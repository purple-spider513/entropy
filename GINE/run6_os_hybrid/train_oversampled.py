"""Train the run5 hyperparameter winners with three-segment oversampling.

Training rows only are copied: fewer than 12 heavy atoms five times, 12 to 15
heavy atoms three times, and larger molecules once. Development and test rows
stay as one copy. Existing training code is imported and not modified.
"""

from __future__ import annotations

import json
import random
import sys
from argparse import Namespace
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from entropy_gnn.data import prepare_dataset
from entropy_gnn.models import ModelConfig
from entropy_gnn.reporting import format_eval_metrics, format_metrics, round_metrics, write_json, write_predictions_csv
from entropy_gnn.training import evaluate_model, load_run_model, resolve_device, select_candidate, train_candidate
from train import checkpoint_dict, environment_versions, evaluate_external_csvs, save_dataset_artifacts, save_history

CSV = ROOT / "hybrid_data" / "hybrid_train_091926.csv"
OUTPUT = ROOT / "run6" / "minimal"
SEED = 42
BATCH_SIZE = 64
MAX_EPOCHS = 300
PATIENCE = 30
GPU_INDEX = 2

WINNERS = (
    (
        "gnn_only",
        ModelConfig(
            mode="gnn",
            architecture="gine",
            hidden_channels=128,
            num_layers=3,
            dropout=0.0,
            use_distance_features=False,
        ),
    ),
    (
        "hybrid",
        ModelConfig(
            mode="fused",
            architecture="gine",
            hidden_channels=128,
            num_layers=5,
            dropout=0.0,
            use_distance_features=True,
        ),
    ),
)

EVAL_CSVS = [
    ["std109", str(ROOT / "hybrid_data" / "hybrid_std109.csv")],
    ["std128", str(ROOT / "hybrid_data" / "hybrid_std128.csv")],
    ["std6074", str(ROOT / "hybrid_data" / "hybrid_std6074.csv")],
    ["pdbbind", str(ROOT / "hybrid_data" / "hybrid_pdbbind.csv")],
]


def copy_count(heavy_atoms: float) -> int:
    if heavy_atoms < 12:
        return 5
    if heavy_atoms <= 15:
        return 3
    return 1


def oversample_train(heavy: pd.Series, train_indices: list[int], seed: int) -> tuple[list[int], dict[str, int]]:
    expanded: list[int] = []
    bins = {"below_12_x5": 0, "from_12_to_15_x3": 0, "above_15_x1": 0}
    for index in train_indices:
        copies = copy_count(float(heavy.iloc[index]))
        expanded.extend([index] * copies)
        if copies == 5:
            bins["below_12_x5"] += 1
        elif copies == 3:
            bins["from_12_to_15_x3"] += 1
        else:
            bins["above_15_x1"] += 1
    random.Random(seed).shuffle(expanded)
    bins["unique_train"] = len(train_indices)
    bins["oversampled_train_rows"] = len(expanded)
    return expanded, bins


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "histories").mkdir(exist_ok=True)
    device = resolve_device("cuda", GPU_INDEX)
    print(f"device {device} ({torch.cuda.get_device_name(device.index)})", flush=True)
    args = Namespace(
        output_dir=OUTPUT,
        seed=SEED,
        embed_3d=True,
        batch_size=BATCH_SIZE,
        selection_objective="pearson",
    )
    write_json(
        OUTPUT / "run_config.json",
        {
            "csv": str(CSV),
            "output_dir": str(OUTPUT),
            "seed": SEED,
            "source_run": "run5 winners",
            "oversampling": "three-segment: heavy<12 x5, 12-15 x3, >15 x1; train split only",
            "max_epochs": MAX_EPOCHS,
            "patience": PATIENCE,
            "batch_size": BATCH_SIZE,
            "device": str(device),
            "gpu_index": GPU_INDEX,
            "embed_3d": True,
            "selection_objective": "pearson",
            "candidates": {name: config.to_dict() for name, config in WINNERS},
            "eval_csv": EVAL_CSVS,
        },
    )
    write_json(OUTPUT / "environment.json", environment_versions())

    dataset = prepare_dataset(CSV, seed=SEED, embed_3d=True)
    save_dataset_artifacts(dataset, OUTPUT)
    if "heavy" not in dataset.source_frame.columns:
        raise SystemExit("training table has no heavy column")
    expanded, bins = oversample_train(dataset.source_frame["heavy"], dataset.split_indices["train"], SEED)
    write_json(OUTPUT / "oversampling.json", bins)
    print(
        f"oversampled train rows {bins['unique_train']} -> {bins['oversampled_train_rows']} "
        f"(<12 x5: {bins['below_12_x5']}, 12-15 x3: {bins['from_12_to_15_x3']}, >15 x1: {bins['above_15_x1']})",
        flush=True,
    )
    split_indices = {
        "train": expanded,
        "dev": dataset.split_indices["dev"],
        "test": dataset.split_indices["test"],
    }

    runs = []
    for name, config in WINNERS:
        print(f"training {name}: {config}", flush=True)
        run = train_candidate(
            name=name,
            config=config,
            graphs=dataset.graphs,
            split_indices=split_indices,
            feature_schema=dataset.feature_schema,
            preprocessor=dataset.preprocessor,
            seed=SEED,
            device=device,
            batch_size=BATCH_SIZE,
            max_epochs=MAX_EPOCHS,
            patience=PATIENCE,
            learning_rate=1e-3,
            monitor="pearson",
            loss_name="mse",
        )
        save_history(run, args.output_dir)
        print(f"{name} development metrics: {format_metrics(run.dev_metrics)}", flush=True)
        runs.append(run)

    named = {run.name: run for run in runs}
    best = select_candidate(runs, "pearson")
    mode_metrics = {}
    for run in runs:
        mode_metrics[run.name] = {}
        frames = []
        for split in ("train", "dev", "test"):
            metrics, target, prediction, molecule_ids = evaluate_model(
                load_run_model(run, dataset.feature_schema, device),
                dataset.graphs,
                dataset.split_indices[split],
                dataset.preprocessor,
                BATCH_SIZE,
                device,
            )
            mode_metrics[run.name][split] = metrics
            frames.append(
                pd.DataFrame(
                    {
                        "ID": molecule_ids,
                        "split": split,
                        "entropy": target,
                        "predicted_entropy": prediction,
                        "residual": prediction - target,
                    }
                )
            )
            print(f"  {run.name} {split} {format_metrics(metrics)}", flush=True)
        write_predictions_csv(OUTPUT / f"{run.name}_predictions.csv", pd.concat(frames, ignore_index=True))

    best_mode = best.name
    test_metrics = mode_metrics[best_mode]["test"]
    accepted = bool(test_metrics["pearson"] >= 0.95)
    payload = {
        "selected_candidate": best.name,
        "selected_mode": best_mode,
        "model_config": best.config.to_dict(),
        "selection_objective": "pearson",
        "oversampling": bins,
        "accepted": accepted,
        "pearson_target": 0.95,
        "selected_dev_metrics": round_metrics(best.dev_metrics),
        "splits": {split: round_metrics(values) for split, values in mode_metrics[best_mode].items()},
        "mode_winners": {
            name: {
                "candidate": run.name,
                "model_config": run.config.to_dict(),
                "dev_metrics": round_metrics(run.dev_metrics),
            }
            for name, run in named.items()
        },
        "modes": {
            name: {split: round_metrics(values) for split, values in splits.items()}
            for name, splits in mode_metrics.items()
        },
    }
    write_json(OUTPUT / "metrics.json", payload)
    checkpoints = {}
    for run in runs:
        path = OUTPUT / f"{run.name}.pt"
        torch.save(checkpoint_dict(run, dataset, args, payload, accepted), path)
        checkpoints[run.name] = path
    torch.save(checkpoint_dict(best, dataset, args, payload, accepted), OUTPUT / "best_model.pt")
    print(f"selected {best_mode} / {best.name}", flush=True)
    external = evaluate_external_csvs(checkpoints, EVAL_CSVS, OUTPUT, device, BATCH_SIZE)
    write_json(OUTPUT / "external_metrics.json", external)
    print(json.dumps({mode: {name: format_eval_metrics(metrics) for name, metrics in sets.items()} for mode, sets in external.items()}))


if __name__ == "__main__":
    main()
