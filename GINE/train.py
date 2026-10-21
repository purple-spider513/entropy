from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from entropy_gnn.data import (
    PreparedDataset,
    load_prepared_dataset,
    prepare_dataset,
    save_prepared_dataset,
    sha256_file,
)
from entropy_gnn.inference import evaluate_csv
from entropy_gnn.models import ModelConfig
from entropy_gnn.reporting import (
    format_eval_metrics,
    format_metrics,
    round_metrics,
    write_json,
    write_predictions_csv,
)
from entropy_gnn.training import (
    CandidateRun,
    evaluate_model,
    load_run_model,
    resolve_device,
    select_candidate,
    train_candidate,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train entropy models from a combined hybrid CSV "
        "(GNN-only from SMILES graphs, and GNN+features hybrid)."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        required=True,
        help="Combined training CSV with ID, SMILES, entropy, and ligand descriptors.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-limit", type=int, default=0, help="Cap usable molecules; 0 means all.")
    parser.add_argument("--pearson-target", type=float, default=0.95)
    parser.add_argument(
        "--mae-target",
        type=float,
        default=14.0,
        help="Held-out MAE threshold used when --selection-objective mae.",
    )
    parser.add_argument(
        "--selection-objective",
        choices=["pearson", "mae"],
        default="pearson",
        help="Choose each mode's winner by development Pearson or by lowest development MAE.",
    )
    parser.add_argument(
        "--min-dev-pearson",
        type=float,
        default=-1.0,
        help="Hard minimum development Pearson required before a candidate can be selected.",
    )
    parser.add_argument("--expand-on-miss", action="store_true")
    parser.add_argument(
        "--always-tune",
        action="store_true",
        help="Run focused hyperparameter grids for GNN-only and hybrid models.",
    )
    parser.add_argument("--max-epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument(
        "--gpu-index",
        type=int,
        default=-1,
        help="CUDA device index; -1 selects the GPU with the most free memory.",
    )
    parser.add_argument("--embed-3d", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Optional directory of prepared dataset caches. Loaded if present and written after graph construction.",
    )
    parser.add_argument(
        "--eval-csv",
        action="append",
        nargs=2,
        metavar=("NAME", "CSV"),
        default=[],
        help="Evaluate every saved mode on a labelled hybrid CSV after training. Repeatable.",
    )
    return parser.parse_args()


def dataset_cache_path(
    cache_dir: Path,
    csv_path: Path,
    seed: int,
    sample_limit: int,
    embed_3d: bool,
) -> Path:
    payload = {
        "csv": sha256_file(csv_path),
        "seed": seed,
        "sample_limit": sample_limit,
        "embed_3d": embed_3d,
        "format": 3,
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:20]
    return cache_dir / f"{digest}.pt"


def load_or_prepare_dataset(args: argparse.Namespace) -> PreparedDataset:
    cache_path = None
    if args.cache_dir is not None:
        cache_path = dataset_cache_path(
            args.cache_dir,
            args.csv,
            args.seed,
            args.sample_limit,
            args.embed_3d,
        )
        if cache_path.exists():
            print(f"loading cached dataset from {cache_path}", flush=True)
            return load_prepared_dataset(cache_path)
    dataset = prepare_dataset(
        args.csv,
        sample_limit=args.sample_limit,
        seed=args.seed,
        embed_3d=args.embed_3d,
    )
    if cache_path is not None:
        print(f"writing dataset cache to {cache_path}", flush=True)
        save_prepared_dataset(dataset, cache_path)
    return dataset


def json_dump(path: Path, values: Any) -> None:
    write_json(path, values)


def safe_metrics(values: dict[str, float]) -> dict[str, float | None]:
    return round_metrics(values)


def environment_versions() -> dict[str, Any]:
    packages = ["numpy", "torch", "torch-geometric", "rdkit", "pandas", "scikit-learn", "scipy"]
    versions: dict[str, str] = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "missing"
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": versions,
        "cuda_available": torch.cuda.is_available(),
    }


def save_dataset_artifacts(dataset: PreparedDataset, directory: Path, prefix: str = "") -> None:
    stem = f"{prefix}_" if prefix else ""
    dataset.manifest.to_csv(directory / f"{stem}dataset_manifest.csv", index=False)
    dataset.audit.to_csv(directory / f"{stem}audit.csv", index=False)
    json_dump(directory / f"{stem}preprocessing.json", dataset.preprocessor.to_dict())
    json_dump(directory / f"{stem}feature_schema.json", dataset.feature_schema)
    json_dump(directory / f"{stem}training_identities.json", dataset.identities.to_dict())


def save_history(run: CandidateRun, output_dir: Path) -> None:
    pd.DataFrame(run.history).to_csv(
        output_dir / "histories" / f"{run.name}.csv",
        index=False,
        float_format="%.6g",
    )


def print_training_summary(
    metrics_payload: dict[str, Any],
    named_runs: dict[str, CandidateRun],
    mode_metrics: dict[str, dict[str, dict[str, float]]],
    accepted: bool,
) -> None:
    status = "accepted" if accepted else "rejected"
    print(
        f"selected {metrics_payload['selected_mode']} / {metrics_payload['selected_candidate']} ({status})",
        flush=True,
    )
    for split in ("train", "dev", "test"):
        print(f"  {split:<5} {format_metrics(metrics_payload['splits'][split])}", flush=True)
    print("mode winners:", flush=True)
    for mode_name, run in named_runs.items():
        test = mode_metrics[mode_name]["test"]
        print(f"  {mode_name:<14} {run.name}  test {format_metrics(test)}", flush=True)


def run_candidate(
    name: str,
    config: ModelConfig,
    dataset: PreparedDataset,
    args: argparse.Namespace,
    device: torch.device,
    learning_rate: float = 1e-3,
) -> CandidateRun:
    loss_name = "smoothl1" if args.selection_objective == "mae" else "mse"
    print(f"training {name}: {config} monitor={args.selection_objective} loss={loss_name} lr={learning_rate}", flush=True)
    run = train_candidate(
        name=name,
        config=config,
        graphs=dataset.graphs,
        split_indices=dataset.split_indices,
        feature_schema=dataset.feature_schema,
        preprocessor=dataset.preprocessor,
        seed=args.seed,
        device=device,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        patience=args.patience,
        learning_rate=learning_rate,
        monitor=args.selection_objective,
        loss_name=loss_name,
    )
    save_history(run, args.output_dir)
    print(f"{name} development metrics: {format_metrics(run.dev_metrics)}", flush=True)
    return run


def split_metrics_for_run(
    run: CandidateRun,
    dataset: PreparedDataset,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[dict[str, dict[str, float]], pd.DataFrame]:
    model = load_run_model(run, dataset.feature_schema, device)
    all_metrics: dict[str, dict[str, float]] = {}
    prediction_frames: list[pd.DataFrame] = []
    for split in ("train", "dev", "test"):
        metrics, target, prediction, molecule_ids = evaluate_model(
            model,
            dataset.graphs,
            dataset.split_indices[split],
            dataset.preprocessor,
            args.batch_size,
            device,
        )
        all_metrics[split] = metrics
        prediction_frames.append(
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
    return all_metrics, pd.concat(prediction_frames, ignore_index=True)


def checkpoint_dict(
    run: CandidateRun,
    dataset: PreparedDataset,
    args: argparse.Namespace,
    metrics_payload: dict[str, Any],
    accepted: bool,
) -> dict[str, Any]:
    return {
        "format_version": 3,
        "state_dict": run.state_dict,
        "model_config": run.config.to_dict(),
        "feature_schema": dataset.feature_schema,
        "preprocessor": dataset.preprocessor.to_dict(),
        "predictor_columns": dataset.predictor_columns,
        "training_identities": dataset.identities.to_dict(),
        "embed_3d": args.embed_3d,
        "metrics": metrics_payload,
        "accepted": accepted,
        "seed": args.seed,
    }


def evaluate_external_csvs(
    checkpoints: dict[str, Path],
    eval_csvs: list[list[str]],
    output_dir: Path,
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for mode_name, checkpoint in checkpoints.items():
        summary[mode_name] = {}
        for name, csv_path in eval_csvs:
            target = output_dir / "external" / mode_name / name
            target.mkdir(parents=True, exist_ok=True)
            print(f"evaluating {mode_name} on {name}", flush=True)
            metrics, predictions = evaluate_csv(
                checkpoint,
                Path(csv_path),
                device,
                batch_size=batch_size,
            )
            write_predictions_csv(target / "predictions.csv", predictions)
            overlap = predictions.loc[predictions["overlap_with_train"], ["ID", "smiles", "inchikey"]]
            overlap.to_csv(target / "excluded_overlap.csv", index=False)
            json_dump(target / "metrics.json", metrics)
            summary[mode_name][name] = metrics
            print(f"  {mode_name}/{name}  {format_eval_metrics(metrics)}", flush=True)
    return summary


def main() -> None:
    args = parse_args()
    if args.sample_limit < 0:
        raise ValueError("--sample-limit must be nonnegative")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "histories").mkdir(exist_ok=True)
    device = resolve_device(args.device, args.gpu_index)

    input_hashes = {"csv": sha256_file(args.csv)}
    run_configuration = {
        "csv": str(args.csv.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "seed": args.seed,
        "sample_limit": args.sample_limit,
        "pearson_target": args.pearson_target,
        "mae_target": args.mae_target,
        "selection_objective": args.selection_objective,
        "min_dev_pearson": args.min_dev_pearson,
        "expand_on_miss": args.expand_on_miss,
        "always_tune": args.always_tune,
        "max_epochs": args.max_epochs,
        "patience": args.patience,
        "batch_size": args.batch_size,
        "device": str(device),
        "gpu_index": args.gpu_index,
        "embed_3d": args.embed_3d,
        "cache_dir": str(args.cache_dir) if args.cache_dir is not None else None,
        "eval_csv": args.eval_csv,
        "input_hashes": input_hashes,
    }
    json_dump(args.output_dir / "run_config.json", run_configuration)
    json_dump(args.output_dir / "environment.json", environment_versions())

    dataset = load_or_prepare_dataset(args)
    save_dataset_artifacts(dataset, args.output_dir)
    print(
        f"prepared {len(dataset.graphs)} SMILES graphs; predictors={len(dataset.predictor_columns)}, "
        f"constant={len(dataset.preprocessor.constant_columns)}, retained={len(dataset.preprocessor.retained_columns)}",
        flush=True,
    )

    use_distances = args.embed_3d
    mae_mode = args.selection_objective == "mae"
    baseline_dropout = 0.0 if mae_mode else 0.1
    baseline_layers = 3 if mae_mode else 4
    gnn_runs = [
        run_candidate(
            "gnn_only",
            ModelConfig(
                mode="gnn",
                num_layers=baseline_layers,
                dropout=baseline_dropout,
                use_distance_features=use_distances,
            ),
            dataset,
            args,
            device,
        )
    ]
    fused_runs = [
        run_candidate(
            "hybrid",
            ModelConfig(
                mode="fused",
                num_layers=baseline_layers,
                dropout=baseline_dropout,
                use_distance_features=use_distances,
            ),
            dataset,
            args,
            device,
        )
    ]
    diagnostic_runs = [*gnn_runs, *fused_runs]
    best_fused_pearson = max(run.dev_metrics["pearson"] for run in fused_runs)
    best_baseline_mae = min(run.dev_metrics["mae"] for run in diagnostic_runs)
    should_tune = args.always_tune or (
        args.expand_on_miss
        and (
            (mae_mode and best_baseline_mae >= args.mae_target)
            or (not mae_mode and best_fused_pearson < args.pearson_target)
        )
    )
    if should_tune:
        focused = args.always_tune and (mae_mode or best_fused_pearson >= args.pearson_target)
        if focused:
            for hidden, layers, dropout, distances in (
                (128, 3, 0.0, True),
                (128, 3, 0.0, False),
                (128, 4, 0.0, True),
                (128, 5, 0.0, True),
                (256, 3, 0.0, True),
                (256, 4, 0.0, True),
                (256, 5, 0.0, True),
            ):
                name = f"gnn_h{hidden}_l{layers}_d{dropout:g}_xyz{int(distances)}"
                if mae_mode and hidden == 128 and layers == 3 and distances == use_distances and dropout == baseline_dropout:
                    continue
                gnn_runs.append(
                    run_candidate(
                        name,
                        ModelConfig(
                            mode="gnn",
                            hidden_channels=hidden,
                            num_layers=layers,
                            dropout=dropout,
                            use_distance_features=distances,
                        ),
                        dataset,
                        args,
                        device,
                    )
                )
            fused_grid = [
                (128, 1, 0.0, True),
                (128, 3, 0.0, True),
                (128, 4, 0.0, True),
                (128, 5, 0.0, True),
                (256, 1, 0.0, True),
                (256, 3, 0.0, True),
                (256, 4, 0.0, True),
                (256, 5, 0.0, True),
            ]
            if mae_mode:
                fused_grid = [
                    item for item in fused_grid
                    if not (item[0] == 128 and item[1] == baseline_layers and item[2] == baseline_dropout and item[3] == use_distances)
                ]
        else:
            fused_grid = [
                (hidden, layers, dropout, distances)
                for hidden in (128, 256)
                for layers in (1, 3, 5)
                for dropout in (0.0, 0.15)
                for distances in (False, True)
            ]
        for hidden, layers, dropout, distances in fused_grid:
            name = f"gine_h{hidden}_l{layers}_d{dropout:g}_xyz{int(distances)}"
            fused_runs.append(
                run_candidate(
                    name,
                    ModelConfig(
                        mode="fused",
                        architecture="gine",
                        hidden_channels=hidden,
                        num_layers=layers,
                        dropout=dropout,
                        use_distance_features=distances,
                    ),
                    dataset,
                    args,
                    device,
                )
            )
        if focused:
            fused_runs.append(
                run_candidate(
                    "gine_h128_l3_d0_xyz1_lr3e4",
                    ModelConfig(
                        mode="fused",
                        architecture="gine",
                        hidden_channels=128,
                        num_layers=3,
                        dropout=0.0,
                        use_distance_features=True,
                    ),
                    dataset,
                    args,
                    device,
                    learning_rate=3e-4,
                )
            )

        if args.selection_objective == "pearson" and max(run.dev_metrics["pearson"] for run in fused_runs) < args.pearson_target:
            for hidden in (128, 256):
                for layers in (3, 5):
                    fused_runs.append(
                        run_candidate(
                            f"attentivefp_h{hidden}_l{layers}",
                            ModelConfig(
                                mode="fused",
                                architecture="attentivefp",
                                hidden_channels=hidden,
                                num_layers=layers,
                                dropout=0.1,
                                use_distance_features=True,
                            ),
                            dataset,
                            args,
                            device,
                        )
                    )

    named_runs = {
        "gnn_only": select_candidate(gnn_runs, args.selection_objective, args.min_dev_pearson),
        "hybrid": select_candidate(fused_runs, args.selection_objective, args.min_dev_pearson),
    }
    best = select_candidate(list(named_runs.values()), args.selection_objective, args.min_dev_pearson)
    best_mode = next(name for name, run in named_runs.items() if run is best)

    mode_metrics: dict[str, dict[str, dict[str, float]]] = {}
    for name, run in named_runs.items():
        split_values, predictions = split_metrics_for_run(run, dataset, args, device)
        mode_metrics[name] = split_values
        write_predictions_csv(args.output_dir / f"{name}_predictions.csv", predictions)

    test_metrics = mode_metrics[best_mode]["test"]
    if args.selection_objective == "mae":
        accepted = bool(test_metrics["mae"] < args.mae_target)
    else:
        accepted = bool(test_metrics["pearson"] >= args.pearson_target)
    all_candidates = [*gnn_runs, *fused_runs]
    metrics_payload = {
        "selected_candidate": best.name,
        "selected_mode": best_mode,
        "model_config": best.config.to_dict(),
        "selection_objective": args.selection_objective,
        "min_dev_pearson": args.min_dev_pearson,
        "selected_dev_metrics": safe_metrics(best.dev_metrics),
        "accepted": accepted,
        "pearson_target": args.pearson_target,
        "mae_target": args.mae_target,
        "splits": {split: safe_metrics(values) for split, values in mode_metrics[best_mode].items()},
        "mode_winners": {
            name: {
                "candidate": run.name,
                "model_config": run.config.to_dict(),
                "dev_metrics": safe_metrics(run.dev_metrics),
            }
            for name, run in named_runs.items()
        },
        "modes": {
            name: {split: safe_metrics(values) for split, values in splits.items()}
            for name, splits in mode_metrics.items()
        },
        "diagnostics": {run.name: safe_metrics(run.dev_metrics) for run in diagnostic_runs},
    }
    pd.DataFrame(
        [
            {
                "candidate": run.name,
                "model_config": json.dumps(run.config.to_dict(), sort_keys=True),
                **safe_metrics(run.dev_metrics),
            }
            for run in all_candidates
        ]
    ).to_csv(args.output_dir / "candidate_metrics.csv", index=False, float_format="%.6g")
    json_dump(args.output_dir / "metrics.json", metrics_payload)

    checkpoints = {
        "gnn_only": args.output_dir / "gnn_only.pt",
        "hybrid": args.output_dir / "hybrid.pt",
    }
    for name, path in checkpoints.items():
        torch.save(checkpoint_dict(named_runs[name], dataset, args, metrics_payload, accepted), path)
    torch.save(checkpoint_dict(best, dataset, args, metrics_payload, accepted), args.output_dir / "best_model.pt")
    print_training_summary(metrics_payload, named_runs, mode_metrics, accepted)
    if not accepted:
        if args.selection_objective == "mae":
            print(
                f"REJECTED: held-out test MAE {test_metrics['mae']:.6f} "
                f"is not below {args.mae_target:.6f}",
                flush=True,
            )
        else:
            print(
                f"REJECTED: held-out test Pearson {test_metrics['pearson']:.6f} "
                f"is below {args.pearson_target:.6f}",
                flush=True,
            )

    if args.eval_csv:
        external = evaluate_external_csvs(
            checkpoints,
            args.eval_csv,
            args.output_dir,
            device,
            args.batch_size,
        )
        json_dump(args.output_dir / "external_metrics.json", external)


if __name__ == "__main__":
    main()
