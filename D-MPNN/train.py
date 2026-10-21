#!/usr/bin/env python3
"""Train CheMeleon D-MPNN (SMILES) and/or hybrid (SMILES + extra features).

  conda run --no-capture-output -n chemprop python train.py \\
      -i hybrid_data/hybrid_train_091926.csv -o my_run --gpu 0

After training, unique-molecule train/val/test MAE and R2 are written next to
the run. Hyperparameter grid search:

  conda run --no-capture-output -n chemprop python train.py \\
      --hpopt --source-run run2 -o hpopt_run2 --gpu 0
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from predict import (
    format_metrics_row,
    load_feature_names,
    load_hybrid_table,
    load_split,
    metrics,
    metrics_header,
    model_path,
    predict_one,
    run_predict,
)

ROOT = Path(__file__).resolve().parent
TARGET_COL = "entropy"
SMILES_COL = "SMILES"
RESERVED = {"ID", SMILES_COL, TARGET_COL}

MODE_HPARAMS = {
    "dmpnn": {
        "ffn_hidden_dim": 300,
        "ffn_num_layers": 1,
        "dropout": 0.0,
        "warmup_epochs": 2,
        "init_lr": 1e-4,
        "max_lr": 1e-3,
        "final_lr": 1e-4,
    },
    "hybrid": {
        "ffn_hidden_dim": 1024,
        "ffn_num_layers": 3,
        "dropout": 0.1,
        "warmup_epochs": 4,
        "init_lr": 1e-5,
        "max_lr": 1e-4,
        "final_lr": 1e-5,
    },
}

DMPNN_TRIALS = [
    {"name": "ffn1_300_d00_lr1e3", "ffn_hidden_dim": 300, "ffn_num_layers": 1, "dropout": 0.0, "warmup_epochs": 2, "init_lr": 1e-4, "max_lr": 1e-3, "final_lr": 1e-4, "baseline": True},
    {"name": "ffn1_300_d10_lr1e3", "ffn_hidden_dim": 300, "ffn_num_layers": 1, "dropout": 0.1, "warmup_epochs": 2, "init_lr": 1e-4, "max_lr": 1e-3, "final_lr": 1e-4},
    {"name": "ffn2_300_d05_lr1e3", "ffn_hidden_dim": 300, "ffn_num_layers": 2, "dropout": 0.05, "warmup_epochs": 2, "init_lr": 1e-4, "max_lr": 1e-3, "final_lr": 1e-4},
    {"name": "ffn1_512_d05_lr1e3", "ffn_hidden_dim": 512, "ffn_num_layers": 1, "dropout": 0.05, "warmup_epochs": 2, "init_lr": 1e-4, "max_lr": 1e-3, "final_lr": 1e-4},
    {"name": "ffn2_512_d10_lr5e4", "ffn_hidden_dim": 512, "ffn_num_layers": 2, "dropout": 0.1, "warmup_epochs": 4, "init_lr": 5e-5, "max_lr": 5e-4, "final_lr": 5e-5},
    {"name": "ffn3_300_d10_lr1e4", "ffn_hidden_dim": 300, "ffn_num_layers": 3, "dropout": 0.1, "warmup_epochs": 4, "init_lr": 1e-5, "max_lr": 1e-4, "final_lr": 1e-5},
]

HYBRID_TRIALS = [
    {"name": "ffn3_1024_d10_lr1e4", "ffn_hidden_dim": 1024, "ffn_num_layers": 3, "dropout": 0.1, "warmup_epochs": 4, "init_lr": 1e-5, "max_lr": 1e-4, "final_lr": 1e-5, "baseline": True},
    {"name": "ffn2_1024_d10_lr1e4", "ffn_hidden_dim": 1024, "ffn_num_layers": 2, "dropout": 0.1, "warmup_epochs": 4, "init_lr": 1e-5, "max_lr": 1e-4, "final_lr": 1e-5},
    {"name": "ffn3_512_d10_lr1e4", "ffn_hidden_dim": 512, "ffn_num_layers": 3, "dropout": 0.1, "warmup_epochs": 4, "init_lr": 1e-5, "max_lr": 1e-4, "final_lr": 1e-5},
    {"name": "ffn2_512_d05_lr1e4", "ffn_hidden_dim": 512, "ffn_num_layers": 2, "dropout": 0.05, "warmup_epochs": 4, "init_lr": 1e-5, "max_lr": 1e-4, "final_lr": 1e-5},
    {"name": "ffn3_1024_d20_lr5e5", "ffn_hidden_dim": 1024, "ffn_num_layers": 3, "dropout": 0.2, "warmup_epochs": 4, "init_lr": 5e-6, "max_lr": 5e-5, "final_lr": 5e-6},
    {"name": "ffn3_1024_d10_lr3e4", "ffn_hidden_dim": 1024, "ffn_num_layers": 3, "dropout": 0.1, "warmup_epochs": 4, "init_lr": 3e-5, "max_lr": 3e-4, "final_lr": 3e-5},
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("both", "dmpnn", "hybrid"), default="both")
    p.add_argument("-i", "--input", dest="input_csv", type=Path, help="Training CSV with SMILES, entropy, and hybrid feature columns.")
    p.add_argument("-o", "--output", type=Path, required=True, help="New output folder. Do not point this at an existing run you want to keep.")
    p.add_argument("--gpu", default="0", help="nvidia-smi GPU id, or 'cpu'")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--keep-constant-features", action="store_true", help="Hybrid: keep zero-variance columns.")
    p.add_argument(
        "--oversample-heavy",
        action="store_true",
        help="Oversample train only: 5 copies if heavy < 12, 3 if 12-15, 1 if heavy > 15.",
    )
    p.add_argument("--splits-file", type=Path, default=None, help="Reuse an existing unique-molecule splits.json.")
    p.add_argument("--ffn-hidden-dim", type=int, default=None)
    p.add_argument("--ffn-num-layers", type=int, default=None)
    p.add_argument("--dropout", type=float, default=None)
    p.add_argument("--warmup-epochs", type=int, default=None)
    p.add_argument("--init-lr", type=float, default=None)
    p.add_argument("--max-lr", type=float, default=None)
    p.add_argument("--final-lr", type=float, default=None)
    p.add_argument("--hpopt", action="store_true", help="Grid-search FFN/dropout/LR on an existing run's data and splits.")
    p.add_argument("--source-run", type=Path, default=None, help="Existing run folder to reuse for --hpopt (read-only).")
    p.add_argument(
        "--eval-csv",
        type=Path,
        default=None,
        help="Unique-molecule CSV for --hpopt scoring (default: hybrid_data/hybrid_train_091926.csv).",
    )
    return p.parse_args()


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in RESERVED]


def feature_matrix(df: pd.DataFrame, columns: list[str]) -> np.ndarray:
    X = df.loc[:, columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


def drop_constant_columns(X: np.ndarray, columns: list[str]) -> list[str]:
    return [c for i, c in enumerate(columns) if np.unique(X[:, i]).size > 1]


def write_smiles_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df[[SMILES_COL, TARGET_COL]].to_csv(path, index=False)


def write_descriptors_npz(X: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, X)


def make_scaffold_split(smiles: pd.Series, out_path: Path, seed: int = 0) -> dict:
    from rdkit import Chem
    from chemprop.data.splitting import make_split_indices

    mols = [Chem.MolFromSmiles(s) for s in smiles]
    if any(m is None for m in mols):
        bad = [i for i, m in enumerate(mols) if m is None]
        raise ValueError(f"Invalid SMILES at rows {bad[:10]}")
    train_idx, val_idx, test_idx = make_split_indices(
        mols, split="scaffold_balanced", sizes=(0.8, 0.1, 0.1), seed=seed, num_replicates=1
    )
    split = {
        "train": [int(i) for i in train_idx[0]],
        "val": [int(i) for i in val_idx[0]],
        "test": [int(i) for i in test_idx[0]],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps([split]))
    return split


def run_chemprop(args: list[str], gpu: str | None) -> None:
    env = os.environ.copy()
    conda = shutil.which("conda")
    extra_env: list[str] = []
    if gpu is not None and str(gpu).lower() != "cpu":
        extra_env = ["env", f"CUDA_VISIBLE_DEVICES={gpu}"]
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    if conda:
        cmd = [conda, "run", "--no-capture-output", "-n", "chemprop", *extra_env, "chemprop", *args]
    else:
        cmd = ["chemprop", *args]
    if gpu is None or str(gpu).lower() == "cpu":
        cmd.extend(["--accelerator", "cpu"])
    else:
        cmd.extend(["--accelerator", "gpu", "--devices", "1"])
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env=env, cwd=ROOT)


def heavy_copy_number(n_heavy: float) -> int:
    if n_heavy < 12:
        return 5
    if n_heavy <= 15:
        return 3
    return 1


def oversample_train_by_heavy(df: pd.DataFrame, split: dict, seed: int) -> tuple[pd.DataFrame, dict, dict]:
    if "heavy" not in df.columns:
        raise ValueError("oversampling requires a 'heavy' column")
    heavy = pd.to_numeric(df["heavy"], errors="coerce").fillna(0).to_numpy()
    rng = np.random.default_rng(seed)
    train_rows: list[int] = []
    bins = {"<12": 0, "12-15": 0, ">15": 0}
    for i in split["train"]:
        n_heavy = float(heavy[i])
        copies = heavy_copy_number(n_heavy)
        if n_heavy < 12:
            bins["<12"] += 1
        elif n_heavy <= 15:
            bins["12-15"] += 1
        else:
            bins[">15"] += 1
        train_rows.extend([int(i)] * copies)
    rng.shuffle(train_rows)
    row_order: list[int] = []
    new_split = {"train": [], "val": [], "test": []}
    for i in train_rows:
        new_split["train"].append(len(row_order))
        row_order.append(i)
    for part in ("val", "test"):
        for i in split[part]:
            new_split[part].append(len(row_order))
            row_order.append(int(i))
    over_df = df.iloc[row_order].reset_index(drop=True)
    stats = {
        "unique_train": len(split["train"]),
        "oversampled_train": len(new_split["train"]),
        "val": len(new_split["val"]),
        "test": len(new_split["test"]),
        "bins_unique_train": bins,
    }
    return over_df, new_split, stats


def modes_to_train(mode: str) -> list[str]:
    return ["dmpnn", "hybrid"] if mode == "both" else [mode]


def run_dir_for(mode: str, output: Path, n_modes: int) -> Path:
    return output / mode if n_modes > 1 else output


def hparams_for(mode: str, args: argparse.Namespace) -> dict:
    hp = dict(MODE_HPARAMS[mode])
    for key in ("ffn_hidden_dim", "ffn_num_layers", "dropout", "warmup_epochs", "init_lr", "max_lr", "final_lr"):
        val = getattr(args, key)
        if val is not None:
            hp[key] = val
    return hp


def prepare_hybrid_features(df, out_dir: Path, keep_constant: bool) -> Path:
    cols = feature_columns(df)
    X = feature_matrix(df, cols)
    if not keep_constant:
        cols = drop_constant_columns(X, cols)
        X = feature_matrix(df, cols)
    pd.Series(cols, name="name").to_csv(out_dir / "feature_names.csv", index=False)
    npz_path = out_dir / "descriptors.npz"
    write_descriptors_npz(X, npz_path)
    print(f"hybrid features: {X.shape[1]} columns -> {npz_path}")
    return npz_path


def train_one(
    mode: str,
    df,
    out_dir: Path,
    chemprop_splits_path: Path,
    eval_splits_path: Path,
    args: argparse.Namespace,
    hp: dict | None = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    hp = hp or hparams_for(mode, args)
    smiles_csv = out_dir / "train.csv"
    write_smiles_csv(df, smiles_csv)
    dest_splits = out_dir / "splits_chemprop.json"
    if chemprop_splits_path.resolve() != dest_splits.resolve():
        shutil.copy2(chemprop_splits_path, dest_splits)
    eval_dest = out_dir / "splits.json"
    if eval_splits_path.resolve() != eval_dest.resolve():
        shutil.copy2(eval_splits_path, eval_dest)
    cmd = [
        "train",
        "-i", str(smiles_csv),
        "-o", str(out_dir),
        "--task-type", "regression",
        "--target-columns", TARGET_COL,
        "--from-foundation", "CheMeleon",
        "--metrics", "r2", "mae", "rmse",
        "--tracking-metric", "val_loss",
        "--splits-file", str(dest_splits),
        "--epochs", str(args.epochs),
        "--patience", str(args.patience),
        "--ffn-hidden-dim", str(hp["ffn_hidden_dim"]),
        "--ffn-num-layers", str(hp["ffn_num_layers"]),
        "--dropout", str(hp["dropout"]),
        "--warmup-epochs", str(hp["warmup_epochs"]),
        "--init-lr", str(hp["init_lr"]),
        "--max-lr", str(hp["max_lr"]),
        "--final-lr", str(hp["final_lr"]),
        "--num-workers", str(args.num_workers),
        "--logfile", str(out_dir / "train.log"),
    ]
    if mode == "hybrid":
        npz_path = prepare_hybrid_features(df, out_dir, args.keep_constant_features)
        cmd.extend(["--descriptors-path", str(npz_path)])
    print(f"=== train {mode} -> {out_dir} ===", flush=True)
    run_chemprop(cmd, args.gpu)
    shutil.copy2(eval_splits_path, eval_dest)
    print(f"done {mode}. checkpoint: {out_dir / 'model_0' / 'best.pt'}", flush=True)


def chemprop_splits_path(model_dir: Path) -> Path:
    chemprop = model_dir / "splits_chemprop.json"
    return chemprop if chemprop.is_file() else model_dir / "splits.json"


def write_hpopt_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "source_run", "model", "trial", "baseline", "ckpt_dir", "split",
        "n", "mae", "rmse", "r2", "bias",
        "ffn_hidden_dim", "ffn_num_layers", "dropout", "warmup_epochs",
        "init_lr", "max_lr", "final_lr",
    ]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def train_hpopt_trial(mode: str, trial: dict, source_model_dir: Path, out_dir: Path, args: argparse.Namespace) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    train_csv = source_model_dir / "train.csv"
    splits = chemprop_splits_path(source_model_dir)
    dest_splits = out_dir / splits.name
    if splits.resolve() != dest_splits.resolve():
        shutil.copy2(splits, dest_splits)
    cmd = [
        "train", "-i", str(train_csv), "-o", str(out_dir),
        "--task-type", "regression", "--target-columns", TARGET_COL,
        "--from-foundation", "CheMeleon", "--metrics", "r2", "mae", "rmse",
        "--tracking-metric", "val_loss", "--splits-file", str(dest_splits),
        "--epochs", str(args.epochs), "--patience", str(args.patience),
        "--ffn-hidden-dim", str(trial["ffn_hidden_dim"]),
        "--ffn-num-layers", str(trial["ffn_num_layers"]),
        "--dropout", str(trial["dropout"]),
        "--warmup-epochs", str(trial["warmup_epochs"]),
        "--init-lr", str(trial["init_lr"]),
        "--max-lr", str(trial["max_lr"]),
        "--final-lr", str(trial["final_lr"]),
        "--num-workers", str(args.num_workers),
        "--logfile", str(out_dir / "train.log"),
        "--remove-checkpoints",
    ]
    if mode == "hybrid":
        npz = source_model_dir / "descriptors.npz"
        names = source_model_dir / "feature_names.csv"
        if not npz.is_file() or not names.is_file():
            raise FileNotFoundError(f"hybrid features missing in {source_model_dir}")
        shutil.copy2(names, out_dir / "feature_names.csv")
        cmd.extend(["--descriptors-path", str(npz)])
    print(f"=== train {mode} {trial['name']} -> {out_dir} ===", flush=True)
    run_chemprop(cmd, args.gpu)
    unique_splits = source_model_dir.parent / "splits.json"
    if unique_splits.is_file():
        shutil.copy2(unique_splits, out_dir / "splits.json")
    print(f"done {mode} {trial['name']}. checkpoint: {model_path(out_dir)}", flush=True)
    return out_dir


def score_hpopt_trial(mode: str, ckpt_dir: Path, eval_df: pd.DataFrame, split: dict, work_dir: Path, gpu: str) -> dict[str, dict]:
    features = load_feature_names(ckpt_dir) if mode == "hybrid" else None
    yhat = predict_one(eval_df, work_dir, f"{ckpt_dir.parent.name}_{ckpt_dir.name}_{mode}", model_path(ckpt_dir), gpu, features)
    y = eval_df[TARGET_COL].to_numpy(dtype=float)
    out = {"all": metrics(y, yhat.to_numpy())}
    for part in ("train", "val", "test"):
        idx = split[part]
        out[part] = metrics(y[idx], yhat.to_numpy()[idx])
    return out


def link_best(dest: Path, src: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_symlink() or dest.is_file():
        dest.unlink()
    elif dest.is_dir():
        shutil.rmtree(dest)
    dest.symlink_to(src.resolve())


def run_hpopt_grid(mode: str, trials: list[dict], args: argparse.Namespace, rows: list[dict]) -> dict:
    source_run = args.source_run.resolve()
    source_model_dir = source_run / mode
    if not model_path(source_model_dir).is_file():
        raise FileNotFoundError(model_path(source_model_dir))
    eval_csv = args.eval_csv or (ROOT / "hybrid_data" / "hybrid_train_091926.csv")
    eval_df = load_hybrid_table(eval_csv)
    split = load_split(source_run / "splits.json")
    work_dir = args.output / "_work"
    work_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output / "summary.csv"
    best: dict | None = None
    print(f"{'model':8s} {metrics_header()}", flush=True)
    for trial in trials:
        out_dir = args.output / "trials" / mode / trial["name"]
        is_baseline = bool(trial.get("baseline"))
        rows[:] = [r for r in rows if not (str(r.get("model")) == mode and str(r.get("trial")) == trial["name"])]
        if is_baseline:
            ckpt_dir = source_model_dir
            print(f"=== reuse baseline {mode} {trial['name']} from {ckpt_dir} ===", flush=True)
        elif model_path(out_dir).is_file():
            ckpt_dir = out_dir
            print(f"=== resume {mode} {trial['name']} (checkpoint exists) ===", flush=True)
        else:
            ckpt_dir = train_hpopt_trial(mode, trial, source_model_dir, out_dir, args)
        scores = score_hpopt_trial(mode, ckpt_dir, eval_df, split, work_dir, args.gpu)
        for part, m in scores.items():
            rows.append({
                "source_run": source_run.name, "model": mode, "trial": trial["name"],
                "baseline": is_baseline, "ckpt_dir": str(ckpt_dir), "split": part, **m,
                "ffn_hidden_dim": trial["ffn_hidden_dim"], "ffn_num_layers": trial["ffn_num_layers"],
                "dropout": trial["dropout"], "warmup_epochs": trial["warmup_epochs"],
                "init_lr": trial["init_lr"], "max_lr": trial["max_lr"], "final_lr": trial["final_lr"],
            })
            if part in ("val", "test"):
                label = f"{trial['name']}/{part}"
                print(f"{mode:8s} {format_metrics_row(label, m)}", flush=True)
        write_hpopt_rows(summary_path, rows)
        val_mae = scores["val"]["mae"]
        if best is None or val_mae < best["val_mae"]:
            best = {
                "trial": trial["name"], "ckpt_dir": str(ckpt_dir),
                "val_mae": val_mae, "test_mae": scores["test"]["mae"], "test_r2": scores["test"]["r2"],
                "hparams": {k: trial[k] for k in ("ffn_hidden_dim", "ffn_num_layers", "dropout", "warmup_epochs", "init_lr", "max_lr", "final_lr")},
            }
    if best is None:
        raise RuntimeError(f"no {mode} trials completed")
    link_best(args.output / mode, Path(best["ckpt_dir"]))
    (args.output / f"best_{mode}.json").write_text(json.dumps(best, indent=2))
    print(
        f"best {mode}: {best['trial']}  val MAE {best['val_mae']:.2f}  "
        f"test MAE {best['test_mae']:.2f}  test R2 {best['test_r2']:.4f}",
        flush=True,
    )
    return best


def run_hpopt(args: argparse.Namespace) -> None:
    if args.source_run is None:
        raise ValueError("--hpopt requires --source-run")
    args.source_run = args.source_run.resolve()
    args.output = args.output.resolve()
    if not args.source_run.is_dir():
        raise FileNotFoundError(args.source_run)
    if args.output == args.source_run:
        raise ValueError("output folder must be different from the source run")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "search_space.json").write_text(json.dumps({"dmpnn": DMPNN_TRIALS, "hybrid": HYBRID_TRIALS}, indent=2))
    shutil.copy2(args.source_run / "splits.json", args.output / "splits.json")
    rows: list[dict] = []
    summary_existing = args.output / "summary.csv"
    if summary_existing.is_file():
        rows = pd.read_csv(summary_existing).to_dict(orient="records")
    modes = modes_to_train(args.mode)
    for mode in modes:
        run_hpopt_grid(mode, DMPNN_TRIALS if mode == "dmpnn" else HYBRID_TRIALS, args, rows)
    eval_csv = args.eval_csv or (ROOT / "hybrid_data" / "hybrid_train_091926.csv")
    print("=== best-model metrics (unique train/val/test) ===", flush=True)
    run_predict(
        input_csvs=[eval_csv],
        output=args.output / "predictions.csv",
        dmpnn_dir=args.output / "dmpnn" if "dmpnn" in modes else None,
        hybrid_dir=args.output / "hybrid" if "hybrid" in modes else None,
        gpu=args.gpu,
    )


def run_train(args: argparse.Namespace) -> None:
    if args.input_csv is None:
        raise ValueError("training requires -i / --input")
    if not args.input_csv.is_file():
        raise FileNotFoundError(args.input_csv)
    modes = modes_to_train(args.mode)
    df = load_hybrid_table(args.input_csv)
    print(f"input: {args.input_csv}  n={len(df)}")
    parent = args.output
    parent.mkdir(parents=True, exist_ok=True)
    splits_path = parent / "splits.json"
    if args.splits_file is not None:
        shutil.copy2(args.splits_file, splits_path)
        split = load_split(splits_path)
        print(f"reused splits {args.splits_file}")
    else:
        split = make_scaffold_split(df["SMILES"], splits_path, seed=args.seed)
    for part in ("train", "val", "test"):
        print(f"  {part:5s} {len(split[part]):5d}  {len(split[part]) / len(df):.3f}")

    n_modes = len(modes)
    train_df = df
    chemprop_splits = splits_path
    if args.oversample_heavy:
        train_df, over_split, stats = oversample_train_by_heavy(df, split, seed=args.seed)
        chemprop_splits = parent / "splits_oversample.json"
        chemprop_splits.write_text(json.dumps([over_split]))
        bins = stats["bins_unique_train"]
        print(
            "oversample unique train "
            f"<12 x5: {bins['<12']}, 12-15 x3: {bins['12-15']}, >15 x1: {bins['>15']}"
        )
        print(
            f"  chemprop train rows {stats['unique_train']} -> {stats['oversampled_train']}  "
            f"val {stats['val']}  test {stats['test']}"
        )
        (parent / "oversample_stats.json").write_text(json.dumps(stats, indent=2))

    for mode in modes:
        train_one(mode, train_df, run_dir_for(mode, args.output, n_modes), chemprop_splits, splits_path, args)

    print("=== metrics (train/val/test) ===")
    run_predict(
        input_csvs=[args.input_csv],
        output=parent / "predictions.csv",
        dmpnn_dir=run_dir_for("dmpnn", args.output, n_modes) if "dmpnn" in modes else None,
        hybrid_dir=run_dir_for("hybrid", args.output, n_modes) if "hybrid" in modes else None,
        gpu=args.gpu,
    )


def main() -> None:
    args = parse_args()
    if args.hpopt:
        run_hpopt(args)
    else:
        run_train(args)


if __name__ == "__main__":
    main()
