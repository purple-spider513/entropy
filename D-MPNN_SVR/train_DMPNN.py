#!/usr/bin/env python3
"""Train CheMeleon D-MPNN (SMILES only) and/or hybrid (SMILES + descriptors) entropy models.

Self-contained: needs only numpy, pandas, RDKit and Chemprop 2.3 (the `chemprop` conda env).
Merges train.py and the helpers it imported from predict.py.

Feature selection for the hybrid model is a training input:
  (default)                     every extra column of the training CSV except ID, SMILES and
                                entropy; zero-variance columns are dropped
                                (--keep-constant-features keeps them)
  --feature-file selected_features
                                exactly the listed columns (one per line), constant ones included

The list used is saved as <run>/hybrid/feature_names.csv; predict_DMPNN.py reads it from there.

Examples (run from this folder)
  # run2_102926_allfeat recipe: scaffold split, no oversampling, all descriptors
  conda run --no-capture-output -n chemprop python train_DMPNN.py \\
      -i hybrid_data/hybrid_train_102926.csv -o my_run2 --gpu 0

  # run3_102926_allfeat recipe: same split, heavy-atom oversampling of the training part
  conda run --no-capture-output -n chemprop python train_DMPNN.py \\
      -i hybrid_data/hybrid_train_102926.csv -o my_run3 --gpu 0 \\
      --oversample-heavy --splits-file my_run2/splits.json

  # selected-features version (run2_102926): add --feature-file selected_features
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ID_COL = "ID"
SMILES_COL = "SMILES"
TARGET_COL = "entropy"
RESERVED = {ID_COL, SMILES_COL, TARGET_COL}
CONDA_ENV = "chemprop"

# D-MPNN matches the original CheMeleon baseline FFN; the hybrid uses a wider FFN.
MODE_HPARAMS = {
    "dmpnn": {"ffn_hidden_dim": 300, "ffn_num_layers": 1, "dropout": 0.0,
              "warmup_epochs": 2, "init_lr": 1e-4, "max_lr": 1e-3, "final_lr": 1e-4},
    "hybrid": {"ffn_hidden_dim": 1024, "ffn_num_layers": 3, "dropout": 0.1,
               "warmup_epochs": 4, "init_lr": 1e-5, "max_lr": 1e-4, "final_lr": 1e-5},
}
HPARAM_KEYS = ("ffn_hidden_dim", "ffn_num_layers", "dropout", "warmup_epochs", "init_lr", "max_lr", "final_lr")


# ---------------------------------------------------------------- tables and features
def load_table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in (ID_COL, SMILES_COL, TARGET_COL) if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    return df


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in RESERVED]


def feature_matrix(df: pd.DataFrame, columns: list[str]) -> np.ndarray:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing[:8]}{'...' if len(missing) > 8 else ''}")
    X = df.loc[:, columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


def drop_constant_columns(X: np.ndarray, columns: list[str]) -> list[str]:
    return [c for i, c in enumerate(columns) if np.unique(X[:, i]).size > 1]


def read_feature_list(path: Path) -> list[str]:
    names = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if not names:
        raise ValueError(f"{path} has no feature names")
    dup = sorted({n for n in names if names.count(n) > 1})
    if dup:
        raise ValueError(f"{path} repeats features: {dup[:8]}")
    clash = [n for n in names if n in RESERVED]
    if clash:
        raise ValueError(f"{path} lists reserved columns: {clash}")
    return names


def load_feature_names(run_dir: Path) -> list[str]:
    names_path = run_dir / "feature_names.csv"
    if not names_path.is_file():
        raise FileNotFoundError(names_path)
    return pd.read_csv(names_path)["name"].tolist()


def write_smiles_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df[[SMILES_COL, TARGET_COL]].to_csv(path, index=False)


def write_descriptors_npz(X: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, X)


# ---------------------------------------------------------------- splits and oversampling
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


def load_split(path: Path) -> dict:
    return json.loads(path.read_text())[0]


def heavy_copy_number(n_heavy: float) -> int:
    if n_heavy < 12:
        return 5
    if n_heavy <= 15:
        return 3
    return 1


def oversample_train_by_heavy(df: pd.DataFrame, split: dict, seed: int) -> tuple[pd.DataFrame, dict, dict]:
    """Copy training rows by heavy-atom count (<12 x5, 12-15 x3, >15 x1); val/test stay unique."""
    if "heavy" not in df.columns:
        raise ValueError("oversampling requires a 'heavy' column")
    heavy = pd.to_numeric(df["heavy"], errors="coerce").fillna(0).to_numpy()
    rng = np.random.default_rng(seed)
    train_rows: list[int] = []
    bins = {"<12": 0, "12-15": 0, ">15": 0}
    for i in split["train"]:
        n_heavy = float(heavy[i])
        if n_heavy < 12:
            bins["<12"] += 1
        elif n_heavy <= 15:
            bins["12-15"] += 1
        else:
            bins[">15"] += 1
        train_rows.extend([int(i)] * heavy_copy_number(n_heavy))
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


# ---------------------------------------------------------------- chemprop command line
def chemprop_command() -> list[str]:
    """Use the chemprop executable of the running env; otherwise go through `conda run`."""
    local = Path(sys.executable).parent / "chemprop"
    if local.is_file():
        return [str(local)]
    conda = shutil.which("conda")
    if conda:
        return [conda, "run", "--no-capture-output", "-n", CONDA_ENV, "chemprop"]
    return ["chemprop"]


def run_chemprop(args: list[str], gpu: str | None) -> None:
    env = os.environ.copy()
    cmd = [*chemprop_command(), *args]
    if gpu is None or str(gpu).lower() == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""
        cmd.extend(["--accelerator", "cpu"])
    else:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        cmd.extend(["--accelerator", "gpu", "--devices", "1"])
    print(f"CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']} " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env=env)


def model_path(run_dir: Path) -> Path:
    return run_dir / "model_0" / "best.pt"


# ---------------------------------------------------------------- scoring
def metrics(y: np.ndarray, yhat: np.ndarray) -> dict[str, float]:
    err = yhat - y
    denom = ((y - y.mean()) ** 2).sum()
    r2 = 1.0 - (err**2).sum() / denom if denom > 0 else float("nan")
    return {
        "n": int(len(y)),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "r2": float(r2),
        "bias": float(err.mean()),
    }


def format_metrics_row(name: str, m: dict[str, float]) -> str:
    return f"{name:24s} {m['n']:5d} {m['mae']:8.2f} {m['rmse']:8.2f} {m['r2']:8.4f} {m['bias']:8.2f}"


def metrics_header() -> str:
    return f"{'split':24s} {'n':>5} {'MAE':>8} {'RMSE':>8} {'R2':>8} {'bias':>8}"


def predict_one(df: pd.DataFrame, work_dir: Path, stem: str, ckpt: Path, gpu: str,
                feature_names: list[str] | None) -> np.ndarray:
    smiles_csv = work_dir / f"{stem}_input.csv"
    pred_csv = work_dir / f"{stem}_raw.csv"
    write_smiles_csv(df, smiles_csv)
    cmd = ["predict", "-i", str(smiles_csv), "-o", str(pred_csv), "--model-paths", str(ckpt)]
    if feature_names is not None:
        npz_path = work_dir / f"{stem}_descriptors.npz"
        write_descriptors_npz(feature_matrix(df, feature_names), npz_path)
        cmd.extend(["--descriptors-path", str(npz_path)])
    run_chemprop(cmd, gpu)
    yhat = pd.read_csv(pred_csv)[TARGET_COL].to_numpy(dtype=float)
    if len(yhat) != len(df):
        raise RuntimeError(f"prediction length {len(yhat)} != input {len(df)}")
    return yhat


def write_metrics_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["model", "input", "split", "n", "mae", "rmse", "r2", "bias"]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def print_comparison(rows: list[dict]) -> None:
    by_key: dict[tuple[str, str], dict[str, dict]] = {}
    for row in rows:
        by_key.setdefault((row["input"], row["split"]), {})[row["model"]] = row
    print()
    print(f"{'input':20s} {'split':8s} {'n':>5} {'dmpnn_MAE':>10} {'dmpnn_R2':>9} {'hybrid_MAE':>11} {'hybrid_R2':>10}")
    for (name, split), models in by_key.items():
        n = next(iter(models.values()))["n"]
        d, h = models.get("dmpnn"), models.get("hybrid")
        d_mae = f"{d['mae']:10.2f}" if d else f"{'—':>10}"
        d_r2 = f"{d['r2']:9.4f}" if d else f"{'—':>9}"
        h_mae = f"{h['mae']:11.2f}" if h else f"{'—':>11}"
        h_r2 = f"{h['r2']:10.4f}" if h else f"{'—':>10}"
        print(f"{name:20s} {split:8s} {n:5d} {d_mae} {d_r2} {h_mae} {h_r2}")


def score_training_table(input_csv: Path, run_root: Path, split: dict, model_dirs: dict[str, Path], gpu: str) -> None:
    """Predict the unique training table and report all/train/val/test (writes predictions.csv, metrics.csv)."""
    df = load_table(input_csv)
    stem = input_csv.stem
    work_dir = run_root / "_work"
    work_dir.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame({"source": stem, ID_COL: df[ID_COL], SMILES_COL: df[SMILES_COL], TARGET_COL: df[TARGET_COL]})
    y = df[TARGET_COL].to_numpy(dtype=float)
    rows: list[dict] = []
    print(f"{'model':8s} {metrics_header()}")
    for name, run_dir in model_dirs.items():
        features = load_feature_names(run_dir) if name == "hybrid" else None
        yhat = predict_one(df, work_dir, f"{stem}_{name}", model_path(run_dir), gpu, features)
        out[f"{name}_pred"] = yhat
        m = metrics(y, yhat)
        rows.append({"model": name, "input": stem, "split": "all", **m})
        print(f"{name:8s} {format_metrics_row(stem, m)}")
        for part in ("train", "val", "test"):
            idx = split[part]
            sm = metrics(y[idx], yhat[idx])
            rows.append({"model": name, "input": stem, "split": part, **sm})
            print(f"{name:8s} {format_metrics_row(part, sm)}")
    out.to_csv(run_root / "predictions.csv", index=False)
    write_metrics_csv(rows, run_root / "metrics.csv")
    print_comparison(rows)
    print(f"wrote predictions {run_root / 'predictions.csv'}")
    print(f"wrote metrics {run_root / 'metrics.csv'}")


# ---------------------------------------------------------------- training
def hparams_for(mode: str, args: argparse.Namespace) -> dict:
    hp = dict(MODE_HPARAMS[mode])
    for key in HPARAM_KEYS:
        val = getattr(args, key)
        if val is not None:
            hp[key] = val
    return hp


def prepare_hybrid_features(df: pd.DataFrame, out_dir: Path, keep_constant: bool, feature_file: Path | None) -> Path:
    if feature_file is not None:
        cols = read_feature_list(feature_file)
        X = feature_matrix(df, cols)
        n_const = len(cols) - len(drop_constant_columns(X, cols))
        shutil.copy2(feature_file, out_dir / "selected_features")
        print(f"feature file {feature_file}: {len(cols)} columns ({n_const} constant in this training set)")
    else:
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


def train_one(mode: str, df: pd.DataFrame, out_dir: Path, chemprop_splits: Path, unique_splits: Path,
              args: argparse.Namespace) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    hp = hparams_for(mode, args)
    smiles_csv = out_dir / "train.csv"
    write_smiles_csv(df, smiles_csv)
    dest_splits = out_dir / "splits_chemprop.json"
    shutil.copy2(chemprop_splits, dest_splits)
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
        npz_path = prepare_hybrid_features(df, out_dir, args.keep_constant_features, args.feature_file)
        cmd.extend(["--descriptors-path", str(npz_path)])
    print(f"=== train {mode} -> {out_dir} ===", flush=True)
    unique_bytes = unique_splits.read_bytes()
    run_chemprop(cmd, args.gpu)
    # Chemprop writes its own (possibly oversampled) splits.json into out_dir, which is the run
    # folder itself in single-mode runs; put the unique-molecule split back.
    (out_dir / "splits.json").write_bytes(unique_bytes)
    print(f"done {mode}. checkpoint: {model_path(out_dir)}", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-i", "--input", dest="input_csv", type=Path, required=True,
                   help="Training CSV with ID, SMILES, entropy and descriptor columns (and 'heavy' for --oversample-heavy).")
    p.add_argument("-o", "--output", type=Path, required=True, help="New run folder.")
    p.add_argument("--mode", choices=("both", "dmpnn", "hybrid"), default="both",
                   help="Models to train. 'both' writes <output>/dmpnn and <output>/hybrid; one mode writes into <output>.")
    p.add_argument("--feature-file", type=Path, default=None,
                   help="Hybrid: use exactly these columns (one per line, e.g. selected_features). "
                        "Default: every extra column of the training CSV.")
    p.add_argument("--keep-constant-features", action="store_true",
                   help="Hybrid without --feature-file: keep zero-variance columns.")
    p.add_argument("--oversample-heavy", action="store_true",
                   help="Copy training molecules by heavy-atom count: <12 x5, 12-15 x3, >15 x1. Val/test stay unique.")
    p.add_argument("--splits-file", type=Path, default=None,
                   help="Reuse an existing unique-molecule splits.json (e.g. the run2 split for run3).")
    p.add_argument("--gpu", default="0", help="CUDA device index (CUDA_VISIBLE_DEVICES), or 'cpu'.")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--seed", type=int, default=0, help="Scaffold-split and oversampling seed.")
    p.add_argument("--num-workers", type=int, default=4)
    for key in HPARAM_KEYS:
        p.add_argument(f"--{key.replace('_', '-')}", type=int if key in ("ffn_hidden_dim", "ffn_num_layers", "warmup_epochs") else float,
                       default=None, help=f"Override {key} for every trained model.")
    p.add_argument("--overwrite", action="store_true", help="Allow an output folder that is not empty.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input_csv.is_file():
        raise FileNotFoundError(args.input_csv)
    out = args.output
    if out.exists() and any(out.iterdir()) and not args.overwrite:
        raise SystemExit(f"{out} is not empty. Choose a new -o, or pass --overwrite to replace its contents.")
    if args.splits_file is not None:
        if not args.splits_file.is_file():
            raise FileNotFoundError(args.splits_file)
        if args.splits_file.resolve() == (out / "splits.json").resolve():
            raise SystemExit("--splits-file must not be the output folder's own splits.json")
    if args.feature_file is not None and not args.feature_file.is_file():
        raise FileNotFoundError(args.feature_file)

    modes = ["dmpnn", "hybrid"] if args.mode == "both" else [args.mode]
    df = load_table(args.input_csv)
    print(f"input: {args.input_csv}  n={len(df)}")
    if args.feature_file is not None and "hybrid" in modes:
        feature_matrix(df, read_feature_list(args.feature_file))  # fail early if a column is missing

    out.mkdir(parents=True, exist_ok=True)
    splits_path = out / "splits.json"
    if args.splits_file is not None:
        shutil.copy2(args.splits_file, splits_path)
        split = load_split(splits_path)
        if len(split["train"]) + len(split["val"]) + len(split["test"]) != len(df):
            raise SystemExit(f"{args.splits_file} does not cover the {len(df)} rows of {args.input_csv}")
        print(f"reused splits {args.splits_file}")
    else:
        split = make_scaffold_split(df[SMILES_COL], splits_path, seed=args.seed)
    for part in ("train", "val", "test"):
        print(f"  {part:5s} {len(split[part]):5d}  {len(split[part]) / len(df):.3f}")

    train_df, chemprop_splits = df, splits_path
    if args.oversample_heavy:
        train_df, over_split, stats = oversample_train_by_heavy(df, split, seed=args.seed)
        chemprop_splits = out / "splits_oversample.json"
        chemprop_splits.write_text(json.dumps([over_split]))
        bins = stats["bins_unique_train"]
        print(f"oversample unique train <12 x5: {bins['<12']}, 12-15 x3: {bins['12-15']}, >15 x1: {bins['>15']}")
        print(f"  chemprop train rows {stats['unique_train']} -> {stats['oversampled_train']}  "
              f"val {stats['val']}  test {stats['test']}")
        (out / "oversample_stats.json").write_text(json.dumps(stats, indent=2))

    model_dirs = {mode: (out / mode if len(modes) > 1 else out) for mode in modes}
    for mode in modes:
        train_one(mode, train_df, model_dirs[mode], chemprop_splits, splits_path, args)

    print("=== metrics (train/val/test) ===")
    score_training_table(args.input_csv, out, split, model_dirs, args.gpu)


if __name__ == "__main__":
    main()
