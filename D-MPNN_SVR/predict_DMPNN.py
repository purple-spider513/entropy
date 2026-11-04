#!/usr/bin/env python3
"""Predict gas-phase standard molar entropy with models trained by train_DMPNN.py (or train.py).

Self-contained: needs only numpy, pandas and Chemprop 2.3 (the `chemprop` conda env).
Replaces predict.py.

The hybrid model's descriptor list is read from <hybrid-dir>/feature_names.csv, so the
features chosen at training time are used automatically. --feature-file only checks that a
list matches the model.

Each input CSV needs ID, SMILES, entropy and (for the hybrid) the model's descriptor columns.
Writes per-file and combined predictions (entropy, dmpnn_pred, hybrid_pred) and metrics.csv
with MAE, RMSE, R2 and bias. An input whose row count equals the training run's splits.json
is also scored on its train/val/test parts.

Example (run from this folder)
  D=hybrid_data
  conda run --no-capture-output -n chemprop python predict_DMPNN.py \\
      -i $D/hybrid_std109_102926.csv -i $D/hybrid_std128_102926.csv \\
      -i $D/hybrid_pdbbind_102926.csv -i $D/hybrid_std6074_102926.csv \\
      -o eval_my_run2 --dmpnn-dir my_run2/dmpnn --hybrid-dir my_run2/hybrid --gpu 0
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


# ---------------------------------------------------------------- tables and features
def load_table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in (ID_COL, SMILES_COL, TARGET_COL) if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    return df


def feature_matrix(df: pd.DataFrame, columns: list[str]) -> np.ndarray:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing[:8]}{'...' if len(missing) > 8 else ''}")
    X = df.loc[:, columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


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


def check_feature_selection(model_features: list[str], feature_file: Path) -> None:
    wanted = read_feature_list(feature_file)
    if wanted != model_features:
        only_file = [n for n in wanted if n not in model_features]
        only_model = [n for n in model_features if n not in wanted]
        raise ValueError(
            f"{feature_file} ({len(wanted)} features) does not match the hybrid model "
            f"({len(model_features)} features). In file only: {only_file[:8]}; in model only: {only_model[:8]}"
        )


def write_smiles_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df[[SMILES_COL, TARGET_COL]].to_csv(path, index=False)


def write_descriptors_npz(X: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, X)


def load_split(path: Path) -> dict:
    return json.loads(path.read_text())[0]


def model_path(run_dir: Path) -> Path:
    return run_dir / "model_0" / "best.pt"


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


# ---------------------------------------------------------------- prediction
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


def split_candidates(available: dict) -> list[Path]:
    """The training run's unique split: <model dir>/splits.json, then its parent run folder."""
    out: list[Path] = []
    seen: set[Path] = set()
    for spec in available.values():
        for p in (spec["dir"] / "splits.json", spec["dir"].parent / "splits.json"):
            key = p.resolve()
            if key not in seen:
                seen.add(key)
                out.append(p)
    return out


def matching_split(candidates: list[Path], n_rows: int) -> dict | None:
    for path in candidates:
        if path.is_file():
            split = load_split(path)
            if len(split["train"]) + len(split["val"]) + len(split["test"]) == n_rows:
                return split
    return None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-i", "--input", dest="input_csvs", type=Path, action="append", required=True,
                   help="CSV to score. Repeat for several files.")
    p.add_argument("-o", "--output", type=Path, required=True,
                   help="Output folder, or a .csv file when scoring a single input.")
    p.add_argument("--dmpnn-dir", type=Path, default=None, help="D-MPNN model folder (contains model_0/best.pt).")
    p.add_argument("--hybrid-dir", type=Path, default=None, help="Hybrid model folder (contains model_0/best.pt).")
    p.add_argument("--gpu", default="0", help="CUDA device index (CUDA_VISIBLE_DEVICES), or 'cpu'.")
    p.add_argument("--feature-file", type=Path, default=None,
                   help="Optional check: fail unless this list equals the hybrid model's feature_names.csv.")
    p.add_argument("--overwrite", action="store_true", help="Allow writing into an existing non-empty output.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.dmpnn_dir is None and args.hybrid_dir is None:
        raise SystemExit("Pass --dmpnn-dir and/or --hybrid-dir")
    for path in args.input_csvs:
        if not path.is_file():
            raise FileNotFoundError(path)

    available: dict[str, dict] = {}
    for name, run_dir in (("dmpnn", args.dmpnn_dir), ("hybrid", args.hybrid_dir)):
        if run_dir is None:
            continue
        ckpt = model_path(run_dir)
        if not ckpt.is_file():
            raise FileNotFoundError(f"no {name} checkpoint at {ckpt}")
        features = None
        if name == "hybrid":
            features = load_feature_names(run_dir)
            if args.feature_file is not None:
                check_feature_selection(features, args.feature_file)
                print(f"hybrid features match {args.feature_file} ({len(features)} columns)")
            else:
                print(f"hybrid features: {len(features)} columns from {run_dir / 'feature_names.csv'}")
        available[name] = {"dir": run_dir, "ckpt": ckpt, "features": features}

    single_csv = args.output.suffix.lower() == ".csv"
    if single_csv and len(args.input_csvs) > 1:
        raise SystemExit("Use a folder for -o when scoring several inputs.")
    out_dir = args.output.parent if single_csv else args.output
    if (out_dir / "model_0").exists() or (out_dir / "splits.json").exists():
        raise SystemExit(f"{out_dir} holds a trained model or run; choose another output.")
    taken = args.output.exists() if single_csv else (out_dir.exists() and any(out_dir.iterdir()))
    if taken and not args.overwrite:
        raise SystemExit(f"{args.output} already exists. Choose a new -o, or pass --overwrite.")
    out_dir.mkdir(parents=True, exist_ok=True)
    work_dir = out_dir / "_work"
    work_dir.mkdir(parents=True, exist_ok=True)
    candidates = split_candidates(available)

    rows: list[dict] = []
    frames: list[pd.DataFrame] = []
    print(f"{'model':8s} {metrics_header()}")
    for path in args.input_csvs:
        df = load_table(path)
        stem = path.stem
        out = pd.DataFrame({"source": stem, ID_COL: df[ID_COL], SMILES_COL: df[SMILES_COL], TARGET_COL: df[TARGET_COL]})
        y = df[TARGET_COL].to_numpy(dtype=float)
        split = matching_split(candidates, len(df))
        for name, spec in available.items():
            yhat = predict_one(df, work_dir, f"{stem}_{name}", spec["ckpt"], args.gpu, spec["features"])
            out[f"{name}_pred"] = yhat
            m = metrics(y, yhat)
            rows.append({"model": name, "input": stem, "split": "all", **m})
            print(f"{name:8s} {format_metrics_row(stem, m)}")
            if split is not None:
                for part in ("train", "val", "test"):
                    idx = split[part]
                    sm = metrics(y[idx], yhat[idx])
                    rows.append({"model": name, "input": stem, "split": part, **sm})
                    print(f"{name:8s} {format_metrics_row(part, sm)}")
        frames.append(out)

    combined = pd.concat(frames, ignore_index=True)
    pred_path = args.output if single_csv else out_dir / "predictions.csv"
    combined.to_csv(pred_path, index=False)
    if len(frames) > 1:
        for frame, path in zip(frames, args.input_csvs):
            frame.to_csv(out_dir / f"{path.stem}_predictions.csv", index=False)
    metrics_path = out_dir / "metrics.csv"
    write_metrics_csv(rows, metrics_path)
    print_comparison(rows)
    print(f"wrote predictions {pred_path}")
    print(f"wrote metrics {metrics_path}")


if __name__ == "__main__":
    main()
