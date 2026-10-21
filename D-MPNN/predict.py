#!/usr/bin/env python3
"""Predict gas-phase standard molar entropy with D-MPNN and/or hybrid models.

  conda run --no-capture-output -n chemprop python predict.py \\
      -i hybrid_data/hybrid_std128.csv -o preds/std128.csv \\
      --dmpnn-dir run2/dmpnn --hybrid-dir run2/hybrid --gpu 0

Required: -i (repeat for several CSVs) and the model folders. Writes a CSV with
true entropy plus dmpnn_pred / hybrid_pred, and prints MAE and R2.
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

ROOT = Path(__file__).resolve().parent
ID_COL = "ID"
SMILES_COL = "SMILES"
TARGET_COL = "entropy"
RESERVED = {ID_COL, SMILES_COL, TARGET_COL}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "-i",
        "--input",
        dest="input_csvs",
        type=Path,
        action="append",
        required=True,
        help="Input CSV to score. Repeat for multiple files.",
    )
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="Output CSV file or folder.",
    )
    p.add_argument(
        "--dmpnn-dir",
        type=Path,
        default=None,
        help="D-MPNN run folder containing model_0/best.pt.",
    )
    p.add_argument(
        "--hybrid-dir",
        type=Path,
        default=None,
        help="Hybrid run folder containing model_0/best.pt.",
    )
    p.add_argument("--gpu", default="0", help="nvidia-smi GPU id, or 'cpu'")
    return p.parse_args()


def resolve_output_path(path: Path, default_name: str) -> tuple[Path, Path]:
    if path.suffix.lower() == ".csv":
        path.parent.mkdir(parents=True, exist_ok=True)
        return path, path.parent
    path.mkdir(parents=True, exist_ok=True)
    return path / default_name, path


def load_hybrid_table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in (ID_COL, SMILES_COL, TARGET_COL) if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    return df


def feature_matrix(df: pd.DataFrame, columns: list[str]) -> np.ndarray:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing feature columns: {missing[:8]}{'...' if len(missing) > 8 else ''}"
        )
    X = df.loc[:, columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


def write_smiles_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df[[SMILES_COL, TARGET_COL]].to_csv(path, index=False)


def write_descriptors_npz(X: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, X)


def load_split(path: Path) -> dict:
    return json.loads(path.read_text())[0]


def load_feature_names(run_dir: Path) -> list[str]:
    names_path = run_dir / "feature_names.csv"
    if not names_path.is_file():
        raise FileNotFoundError(names_path)
    return pd.read_csv(names_path)["name"].tolist()


def model_path(out_dir: Path) -> Path:
    return out_dir / "model_0" / "best.pt"


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


def predict_one(
    df: pd.DataFrame,
    work_dir: Path,
    stem: str,
    ckpt: Path,
    gpu: str,
    feature_names: list[str] | None,
) -> pd.Series:
    smiles_csv = work_dir / f"{stem}_input.csv"
    pred_csv = work_dir / f"{stem}_raw.csv"
    write_smiles_csv(df, smiles_csv)
    cmd = [
        "predict",
        "-i", str(smiles_csv),
        "-o", str(pred_csv),
        "--model-paths", str(ckpt),
    ]
    if feature_names is not None:
        npz_path = work_dir / f"{stem}_descriptors.npz"
        write_descriptors_npz(feature_matrix(df, feature_names), npz_path)
        cmd.extend(["--descriptors-path", str(npz_path)])
    run_chemprop(cmd, gpu)
    yhat = pd.read_csv(pred_csv)[TARGET_COL].to_numpy(dtype=float)
    if len(yhat) != len(df):
        raise RuntimeError(f"prediction length {len(yhat)} != input {len(df)}")
    return pd.Series(yhat, index=df.index)


def split_candidates(available: dict, extra: list[Path] | None = None) -> list[Path]:
    paths: list[Path] = []
    for spec in available.values():
        d = spec["dir"]
        paths.extend([d / "splits.json", d.parent / "splits.json"])
    if extra:
        paths.extend(extra)
    seen: set[Path] = set()
    out: list[Path] = []
    for p in paths:
        key = p.resolve() if p.exists() else p
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out


def matching_split(candidates: list[Path], n_rows: int) -> dict | None:
    for path in candidates:
        if not path.is_file():
            continue
        splits = load_split(path)
        n_split = len(splits["train"]) + len(splits["val"]) + len(splits["test"])
        if n_split == n_rows:
            return splits
    return None


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
        key = (row["input"], row["split"])
        by_key.setdefault(key, {})[row["model"]] = row
    print()
    print(
        f"{'input':20s} {'split':8s} {'n':>5} "
        f"{'dmpnn_MAE':>10} {'dmpnn_R2':>9} {'hybrid_MAE':>11} {'hybrid_R2':>10}"
    )
    for (name, split), models in by_key.items():
        n = next(iter(models.values()))["n"]
        d = models.get("dmpnn")
        h = models.get("hybrid")
        d_mae = f"{d['mae']:10.2f}" if d else f"{'—':>10}"
        d_r2 = f"{d['r2']:9.4f}" if d else f"{'—':>9}"
        h_mae = f"{h['mae']:11.2f}" if h else f"{'—':>11}"
        h_r2 = f"{h['r2']:10.4f}" if h else f"{'—':>10}"
        print(f"{name:20s} {split:8s} {n:5d} {d_mae} {d_r2} {h_mae} {h_r2}")


def run_predict(
    input_csvs: list[Path],
    output: Path,
    dmpnn_dir: Path | None = None,
    hybrid_dir: Path | None = None,
    gpu: str = "0",
) -> list[dict]:
    if not input_csvs:
        raise ValueError("Pass at least one input CSV with -i / --input")
    models = {
        "dmpnn": {"dir": dmpnn_dir, "ckpt": model_path(dmpnn_dir) if dmpnn_dir else None, "features": None},
        "hybrid": {"dir": hybrid_dir, "ckpt": model_path(hybrid_dir) if hybrid_dir else None, "features": None},
    }
    available = {}
    for name, spec in models.items():
        ckpt = spec["ckpt"]
        if spec["dir"] is not None and ckpt is not None and ckpt.is_file():
            if name == "hybrid":
                spec["features"] = load_feature_names(spec["dir"])
            available[name] = spec
        elif spec["dir"] is not None:
            print(f"warning: no {name} checkpoint at {ckpt} (skipping)")
    if not available:
        raise FileNotFoundError("Pass --dmpnn-dir and/or --hybrid-dir with model_0/best.pt")
    if "dmpnn" not in available or "hybrid" not in available:
        print("warning: reporting the trained model(s) only; train the missing one to compare both")

    for path in input_csvs:
        if not path.is_file():
            raise FileNotFoundError(path)

    out_csv, out_dir = resolve_output_path(output, "predictions.csv")
    work_dir = out_dir / "_work"
    work_dir.mkdir(parents=True, exist_ok=True)
    candidates = split_candidates(available, [out_dir / "splits.json"])

    metric_rows: list[dict] = []
    combined_frames: list[pd.DataFrame] = []
    print(f"{'model':8s} {metrics_header()}")

    for path in input_csvs:
        df = load_hybrid_table(path)
        stem = path.stem
        out = pd.DataFrame({
            "source": stem,
            ID_COL: df[ID_COL],
            SMILES_COL: df[SMILES_COL],
            TARGET_COL: df[TARGET_COL],
        })
        y = df[TARGET_COL].to_numpy(dtype=float)
        splits = matching_split(candidates, len(df))

        for model_name, spec in available.items():
            yhat = predict_one(
                df, work_dir, f"{stem}_{model_name}", spec["ckpt"], gpu, spec["features"]
            )
            out[f"{model_name}_pred"] = yhat.to_numpy()
            m = metrics(y, yhat.to_numpy())
            metric_rows.append({"model": model_name, "input": stem, "split": "all", **m})
            print(f"{model_name:8s} {format_metrics_row(stem, m)}")
            if splits is not None:
                for part in ("train", "val", "test"):
                    idx = splits[part]
                    sm = metrics(y[idx], yhat.to_numpy()[idx])
                    metric_rows.append({"model": model_name, "input": stem, "split": part, **sm})
                    print(f"{model_name:8s} {format_metrics_row(part, sm)}")
        combined_frames.append(out)

    combined = pd.concat(combined_frames, ignore_index=True)
    if len(input_csvs) == 1:
        combined.to_csv(out_csv, index=False)
        pred_path = out_csv
    else:
        pred_path = out_dir / "predictions.csv" if out_csv.name == "predictions.csv" else out_csv
        combined.to_csv(pred_path, index=False)
        for frame, path in zip(combined_frames, input_csvs):
            frame.to_csv(out_dir / f"{path.stem}_predictions.csv", index=False)

    metrics_path = out_dir / "metrics.csv"
    write_metrics_csv(metric_rows, metrics_path)
    print_comparison(metric_rows)
    print(f"wrote predictions {pred_path}")
    print(f"wrote metrics {metrics_path}")
    return metric_rows


def main() -> None:
    args = parse_args()
    if args.dmpnn_dir is None and args.hybrid_dir is None:
        raise ValueError("Pass --dmpnn-dir and/or --hybrid-dir")
    run_predict(
        input_csvs=args.input_csvs,
        output=args.output,
        dmpnn_dir=args.dmpnn_dir,
        hybrid_dir=args.hybrid_dir,
        gpu=args.gpu,
    )


if __name__ == "__main__":
    main()
