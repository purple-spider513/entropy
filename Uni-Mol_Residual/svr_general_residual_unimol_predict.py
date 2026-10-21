#!/usr/bin/env python3
"""Predict entropy with an SVR + Uni-Mol residual hybrid.

The model directory is one written by svr_general_residual_unimol_train.py:
tabular.joblib plus unimol/. Prediction reads SMILES and the saved descriptor
columns from each input CSV. Labeled tables also receive MAE, RMSE, R2, and bias.
Rows whose canonical SMILES appear in the training split are left out of the
scores unless --no-overlap is set.

Example:
  python svr_general_residual_unimol_predict.py \
    --model svr_general_residual_run \
    --input hybrid_data/hybrid_std128.csv hybrid_data/hybrid_std109.csv \
    --output svr_general_residual_run_eval
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

GPU_UUID = "GPU-11559549-d592-eb13-a511-44849442d2a5"
READABLE_FLOAT_COLS = {
    "Entropy",
    "entropy",
    "Residual",
    "Predicted_Entropy",
    "Absolute_Error",
    "SVRPred",
    "SVR_Error",
    "Signed_Error",
}


def normalize_table(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "Entropy" not in df.columns and "entropy" in df.columns:
        df["Entropy"] = df["entropy"]
    if "Entropy" in df.columns:
        df["Entropy"] = pd.to_numeric(df["Entropy"], errors="coerce")
    if "n_heavy" not in df.columns and "heavy" in df.columns:
        df["n_heavy"] = pd.to_numeric(df["heavy"], errors="coerce").fillna(1).astype(int)
    return df


def write_readable_csv(df: pd.DataFrame, path, decimals=4) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    for col in out.columns:
        if col in {"n_heavy", "heavy"}:
            out[col] = pd.to_numeric(out[col], errors="coerce").astype("Int64")
            continue
        if col == "TrainOverlap":
            out[col] = [bool(v) for v in out[col]]
            continue
        if col not in READABLE_FLOAT_COLS:
            continue
        rounded = []
        for v in out[col]:
            if pd.isna(v):
                rounded.append(np.nan)
            else:
                rounded.append(round(float(v), decimals))
        out[col] = rounded
    out.to_csv(path, index=False, float_format=f"%.{decimals}f")
    return path


def canon(smiles: str):
    from rdkit import Chem

    if smiles is None or (isinstance(smiles, float) and np.isnan(smiles)):
        return None
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def canonical_smiles_set(smiles) -> set:
    out = set()
    for smi in smiles:
        if pd.isna(smi):
            continue
        value = canon(smi)
        if value:
            out.add(value)
    return out


def overlap_row_mask(df: pd.DataFrame, banned_canons) -> pd.Series:
    banned = set(banned_canons)
    flags = []
    for smi in df["SMILES"]:
        if pd.isna(smi):
            flags.append(False)
            continue
        value = canon(smi)
        flags.append(bool(value) and value in banned)
    return pd.Series(flags, index=df.index)


def regression_metrics(y, p) -> dict:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    ok = np.isfinite(y) & np.isfinite(p)
    y = y[ok]
    p = p[ok]
    if len(y) == 0:
        return {"n": 0, "mae": None, "rmse": None, "r2": None, "bias": None}
    return {
        "n": int(len(y)),
        "mae": round(float(mean_absolute_error(y, p)), 4),
        "rmse": round(float(np.sqrt(mean_squared_error(y, p))), 4),
        "r2": round(float(r2_score(y, p)), 4) if len(y) > 1 else None,
        "bias": round(float(np.mean(p - y)), 4),
    }


def report_metrics(name: str, y, p, n_excluded: int = 0) -> dict:
    stats = regression_metrics(y, p)
    mae = "nan" if stats["mae"] is None else f"{stats['mae']:.4f}"
    rmse = "nan" if stats["rmse"] is None else f"{stats['rmse']:.4f}"
    r2 = "nan" if stats["r2"] is None else f"{stats['r2']:.4f}"
    bias = "nan" if stats["bias"] is None else f"{stats['bias']:.4f}"
    print(
        f"{name:22} n={stats['n']:5d}  MAE={mae}  RMSE={rmse}  "
        f"R2={r2}  bias={bias}  excluded_overlap={n_excluded}"
    )
    return stats


def feature_matrix(df: pd.DataFrame, feat_names: list[str]) -> np.ndarray:
    missing = [c for c in feat_names if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing {len(missing)} features, e.g. {missing[:8]}")
    return df[feat_names].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(float)


def load_model(model_dir: Path):
    tabular = model_dir / "tabular.joblib"
    if not tabular.is_file():
        raise SystemExit(f"Missing {tabular}")
    payload = joblib.load(tabular)
    svr = payload["estimator"]
    feat_names = list(payload["feat_names"])
    config_path = model_dir / "config.json"
    cfg = json.loads(config_path.read_text()) if config_path.is_file() else {}
    unimol_dir = Path(cfg["unimol"]) if cfg.get("unimol") else model_dir / "unimol"
    if not unimol_dir.is_dir():
        raise SystemExit(f"Missing Uni-Mol directory: {unimol_dir}")
    train_csv = cfg.get("train", "")
    return svr, feat_names, unimol_dir, train_csv


def predict_table(df, feat_names, svr, predictor, conformer_dir: Path):
    svr_pred = np.asarray(svr.predict(feature_matrix(df, feat_names)), dtype=float)
    conformer_dir.mkdir(parents=True, exist_ok=True)
    resid = np.asarray(
        predictor.predict(
            pd.DataFrame({"SMILES": df["SMILES"].astype(str)}),
            save_path=str(conformer_dir),
        ),
        dtype=float,
    ).reshape(-1)
    if len(svr_pred) != len(df) or len(resid) != len(df):
        raise SystemExit(
            f"Length mismatch rows={len(df)} svr={len(svr_pred)} unimol={len(resid)}"
        )
    out = df.copy()
    out["SVRPred"] = svr_pred
    out["Residual"] = resid
    out["Predicted_Entropy"] = svr_pred + resid
    if "Entropy" in out.columns:
        y = out["Entropy"].to_numpy(float)
        out["SVR_Error"] = np.abs(svr_pred - y)
        out["Absolute_Error"] = np.abs(out["Predicted_Entropy"] - y)
        out["Signed_Error"] = out["Predicted_Entropy"] - y
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Predict with an SVR + Uni-Mol residual hybrid model."
    )
    parser.add_argument("--model", required=True, help="Model directory with tabular.joblib and unimol/.")
    parser.add_argument(
        "--input",
        nargs="+",
        required=True,
        help="One or more CSV files containing SMILES and the model descriptors.",
    )
    parser.add_argument("-o", "--output", required=True, help="Directory for prediction CSVs. Not the model directory.")
    parser.add_argument(
        "--overlap-with",
        default="",
        help="CSV of training SMILES used to drop overlapping test rows from the scores. Default is the train split recorded in config.json.",
    )
    parser.add_argument(
        "--no-overlap",
        action="store_true",
        help="Score every labeled row, including molecules that also appear in training.",
    )
    parser.add_argument("--gpu", default=GPU_UUID, help="CUDA_VISIBLE_DEVICES value. Use 'cpu' to skip the GPU.")
    return parser


def main():
    args = build_parser().parse_args()
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    if args.gpu.lower() != "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    model_dir = Path(args.model).resolve()
    out_dir = Path(args.output).resolve()
    if out_dir == model_dir:
        raise SystemExit("Choose an --output directory that is not the model directory.")
    svr, feat_names, unimol_dir, recorded_train = load_model(model_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    banned = set()
    if not args.no_overlap:
        overlap_path = Path(args.overlap_with) if args.overlap_with else Path(recorded_train) if recorded_train else None
        if overlap_path is not None and overlap_path.is_file():
            train_df = normalize_table(pd.read_csv(overlap_path))
            if "SMILES" not in train_df.columns:
                raise SystemExit(f"{overlap_path} has no SMILES column.")
            banned = canonical_smiles_set(train_df["SMILES"])
            print(f"Overlap reference: {overlap_path}  n_train_smiles={len(banned)}")
        else:
            print("No training SMILES file found. Scores include every labeled row.")

    import unimol_tools

    predictor = unimol_tools.MolPredict(load_model=str(unimol_dir))
    if hasattr(predictor, "config") and "save_path" in predictor.config:
        predictor.config.pop("save_path")

    summary = []
    scored_frames = []
    keep = [
        "ID",
        "SMILES",
        "Entropy",
        "n_heavy",
        "SVRPred",
        "Residual",
        "Predicted_Entropy",
        "SVR_Error",
        "Absolute_Error",
        "Signed_Error",
        "TrainOverlap",
    ]
    for test in args.input:
        path = Path(test)
        if not path.is_file():
            raise SystemExit(f"Missing input CSV: {path}")
        print(f"Predicting {path}")
        df = normalize_table(pd.read_csv(path))
        if "SMILES" not in df.columns:
            raise SystemExit(f"{path} has no SMILES column.")
        pred = predict_table(df, feat_names, svr, predictor, out_dir / f"{path.stem}_conformers")
        n_excluded = 0
        scored = pred
        if banned and "SMILES" in pred.columns:
            mask = overlap_row_mask(pred, banned)
            pred["TrainOverlap"] = mask.to_numpy()
            n_excluded = int(mask.sum())
            scored = pred.loc[~mask]
        row = {"set": path.stem, "n_rows": int(len(pred)), "excluded_overlap": n_excluded}
        if "Entropy" in scored.columns:
            row["svr"] = report_metrics(f"{path.stem} SVR", scored["Entropy"], scored["SVRPred"], n_excluded)
            row["hybrid"] = report_metrics(
                f"{path.stem} SVR+res",
                scored["Entropy"],
                scored["Predicted_Entropy"],
                n_excluded,
            )
            if len(scored):
                scored_frames.append(scored)
        summary.append(row)
        dest = out_dir / f"{path.stem}_preds.csv"
        write_readable_csv(pred[[c for c in keep if c in pred.columns]], dest)
        print(f"Wrote {dest} ({len(pred)} rows)")

    combined = None
    if scored_frames:
        all_df = pd.concat(scored_frames, ignore_index=True)
        combined = {
            "svr": report_metrics("combined SVR", all_df["Entropy"], all_df["SVRPred"]),
            "hybrid": report_metrics(
                "combined SVR+res", all_df["Entropy"], all_df["Predicted_Entropy"]
            ),
        }
    payload = {"model": str(model_dir), "sets": summary, "combined": combined}
    (out_dir / "metrics.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Saved {out_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
