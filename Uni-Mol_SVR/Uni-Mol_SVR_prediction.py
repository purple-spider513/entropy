#!/usr/bin/env python3
"""Predict S° with a trained Uni-Mol fine-tune + SVR stack (made by Uni-Mol_SVR_train.py).

Self-contained: needs only numpy, pandas, scikit-learn, joblib, RDKit and unimol_tools.

For each input CSV, the fine-tuned Uni-Mol predicts S° from SMILES (UniMolPred),
then the SVR combines it with the descriptor columns stored in the model
(SVRPred, the final prediction). Each input needs SMILES and those descriptor
columns. If it also has an 'entropy' or 'Entropy' column, MAE, RMSE, R² and
bias are reported for UniMolPred and SVRPred.

Labeled rows whose canonical SMILES are in the model's training split are kept
in the output with TrainOverlap=True but left out of the scores, unless
--no-overlap is given.

Example
  python Uni-Mol_SVR_prediction.py --model os3seg_selfeat_svr \\
      --input hybrid_data/hybrid_std128_102926.csv hybrid_data/hybrid_std109_102926.csv \\
      -o os3seg_selfeat_svr_eval
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

GPU_UUID = "GPU-11559549-d592-eb13-a511-44849442d2a5"  # RTX 5070 Ti
MODEL_FILE = "svr_unimol_stack.joblib"
OUTPUT_COLS = ["ID", "SMILES", "Entropy", "n_heavy", "UniMolPred", "SVRPred",
               "UniMol_Error", "SVR_Error", "TrainOverlap"]
READABLE_FLOAT_COLS = {"Entropy", "UniMolPred", "SVRPred", "UniMol_Error", "SVR_Error"}


# ---------------------------------------------------------------- tables
def normalize_table(df: pd.DataFrame) -> pd.DataFrame:
    """Add an 'Entropy' column (from 'entropy') and 'n_heavy' (from 'heavy')."""
    df = df.copy()
    if "Entropy" not in df.columns and "entropy" in df.columns:
        df["Entropy"] = df["entropy"]
    if "Entropy" in df.columns:
        df["Entropy"] = pd.to_numeric(df["Entropy"], errors="coerce")
    if "n_heavy" not in df.columns and "heavy" in df.columns:
        df["n_heavy"] = pd.to_numeric(df["heavy"], errors="coerce").fillna(1).astype(int)
    return df


def write_readable_csv(df: pd.DataFrame, path: Path, decimals: int = 4) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    for col in out.columns:
        if col == "n_heavy":
            out[col] = pd.to_numeric(out[col], errors="coerce").astype("Int64")
        elif col == "TrainOverlap":
            out[col] = [bool(v) for v in out[col]]
        elif col in READABLE_FLOAT_COLS and not pd.api.types.is_integer_dtype(out[col]):
            # Whole-number labels (e.g. std109) stay as written in the input.
            out[col] = [np.nan if pd.isna(v) else round(float(v), decimals) for v in out[col]]
    out.to_csv(path, index=False, float_format=f"%.{decimals}f")


# ---------------------------------------------------------------- overlap and scores
def canon(smiles):
    from rdkit import Chem

    if smiles is None or pd.isna(smiles):
        return None
    mol = Chem.MolFromSmiles(str(smiles))
    return None if mol is None else Chem.MolToSmiles(mol, canonical=True)


def canonical_smiles_set(smiles) -> set:
    return {c for c in (canon(s) for s in smiles) if c}


def overlap_row_mask(df: pd.DataFrame, banned: set) -> pd.Series:
    return pd.Series([bool(c) and c in banned for c in (canon(s) for s in df["SMILES"])], index=df.index)


def regression_metrics(y, p) -> dict:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    ok = np.isfinite(y) & np.isfinite(p)
    y, p = y[ok], p[ok]
    if len(y) == 0:
        return {"n": 0, "mae": None, "rmse": None, "r2": None, "bias": None}
    return {
        "n": int(len(y)),
        "mae": round(float(mean_absolute_error(y, p)), 4),
        "rmse": round(float(np.sqrt(mean_squared_error(y, p))), 4),
        "r2": round(float(r2_score(y, p)), 4) if len(y) > 1 else None,
        "bias": round(float(np.mean(p - y)), 4),
    }


def report(name: str, y, p, n_excluded: int = 0) -> dict:
    m = regression_metrics(y, p)
    fmt = lambda v: "nan" if v is None else f"{v:.4f}"
    print(f"{name:22} n={m['n']:5d}  MAE={fmt(m['mae'])}  RMSE={fmt(m['rmse'])}  "
          f"R2={fmt(m['r2'])}  bias={fmt(m['bias'])}  excluded_overlap={n_excluded}", flush=True)
    return m


# ---------------------------------------------------------------- model
def load_model(model: Path):
    """Return (SVR, descriptor names, Uni-Mol dir, config, model dir)."""
    path = model / MODEL_FILE if model.is_dir() else model
    if not path.is_file():
        raise SystemExit(f"Missing {path}")
    model_dir = path.parent
    payload = joblib.load(path)
    config_path = model_dir / "config.json"
    config = json.loads(config_path.read_text()) if config_path.is_file() else {}
    features = [n for n in payload["feat_names"] if n != "UniMolPred"]
    unimol_dir = model_dir / "unimol"
    if not unimol_dir.is_dir():
        unimol_dir = Path(payload.get("unimol") or config.get("unimol", ""))
    return payload["estimator"], features, unimol_dir, config, model_dir


def unimol_predict(unimol_dir: Path, smiles: pd.Series, conformer_dir: Path) -> np.ndarray:
    from unimol_tools import MolPredict

    predictor = MolPredict(load_model=str(unimol_dir))
    if hasattr(predictor, "config") and "save_path" in predictor.config:
        predictor.config.pop("save_path")
    conformer_dir.mkdir(parents=True, exist_ok=True)
    pred = np.asarray(
        predictor.predict(pd.DataFrame({"SMILES": smiles.astype(str)}), save_path=str(conformer_dir)),
        dtype=float,
    ).reshape(-1)
    if len(pred) != len(smiles):
        raise SystemExit(f"Uni-Mol returned {len(pred)} predictions for {len(smiles)} SMILES")
    return pred


def stack_matrix(frame: pd.DataFrame, features: list[str], unimol_pred) -> np.ndarray:
    missing = [n for n in features if n not in frame.columns]
    if missing:
        raise SystemExit(f"Missing {len(missing)} descriptor columns, e.g. {missing[:8]}")
    tabular = frame[features].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(float)
    return np.hstack([tabular, np.asarray(unimol_pred, dtype=float).reshape(-1, 1)])


def training_smiles_file(config: dict, model_dir: Path, override: str):
    if override:
        return Path(override)
    local = model_dir / "split_train.csv"
    if local.is_file():
        return local
    recorded = Path(config.get("train", ""))
    return recorded if recorded.is_file() else None


# ---------------------------------------------------------------- main
def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help=f"Model directory from Uni-Mol_SVR_train.py, or its {MODEL_FILE}.")
    p.add_argument("--input", nargs="+", required=True, help="One or more CSVs with SMILES and the model's descriptors.")
    p.add_argument("-o", "--output", required=True, help="Directory for predictions. Not the model directory.")
    p.add_argument("--unimol", default="", help="Uni-Mol model directory (default: <model>/unimol).")
    p.add_argument("--overlap-with", default="",
                   help="CSV of training SMILES to leave out of the scores (default: <model>/split_train.csv).")
    p.add_argument("--no-overlap", action="store_true", help="Score every labeled row.")
    p.add_argument("--gpu", default=GPU_UUID, help="CUDA device UUID or index, or 'cpu'.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    os.environ["CUDA_VISIBLE_DEVICES"] = "" if args.gpu.lower() == "cpu" else args.gpu

    estimator, features, unimol_dir, config, model_dir = load_model(Path(args.model))
    if args.unimol:
        unimol_dir = Path(args.unimol)
    if not (unimol_dir / "config.yaml").is_file():
        raise SystemExit(f"No Uni-Mol model in {unimol_dir}; pass --unimol.")
    out_dir = Path(args.output).resolve()
    if out_dir == model_dir.resolve():
        raise SystemExit("Choose an --output directory that is not the model directory.")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Model {model_dir}  descriptors={len(features)}  Uni-Mol {unimol_dir}", flush=True)

    banned = set()
    if not args.no_overlap:
        ref = training_smiles_file(config, model_dir, args.overlap_with)
        if ref is None:
            print("No training SMILES file found; every labeled row is scored.", flush=True)
        else:
            ref_df = pd.read_csv(ref)
            if "SMILES" not in ref_df.columns:
                raise SystemExit(f"{ref} has no SMILES column.")
            banned = canonical_smiles_set(ref_df["SMILES"])
            print(f"Overlap reference: {ref}  n_train_smiles={len(banned)}", flush=True)

    summary, scored_frames = [], []
    for item in args.input:
        path = Path(item)
        if not path.is_file():
            raise SystemExit(f"Missing input CSV: {path}")
        print(f"Predicting {path}", flush=True)
        frame = normalize_table(pd.read_csv(path))
        if "SMILES" not in frame.columns:
            raise SystemExit(f"{path} has no SMILES column.")
        pred = frame.copy()
        pred["UniMolPred"] = unimol_predict(unimol_dir, frame["SMILES"], out_dir / f"{path.stem}_conformers")
        pred["SVRPred"] = estimator.predict(stack_matrix(frame, features, pred["UniMolPred"]))
        labeled = "Entropy" in pred.columns
        if labeled:
            pred["UniMol_Error"] = np.abs(pred["UniMolPred"] - pred["Entropy"])
            pred["SVR_Error"] = np.abs(pred["SVRPred"] - pred["Entropy"])
        n_excluded, scored = 0, pred
        if banned:
            mask = overlap_row_mask(pred, banned)
            pred["TrainOverlap"] = mask.to_numpy()
            n_excluded = int(mask.sum())
            scored = pred.loc[~mask]
        row = {"set": path.stem, "n_rows": int(len(pred)), "excluded_overlap": n_excluded}
        if labeled:
            row["unimol"] = report(f"{path.stem} Uni-Mol", scored["Entropy"], scored["UniMolPred"], n_excluded)
            row["unimol_svr"] = report(f"{path.stem} Uni-Mol+SVR", scored["Entropy"], scored["SVRPred"], n_excluded)
            if len(scored):
                scored_frames.append(scored)
        summary.append(row)
        dest = out_dir / f"{path.stem}_preds.csv"
        write_readable_csv(pred[[c for c in OUTPUT_COLS if c in pred.columns]], dest)
        print(f"Wrote {dest} ({len(pred)} rows)", flush=True)

    combined = None
    if len(scored_frames) > 1:
        all_df = pd.concat(scored_frames, ignore_index=True)
        combined = {
            "unimol": report("combined Uni-Mol", all_df["Entropy"], all_df["UniMolPred"]),
            "unimol_svr": report("combined Uni-Mol+SVR", all_df["Entropy"], all_df["SVRPred"]),
        }
    payload = {"model": str(model_dir.resolve()), "sets": summary, "combined": combined}
    (out_dir / "metrics.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Saved {out_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
