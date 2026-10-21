#!/usr/bin/env python3
"""Score CSVs with a saved Uni-Mol + descriptor SVR stack.

The model directory is one written by unimol_ft_svr_stack_train.py.
Each input needs the descriptor columns saved in the model. UniMolPred can
come from the input CSV, from --unimol-preds, or from a Uni-Mol checkpoint
passed with --unimol. Labeled rows whose canonical SMILES appear in the
training split are left out of the scores unless --no-overlap is set.
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
    "UniMolPred",
    "SVRPred",
    "UniMol_Error",
    "SVR_Error",
    "Absolute_Error",
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


def write_readable_csv(df: pd.DataFrame, path, decimals: int = 4) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    for column in out.columns:
        if column in {"n_heavy", "heavy"}:
            out[column] = pd.to_numeric(out[column], errors="coerce").astype("Int64")
            continue
        if column == "TrainOverlap":
            out[column] = [bool(value) for value in out[column]]
            continue
        if column not in READABLE_FLOAT_COLS:
            continue
        rounded = []
        for value in out[column]:
            rounded.append(np.nan if pd.isna(value) else round(float(value), decimals))
        out[column] = rounded
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


def regression_metrics(y, prediction) -> dict:
    y = np.asarray(y, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    ok = np.isfinite(y) & np.isfinite(prediction)
    y = y[ok]
    prediction = prediction[ok]
    if len(y) == 0:
        return {"n": 0, "mae": None, "rmse": None, "r2": None, "bias": None}
    return {
        "n": int(len(y)),
        "mae": round(float(mean_absolute_error(y, prediction)), 4),
        "rmse": round(float(np.sqrt(mean_squared_error(y, prediction))), 4),
        "r2": round(float(r2_score(y, prediction)), 4) if len(y) > 1 else None,
        "bias": round(float(np.mean(prediction - y)), 4),
    }


def report_metrics(name: str, y, prediction, n_excluded: int = 0) -> dict:
    scored = regression_metrics(y, prediction)
    def fmt(value):
        return "nan" if value is None else f"{value:.4f}"
    print(
        f"{name:22} n={scored['n']:5d}  MAE={fmt(scored['mae'])}  "
        f"RMSE={fmt(scored['rmse'])}  R2={fmt(scored['r2'])}  "
        f"bias={fmt(scored['bias'])}  excluded_overlap={n_excluded}"
    )
    return scored


def stack_matrix(frame: pd.DataFrame, columns: list[str], unimol_pred: np.ndarray) -> np.ndarray:
    descriptor_columns = [name for name in columns if name != "UniMolPred"]
    missing = [name for name in descriptor_columns if name not in frame.columns]
    if missing:
        raise SystemExit(f"Missing {len(missing)} descriptor columns, e.g. {missing[:8]}")
    tabular = frame[descriptor_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(float)
    return np.hstack([tabular, np.asarray(unimol_pred, dtype=float).reshape(-1, 1)])


def load_model(model: Path):
    path = model / "svr_unimol_stack.joblib" if model.is_dir() else model
    if not path.is_file():
        raise SystemExit(f"Missing {path}")
    payload = joblib.load(path)
    config_path = path.parent / "config.json"
    config = json.loads(config_path.read_text()) if config_path.is_file() else {}
    return payload["estimator"], list(payload["feat_names"]), config, path.parent


def prediction_map(path: Path) -> pd.Series:
    frame = pd.read_csv(path)
    if "ID" not in frame.columns or "UniMolPred" not in frame.columns:
        raise SystemExit(f"{path} needs ID and UniMolPred columns.")
    if frame["ID"].duplicated().any():
        raise SystemExit(f"Duplicate IDs in {path}")
    return frame.set_index("ID")["UniMolPred"].astype(float)


def predict_unimol(smiles: pd.Series, unimol_dir: Path, save_path: Path) -> np.ndarray:
    import unimol_tools

    predictor = unimol_tools.MolPredict(load_model=str(unimol_dir))
    if hasattr(predictor, "config") and "save_path" in predictor.config:
        predictor.config.pop("save_path")
    save_path.mkdir(parents=True, exist_ok=True)
    values = np.asarray(
        predictor.predict(pd.DataFrame({"SMILES": smiles.astype(str)}), save_path=str(save_path)),
        dtype=float,
    ).reshape(-1)
    if len(values) != len(smiles):
        raise SystemExit(f"Uni-Mol returned {len(values)} rows for {len(smiles)} SMILES.")
    return values


def unimol_column(frame: pd.DataFrame, args, conformer_dir: Path) -> np.ndarray:
    if "UniMolPred" in frame.columns and not args.unimol and not args.unimol_preds:
        return pd.to_numeric(frame["UniMolPred"], errors="coerce").to_numpy(float)
    if args.unimol_preds:
        predictions = prediction_map(Path(args.unimol_preds))
        missing = [item for item in frame["ID"] if item not in predictions.index]
        if missing:
            raise SystemExit(f"Uni-Mol prediction file is missing {len(missing)} IDs, e.g. {missing[:5]}")
        return predictions.loc[frame["ID"]].to_numpy(float)
    if args.unimol:
        if "SMILES" not in frame.columns:
            raise SystemExit("Uni-Mol prediction requires a SMILES column.")
        return predict_unimol(frame["SMILES"], Path(args.unimol), conformer_dir)
    raise SystemExit("Provide UniMolPred in the CSV, or pass --unimol-preds, or pass --unimol.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Predict with a Uni-Mol + descriptor SVR stack.")
    parser.add_argument("--model", required=True, help="SVR stack directory or svr_unimol_stack.joblib.")
    parser.add_argument("--input", nargs="+", required=True, help="One or more CSV files.")
    parser.add_argument("-o", "--output", required=True, help="Directory for prediction CSVs.")
    parser.add_argument("--unimol", default="", help="Uni-Mol checkpoint directory. Used when the CSV has no UniMolPred.")
    parser.add_argument("--unimol-preds", default="", help="CSV with ID and UniMolPred to attach before the SVR.")
    parser.add_argument("--overlap-with", default="", help="Training CSV whose SMILES are excluded from the scores.")
    parser.add_argument("--no-overlap", action="store_true", help="Score every labeled row.")
    parser.add_argument("--gpu", default=GPU_UUID, help="CUDA_VISIBLE_DEVICES value. Use cpu to skip the GPU.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    if args.gpu.lower() != "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    estimator, feat_names, config, model_dir = load_model(Path(args.model))
    out_dir = Path(args.output).resolve()
    if out_dir == model_dir.resolve():
        raise SystemExit("Choose an --output directory that is not the model directory.")
    out_dir.mkdir(parents=True, exist_ok=True)

    banned = set()
    if not args.no_overlap:
        overlap_path = Path(args.overlap_with) if args.overlap_with else None
        if overlap_path is None and config.get("train"):
            overlap_path = Path(config["train"])
        if overlap_path is not None and overlap_path.is_file():
            train_df = normalize_table(pd.read_csv(overlap_path))
            if "SMILES" not in train_df.columns:
                raise SystemExit(f"{overlap_path} has no SMILES column.")
            banned = canonical_smiles_set(train_df["SMILES"])
            print(f"Overlap reference: {overlap_path}  n_train_smiles={len(banned)}")

    summary = []
    scored_frames = []
    keep = [
        "ID",
        "SMILES",
        "Entropy",
        "n_heavy",
        "UniMolPred",
        "SVRPred",
        "UniMol_Error",
        "SVR_Error",
        "TrainOverlap",
    ]
    for test in args.input:
        path = Path(test)
        if not path.is_file():
            raise SystemExit(f"Missing input CSV: {path}")
        print(f"Predicting {path}")
        frame = normalize_table(pd.read_csv(path))
        if "ID" not in frame.columns:
            raise SystemExit(f"{path} has no ID column.")
        unimol_pred = unimol_column(frame, args, out_dir / f"{path.stem}_conformers")
        svr_pred = np.asarray(estimator.predict(stack_matrix(frame, feat_names, unimol_pred)), dtype=float)
        pred = frame.copy()
        pred["UniMolPred"] = unimol_pred
        pred["SVRPred"] = svr_pred
        if "Entropy" in pred.columns:
            pred["UniMol_Error"] = np.abs(pred["UniMolPred"] - pred["Entropy"])
            pred["SVR_Error"] = np.abs(pred["SVRPred"] - pred["Entropy"])
        n_excluded = 0
        scored = pred
        if banned and "SMILES" in pred.columns:
            mask = overlap_row_mask(pred, banned)
            pred["TrainOverlap"] = mask.to_numpy()
            n_excluded = int(mask.sum())
            scored = pred.loc[~mask]
        row = {"set": path.stem, "n_rows": int(len(pred)), "excluded_overlap": n_excluded}
        if "Entropy" in scored.columns:
            row["unimol"] = report_metrics(f"{path.stem} Uni-Mol", scored["Entropy"], scored["UniMolPred"], n_excluded)
            row["unimol_svr"] = report_metrics(f"{path.stem} Uni-Mol+SVR", scored["Entropy"], scored["SVRPred"], n_excluded)
            if len(scored):
                scored_frames.append(scored)
        summary.append(row)
        dest = out_dir / f"{path.stem}_preds.csv"
        write_readable_csv(pred[[column for column in keep if column in pred.columns]], dest)
        print(f"Wrote {dest} ({len(pred)} rows)")

    combined = None
    if scored_frames:
        all_df = pd.concat(scored_frames, ignore_index=True)
        combined = {
            "unimol": report_metrics("combined Uni-Mol", all_df["Entropy"], all_df["UniMolPred"]),
            "unimol_svr": report_metrics("combined Uni-Mol+SVR", all_df["Entropy"], all_df["SVRPred"]),
        }
    payload = {"model": str(Path(args.model).resolve()), "sets": summary, "combined": combined}
    (out_dir / "metrics.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Saved {out_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
