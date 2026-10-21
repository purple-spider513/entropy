#!/usr/bin/env python3
"""Fit an unscaled SVR on descriptors plus a Uni-Mol entropy prediction.

This is the stack used by
unimol_ft_entropy_no_oversample_091926_80_10_10_svr and
unimol_ft_entropy_os3seg_091926_80_10_10_svr. Uni-Mol is not retrained.
Pass the finished Uni-Mol run that already contains train, validation, and
test prediction CSVs. --oversample records which Uni-Mol run you are stacking:

  none   Uni-Mol trained on the unique 80% rows
  3seg   Uni-Mol trained with n<12 x5, 12-15 x3, >15 x1

SVR itself is always fit on the unique 80% rows. The feature vector is every
CSV descriptor except identifiers, SMILES, entropy, and heavy, plus UniMolPred.
SVR is unscaled RBF, C=10000, epsilon=0.1.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR

SKIP_COLS = {
    "ID",
    "entropy",
    "Entropy",
    "SMILES",
    "Ligands",
    "n_heavy",
    "heavy",
    "canon",
    "LinearPred",
    "Residual",
    "HybridPred",
    "Predicted_Entropy",
    "UniMolPred",
    "FeatPred",
    "Absolute_Error",
    "UniMol_Error",
    "Feat_Error",
    "SVRPred",
    "SVR_Error",
    "TrainOverlap",
}
READABLE_FLOAT_COLS = {
    "Entropy",
    "entropy",
    "UniMolPred",
    "FeatPred",
    "Predicted_Entropy",
    "Absolute_Error",
    "UniMol_Error",
    "Feat_Error",
    "SVRPred",
    "SVR_Error",
}

SVR_KERNEL = "rbf"
SVR_C = 10000.0
SVR_EPSILON = 0.1
SVR_GAMMA = "scale"
SVR_CACHE_MB = 4096
SPLIT_SEED = 42
HOLD_FRACTION = 0.2


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [column for column in df.columns if column not in SKIP_COLS]


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


def regression_metrics(y, prediction) -> dict:
    y = np.asarray(y, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    return {
        "n": int(len(y)),
        "mae": round(float(mean_absolute_error(y, prediction)), 4),
        "rmse": round(float(np.sqrt(mean_squared_error(y, prediction))), 4),
        "r2": round(float(r2_score(y, prediction)), 4),
        "bias": round(float(np.mean(prediction - y)), 4),
    }


def report_metrics(name: str, y, prediction) -> dict:
    scored = regression_metrics(y, prediction)
    print(
        f"{name:22} n={scored['n']:5d}  MAE={scored['mae']:.4f}  "
        f"RMSE={scored['rmse']:.4f}  R2={scored['r2']:.4f}  bias={scored['bias']:.4f}"
    )
    return scored


def split_frame(frame: pd.DataFrame, hold_fraction: float, seed: int):
    indices = np.arange(len(frame))
    train_idx, temp_idx = train_test_split(
        indices, test_size=hold_fraction, random_state=seed, shuffle=True
    )
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=seed, shuffle=True)
    return (
        frame.iloc[train_idx].reset_index(drop=True),
        frame.iloc[val_idx].reset_index(drop=True),
        frame.iloc[test_idx].reset_index(drop=True),
    )


def prediction_map(path: Path) -> pd.Series:
    frame = pd.read_csv(path, usecols=["ID", "UniMolPred"])
    if frame["ID"].duplicated().any():
        raise SystemExit(f"Duplicate IDs in {path}")
    return frame.set_index("ID")["UniMolPred"].astype(float)


def attach_unimol(frame: pd.DataFrame, predictions: pd.Series, label: str) -> np.ndarray:
    missing = [item for item in frame["ID"] if item not in predictions.index]
    if missing:
        raise SystemExit(f"{label} is missing UniMolPred for {len(missing)} IDs, e.g. {missing[:5]}")
    return predictions.loc[frame["ID"]].to_numpy(float)


def stack_matrix(frame: pd.DataFrame, columns: list[str], unimol_pred: np.ndarray) -> np.ndarray:
    missing = [name for name in columns if name not in frame.columns]
    if missing:
        raise SystemExit(f"Missing {len(missing)} descriptor columns, e.g. {missing[:8]}")
    tabular = frame[columns].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(float)
    return np.hstack([tabular, np.asarray(unimol_pred, dtype=float).reshape(-1, 1)])


def score_frame(frame, columns, unimol_pred, estimator) -> pd.DataFrame:
    prediction = estimator.predict(stack_matrix(frame, columns, unimol_pred))
    out = frame[["ID", "SMILES", "Entropy"]].copy()
    if "n_heavy" in frame.columns:
        out["n_heavy"] = frame["n_heavy"]
    out["UniMolPred"] = unimol_pred
    out["SVRPred"] = prediction
    out["UniMol_Error"] = np.abs(out["UniMolPred"] - out["Entropy"])
    out["SVR_Error"] = np.abs(out["SVRPred"] - out["Entropy"])
    return out


def load_internal_predictions(unimol_source: Path) -> pd.Series:
    pieces = []
    for name in ("train_predictions.csv", "val_10pct_preds.csv", "test_10pct_preds.csv"):
        path = unimol_source / name
        if not path.is_file():
            raise SystemExit(f"Missing {path}")
        pieces.append(prediction_map(path))
    return pd.concat(pieces)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the Uni-Mol prediction + descriptor SVR stack.")
    parser.add_argument("--train", required=True, help="Training CSV with SMILES, entropy, descriptors, and ID.")
    parser.add_argument(
        "--unimol-source",
        required=True,
        help="Finished Uni-Mol run with train_predictions.csv, val_10pct_preds.csv, and test_10pct_preds.csv.",
    )
    parser.add_argument("-o", "--output", required=True, help="New directory for the SVR stack.")
    parser.add_argument(
        "--oversample",
        choices=["none", "3seg"],
        required=True,
        help="none matches the no-oversample Uni-Mol run. 3seg matches n<12 x5, 12-15 x3, >15 x1.",
    )
    parser.add_argument("--hold-fraction", type=float, default=HOLD_FRACTION)
    parser.add_argument("--split-seed", type=int, default=SPLIT_SEED)
    parser.add_argument("--svr-kernel", default=SVR_KERNEL)
    parser.add_argument("--svr-C", type=float, default=SVR_C)
    parser.add_argument("--svr-epsilon", type=float, default=SVR_EPSILON)
    parser.add_argument("--svr-gamma", default=SVR_GAMMA)
    parser.add_argument("--svr-cache-mb", type=float, default=SVR_CACHE_MB)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    train_path = Path(args.train)
    unimol_source = Path(args.unimol_source)
    out_dir = Path(args.output)
    if not train_path.is_file():
        raise SystemExit(f"Missing {train_path}")
    if not unimol_source.is_dir():
        raise SystemExit(f"Missing Uni-Mol run {unimol_source}")
    if out_dir.resolve() == unimol_source.resolve():
        raise SystemExit("Choose an output directory that is not the Uni-Mol source.")
    if (out_dir / "config.json").exists() or (out_dir / "svr_unimol_stack.joblib").exists():
        raise SystemExit(f"{out_dir} already contains a model. Choose a new --output directory.")

    table = normalize_table(pd.read_csv(train_path))
    if "ID" not in table.columns or "SMILES" not in table.columns or "Entropy" not in table.columns:
        raise SystemExit("Training CSV must contain ID, SMILES, and entropy.")
    columns = feature_columns(table)
    train_df, val_df, test_df = split_frame(table, args.hold_fraction, args.split_seed)

    saved_train = set(pd.read_csv(unimol_source / "train_predictions.csv", usecols=["ID"])["ID"])
    if set(train_df["ID"]) != saved_train:
        raise SystemExit("The 80% split IDs do not match the saved Uni-Mol training predictions.")

    internal_pred = load_internal_predictions(unimol_source)
    train_um = attach_unimol(train_df, internal_pred, "train")
    val_um = attach_unimol(val_df, internal_pred, "val")
    test_um = attach_unimol(test_df, internal_pred, "test")

    estimator = SVR(
        kernel=args.svr_kernel,
        C=args.svr_C,
        epsilon=args.svr_epsilon,
        gamma=args.svr_gamma,
        cache_size=args.svr_cache_mb,
    )
    print(
        f"oversample={args.oversample}  n_train={len(train_df)}  "
        f"descriptors={len(columns)}  stack={len(columns) + 1}"
    )
    print(
        f"Fitting unscaled SVR kernel={args.svr_kernel} C={args.svr_C} "
        f"epsilon={args.svr_epsilon} gamma={args.svr_gamma}"
    )
    estimator.fit(stack_matrix(train_df, columns, train_um), train_df["Entropy"].to_numpy(float))
    print(f"SVR support vectors: {len(estimator.support_)}")

    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "svr_unimol_stack.joblib"
    joblib.dump(
        {
            "estimator": estimator,
            "feat_names": columns + ["UniMolPred"],
            "kind": "svr_unimol_feat_stack",
            "kernel": args.svr_kernel,
            "C": args.svr_C,
            "epsilon": args.svr_epsilon,
            "gamma": args.svr_gamma,
            "scaled": False,
            "unimol_source": str(unimol_source),
            "oversample": args.oversample,
        },
        model_path,
    )
    print(f"Saved {model_path}")

    train_out = score_frame(train_df, columns, train_um, estimator)
    val_out = score_frame(val_df, columns, val_um, estimator)
    test_out = score_frame(test_df, columns, test_um, estimator)
    print("Internal splits:")
    split_metrics = {
        "train_unimol": report_metrics("train Uni-Mol", train_out["Entropy"], train_out["UniMolPred"]),
        "train_unimol_svr": report_metrics("train Uni-Mol+SVR", train_out["Entropy"], train_out["SVRPred"]),
        "val_unimol": report_metrics("val Uni-Mol", val_out["Entropy"], val_out["UniMolPred"]),
        "val_unimol_svr": report_metrics("val Uni-Mol+SVR", val_out["Entropy"], val_out["SVRPred"]),
        "test_unimol": report_metrics("test Uni-Mol", test_out["Entropy"], test_out["UniMolPred"]),
        "test_unimol_svr": report_metrics("test Uni-Mol+SVR", test_out["Entropy"], test_out["SVRPred"]),
    }
    write_readable_csv(train_out, out_dir / "train_predictions.csv")
    write_readable_csv(train_df[["ID", "SMILES", "Entropy"]], out_dir / "split_train.csv")
    write_readable_csv(val_out, out_dir / "val_10pct_preds.csv")
    write_readable_csv(test_out, out_dir / "test_10pct_preds.csv")
    (out_dir / "split_metrics.json").write_text(json.dumps(split_metrics, indent=2) + "\n")

    scheme = "n<12 x5, 12<=n<=15 x3, n>15 x1" if args.oversample == "3seg" else "none"
    config = {
        "kind": "unimol_ft_svr_stack",
        "train_source": str(train_path),
        "unimol_source": str(unimol_source),
        "output": str(out_dir),
        "model": str(model_path),
        "train": str(out_dir / "split_train.csv"),
        "n_pool": int(len(table)),
        "n_train": int(len(train_df)),
        "n_val": int(len(val_df)),
        "n_test": int(len(test_df)),
        "oversample": args.oversample == "3seg",
        "oversample_scheme": scheme,
        "n_descriptors": int(len(columns)),
        "stack_features": int(len(columns) + 1),
        "split_seed": args.split_seed,
        "train_fraction": round(1.0 - args.hold_fraction, 4),
        "val_fraction": round(args.hold_fraction / 2.0, 4),
        "test_fraction": round(args.hold_fraction / 2.0, 4),
        "svr": {
            "kernel": args.svr_kernel,
            "C": args.svr_C,
            "epsilon": args.svr_epsilon,
            "gamma": args.svr_gamma,
            "scaled": False,
            "cache_size_mb": args.svr_cache_mb,
            "n_support": int(len(estimator.support_)),
        },
        "note": (
            "Unscaled RBF SVR on CSV descriptors plus UniMolPred. "
            "SVR is fit on the unique training split. "
            "3seg means the source Uni-Mol model was trained with size copies."
        ),
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(f"Finished. Output: {out_dir}")


if __name__ == "__main__":
    main()
