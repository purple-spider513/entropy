#!/usr/bin/env python3
"""Train an SVR + Uni-Mol residual hybrid on bond features plus general descriptors.

The descriptor set is every numeric CSV column except identifiers, entropy,
and SMILES. The heavy column stays in the SVR input, as it does in the
bond-feature run. A copy named n_heavy is used only for 3-segment residual
oversampling:
  n_heavy < 12 -> 5 copies, 12 <= n_heavy <= 15 -> 3 copies, n_heavy > 15 -> 1 copy.
SVR itself is fit on the unique training rows.

SVR is the unscaled RBF bond-feature model from classical_SVR/SVR_predS_bond.py
(C=10000, epsilon=0.1).
Uni-Mol v2 84m learns the residual from SMILES. Helper routines are copied
here so this script does not import train.py.

Example:
  python svr_general_residual_unimol_train.py \
    --train hybrid_data/hybrid_train_091926.csv \
    --output svr_general_residual_run
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
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR

ROOT = Path(__file__).resolve().parent
GPU_UUID = "GPU-11559549-d592-eb13-a511-44849442d2a5"

SKIP_COLS = {
    "ID",
    "entropy",
    "Entropy",
    "SMILES",
    "Ligands",
    "n_heavy",
    "canon",
    "LinearPred",
    "Residual",
    "HybridPred",
    "Predicted_Entropy",
    "Absolute_Error",
    "TrainOverlap",
    "SVRPred",
    "SVR_Error",
    "Signed_Error",
}
READABLE_FLOAT_COLS = {
    "Entropy",
    "entropy",
    "HuberPred",
    "Residual",
    "Predicted_Entropy",
    "Absolute_Error",
    "Predicted_Residual",
    "True_Residual",
    "SVRPred",
    "SVR_Error",
    "Signed_Error",
}

OVERSAMPLE_SEED = 42
SMALL_HEAVY = 12
MEDIUM_HEAVY = 16
SMALL_COPIES = 5
MEDIUM_COPIES = 3

UNIMOL_EPOCHS = 25
UNIMOL_BATCH_SIZE = 8
UNIMOL_LR = 5e-5
UNIMOL_MODEL_NAME = "unimolv2"
UNIMOL_MODEL_SIZE = "84m"
UNIMOL_EARLY_STOPPING = 20
UNIMOL_WARMUP_RATIO = 0.03
UNIMOL_MAX_NORM = 5.0
UNIMOL_SEED = 42

SVR_KERNEL = "rbf"
SVR_C = 10000
SVR_EPSILON = 0.1
SVR_CACHE_MB = 2048
SPLIT_SEED = 42
HOLD_FRACTION = 0.2


def feature_columns(df: pd.DataFrame):
    return [c for c in df.columns if c not in SKIP_COLS]


def load_feature_file(path: Path) -> list[str]:
    names = []
    for line in path.read_text().splitlines():
        name = line.split("#", 1)[0].strip().strip(",").strip('"').strip("'")
        if name:
            names.append(name)
    if not names:
        raise SystemExit(f"No feature names in {path}")
    return names


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


def oversample_3seg(
    df: pd.DataFrame,
    small_heavy: int = SMALL_HEAVY,
    medium_heavy: int = MEDIUM_HEAVY,
    small_copies: int = SMALL_COPIES,
    medium_copies: int = MEDIUM_COPIES,
    seed: int = OVERSAMPLE_SEED,
) -> pd.DataFrame:
    """3-segment residual oversampling.

    Default bins match the earlier residual runs and train.py:
    n_heavy < 12 is repeated 5 times, 12 <= n_heavy <= 15 is repeated 3 times,
    and n_heavy > 15 is kept once. The upper edge is exclusive, so medium_heavy=16
    covers 12 through 15.
    """
    n_heavy = pd.to_numeric(df["n_heavy"], errors="coerce")
    small = df.loc[n_heavy < small_heavy]
    medium = df.loc[(n_heavy >= small_heavy) & (n_heavy < medium_heavy)]
    large = df.loc[n_heavy.isna() | (n_heavy >= medium_heavy)]
    parts = []
    if len(small):
        parts.append(pd.concat([small] * small_copies, ignore_index=True))
    if len(medium):
        parts.append(pd.concat([medium] * medium_copies, ignore_index=True))
    if len(large):
        parts.append(large.reset_index(drop=True))
    if not parts:
        raise SystemExit("3-segment oversampling received no rows.")
    print(
        f"3-segment oversampling: n<{small_heavy} n={len(small)} x{small_copies}, "
        f"{small_heavy}<=n<{medium_heavy} n={len(medium)} x{medium_copies}, "
        f"n>={medium_heavy} n={len(large)} x1"
    )
    return (
        pd.concat(parts, ignore_index=True)
        .sample(frac=1.0, random_state=seed)
        .reset_index(drop=True)
    )


def train_unimol(residual_df: pd.DataFrame, save_path: Path, use_cuda: bool, epochs, batch_size, lr):
    from unimol_tools import MolTrain

    trainer = MolTrain(
        task="regression",
        data_type="molecule",
        smiles_col="SMILES",
        target_cols="Entropy",
        model_name=UNIMOL_MODEL_NAME,
        model_size=UNIMOL_MODEL_SIZE,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=lr,
        kfold=1,
        metrics="mae",
        target_normalize="standard",
        remove_hs=False,
        smiles_check="filter",
        use_cuda=use_cuda,
        use_amp=use_cuda,
        early_stopping=UNIMOL_EARLY_STOPPING,
        max_norm=UNIMOL_MAX_NORM,
        save_path=str(save_path),
        conf_cache_level=1,
        seed=UNIMOL_SEED,
    )
    trainer.fit(residual_df[["SMILES", "Entropy"]])


def write_unimol_readable_outputs(unimol_dir: Path, residual_train_csv=None):
    """Write Uni-Mol dumps as JSON/CSV with Python floats, not np.float64."""

    unimol_dir = Path(unimol_dir)
    metric_path = unimol_dir / "metric.result"
    if metric_path.exists():
        try:
            payload = joblib.load(metric_path)
        except Exception:
            payload = json.loads(metric_path.read_text())
        if isinstance(payload, dict):
            clean = {}
            for key, val in payload.items():
                if hasattr(val, "item"):
                    val = val.item()
                clean[key] = round(float(val), 4)
            text = json.dumps(clean, indent=2) + "\n"
            (unimol_dir / "metric.json").write_text(text)
            metric_path.write_text(text)
            print(f"Saved {unimol_dir / 'metric.json'}")

    scaler_path = unimol_dir / "target_scaler.ss"
    scaler = None
    if scaler_path.exists():
        scaler = joblib.load(scaler_path)
        n_seen = scaler.n_samples_seen_
        if hasattr(n_seen, "item"):
            n_seen = n_seen.item()
        scaler_json = {
            "mean": round(float(np.asarray(scaler.mean_).reshape(-1)[0]), 4),
            "scale": round(float(np.asarray(scaler.scale_).reshape(-1)[0]), 4),
            "var": round(float(np.asarray(scaler.var_).reshape(-1)[0]), 4),
            "n_samples": int(n_seen),
        }
        (unimol_dir / "target_scaler.json").write_text(
            json.dumps(scaler_json, indent=2) + "\n"
        )
        print(f"Saved {unimol_dir / 'target_scaler.json'}")

    cv_path = unimol_dir / "cv.data"
    if cv_path.exists():
        pred = np.asarray(joblib.load(cv_path), dtype=float).reshape(-1, 1)
        if scaler is not None:
            pred = scaler.inverse_transform(pred)
        pred = pred.reshape(-1)
        out = pd.DataFrame({"Predicted_Residual": pred})
        if residual_train_csv and Path(residual_train_csv).exists():
            resid = pd.read_csv(residual_train_csv)
            if len(resid) == len(out):
                out.insert(0, "SMILES", resid["SMILES"].astype(str).values)
                if "Entropy" in resid.columns:
                    true = pd.to_numeric(resid["Entropy"], errors="coerce").to_numpy(float)
                    out["True_Residual"] = true
                    out["Absolute_Error"] = np.abs(pred - true)
        write_readable_csv(out, unimol_dir / "cv.csv")
        print(f"Saved {unimol_dir / 'cv.csv'}")

    cfg_path = unimol_dir / "config.yaml"
    if cfg_path.exists():
        text = cfg_path.read_text()
        text = text.replace("learning_rate: 5.0e-05", "learning_rate: 0.00005")
        cfg_path.write_text(text)


def write_unimol_metrics(unimol_dir: Path, residual_train_csv=None):
    write_unimol_readable_outputs(unimol_dir, residual_train_csv)


def feature_matrix(df: pd.DataFrame, feat_names: list[str]) -> np.ndarray:
    missing = [c for c in feat_names if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing {len(missing)} features, e.g. {missing[:8]}")
    return df[feat_names].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(float)


def regression_metrics(y, p) -> dict:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    return {
        "n": int(len(y)),
        "mae": round(float(mean_absolute_error(y, p)), 4),
        "rmse": round(float(np.sqrt(mean_squared_error(y, p))), 4),
        "r2": round(float(r2_score(y, p)), 4),
        "bias": round(float(np.mean(p - y)), 4),
    }


def report_metrics(name: str, y, p) -> dict:
    stats = regression_metrics(y, p)
    print(
        f"{name:22} n={stats['n']:5d}  MAE={stats['mae']:.4f}  "
        f"RMSE={stats['rmse']:.4f}  R2={stats['r2']:.4f}  bias={stats['bias']:.4f}"
    )
    return stats


def score_split(df_split, feat_names, svr, unimol_dir=None, save_path=None):
    y = df_split["Entropy"].to_numpy(float)
    svr_pred = svr.predict(feature_matrix(df_split, feat_names))
    out = df_split.copy()
    out["SVRPred"] = svr_pred
    out["SVR_Error"] = np.abs(svr_pred - y)
    if unimol_dir is not None:
        import unimol_tools

        predictor = unimol_tools.MolPredict(load_model=str(unimol_dir))
        if hasattr(predictor, "config") and "save_path" in predictor.config:
            predictor.config.pop("save_path")
        kwargs = {}
        if save_path is not None:
            kwargs["save_path"] = str(save_path)
        resid = np.asarray(
            predictor.predict(pd.DataFrame({"SMILES": out["SMILES"].astype(str)}), **kwargs),
            dtype=float,
        ).reshape(-1)
        out["Residual"] = resid
        out["Predicted_Entropy"] = svr_pred + resid
        out["Absolute_Error"] = np.abs(out["Predicted_Entropy"] - y)
        out["Signed_Error"] = out["Predicted_Entropy"] - y
    return out


def write_score_csv(df_out, path, hybrid=False):
    cols = ["ID", "SMILES", "Entropy", "n_heavy", "SVRPred", "SVR_Error"]
    if hybrid:
        cols.extend(["Residual", "Predicted_Entropy", "Absolute_Error", "Signed_Error"])
    write_readable_csv(df_out[[c for c in cols if c in df_out.columns]], path)


def split_pool(df: pd.DataFrame, hold_fraction: float, seed: int):
    if hold_fraction <= 0:
        empty = df.iloc[0:0].copy()
        return df.reset_index(drop=True), empty, empty
    if not 0 < hold_fraction < 1:
        raise SystemExit("--hold-fraction must be 0 or between 0 and 1.")
    idx = np.arange(len(df))
    train_idx, temp_idx = train_test_split(
        idx, test_size=hold_fraction, random_state=seed, shuffle=True
    )
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.5, random_state=seed, shuffle=True
    )
    return (
        df.iloc[train_idx].reset_index(drop=True),
        df.iloc[val_idx].reset_index(drop=True),
        df.iloc[test_idx].reset_index(drop=True),
    )


def refuse_existing_output(save_dir: Path):
    if (save_dir / "config.json").exists() or (save_dir / "tabular.joblib").exists():
        raise SystemExit(
            f"{save_dir} already contains a model. Choose a new --output directory."
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train SVR + Uni-Mol residual hybrid on bond and general descriptors."
    )
    parser.add_argument("--train", required=True, help="Training CSV with SMILES, entropy, and descriptors.")
    parser.add_argument("-o", "--output", required=True, help="New directory for this model. Must not already hold a model.")
    parser.add_argument(
        "--feature-file",
        default="",
        help="Optional text file of descriptor names, one per line. Default uses every CSV descriptor, including heavy.",
    )
    parser.add_argument(
        "--include-heavy",
        action="store_true",
        help="Add heavy to a custom --feature-file that omitted it. The default descriptor list already keeps heavy.",
    )
    parser.add_argument("--svr-kernel", default=SVR_KERNEL)
    parser.add_argument("--svr-C", type=float, default=SVR_C)
    parser.add_argument("--svr-epsilon", type=float, default=SVR_EPSILON)
    parser.add_argument("--svr-cache-mb", type=float, default=SVR_CACHE_MB)
    parser.add_argument("--epochs", type=int, default=UNIMOL_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=UNIMOL_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=UNIMOL_LR)
    parser.add_argument(
        "--hold-fraction",
        type=float,
        default=HOLD_FRACTION,
        help="Fraction held out, then split in half into val and test. 0.2 gives 80/10/10. 0 trains on every row.",
    )
    parser.add_argument("--split-seed", type=int, default=SPLIT_SEED)
    parser.add_argument(
        "--small-heavy",
        type=int,
        default=SMALL_HEAVY,
        help="3-segment edge. n_heavy below this value gets --small-copies (default 12).",
    )
    parser.add_argument(
        "--medium-heavy",
        type=int,
        default=MEDIUM_HEAVY,
        help="3-segment edge. n_heavy below this value and at or above --small-heavy gets --medium-copies (default 16, so 12-15).",
    )
    parser.add_argument("--small-copies", type=int, default=SMALL_COPIES, help="Copies for the small segment (default 5).")
    parser.add_argument("--medium-copies", type=int, default=MEDIUM_COPIES, help="Copies for the middle segment (default 3).")
    parser.add_argument(
        "--oversample-seed",
        type=int,
        default=OVERSAMPLE_SEED,
        help="Shuffle seed for the 3-segment residual rows (default 42).",
    )
    parser.add_argument("--gpu", default=GPU_UUID, help="CUDA_VISIBLE_DEVICES value. Use 'cpu' to skip the GPU.")
    parser.add_argument("--tabular-only", action="store_true", help="Fit SVR only and skip the residual Uni-Mol.")
    return parser


def main():
    args = build_parser().parse_args()
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    if args.gpu.lower() != "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    train_path = Path(args.train)
    if not train_path.is_file():
        raise SystemExit(f"Missing training CSV: {train_path}")
    save_dir = Path(args.output)
    refuse_existing_output(save_dir)

    df = normalize_table(pd.read_csv(train_path))
    if "SMILES" not in df.columns or "Entropy" not in df.columns:
        raise SystemExit("Training CSV must contain SMILES and entropy.")
    if "n_heavy" not in df.columns:
        df["n_heavy"] = MEDIUM_HEAVY

    if args.feature_file:
        feat_names = load_feature_file(Path(args.feature_file))
    else:
        feat_names = feature_columns(df)
    if args.include_heavy and "heavy" in df.columns and "heavy" not in feat_names:
        feat_names.append("heavy")
    missing = [c for c in feat_names if c not in df.columns]
    if missing:
        raise SystemExit(f"Training CSV missing {len(missing)} features, e.g. {missing[:8]}")

    df_tr, df_va, df_te = split_pool(df, args.hold_fraction, args.split_seed)
    print(
        f"n_features={len(feat_names)}  n_train={len(df_tr)}  "
        f"n_val={len(df_va)}  n_test={len(df_te)}  seed={args.split_seed}"
    )

    svr = SVR(
        kernel=args.svr_kernel,
        C=args.svr_C,
        epsilon=args.svr_epsilon,
        cache_size=args.svr_cache_mb,
    )
    y_tr = df_tr["Entropy"].to_numpy(float)
    print(
        f"Fitting unscaled SVR kernel={args.svr_kernel} C={args.svr_C} "
        f"epsilon={args.svr_epsilon}  n={len(df_tr)}  p={len(feat_names)}"
    )
    svr.fit(feature_matrix(df_tr, feat_names), y_tr)
    train_pred = svr.predict(feature_matrix(df_tr, feat_names))
    residual = y_tr - train_pred
    print(f"Train SVR MAE={np.mean(np.abs(residual)):.4f}  n_support={len(svr.support_)}")

    save_dir.mkdir(parents=True, exist_ok=True)
    split_cols = [c for c in ["ID", "SMILES", "Entropy", "n_heavy"] if c in df_tr.columns]
    split_train_csv = save_dir / "split_train.csv"
    write_readable_csv(df_tr[split_cols], split_train_csv)
    if len(df_va):
        write_readable_csv(df_va[split_cols], save_dir / "split_val.csv")
        write_readable_csv(df_te[split_cols], save_dir / "split_test.csv")

    val_svr_m = test_svr_m = None
    if len(df_va):
        val_svr = score_split(df_va, feat_names, svr)
        test_svr = score_split(df_te, feat_names, svr)
        print("Validation SVR:")
        val_svr_m = report_metrics("val SVR", val_svr["Entropy"], val_svr["SVRPred"])
        print("Test SVR:")
        test_svr_m = report_metrics("test SVR", test_svr["Entropy"], test_svr["SVRPred"])
        write_score_csv(val_svr, save_dir / "val_svr.csv")
        write_score_csv(test_svr, save_dir / "test_svr.csv")

    tabular_path = save_dir / "tabular.joblib"
    joblib.dump(
        {
            "feat_names": feat_names,
            "estimator": svr,
            "kind": "svr_rbf_unscaled",
            "kernel": args.svr_kernel,
            "C": args.svr_C,
            "epsilon": args.svr_epsilon,
            "scaled": False,
            "feature_set": "bond_plus_general" if not args.feature_file else "custom",
            "include_heavy": "heavy" in feat_names,
        },
        tabular_path,
    )
    print(f"Saved {tabular_path}")

    resid_df = df_tr[["SMILES", "Entropy", "n_heavy"]].copy()
    resid_df["Entropy"] = residual
    write_readable_csv(resid_df, save_dir / "residual_base.csv")
    sampled = oversample_3seg(
        resid_df,
        small_heavy=args.small_heavy,
        medium_heavy=args.medium_heavy,
        small_copies=args.small_copies,
        medium_copies=args.medium_copies,
        seed=args.oversample_seed,
    )
    residual_train_csv = save_dir / "residual_train.csv"
    write_readable_csv(sampled[["SMILES", "Entropy"]], residual_train_csv)
    print(f"Residual train rows after size copies: {len(sampled)}")

    unimol_dir = save_dir / "unimol"
    val_hyb_m = test_hyb_m = None
    if args.tabular_only:
        print("Skipping Uni-Mol (--tabular-only).")
        unimol_record = ""
    else:
        import torch

        use_cuda = args.gpu.lower() != "cpu" and torch.cuda.is_available()
        print(f"CUDA: {use_cuda}  batch_size={args.batch_size}")
        if use_cuda:
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        train_unimol(sampled, unimol_dir, use_cuda, args.epochs, args.batch_size, args.lr)
        write_unimol_metrics(unimol_dir, residual_train_csv)
        unimol_record = str(unimol_dir)
        if len(df_va):
            val_hyb = score_split(df_va, feat_names, svr, unimol_dir, save_dir / "val_conformers")
            test_hyb = score_split(df_te, feat_names, svr, unimol_dir, save_dir / "test_conformers")
            print("Validation SVR+residual:")
            val_hyb_m = report_metrics("val SVR+res", val_hyb["Entropy"], val_hyb["Predicted_Entropy"])
            print("Test SVR+residual:")
            test_hyb_m = report_metrics("test SVR+res", test_hyb["Entropy"], test_hyb["Predicted_Entropy"])
            write_score_csv(val_hyb, save_dir / "val_preds.csv", hybrid=True)
            write_score_csv(test_hyb, save_dir / "test_preds.csv", hybrid=True)

    held = args.hold_fraction
    meta = {
        "kind": "svr_general_residual_unimol",
        "train_source": str(train_path),
        "train": str(split_train_csv),
        "output": str(save_dir),
        "tabular": str(tabular_path),
        "unimol": unimol_record,
        "feature_file": args.feature_file,
        "include_heavy": "heavy" in feat_names,
        "n_features": int(len(feat_names)),
        "svr_kernel": args.svr_kernel,
        "svr_C": args.svr_C,
        "svr_epsilon": args.svr_epsilon,
        "scaled": False,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "model_name": UNIMOL_MODEL_NAME,
        "model_size": UNIMOL_MODEL_SIZE,
        "target_normalize": "standard",
        "remove_hs": False,
        "warmup_ratio_constant": UNIMOL_WARMUP_RATIO,
        "hold_fraction": held,
        "split_seed": args.split_seed,
        "n_pool": int(len(df)),
        "n_train": int(len(df_tr)),
        "n_val": int(len(df_va)),
        "n_test": int(len(df_te)),
        "n_support": int(len(svr.support_)),
        "n_residual_train": int(len(sampled)),
        "n_unique_train_smiles": int(df_tr["SMILES"].astype(str).nunique()),
        "oversample": "3-segment",
        "oversample_rule": (
            f"n<{args.small_heavy} x{args.small_copies}, "
            f"{args.small_heavy}<=n<{args.medium_heavy} x{args.medium_copies}, "
            f"n>={args.medium_heavy} x1"
        ),
        "oversample_seed": args.oversample_seed,
        "val_svr": val_svr_m,
        "test_svr": test_svr_m,
        "val_hybrid": val_hyb_m,
        "test_hybrid": test_hyb_m,
        "note": (
            "Unscaled SVR on bond plus general descriptors, including heavy, then Uni-Mol residual. "
            "3-segment oversampling is applied to residual Uni-Mol rows only."
        ),
    }
    (save_dir / "config.json").write_text(json.dumps(meta, indent=2) + "\n")
    (save_dir / "split_metrics.json").write_text(
        json.dumps(
            {
                "val_svr": val_svr_m,
                "test_svr": test_svr_m,
                "val_hybrid": val_hyb_m,
                "test_hybrid": test_hyb_m,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"SVR residual hybrid finished. Model dir: {save_dir}")


if __name__ == "__main__":
    main()
