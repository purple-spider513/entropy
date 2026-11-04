#!/usr/bin/env python3
"""Train a Uni-Mol fine-tune + SVR stack for gas-phase standard entropy (S°).

Self-contained: needs only numpy, pandas, scikit-learn, joblib, RDKit and unimol_tools.

Model
  1. Uni-Mol v2 84m is fine-tuned on S° from SMILES.
     --oversample none   each training molecule once
     --oversample 3seg   copies by heavy-atom count: n<12 x5, 12<=n<=15 x3, n>15 x1
  2. An unscaled RBF SVR (C=10000, epsilon=0.1, gamma="scale") is fit on the
     descriptor columns plus the Uni-Mol prediction (UniMolPred), on each
     training molecule once.
  Both steps use the 80% training split only (80/10/10, seed 42).

Descriptors
  Default: every descriptor column of the training CSV (all columns except ID,
  SMILES, entropy and derived columns). With --feature-file, only the columns
  listed in that file (one name per line; blank lines and '#' comments ignored).

After training, the 10% validation and test splits are scored, and each CSV
passed with --external is scored into <output>/official_eval/. External rows
whose canonical SMILES are in the training split are marked TrainOverlap and
left out of the scores.

Examples
  python Uni-Mol_SVR_train.py --train hybrid_data/hybrid_train_102926.csv --oversample 3seg \\
      --feature-file selected_features -o os3seg_selfeat_svr \\
      --external hybrid_data/hybrid_std128_102926.csv hybrid_data/hybrid_std109_102926.csv
  python Uni-Mol_SVR_train.py --train hybrid_data/hybrid_train_102926.csv --oversample none -o no_os_allfeat_svr
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

# ---------------------------------------------------------------- settings
GPU_UUID = "GPU-11559549-d592-eb13-a511-44849442d2a5"  # RTX 5070 Ti

SPLIT_SEED = 42
HOLD_FRACTION = 0.2  # held out, then split in half into validation and test

UNIMOL_MODEL_NAME = "unimolv2"
UNIMOL_MODEL_SIZE = "84m"
UNIMOL_EPOCHS = 25
UNIMOL_BATCH_SIZE = 8
UNIMOL_LR = 5e-5
UNIMOL_EARLY_STOPPING = 20
UNIMOL_MAX_NORM = 5.0
UNIMOL_SEED = 42

# 3-segment oversampling: (lowest n_heavy inclusive, highest exclusive, total copies)
OVERSAMPLE_SEGMENTS = ((None, 12, 5), (12, 16, 3), (16, None, 1))
OVERSAMPLE_SEED = 42
OVERSAMPLE_RULE = "n<12 x5, 12<=n<=15 x3, n>15 x1"

SVR_KERNEL = "rbf"
SVR_C = 10000
SVR_EPSILON = 0.1
SVR_GAMMA = "scale"
SVR_CACHE_MB = 4096

MODEL_FILE = "svr_unimol_stack.joblib"

# Columns that are never descriptors.
NON_FEATURE_COLS = {
    "ID", "entropy", "Entropy", "SMILES", "Ligands", "n_heavy", "canon",
    "UniMolPred", "SVRPred", "UniMol_Error", "SVR_Error", "TrainOverlap",
    "LinearPred", "Residual", "HybridPred", "Predicted_Entropy", "FeatPred",
    "Absolute_Error", "Signed_Error", "Feat_Error",
}
READABLE_FLOAT_COLS = {"Entropy", "entropy"}


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
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    for col in out.columns:
        if col in {"n_heavy", "heavy"}:
            out[col] = pd.to_numeric(out[col], errors="coerce").astype("Int64")
        elif col in READABLE_FLOAT_COLS:
            out[col] = [np.nan if pd.isna(v) else round(float(v), decimals) for v in out[col]]
    out.to_csv(path, index=False, float_format=f"%.{decimals}f")


def write_preds(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, float_format="%.4f")


def dump_json(path: Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


# ---------------------------------------------------------------- descriptors
def load_feature_file(path: Path) -> list[str]:
    names = [ln.strip() for ln in Path(path).read_text().splitlines()]
    names = [n for n in names if n and not n.startswith("#")]
    dup = sorted({n for n in names if names.count(n) > 1})
    if dup:
        raise SystemExit(f"{path} lists duplicate features, e.g. {dup[:8]}")
    if not names:
        raise SystemExit(f"{path} lists no features")
    return names


def all_descriptor_columns(frame: pd.DataFrame) -> list[str]:
    """Every column except identifiers, SMILES, entropy and derived columns, in CSV order."""
    names, skipped = [], []
    for col in frame.columns:
        if col in NON_FEATURE_COLS:
            continue
        if pd.to_numeric(frame[col], errors="coerce").notna().any():
            names.append(col)
        else:
            skipped.append(col)
    if skipped:
        print(f"Skipping non-numeric columns: {skipped}", flush=True)
    if not names:
        raise SystemExit("No descriptor columns found in the training CSV.")
    return names


def check_features(frame: pd.DataFrame, features: list[str], label: str) -> None:
    missing = [n for n in features if n not in frame.columns]
    if missing:
        raise SystemExit(f"{label} is missing {len(missing)} descriptor columns, e.g. {missing[:8]}")


def stack_matrix(frame: pd.DataFrame, features: list[str], unimol_pred) -> np.ndarray:
    tabular = frame[features].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(float)
    return np.hstack([tabular, np.asarray(unimol_pred, dtype=float).reshape(-1, 1)])


# ---------------------------------------------------------------- oversampling
def oversample_3seg(df: pd.DataFrame) -> pd.DataFrame:
    heavy = pd.to_numeric(df["n_heavy"], errors="coerce")
    parts = []
    for lo, hi, copies in OVERSAMPLE_SEGMENTS:
        mask = pd.Series(True, index=df.index)
        if lo is not None:
            mask &= heavy >= lo
        if hi is not None:
            mask &= heavy < hi
        block = df.loc[mask]
        if len(block):
            parts.append(pd.concat([block] * copies, ignore_index=True))
    return (
        pd.concat(parts, ignore_index=True)
        .sample(frac=1.0, random_state=OVERSAMPLE_SEED)
        .reset_index(drop=True)
    )


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


def metrics(y, p) -> dict:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    return {
        "n": int(len(y)),
        "mae": round(float(mean_absolute_error(y, p)), 4),
        "rmse": round(float(np.sqrt(mean_squared_error(y, p))), 4),
        "r2": round(float(r2_score(y, p)), 4),
        "bias": round(float(np.mean(p - y)), 4),
    }


def report(name: str, y, p) -> dict:
    m = metrics(y, p)
    print(f"{name:22} n={m['n']:5d}  MAE={m['mae']:.4f}  RMSE={m['rmse']:.4f}  "
          f"R2={m['r2']:.4f}  bias={m['bias']:.4f}", flush=True)
    return m


# ---------------------------------------------------------------- Uni-Mol
def train_unimol(rows: pd.DataFrame, save_path: Path, use_cuda: bool, epochs: int, batch_size: int, lr: float):
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
    trainer.fit(rows[["SMILES", "Entropy"]])


def tidy_unimol_dir(unimol_dir: Path) -> None:
    """Write metric.result as readable JSON (also metric.json) and a plain learning rate in config.yaml."""
    metric_path = unimol_dir / "metric.result"
    if metric_path.exists():
        try:
            payload = joblib.load(metric_path)
        except Exception:
            payload = json.loads(metric_path.read_text())
        if isinstance(payload, dict):
            clean = {k: round(float(v.item() if hasattr(v, "item") else v), 4) for k, v in payload.items()}
            dump_json(unimol_dir / "metric.json", clean)
            metric_path.write_text(json.dumps(clean, indent=2) + "\n")
    cfg_path = unimol_dir / "config.yaml"
    if cfg_path.exists():
        cfg_path.write_text(cfg_path.read_text().replace("learning_rate: 5.0e-05", "learning_rate: 0.00005"))


def unimol_trained(unimol_dir: Path) -> bool:
    return (unimol_dir / "config.yaml").is_file() and any(unimol_dir.glob("*.pth"))


def unimol_predict(unimol_dir: Path, smiles: pd.Series) -> np.ndarray:
    from unimol_tools import MolPredict

    predictor = MolPredict(load_model=str(unimol_dir))
    if hasattr(predictor, "config") and "save_path" in predictor.config:
        predictor.config.pop("save_path")
    pred = np.asarray(predictor.predict(pd.DataFrame({"SMILES": smiles.astype(str)})), dtype=float).reshape(-1)
    if len(pred) != len(smiles):
        raise RuntimeError(f"Uni-Mol returned {len(pred)} predictions for {len(smiles)} SMILES")
    return pred


def score_frame(frame: pd.DataFrame, features: list[str], unimol_pred, estimator) -> pd.DataFrame:
    out = frame[[c for c in ("ID", "SMILES", "Entropy", "n_heavy") if c in frame.columns]].copy()
    out["UniMolPred"] = unimol_pred
    out["SVRPred"] = estimator.predict(stack_matrix(frame, features, unimol_pred))
    out["UniMol_Error"] = np.abs(out["UniMolPred"] - out["Entropy"])
    out["SVR_Error"] = np.abs(out["SVRPred"] - out["Entropy"])
    return out


# ---------------------------------------------------------------- main
def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train", required=True, help="Training CSV with ID, SMILES, entropy and descriptor columns.")
    p.add_argument("-o", "--output", required=True, help="New model directory.")
    p.add_argument("--oversample", choices=["none", "3seg"], default="none",
                   help="Copies of Uni-Mol training rows by heavy-atom count (default: none).")
    p.add_argument("--feature-file", default="",
                   help="Optional text file, one descriptor column per line. Default: every descriptor column.")
    p.add_argument("--external", nargs="*", default=[], help="Labeled CSVs to score after training.")
    p.add_argument("--epochs", type=int, default=UNIMOL_EPOCHS)
    p.add_argument("--batch-size", type=int, default=UNIMOL_BATCH_SIZE)
    p.add_argument("--lr", type=float, default=UNIMOL_LR)
    p.add_argument("--split-seed", type=int, default=SPLIT_SEED)
    p.add_argument("--gpu", default=GPU_UUID, help="CUDA device UUID or index (CUDA_VISIBLE_DEVICES).")
    p.add_argument("--resume", action="store_true",
                   help="Allow an existing output directory: reuse its trained Uni-Mol, refit the SVR, rewrite the rest.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    train_path = Path(args.train)
    if not train_path.is_file():
        raise SystemExit(f"Missing {train_path}")
    out_dir = Path(args.output)
    unimol_dir = out_dir / "unimol"
    existing = [p.name for p in (out_dir / "config.json", out_dir / MODEL_FILE, unimol_dir) if p.exists()]
    if existing and not args.resume:
        raise SystemExit(f"{out_dir} already holds a model ({', '.join(existing)}). "
                         "Choose a new --output, or pass --resume to reuse its Uni-Mol and refit the SVR.")
    if args.resume and (out_dir / "config.json").is_file():
        old = json.loads((out_dir / "config.json").read_text())
        if old.get("oversample") != (args.oversample == "3seg") or old.get("split_seed") != args.split_seed:
            raise SystemExit(f"--resume: {out_dir} was trained with a different --oversample or --split-seed.")
    out_dir.mkdir(parents=True, exist_ok=True)

    table = normalize_table(pd.read_csv(train_path))
    for col in ("SMILES", "Entropy"):
        if col not in table.columns:
            raise SystemExit(f"{train_path} has no {col} column")
    if args.oversample == "3seg" and "n_heavy" not in table.columns:
        raise SystemExit("--oversample 3seg needs a 'heavy' or 'n_heavy' column.")
    if args.feature_file:
        features = load_feature_file(Path(args.feature_file))
        feature_source = str(Path(args.feature_file).resolve())
    else:
        features = all_descriptor_columns(table)
        feature_source = "all descriptor columns"
    check_features(table, features, train_path.name)
    (out_dir / "features.txt").write_text("\n".join(features) + "\n")

    idx = np.arange(len(table))
    tr_idx, tmp_idx = train_test_split(idx, test_size=HOLD_FRACTION, random_state=args.split_seed, shuffle=True)
    va_idx, te_idx = train_test_split(tmp_idx, test_size=0.5, random_state=args.split_seed, shuffle=True)
    df_tr = table.iloc[tr_idx].reset_index(drop=True)
    df_va = table.iloc[va_idx].reset_index(drop=True)
    df_te = table.iloc[te_idx].reset_index(drop=True)
    print(f"80/10/10 split seed={args.split_seed}: n_train={len(df_tr)} n_val={len(df_va)} "
          f"n_test={len(df_te)}  descriptors={len(features)} ({feature_source})", flush=True)

    split_train_csv = out_dir / "split_train.csv"
    write_readable_csv(df_tr[["SMILES", "Entropy"]], split_train_csv)
    write_readable_csv(df_va, out_dir / "split_val.csv")
    write_readable_csv(df_te, out_dir / "split_test.csv")

    if args.oversample == "3seg":
        sampled = oversample_3seg(df_tr[["SMILES", "Entropy", "n_heavy"]].copy())
        print(f"Uni-Mol fine-tune rows after 3-segment oversampling ({OVERSAMPLE_RULE}): {len(sampled)}", flush=True)
    else:
        sampled = df_tr[["SMILES", "Entropy"]].copy()
        print(f"Uni-Mol fine-tune rows (no oversampling): {len(sampled)}", flush=True)
    write_readable_csv(sampled[["SMILES", "Entropy"]], out_dir / "unimol_ft_train.csv")

    # Step 1: Uni-Mol fine-tune on S°
    if unimol_trained(unimol_dir):
        print(f"Reusing the trained Uni-Mol in {unimol_dir}", flush=True)
    else:
        import torch

        use_cuda = torch.cuda.is_available()
        print(f"CUDA: {use_cuda}  batch_size={args.batch_size}", flush=True)
        train_unimol(sampled[["SMILES", "Entropy"]], unimol_dir, use_cuda, args.epochs, args.batch_size, args.lr)
        tidy_unimol_dir(unimol_dir)

    print("Uni-Mol predictions on the train/val/test splits...", flush=True)
    um_tr = unimol_predict(unimol_dir, df_tr["SMILES"])
    um_va = unimol_predict(unimol_dir, df_va["SMILES"])
    um_te = unimol_predict(unimol_dir, df_te["SMILES"])

    # Step 2: SVR on descriptors + UniMolPred
    estimator = SVR(kernel=SVR_KERNEL, C=SVR_C, epsilon=SVR_EPSILON, gamma=SVR_GAMMA, cache_size=SVR_CACHE_MB)
    print(f"Fitting SVR on n={len(df_tr)}  inputs={len(features) + 1}  kernel={SVR_KERNEL}  "
          f"C={SVR_C}  epsilon={SVR_EPSILON}  gamma={SVR_GAMMA}", flush=True)
    estimator.fit(stack_matrix(df_tr, features, um_tr), df_tr["Entropy"].to_numpy(float))
    print(f"SVR support vectors: {len(estimator.support_)}", flush=True)
    model_path = out_dir / MODEL_FILE
    joblib.dump({
        "estimator": estimator,
        "feat_names": features + ["UniMolPred"],
        "kind": "svr_unimol_feat_stack",
        "kernel": SVR_KERNEL, "C": SVR_C, "epsilon": SVR_EPSILON, "gamma": SVR_GAMMA,
        "scaled": False,
        "feature_file": feature_source,
        "unimol": str(unimol_dir),
    }, model_path)
    print(f"Saved {model_path}", flush=True)

    # Internal splits
    tr_out = score_frame(df_tr, features, um_tr, estimator)
    va_out = score_frame(df_va, features, um_va, estimator)
    te_out = score_frame(df_te, features, um_te, estimator)
    print("Internal splits:")
    split_metrics = {
        "train_unimol": report("train Uni-Mol", tr_out["Entropy"], tr_out["UniMolPred"]),
        "train_unimol_svr": report("train Uni-Mol+SVR", tr_out["Entropy"], tr_out["SVRPred"]),
        "val_unimol": report("val Uni-Mol", va_out["Entropy"], va_out["UniMolPred"]),
        "val_unimol_svr": report("val Uni-Mol+SVR", va_out["Entropy"], va_out["SVRPred"]),
        "test_unimol": report("test Uni-Mol", te_out["Entropy"], te_out["UniMolPred"]),
        "test_unimol_svr": report("test Uni-Mol+SVR", te_out["Entropy"], te_out["SVRPred"]),
    }
    write_preds(tr_out, out_dir / "train_predictions.csv")
    write_preds(va_out, out_dir / "val_10pct_preds.csv")
    write_preds(te_out, out_dir / "test_10pct_preds.csv")
    dump_json(out_dir / "split_metrics.json", split_metrics)

    # External sets
    if args.external:
        banned = canonical_smiles_set(df_tr["SMILES"])
        eval_dir = out_dir / "official_eval"
        summary, parts = [], []
        print("External sets (rows whose SMILES are in the 80% training split are not scored):")
        for ext in args.external:
            ext_path = Path(ext)
            raw = normalize_table(pd.read_csv(ext_path))
            check_features(raw, features, ext_path.name)
            um = unimol_predict(unimol_dir, raw["SMILES"])
            scored = score_frame(raw, features, um, estimator)
            overlap = overlap_row_mask(scored, banned)
            scored["TrainOverlap"] = overlap.to_numpy()
            kept = scored.loc[~overlap]
            summary.append({
                "set": ext_path.stem,
                "excluded_overlap": int(overlap.sum()),
                "unimol": report(f"{ext_path.stem} Uni-Mol", kept["Entropy"], kept["UniMolPred"]),
                "unimol_svr": report(f"{ext_path.stem} Uni-Mol+SVR", kept["Entropy"], kept["SVRPred"]),
            })
            parts.append(kept)
            write_preds(scored, eval_dir / f"{ext_path.stem}_preds.csv")
        combined = pd.concat(parts, ignore_index=True)
        dump_json(eval_dir / "metrics.json", {
            "sets": summary,
            "combined": {
                "unimol": report("combined Uni-Mol", combined["Entropy"], combined["UniMolPred"]),
                "unimol_svr": report("combined Uni-Mol+SVR", combined["Entropy"], combined["SVRPred"]),
            },
        })

    three = args.oversample == "3seg"
    dump_json(out_dir / "config.json", {
        "kind": f"unimol_ft_entropy_{'os3seg' if three else 'no_oversample'}_80_10_10_svr_stack",
        "train_source": str(train_path.resolve()),
        "train": str(split_train_csv.resolve()),
        "output": str(out_dir.resolve()),
        "unimol": str(unimol_dir.resolve()),
        "model": str(model_path.resolve()),
        "feature_file": feature_source,
        "n_descriptors": len(features),
        "stack_features": len(features) + 1,
        "n_pool": int(len(table)),
        "n_train": int(len(df_tr)), "n_val": int(len(df_va)), "n_test": int(len(df_te)),
        "n_unimol_rows": int(len(sampled)),
        "oversample": three,
        "oversample_scheme": OVERSAMPLE_RULE if three else None,
        "split_seed": args.split_seed,
        "epochs": args.epochs, "batch_size": args.batch_size, "learning_rate": args.lr,
        "model_name": UNIMOL_MODEL_NAME, "model_size": UNIMOL_MODEL_SIZE,
        "svr": {"kernel": SVR_KERNEL, "C": SVR_C, "epsilon": SVR_EPSILON, "gamma": SVR_GAMMA,
                "scaled": False, "n_support": int(len(estimator.support_))},
        "external": [str(Path(x).resolve()) for x in args.external],
        "note": "Uni-Mol fine-tuned on S°, then unscaled RBF SVR on the descriptors plus UniMolPred. "
                "Both fit on the 80% split only.",
    })
    print(f"Finished. Output: {out_dir}")


if __name__ == "__main__":
    main()
