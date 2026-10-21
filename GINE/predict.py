from __future__ import annotations

import argparse
from pathlib import Path

from entropy_gnn.inference import (
    load_checkpoint,
    mode_requires_features,
    mode_requires_smiles,
    predict_table,
    _align_prediction_table,
)
from entropy_gnn.reporting import write_predictions_csv
from entropy_gnn.training import resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict entropy from a combined hybrid CSV (SMILES + ligand features)."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--csv",
        type=Path,
        required=True,
        help="Combined CSV with ID, SMILES, and ligand descriptors.",
    )
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument(
        "--gpu-index",
        type=int,
        default=-1,
        help="CUDA device index; -1 selects the GPU with the most free memory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device, args.gpu_index)
    loaded = load_checkpoint(args.checkpoint, device)
    frame = _align_prediction_table(
        args.csv,
        loaded["predictor_columns"],
        require_features=mode_requires_features(loaded["mode"]),
        require_smiles=mode_requires_smiles(loaded["mode"]),
        require_entropy=False,
    )
    output = predict_table(
        loaded["model"],
        loaded["preprocessor"],
        frame,
        loaded["mode"],
        device,
        args.batch_size,
        loaded["embed_3d"],
        loaded["seed"],
    )
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    write_predictions_csv(args.output_csv, output)
    successful = int(output["predicted_entropy"].notna().sum())
    print(f"wrote {args.output_csv} ({successful} successful, {len(output) - successful} failed)")


if __name__ == "__main__":
    main()
