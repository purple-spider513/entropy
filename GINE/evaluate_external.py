"""Evaluate a saved entropy checkpoint on a labelled hybrid CSV."""
from __future__ import annotations

import argparse
from pathlib import Path

from entropy_gnn.inference import evaluate_csv
from entropy_gnn.reporting import format_eval_metrics, write_json, write_predictions_csv
from entropy_gnn.training import resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--csv",
        type=Path,
        required=True,
        help="Combined labelled CSV with ID, SMILES, entropy, and ligand descriptors.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument(
        "--gpu-index",
        type=int,
        default=-1,
        help="CUDA device index; -1 selects the GPU with the most free memory.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--entropy-scale",
        type=float,
        default=1.0,
        help="Multiply supplied entropy labels by this factor before scoring (e.g. 4.184 for cal to J).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.entropy_scale <= 0.0:
        raise ValueError("--entropy-scale must be a finite positive value")
    device = resolve_device(args.device, args.gpu_index)
    metrics, predictions = evaluate_csv(
        args.checkpoint,
        args.csv,
        device,
        batch_size=args.batch_size,
        entropy_scale=args.entropy_scale,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_predictions_csv(args.output_dir / "predictions.csv", predictions)
    predictions.loc[predictions["overlap_with_train"], ["ID", "smiles", "inchikey"]].to_csv(
        args.output_dir / "excluded_overlap.csv",
        index=False,
    )
    write_json(args.output_dir / "metrics.json", metrics)
    print(format_eval_metrics(metrics))


if __name__ == "__main__":
    main()
