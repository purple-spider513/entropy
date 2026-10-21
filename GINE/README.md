# GINE models

Graph isomorphism networks with edge features (GINE), trained on molecular graphs built from SMILES, in two modes:

- **GNN-only**: the molecular graph alone.
- **Hybrid**: the graph plus the descriptor columns.

## Files

| File | What it does |
|---|---|
| `train.py` | Trains both modes from one CSV, optionally tunes hyperparameters (`--always-tune`), picks a winner for each mode, and can score extra labelled CSVs (`--eval-csv NAME CSV`). |
| `predict.py` | Predicts entropy for a CSV with a saved checkpoint. |
| `evaluate_external.py` | Scores a saved checkpoint on a labelled CSV and writes metrics and predictions. |
| `run5_GINE_tune_hybrid/` | Tuned GNN-only and hybrid models. `minimal/` holds the checkpoints (`gnn_only.pt`, `hybrid.pt`, `best_model.pt`), metrics, predictions and training histories. |
| `run6_os_hybrid/` | The run5 winners retrained with three-segment oversampling of the training rows. `minimal/` holds their results. |
| `run6_os_hybrid/train_oversampled.py` | The script for that retraining. It takes no command-line options. Its settings are constants at the top of the file, and its default output folder is `run6/minimal`. |
| `hybrid_data/` | Copies of the training and external CSVs. |

## Requirements

PyTorch (GPU optional) and the `entropy_gnn` package. The scripts import `entropy_gnn`, but it is not included in this repository. Install it, or add it to `PYTHONPATH`, before running anything.

## Usage

```bash
cd GINE
python train.py --csv hybrid_data/hybrid_train_091926.csv --output-dir my_run \
    --always-tune --eval-csv std6074 hybrid_data/hybrid_std6074.csv
python predict.py --checkpoint my_run/hybrid.pt --csv hybrid_data/hybrid_std128.csv \
    --output-csv std128_predictions.csv
```

Run `python train.py --help` for the other options, such as seed, epochs, patience, batch size and device.
