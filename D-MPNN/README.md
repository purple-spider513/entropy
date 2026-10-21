# D-MPNN models

Directed message-passing neural networks (Chemprop) fine-tuned from the CheMeleon foundation weights, in two modes:

- `dmpnn`: SMILES only.
- `hybrid`: SMILES plus the descriptor columns.

## Files

| File | What it does |
|---|---|
| `train.py` | Trains one or both modes on an 80/10/10 scaffold split. Options include `--mode`, `--oversample-heavy` and `--hpopt` (hyperparameter grid). |
| `predict.py` | Scores one or more CSVs with trained models. Writes predictions and `metrics.csv`. |
| `hybrid_data/` | Copies of the training and external CSVs. |
| `run2/` | Trained run with default hyperparameters and no oversampling. |
| `run3/` | Same split as `run2`, with heavy-atom oversampling of the training rows. |
| `eval_run2/`, `eval_run3/` | Predictions of `run2` and `run3` on the four external sets, with `metrics.csv`. |
| `*/_work/` | Intermediate files written during training and prediction. |

Each run folder contains `dmpnn/model_0/best.pt`, `hybrid/model_0/best.pt`, `splits.json`, `predictions.csv` and `metrics.csv`.

## Setup

The original runs used Python 3.11 and Chemprop 2.3.1. CheMeleon needs Chemprop 2.2 or newer.

```bash
conda create -n chemprop python=3.11 -y
conda activate chemprop
pip install chemprop==2.3.1
```

`train.py` and `predict.py` call Chemprop through `conda run -n chemprop` when conda is available, and through `chemprop` on the PATH otherwise. A GPU is optional. Without one, pass `--gpu cpu`.

## Usage

```bash
cd D-MPNN
python train.py -i hybrid_data/hybrid_train_091926.csv -o my_run --gpu 0
python predict.py -i hybrid_data/hybrid_std128.csv -o std128_predictions.csv \
    --dmpnn-dir my_run/dmpnn --hybrid-dir my_run/hybrid --gpu 0
```

Use a new `-o` folder for training so that `run2/` and `run3/` are not overwritten.

The config and log files in the run folders contain paths from the original machine. They are a record of each run and are not needed to use the models.
