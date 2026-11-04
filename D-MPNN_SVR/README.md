# D-MPNN models

Directed message-passing neural networks (Chemprop 2.3.1) fine-tuned from the CheMeleon foundation encoder, in two modes:

- **D-MPNN**: SMILES only. FFN 1×300, dropout 0, learning rate 1e-4 → 1e-3 → 1e-4 (2 warmup epochs).
- **Hybrid**: the same encoder plus the descriptor columns. FFN 3×1024, dropout 0.1, learning rate 1e-5 → 1e-4 → 1e-5 (4 warmup epochs).

Both use an 80/10/10 scaffold-balanced split (seed 0), up to 80 epochs, and early stopping with patience 15. With `--oversample-heavy` (run3), training molecules are copied by heavy-atom count: <12 ×5, 12–15 ×3, >15 ×1. Validation and test stay unique.

## Files

| File | What it does |
|---|---|
| `train_DMPNN.py` | Trains one or both modes. Self-contained; needs only Chemprop, numpy, pandas and RDKit. |
| `predict_DMPNN.py` | Scores one or more CSVs with trained models. Writes predictions and `metrics.csv`. |
| `selected_features` | The 256 descriptors used by the hybrid models in `run2_102926` and `run3_102926`. |
| `run2_102926/` | Trained run, no oversampling. |
| `run3_102926/` | Same split as `run2_102926`, with heavy-atom oversampling. |
| `eval_run2_102926/`, `eval_run3_102926/` | Predictions of the two runs on the four external sets, with `metrics.csv`. |

Both runs use `--feature-file selected_features`. All-features runs (`*_allfeat`) were made with the same scripts but are not included. The hyperparameter grid search from the original training code is not included either.

## Setup

```bash
conda create -n chemprop python=3.11 -y
conda activate chemprop
pip install chemprop==2.3.1
```

You can also run the commands below without activating the env, by prefixing them with `conda run --no-capture-output -n chemprop`.

## Feature selection (hybrid model)

| `train_DMPNN.py` option | Hybrid descriptors |
|---|---|
| none (default) | Every column of the training CSV except `ID`, `SMILES` and `entropy`, with zero-variance columns dropped. For `hybrid_train_102926.csv` that is 198 columns. |
| `--keep-constant-features` | The same, but constant columns kept (289). |
| `--feature-file selected_features` | Exactly the 256 listed columns, constant ones included (`run2_102926`, `run3_102926`). |

The list used is saved as `<run>/hybrid/feature_names.csv`, and `predict_DMPNN.py` reads it from there. `predict_DMPNN.py --feature-file X` only checks that `X` matches the model.

## Train

Run from this folder, after copying the data into `hybrid_data/` (see the top-level README). Each run takes about 10 minutes on an RTX 5070 Ti.

```bash
cd D-MPNN_SVR
TRAIN=hybrid_data/hybrid_train_102926.csv

# run2 recipe: no oversampling
python train_DMPNN.py -i $TRAIN -o my_run2 --feature-file selected_features --gpu 0 > my_run2.train.log 2>&1

# run3 recipe: run2's split plus heavy-atom oversampling
python train_DMPNN.py -i $TRAIN -o my_run3 --feature-file selected_features --gpu 0 \
    --oversample-heavy --splits-file my_run2/splits.json > my_run3.train.log 2>&1
```

Drop `--feature-file selected_features` to train on all descriptors.

| Option | Default | Meaning |
|---|---|---|
| `-i` | required | Training CSV with `ID`, `SMILES`, `entropy`, descriptors (and `heavy` for oversampling) |
| `-o` | required | New run folder. A non-empty folder is refused unless `--overwrite` is given. |
| `--mode` | `both` | `both` writes `<o>/dmpnn` and `<o>/hybrid`; `dmpnn` or `hybrid` writes the model straight into `<o>` |
| `--feature-file`, `--keep-constant-features` | none | Hybrid descriptor selection (table above) |
| `--oversample-heavy` | off | run3 oversampling |
| `--splits-file` | none | Reuse a `splits.json` (must cover the same rows) |
| `--gpu` | `0` | CUDA device index, or `cpu` |
| `--epochs`, `--patience`, `--seed`, `--num-workers` | 80, 15, 0, 4 | |
| `--ffn-hidden-dim`, `--ffn-num-layers`, `--dropout`, `--warmup-epochs`, `--init-lr`, `--max-lr`, `--final-lr` | per model (above) | Override for every trained model |

A run folder contains:
- `splits.json`: the unique-molecule split.
- `dmpnn/` and `hybrid/`: each with `model_0/best.pt`, `config.toml` and `train.log`. The hybrid folder also has `feature_names.csv` and `descriptors.npz`.
- `predictions.csv` and `metrics.csv`: the training table scored as all/train/val/test.
- Oversampled runs also have `splits_oversample.json` and `oversample_stats.json`.

## Predict and evaluate

```bash
D=hybrid_data
EVAL="-i $D/hybrid_std109_102926.csv -i $D/hybrid_std128_102926.csv -i $D/hybrid_pdbbind_102926.csv -i $D/hybrid_std6074_102926.csv"
python predict_DMPNN.py $EVAL -o my_eval_run2 \
    --dmpnn-dir run2_102926/dmpnn --hybrid-dir run2_102926/hybrid --gpu 0
```

- Inputs need `ID`, `SMILES`, `entropy` and, for the hybrid, the model's descriptor columns.
- The output folder gets `predictions.csv` (all inputs), one `<input>_predictions.csv` per input (columns `source, ID, SMILES, entropy, dmpnn_pred, hybrid_pred`) and `metrics.csv` (MAE, RMSE, R², bias). For one input, `-o file.csv` also works.
- An input with the same row count as the run's `splits.json` (the training table) is also scored on train/val/test.
- Either model folder can be given alone. Existing outputs and model or run folders are refused (`--overwrite` allows an existing output).

## Reproducibility checks (2026-11-03)

- **Prediction**: `predict_DMPNN.py` reproduced the saved evaluations of the original prediction code to within 0.0002 J⋅mol⁻¹⋅K⁻¹ per molecule; metrics agree to 10⁻⁵. Rerunning the original code on the same model differs by the same amount (GPU rounding).
- **Training**: 5-epoch runs of both recipes produced byte-identical `splits.json`, `splits_oversample.json`, `oversample_stats.json`, `train.csv`, Chemprop split files, `feature_names.csv` and `descriptors.npz` to the originals. Chemprop's saved `config.toml` differs only in the epoch count. The same holds for `--mode hybrid --feature-file selected_features --oversample-heavy` against `run3_102926/hybrid`.
- A full retrain will not reproduce the checkpoints bit for bit, because GPU training is not deterministic.
