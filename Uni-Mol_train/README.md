# Uni-Mol + SVR stack

An unscaled RBF SVR (`C=10000`, `epsilon=0.1`) fitted to the descriptor columns plus a Uni-Mol entropy prediction (`UniMolPred`). Uni-Mol itself is not retrained here. The scripts take prediction CSVs from a finished Uni-Mol run.

## Files

| File | What it does |
|---|---|
| `unimol_ft_svr_stack_train.py` | Fits the SVR stack and reports train, validation and test metrics. |
| `unimol_ft_svr_stack_predict.py` | Applies a saved stack to new CSVs. |
| `unimol_ft_entropy_no_oversample_091926_80_10_10_svr/` | Stack built on Uni-Mol predictions from a run without oversampling. |
| `unimol_ft_entropy_os3seg_091926_80_10_10_svr/` | Stack built on Uni-Mol predictions from a run with three-segment oversampling. |

Each stack folder contains `svr_unimol_stack.joblib`, `config.json`, `split_metrics.json`, the Uni-Mol prediction CSVs for the train, validation and test splits, and `official_eval/`, which holds predictions for the external sets.

## Usage

```bash
cd Uni-Mol_train
python unimol_ft_svr_stack_train.py --train ../hybrid_data/hybrid_train_091926.csv \
    --unimol-source unimol_ft_entropy_no_oversample_091926_80_10_10_svr \
    --oversample none --output my_stack
python unimol_ft_svr_stack_predict.py --model unimol_ft_entropy_os3seg_091926_80_10_10_svr \
    --unimol-preds unimol_ft_entropy_os3seg_091926_80_10_10_svr/official_eval/hybrid_std128_preds.csv \
    --input ../hybrid_data/hybrid_std128.csv --output my_predictions
```

Notes:
- `--unimol-source` must contain `train_predictions.csv`, `val_10pct_preds.csv` and `test_10pct_preds.csv`. The stack folders include these files.
- The Uni-Mol checkpoints are not in this folder, so use `--unimol-preds` rather than `--unimol` when predicting.
- To drop external molecules that also appear in the training data, add `--overlap-with ../hybrid_data/hybrid_train_091926.csv` when predicting. The saved configs do not record this path.
- The `config.json` files contain paths from the original machine.
