# SVR + Uni-Mol residual models

Prediction = SVR(descriptors) + Uni-Mol(SMILES residual).

The SVR is an unscaled RBF model (`C=10000`, `epsilon=0.1`). Uni-Mol v2 (84m) is trained on the SVR residuals. Before that, the residual rows are copied with three-segment oversampling by heavy-atom count: fewer than 12 heavy atoms get 5 copies, 12 to 15 get 3 copies, and 16 or more get 1 copy.

## Files

| File | What it does |
|---|---|
| `svr_general_residual_unimol_train.py` | Trains the SVR and the Uni-Mol residual model. Options include `--feature-file`, `--hold-fraction` and `--epochs`. |
| `svr_general_residual_unimol_predict.py` | Predicts new CSVs with a trained model folder. |
| `svr_general_residual_run_bond/` | Model that uses the 256 features in the top-level `selected_features` file. |
| `svr_general_residual_run_heavy/` | Model that uses all 289 descriptor columns. |
| `svr_general_residual_run_eval_bond2/`, `svr_general_residual_run_eval_heavy2/` | Predictions of the two models above on the four external sets. |
| `rate_evaluation/` | Predictions for 10k to 500k molecules. The `Entropy` column is all zeros in these files, so their metrics are not meaningful. Use `Predicted_Entropy`. |
| `svr_residual_bondfeat_std6074_80_10_10/` | Model trained and tested on an 80/10/10 split of `hybrid_std6074.csv`. Includes `calc_metrics.py`, which computes R², MAE, RMSE, Pearson r and MRE from a CSV with `obs` and `pred` columns, and `bin.py`, which bins the deviations in `test.dat`. |

Each model folder contains `tabular.joblib` (the SVR), `unimol/` (the Uni-Mol checkpoint), `config.json`, the split files and the predictions.

## Setup

Python 3.12 with PyTorch (CUDA), `unimol_tools`, RDKit, scikit-learn, pandas, numpy and joblib. Install PyTorch before `unimol_tools`, or pip may install a CPU-only build.

```bash
conda create -n unimol python=3.12 -y
conda activate unimol
python -m pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu130
python -m pip install unimol_tools==0.1.6 rdkit==2026.3.5 scikit-learn==1.9.0 \
    pandas==3.0.5 numpy==2.2.6 joblib==1.5.3
```

## Usage

Run these commands from inside this folder. The saved configs use paths relative to it.

```bash
cd Uni-Mol_Residual
python svr_general_residual_unimol_train.py --train ../hybrid_data/hybrid_train_091926.csv \
    --output my_run --gpu 0
python svr_general_residual_unimol_predict.py --model svr_general_residual_run_bond \
    --input ../hybrid_data/hybrid_std128.csv --output my_predictions --gpu 0
```

Notes:
- `--output` must be a new folder. Training refuses to overwrite an existing model.
- The default `--gpu` is a GPU ID from the original machine. Pass `--gpu 0` or `--gpu cpu`.
- Prediction needs the Uni-Mol checkpoints (`unimol/model_0.pth`). See the Large files section of the top-level README.
