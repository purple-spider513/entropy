# Uni-Mol + SVR stack

A two-step model:

1. **Uni-Mol v2 84m** is fine-tuned to predict S° directly from SMILES (25 epochs, batch 8, learning rate 5e-5).
   - `--oversample none`: each training molecule once (12,436 rows).
   - `--oversample 3seg`: copies by heavy-atom count: fewer than 12 heavy atoms ×5, 12–15 ×3, 16 or more ×1 (20,012 rows).
2. **An unscaled RBF SVR** (`C=10000`, `epsilon=0.1`, `gamma="scale"`) is fitted on the descriptors plus the Uni-Mol prediction (`UniMolPred`). It uses each training molecule once.

Both steps use only the 80% training split of `hybrid_train_102926.csv` (80/10/10, seed 42, giving 12,436 / 1,555 / 1,555). The 10% validation and test splits are scored after training.

This is the reverse of `../unimol_residual`: there the SVR comes first and Uni-Mol learns its error; here Uni-Mol predicts S° first and the SVR corrects it with descriptors.

## Files

| File | What it does |
|---|---|
| `Uni-Mol_SVR_train.py` | Fine-tunes Uni-Mol, fits the SVR, scores the internal splits and, optionally, external sets. Self-contained. |
| `Uni-Mol_SVR_prediction.py` | Predicts and scores CSVs with a trained model folder. Self-contained. |
| `selected_features` | The 256 SVR descriptors used by both included models. |
| `unimol_ft_entropy_no_oversample_102926_80_10_10_selfeat_svr/` | Model trained with `--oversample none` (manuscript Table 2, Uni-Mol columns). |
| `unimol_ft_entropy_os3seg_102926_80_10_10_selfeat_svr/` | Model trained with `--oversample 3seg` (manuscript Table 3, Uni-Mol columns). |

Each model folder contains:
- `unimol/`, the fine-tuned Uni-Mol model, and `svr_unimol_stack.joblib`, the SVR with its 257 input names (256 descriptors + `UniMolPred`).
- `config.json` with the settings. Its paths are from the original machine and are not needed for prediction.
- The splits `split_train.csv`, `split_val.csv`, `split_test.csv`, and `unimol_ft_train.csv` (the rows Uni-Mol was trained on, after oversampling).
- Internal results: `train_predictions.csv`, `val_10pct_preds.csv`, `test_10pct_preds.csv`, `split_metrics.json`.
- `official_eval/`: predictions and `metrics.json` for the four external sets. `official_eval_std128swap_110226/` holds std128 predictions rescored with the corrected benzyl alcohol and benzaldehyde labels.

The Uni-Mol checkpoints (`unimol/model_0.pth`) are too large for the repository. Download them from the release and place them as listed in the top-level README.

## Feature selection

| `Uni-Mol_SVR_train.py` option | SVR descriptors |
|---|---|
| `--feature-file selected_features` | Only the 256 listed columns (one per line; blank lines and `#` comments ignored). Both included models used this. |
| no `--feature-file` (default) | Every descriptor column in the training CSV: all columns except `ID`, `SMILES`, `entropy` and derived columns. That is 289 for `hybrid_train_102926.csv`. |

Any other list works too: write the column names into a text file and pass it with `--feature-file`. Training stops if a listed column is missing from the training CSV.

The chosen list is saved in `svr_unimol_stack.joblib` (and, for new runs, in `features.txt`). `Uni-Mol_SVR_prediction.py` reads it from there, so it takes no feature option. Its inputs only need to contain those columns.

## Install the software

Use the same `unimol` conda environment as `../unimol_residual`. Uni-Mol is distributed on PyPI as `unimol_tools`, so use conda only for a clean Python 3.12 environment and install the rest with pip. Install PyTorch first, or pip pulls in a CPU-only build.

| Package | Version |
|---|---|
| Python | 3.12.13 (conda-forge) |
| PyTorch | 2.13.0, CUDA 13.0 build |
| unimol_tools | 0.1.6 |
| RDKit | 2026.3.5 (pip wheel) |
| scikit-learn | 1.9.0 |
| pandas / NumPy / joblib | 3.0.5 / 2.2.6 / 1.5.3 |

```bash
conda create -n unimol python=3.12 -y
conda activate unimol
python -m pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu130
python -m pip install unimol_tools==0.1.6 rdkit==2026.3.5 scikit-learn==1.9.0 \
  pandas==3.0.5 numpy==2.2.6 joblib==1.5.3
```

- The CUDA 13.0 build needs NVIDIA driver 580 or newer. Check with `nvidia-smi`: "CUDA Version" in the top right must be 13.0 or higher. For an older driver or GPU, pick a matching build at https://pytorch.org/get-started/locally/, for example `--index-url https://download.pytorch.org/whl/cu126`.
- To copy the original environment exactly, use `python -m pip install -r ../unimol_residual/unimol_env_requirements.txt --extra-index-url https://download.pytorch.org/whl/cu130`.
- The first training run downloads the Uni-Mol v2 84m pretrained weights (322 MB) from Hugging Face. Fetching them in advance, and using them offline with `UNIMOL_WEIGHT_DIR`, is described in section 0.6 of `../unimol_residual/README.md`.
- Either activate the environment and run `python ...`, or prefix commands with `conda run --no-capture-output -n unimol`, which works without activating (for example inside `nohup`).

## Train

Run from this folder, after copying the data into `hybrid_data/` (see the top-level README). Training takes about 2 h (`none`) or 2 h 50 min (`3seg`) on an RTX 5070 Ti. Run one job at a time on the GPU.

```bash
cd Uni-Mol_SVR
PY="conda run --no-capture-output -n unimol python"
GPU=0     # CUDA device index or GPU UUID; the script default is a UUID from the original machine
D=hybrid_data
TESTS="$D/hybrid_std128_102926.csv $D/hybrid_std109_102926.csv $D/hybrid_pdbbind_102926.csv $D/hybrid_std6074_102926.csv"

# Same settings as unimol_ft_entropy_no_oversample_102926_80_10_10_selfeat_svr
nohup $PY Uni-Mol_SVR_train.py --train $D/hybrid_train_102926.csv --oversample none \
  --feature-file selected_features --external $TESTS --gpu $GPU \
  -o my_no_oversample_selfeat_svr > my_no_oversample_selfeat_svr.log 2>&1 &

# Same settings as unimol_ft_entropy_os3seg_102926_80_10_10_selfeat_svr (start after the first one finishes)
nohup $PY Uni-Mol_SVR_train.py --train $D/hybrid_train_102926.csv --oversample 3seg \
  --feature-file selected_features --external $TESTS --gpu $GPU \
  -o my_os3seg_selfeat_svr > my_os3seg_selfeat_svr.log 2>&1 &
```

Leave out `--feature-file selected_features` to use all 289 descriptors.

| Option | Default | Meaning |
|---|---|---|
| `--train` | required | Training CSV with `SMILES`, `entropy` (or `Entropy`), descriptors, and `heavy` (needed for `3seg`) |
| `-o, --output` | required | New model folder |
| `--oversample` | `none` | `none` or `3seg` |
| `--feature-file` | none (all descriptors) | Descriptor list for the SVR |
| `--external` | none | Labelled CSVs to score after training, written to `<output>/official_eval/` |
| `--epochs`, `--batch-size`, `--lr` | 25, 8, 5e-5 | Uni-Mol fine-tuning |
| `--split-seed` | 42 | 80/10/10 split |
| `--gpu` | a GPU UUID from the original machine | `CUDA_VISIBLE_DEVICES` value (a UUID or an index). Pass your own. |
| `--resume` | off | Allow an existing output folder: reuse its trained Uni-Mol, refit the SVR, rewrite everything else |

The script refuses an output folder that already holds a model unless `--resume` is given.

## Predict and evaluate

```bash
$PY Uni-Mol_SVR_prediction.py --model unimol_ft_entropy_os3seg_102926_80_10_10_selfeat_svr \
  --input $TESTS --gpu $GPU -o eval_os3seg_selfeat_svr
```

- `--model` is a model folder: one of the included models or one made by `Uni-Mol_SVR_train.py`. The output folder must differ from it.
- Each input needs `SMILES` and the model's descriptor columns. With an `entropy` column, MAE, RMSE, R² and bias are reported for Uni-Mol alone and for the stack. Without one, the script only predicts.
- Output, one file per input: `<name>_preds.csv` with columns `ID, SMILES, Entropy, n_heavy, UniMolPred, SVRPred, UniMol_Error, SVR_Error, TrainOverlap`. **`SVRPred` is the final prediction**; `UniMolPred` is Uni-Mol alone. `metrics.json` holds the per-file scores (`unimol` and `unimol_svr`) and a combined score. Uni-Mol conformers go to `<name>_conformers/`.
- Rows whose canonical SMILES are in the model's `split_train.csv` are marked `TrainOverlap=True` and not scored. This removes 1 molecule from std128, 2 from std109, 1 from PDBbind and 12 from std6074. `--overlap-with other.csv` changes the reference list, and `--no-overlap` scores every row.
- `--unimol DIR` points to a Uni-Mol model elsewhere (default `<model>/unimol`). `--gpu cpu` runs without a GPU, slowly.

## Expected results

MAE in J⋅mol⁻¹⋅K⁻¹, with R² in parentheses, from `official_eval/metrics.json`:

| Set | n | No oversampling: Uni-Mol | No oversampling: Uni-Mol + SVR | 3-segment: Uni-Mol | 3-segment: Uni-Mol + SVR |
|---|---|---|---|---|---|
| std128 | 127 | 38.53 (0.362) | 40.53 (0.314) | 10.30 (0.935) | 12.30 (0.914) |
| std109 | 107 | 63.04 (−1.184) | 64.57 (−1.204) | 29.25 (0.289) | 31.99 (0.193) |
| PDBbind | 310 | 15.29 (0.990) | 14.42 (0.992) | 14.03 (0.991) | 13.29 (0.992) |
| std6074 | 6,062 | 33.81 (0.669) | 35.44 (0.650) | 12.90 (0.933) | 13.36 (0.928) |

Internal 10% test split (1,555 molecules, DFT labels):

| Model | Uni-Mol MAE | Uni-Mol + SVR MAE |
|---|---|---|
| No oversampling | 11.80 | 9.86 |
| 3-segment | 11.99 | 10.79 |

A retrained model will not match exactly, because Uni-Mol fine-tuning on a GPU is not bit-for-bit deterministic. The SVR is refit on the new Uni-Mol predictions, so both columns can move.

## Reproducibility checks (2026-11-03)

- `Uni-Mol_SVR_prediction.py` on both included models reproduces all four `official_eval/*_preds.csv` files byte for byte, and the same scores in `metrics.json`. The new `metrics.json` also records the model path and row counts.
- Rerunning the os3seg recipe with `--resume` on a copy of the trained Uni-Mol gave byte-identical splits, `unimol_ft_train.csv`, internal predictions, `split_metrics.json` and `official_eval/` files.
- A 1-epoch run on 400 molecules exercised Uni-Mol training and the all-descriptor default (289 columns).
