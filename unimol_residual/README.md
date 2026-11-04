# SVR + Uni-Mol residual models

Prediction = SVR(descriptors) + Uni-Mol(SMILES → SVR residual).

This covers two models:

| Model directory | SVR inputs | Feature list |
|---|---|---|
| `svr_general_residual_run_bond_conformer_symmetry` | 256 descriptors (bond features + `ring`, `heavy`, …) | `./selected_features` |
| `svr_general_residual_run_all_conformer_symmetry` | 289 descriptors (every descriptor column in the CSV) | none (default) |

Both use the same method: unscaled RBF SVR (`C=10000`, `epsilon=0.1`) on the descriptors, then Uni-Mol v2 84m trained on the SVR residual with 3-segment oversampling (fewer than 12 heavy atoms ×5, 12–15 ×3, 16 or more ×1). The final prediction is SVR + Uni-Mol residual. In the manuscript, the all-features model is the "All features residual" row in Table 4 and the bond model is the "Bond-features residual" row.

## Files

| File | What it does |
|---|---|
| `svr_general_residual_unimol_train.py` | Trains the SVR and the Uni-Mol residual model on an 80/10/10 split (seed 42). Run with `--help` for all options. |
| `svr_general_residual_unimol_predict.py` | Predicts new CSVs with a trained model folder. |
| `svr_general_residual_run_bond_conformer_symmetry/` | Bond-features model (256 descriptors). |
| `svr_general_residual_run_all_conformer_symmetry/` | All-features model (289 descriptors). |
| `svr_general_residual_run_eval_{bond,all}_conformer_symmetry/` | Predictions of the two models on the four external sets, with `metrics.json`. |
| `svr_residual_allfeat_std6074_102926_80_10_10/` | Same method, trained and tested on an 80/10/10 split of `hybrid_std6074_102926.csv` (section 6). |
| `unimol_env_requirements.txt` | Exact pip package list of the environment used. |

The Uni-Mol checkpoints (`unimol/model_0.pth`) are too large for the repository. Download them from the release and place them as listed in the top-level README.

## 0. Install the software

Uni-Mol is not a conda package. It is distributed on PyPI as `unimol_tools`. Use conda only to create a clean Python 3.12 environment, then install everything inside it with pip. The `unimol` environment used for these runs was built that way:

| Package | Version |
|---|---|
| Python | 3.12.13 (conda-forge) |
| PyTorch | 2.13.0, CUDA 13.0 build |
| unimol_tools | 0.1.6 |
| RDKit | 2026.3.5 (pip wheel) |
| scikit-learn | 1.9.0 |
| pandas / NumPy / joblib | 3.0.5 / 2.2.6 / 1.5.3 |

If the `unimol` environment already exists (`conda env list`), skip to step 1.

### 0.1 Conda

If conda is missing, install Miniconda in your home directory:

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3
$HOME/miniconda3/bin/conda init bash
exec bash
```

### 0.2 Create the environment

```bash
conda create -n unimol python=3.12 -y
conda activate unimol
```

### 0.3 Install PyTorch with GPU support (first)

Install PyTorch before `unimol_tools`. Otherwise pip pulls in a CPU-only PyTorch.

```bash
python -m pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu130
```

The CUDA 13.0 build needs an NVIDIA driver of version 580 or newer. The original machine had driver 595.80. Check yours with `nvidia-smi` (top right, "CUDA Version" must be 13.0 or higher). The CUDA toolkit does not need to be installed separately, because the pip wheel brings its own CUDA libraries. The RTX 5070 Ti (Blackwell) needs a CUDA 12.8 or newer build. On an older driver, choose a matching build from https://pytorch.org/get-started/locally/, for example `--index-url https://download.pytorch.org/whl/cu126`, for a GPU older than the RTX 50 series.

### 0.4 Install Uni-Mol and the other packages

```bash
python -m pip install \
  unimol_tools==0.1.6 \
  rdkit==2026.3.5 \
  scikit-learn==1.9.0 \
  pandas==3.0.5 \
  numpy==2.2.6 \
  joblib==1.5.3
```

`unimol_tools` installs its own dependencies (Hydra, LMDB, Numba, Transformers, TensorBoard and others).

To copy the original environment package for package instead of running 0.3 and 0.4, use the saved list `unimol_env_requirements.txt` in this folder:

```bash
python -m pip install -r unimol_env_requirements.txt \
  --extra-index-url https://download.pytorch.org/whl/cu130
```

### 0.5 Check the install

```bash
python - <<'PY'
import torch, unimol_tools, rdkit, sklearn, pandas, numpy, joblib
print("torch", torch.__version__, "CUDA", torch.version.cuda, "GPU available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
print("rdkit", rdkit.__version__, "| scikit-learn", sklearn.__version__, "| pandas", pandas.__version__)
PY
```

`GPU available:` must print `True` on the training machine.

### 0.6 Download the Uni-Mol 2 pretrained weights

The first training run downloads the Uni-Mol v2 84m checkpoint (322 MB) from Hugging Face (`dptech/Uni-Mol2`) automatically. To fetch it in advance, for example on a node with internet access:

```bash
python -c "
from unimol_tools.config import MODEL_CONFIG_V2
from unimol_tools.weights import weight_download_v2, get_weight_dir
weight_download_v2(MODEL_CONFIG_V2['weight']['84m'], get_weight_dir())"
```

The file lands in `<env>/lib/python3.12/site-packages/unimol_tools/weights/modelzoo/84M/checkpoint.pt`. To keep the weights somewhere else, or on a machine without internet access, copy `modelzoo/84M/checkpoint.pt` into a folder and set `export UNIMOL_WEIGHT_DIR=/path/to/that/folder` before training. That folder must contain `modelzoo/84M/checkpoint.pt`. If Hugging Face is blocked, unimol_tools tries a mirror automatically, or you can set `HF_ENDPOINT`.

### 0.7 Using the environment

Either activate it (`conda activate unimol`) and drop the `conda run ...` prefix from the commands below, or keep the prefix `conda run --no-capture-output -n unimol`, which works without activating, for example inside `nohup` or a script.

## 1. Environment and data

Run everything from this folder with the `unimol` conda environment. First copy the data and `selected_features` here (see the top-level README):

```bash
cd unimol_residual
GPU=0     # CUDA device index or GPU UUID; the script default is a UUID from the original machine
```

The scripts set `CUDA_VISIBLE_DEVICES` to this value. List your GPUs with `nvidia-smi --query-gpu=index,name,uuid --format=csv`. Use `--gpu cpu` only for prediction; training on a CPU is far too slow.

Input files in `hybrid_data/`:

| File | Rows | Role |
|---|---|---|
| `hybrid_train_102926.csv` | 15,546 | Training pool. Labels include the conformer-multiplicity and rotational-symmetry corrections. |
| `hybrid_std128_102926.csv` | 128 | Experimental test set. Labels for 100.mol2 (benzyl alcohol, 373.589) and 105.mol2 (benzaldehyde, 336.017) were corrected on 2026-11-02. |
| `hybrid_std109_102926.csv` | 109 | Experimental test set |
| `hybrid_pdbbind_102926.csv` | 311 | DFT-labeled PDBbind ligands |
| `hybrid_std6074_102926.csv` | 6,074 | Experimental test set |

Each CSV has `ID`, `SMILES`, `entropy` and the descriptor columns.

## 2. Train

The train script refuses an output directory that already holds `config.json` or `tabular.joblib`. To rerun without touching the existing models, give the output a new name (the `_rerun` suffix below), or move the old directory away first.

Bond-features model (about 1 h 45 min on the 5070 Ti):

```bash
nohup conda run --no-capture-output -n unimol python svr_general_residual_unimol_train.py \
  --train hybrid_data/hybrid_train_102926.csv \
  --feature-file ./selected_features --include-heavy \
  --gpu $GPU \
  --output svr_general_residual_run_bond_conformer_symmetry_rerun \
  > train_bond_rerun.log 2>&1 &
```

All-features model (about 2 h 45 min):

```bash
nohup conda run --no-capture-output -n unimol python svr_general_residual_unimol_train.py \
  --train hybrid_data/hybrid_train_102926.csv \
  --include-heavy \
  --gpu $GPU \
  --output svr_general_residual_run_all_conformer_symmetry_rerun \
  > train_all_rerun.log 2>&1 &
```

Run the two jobs one after the other on the same GPU, not at the same time.

The default 80/10/10 split (seed 42) gives 12,436 training, 1,555 validation and 1,555 test molecules. SVR and Uni-Mol see only the 80%. `--include-heavy` changes nothing here (`heavy` is already in `selected_features` and in the default list); it is kept to match the original run.

Each model directory contains:

| File | Content |
|---|---|
| `tabular.joblib` | Fitted SVR and its feature list |
| `unimol/` | Uni-Mol residual model (`model_0.pth`, scaler, config) |
| `config.json` | All settings, split sizes and the val/test scores |
| `split_train.csv`, `split_val.csv`, `split_test.csv` | The split. `split_train.csv` is also used to flag overlap at prediction time. |
| `residual_base.csv`, `residual_train.csv` | SVR residuals before and after oversampling |
| `val_preds.csv`, `test_preds.csv`, `split_metrics.json` | Internal 10% validation and test results |

Internal test-split results of the existing models (`split_metrics.json`, MAE in J⋅mol⁻¹⋅K⁻¹):

| Model | SVR MAE | SVR + Uni-Mol MAE | R² |
|---|---|---|---|
| bond | 12.88 | 8.93 | 0.9954 |
| all | 12.78 | 8.77 | 0.9956 |

## 3. Evaluate on the four external sets

Prediction takes under a minute per model. The output directory must differ from the model directory.

```bash
TESTS="hybrid_data/hybrid_std128_102926.csv hybrid_data/hybrid_std109_102926.csv \
       hybrid_data/hybrid_pdbbind_102926.csv hybrid_data/hybrid_std6074_102926.csv"

for tag in bond all; do
  conda run --no-capture-output -n unimol python svr_general_residual_unimol_predict.py \
    --model svr_general_residual_run_${tag}_conformer_symmetry \
    --input $TESTS --gpu $GPU \
    --output svr_general_residual_run_eval_${tag}_conformer_symmetry_rerun \
    > eval_${tag}_rerun.log 2>&1
done
```

To score a retrained model, point `--model` at the `_rerun` directory.

Test molecules whose canonical SMILES appear in the model's `split_train.csv` stay in the prediction CSV with `TrainOverlap=True` but are left out of every score. This removes 1 molecule from std128, 2 from std109, 1 from PDBbind and 12 from std6074, which leaves 127, 107, 310 and 6,062 scored molecules. Use `--no-overlap` to score every row, or `--overlap-with other.csv` to use a different training list.

Outputs in the eval directory:

- `<set>_preds.csv`, one row per molecule with these columns: `ID, SMILES, Entropy, n_heavy, SVRPred, Residual, Predicted_Entropy, SVR_Error, Absolute_Error, Signed_Error, TrainOverlap`. `Predicted_Entropy` is SVR + Uni-Mol; `SVRPred` is the SVR stage alone.
- `metrics.json` with MAE, RMSE, R² and bias for the SVR stage and for the hybrid model, per set and combined.
- `<set>_conformers/` with the Uni-Mol conformers.

## 4. Mean relative error

`metrics.json` does not include the mean relative error (MRE) quoted in the manuscript. Compute it from the prediction CSVs, leaving out the overlap rows:

```bash
conda run --no-capture-output -n unimol python - <<'PY'
import pandas as pd
for tag in ["bond", "all"]:
    ev = f"svr_general_residual_run_eval_{tag}_conformer_symmetry"   # or the _rerun directory
    for s in ["std128", "std109", "std6074", "pdbbind"]:
        d = pd.read_csv(f"{ev}/hybrid_{s}_102926_preds.csv")
        d = d[~d.TrainOverlap]
        mae = d.Absolute_Error.mean()
        mre = (d.Absolute_Error / d.Entropy).mean() * 100
        print(f"{tag:4s} {s:8s} n={len(d):5d}  MAE {mae:6.2f}  MRE {mre:.2f}%")
PY
```

## 5. Expected results

These are the current results in `svr_general_residual_run_eval_{bond,all}_conformer_symmetry/` (MAE in J⋅mol⁻¹⋅K⁻¹):

| Set | n | Bond: SVR MAE | Bond: hybrid MAE | Bond: hybrid MRE | All: SVR MAE | All: hybrid MAE | All: hybrid MRE |
|---|---|---|---|---|---|---|---|
| std128 | 127 | 7.19 | 7.33 | 2.00% | 7.11 | 6.51 | 1.79% |
| std109 | 107 | 8.24 | 6.78 | 2.29% | 8.26 | 6.54 | 2.21% |
| PDBbind | 310 | 14.38 | 11.51 | 1.65% | 14.33 | 11.73 | 1.73% |
| std6074 | 6,062 | 12.42 | 12.76 | 3.17% | 12.45 | 12.58 | 3.12% |

The SVR stage of the bond model gives the same predictions as the stand-alone bond-features SVR (MRE 2.03%, 2.79% and 3.10% on std128, std109 and std6074; Table 1). Table 1 itself comes from a separate scikit-learn SVR rerun, which explains the 0.01 differences, for example RMSE 16.67 vs 16.68 on std6074.

A retrained model will not match these numbers exactly. Uni-Mol training on a GPU is not bit-for-bit deterministic, so expect hybrid MAEs to move by a few tenths. The SVR-stage numbers should reproduce exactly.

## 6. Model trained on std6074: `svr_residual_allfeat_std6074_102926_80_10_10`

This model uses the same method (unscaled RBF SVR on all 289 descriptors, then Uni-Mol v2 84m on the SVR residual with 3-segment oversampling), but it is trained on the experimental std6074 set instead of the DFT table. It is scored on its own held-out 10% of std6074. The manuscript gives this result as "retrained on std6074, it reached 1.12% on held-out molecules".

It was trained with a different script, `svr_residual_hybrid_80_10_10_train.py`, which is **not included in this repository** (it depends on helper modules from the original working folder). The trained model and its results are included. For reference, the original command was:

```bash
export CUDA_VISIBLE_DEVICES=GPU-11559549-d592-eb13-a511-44849442d2a5   # RTX 5070 Ti (script default)

nohup conda run --no-capture-output -n unimol python svr_residual_hybrid_80_10_10_train.py \
  --train hybrid_data/hybrid_std6074_102926.csv \
  --feature-set all \
  --output svr_residual_allfeat_std6074_102926_80_10_10_rerun \
  > svr_residual_allfeat_std6074_102926_80_10_10_rerun.train.log 2>&1 &
```

- `--feature-set all` uses every descriptor column (289). `--feature-set bond` uses the bond list read from `SVR_predS_bond.py` (the `svr_residual_bondfeat_std6074_102926_80_10_10` run).
- The script has no `--gpu` option. Choose the GPU with `CUDA_VISIBLE_DEVICES`, as above.
- **This script does not protect an existing output directory.** Always give a new `--output` name, or the original model is overwritten.
- The split is 80/10/10 with seed 42: 4,859 training, 607 validation and 608 test molecules from the 6,074. Oversampling turns the 4,859 residual rows into 22,399.
- Training takes about 1 h 20 min on the 5070 Ti, most of it Uni-Mol.

### Evaluate

The script scores the held-out splits itself at the end of training; no separate prediction step is needed. The results are in the output directory:

| File | Content |
|---|---|
| `split_metrics.json` (also in `config.json`) | MAE, RMSE, R² and bias for the SVR stage and for SVR + Uni-Mol on the 10% validation and 10% test splits |
| `test_10pct_preds.csv`, `val_10pct_preds.csv` | Per-molecule `Entropy`, `SVRPred`, `Residual`, `Predicted_Entropy`, `Absolute_Error`, `Signed_Error` |
| `test_10pct_svr.csv`, `val_10pct_svr.csv` | SVR-stage predictions only |
| `split_train.csv`, `split_val.csv`, `split_test.csv` | The split |
| `tabular.joblib`, `unimol/` | The trained SVR and Uni-Mol residual model |

Mean relative error:

```bash
conda run --no-capture-output -n unimol python - <<'PY'
import pandas as pd
d = "svr_residual_allfeat_std6074_102926_80_10_10"   # or the _rerun directory
for f in ["val_10pct_preds", "test_10pct_preds"]:
    p = pd.read_csv(f"{d}/{f}.csv")
    print(f, len(p),
          "MAE %.2f" % p.Absolute_Error.mean(),
          "MRE %.2f%%" % ((p.Absolute_Error / p.Entropy).mean() * 100),
          "SVR-stage MRE %.2f%%" % ((p.SVR_Error.abs() / p.Entropy).mean() * 100))
PY
```

Expected results of the existing model (MAE in J⋅mol⁻¹⋅K⁻¹):

| Split | n | SVR MAE | SVR + Uni-Mol MAE | R² | MRE | SVR-stage MRE |
|---|---|---|---|---|---|---|
| validation | 607 | 8.30 | 5.18 | 0.9879 | 1.29% | 2.08% |
| test | 608 | 8.02 | 4.52 | 0.9904 | 1.12% | 1.99% |

Do not use this model on std6074 itself, because 80% of std6074 is in its training data.

### 10-fold cross-validation (Benson comparison)

The cross-validated MAE compared with Benson group additivity in the manuscript comes from `svr_residual_allfeat_std6074_102926_10fold_train.py`, which is **not included in this repository**, and neither are its outputs. It repeats the run above for 10 disjoint test folds (KFold, shuffle, seed 42). Inside each fold, it keeps an 80/10/10 train/validation/test split.

```bash
nohup conda run --no-capture-output -n unimol python svr_residual_allfeat_std6074_102926_10fold_train.py \
  --gpu GPU-11559549-d592-eb13-a511-44849442d2a5 \
  --output svr_residual_allfeat_std6074_102926_10fold_rerun \
  > svr_residual_allfeat_std6074_102926_10fold_rerun.log 2>&1 &
```

- The default `--train` is `hybrid_data/hybrid_std6074_102926.csv`.
- A finished fold is skipped on restart.
- To use several GPUs, start one process per GPU with `--worker`, each with its own `--gpu`. The processes share the folds, and the last one pools the results.
- `--aggregate-only` rebuilds the summary from finished folds.
- Ten folds take roughly ten times the single run, about 13 h on one GPU. The original run trained several folds in parallel with `--worker` and finished in about 6.5 h on 2026-11-01.

The outputs were `cv_summary.json`, `fold_metrics.csv`, `oof_predictions.csv` and `std6074_benson_residual_bruce_comparison_102926.csv`. The original run gave a pooled out-of-fold MAE of 1.20 cal⋅mol⁻¹⋅K⁻¹ over all 6,074 molecules (fold-to-fold standard deviation 0.04), against 4.20 cal⋅mol⁻¹⋅K⁻¹ for Benson.

## 7. Notes

- The std128 labels for benzyl alcohol and benzaldehyde were corrected in `hybrid_std128_102926.csv` on 2026-11-02. The std128 prediction files and `metrics.json` in both eval folders were rescored with the corrected labels; the predictions themselves did not change.
- To train on all 15,546 molecules with no holdout, add `--hold-fraction 0`. The run then has no internal val/test scores, and `split_train.csv` contains every molecule, so more external molecules get flagged as overlap.
