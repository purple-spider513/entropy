# GINE models

Graph isomorphism networks with edge features (GINE) that predict the gas-phase standard molar entropy S° (J⋅mol⁻¹⋅K⁻¹) from a molecular graph, in two modes: the graph alone (`gnn_only`), and the graph plus the descriptor columns (`hybrid`).

## Files

| File | What it does |
|---|---|
| `train_GINE.py` | Trains both modes, optionally tunes hyperparameters (`--tune`), retrains winners with oversampling (`--source-run`, `--oversample 3seg`), and can score extra labelled CSVs (`--eval-csv NAME CSV`). |
| `predict_GINE.py` | Scores CSVs with a saved run or checkpoint. |
| `selected_features` | The 256 descriptors used by the hybrid models. |
| `run5_102926/` | Tuned GNN-only and hybrid models. `minimal/` holds the checkpoints (`gnn_only.pt`, `hybrid.pt`, `best_model.pt`), metrics, predictions, training histories and external results; `train.log` is the training log. |
| `run6_102926/` | The run5 winners retrained with 3-segment oversampling, laid out the same way. |

The two scripts are self-contained and reproduce `run5_102926` and `run6_102926`. They import nothing else from this repository.

## 1. The models

Both models work on a molecular graph built from the SMILES string with RDKit.

- **Atom features:** element, degree, formal charge, hybridization, aromaticity, ring membership, chirality and mass.
- **Bond features:** bond type, conjugation, ring membership and stereo, plus 16 radial-basis functions of the bond length from one ETKDG 3D conformer.

| Model | Inputs |
|---|---|
| `gnn_only` | GINE graph encoder (residual GINE layers, batch norm, sum + mean pooling) |
| `hybrid` | the same GINE encoder fused with an MLP over the CSV ligand descriptors |

Training settings:
- **Split:** the training CSV is split 80/10/10 into train/dev/test with seed 42, each molecule in one part only.
- **Descriptors:** count descriptors get log1p, then z-scores from the training part. The target is z-scored too.
- **Optimisation:** MSE loss, AdamW (learning rate 1e-3, weight decay 1e-5), mixed precision on GPU, batch size 64, up to 300 epochs.
- **Model selection:** the epoch with the best dev Pearson r is kept. The learning rate halves after 10 epochs without improvement, and training stops after 30.

The two recipes:
- **run5** (`--tune`): two baselines (4 layers, 128 channels, dropout 0.1), then a hyperparameter grid. The `gnn_only` and `hybrid` winners are chosen by dev Pearson r. In `run5_102926` both winners were 256 channels, 5 layers, dropout 0, with distance features.
- **run6** (`--source-run <run5> --oversample 3seg`): run5's two winners retrained with 3-segment oversampling of the training part only. Molecules with fewer than 12 heavy atoms are copied 5 times, 12–15 heavy atoms 3 times, and the rest once. Dev and test stay unique, and all scores are on unique molecules.

### Feature selection (training input)

| `train_GINE.py` option | Hybrid descriptors |
|---|---|
| no `--feature-file` | every column except ID, SMILES and entropy (all 289 descriptors of `hybrid_train_102926.csv`) |
| `--feature-file selected_features` | only the 256 listed columns, one per line, with `#` comments allowed. run5/run6 used this. |

The list is stored in every checkpoint, together with the descriptor scaling and the training molecules. `predict_GINE.py` therefore needs no feature option; its `--feature-file` only checks that a list matches the checkpoint.

## 2. Install the software

The scripts need Python 3.12, PyTorch with CUDA, PyTorch Geometric, RDKit, numpy, pandas, scipy and scikit-learn. This project used a conda environment named `GNN` with these versions:

| Package | Version | Source |
|---|---|---|
| python | 3.12.13 | conda-forge |
| rdkit | 2026.03.5 | conda-forge |
| numpy / pandas / scipy / scikit-learn | 2.5.2 / 3.0.5 / 1.18.0 / 1.9.0 | conda-forge |
| torch | 2.12.1+cu132 (CUDA 13.2) | pip, PyTorch wheel index |
| torch-geometric | 2.8.0.post1 | pip |

No PyG extension packages (`torch_scatter`, `torch_sparse`, `pyg_lib`) are needed.

### 2.1 Conda

Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) (or Anaconda) if `conda --version` does not work. For a GPU, check the NVIDIA driver with `nvidia-smi`. The "CUDA Version" it shows is the newest CUDA build the driver supports; the original machine showed 13.2 (driver 595.80).

### 2.2 Create the environment (recommended: conda + pip PyTorch)

PyTorch no longer publishes new versions on its own conda channel, so install the chemistry and science stack with conda and PyTorch with pip inside the conda env:

```bash
conda create -n GNN -c conda-forge python=3.12 rdkit numpy pandas scipy scikit-learn -y
conda activate GNN

# PyTorch with CUDA 13.2 (pick the cuXXX index that matches your driver at https://pytorch.org/get-started/locally/)
pip install torch==2.12.1 --index-url https://download.pytorch.org/whl/cu132
# CPU only:  pip install torch==2.12.1 --index-url https://download.pytorch.org/whl/cpu

pip install torch-geometric==2.8.0.post1
```

### 2.3 Alternative: everything from conda-forge

conda-forge also builds PyTorch and PyTorch Geometric:

```bash
conda create -n GNN -c conda-forge python=3.12 rdkit numpy pandas scipy scikit-learn pytorch-gpu pytorch_geometric -y
# CPU only: replace pytorch-gpu with pytorch
```

conda-forge picks its own PyTorch and CUDA versions. This route was not used for the runs here, so results can differ slightly.

### 2.4 Check the install

```bash
conda activate GNN
python -c "import torch, torch_geometric, rdkit; print(torch.__version__, torch.cuda.is_available(), torch_geometric.__version__, rdkit.__version__)"
python -c "import torch; [print(i, torch.cuda.get_device_name(i)) for i in range(torch.cuda.device_count())]"
```

The first line should end in `True` on a GPU machine. The second lists PyTorch's GPU indices, which `--gpu-index` uses. On the original machine `0` was the RTX 5070 Ti, because PyTorch lists the fastest card first; `nvidia-smi` numbers the cards differently.

You can also run every command below without activating the env, as `conda run --no-capture-output -n GNN python -u ...`.

## 3. Data

Run from this folder, after copying the data into `hybrid_data/` (see the top-level README). Each CSV needs an ID column, `SMILES`, `entropy` and the descriptor columns. For `--oversample 3seg`, the training CSV also needs `heavy`.

| File | Rows | Role |
|---|---|---|
| `hybrid_data/hybrid_train_102926.csv` | 15,546 | Training table (split 12,436 / 1,555 / 1,555) |
| `hybrid_data/hybrid_std128_102926.csv` | 128 | External test (benzyl alcohol / benzaldehyde labels corrected 2026-11-02) |
| `hybrid_data/hybrid_std109_102926.csv` | 109 | External test |
| `hybrid_data/hybrid_pdbbind_102926.csv` | 311 | External test |
| `hybrid_data/hybrid_std6074_102926.csv` | 6,074 | External test |

## 4. Train

```bash
cd GINE_SVR
conda activate GNN
D=hybrid_data
EVAL="--eval-csv std109 $D/hybrid_std109_102926.csv --eval-csv std128 $D/hybrid_std128_102926.csv \
      --eval-csv std6074 $D/hybrid_std6074_102926.csv --eval-csv pdbbind $D/hybrid_pdbbind_102926.csv"

# run5 recipe (about 1 h on the RTX 5070 Ti, including about 10 min of graph building)
nohup python -u train_GINE.py --csv $D/hybrid_train_102926.csv --output-dir my_run5 \
    --feature-file selected_features --tune --cache-dir my_cache --gpu-index 0 $EVAL \
    > my_run5.log 2>&1 &

# run6 recipe, after run5 has finished (about 20 min)
nohup python -u train_GINE.py --csv $D/hybrid_train_102926.csv --output-dir my_run6 \
    --feature-file selected_features --source-run my_run5 --oversample 3seg --cache-dir my_cache --gpu-index 0 $EVAL \
    > my_run6.log 2>&1 &
```

- **Cache:** with the same `--cache-dir`, run6 reuses run5's prepared graphs (about 250 MB) and skips the 10-minute rebuild.
- **All descriptors:** drop `--feature-file selected_features` to train on every descriptor column.
- **Quick test:** `--sample-limit 500 --max-epochs 2` checks the setup in a minute or two.

| Option | Default | Meaning |
|---|---|---|
| `--csv` | required | Training CSV |
| `--output-dir` | required | New run folder. A non-empty folder is refused unless `--overwrite` is given. |
| `--feature-file` | none (all descriptors) | Hybrid descriptor list |
| `--tune` | off | run5 hyperparameter grid after the baselines |
| `--source-run DIR` | none | Train the two winners recorded in `DIR/metrics.json` (run6) |
| `--oversample` | `none` | `3seg` for the run6 heavy-atom oversampling |
| `--eval-csv NAME CSV` | none | Score both models on a labelled CSV after training. Repeatable. |
| `--seed` | 42 | Split, oversampling, ETKDG and training seed |
| `--max-epochs`, `--patience`, `--batch-size` | 300, 30, 64 | |
| `--embed-3d` / `--no-embed-3d` | on | 3D bond lengths, or a 2D layout without them |
| `--device`, `--gpu-index` | auto, -1 | `cpu`/`cuda`; GPU index (-1 = the one with most free memory) |
| `--cache-dir` | none | Folder for prepared-graph caches |
| `--sample-limit` | 0 (all) | Use a random subset, for tests |

Without `--tune` or `--source-run`, only the two baselines are trained.

A run folder contains:
- **Checkpoints:** `gnn_only.pt`, `hybrid.pt` and `best_model.pt` (the better of the two by dev Pearson r).
- **Results:** `metrics.json`, with train/dev/test scores for both models, the winners and the oversampling counts. `candidate_metrics.csv` holds every trained candidate's dev scores.
- **Training curves:** `histories/<candidate>.csv`.
- **Predictions:** `gnn_only_predictions.csv` and `hybrid_predictions.csv` (train/dev/test).
- **Records of the data used:** `dataset_manifest.csv` (split of every molecule), `audit.csv` (molecules that failed), `preprocessing.json`, `feature_schema.json` and `training_identities.json`. `oversampling.json` is added for run6.
- **Settings:** `run_config.json` and `environment.json`.
- **External results** (with `--eval-csv`): `external/<model>/<set>/{predictions.csv, metrics.json, excluded_overlap.csv}` and `external_metrics.json`.

## 5. Predict and evaluate

```bash
python -u predict_GINE.py --run-dir run5_102926/minimal \
    --csv std109=$D/hybrid_std109_102926.csv --csv std128=$D/hybrid_std128_102926.csv \
    --csv std6074=$D/hybrid_std6074_102926.csv --csv pdbbind=$D/hybrid_pdbbind_102926.csv \
    --output-dir run5_eval --gpu-index 0
```

- **Models:** `--run-dir` uses the folder's `gnn_only.pt` and `hybrid.pt`. To score one model, use `--checkpoint path.pt` instead. Both options are repeatable.
- **Inputs:** `--csv NAME=path.csv`, repeatable. Without `NAME=`, the file name is used.
- **Output:** for each model and input, `predictions.csv`, `metrics.json` and `excluded_overlap.csv` go in `<output-dir>/<model>/<name>/`, laid out like the run's `external/` folder. `external_metrics.json` sums them up.
- **Prediction columns:** `ID, smiles, predicted_entropy, status, error, inchikey, canonical_smiles, entropy, overlap_with_train`.
- **Unlabelled CSVs:** a CSV without `entropy` gets predictions only.
- **Overlap:** molecules found in the training table by ID, InChIKey or canonical SMILES are flagged and left out of the scores. That removes 2 from std109, 1 from std128, 1 from PDBbind and 13 from std6074.
- **Other options:** `--entropy-scale 4.184` converts cal-based labels to J. `--feature-file selected_features` checks the checkpoint's descriptor list. `--device cpu` runs without a GPU, and `--overwrite` allows writing into an existing output folder.
- **Speed:** graphs are built once per input CSV and shared by all models. std6074 takes about 3 minutes.

## 6. Expected results

MAE in J⋅mol⁻¹⋅K⁻¹, with R² in parentheses, from `run*_102926/minimal/external/*/*/metrics.json` (std128 rescored with the corrected labels). The internal test set is the 10% test split (1,555 molecules).

| Model | Internal test | std128 (127) | std109 (107) | PDBbind (310) | std6074 (6,061) |
|---|---|---|---|---|---|
| run5 GNN only | 12.00 (0.992) | 10.70 (0.935) | 21.88 (0.345) | 14.24 (0.991) | 12.44 (0.937) |
| run5 hybrid | 10.13 (0.994) | 6.87 (0.966) | 15.09 (0.723) | 22.96 (0.973) | 20.32 (0.753) |
| run6 GNN only | 12.08 (0.992) | 12.16 (0.915) | 22.68 (0.230) | 15.94 (0.989) | 13.41 (0.925) |
| run6 hybrid | 10.82 (0.994) | 8.58 (0.952) | 13.49 (0.786) | 22.65 (0.973) | 17.53 (0.825) |

## 7. Reproducibility checks (2026-11-03)

- **Prediction on CPU:** `predict_GINE.py` gives byte-identical `predictions.csv`, `excluded_overlap.csv` and `metrics.json` to the original evaluation code, tested on std128 with run5 `gnn_only.pt` and run6 `hybrid.pt`.
- **Prediction on GPU:** all 16 model/test-set results agree with the saved ones within 0.01 in MAE.
  - Single molecules move by 0.2 or 0.4 J⋅mol⁻¹⋅K⁻¹, the rounding unit of the GPU's half-precision output.
  - The original code does the same: two identical runs differed by up to 0.2.
- **Training data and early epochs:** a 2-epoch run of the run5 recipe gave byte-identical `dataset_manifest.csv`, `audit.csv`, `preprocessing.json`, `feature_schema.json` and `training_identities.json` to `run5_102926`. Both models' first two training epochs also matched to every saved digit.
- **run6 recipe:** the same 2-epoch check of `--source-run run5_102926/minimal --oversample 3seg` gave byte-identical dataset files and `oversampling.json` to `run6_102926`, and identical first two epochs for both models.
- **Other code paths:** a short `--tune` run on 200 molecules with all 289 descriptors ran through the broad grid and the AttentiveFP fallback.
- **Full retrain:** not bit-for-bit repeatable on a GPU, so expect small changes in the last digits of the scores.
