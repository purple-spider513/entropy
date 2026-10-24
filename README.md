# Gas-phase entropy prediction

Code, data, and trained models for predicting the gas-phase entropy of small molecules (ligands) from their structure. The repository includes support-vector regression (SVR) on count descriptors, a Chemprop D-MPNN, a GINE graph neural network, and two models that combine SVR with Uni-Mol.

## Layout

```
.
├── hybrid_data/             Training and external test CSV files
├── descriptor_preparation/  Builds atom, bond and element count tables from mol2 files
├── classical_SVR/           Support-vector regression baselines
├── D-MPNN/                  Chemprop D-MPNN models (SMILES only, and SMILES plus descriptors)
├── GINE/                    GINE graph neural network models
├── Uni-Mol_train/           SVR stacked on a Uni-Mol prediction
├── Uni-Mol_Residual/        SVR plus a Uni-Mol model that learns the SVR residual
└── selected_features        The 256 descriptor names used by the bond-feature residual model
```

Each folder has its own README that lists its files and shows how to run them.

## Data

- `hybrid_data/hybrid_train_091926.csv`: main training set, 14,459 molecules. Columns are `ID`, `SMILES`, `entropy` (the target) and 289 descriptor columns.
- `hybrid_data/hybrid_pdbbind.csv`, `hybrid_std109.csv`, `hybrid_std128.csv` and `hybrid_std6074.csv`: external test sets with 311, 109, 128 and 6,074 molecules.

`classical_SVR/`, `D-MPNN/hybrid_data/` and `GINE/hybrid_data/` hold copies of `hybrid_train_091926.csv` and the four external sets, so that each model folder can run on its own.

## Trained Uni-Mol checkpoints

The three Uni-Mol checkpoints are about 334 MB each, too large for this repository. Download them from the model checkpoints release and place them at these paths:

| Release file | Place at |
| --- | --- |
| `svr_general_residual_run_bond_model_0.pth` | `Uni-Mol_Residual/svr_general_residual_run_bond/unimol/model_0.pth` |
| `svr_general_residual_run_heavy_model_0.pth` | `Uni-Mol_Residual/svr_general_residual_run_heavy/unimol/model_0.pth` |
| `svr_residual_bondfeat_std6074_80_10_10_model_0.pth` | `Uni-Mol_Residual/svr_residual_bondfeat_std6074_80_10_10/unimol/model_0.pth` |

The D-MPNN checkpoints and the Uni-Mol prediction file (`Uni-Mol_Residual/predict_rate/svr_general_residual_run_bond_500k_preds/hybrid_test_500k_preds.csv`) are included in the repository, and are between 50 and 100 MB each.

## Before running

- Run each script from inside its own folder. The scripts use relative paths.
- Several scripts contain paths and GPU IDs from the machine they were written on. The folder READMEs say what to change.
- `GINE` needs the `entropy_gnn` package, which is not included in this repository.
