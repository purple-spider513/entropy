# Gas-phase entropy prediction

Code, data, and trained models for predicting the gas-phase standard molar entropy S° (J⋅mol⁻¹⋅K⁻¹) of small molecules (ligands) from their structure. The repository includes support-vector regression (SVR) on count descriptors, a Chemprop D-MPNN, a GINE graph neural network, and two models that combine SVR with Uni-Mol.

## Layout

```
.
├── descriptor_preparation/  Builds descriptor tables and corrected entropy labels from mol2 files
├── classical_SVR/           SVR baselines, plus the training and external test CSVs
├── D-MPNN_SVR/              Chemprop D-MPNN models (SMILES only, and SMILES plus descriptors)
├── GINE_SVR/                GINE graph neural network models (graph only, and graph plus descriptors)
├── Uni-Mol_SVR/             Fine-tuned Uni-Mol, then an SVR on descriptors plus the Uni-Mol prediction
├── unimol_residual/         SVR first, then Uni-Mol trained on the SVR residual
└── eval_500k/               Prediction speed test of the residual model on 10k to 500k molecules
```

Each folder has its own README that lists its files and shows how to run them.

## Data

All data files are in `classical_SVR/`:

| File | Rows | Role |
|---|---|---|
| `hybrid_train_102926.csv` | 15,546 | Training set. Labels include the conformer-multiplicity and rotational-symmetry corrections. |
| `hybrid_std128_102926.csv` | 128 | External test set (experimental) |
| `hybrid_std109_102926.csv` | 109 | External test set (experimental) |
| `hybrid_pdbbind_102926.csv` | 311 | External test set (DFT-labelled PDBbind ligands) |
| `hybrid_std6074_102926.csv` | 6,074 | External test set (experimental) |

Each CSV has `ID`, `SMILES`, `entropy` (the target) and 289 descriptor columns. In `hybrid_std128_102926.csv`, the labels of benzyl alcohol and benzaldehyde were corrected on 2026-11-02.

`selected_features` (in `D-MPNN_SVR/`, `GINE_SVR/` and `Uni-Mol_SVR/`, all identical) lists the 256 descriptors used by the "bond features" models.

To give the other model folders the `hybrid_data/` folder their commands expect, run this once from the repository root:

```bash
for d in D-MPNN_SVR GINE_SVR Uni-Mol_SVR unimol_residual; do
  mkdir -p $d/hybrid_data && cp classical_SVR/hybrid_*_102926.csv $d/hybrid_data/
done
cp D-MPNN_SVR/selected_features unimol_residual/
```

## Trained checkpoints

Files over 100 MB cannot be stored in a GitHub repository. Download them from the release attached to this repository and place them at these paths:

| Release file | Place at |
|---|---|
| `unimol_residual_bond_model_0.pth` | `unimol_residual/svr_general_residual_run_bond_conformer_symmetry/unimol/model_0.pth` |
| `unimol_residual_all_model_0.pth` | `unimol_residual/svr_general_residual_run_all_conformer_symmetry/unimol/model_0.pth` |
| `unimol_residual_std6074_model_0.pth` | `unimol_residual/svr_residual_allfeat_std6074_102926_80_10_10/unimol/model_0.pth` |
| `unimol_svr_no_oversample_model_0.pth` | `Uni-Mol_SVR/unimol_ft_entropy_no_oversample_102926_80_10_10_selfeat_svr/unimol/model_0.pth` |
| `unimol_svr_os3seg_model_0.pth` | `Uni-Mol_SVR/unimol_ft_entropy_os3seg_102926_80_10_10_selfeat_svr/unimol/model_0.pth` |

The D-MPNN models (`best.pt`, about 50 MB each) and the GINE checkpoints are in the repository. The PyTorch Lightning `checkpoints/*.ckpt` files from D-MPNN training are not included; prediction uses `best.pt` only.

## Before running

- Run each script from inside its own folder. The scripts use relative paths.
- Several scripts default to a GPU ID from the machine they were written on. Pass `--gpu 0` (or `--gpu cpu`) as described in each folder README.
- Some saved config and log files contain paths from the original machine. They are a record of each run and are not needed to use the models.
