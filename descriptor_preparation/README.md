# Descriptor preparation

Builds the count-based descriptors used by the SVR and hybrid models from mol2 structure files and corrected entropies.

## Files

| File | What it does |
|---|---|
| `mw300_mol2.zip` | The 854 ligand structures in mol2 format. Extract it first (see below). |
| `corrected_entropy_mw300` | Ligand ID, symmetry correction and corrected entropy, whitespace-separated. |
| `unique_atom_type` | The 30 SYBYL atom types used as columns. |
| `unique_bond_type` | The 250 bond types used as columns. |
| `mol2_utils.py` | Shared mol2 reader used by the scripts below. |
| `gene_entropy_input_atom_v2.py` | Counts atom types per ligand. Writes `train_atom_v2_mw300.csv`. |
| `gene_entropy_input_bond_v2.py` | Counts bond types per ligand. Writes `train_bond_v2_mw300.csv`. |
| `gene_entropy_input_elements_v3.py` | Sums atom types into element totals. Writes `train_element_v2_mw300.csv`. |
| `merge_train_v3.py` | Joins the three tables into one training table, `all_train_v2_mw300.csv`. |
| `count_rotational_conformations_v3.py` | Optional. Counts rotational conformers with RDKit as a product of rotor factors, with no per-molecule overrides. Not used by the four scripts above. |
| `mw300_conformer` | Conformer count per ligand, the output of the conformer script. |
| `failed_info` | Log of molecules the conformer script could not process. Only a header row here, so nothing failed. |
| `train_*_mw300.csv`, `all_train_v2_mw300.csv` | Outputs of the pipeline, 854 rows each. |

## Usage

```bash
cd descriptor_preparation
unzip mw300_mol2.zip
python gene_entropy_input_atom_v2.py
python gene_entropy_input_bond_v2.py
python gene_entropy_input_elements_v3.py
python merge_train_v3.py
```

The scripts read `mw300_mol2/` from the current folder, so extract the zip first. The four CSV scripts need only the Python standard library. Running them reproduces the bundled CSVs exactly.

To regenerate `mw300_conformer` (needs RDKit):

```bash
python count_rotational_conformations_v3.py mw300_mol2 --workers 10 > mw300_conformer
```
