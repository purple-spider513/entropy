# Descriptor preparation

Builds the count descriptors and corrected entropy labels used by the models in this repository, starting from mol2 structures and DFT entropies. The example data are 854 ligands with molecular weight up to 300.

The corrected entropy is the DFT entropy plus a conformer-multiplicity and rotational-symmetry correction (J⋅mol⁻¹⋅K⁻¹).

## Files

### Inputs

| File | Content |
|---|---|
| `mw300_mol2/` | 854 ligand structures in mol2 format, one file per ligand (`<ID>.mol2`). |
| `mw300_DFT` | Ligand ID and DFT entropy, whitespace-separated, no header. |
| `mw300.smi` | SMILES and ligand ID, tab-separated. |
| `unique_atom_type` | The 30 SYBYL atom types used as columns. |
| `unique_bond_type` | The bond types (pairs of SYBYL atom types) used as columns. |

### Scripts

| Script | What it does |
|---|---|
| `count_rotational_conformations_and_symmetry_v2.py` | Counts conformers and the rotational symmetry number of each mol2 structure, and writes the matching entropy corrections. Writes `mw300_conformers-symmetry.csv` and a `failed_info_symmetry_*.csv` log. Needs RDKit, NumPy and SciPy. |
| `gene_entropy_input_atom_v2.py` | Counts atom types per ligand, plus ring count, heavy atoms, single heavy-atom bonds and molecular weight. Writes `train_atom_mw300.csv`. |
| `gene_entropy_input_bond_v2.py` | Counts bond types per ligand, plus the same general descriptors and the hydrogen count. Writes `train_bond_mw300.csv`. |
| `gene_entropy_input_elements_v3.py` | Sums atom types into element totals. Reads `train_atom_mw300.csv`, writes `train_element_mw300.csv`. |
| `merge_train_v3.py` | Joins the atom, element and bond tables on ID and adds SMILES. Writes `all_feature_mw300.csv`. |

### Outputs (included)

| File | Content |
|---|---|
| `mw300_conformers-symmetry.csv` | Per ligand: conformer count, symmetry number, the two entropy corrections and their sum (`net_ent`). Ligands that could not be processed have status `error`. |
| `failed_info_symmetry_20261104T013043Z_dsdoo9g7.csv` | The 3 ligands the conformer script could not process (RDKit could not kekulize them). |
| `train_atom_mw300.csv`, `train_bond_mw300.csv`, `train_element_mw300.csv` | Descriptor tables, 851 ligands each. The last two columns are `symmetry` (the total correction) and `entropy` (DFT entropy + correction). |
| `all_feature_mw300.csv` | Merged table: ID, all descriptors, `symmetry`, `entropy` and `SMILES`. |

## Usage

Run from this folder, in this order:

```bash
cd descriptor_preparation

# 1. Conformer and symmetry corrections (needs RDKit, NumPy, SciPy)
python count_rotational_conformations_and_symmetry_v2.py mw300_mol2 --workers 10 > mw300_conformers-symmetry.csv

# 2. Descriptor tables (Python standard library only)
python gene_entropy_input_atom_v2.py
python gene_entropy_input_bond_v2.py
python gene_entropy_input_elements_v3.py
python merge_train_v3.py
```

- Step 1 writes a new `failed_info_symmetry_<timestamp>.csv` each run and never overwrites an old one. Run `python count_rotational_conformations_and_symmetry_v2.py --help` for the symmetry options (`--symmetry-mode`, `--symmetry-policy`, tolerances).
- Step 2 can be run on their own with the included `mw300_conformers-symmetry.csv`. They skip ligands marked `error`, which leaves 851 of the 854.
- Running step 2 reproduces the included `train_*_mw300.csv` and `all_feature_mw300.csv` files exactly (checked 2026-11-03).
- The input and output file names are set at the top of each script. Edit them there to run on another data set.
