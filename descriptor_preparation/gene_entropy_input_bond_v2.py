#!/usr/bin/python3
"""Build the bond-type descriptor table (one row per ligand).

Columns: ID, ring, heavy, H, s_heavy_bond, MW, <one count per bond type>,
symmetry, entropy. A bond type is the two SYBYL atom types joined by "-" in
sorted order, e.g. "C.ar-N.am".
"""

import argparse
from pathlib import Path

from mol2_utils import read_entropy_table, read_mol2, read_type_library


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--bond-types", default="unique_bond_type")
    parser.add_argument("--entropy-table", default="corrected_entropy_mw300")
    parser.add_argument("--mol2-dir", default="mw300_mol2")
    parser.add_argument("--output", default="train_bond_v2_mw300.csv")
    return parser.parse_args()


def bond_type_counts(mol, bond_types):
    counts = dict.fromkeys(bond_types, 0)
    types = mol.type_by_atom_id
    for bond in mol.bonds:
        first, second = sorted((types[bond.atom_1], types[bond.atom_2]))
        counts[f"{first}-{second}"] += 1
    return counts


def main():
    args = parse_args()
    bond_types = read_type_library(args.bond_types)

    n_written = 0
    with open(args.output, "w") as out:
        out.write("ID,ring,heavy,H,s_heavy_bond,MW,")
        out.write("".join(f"{bond_type}," for bond_type in bond_types))
        out.write("symmetry,entropy\n")

        for ligand, symmetry, entropy in read_entropy_table(args.entropy_table):
            mol2_path = Path(args.mol2_dir) / f"{ligand}.mol2"
            if not mol2_path.exists():
                print(f"missing mol2 for {ligand}")
                continue
            mol = read_mol2(mol2_path)
            counts = bond_type_counts(mol, bond_types)

            out.write(f"{ligand}.mol2,{mol.ring_count},{mol.heavy_atom_count},")
            out.write(f"{mol.hydrogen_count},{mol.single_heavy_bond_count},{mol.molecular_weight},")
            out.write("".join(f"{counts[bond_type]}," for bond_type in bond_types))
            out.write(f"{symmetry},{entropy}\n")
            n_written += 1

    print(f"total {n_written} mols")


if __name__ == "__main__":
    main()
