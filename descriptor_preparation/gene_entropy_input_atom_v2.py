#!/usr/bin/python3
"""Build the atom-type descriptor table (one row per ligand).

Columns: ID, ring, heavy, s_heavy_bond, MW, <one count per SYBYL atom type>,
symmetry, entropy.
"""

import argparse
from collections import Counter
from pathlib import Path

from mol2_utils import read_entropy_table, read_mol2, read_type_library


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--atom-types", default="unique_atom_type")
    parser.add_argument("--entropy-table", default="corrected_entropy_mw300")
    parser.add_argument("--mol2-dir", default="mw300_mol2")
    parser.add_argument("--output", default="train_atom_v2_mw300.csv")
    return parser.parse_args()


def main():
    args = parse_args()
    atom_types = read_type_library(args.atom_types)

    n_written = 0
    with open(args.output, "w") as out:
        out.write("ID,ring,heavy,s_heavy_bond,MW,")
        out.write("".join(f"{atom_type}," for atom_type in atom_types))
        out.write("symmetry,entropy\n")

        for ligand, symmetry, entropy in read_entropy_table(args.entropy_table):
            mol2_path = Path(args.mol2_dir) / f"{ligand}.mol2"
            if not mol2_path.exists():
                continue
            mol = read_mol2(mol2_path)

            type_counts = Counter(atom.atom_type for atom in mol.atoms)
            unknown = set(type_counts) - set(atom_types)
            if unknown:
                print(f"{ligand}: atom types not in library: {sorted(unknown)}")

            out.write(f"{ligand}.mol2,{mol.ring_count},{mol.heavy_atom_count},")
            out.write(f"{mol.single_heavy_bond_count},{mol.molecular_weight},")
            out.write("".join(f"{type_counts[atom_type]}," for atom_type in atom_types))
            out.write(f"{symmetry},{entropy}\n")
            n_written += 1

    print(f"total {n_written} mols")


if __name__ == "__main__":
    main()
