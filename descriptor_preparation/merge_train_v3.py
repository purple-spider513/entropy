#!/usr/bin/python3
"""Merge the atom-type, element and bond-type tables into one training set.

Output columns: the atom-type table (without symmetry/entropy), the C/N/O/S
element totals, then the bond-type table (without its ID/ring/heavy/H/
s_heavy_bond/MW columns) ending in symmetry and entropy.
"""

import argparse
import csv

ELEMENT_TOTALS = ["C", "N", "O", "S"]
BOND_TABLE_LEADING_COLUMNS = 6  # ID, ring, heavy, H, s_heavy_bond, MW


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--atom-table", default="train_atom_v2_mw300.csv")
    parser.add_argument("--element-table", default="train_element_v2_mw300.csv")
    parser.add_argument("--bond-table", default="train_bond_v2_mw300.csv")
    parser.add_argument("--output", default="all_train_v2_mw300.csv")
    return parser.parse_args()


def read_rows(path):
    with open(path, newline="") as handle:
        return list(csv.reader(handle))


def main():
    args = parse_args()
    atom_rows = read_rows(args.atom_table)
    element_rows = read_rows(args.element_table)
    bond_rows = read_rows(args.bond_table)

    element_header = element_rows[0]
    element_columns = [element_header.index(name) for name in ELEMENT_TOTALS]
    elements_by_id = {row[0]: row for row in element_rows}
    bonds_by_id = {row[0]: row for row in bond_rows}

    n_merged = 0
    with open(args.output, "w", newline="") as out:
        writer = csv.writer(out, lineterminator="\n")
        for atom_row in atom_rows:
            ligand = atom_row[0]
            if ligand not in elements_by_id or ligand not in bonds_by_id:
                print(f"skipping {ligand}: missing from element or bond table")
                continue

            atom_part = atom_row[:-2]  # drop symmetry and entropy
            element_part = [elements_by_id[ligand][i] for i in element_columns]
            bond_part = bonds_by_id[ligand][BOND_TABLE_LEADING_COLUMNS:]
            writer.writerow(atom_part + element_part + bond_part)

            n_merged += 1
            if n_merged % 100 == 0:
                print(f"I have processed {n_merged} ...")


if __name__ == "__main__":
    main()
