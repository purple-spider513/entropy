#!/usr/bin/python3
"""Collapse the atom-type table into per-element counts.

Reads the output of gene_entropy_input_atom_v2.py and sums the SYBYL atom types
of each element (C.1 + C.2 + ... -> C, N.*, O.*, S.*). Boron support was added
in v3.
"""

import argparse
import csv
from collections import defaultdict

# Elements in the output order.
ELEMENTS = ["B", "Br", "C", "Cl", "F", "Fe", "H", "I", "N", "O", "P", "S", "Si"]
LEADING_COLUMNS = ["ID", "ring", "heavy", "s_heavy_bond", "MW"]
TRAILING_COLUMNS = ["symmetry", "entropy"]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--atom-table", default="train_atom_v2_mw300.csv")
    parser.add_argument("--output", default="train_element_v2_mw300.csv")
    return parser.parse_args()


def element_of(atom_type):
    return atom_type.split(".")[0]


def main():
    args = parse_args()

    with open(args.atom_table, newline="") as source, open(args.output, "w") as out:
        reader = csv.DictReader(source)
        type_columns = [
            name for name in reader.fieldnames
            if name not in LEADING_COLUMNS + TRAILING_COLUMNS
        ]
        out.write(",".join(LEADING_COLUMNS + ELEMENTS + TRAILING_COLUMNS) + "\n")

        for row in reader:
            counts = defaultdict(int)
            for atom_type in type_columns:
                counts[element_of(atom_type)] += int(row[atom_type])

            entropy = round(float(row["entropy"]), 3)
            leading = [row[name] for name in LEADING_COLUMNS]
            element_counts = [str(counts[element]) for element in ELEMENTS]
            out.write(",".join(leading + element_counts + [row["symmetry"], str(entropy)]) + "\n")


if __name__ == "__main__":
    main()
