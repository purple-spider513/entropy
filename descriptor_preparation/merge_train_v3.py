#!/usr/bin/env python3
"""
The final feature table used in this project for Graph-based training.
Merge atom, element, and bond entropy training tables on molecule ID.
Element, bond, and SMILES rows are indexed once so each atom row is a dictionary lookup.
SMILES is written as the last column.

"""

atom_path = "train_atom_mw300.csv"
element_path = "train_element_mw300.csv"
bond_path = "train_bond_mw300.csv"
smi_path = "mw300.smi"
out_path = "all_feature_mw300.csv"

# First match wins, matching the old nested-loop break.
element_by_id = {}
with open(element_path, "r") as fo2:
    for line2 in fo2:
        ln2 = line2.strip().split(",")
        mol_id = ln2[0]
        if mol_id not in element_by_id:
            # C, N, O, S counts
            element_by_id[mol_id] = ",".join((ln2[7], ln2[13], ln2[14], ln2[16]))

bond_by_id = {}
with open(bond_path, "r") as fo3:
    for line3 in fo3:
        ln3 = line3.split(",")
        mol_id = ln3[0]
        if mol_id not in bond_by_id:
            tail = ln3[6:]
            if tail:
                tail[-1] = tail[-1].rstrip("\r\n")
            bond_by_id[mol_id] = ",".join(tail)

# Ligand ID in the .smi file is the molecule ID without the .mol2 suffix.
# Fields are tab-separated. Some SMILES include a CXSMILES suffix after a space
# (" |&1:11|"), so splitting on whitespace would shift the ligand ID.
smiles_by_id = {}
with open(smi_path, "r") as fos:
    for line_smi in fos:
        fields = line_smi.rstrip("\r\n").split("\t")
        if len(fields) < 2:
            continue
        lig_id = fields[1]
        if lig_id not in smiles_by_id:
            smiles = fields[0]
            if any(ch in smiles for ch in ',"\n'):
                smiles = '"' + smiles.replace('"', '""') + '"'
            smiles_by_id[lig_id] = smiles

n = 0
written = 0
missing_smiles = 0
with open(atom_path, "r") as fo1, open(out_path, "w") as out:
    for line in fo1:
        n += 1
        ln1 = line.split(",")[:-2]
        mol_id = ln1[0]
        if n % 100 == 0:
            print(f"I have processed {n} ...")
        bond = bond_by_id.get(mol_id)
        if bond is None:
            continue
        if mol_id == "ID":
            smiles = "SMILES"
        else:
            lig_id = mol_id[:-5] if mol_id.endswith(".mol2") else mol_id
            smiles = smiles_by_id.get(lig_id, "")
            if smiles == "":
                missing_smiles += 1
        atom = ",".join(ln1)
        element = element_by_id.get(mol_id, "")
        out.write(f"{atom},{element},{bond},{smiles}\n")
        written += 1
        if n>10000:
            break

print(f"Done. Read {n} atom rows, wrote {written} merged rows, missing SMILES: {missing_smiles}.")
