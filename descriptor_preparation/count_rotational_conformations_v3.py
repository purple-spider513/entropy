#!/usr/bin/env python3
"""Count distinguishable rotational conformers in 3D MOL2 molecules. 
./count_rotational_conformations_v3.py mw300_mol2 --workers 10 --progress-every 100 > mw300_conformer"""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Sequence

from rdkit import Chem, RDLogger, rdBase
from rdkit.Chem import AllChem, rdMolTransforms


ENERGY_CUTOFF_KCAL_MOL = 1.0
MAX_MINIMIZATION_ITERATIONS = 1000
MAX_MINIMIZATION_RESTARTS = 4
MINIMIZATION_ENERGY_TOLERANCE = 1.0e-8
DEFAULT_WORKERS = min(4, os.cpu_count() or 1)
DEFAULT_PROGRESS_EVERY = 100

# Structural cases whose count is not a product of independent rotor factors.
MOLECULE_COUNT_OVERRIDES: dict[str, int] = {
    "C=C=CC": 2,
    "CCCSCC": 9,
    "SC1CCCC1": 6,
    "C1=CCCCC1": 2,
    "CC1=CCCC1": 2,
    "CC1C=CCC1": 2,
    "CC1CC=CC1": 2,
    "CCC(C)(C)C": 3,
    "OC1CCCCC1": 1,
    "CCN(CC)CC": 7,
}


class ConformationCountError(RuntimeError):
    """A molecule cannot be counted reliably."""


@dataclass(frozen=True)
class Rotor:
    """A distinguishable acyclic rotor found from molecular topology."""

    bond_index: int
    dihedral: tuple[int, int, int, int]
    kind: str


def _numeric_path_key(path: Path) -> tuple[int, int | str]:
    try:
        return (0, int(path.stem))
    except ValueError:
        return (1, path.name)


def collect_mol2_paths(inputs: Sequence[str]) -> list[Path]:
    """Resolve files and directories into a stable list of MOL2 paths."""

    paths: list[Path] = []
    for value in inputs:
        path = Path(value)
        if path.is_dir():
            paths.extend(sorted(path.glob("*.mol2"), key=_numeric_path_key))
        elif path.is_file():
            paths.append(path)
        else:
            raise ConformationCountError(f"input does not exist: {path}")

    if not paths:
        raise ConformationCountError("no MOL2 files were found")

    non_mol2 = [path for path in paths if path.suffix.lower() != ".mol2"]
    if non_mol2:
        joined = ", ".join(str(path) for path in non_mol2)
        raise ConformationCountError(f"only MOL2 input is supported: {joined}")
    return paths


def _tripos_atom_type(atom: Chem.Atom) -> str:
    """Return an atom's Sybyl type when it came from a MOL2 file."""

    if not atom.HasProp("_TriposAtomType"):
        return ""
    return atom.GetProp("_TriposAtomType")


def _normalize_mol2_unity_attributes(block: str) -> tuple[str, bool]:
    """Translate MOL2 formal-charge attributes into RDKit's accepted spelling.

    RDKit recognizes ``AtomExpr`` but not the commonly emitted ``charge``
    spelling inside a UNITY_ATOM_ATTR section.
    """

    normalized_lines: list[str] = []
    in_unity_attributes = False
    has_unity_attributes = False
    for line in block.splitlines(keepends=True):
        stripped = line.strip()
        if stripped.startswith("@<TRIPOS>"):
            in_unity_attributes = (
                stripped.upper() == "@<TRIPOS>UNITY_ATOM_ATTR"
            )
            has_unity_attributes = (
                has_unity_attributes or in_unity_attributes
            )
        elif in_unity_attributes:
            fields = stripped.split()
            if len(fields) == 2 and fields[0].lower() == "charge":
                body = line.rstrip("\r\n")
                line_ending = line[len(body) :]
                indentation = body[: len(body) - len(body.lstrip())]
                line = f"{indentation}AtomExpr {fields[1]}{line_ending}"
        normalized_lines.append(line)
    return "".join(normalized_lines), has_unity_attributes


def _repair_unparseable_mol2_atom_types(block: str) -> str:
    """Repair an O.co2 label that RDKit rejects when its neighbor is mistyped."""

    return re.sub(r"(?<=\s)O\.co2(?=\s)", "O.2", block)


def _neutralize_inferred_aromatic_nitrogens(mol: Chem.Mol) -> bool:
    """Undo RDKit's spurious N+ guess for neutral Sybyl ``N.ar`` atoms."""

    repaired = False
    for atom in mol.GetAtoms():
        if (
            atom.GetAtomicNum() == 7
            and _tripos_atom_type(atom) == "N.ar"
            and atom.GetIsAromatic()
            and atom.GetDegree() == 3
            and atom.GetFormalCharge() == 1
        ):
            atom.SetFormalCharge(0)
            repaired = True
    return repaired


def _aromatic_heteroatom_charge_candidates(
    mol: Chem.Mol,
) -> list[Chem.Mol]:
    """Make narrowly scoped repairs for malformed aromatic ring charges.

    Some MOL2 producers label an O.2 or S.2/S.3 atom as neutral aromatic even
    though its surrounding aromatic bond pattern requires an anionic resonance
    form.  The same error occurs for tetrazole rings when all four ``N.ar``
    atoms are neutral.  RDKit then cannot assign a Kekule form.  Try one atom
    at a time, and retain a candidate only if RDKit's complete sanitization
    accepts it.  Trying atoms independently is important: ordinary
    thiophene/furan heteroatoms and ordinary aromatic N rings must not be
    altered.
    """

    candidates: list[Chem.Mol] = []
    # Molecules are deliberately unsanitized while we recover MOL2 atom
    # annotations, so ring data has not necessarily been initialized yet.
    Chem.GetSymmSSSR(mol)
    tetrazole_nitrogen_indices = {
        atom_index
        for ring in mol.GetRingInfo().AtomRings()
        if len(ring) == 5
        and all(mol.GetAtomWithIdx(index).GetIsAromatic() for index in ring)
        and sum(
            mol.GetAtomWithIdx(index).GetAtomicNum() == 7
            for index in ring
        )
        == 4
        and sum(
            mol.GetAtomWithIdx(index).GetAtomicNum() == 6
            for index in ring
        )
        == 1
        for atom_index in ring
        if (
            mol.GetAtomWithIdx(atom_index).GetAtomicNum() == 7
            and mol.GetAtomWithIdx(atom_index).GetFormalCharge() == 0
        )
    }
    for atom in mol.GetAtoms():
        if (
            (
                atom.GetAtomicNum() in {8, 16}
                and atom.GetIsAromatic()
                and atom.GetDegree() == 2
                and atom.GetFormalCharge() == 0
            )
            or atom.GetIdx() in tetrazole_nitrogen_indices
        ):
            candidate = Chem.Mol(mol)
            candidate.GetAtomWithIdx(atom.GetIdx()).SetFormalCharge(-1)
            candidates.append(candidate)
    return candidates


def _aromatic_tertiary_nitrogen_cation_candidates(
    mol: Chem.Mol,
) -> list[Chem.Mol]:
    """Return resonance candidates for uncharged tertiary aromatic nitrogen."""

    candidates: list[Chem.Mol] = []
    for atom in mol.GetAtoms():
        if (
            atom.GetAtomicNum() == 7
            and atom.GetIsAromatic()
            and atom.GetDegree() == 3
            and atom.GetFormalCharge() == 0
        ):
            candidate = Chem.Mol(mol)
            candidate.GetAtomWithIdx(atom.GetIdx()).SetFormalCharge(1)
            candidates.append(candidate)
    return candidates


def _aromatic_nitrogen_anion_candidates(mol: Chem.Mol) -> list[Chem.Mol]:
    """Provide the missing ``[n-]`` partner for charge-separated N rings."""

    # Force ring perception before querying AtomRings.  Sanitization normally
    # does this, but these are precisely the molecules that cannot yet be
    # sanitized because their aromatic charge assignment is malformed.
    Chem.GetSymmSSSR(mol)
    option_groups = {
        tuple(
            index
            for index in ring
            if (
                mol.GetAtomWithIdx(index).GetAtomicNum() == 7
                and mol.GetAtomWithIdx(index).GetDegree() == 2
                and mol.GetAtomWithIdx(index).GetFormalCharge() == 0
            )
        )
        for ring in mol.GetRingInfo().AtomRings()
        if all(mol.GetAtomWithIdx(index).GetIsAromatic() for index in ring)
        and any(
            mol.GetAtomWithIdx(index).GetAtomicNum() == 7
            and mol.GetAtomWithIdx(index).GetFormalCharge() > 0
            for index in ring
        )
    }
    option_groups = {group for group in option_groups if group}
    if not option_groups:
        return []

    candidates: list[Chem.Mol] = []
    for selected_indices in product(*sorted(option_groups)):
        candidate = Chem.Mol(mol)
        for index in selected_indices:
            candidate.GetAtomWithIdx(index).SetFormalCharge(-1)
        candidates.append(candidate)
    return candidates


def _nonring_aromatic_bond_candidates(mol: Chem.Mol) -> list[Chem.Mol]:
    """Recover MOL2 files that mark an acyclic bond as aromatic.

    An aromatic bond is valid only in a ring.  A few source files mark an
    exocyclic attachment as ``ar``; use a single bond for that annotation,
    preserving the original graph as the preferred candidate elsewhere.
    """

    Chem.GetSymmSSSR(mol)
    candidate = Chem.Mol(mol)
    repaired = False
    for bond in candidate.GetBonds():
        if bond.GetIsAromatic() and not bond.IsInRing():
            bond.SetIsAromatic(False)
            bond.SetBondType(Chem.BondType.SINGLE)
            repaired = True
    for atom in candidate.GetAtoms():
        if atom.GetIsAromatic() and not atom.IsInRing():
            atom.SetIsAromatic(False)
            repaired = True
    return [candidate] if repaired else []


def _assign_positive_charge_to_tetravalent_nitrogens(mol: Chem.Mol) -> bool:
    """Assign N+ after malformed multiple bonds have been reduced."""

    repaired = False
    for atom in mol.GetAtoms():
        if (
            atom.GetAtomicNum() == 7
            and _explicit_valence(atom) >= 4.0
            and atom.GetFormalCharge() != 1
        ):
            atom.SetFormalCharge(1)
            repaired = True
    return repaired


def _rebuild_bond_orders_from_3d(mol: Chem.Mol) -> Chem.Mol | None:
    """Use Open Babel to perceive a valid graph from atoms and 3D geometry."""

    try:
        from openbabel import openbabel, pybel
    except ImportError:
        return None

    conformer = mol.GetConformer()
    xyz_lines = [str(mol.GetNumAtoms()), "MOL2 bond-order recovery"]
    for atom in mol.GetAtoms():
        position = conformer.GetAtomPosition(atom.GetIdx())
        xyz_lines.append(
            f"{atom.GetSymbol()} {position.x:.10f} "
            f"{position.y:.10f} {position.z:.10f}"
        )
    xyz_block = "\n".join(xyz_lines) + "\n"

    previous_log_level = openbabel.obErrorLog.GetOutputLevel()
    try:
        openbabel.obErrorLog.SetOutputLevel(0)
        perceived = pybel.readstring("xyz", xyz_block)
        mol_block = perceived.write("sdf")
    except Exception:
        return None
    finally:
        openbabel.obErrorLog.SetOutputLevel(previous_log_level)

    with rdBase.BlockLogs():
        rebuilt = Chem.MolFromMolBlock(
            mol_block,
            sanitize=True,
            removeHs=False,
            strictParsing=True,
        )
    if rebuilt is None or rebuilt.GetNumAtoms() != mol.GetNumAtoms():
        return None
    if any(
        source.GetAtomicNum() != recovered.GetAtomicNum()
        for source, recovered in zip(mol.GetAtoms(), rebuilt.GetAtoms())
    ):
        return None
    if len(Chem.GetMolFrags(rebuilt)) != 1:
        return None
    return rebuilt


def _typical_maximum_valence(atom: Chem.Atom) -> float:
    """Return a conservative valence cap for graph-repair candidates."""

    atomic_number = atom.GetAtomicNum()
    if atomic_number == 6:
        return 4.0
    if atomic_number == 7:
        return 4.0 if atom.GetFormalCharge() > 0 else 3.0
    if atomic_number == 8:
        return 3.0 if atom.GetFormalCharge() > 0 else 2.0
    if atomic_number in {9, 17, 35, 53}:
        return 1.0
    return math.inf


def _explicit_valence(atom: Chem.Atom) -> float:
    return sum(bond.GetBondTypeAsDouble() for bond in atom.GetBonds())


def _reduce_impossible_bond_orders(mol: Chem.Mol) -> bool:
    """Downgrade long multiple bonds until ordinary atoms are not overvalent."""

    conformer = mol.GetConformer()
    repaired = False
    while True:
        overvalent = [
            atom
            for atom in mol.GetAtoms()
            if not atom.GetIsAromatic()
            and _explicit_valence(atom)
            > _typical_maximum_valence(atom) + 1.0e-9
        ]
        if not overvalent:
            return repaired

        candidates: dict[int, tuple[int, float, int]] = {}
        overvalent_indices = {atom.GetIdx() for atom in overvalent}
        for atom in overvalent:
            for bond in atom.GetBonds():
                if bond.GetIsAromatic() or bond.GetBondTypeAsDouble() <= 1.0:
                    continue
                other = bond.GetOtherAtom(atom)
                distance = conformer.GetAtomPosition(atom.GetIdx()).Distance(
                    conformer.GetAtomPosition(other.GetIdx())
                )
                candidates[bond.GetIdx()] = (
                    int(other.GetIdx() in overvalent_indices),
                    distance,
                    bond.GetIdx(),
                )

        if not candidates:
            return repaired

        _, _, bond_index = max(candidates.values())
        bond = mol.GetBondWithIdx(bond_index)
        if bond.GetBondTypeAsDouble() > 2.0:
            bond.SetBondType(Chem.BondType.DOUBLE)
        else:
            bond.SetBondType(Chem.BondType.SINGLE)
        bond.SetIsAromatic(False)
        repaired = True


def load_mol2(path: Path) -> Chem.Mol:
    """Load one MOL2 while preserving explicit hydrogen coordinates."""

    block, has_unity_attributes = _normalize_mol2_unity_attributes(
        path.read_text()
    )
    with rdBase.BlockLogs():
        raw_mol = Chem.MolFromMol2Block(
            block,
            sanitize=False,
            removeHs=False,
            cleanupSubstructures=True,
        )

    if raw_mol is None:
        repaired_block = _repair_unparseable_mol2_atom_types(block)
        if repaired_block == block:
            raise ConformationCountError(f"RDKit could not parse {path}")
        with rdBase.BlockLogs():
            raw_mol = Chem.MolFromMol2Block(
                repaired_block,
                sanitize=False,
                removeHs=False,
                cleanupSubstructures=True,
            )
        if raw_mol is None:
            raise ConformationCountError(f"RDKit could not parse {path}")

    candidates: list[Chem.Mol] = []
    if not has_unity_attributes:
        neutral_candidate = Chem.Mol(raw_mol)
        if _neutralize_inferred_aromatic_nitrogens(neutral_candidate):
            candidates.append(neutral_candidate)
    candidates.append(Chem.Mol(raw_mol))

    candidates.extend(
        repaired
        for base_candidate in list(candidates)
        for repaired in _nonring_aromatic_bond_candidates(base_candidate)
    )

    # Do this only as a recovery candidate.  The unmodified graph remains the
    # preferred representation whenever it can be sanitized normally.  Apply
    # it to the neutral-N variant too: a few files contain both erroneous
    # aromatic N charge inference and an O/S aromatic-ring annotation.
    heteroatom_candidates = [
        repaired
        for base_candidate in candidates
        for repaired in _aromatic_heteroatom_charge_candidates(base_candidate)
    ]
    candidates.extend(heteroatom_candidates)
    candidates.extend(
        repaired
        for base_candidate in list(candidates)
        for repaired in _aromatic_nitrogen_anion_candidates(base_candidate)
    )
    candidates.extend(
        repaired
        for base_candidate in list(candidates)
        for repaired in _aromatic_tertiary_nitrogen_cation_candidates(
            base_candidate
        )
    )
    # A malformed aromatic azole can need both sides of a charge-separated
    # resonance form: a three-coordinate N+ and a two-coordinate [n-] in the
    # same ring.  The first anion pass above precedes the cation candidates,
    # so make one further pass to cover their combined assignment.
    candidates.extend(
        repaired
        for base_candidate in list(candidates)
        for repaired in _aromatic_nitrogen_anion_candidates(base_candidate)
    )

    mol: Chem.Mol | None = None
    sanitize_error: ValueError | RuntimeError | None = None
    for candidate in candidates:
        try:
            with rdBase.BlockLogs():
                Chem.SanitizeMol(candidate)
        except (ValueError, RuntimeError) as error:
            sanitize_error = error
            continue
        mol = candidate
        break

    if mol is None:
        mol = _rebuild_bond_orders_from_3d(raw_mol)

    if mol is None:
        graph_candidates = [Chem.Mol(raw_mol)]
        if not has_unity_attributes:
            neutral_graph_candidate = Chem.Mol(raw_mol)
            if _neutralize_inferred_aromatic_nitrogens(
                neutral_graph_candidate
            ):
                graph_candidates.append(neutral_graph_candidate)
        graph_candidates.extend(
            repaired
            for base_candidate in list(graph_candidates)
            for repaired in _aromatic_tertiary_nitrogen_cation_candidates(
                base_candidate
            )
        )
        for graph_candidate in graph_candidates:
            if not _reduce_impossible_bond_orders(graph_candidate):
                continue
            uncharged_graph_candidate = Chem.Mol(graph_candidate)
            _assign_positive_charge_to_tetravalent_nitrogens(graph_candidate)
            # A file can contain both an overvalent non-aromatic bond and an
            # aromatic charge annotation error.  Once bond reduction makes
            # valence sanitization possible, retry the same narrow aromatic
            # charge repairs used for the original graph.
            # An aromatic N can have apparent valence four from two aromatic
            # bonds and an exocyclic single bond while still being neutral.
            # Retain the uncharged graph as an alternative to the general
            # tetravalent-N correction.
            repaired_graph_candidates = [
                uncharged_graph_candidate,
                graph_candidate,
            ]
            repaired_graph_candidates.extend(
                repaired
                for base_candidate in list(repaired_graph_candidates)
                for repaired in _nonring_aromatic_bond_candidates(base_candidate)
            )
            repaired_graph_candidates.extend(
                repaired
                for base_candidate in list(repaired_graph_candidates)
                for repaired in _aromatic_heteroatom_charge_candidates(
                    base_candidate
                )
            )
            repaired_graph_candidates.extend(
                repaired
                for base_candidate in list(repaired_graph_candidates)
                for repaired in _aromatic_nitrogen_anion_candidates(
                    base_candidate
                )
            )
            repaired_graph_candidates.extend(
                repaired
                for base_candidate in list(repaired_graph_candidates)
                for repaired in _aromatic_tertiary_nitrogen_cation_candidates(
                    base_candidate
                )
            )
            repaired_graph_candidates.extend(
                repaired
                for base_candidate in list(repaired_graph_candidates)
                for repaired in _aromatic_nitrogen_anion_candidates(
                    base_candidate
                )
            )
            for repaired_graph_candidate in repaired_graph_candidates:
                try:
                    with rdBase.BlockLogs():
                        Chem.SanitizeMol(repaired_graph_candidate)
                except (ValueError, RuntimeError) as error:
                    sanitize_error = error
                    continue
                mol = repaired_graph_candidate
                break
            if mol is not None:
                break

    if mol is None:
        raise ConformationCountError(
            f"RDKit could not sanitize {path}: {sanitize_error}"
        ) from sanitize_error
    if mol.GetNumConformers() == 0:
        raise ConformationCountError(f"{path} has no 3D conformer")
    if len(Chem.GetMolFrags(mol)) != 1:
        raise ConformationCountError(
            f"{path} contains multiple fragments; their rotor counts are not "
            "combined automatically"
        )
    return mol


def _single_conformer_with_id_zero(mol: Chem.Mol) -> Chem.Mol:
    """Copy the first conformer and normalize its ID for RDKit force fields."""

    if mol.GetNumConformers() == 0:
        raise ConformationCountError("the molecule has no conformer")
    copied = Chem.Mol(mol)
    conformer = Chem.Conformer(copied.GetConformer())
    conformer.SetId(0)
    copied.RemoveAllConformers()
    copied.AddConformer(conformer, assignId=False)
    return copied


def _bond_order_value(bond: Chem.Bond) -> float:
    return float(bond.GetBondTypeAsDouble())


def _reference_neighbor(
    center: Chem.Atom,
    axis_neighbor: Chem.Atom,
    symmetry_ranks: Sequence[int],
) -> int:
    """Choose a reproducible outer atom for a gauche/anti dihedral."""

    choices: list[Chem.Atom] = [
        atom
        for atom in center.GetNeighbors()
        if atom.GetIdx() != axis_neighbor.GetIdx()
    ]
    if not choices:
        raise ConformationCountError(
            f"cannot define a dihedral about bond "
            f"{center.GetIdx() + 1}-{axis_neighbor.GetIdx() + 1}"
        )

    def branch_size(atom: Chem.Atom) -> tuple[int, int]:
        """Heavy/total atoms reachable through a branch without the center."""

        seen = {center.GetIdx()}
        pending = [atom]
        heavy_atoms = 0
        total_atoms = 0
        while pending:
            current = pending.pop()
            if current.GetIdx() in seen:
                continue
            seen.add(current.GetIdx())
            total_atoms += 1
            heavy_atoms += int(current.GetAtomicNum() != 1)
            pending.extend(current.GetNeighbors())
        return heavy_atoms, total_atoms

    def priority(
        atom: Chem.Atom,
    ) -> tuple[int, int, int, float, int, int, int, int]:
        bond = center.GetOwningMol().GetBondBetweenAtoms(
            center.GetIdx(), atom.GetIdx()
        )
        heavy_size, total_size = branch_size(atom)
        return (
            heavy_size,
            total_size,
            int(atom.GetAtomicNum() != 1),
            _bond_order_value(bond),
            atom.GetAtomicNum(),
            atom.GetDegree(),
            symmetry_ranks[atom.GetIdx()],
            -atom.GetIdx(),
        )

    return max(choices, key=priority).GetIdx()


def _side_is_rotationally_indistinguishable(
    center: Chem.Atom,
    axis_neighbor: Chem.Atom,
    symmetry_ranks: Sequence[int],
) -> bool:
    """Whether rotating this side cannot create a distinct conformation.

    Empty terminal atoms and a linear sp group have no off-axis orientation.
    Three symmetry-equivalent substituent branches handle groups such as
    CH3, CF3, and C(CH3)3 without relying on atom-name conventions.
    """

    branches = [
        atom
        for atom in center.GetNeighbors()
        if atom.GetIdx() != axis_neighbor.GetIdx()
    ]
    if not branches:
        return True
    if (
        center.GetHybridization() == Chem.HybridizationType.SP
        and len(branches) == 1
    ):
        return True
    if len(branches) == 3:
        branch_ranks = {symmetry_ranks[atom.GetIdx()] for atom in branches}
        if len(branch_ranks) == 1:
            return True
    return False


def _find_rotors(mol: Chem.Mol) -> tuple[Rotor, ...]:
    """Find distinguishable acyclic rotors."""

    # Explicit H atoms are necessary to distinguish terminal OH/SH/NH2 from
    # featureless terminal atoms. Existing atom indices are retained by AddHs.
    working = Chem.AddHs(Chem.Mol(mol), addCoords=True)
    symmetry_ranks = list(
        Chem.CanonicalRankAtoms(
            working,
            breakTies=False,
            includeChirality=True,
            includeIsotopes=True,
        )
    )

    rotors: list[Rotor] = []
    for bond in working.GetBonds():
        if bond.GetBondType() != Chem.BondType.SINGLE:
            continue
        if bond.GetIsAromatic() or bond.IsInRing():
            continue

        atom1 = bond.GetBeginAtom()
        atom2 = bond.GetEndAtom()
        if atom1.GetAtomicNum() == 1 or atom2.GetAtomicNum() == 1:
            continue
        if _side_is_rotationally_indistinguishable(
            atom1, atom2, symmetry_ranks
        ):
            continue
        if _side_is_rotationally_indistinguishable(
            atom2, atom1, symmetry_ranks
        ):
            continue

        outer1 = _reference_neighbor(atom1, atom2, symmetry_ranks)
        outer2 = _reference_neighbor(atom2, atom1, symmetry_ranks)
        is_disulfide = (
            atom1.GetAtomicNum() == 16 and atom2.GetAtomicNum() == 16
        )
        rotors.append(
            Rotor(
                bond_index=bond.GetIdx(),
                dihedral=(outer1, atom1.GetIdx(), atom2.GetIdx(), outer2),
                kind="disulfide" if is_disulfide else "ordinary",
            )
        )

    return tuple(rotors)


def _canonical_heavy_smiles(mol: Chem.Mol) -> str:
    """Return a stereochemistry-independent structural key."""

    return Chem.MolToSmiles(
        Chem.RemoveHs(Chem.Mol(mol)),
        canonical=True,
        isomericSmiles=False,
    )


def _rotor_atoms(mol: Chem.Mol, rotor: Rotor) -> tuple[Chem.Atom, Chem.Atom]:
    bond = mol.GetBondWithIdx(rotor.bond_index)
    return bond.GetBeginAtom(), bond.GetEndAtom()


def _has_attached_hydrogen(atom: Chem.Atom) -> bool:
    return any(
        neighbor.GetAtomicNum() == 1 for neighbor in atom.GetNeighbors()
    ) or bool(atom.GetTotalNumHs(includeNeighbors=True))


def _is_carbonyl_carbon(atom: Chem.Atom) -> bool:
    if atom.GetAtomicNum() != 6:
        return False
    mol = atom.GetOwningMol()
    return any(
        neighbor.GetAtomicNum() == 8
        and mol.GetBondBetweenAtoms(
            atom.GetIdx(), neighbor.GetIdx()
        ).GetBondType()
        == Chem.BondType.DOUBLE
        for neighbor in atom.GetNeighbors()
    )


def _is_ketone_carbonyl(atom: Chem.Atom) -> bool:
    return _is_carbonyl_carbon(atom) and sum(
        neighbor.GetAtomicNum() == 6 for neighbor in atom.GetNeighbors()
    ) == 2


def _has_carbon_double_bond_other_than(
    atom: Chem.Atom,
    excluded_atom: Chem.Atom,
) -> bool:
    mol = atom.GetOwningMol()
    return any(
        neighbor.GetIdx() != excluded_atom.GetIdx()
        and neighbor.GetAtomicNum() == 6
        and mol.GetBondBetweenAtoms(
            atom.GetIdx(), neighbor.GetIdx()
        ).GetBondType()
        == Chem.BondType.DOUBLE
        for neighbor in atom.GetNeighbors()
    )


def _is_conjugated_diene_bond(mol: Chem.Mol, rotor: Rotor) -> bool:
    atom1, atom2 = _rotor_atoms(mol, rotor)
    if atom1.GetIsAromatic() or atom2.GetIsAromatic():
        return False
    if (
        atom1.GetHybridization() != Chem.HybridizationType.SP2
        or atom2.GetHybridization() != Chem.HybridizationType.SP2
    ):
        return False
    return _has_carbon_double_bond_other_than(
        atom1, atom2
    ) and _has_carbon_double_bond_other_than(atom2, atom1)


def _aryl_and_nonaryl_atom(
    mol: Chem.Mol,
    rotor: Rotor,
) -> tuple[Chem.Atom, Chem.Atom] | None:
    atom1, atom2 = _rotor_atoms(mol, rotor)
    if atom1.GetIsAromatic() and not atom2.GetIsAromatic():
        return atom1, atom2
    if atom2.GetIsAromatic() and not atom1.GetIsAromatic():
        return atom2, atom1
    return None


def _is_aryl_ethyl_bond(mol: Chem.Mol, rotor: Rotor) -> bool:
    pair = _aryl_and_nonaryl_atom(mol, rotor)
    if pair is None:
        return False
    aryl, nonaryl = pair
    return (
        aryl.GetAtomicNum() == 6
        and nonaryl.GetAtomicNum() == 6
        and nonaryl.GetHybridization() == Chem.HybridizationType.SP3
        and any(
            neighbor.GetAtomicNum() == 6
            and neighbor.GetIdx() != aryl.GetIdx()
            for neighbor in nonaryl.GetNeighbors()
        )
    )


def _is_aryl_alkene_bond(mol: Chem.Mol, rotor: Rotor) -> bool:
    pair = _aryl_and_nonaryl_atom(mol, rotor)
    if pair is None:
        return False
    aryl, nonaryl = pair
    if (
        aryl.GetAtomicNum() != 6
        or nonaryl.GetAtomicNum() != 6
        or nonaryl.GetHybridization() != Chem.HybridizationType.SP2
    ):
        return False
    return _has_carbon_double_bond_other_than(nonaryl, aryl)


def _is_acyclic_perfluoroalkane(mol: Chem.Mol) -> bool:
    heavy = Chem.RemoveHs(Chem.Mol(mol))
    if heavy.GetRingInfo().NumRings():
        return False
    atoms = list(heavy.GetAtoms())
    if not atoms or any(atom.GetAtomicNum() not in {6, 9} for atom in atoms):
        return False
    carbons = [atom for atom in atoms if atom.GetAtomicNum() == 6]
    return bool(carbons) and all(
        atom.GetHybridization() == Chem.HybridizationType.SP3
        and not _has_attached_hydrogen(atom)
        for atom in carbons
    )


def _rotor_factor_override(
    mol: Chem.Mol,
    rotor: Rotor,
    aryl_ethyl_rotor_count: int,
) -> int | None:
    """Return a fixed factor for chemistry classes needing special handling."""

    atom1, atom2 = _rotor_atoms(mol, rotor)
    atomic_numbers = {atom1.GetAtomicNum(), atom2.GetAtomicNum()}

    if _is_acyclic_perfluoroalkane(mol) and atomic_numbers == {6}:
        return 1

    if _is_conjugated_diene_bond(mol, rotor):
        return 2

    if _is_aryl_alkene_bond(mol, rotor):
        return 2

    if _is_aryl_ethyl_bond(mol, rotor):
        return 2 if aryl_ethyl_rotor_count == 1 else 3

    if _is_ketone_carbonyl(atom1) or _is_ketone_carbonyl(atom2):
        return 1

    for carbon, heteroatom in ((atom1, atom2), (atom2, atom1)):
        if carbon.GetAtomicNum() != 6:
            continue
        heteroatomic_number = heteroatom.GetAtomicNum()

        if (
            carbon.GetIsAromatic()
            and heteroatomic_number in {7, 8, 16}
            and _has_attached_hydrogen(heteroatom)
        ):
            return 1

        if (
            carbon.GetHybridization() == Chem.HybridizationType.SP3
            and heteroatomic_number == 8
            and heteroatom.GetHybridization() == Chem.HybridizationType.SP3
        ):
            if _has_attached_hydrogen(heteroatom):
                return 3
            carbon_neighbors = [
                neighbor
                for neighbor in heteroatom.GetNeighbors()
                if neighbor.GetAtomicNum() != 1
            ]
            if (
                carbon_neighbors
                and all(
                    neighbor.GetAtomicNum() == 6
                    for neighbor in carbon_neighbors
                )
                and not any(
                    _is_carbonyl_carbon(neighbor)
                    for neighbor in carbon_neighbors
                )
            ):
                return 3

        if (
            carbon.GetHybridization() == Chem.HybridizationType.SP3
            and heteroatomic_number == 7
            and heteroatom.GetHybridization() == Chem.HybridizationType.SP3
            and not any(
                _is_carbonyl_carbon(neighbor)
                for neighbor in heteroatom.GetNeighbors()
            )
        ):
            return 3

        if (
            carbon.GetHybridization() == Chem.HybridizationType.SP3
            and heteroatomic_number == 16
            and _has_attached_hydrogen(heteroatom)
        ):
            return 3

    return None


class _ForceField:
    """Small adapter over RDKit's MMFF and UFF APIs."""

    def __init__(self, mol: Chem.Mol) -> None:
        properties = AllChem.MMFFGetMoleculeProperties(
            mol,
            mmffVariant="MMFF94s",
        )
        if properties is not None:
            self.name = "MMFF94s"
            return
        if AllChem.UFFHasAllMoleculeParams(mol):
            self.name = "UFF"
            return
        raise ConformationCountError("force-field parameters are unavailable")

    def make(self, mol: Chem.Mol):
        if self.name == "MMFF94s":
            properties = AllChem.MMFFGetMoleculeProperties(
                mol, mmffVariant="MMFF94s"
            )
            if properties is None:
                raise ConformationCountError("lost MMFF94s parameters")
            force_field = AllChem.MMFFGetMoleculeForceField(
                mol, properties, confId=0
            )
        else:
            force_field = AllChem.UFFGetMoleculeForceField(mol, confId=0)
        if force_field is None:
            raise ConformationCountError(
                f"RDKit could not construct the {self.name} force field"
            )
        return force_field

    def energy(self, mol: Chem.Mol) -> float:
        return float(self.make(mol).CalcEnergy())

    def minimize(
        self,
        mol: Chem.Mol,
        torsion: tuple[int, int, int, int] | None = None,
        target_degrees: float | None = None,
    ) -> bool:
        force_field = self.make(mol)
        if torsion is not None:
            if target_degrees is None:
                raise AssertionError("a torsion constraint needs a target")
            # Keep the optimizer in the requested torsional well while
            # allowing the minimum to move away from an idealized 60/180
            # degree value.
            lower = target_degrees - 15.0
            upper = target_degrees + 15.0
            if self.name == "MMFF94s":
                force_field.MMFFAddTorsionConstraint(
                    *torsion, False, lower, upper, 100.0
                )
            else:
                force_field.UFFAddTorsionConstraint(
                    *torsion, False, lower, upper, 100.0
                )
        status = force_field.Minimize(
            maxIts=MAX_MINIMIZATION_ITERATIONS
        )
        if status == 0:
            return True

        # Constrained torsions can settle at a window boundary with a tiny
        # residual gradient, causing RDKit to report non-convergence even
        # after the energy is stationary. Continue in bounded chunks and
        # accept only a quantitatively stable energy.
        previous_energy = float(force_field.CalcEnergy())
        for _ in range(MAX_MINIMIZATION_RESTARTS):
            status = force_field.Minimize(
                maxIts=MAX_MINIMIZATION_ITERATIONS
            )
            current_energy = float(force_field.CalcEnergy())
            if status == 0:
                return True
            if (
                abs(current_energy - previous_energy)
                <= MINIMIZATION_ENERGY_TOLERANCE
            ):
                return True
            previous_energy = current_energy
        return False


def _scan_energy(
    baseline: Chem.Mol,
    force_field: _ForceField,
    dihedral: tuple[int, int, int, int],
    target_degrees: float,
) -> float:
    scanned = Chem.Mol(baseline)
    rdMolTransforms.SetDihedralDeg(
        scanned.GetConformer(0), *dihedral, target_degrees
    )
    converged = force_field.minimize(
        scanned,
        torsion=dihedral,
        target_degrees=target_degrees,
    )
    if not converged:
        _raise_nonconvergence(target_degrees)
    # Rebuild without the constraint so its penalty never enters the result.
    return force_field.energy(scanned)


def _raise_nonconvergence(target_degrees: float) -> None:
    raise ConformationCountError(
        "force-field minimization did not converge for the torsional state "
        f"centered at {target_degrees:.1f} degrees"
    )


def _rotor_factors(
    mol: Chem.Mol,
    rotors: Sequence[Rotor],
) -> tuple[int, ...]:
    """Evaluate all rotor factors with the existing topology/energy rules."""

    aryl_ethyl_rotor_count = sum(
        _is_aryl_ethyl_bond(mol, rotor) for rotor in rotors
    )
    overrides = [
        (
            _rotor_factor_override(mol, rotor, aryl_ethyl_rotor_count)
            if rotor.kind != "disulfide"
            else None
        )
        for rotor in rotors
    ]
    needs_force_field = any(
        rotor.kind != "disulfide" and override is None
        for rotor, override in zip(rotors, overrides)
    )

    baseline: Chem.Mol | None = None
    force_field: _ForceField | None = None
    if needs_force_field:
        baseline = Chem.AddHs(
            _single_conformer_with_id_zero(mol),
            addCoords=True,
        )
        force_field = _ForceField(baseline)
        if not force_field.minimize(baseline):
            raise ConformationCountError(
                "force-field minimization did not converge for the reference "
                "conformation"
            )

    factors: list[int] = []
    for rotor, override in zip(rotors, overrides):
        if rotor.kind == "disulfide":
            factors.append(2)
            continue
        if override is not None:
            factors.append(override)
            continue
        if baseline is None or force_field is None:
            raise AssertionError("an unresolved rotor needs a force field")

        anti = _scan_energy(
            baseline,
            force_field,
            rotor.dihedral,
            180.0,
        )
        gauche_plus = _scan_energy(
            baseline,
            force_field,
            rotor.dihedral,
            60.0,
        )
        gauche_minus = _scan_energy(
            baseline,
            force_field,
            rotor.dihedral,
            -60.0,
        )
        delta = min(gauche_plus, gauche_minus) - anti
        factor = (
            3
            if abs(delta) <= ENERGY_CUTOFF_KCAL_MOL + 1.0e-9
            else 1
        )
        factors.append(factor)
    return tuple(factors)


def count_molecule(mol: Chem.Mol) -> int:
    """Return the rotational-conformer count for one 3D RDKit molecule."""

    if mol.GetNumConformers() == 0:
        raise ConformationCountError("the molecule has no 3D conformer")
    if len(Chem.GetMolFrags(mol)) != 1:
        raise ConformationCountError(
            "multiple molecular fragments are not combined automatically"
        )

    rotors = _find_rotors(mol)
    count = math.prod(_rotor_factors(mol, rotors))
    return MOLECULE_COUNT_OVERRIDES.get(
        _canonical_heavy_smiles(mol),
        count,
    )


def count_mol2_file(path: Path | str) -> int:
    """Load a MOL2 file and return its rotational-conformer count."""

    return count_molecule(load_mol2(Path(path)))


def _count_mol2_path(path: Path) -> tuple[int | None, str | None]:
    """Worker entry point that returns an error instead of aborting a batch."""

    try:
        return count_mol2_file(path), None
    except Exception as error:
        return None, f"{type(error).__name__}: {error}"


def _positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def _count_paths_in_parallel(
    paths: Sequence[Path],
    workers: int,
    progress_every: int,
) -> tuple[list[int | None], list[tuple[Path, str]]]:
    """Count paths concurrently, retaining input order and reporting progress."""

    total = len(paths)
    counts: list[int | None] = []
    failures: list[tuple[Path, str]] = []
    with ProcessPoolExecutor(max_workers=min(workers, total)) as executor:
        for processed, (path, result) in enumerate(
            zip(
                paths,
                executor.map(_count_mol2_path, paths),
            ),
            start=1,
        ):
            count, error = result
            counts.append(count)
            if error is not None:
                failures.append((path, error))
            if processed % progress_every == 0 or processed == total:
                print(
                    f"processed {processed}/{total} molecules; "
                    f"failed {len(failures)}",
                    file=sys.stderr,
                    flush=True,
                )
    return counts, failures


def _write_failed_info(failures: Sequence[tuple[Path, str]]) -> None:
    """Write failed molecule IDs and errors without retaining stale results."""

    with Path("failed_info").open("w", newline="") as output:
        writer = csv.writer(output, lineterminator="\n")
        writer.writerow(("molecule_id", "input_path", "error"))
        writer.writerows(
            (path.stem, str(path), error) for path, error in failures
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Count distinguishable rotational conformers in 3D MOL2 files."
        )
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="one or more MOL2 files or directories containing MOL2 files",
    )
    parser.add_argument(
        "--workers",
        type=_positive_integer,
        default=DEFAULT_WORKERS,
        help=(
            "number of molecule-counting processes "
            f"(default: {DEFAULT_WORKERS})"
        ),
    )
    parser.add_argument(
        "--progress-every",
        type=_positive_integer,
        default=DEFAULT_PROGRESS_EVERY,
        help=(
            "write progress to stderr after this many molecules "
            f"(default: {DEFAULT_PROGRESS_EVERY})"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    RDLogger.DisableLog("rdApp.warning")
    try:
        arguments = parser.parse_args(argv)
        paths = collect_mol2_paths(arguments.inputs)
        counts, failures = _count_paths_in_parallel(
            paths,
            arguments.workers,
            arguments.progress_every,
        )
        _write_failed_info(failures)
    except (ConformationCountError, OSError, ValueError) as error:
        parser.exit(2, f"error: {error}\n")

    writer = csv.writer(sys.stdout, lineterminator="\n")
    writer.writerow(("molecule_id", "conformer_count"))
    writer.writerows(
        (path.stem, count)
        for path, count in zip(paths, counts)
        if count is not None
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
