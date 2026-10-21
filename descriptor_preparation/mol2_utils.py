"""Small helpers for reading the Tripos mol2 files used to build descriptors."""

from dataclasses import dataclass

# Atomic masses in the order they are tested. Each entry is
# (SYBYL element symbol, upper-case atom name that also counts, mass).
HEAVY_ATOM_MASSES = [
    ("B", None, 10.81),
    ("Cl", "CL", 35.453),
    ("Br", "BR", 79.904),
    ("Si", "SI", 28.085),
    ("C", None, 12.011),
    ("O", None, 15.999),
    ("S", None, 32.06),
    ("N", None, 14.007),
    ("P", None, 30.974),
    ("F", None, 18.998),
    ("I", None, 126.90),
    ("Fe", "FE", 55.845),
]
HYDROGEN_MASS = 1.008


@dataclass
class Atom:
    atom_id: str
    name: str
    atom_type: str  # SYBYL type such as "C.ar"

    @property
    def element(self):
        return self.atom_type.split(".")[0]

    @property
    def is_hydrogen(self):
        return self.name.startswith("H")

    @property
    def mass(self):
        if self.is_hydrogen:
            return HYDROGEN_MASS
        for element, upper_name, mass in HEAVY_ATOM_MASSES:
            if self.element == element or self.name == upper_name:
                return mass
        return 0.0  # unknown elements contribute nothing, as in the original scripts


@dataclass
class Bond:
    atom_1: str
    atom_2: str
    order: str


@dataclass
class Mol2:
    ring_count: int
    atoms: list
    bonds: list

    @property
    def type_by_atom_id(self):
        return {atom.atom_id: atom.atom_type for atom in self.atoms}

    @property
    def heavy_atom_count(self):
        return sum(not atom.is_hydrogen for atom in self.atoms)

    @property
    def hydrogen_count(self):
        return sum(atom.is_hydrogen for atom in self.atoms)

    @property
    def molecular_weight(self):
        return round(sum(atom.mass for atom in self.atoms), 3)

    @property
    def single_heavy_bond_count(self):
        """Number of single bonds that do not involve a hydrogen."""
        types = self.type_by_atom_id
        return sum(
            bond.order == "1" and types[bond.atom_1] != "H" and types[bond.atom_2] != "H"
            for bond in self.bonds
        )


def read_mol2(path):
    """Parse the MOLECULE, ATOM and BOND sections of a mol2 file."""
    with open(path) as handle:
        lines = handle.readlines()

    atom_count, bond_count = lines[2].split()[:2]
    ring_count = int(bond_count) - int(atom_count) + 1

    atoms, bonds = [], []
    section = None
    for line in lines:
        if line.startswith("@<TRIPOS>"):
            section = line.strip()[len("@<TRIPOS>"):]
            if section == "SUBSTRUCTURE":
                break
            continue
        if not line.strip():
            continue
        fields = line.split()
        if section == "ATOM":
            atoms.append(Atom(atom_id=fields[0], name=fields[1], atom_type=fields[5]))
        elif section == "BOND":
            bonds.append(Bond(atom_1=fields[1], atom_2=fields[2], order=fields[3]))

    return Mol2(ring_count=ring_count, atoms=atoms, bonds=bonds)


def read_type_library(path):
    """Read a one-type-per-line library file, sorted as the CSV columns are."""
    with open(path) as handle:
        return sorted(line.strip() for line in handle if line.strip())


def read_entropy_table(path):
    """Yield (ligand_id, symmetry_correction, corrected_entropy) strings."""
    with open(path) as handle:
        for line in handle:
            fields = line.split()
            if fields:
                yield fields[0], fields[1], fields[2]
