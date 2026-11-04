#!/usr/bin/env python3
"""Train GINE graph neural networks for gas-phase standard molar entropy (S°).

Self-contained: needs only numpy, pandas, scipy, scikit-learn, RDKit, PyTorch and PyTorch
Geometric (the `GNN` conda env). Merges train.py, run6/train_oversampled.py and the
entropy_gnn package for the run5_102926 / run6_102926 recipes.

Two models are trained from each molecule's SMILES graph (RDKit atoms and bonds, plus
interatomic-distance features from an ETKDG 3D embedding):
  gnn_only  GINE graph encoder only
  hybrid    GINE graph encoder fused with an MLP over the CSV ligand descriptors

Data: one combined CSV with ID, SMILES, entropy and descriptor columns, split 80/10/10
(train/dev/test, seed 42). Each mode's winner is chosen by development-set Pearson r.

Feature selection (hybrid descriptors) is a training input:
  (default)                         every column except ID, SMILES and entropy
  --feature-file selected_features  only the listed columns (one per line, '#' comments allowed)
The list is stored in each checkpoint; predict_GINE.py uses it automatically.

Recipes (run from this folder):
  # run5_102926: baselines + hyperparameter grid, no oversampling
  python train_GINE.py --csv hybrid_data/hybrid_train_102926.csv --output-dir my_run5 \\
      --feature-file selected_features --tune --gpu-index 0 --eval-csv std128 hybrid_data/hybrid_std128_102926.csv

  # run6_102926: run5's two mode winners retrained with 3-segment oversampling
  python train_GINE.py --csv hybrid_data/hybrid_train_102926.csv --output-dir my_run6 \\
      --feature-file selected_features --source-run my_run5 --oversample 3seg --gpu-index 0
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import math
import platform
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import AttentiveFP, BatchNorm, GINEConv, global_add_pool, global_mean_pool

RDLogger.DisableLog("rdApp.*")

# ====================================================================== constants
ID_COLUMN = "molecule_id"
SMILES_COLUMN = "smiles"
TARGET_COLUMN = "entropy"
NON_COUNT_COLUMNS = {"MW", "symmetry"}  # every other descriptor is a count and gets log1p
RBF_CENTERS = np.linspace(0.0, 3.0, 16, dtype=np.float32)
RBF_GAMMA = float(1.0 / (RBF_CENTERS[1] - RBF_CENTERS[0]) ** 2)

SMILES_COLUMN_ALIASES = ("SMILES", "smiles", "Smiles")
ID_COLUMN_ALIASES = ("Ligands", "ID", "Number_in_Database", "id", "ligand", "Ligand")
TARGET_COLUMN_ALIASES = ("entropy", "Entropy")

ATOMIC_NUMBERS = [1, 6, 7, 8, 9, 14, 15, 16, 17, 26, 35, 53]
HYBRIDIZATIONS = [
    Chem.HybridizationType.UNSPECIFIED,
    Chem.HybridizationType.S,
    Chem.HybridizationType.SP,
    Chem.HybridizationType.SP2,
    Chem.HybridizationType.SP3,
    Chem.HybridizationType.SP3D,
    Chem.HybridizationType.SP3D2,
]
CHIRAL_TAGS = [
    Chem.ChiralType.CHI_UNSPECIFIED,
    Chem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.ChiralType.CHI_TETRAHEDRAL_CCW,
]
BOND_TYPES = [Chem.BondType.SINGLE, Chem.BondType.DOUBLE, Chem.BondType.TRIPLE, Chem.BondType.AROMATIC]
BOND_STEREOS = [
    Chem.BondStereo.STEREONONE,
    Chem.BondStereo.STEREOANY,
    Chem.BondStereo.STEREOZ,
    Chem.BondStereo.STEREOE,
    Chem.BondStereo.STEREOCIS,
    Chem.BondStereo.STEREOTRANS,
]

PEARSON_TARGET = 0.95
CACHE_FORMAT = 4  # 3 = old train.py caches (very large files); 4 = this script


# ====================================================================== reporting helpers
JSON_PRECISION = {"r2": 6, "pearson": 6, "mae": 4, "rmse": 4}
DISPLAY_PRECISION = {"r2": 4, "pearson": 4, "mae": 3, "rmse": 3}
PREDICTION_COLUMNS = ("predicted_entropy", "entropy", "residual")
PREDICTION_DECIMALS = 4
METRIC_ORDER = ("r2", "pearson", "mae", "rmse")


def as_python_number(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.ndim != 0:
            return [as_python_number(item) for item in value.detach().cpu().flatten().tolist()]
        value = value.detach().cpu().item()
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            value = value.item()
        else:
            return [as_python_number(item) for item in value.tolist()]
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, bool):
        return value
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, int):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value.item())
    return value


def jsonable(value: Any, key: str | None = None) -> Any:
    """Nested JSON-safe copy; metric fields are rounded Python floats, non-finite values become null."""
    value = as_python_number(value)
    if isinstance(value, dict):
        return {str(item_key): jsonable(item, str(item_key)) for item_key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        digits = JSON_PRECISION.get(key or "", None)
        return round(value, digits) if digits is not None else value
    return value


def round_metrics(values: Mapping[str, Any]) -> dict[str, Any]:
    return {key: jsonable(value, key) for key, value in values.items()}


def format_metrics(values: Mapping[str, Any]) -> str:
    parts: list[str] = []
    keys = [key for key in METRIC_ORDER if key in values] or list(values)
    for key in keys:
        raw = as_python_number(values[key])
        if raw is None or isinstance(raw, float) and not math.isfinite(raw):
            parts.append(f"{key}=nan")
            continue
        parts.append(f"{key}={float(raw):.{DISPLAY_PRECISION.get(key, 4)}f}")
    return "  ".join(parts)


def format_eval_metrics(values: Mapping[str, Any]) -> str:
    prefix = ""
    if "metric_rows" in values and "total_rows" in values:
        prefix = f"n={int(values['metric_rows'])}/{int(values['total_rows'])}"
        if "overlap_rows" in values:
            prefix += f"  overlap={int(values['overlap_rows'])}"
        prefix += "  "
    return prefix + format_metrics(values)


def write_json(path: Path, values: Any) -> None:
    path.write_text(json.dumps(jsonable(values), indent=2, sort_keys=True, allow_nan=False) + "\n")


def write_predictions_csv(path: Path, frame: pd.DataFrame) -> None:
    output = frame.copy()
    for column in PREDICTION_COLUMNS:
        if column in output.columns:
            output[column] = pd.to_numeric(output[column], errors="coerce").round(PREDICTION_DECIMALS)
    output.to_csv(path, index=False, float_format="%.4f")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ====================================================================== molecules and graphs
def _one_hot_with_other(value: Any, choices: Sequence[Any]) -> list[float]:
    result = [float(value == choice) for choice in choices]
    result.append(float(value not in choices))
    return result


def normalize_molecule_id(value: Any) -> str:
    return Path(str(value).strip()).stem


def pick_column(frame: pd.DataFrame, aliases: Sequence[str], kind: str) -> str:
    present = [name for name in aliases if name in frame.columns]
    if not present:
        raise ValueError(f"could not find a {kind} column; expected one of {list(aliases)}, got {list(frame.columns)}")
    return present[0]


def _etkdg_params(seed: int) -> Any:
    try:
        params = AllChem.ETKDGv3()
    except AttributeError:
        params = AllChem.ETKDG()
    params.randomSeed = int(seed)
    params.maxIterations = 3
    return params


@dataclass
class MoleculeLoadResult:
    mol: Chem.Mol | None
    status: str
    error: str = ""
    canonical_smiles: str = ""
    inchikey: str = ""


def smiles_identity(smiles: str) -> tuple[str, str]:
    mol = Chem.MolFromSmiles(str(smiles).strip())
    if mol is None:
        return "", ""
    try:
        canonical = Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        canonical = ""
    try:
        inchikey = Chem.MolToInchiKey(mol)
    except Exception:
        inchikey = ""
    return canonical, inchikey


def load_molecule_from_smiles(smiles: str, seed: int = 42, embed_3d: bool = True) -> MoleculeLoadResult:
    """Parse SMILES; with embed_3d, embed one ETKDG conformer (2D coordinates if embedding fails)."""
    raw = str(smiles).strip()
    if not raw or raw.lower() == "nan":
        return MoleculeLoadResult(None, "unresolved", "empty SMILES")
    try:
        mol = Chem.MolFromSmiles(raw)
    except Exception as exc:
        return MoleculeLoadResult(None, "unresolved", f"{type(exc).__name__}: {exc}")
    if mol is None:
        return MoleculeLoadResult(None, "unresolved", "MolFromSmiles returned None")
    if mol.GetNumAtoms() == 0:
        return MoleculeLoadResult(None, "unresolved", "molecule has no heavy atoms")
    try:
        Chem.SanitizeMol(mol)
    except Exception as exc:
        return MoleculeLoadResult(None, "unresolved", f"sanitize failed: {type(exc).__name__}: {exc}")

    canonical, inchikey = smiles_identity(raw)
    status = "smiles_2d"
    work = Chem.Mol(mol)
    if embed_3d:
        heavy_with_h = Chem.AddHs(work)
        code = AllChem.EmbedMolecule(heavy_with_h, _etkdg_params(seed))
        if code == 0:
            try:
                work = Chem.RemoveHs(heavy_with_h, sanitize=True)
                if work.GetNumAtoms() == 0:
                    return MoleculeLoadResult(None, "unresolved", "molecule has no heavy atoms")
                status = "smiles_3d"
            except Exception:
                AllChem.Compute2DCoords(work)
                status = "smiles_2d"
        else:
            AllChem.Compute2DCoords(work)
            status = "smiles_2d"
    else:
        AllChem.Compute2DCoords(work)
        status = "smiles_2d"
    if work.GetNumConformers() == 0:
        AllChem.Compute2DCoords(work)
        status = "smiles_2d"
    return MoleculeLoadResult(work, status, canonical_smiles=canonical, inchikey=inchikey)


def atom_features(atom: Chem.Atom) -> list[float]:
    partial_charge = 0.0
    missing_charge = 1.0
    if atom.HasProp("_TriposPartialCharge"):
        try:
            partial_charge = float(atom.GetProp("_TriposPartialCharge"))
            missing_charge = 0.0
        except ValueError:
            pass
    features: list[float] = []
    features.extend(_one_hot_with_other(atom.GetAtomicNum(), ATOMIC_NUMBERS))
    features.extend(_one_hot_with_other(min(atom.GetDegree(), 7), list(range(7))))
    features.extend(_one_hot_with_other(atom.GetFormalCharge(), list(range(-3, 4))))
    features.extend(_one_hot_with_other(atom.GetHybridization(), HYBRIDIZATIONS))
    features.extend([float(atom.GetIsAromatic()), float(atom.IsInRing())])
    features.extend(_one_hot_with_other(atom.GetChiralTag(), CHIRAL_TAGS))
    features.extend([float(atom.GetMass()) / 200.0, partial_charge, missing_charge])
    return features


def bond_features(bond: Chem.Bond, distance: float) -> list[float]:
    features: list[float] = []
    features.extend(_one_hot_with_other(bond.GetBondType(), BOND_TYPES))
    features.extend([float(bond.GetIsConjugated()), float(bond.IsInRing())])
    features.extend(_one_hot_with_other(bond.GetStereo(), BOND_STEREOS))
    rbf = np.exp(-RBF_GAMMA * (RBF_CENTERS - float(distance)) ** 2)
    features.extend(rbf.tolist())
    return features


def edge_feature_dims() -> tuple[int, int]:
    """(topology-only edge width, topology + 16 distance RBFs)."""
    topology = (len(BOND_TYPES) + 1) + 2 + (len(BOND_STEREOS) + 1)
    return topology, topology + len(RBF_CENTERS)


def build_graph(mol: Chem.Mol) -> Data:
    if mol.GetNumConformers() == 0:
        AllChem.Compute2DCoords(mol)
    if mol.GetNumConformers() == 0:
        raise ValueError("molecule has no conformer")
    conformer = mol.GetConformer()
    x = torch.tensor([atom_features(atom) for atom in mol.GetAtoms()], dtype=torch.float32)
    positions = torch.tensor(
        [
            [conformer.GetAtomPosition(i).x, conformer.GetAtomPosition(i).y, conformer.GetAtomPosition(i).z]
            for i in range(mol.GetNumAtoms())
        ],
        dtype=torch.float32,
    )
    edges: list[list[int]] = []
    attributes: list[list[float]] = []
    for bond in mol.GetBonds():
        begin = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        distance = float(torch.linalg.vector_norm(positions[begin] - positions[end]).item())
        attr = bond_features(bond, distance)
        edges.extend([[begin, end], [end, begin]])
        attributes.extend([attr, attr])
    _, full_edge_dim = edge_feature_dims()
    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(attributes, dtype=torch.float32)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, full_edge_dim), dtype=torch.float32)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, pos=positions)


# ====================================================================== descriptors
@dataclass
class FeaturePreprocessor:
    """log1p on count descriptors, then z-scores from the training split; z-scored target."""

    predictor_columns: list[str]
    constant_columns: list[str]
    retained_columns: list[str]
    count_columns: list[str]
    feature_mean: list[float]
    feature_std: list[float]
    target_mean: float
    target_std: float

    @classmethod
    def fit(cls, full_frame: pd.DataFrame, train_frame: pd.DataFrame, predictor_columns: Sequence[str]) -> "FeaturePreprocessor":
        if predictor_columns:
            _validate_numeric_frame(full_frame, predictor_columns)
        constant_columns = [c for c in predictor_columns if bool((full_frame[c].astype(float) == 0.0).all())]
        retained = list(predictor_columns)  # all-zero columns are reported but kept
        count_columns = [c for c in retained if c not in NON_COUNT_COLUMNS]
        if retained:
            transformed = _transform_feature_values(train_frame, retained, count_columns)
            mean = transformed.mean(axis=0)
            std = transformed.std(axis=0, ddof=0)
            std[std == 0.0] = 1.0
            mean_list, std_list = mean.astype(float).tolist(), std.astype(float).tolist()
        else:
            mean_list, std_list = [], []
        target = train_frame[TARGET_COLUMN].to_numpy(dtype=np.float64)
        target_mean = float(target.mean())
        target_std = float(target.std(ddof=0)) or 1.0
        return cls(list(predictor_columns), constant_columns, retained, count_columns,
                   mean_list, std_list, target_mean, target_std)

    def transform_frame(self, frame: pd.DataFrame) -> np.ndarray:
        if not self.predictor_columns:
            return np.zeros((len(frame), 0), dtype=np.float32)
        missing = [c for c in self.predictor_columns if c not in frame.columns]
        if missing:
            raise ValueError(f"missing predictor columns: {missing}")
        _validate_numeric_frame(frame, self.predictor_columns)
        values = _transform_feature_values(frame, self.retained_columns, self.count_columns)
        mean = np.asarray(self.feature_mean, dtype=np.float64)
        std = np.asarray(self.feature_std, dtype=np.float64)
        return ((values - mean) / std).astype(np.float32)

    def transform_target(self, values: Iterable[float]) -> np.ndarray:
        array = np.asarray(list(values), dtype=np.float64)
        return ((array - self.target_mean) / self.target_std).astype(np.float32)

    def inverse_target(self, values: np.ndarray | torch.Tensor) -> np.ndarray:
        if isinstance(values, torch.Tensor):
            values = values.detach().cpu().numpy()
        return np.asarray(values) * self.target_std + self.target_mean

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "FeaturePreprocessor":
        return cls(**values)


def _validate_numeric_frame(frame: pd.DataFrame, columns: Sequence[str]) -> None:
    for column in columns:
        try:
            values = pd.to_numeric(frame[column], errors="raise").to_numpy(dtype=np.float64)
        except Exception as exc:
            raise ValueError(f"column {column!r} is not numeric: {exc}") from exc
        if not np.isfinite(values).all():
            raise ValueError(f"column {column!r} contains non-finite values")


def _transform_feature_values(frame: pd.DataFrame, retained_columns: Sequence[str], count_columns: Sequence[str]) -> np.ndarray:
    values = frame.loc[:, retained_columns].to_numpy(dtype=np.float64, copy=True)
    count_indices = [retained_columns.index(column) for column in count_columns]
    if count_indices:
        counts = values[:, count_indices]
        if (counts < 0.0).any():
            bad_column = count_columns[int(np.where(counts < 0.0)[1][0])]
            raise ValueError(f"count feature {bad_column!r} contains a negative value")
        values[:, count_indices] = np.log1p(counts)
    return values


def read_feature_list(path: str | Path) -> list[str]:
    """One column name per line; blank lines and '#' comments are ignored."""
    names = [line.strip() for line in Path(path).read_text().splitlines()]
    names = [name for name in names if name and not name.startswith("#")]
    if not names:
        raise ValueError(f"{path} lists no features")
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"{path} lists duplicate features: {duplicates[:8]}")
    return names


def load_hybrid_table(path: str | Path, require_entropy: bool = True,
                      feature_columns: Sequence[str] | None = None) -> tuple[pd.DataFrame, list[str]]:
    """Read ID, SMILES, entropy (optional) and descriptors.

    Without feature_columns every other column is a predictor; with it, only those columns
    (kept in CSV order).
    """
    frame = pd.read_csv(path)
    smiles_col = pick_column(frame, SMILES_COLUMN_ALIASES, "SMILES")
    id_col = pick_column(frame, ID_COLUMN_ALIASES, "molecule ID")
    target_col = next((name for name in TARGET_COLUMN_ALIASES if name in frame.columns), None)
    if require_entropy:
        target_col = pick_column(frame, TARGET_COLUMN_ALIASES, "entropy")
    predictor_columns = [c for c in frame.columns if c not in {id_col, smiles_col, target_col}]
    if feature_columns is not None:
        missing = [c for c in feature_columns if c not in predictor_columns]
        if missing:
            raise ValueError(f"{path} is missing {len(missing)} selected features, e.g. {missing[:8]}")
        keep = set(feature_columns)
        predictor_columns = [c for c in predictor_columns if c in keep]
    data: dict[str, Any] = {
        ID_COLUMN: frame[id_col].map(normalize_molecule_id).to_numpy(),
        SMILES_COLUMN: frame[smiles_col].astype(str).to_numpy(),
    }
    if target_col is not None:
        data[TARGET_COLUMN] = pd.to_numeric(frame[target_col], errors="raise").to_numpy()
    for column in predictor_columns:
        data[column] = frame[column].to_numpy()
    out = pd.DataFrame(data)
    validate_columns = predictor_columns + ([TARGET_COLUMN] if target_col is not None else [])
    if out[ID_COLUMN].duplicated().any():
        raise ValueError(f"{path} contains duplicate normalized molecule IDs")
    if validate_columns:
        _validate_numeric_frame(out, validate_columns)
    return out, predictor_columns


def _audit_row(reason: str, **values: Any) -> dict[str, Any]:
    row = {"reason": reason, "molecule_id": "", "smiles": "", "detail": ""}
    row.update(values)
    return row


@dataclass
class TrainingIdentities:
    """IDs, InChIKeys and canonical SMILES of every training-table molecule (for overlap checks)."""

    molecule_ids: list[str]
    inchikeys: list[str]
    canonical_smiles: list[str]

    def to_dict(self) -> dict[str, list[str]]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "TrainingIdentities":
        return cls(list(values.get("molecule_ids", [])), list(values.get("inchikeys", [])),
                   list(values.get("canonical_smiles", [])))

    def overlap_mask(self, molecule_ids: Sequence[str], inchikeys: Sequence[str], canonical_smiles: Sequence[str]) -> np.ndarray:
        train_ids = set(self.molecule_ids)
        train_keys = {v for v in self.inchikeys if v}
        train_smiles = {v for v in self.canonical_smiles if v}
        return np.asarray(
            [
                bool(mid and mid in train_ids) or bool(key and key in train_keys) or bool(can and can in train_smiles)
                for mid, key, can in zip(molecule_ids, inchikeys, canonical_smiles)
            ],
            dtype=bool,
        )


# ====================================================================== dataset
@dataclass
class PreparedDataset:
    graphs: list[Data]
    split_indices: dict[str, list[int]]
    preprocessor: FeaturePreprocessor
    predictor_columns: list[str]
    manifest: pd.DataFrame
    audit: pd.DataFrame
    feature_schema: dict[str, int]
    identities: TrainingIdentities
    source_frame: pd.DataFrame


def save_prepared_dataset(dataset: PreparedDataset, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "graphs": dataset.graphs,
            "split_indices": dataset.split_indices,
            "preprocessor": dataset.preprocessor.to_dict(),
            "predictor_columns": dataset.predictor_columns,
            "manifest": dataset.manifest,
            "audit": dataset.audit,
            "feature_schema": dataset.feature_schema,
            "identities": dataset.identities.to_dict(),
            "source_frame": dataset.source_frame,
        },
        path,
    )


def load_prepared_dataset(path: Path) -> PreparedDataset:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return PreparedDataset(
        graphs=payload["graphs"],
        split_indices=payload["split_indices"],
        preprocessor=FeaturePreprocessor.from_dict(payload["preprocessor"]),
        predictor_columns=list(payload["predictor_columns"]),
        manifest=payload["manifest"],
        audit=payload["audit"],
        feature_schema=payload["feature_schema"],
        identities=TrainingIdentities.from_dict(payload["identities"]),
        source_frame=payload["source_frame"],
    )


def split_indices_80_10_10(count: int, seed: int) -> dict[str, list[int]]:
    if count < 10:
        raise ValueError("an 80/10/10 split requires at least 10 usable molecules")
    indices = np.arange(count)
    train_indices, temporary = train_test_split(indices, test_size=0.2, random_state=seed, shuffle=True)
    dev_indices, test_indices = train_test_split(temporary, test_size=0.5, random_state=seed + 1, shuffle=True)
    return {
        "train": sorted(int(v) for v in train_indices),
        "dev": sorted(int(v) for v in dev_indices),
        "test": sorted(int(v) for v in test_indices),
    }


def prepare_dataset(csv_path: Path, sample_limit: int = 0, seed: int = 42, embed_3d: bool = True,
                    feature_columns: Sequence[str] | None = None) -> PreparedDataset:
    paired, predictor_columns = load_hybrid_table(csv_path, require_entropy=True, feature_columns=feature_columns)
    audit_rows: list[dict[str, Any]] = []
    paired = paired.sort_values(ID_COLUMN).reset_index(drop=True)
    rng = random.Random(seed)
    if sample_limit > 0 and len(paired) > sample_limit:
        chosen = sorted(rng.sample(range(len(paired)), sample_limit))
        paired = paired.iloc[chosen].reset_index(drop=True)

    graphs: list[Data] = []
    kept_rows: list[int] = []
    for position, row in paired.iterrows():
        result = load_molecule_from_smiles(str(row[SMILES_COLUMN]), seed=seed, embed_3d=embed_3d)
        if result.mol is None:
            audit_rows.append(_audit_row("unresolved_smiles", molecule_id=str(row[ID_COLUMN]),
                                         smiles=str(row[SMILES_COLUMN]), detail=result.error))
            continue
        try:
            graph = build_graph(result.mol)
        except Exception as exc:
            audit_rows.append(_audit_row("graph_build_failed", molecule_id=str(row[ID_COLUMN]),
                                         smiles=str(row[SMILES_COLUMN]), detail=f"{type(exc).__name__}: {exc}"))
            continue
        graph.molecule_id = str(row[ID_COLUMN])
        graph.smiles = str(row[SMILES_COLUMN])
        graph.canonical_smiles = result.canonical_smiles
        graph.inchikey = result.inchikey
        graph.chemistry_status = result.status
        graphs.append(graph)
        kept_rows.append(int(position))
        if (len(graphs) % 1000) == 0:
            print(f"built {len(graphs)} SMILES graphs", flush=True)

    if not graphs:
        raise ValueError("no usable labeled molecules were found")
    selected = paired.iloc[kept_rows].reset_index(drop=True)
    split_indices = split_indices_80_10_10(len(graphs), seed)
    split_by_index = {index: split for split, values in split_indices.items() for index in values}
    preprocessor = FeaturePreprocessor.fit(selected, selected.iloc[split_indices["train"]], predictor_columns)
    tabular = preprocessor.transform_frame(selected)
    targets = preprocessor.transform_target(selected[TARGET_COLUMN].astype(float))

    manifest_rows: list[dict[str, Any]] = []
    for index, graph in enumerate(graphs):
        # Own copy of the row: a view of the (column-ordered) matrix would make every saved graph
        # carry nearly the whole matrix, which is why old train.py caches reached hundreds of GB.
        graph.tabular = torch.from_numpy(np.ascontiguousarray(tabular[index : index + 1]))
        graph.y = torch.tensor([targets[index]], dtype=torch.float32)
        source_row = selected.iloc[index]
        manifest_rows.append(
            {
                "molecule_id": graph.molecule_id,
                "smiles": graph.smiles,
                "canonical_smiles": graph.canonical_smiles,
                "inchikey": graph.inchikey,
                "split": split_by_index[index],
                "chemistry_status": graph.chemistry_status,
                "symmetry": float(source_row["symmetry"]) if "symmetry" in selected.columns else None,
                "entropy": float(source_row[TARGET_COLUMN]),
            }
        )

    topology_edge_dim, full_edge_dim = edge_feature_dims()
    schema = {
        "node_dim": int(graphs[0].x.shape[1]),
        "topology_edge_dim": topology_edge_dim,
        "full_edge_dim": full_edge_dim,
        "tabular_dim": len(preprocessor.retained_columns),
    }
    identities = TrainingIdentities(
        molecule_ids=[g.molecule_id for g in graphs],
        inchikeys=[g.inchikey for g in graphs],
        canonical_smiles=[g.canonical_smiles for g in graphs],
    )
    return PreparedDataset(
        graphs=graphs,
        split_indices=split_indices,
        preprocessor=preprocessor,
        predictor_columns=list(predictor_columns),
        manifest=pd.DataFrame(manifest_rows),
        audit=pd.DataFrame(audit_rows, columns=["reason", "molecule_id", "smiles", "detail"]),
        feature_schema=schema,
        identities=identities,
        source_frame=selected,
    )


def dataset_cache_path(cache_dir: Path, csv_path: Path, seed: int, sample_limit: int, embed_3d: bool,
                       feature_columns: list[str] | None) -> Path:
    payload = {"csv": sha256_file(csv_path), "seed": seed, "sample_limit": sample_limit,
               "embed_3d": embed_3d, "format": CACHE_FORMAT}
    if feature_columns is not None:
        payload["features"] = sorted(feature_columns)
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:20]
    return cache_dir / f"{digest}.pt"


def load_or_prepare_dataset(args: argparse.Namespace) -> PreparedDataset:
    cache_path = None
    if args.cache_dir is not None:
        cache_path = dataset_cache_path(args.cache_dir, args.csv, args.seed, args.sample_limit,
                                        args.embed_3d, args.feature_columns)
        if cache_path.exists():
            print(f"loading cached dataset from {cache_path}", flush=True)
            return load_prepared_dataset(cache_path)
    dataset = prepare_dataset(args.csv, sample_limit=args.sample_limit, seed=args.seed,
                              embed_3d=args.embed_3d, feature_columns=args.feature_columns)
    if cache_path is not None:
        print(f"writing dataset cache to {cache_path}", flush=True)
        save_prepared_dataset(dataset, cache_path)
    return dataset


# ====================================================================== oversampling (run6)
def heavy_atom_counts(source_frame: pd.DataFrame, csv: Path) -> pd.Series:
    """Heavy-atom counts aligned to source_frame; read from the CSV when 'heavy' is not a predictor."""
    if "heavy" in source_frame.columns:
        return source_frame["heavy"]
    table = pd.read_csv(csv)
    id_col = pick_column(table, ID_COLUMN_ALIASES, "molecule ID")
    if "heavy" not in table.columns:
        raise ValueError(f"--oversample 3seg needs a 'heavy' column in {csv}")
    lookup = {normalize_molecule_id(i): h for i, h in zip(table[id_col], table["heavy"])}
    return pd.Series([lookup[i] for i in source_frame[ID_COLUMN]], index=source_frame.index)


def copy_count(heavy_atoms: float) -> int:
    if heavy_atoms < 12:
        return 5
    if heavy_atoms <= 15:
        return 3
    return 1


def oversample_train(heavy: pd.Series, train_indices: list[int], seed: int) -> tuple[list[int], dict[str, int]]:
    """Training rows only: heavy < 12 x5, 12-15 x3, > 15 x1, then shuffled."""
    expanded: list[int] = []
    bins = {"below_12_x5": 0, "from_12_to_15_x3": 0, "above_15_x1": 0}
    for index in train_indices:
        copies = copy_count(float(heavy.iloc[index]))
        expanded.extend([index] * copies)
        if copies == 5:
            bins["below_12_x5"] += 1
        elif copies == 3:
            bins["from_12_to_15_x3"] += 1
        else:
            bins["above_15_x1"] += 1
    random.Random(seed).shuffle(expanded)
    bins["unique_train"] = len(train_indices)
    bins["oversampled_train_rows"] = len(expanded)
    return expanded, bins


# ====================================================================== models
@dataclass(frozen=True)
class ModelConfig:
    mode: str = "fused"  # "gnn" (gnn_only) or "fused" (hybrid)
    architecture: str = "gine"  # "gine"; "attentivefp" only as a --tune fallback
    hidden_channels: int = 128
    num_layers: int = 4
    dropout: float = 0.1
    use_distance_features: bool = True
    tabular_hidden: int = 256
    tabular_out: int = 128
    attentive_timesteps: int = 2

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "ModelConfig":
        allowed = set(cls.__dataclass_fields__)
        return cls(**{key: value for key, value in values.items() if key in allowed})


class TabularEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values)


class GINEEncoder(nn.Module):
    """Residual GINE layers with batch norm; sum + mean pooling."""

    def __init__(self, node_dim: int, edge_dim: int, hidden_channels: int, num_layers: int, dropout: float):
        super().__init__()
        self.node_projection = nn.Linear(node_dim, hidden_channels)
        self.convolutions = nn.ModuleList()
        self.normalizations = nn.ModuleList()
        self.dropout = dropout
        for _ in range(num_layers):
            update = nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels * 2),
                nn.ReLU(),
                nn.Linear(hidden_channels * 2, hidden_channels),
            )
            self.convolutions.append(GINEConv(update, edge_dim=edge_dim, train_eps=True))
            self.normalizations.append(BatchNorm(hidden_channels))

    @property
    def output_dim(self) -> int:
        return self.node_projection.out_features * 2

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        x = self.node_projection(x)
        for convolution, normalization in zip(self.convolutions, self.normalizations):
            residual = x
            x = convolution(x, edge_index, edge_attr)
            x = normalization(x)
            x = torch.relu(x + residual)
            x = nn.functional.dropout(x, p=self.dropout, training=self.training)
        return torch.cat([global_add_pool(x, batch), global_mean_pool(x, batch)], dim=-1)


class EntropyRegressor(nn.Module):
    def __init__(self, config: ModelConfig, node_dim: int, topology_edge_dim: int, full_edge_dim: int, tabular_dim: int):
        super().__init__()
        if config.mode not in {"gnn", "fused"}:
            raise ValueError(f"unsupported mode: {config.mode}")
        if config.architecture not in {"gine", "attentivefp"}:
            raise ValueError(f"unsupported architecture: {config.architecture}")
        self.config = config
        self.topology_edge_dim = topology_edge_dim
        self.edge_dim = full_edge_dim if config.use_distance_features else topology_edge_dim
        self.tabular_encoder: nn.Module | None = None
        if config.architecture == "gine":
            encoder = GINEEncoder(node_dim, self.edge_dim, config.hidden_channels, config.num_layers, config.dropout)
            graph_dim = encoder.output_dim
            self.graph_encoder: nn.Module = encoder
        else:
            graph_dim = config.hidden_channels * 2
            self.graph_encoder = AttentiveFP(
                in_channels=node_dim,
                hidden_channels=config.hidden_channels,
                out_channels=graph_dim,
                edge_dim=self.edge_dim,
                num_layers=config.num_layers,
                num_timesteps=config.attentive_timesteps,
                dropout=config.dropout,
            )
        tabular_output_dim = 0
        if config.mode == "fused":
            self.tabular_encoder = TabularEncoder(tabular_dim, config.tabular_hidden, config.tabular_out, config.dropout)
            tabular_output_dim = config.tabular_out
        self.head = nn.Sequential(
            nn.Linear(graph_dim + tabular_output_dim, 256),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, data: Any) -> torch.Tensor:
        edge_attr = data.edge_attr[:, : self.edge_dim]
        representations = [self.graph_encoder(data.x, data.edge_index, edge_attr, data.batch)]
        if self.tabular_encoder is not None:
            representations.append(self.tabular_encoder(data.tabular))
        return self.head(torch.cat(representations, dim=-1)).view(-1)


# ====================================================================== training
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def resolve_device(request: str = "auto", gpu_index: int = -1) -> torch.device:
    if request == "cpu":
        return torch.device("cpu")
    if not torch.cuda.is_available():
        if request == "cuda":
            raise RuntimeError("CUDA was requested but is unavailable. Check the NVIDIA driver and the CUDA build of PyTorch.")
        return torch.device("cpu")
    if gpu_index == -1:
        free = [int(torch.cuda.mem_get_info(i)[0]) for i in range(torch.cuda.device_count())]
        gpu_index = int(np.argmax(free))
        print(f"selected CUDA device {gpu_index} ({torch.cuda.get_device_name(gpu_index)}) as the freest GPU", flush=True)
    if gpu_index < 0 or gpu_index >= torch.cuda.device_count():
        raise ValueError(f"--gpu-index must be -1 (auto) or in [0, {torch.cuda.device_count() - 1}]")
    torch.cuda.set_device(gpu_index)
    return torch.device(f"cuda:{gpu_index}")


def regression_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    prediction = np.asarray(prediction, dtype=np.float64).reshape(-1)
    residual = prediction - target
    mae = as_python_number(np.mean(np.abs(residual)))
    rmse = as_python_number(np.sqrt(np.mean(residual**2)))
    denominator = as_python_number(np.sum((target - target.mean()) ** 2))
    r2 = as_python_number(1.0 - np.sum(residual**2) / denominator) if denominator > 0.0 else float("nan")
    if len(target) < 2 or np.std(target) == 0.0 or np.std(prediction) == 0.0:
        pearson = float("nan")
    else:
        pearson = as_python_number(pearsonr(target, prediction).statistic)
    return {"r2": float(r2), "pearson": float(pearson), "mae": float(mae), "rmse": float(rmse)}


def make_loader(graphs: Sequence[Data], indices: Sequence[int], batch_size: int, shuffle: bool, seed: int,
                pin_memory: bool = False) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader([graphs[i] for i in indices], batch_size=batch_size, shuffle=shuffle, num_workers=0,
                      generator=generator, pin_memory=pin_memory)


@torch.no_grad()
def predict_standardized(model: EntropyRegressor, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray, list[str]]:
    model.eval()
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    molecule_ids: list[str] = []
    amp_enabled = device.type == "cuda"
    for batch in loader:
        batch = batch.to(device, non_blocking=amp_enabled)
        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            prediction = model(batch)
        predictions.append(prediction.detach().float().cpu().numpy())
        targets.append(batch.y.view(-1).detach().cpu().numpy())
        ids = batch.molecule_id
        molecule_ids.extend(ids if isinstance(ids, list) else [ids])
    return np.concatenate(targets), np.concatenate(predictions), molecule_ids


def evaluate_model(model: EntropyRegressor, graphs: Sequence[Data], indices: Sequence[int], preprocessor: FeaturePreprocessor,
                   batch_size: int, device: torch.device) -> tuple[dict[str, float], np.ndarray, np.ndarray, list[str]]:
    loader = make_loader(graphs, indices, batch_size, False, 0, pin_memory=device.type == "cuda")
    target_z, prediction_z, molecule_ids = predict_standardized(model, loader, device)
    target = preprocessor.inverse_target(target_z)
    prediction = preprocessor.inverse_target(prediction_z)
    return regression_metrics(target, prediction), target, prediction, molecule_ids


@dataclass
class CandidateRun:
    name: str
    config: ModelConfig
    state_dict: dict[str, torch.Tensor]
    dev_metrics: dict[str, float]
    history: list[dict[str, float]]
    learning_rate: float = 1e-3


def train_candidate(name: str, config: ModelConfig, graphs: Sequence[Data], split_indices: dict[str, list[int]],
                    feature_schema: dict[str, int], preprocessor: FeaturePreprocessor, seed: int, device: torch.device,
                    batch_size: int = 64, max_epochs: int = 300, patience: int = 30, learning_rate: float = 1e-3,
                    weight_decay: float = 1e-5) -> CandidateRun:
    """MSE on z-scored S°, AdamW, mixed precision on CUDA; keep the epoch with the best dev Pearson r
    (ties: lower RMSE); halve the LR after 10 epochs without improvement; stop after `patience`."""
    set_seed(seed)
    model = EntropyRegressor(config=config, **feature_schema).to(device)
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=10)
    criterion = nn.MSELoss()
    pin_memory = device.type == "cuda"
    train_loader = make_loader(graphs, split_indices["train"], batch_size, True, seed, pin_memory=pin_memory)
    dev_loader = make_loader(graphs, split_indices["dev"], batch_size, False, seed, pin_memory=pin_memory)
    amp_enabled = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    best_score = -math.inf
    best_rmse = math.inf
    best_state: dict[str, torch.Tensor] | None = None
    best_dev_metrics: dict[str, float] | None = None
    wait = 0
    history: list[dict[str, float]] = []
    for epoch in range(1, max_epochs + 1):
        model.train()
        total_loss = 0.0
        count = 0
        for batch in train_loader:
            batch = batch.to(device, non_blocking=pin_memory)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                prediction = model(batch)
                loss = criterion(prediction, batch.y.view(-1))
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.item()) * int(batch.num_graphs)
            count += int(batch.num_graphs)

        target_z, prediction_z, _ = predict_standardized(model, dev_loader, device)
        dev_metrics = regression_metrics(preprocessor.inverse_target(target_z), preprocessor.inverse_target(prediction_z))
        pearson = dev_metrics["pearson"] if math.isfinite(dev_metrics["pearson"]) else -math.inf
        scheduler.step(pearson)
        improved = pearson > best_score + 1e-8 or (abs(pearson - best_score) <= 1e-8 and dev_metrics["rmse"] < best_rmse)
        epoch_row = {
            "epoch": int(epoch),
            "train_loss_standardized": as_python_number(total_loss / max(count, 1)),
            "learning_rate": as_python_number(optimizer.param_groups[0]["lr"]),
            **{f"dev_{key}": as_python_number(value) for key, value in dev_metrics.items()},
        }
        history.append(epoch_row)
        if epoch == 1 or epoch % 10 == 0:
            print(f"{name} epoch {epoch} train_loss={epoch_row['train_loss_standardized']:.4f} "
                  f"dev_pearson={dev_metrics['pearson']:.4f} dev_mae={dev_metrics['mae']:.3f}", flush=True)
        if improved:
            best_score = pearson
            best_rmse = dev_metrics["rmse"]
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            best_dev_metrics = dict(dev_metrics)
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state is None or best_dev_metrics is None:
        raise RuntimeError(f"training candidate {name!r} did not produce a finite development score")
    return CandidateRun(name, config, best_state, best_dev_metrics, history, learning_rate)


def load_run_model(run: CandidateRun, feature_schema: dict[str, int], device: torch.device) -> EntropyRegressor:
    model = EntropyRegressor(config=run.config, **feature_schema)
    model.load_state_dict(copy.deepcopy(run.state_dict))
    return model.to(device)


def select_candidate(runs: Sequence[CandidateRun]) -> CandidateRun:
    """Highest development Pearson r (ties: lower RMSE). The test split is never consulted."""
    eligible = [run for run in runs if math.isfinite(run.dev_metrics["pearson"])]
    if not eligible:
        raise RuntimeError("no candidate produced a finite development Pearson r")
    return max(eligible, key=lambda run: (run.dev_metrics["pearson"], -run.dev_metrics["rmse"]))


# ====================================================================== external scoring
def load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = EntropyRegressor(ModelConfig.from_dict(checkpoint["model_config"]), **checkpoint["feature_schema"])
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    preprocessor = FeaturePreprocessor.from_dict(checkpoint["preprocessor"])
    return {
        "model": model,
        "preprocessor": preprocessor,
        "identities": TrainingIdentities.from_dict(checkpoint.get("training_identities", {})),
        "mode": checkpoint["model_config"]["mode"],
        "predictor_columns": list(checkpoint.get("predictor_columns", preprocessor.predictor_columns)),
        "embed_3d": bool(checkpoint.get("embed_3d", True)),
        "seed": int(checkpoint.get("seed", 42)),
    }


def molecule_graphs(frame: pd.DataFrame, seed: int, embed_3d: bool) -> list[tuple[Data | None, str, str, str, str]]:
    """Per row: (graph or None, status, error, canonical SMILES, InChIKey)."""
    rows = []
    for _, row in frame.iterrows():
        smiles = str(row[SMILES_COLUMN])
        canonical, inchikey = smiles_identity(smiles) if smiles else ("", "")
        try:
            if not smiles:
                raise ValueError("missing SMILES")
            result = load_molecule_from_smiles(smiles, seed=seed, embed_3d=embed_3d)
            if result.mol is None:
                raise ValueError(result.error)
            graph = build_graph(result.mol)
            graph.molecule_id = str(row[ID_COLUMN])
            graph.smiles = smiles
            graph.canonical_smiles = result.canonical_smiles or canonical
            graph.inchikey = result.inchikey or inchikey
            rows.append((graph, result.status, "", graph.canonical_smiles, graph.inchikey))
        except Exception as exc:
            rows.append((None, "error", f"{type(exc).__name__}: {exc}", canonical, inchikey))
    return rows


def predict_frame(loaded: dict[str, Any], frame: pd.DataFrame, built: list, device: torch.device, batch_size: int,
                  entropy_scale: float = 1.0) -> pd.DataFrame:
    """Prediction table: ID, smiles, predicted_entropy, status, error, inchikey, canonical_smiles[, entropy]."""
    preprocessor = loaded["preprocessor"]
    output = pd.DataFrame(
        {
            "ID": frame[ID_COLUMN].astype(str),
            "smiles": frame[SMILES_COLUMN].astype(str),
            "predicted_entropy": np.nan,
            "status": "error",
            "error": "",
            "inchikey": "",
            "canonical_smiles": "",
        }
    )
    if TARGET_COLUMN in frame.columns:
        output["entropy"] = frame[TARGET_COLUMN].astype(float) * entropy_scale
    graphs, positions = [], []
    for position, ((_, row), (graph, status, error, canonical, inchikey)) in enumerate(zip(frame.iterrows(), built)):
        output.loc[position, "canonical_smiles"] = canonical
        output.loc[position, "inchikey"] = inchikey
        if graph is None:
            output.loc[position, "error"] = error
            continue
        try:
            graph = graph.clone()
            graph.tabular = torch.from_numpy(preprocessor.transform_frame(pd.DataFrame([row])))
        except Exception as exc:
            output.loc[position, "error"] = f"{type(exc).__name__}: {exc}"
            continue
        output.loc[position, "status"] = status
        graphs.append(graph)
        positions.append(position)
    if graphs:
        model = loaded["model"]
        predictions: list[np.ndarray] = []
        loader = DataLoader(graphs, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=device.type == "cuda")
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device, non_blocking=device.type == "cuda")
                with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
                    predictions.append(model(batch).float().cpu().numpy())
        output.loc[positions, "predicted_entropy"] = preprocessor.inverse_target(np.concatenate(predictions))
    return output


def score_predictions(output: pd.DataFrame, identities: TrainingIdentities) -> tuple[dict[str, Any], pd.DataFrame]:
    """Flag training-set overlap (ID, InChIKey or canonical SMILES) and score the rest."""
    scored = output.copy()
    scored["overlap_with_train"] = identities.overlap_mask(
        scored["ID"].astype(str).tolist(), scored["inchikey"].astype(str).tolist(),
        scored["canonical_smiles"].astype(str).tolist())
    successful = scored["predicted_entropy"].notna()
    metric_mask = successful & ~scored["overlap_with_train"]
    payload: dict[str, Any] = {
        "total_rows": int(len(scored)),
        "successful_rows": int(successful.sum()),
        "failed_rows": int((~successful).sum()),
        "overlap_rows": int(scored["overlap_with_train"].sum()),
        "metric_rows": int(metric_mask.sum()),
        "overlap_excluded": True,
    }
    if "entropy" in scored.columns and metric_mask.any():
        payload.update(regression_metrics(scored.loc[metric_mask, "entropy"].to_numpy(),
                                          scored.loc[metric_mask, "predicted_entropy"].to_numpy()))
    if "entropy" in scored.columns and successful.any():
        payload["metrics_including_overlap"] = regression_metrics(
            scored.loc[successful, "entropy"].to_numpy(), scored.loc[successful, "predicted_entropy"].to_numpy())
    return jsonable(payload), scored


def evaluate_external_csvs(checkpoints: dict[str, Path], eval_csvs: list[list[str]], output_dir: Path,
                           device: torch.device, batch_size: int) -> dict[str, Any]:
    loaded = {name: load_checkpoint(path, device) for name, path in checkpoints.items()}
    summary: dict[str, Any] = {name: {} for name in checkpoints}
    for set_name, csv_path in eval_csvs:
        frame, _ = load_hybrid_table(csv_path, require_entropy=True)
        frame = frame.reset_index(drop=True)
        graph_sets: dict[tuple[int, bool], list] = {}
        for mode_name, item in loaded.items():
            key = (item["seed"], item["embed_3d"])
            if key not in graph_sets:
                graph_sets[key] = molecule_graphs(frame, *key)
            print(f"evaluating {mode_name} on {set_name}", flush=True)
            metrics, predictions = score_predictions(
                predict_frame(item, frame, graph_sets[key], device, batch_size), item["identities"])
            metrics.update({"entropy_scale_applied": 1.0, "checkpoint_mode": item["mode"],
                            "n_predictor_columns": len(item["predictor_columns"])})
            target = output_dir / "external" / mode_name / set_name
            target.mkdir(parents=True, exist_ok=True)
            write_predictions_csv(target / "predictions.csv", predictions)
            predictions.loc[predictions["overlap_with_train"], ["ID", "smiles", "inchikey"]].to_csv(
                target / "excluded_overlap.csv", index=False)
            write_json(target / "metrics.json", metrics)
            summary[mode_name][set_name] = metrics
            print(f"  {mode_name}/{set_name}  {format_eval_metrics(metrics)}", flush=True)
    return summary


# ====================================================================== run artefacts
def environment_versions() -> dict[str, Any]:
    versions: dict[str, str] = {}
    for package in ["numpy", "torch", "torch-geometric", "rdkit", "pandas", "scikit-learn", "scipy"]:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "missing"
    return {"python": platform.python_version(), "platform": platform.platform(), "packages": versions,
            "cuda_available": torch.cuda.is_available()}


def save_dataset_artifacts(dataset: PreparedDataset, directory: Path) -> None:
    dataset.manifest.to_csv(directory / "dataset_manifest.csv", index=False)
    dataset.audit.to_csv(directory / "audit.csv", index=False)
    write_json(directory / "preprocessing.json", dataset.preprocessor.to_dict())
    write_json(directory / "feature_schema.json", dataset.feature_schema)
    write_json(directory / "training_identities.json", dataset.identities.to_dict())


def checkpoint_dict(run: CandidateRun, dataset: PreparedDataset, args: argparse.Namespace,
                    metrics_payload: dict[str, Any], accepted: bool) -> dict[str, Any]:
    return {
        "format_version": 3,
        "state_dict": run.state_dict,
        "model_config": run.config.to_dict(),
        "learning_rate": run.learning_rate,
        "feature_schema": dataset.feature_schema,
        "preprocessor": dataset.preprocessor.to_dict(),
        "predictor_columns": dataset.predictor_columns,
        "training_identities": dataset.identities.to_dict(),
        "embed_3d": args.embed_3d,
        "metrics": metrics_payload,
        "accepted": accepted,
        "seed": args.seed,
    }


# ====================================================================== candidates
def tuning_candidates(base_dropout: float, base_layers: int, use_distances: bool):
    """The run5 recipe. Yields (stage, name, ModelConfig, learning rate)."""
    yield "baseline", "gnn_only", ModelConfig(mode="gnn", num_layers=base_layers, dropout=base_dropout,
                                              use_distance_features=use_distances), 1e-3
    yield "baseline", "hybrid", ModelConfig(mode="fused", num_layers=base_layers, dropout=base_dropout,
                                            use_distance_features=use_distances), 1e-3


GNN_GRID = ((128, 3, 0.0, True), (128, 3, 0.0, False), (128, 4, 0.0, True), (128, 5, 0.0, True),
            (256, 3, 0.0, True), (256, 4, 0.0, True), (256, 5, 0.0, True))
FOCUSED_FUSED_GRID = ((128, 1, 0.0, True), (128, 3, 0.0, True), (128, 4, 0.0, True), (128, 5, 0.0, True),
                      (256, 1, 0.0, True), (256, 3, 0.0, True), (256, 4, 0.0, True), (256, 5, 0.0, True))
BROAD_FUSED_GRID = tuple((h, l, d, x) for h in (128, 256) for l in (1, 3, 5) for d in (0.0, 0.15) for x in (False, True))


def winners_from_run(source_run: Path) -> list[tuple[str, ModelConfig, float]]:
    metrics = json.loads((source_run / "metrics.json").read_text())
    found = []
    for name in ("gnn_only", "hybrid"):
        winner = metrics["mode_winners"][name]
        # Old train.py runs do not record the learning rate; their winners used 1e-3 (run5_102926 did).
        found.append((name, ModelConfig.from_dict(winner["model_config"]), float(winner.get("learning_rate", 1e-3))))
    return found


# ====================================================================== main
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", type=Path, required=True, help="Training CSV with ID, SMILES, entropy and descriptors.")
    p.add_argument("--output-dir", type=Path, required=True, help="New run folder.")
    p.add_argument("--feature-file", type=Path, default=None,
                   help="Hybrid descriptors: one column per line (e.g. selected_features). Default: every descriptor column.")
    p.add_argument("--tune", action="store_true",
                   help="run5 recipe: after the two baselines, train the GINE hyperparameter grids "
                        "(same as train.py --always-tune).")
    p.add_argument("--source-run", type=Path, default=None,
                   help="run6 recipe: retrain the gnn_only and hybrid winners recorded in <run>/metrics.json.")
    p.add_argument("--oversample", choices=["none", "3seg"], default="none",
                   help="3seg: copy training rows by heavy-atom count (<12 x5, 12-15 x3, >15 x1). Dev/test stay unique.")
    p.add_argument("--eval-csv", action="append", nargs=2, metavar=("NAME", "CSV"), default=[],
                   help="Labelled external CSV scored with both models after training. Repeatable.")
    p.add_argument("--seed", type=int, default=42, help="Split, sampling, embedding and training seed.")
    p.add_argument("--sample-limit", type=int, default=0, help="Use only this many molecules (0 = all); for quick tests.")
    p.add_argument("--max-epochs", type=int, default=300)
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--embed-3d", action=argparse.BooleanOptionalAction, default=True,
                   help="ETKDG 3D coordinates for bond-distance features (--no-embed-3d: 2D layout).")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--gpu-index", type=int, default=-1, help="PyTorch CUDA index; -1 picks the GPU with most free memory.")
    p.add_argument("--cache-dir", type=Path, default=None,
                   help="Folder for prepared-graph caches; reused when the CSV, seed, sample limit, 3D flag and features match.")
    p.add_argument("--overwrite", action="store_true", help="Allow an output folder that is not empty.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.tune and args.source_run is not None:
        raise SystemExit("--tune and --source-run are alternatives; pass one or neither")
    if args.sample_limit < 0:
        raise SystemExit("--sample-limit must be nonnegative")
    out = args.output_dir
    if out.exists() and any(out.iterdir()) and not args.overwrite:
        raise SystemExit(f"{out} is not empty. Choose a new --output-dir, or pass --overwrite.")
    if args.source_run is not None and not (args.source_run / "metrics.json").is_file():
        raise SystemExit(f"{args.source_run / 'metrics.json'} not found")
    for _, csv_path in args.eval_csv:
        if not Path(csv_path).is_file():
            raise SystemExit(f"missing --eval-csv file {csv_path}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "histories").mkdir(exist_ok=True)
    args.feature_columns = read_feature_list(args.feature_file) if args.feature_file is not None else None
    device = resolve_device(args.device, args.gpu_index)
    if device.type == "cuda":
        print(f"device {device} ({torch.cuda.get_device_name(device.index)})", flush=True)
    fixed = winners_from_run(args.source_run) if args.source_run is not None else None

    input_hashes = {"csv": sha256_file(args.csv)}
    if args.feature_file is not None:
        input_hashes["feature_file"] = sha256_file(args.feature_file)
    write_json(out / "run_config.json", {
        "csv": str(args.csv.resolve()),
        "output_dir": str(out.resolve()),
        "seed": args.seed,
        "sample_limit": args.sample_limit,
        "tune": args.tune,
        "source_run": str(args.source_run.resolve()) if args.source_run is not None else None,
        "candidates": {n: {**c.to_dict(), "learning_rate": lr} for n, c, lr in fixed} if fixed else None,
        "oversample": args.oversample,
        "selection_objective": "pearson",
        "pearson_target": PEARSON_TARGET,
        "max_epochs": args.max_epochs,
        "patience": args.patience,
        "batch_size": args.batch_size,
        "device": str(device),
        "gpu_index": args.gpu_index,
        "embed_3d": args.embed_3d,
        "cache_dir": str(args.cache_dir) if args.cache_dir is not None else None,
        "eval_csv": args.eval_csv,
        "feature_file": str(args.feature_file.resolve()) if args.feature_file is not None else None,
        "n_selected_features": len(args.feature_columns) if args.feature_columns is not None else None,
        "input_hashes": input_hashes,
        "script": "train_GINE.py",
    })
    write_json(out / "environment.json", environment_versions())

    dataset = load_or_prepare_dataset(args)
    save_dataset_artifacts(dataset, out)
    print(f"prepared {len(dataset.graphs)} SMILES graphs; predictors={len(dataset.predictor_columns)}, "
          f"constant={len(dataset.preprocessor.constant_columns)}, retained={len(dataset.preprocessor.retained_columns)}",
          flush=True)

    train_split = dict(dataset.split_indices)
    oversampling = None
    if args.oversample == "3seg":
        heavy = heavy_atom_counts(dataset.source_frame, args.csv)
        train_split["train"], oversampling = oversample_train(heavy, dataset.split_indices["train"], args.seed)
        write_json(out / "oversampling.json", oversampling)
        print(f"oversampled train rows {oversampling['unique_train']} -> {oversampling['oversampled_train_rows']} "
              f"(<12 x5: {oversampling['below_12_x5']}, 12-15 x3: {oversampling['from_12_to_15_x3']}, "
              f">15 x1: {oversampling['above_15_x1']})", flush=True)

    def run_candidate(name: str, config: ModelConfig, learning_rate: float) -> CandidateRun:
        print(f"training {name}: {config} monitor=pearson loss=mse lr={learning_rate}", flush=True)
        run = train_candidate(name=name, config=config, graphs=dataset.graphs, split_indices=train_split,
                              feature_schema=dataset.feature_schema, preprocessor=dataset.preprocessor,
                              seed=args.seed, device=device, batch_size=args.batch_size,
                              max_epochs=args.max_epochs, patience=args.patience, learning_rate=learning_rate)
        pd.DataFrame(run.history).to_csv(out / "histories" / f"{run.name}.csv", index=False, float_format="%.6g")
        print(f"{name} development metrics: {format_metrics(run.dev_metrics)}", flush=True)
        return run

    if fixed is not None:
        gnn_runs = [run_candidate(*fixed[0])]
        fused_runs = [run_candidate(*fixed[1])]
    else:
        use_distances = args.embed_3d
        gnn_runs = [run_candidate("gnn_only", ModelConfig(mode="gnn", num_layers=4, dropout=0.1,
                                                          use_distance_features=use_distances), 1e-3)]
        fused_runs = [run_candidate("hybrid", ModelConfig(mode="fused", num_layers=4, dropout=0.1,
                                                          use_distance_features=use_distances), 1e-3)]
    diagnostic_runs = [*gnn_runs, *fused_runs]

    if args.tune:
        # Focused grid when the hybrid baseline already reaches dev r >= 0.95, broad grid otherwise.
        focused = max(run.dev_metrics["pearson"] for run in fused_runs) >= PEARSON_TARGET
        if focused:
            for hidden, layers, dropout, distances in GNN_GRID:
                gnn_runs.append(run_candidate(
                    f"gnn_h{hidden}_l{layers}_d{dropout:g}_xyz{int(distances)}",
                    ModelConfig(mode="gnn", hidden_channels=hidden, num_layers=layers, dropout=dropout,
                                use_distance_features=distances), 1e-3))
        for hidden, layers, dropout, distances in (FOCUSED_FUSED_GRID if focused else BROAD_FUSED_GRID):
            fused_runs.append(run_candidate(
                f"gine_h{hidden}_l{layers}_d{dropout:g}_xyz{int(distances)}",
                ModelConfig(mode="fused", architecture="gine", hidden_channels=hidden, num_layers=layers,
                            dropout=dropout, use_distance_features=distances), 1e-3))
        if focused:
            fused_runs.append(run_candidate(
                "gine_h128_l3_d0_xyz1_lr3e4",
                ModelConfig(mode="fused", architecture="gine", hidden_channels=128, num_layers=3, dropout=0.0,
                            use_distance_features=True), 3e-4))
        if max(run.dev_metrics["pearson"] for run in fused_runs) < PEARSON_TARGET:
            for hidden in (128, 256):
                for layers in (3, 5):
                    fused_runs.append(run_candidate(
                        f"attentivefp_h{hidden}_l{layers}",
                        ModelConfig(mode="fused", architecture="attentivefp", hidden_channels=hidden,
                                    num_layers=layers, dropout=0.1, use_distance_features=True), 1e-3))

    named_runs = {"gnn_only": select_candidate(gnn_runs), "hybrid": select_candidate(fused_runs)}
    best = select_candidate(list(named_runs.values()))
    best_mode = next(name for name, run in named_runs.items() if run is best)

    mode_metrics: dict[str, dict[str, dict[str, float]]] = {}
    for name, run in named_runs.items():
        model = load_run_model(run, dataset.feature_schema, device)
        mode_metrics[name] = {}
        frames = []
        for split in ("train", "dev", "test"):  # unique molecules, also for oversampled runs
            metrics, target, prediction, molecule_ids = evaluate_model(
                model, dataset.graphs, dataset.split_indices[split], dataset.preprocessor, args.batch_size, device)
            mode_metrics[name][split] = metrics
            frames.append(pd.DataFrame({"ID": molecule_ids, "split": split, "entropy": target,
                                        "predicted_entropy": prediction, "residual": prediction - target}))
        write_predictions_csv(out / f"{name}_predictions.csv", pd.concat(frames, ignore_index=True))

    test_metrics = mode_metrics[best_mode]["test"]
    accepted = bool(test_metrics["pearson"] >= PEARSON_TARGET)
    all_candidates = [*gnn_runs, *fused_runs]
    metrics_payload = {
        "selected_candidate": best.name,
        "selected_mode": best_mode,
        "model_config": best.config.to_dict(),
        "selection_objective": "pearson",
        "selected_dev_metrics": round_metrics(best.dev_metrics),
        "accepted": accepted,
        "pearson_target": PEARSON_TARGET,
        "oversampling": oversampling,
        "splits": {split: round_metrics(values) for split, values in mode_metrics[best_mode].items()},
        "mode_winners": {
            name: {"candidate": run.name, "model_config": run.config.to_dict(), "learning_rate": run.learning_rate,
                   "dev_metrics": round_metrics(run.dev_metrics)}
            for name, run in named_runs.items()
        },
        "modes": {name: {split: round_metrics(v) for split, v in splits.items()} for name, splits in mode_metrics.items()},
        "diagnostics": {run.name: round_metrics(run.dev_metrics) for run in diagnostic_runs},
    }
    pd.DataFrame(
        [{"candidate": run.name, "model_config": json.dumps(run.config.to_dict(), sort_keys=True),
          "learning_rate": run.learning_rate, **round_metrics(run.dev_metrics)} for run in all_candidates]
    ).to_csv(out / "candidate_metrics.csv", index=False, float_format="%.6g")
    write_json(out / "metrics.json", metrics_payload)

    checkpoints = {"gnn_only": out / "gnn_only.pt", "hybrid": out / "hybrid.pt"}
    for name, path in checkpoints.items():
        torch.save(checkpoint_dict(named_runs[name], dataset, args, metrics_payload, accepted), path)
    torch.save(checkpoint_dict(best, dataset, args, metrics_payload, accepted), out / "best_model.pt")

    print(f"selected {best_mode} / {best.name} ({'accepted' if accepted else 'rejected'})", flush=True)
    for split in ("train", "dev", "test"):
        print(f"  {split:<5} {format_metrics(metrics_payload['splits'][split])}", flush=True)
    print("mode winners:", flush=True)
    for name, run in named_runs.items():
        print(f"  {name:<14} {run.name}  test {format_metrics(mode_metrics[name]['test'])}", flush=True)
    if not accepted:
        print(f"REJECTED: held-out test Pearson {test_metrics['pearson']:.6f} is below {PEARSON_TARGET:.6f}", flush=True)

    if args.eval_csv:
        external = evaluate_external_csvs(checkpoints, args.eval_csv, out, device, args.batch_size)
        write_json(out / "external_metrics.json", external)


if __name__ == "__main__":
    main()
