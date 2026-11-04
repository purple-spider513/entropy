#!/usr/bin/env python3
"""Predict gas-phase standard molar entropy (S°) with checkpoints from train_GINE.py (or train.py).

Self-contained: needs only numpy, pandas, scipy, RDKit, PyTorch and PyTorch Geometric
(the `GNN` conda env). Merges predict.py, evaluate_external.py and the entropy_gnn package.

Each checkpoint stores its model, the descriptor list chosen at training time (feature
selection), the descriptor scaling and the training-set molecules, so nothing else is needed.

Input CSVs need an ID column, SMILES and the descriptor columns used in training. When an
'entropy' column is present, R2, Pearson r, MAE and RMSE are reported. Molecules that also
occur in the training table (same ID, InChIKey or canonical SMILES) are flagged in
`overlap_with_train` and left out of the scores.

Output per checkpoint and input: <output-dir>/<checkpoint>/<name>/predictions.csv,
metrics.json and excluded_overlap.csv, plus <output-dir>/external_metrics.json.

Example (run from this folder)
  D=hybrid_data
  python predict_GINE.py --run-dir run5_102926/minimal \\
      --csv std109=$D/hybrid_std109_102926.csv --csv std128=$D/hybrid_std128_102926.csv \\
      --csv std6074=$D/hybrid_std6074_102926.csv --csv pdbbind=$D/hybrid_pdbbind_102926.csv \\
      --output-dir run5_eval --gpu-index 0
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from scipy.stats import pearsonr
from torch import nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import AttentiveFP, BatchNorm, GINEConv, global_add_pool, global_mean_pool

RDLogger.DisableLog("rdApp.*")

# ====================================================================== shared with train_GINE.py
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


# ====================================================================== prediction
def check_feature_selection(predictor_columns: list[str], feature_file: Path | None) -> None:
    """Optional safety check: the checkpoint always uses its own columns; fail if a list differs."""
    if feature_file is None:
        return
    expected = set(read_feature_list(feature_file))
    actual = set(predictor_columns)
    if expected != actual:
        raise ValueError(
            f"checkpoint was trained on {len(actual)} predictors but {feature_file} lists {len(expected)}; "
            f"only in checkpoint: {sorted(actual - expected)[:8]}, only in file: {sorted(expected - actual)[:8]}"
        )


def checkpoint_list(args: argparse.Namespace) -> dict[str, Path]:
    paths: list[Path] = list(args.checkpoint)
    for run_dir in args.run_dir:
        found = [run_dir / f"{name}.pt" for name in ("gnn_only", "hybrid") if (run_dir / f"{name}.pt").is_file()]
        if not found:
            raise SystemExit(f"no gnn_only.pt or hybrid.pt in {run_dir}")
        paths.extend(found)
    if not paths:
        raise SystemExit("pass --checkpoint and/or --run-dir")
    stems = [p.stem for p in paths]
    named: dict[str, Path] = {}
    for path in paths:
        if not path.is_file():
            raise SystemExit(f"missing checkpoint {path}")
        name = path.stem if stems.count(path.stem) == 1 else f"{path.parent.name}_{path.stem}"
        if name in named:
            raise SystemExit(f"two checkpoints would be written to the same folder name {name!r}")
        named[name] = path
    return named


def input_list(items: list[str]) -> list[tuple[str, Path]]:
    inputs = []
    for item in items:
        name, sep, path = item.partition("=")
        path_obj = Path(path) if sep else Path(item)
        name = name if sep else path_obj.stem
        if not path_obj.is_file():
            raise SystemExit(f"missing input CSV {path_obj}")
        inputs.append((name, path_obj))
    names = [n for n, _ in inputs]
    if len(set(names)) != len(names):
        raise SystemExit(f"input names must differ: {names}")
    return inputs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, action="append", default=[], help="A .pt checkpoint. Repeatable.")
    p.add_argument("--run-dir", type=Path, action="append", default=[],
                   help="A training run folder; uses its gnn_only.pt and hybrid.pt. Repeatable.")
    p.add_argument("--csv", action="append", required=True, metavar="[NAME=]CSV",
                   help="Input CSV, optionally named (std128=path.csv); default name is the file stem. Repeatable.")
    p.add_argument("--output-dir", type=Path, required=True, help="New folder for the results.")
    p.add_argument("--feature-file", type=Path, default=None,
                   help="Optional check that each checkpoint's descriptor list equals this file (e.g. selected_features).")
    p.add_argument("--entropy-scale", type=float, default=1.0,
                   help="Multiply the CSV's entropy labels by this factor before scoring (e.g. 4.184 for cal to J).")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--gpu-index", type=int, default=-1, help="PyTorch CUDA index; -1 picks the GPU with most free memory.")
    p.add_argument("--overwrite", action="store_true", help="Allow an output folder that is not empty.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not (args.entropy_scale > 0.0 and math.isfinite(args.entropy_scale)):
        raise SystemExit("--entropy-scale must be a finite positive value")
    checkpoints = checkpoint_list(args)
    inputs = input_list(args.csv)
    out = args.output_dir
    if (out / "run_config.json").exists() or any(out.glob("*.pt")):
        raise SystemExit(f"{out} is a training run folder; choose another --output-dir")
    if out.exists() and any(out.iterdir()) and not args.overwrite:
        raise SystemExit(f"{out} is not empty. Choose a new --output-dir, or pass --overwrite.")
    out.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device, args.gpu_index)
    loaded = {}
    for name, path in checkpoints.items():
        item = load_checkpoint(path, device)
        check_feature_selection(item["predictor_columns"], args.feature_file)
        print(f"{name}: {path}  mode={item['mode']}  descriptors={len(item['predictor_columns'])}", flush=True)
        loaded[name] = item

    summary: dict[str, Any] = {name: {} for name in loaded}
    for set_name, csv_path in inputs:
        frame, _ = load_hybrid_table(csv_path, require_entropy=False)
        frame = frame.reset_index(drop=True)
        graph_sets: dict[tuple[int, bool], list] = {}
        for ckpt_name, item in loaded.items():
            key = (item["seed"], item["embed_3d"])
            if key not in graph_sets:
                print(f"building graphs for {set_name} ({len(frame)} rows)", flush=True)
                graph_sets[key] = molecule_graphs(frame, *key)
            output = predict_frame(item, frame, graph_sets[key], device, args.batch_size, args.entropy_scale)
            metrics, predictions = score_predictions(output, item["identities"])
            metrics.update({"entropy_scale_applied": args.entropy_scale, "checkpoint_mode": item["mode"],
                            "n_predictor_columns": len(item["predictor_columns"]),
                            "checkpoint": str(checkpoints[ckpt_name]), "csv": str(csv_path)})
            target = out / ckpt_name / set_name
            target.mkdir(parents=True, exist_ok=True)
            write_predictions_csv(target / "predictions.csv", predictions)
            predictions.loc[predictions["overlap_with_train"], ["ID", "smiles", "inchikey"]].to_csv(
                target / "excluded_overlap.csv", index=False)
            write_json(target / "metrics.json", metrics)
            summary[ckpt_name][set_name] = metrics
            scores = format_eval_metrics(metrics) if "mae" in metrics else (
                f"n={metrics['successful_rows']}/{metrics['total_rows']}  overlap={metrics['overlap_rows']}  (no entropy labels)")
            print(f"  {ckpt_name}/{set_name}  {scores}  failed={metrics['failed_rows']}", flush=True)
    write_json(out / "external_metrics.json", summary)
    print(f"wrote {out / 'external_metrics.json'}", flush=True)


if __name__ == "__main__":
    main()
