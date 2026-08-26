"""Reader for pre-processor ``.npy`` cases (schema ``x,y,z,t,u,v,w,theta,p_prime,source``).

Coordinates are required (never NaN); optional targets may be NaN, in which case
the row is kept, the NaN zero-filled, and a per-entry ``target_mask`` records
which targets are real so the masked loss supervises only those.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

#: Default column split (matches the pre-processor schema).
DEFAULT_COORDINATES: tuple[str, ...] = ("x", "y", "z", "t")
DEFAULT_TARGETS: tuple[str, ...] = ("u", "v", "w")

#: Source category codes in the ``source`` column (from the pre-processor).
SRC_INLET = 0       # HRRR
SRC_SIM = 1         # LES (LASSO)
SRC_SENSOR = 2      # ground observations


@dataclass(frozen=True)
class Case:
    """One normalized case: coordinates, targets, per-entry target mask, source tag."""

    coordinates: np.ndarray
    targets: np.ndarray
    target_mask: np.ndarray
    source: np.ndarray
    coordinate_names: tuple[str, ...]
    target_names: tuple[str, ...]

    @property
    def n_points(self) -> int:
        return self.coordinates.shape[0]

    @property
    def input_dim(self) -> int:
        return self.coordinates.shape[1]

    @property
    def target_dim(self) -> int:
        return self.targets.shape[1]

    def as_torch(self, *, dtype: object | None = None, device: object | None = None):
        """Return coordinates, targets, and target mask as torch tensors."""

        import torch

        tensor_dtype = dtype if dtype is not None else torch.float32
        coordinates = torch.as_tensor(self.coordinates, dtype=tensor_dtype, device=device)
        targets = torch.as_tensor(self.targets, dtype=tensor_dtype, device=device)
        target_mask = torch.as_tensor(self.target_mask, dtype=tensor_dtype, device=device)
        return coordinates, targets, target_mask


@dataclass(frozen=True)
class CaseMetadata:
    """The shared sidecar: column order, source-code map, normalization recipe."""

    columns: tuple[str, ...]
    source_codes: dict[int, str]
    normalization: dict

    @classmethod
    def load(cls, path: Path) -> "CaseMetadata":
        raw = json.loads(Path(path).read_text())
        schema = raw["schema"]
        codes = {int(k): v for k, v in schema.get("source_codes", {}).items()}
        return cls(
            columns=tuple(schema["columns"]),
            source_codes=codes,
            normalization=raw.get("normalization", {}),
        )


def read_case(
    npy_path: str | Path,
    metadata: CaseMetadata,
    *,
    coordinates: tuple[str, ...] = DEFAULT_COORDINATES,
    targets: tuple[str, ...] = DEFAULT_TARGETS,
) -> Case:
    """Read one ``.npy`` case, slicing columns by name per ``metadata``.

    ``coordinates`` and ``targets`` name which schema columns are model inputs
    vs. supervised outputs; they are not hardcoded so the split can change with
    later decisions.
    """

    path = Path(npy_path)
    if not path.exists():
        raise FileNotFoundError(f"Case file not found: {path}.")

    data = np.load(path)
    index = {name: i for i, name in enumerate(metadata.columns)}
    _require_columns(path, index, coordinates + targets + ("source",))

    coord_cols = [index[name] for name in coordinates]
    target_cols = [index[name] for name in targets]

    coord_array = np.ascontiguousarray(data[:, coord_cols], dtype=np.float32)
    target_array = np.ascontiguousarray(data[:, target_cols], dtype=np.float32)
    source_array = data[:, index["source"]].astype(np.int64)

    _check_finite(path, coordinates, coord_array)   # coordinates are required

    # optional targets may be NaN: mask measured entries, zero-fill the rest
    target_mask = np.isfinite(target_array).astype(np.float32)
    target_array = np.where(target_mask > 0.0, target_array, 0.0).astype(np.float32)

    return Case(
        coordinates=coord_array,
        targets=np.ascontiguousarray(target_array),
        target_mask=np.ascontiguousarray(target_mask),
        source=source_array,
        coordinate_names=coordinates,
        target_names=targets,
    )


def _require_columns(path: Path, index: dict[str, int], needed: tuple[str, ...]) -> None:
    missing = sorted({name for name in needed if name not in index})
    if missing:
        raise ValueError(f"Case {path} is missing schema columns: {missing}.")


def _check_finite(path: Path, names: tuple[str, ...], array: np.ndarray) -> None:
    """Reject NaN/inf, which would silently poison training."""

    if np.isfinite(array).all():
        return
    bad = sorted({names[c] for c in np.unique(np.where(~np.isfinite(array))[1])})
    raise ValueError(f"Non-finite values in {path}, columns: {bad}.")
