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
#: HRRR is the anchor condition, not a data source, so it has no code here.
SRC_SIM = 0         # LES (LASSO)
SRC_SENSOR = 1      # ground observations


@dataclass(frozen=True)
class Case:
    """One normalized case: coordinates, targets, per-entry target mask, source tag.

    Arrays are numpy (host read) or torch tensors (GPU read path).
    """

    coordinates: object   # np.ndarray or torch.Tensor
    targets: object
    target_mask: object
    source: object
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
    device: object | None = None,
) -> Case:
    """Read one ``.npy`` case, slicing columns by name per ``metadata``.

    ``coordinates``/``targets`` name the input vs. supervised columns. ``device``
    (a torch device) runs the build on the GPU instead of host numpy; the raw
    array is moved once and the returned ``Case`` holds device tensors.
    """

    path = Path(npy_path)
    if not path.exists():
        raise FileNotFoundError(f"Case file not found: {path}.")

    index = {name: i for i, name in enumerate(metadata.columns)}
    _require_columns(path, index, coordinates + targets + ("source",))
    coord_cols = [index[name] for name in coordinates]
    target_cols = [index[name] for name in targets]

    raw = np.load(path)
    xp = np
    if device is not None:
        import torch
        xp = torch
        raw = torch.as_tensor(raw, device=device)   # single host->device move

    coord_array = _f32(xp, raw[:, coord_cols])
    target_array = _f32(xp, raw[:, target_cols])
    source_array = raw[:, index["source"]].astype(np.int64) if xp is np \
        else raw[:, index["source"]].to(xp.int64)

    _check_finite(path, coordinates, coord_array, xp)   # coordinates are required

    # optional targets may be NaN: mask measured entries, zero-fill the rest
    target_mask = _f32(xp, xp.isfinite(target_array))
    target_array = _f32(xp, xp.where(target_mask > 0.0, target_array, 0.0))

    return Case(
        coordinates=_contig(xp, coord_array),
        targets=_contig(xp, target_array),
        target_mask=_contig(xp, target_mask),
        source=source_array,
        coordinate_names=coordinates,
        target_names=targets,
    )


def _f32(xp, array):
    return array.astype(np.float32) if xp is np else array.to(xp.float32)


def _contig(xp, array):
    return np.ascontiguousarray(array) if xp is np else array.contiguous()


def _require_columns(path: Path, index: dict[str, int], needed: tuple[str, ...]) -> None:
    missing = sorted({name for name in needed if name not in index})
    if missing:
        raise ValueError(f"Case {path} is missing schema columns: {missing}.")


def _check_finite(path: Path, names: tuple[str, ...], array, xp=np) -> None:
    """Reject NaN/inf, which would silently poison training."""

    finite = xp.isfinite(array)
    if bool(finite.all()):
        return
    bad_cols = xp.where(~finite.all(axis=0))[0] if xp is np \
        else xp.where(~finite.all(dim=0))[0]
    bad = sorted({names[int(c)] for c in bad_cols})
    raise ValueError(f"Non-finite values in {path}, columns: {bad}.")
