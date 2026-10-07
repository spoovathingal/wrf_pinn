"""Readers for boundary geometry data."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from wrf_pinn.config.boundary_data import SurfaceFluxConfig, WallSurfaceConfig


@dataclass(frozen=True)
class WallSurfaceGeometry:
    """Normalized 3D wall surface geometry used to build boundary constraints."""

    coordinates: np.ndarray
    coordinate_names: tuple[str, str, str]

    @property
    def n_points(self) -> int:
        """Return the number of wall surface samples."""

        return self.coordinates.shape[0]

    def as_torch(self, *, dtype: object | None = None, device: object | None = None):
        """Return wall ``x,y,z`` surface coordinates as a torch tensor."""

        import torch

        tensor_dtype = dtype if dtype is not None else torch.float32
        return torch.as_tensor(self.coordinates, dtype=tensor_dtype, device=device)

@dataclass(frozen=True)
class SurfaceFluxData:
    """Normalized x,y,t coordinates and physical surface-flux values."""
    coordinates: np.ndarray
    fric_vel: np.ndarray
    ht_flux: np.ndarray

def read_surface_fluxes(config: SurfaceFluxConfig, *, coordinates: np.ndarray | None = None) -> SurfaceFluxData:
    """Read .npy rows ordered x,y,t,fricVel,htFlux without normalizing them."""

    if not config.path:
        raise ValueError("No surface-flux path is configured.")

    path = Path(config.path)
    if path.suffix.lower() != ".npy":
        raise ValueError("Surface-flux data must be a .npy file.")
    if not path.is_file():
        raise FileNotFoundError(f"Surface-flux file not found: {path}.")

    data = np.load(path, mmap_mode="r", allow_pickle=False)
    if not isinstance(data, np.ndarray) or data.ndim != 2 or data.shape[1] != 5:
        raise ValueError("Surface-flux data must have shape (N, 5).")
    if data.shape[0] == 0:
        raise ValueError("Surface-flux data must contain at least one row.")
    if data.dtype.kind != "f" or data.dtype.itemsize not in (4, 8):
        raise ValueError("Surface-flux data must use float32 or float64.")

    def keys(values):
        values = np.ascontiguousarray(values, dtype=np.float32)
        return values.view([("x", np.float32), ("y", np.float32), ("t", np.float32)]).reshape(-1)

    requested = None
    selected = []
    if coordinates is not None:
        coordinates = np.asarray(coordinates)
        if coordinates.ndim != 2 or coordinates.shape[1] != 3 or len(coordinates) == 0:
            raise ValueError("Requested coordinates must have shape (N, 3), N > 0.")
        if not np.isfinite(coordinates).all():
            raise ValueError("Requested coordinates must be finite.")
        requested = np.unique(keys(coordinates))
        found = np.zeros(len(requested), dtype=bool)

    for start in range(0, data.shape[0], 100_000):
        block = data[start:start + 100_000]

        # Preserve existing source-data validation.
        if not np.isfinite(block).all():
            raise ValueError("Surface-flux data contain NaN or infinity.")
        if np.any((block[:, :3] < -1.0e-6) | (block[:, :3] > 1.0 + 1.0e-6)):
            raise ValueError("Surface-flux x,y,t must use global [0, 1] normalization.")
        if np.any(block[:, 3] < 0.0):
            raise ValueError("Surface friction velocity must be nonnegative.")

        if requested is not None:
            block_keys = keys(block[:, :3])
            positions = np.searchsorted(requested, block_keys)
            matches = positions < len(requested)
            matches[matches] = (requested[positions[matches]] == block_keys[matches])
            if matches.any():
                selected.append(block[matches])  # Copies matching rows only.
                found[positions[matches]] = True

    if requested is not None:
        if not found.all():
            raise ValueError("No exact surface-flux (x,y,t) match; "
                "check coverage and shared normalization.")
        data = np.concatenate(selected)
    return SurfaceFluxData(coordinates=data[:, :3], fric_vel=data[:, 3:4], ht_flux=data[:, 4:5])

def read_wall_surface_geometry(config: WallSurfaceConfig) -> WallSurfaceGeometry:
    """Read no-slip wall surface geometry from the configured source."""

    if config.file_format != "csv":
        raise ValueError(f"Unsupported wall surface format: {config.file_format}.")

    return read_wall_surface_geometry_csv(
        path=config.path,
        coordinate_columns=config.coordinate_columns,
    )


def read_wall_surface_geometry_csv(
    path: str | Path,
    coordinate_columns: tuple[str, str, str] = ("x", "y", "z"),
) -> WallSurfaceGeometry:
    """Read normalized wall ``x,y,z`` surface geometry from a CSV file."""

    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Wall surface CSV not found: {csv_path}.")

    coordinate_rows: list[list[float]] = []

    with csv_path.open(newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None:
            raise ValueError(f"Wall surface CSV has no header: {csv_path}.")

        _validate_coordinate_columns(csv_path, reader.fieldnames, coordinate_columns)

        for row in reader:
            coordinate_rows.append([float(row[name]) for name in coordinate_columns])

    if not coordinate_rows:
        raise ValueError(f"Wall surface CSV contains no data rows: {csv_path}.")

    return WallSurfaceGeometry(
        coordinates=np.asarray(coordinate_rows, dtype=np.float32),
        coordinate_names=coordinate_columns,
    )


def _validate_coordinate_columns(
    path: Path,
    fieldnames: list[str],
    coordinate_columns: tuple[str, str, str],
) -> None:
    required = set(coordinate_columns)
    available = set(fieldnames)
    missing = sorted(required - available)
    if missing:
        raise ValueError(f"Missing wall surface coordinate columns in {path}: {missing}.")
