"""Readers for boundary geometry data."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from wrf_pinn.config.boundary_data import WallSurfaceConfig


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
