"""Readers for normalized dense flow-field training data."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from wrf_pinn.config.data import ColumnSpec, FlowFieldDataConfig


@dataclass(frozen=True)
class FlowFieldData:
    """Normalized coordinate and target arrays for flow-field training."""

    coordinates: np.ndarray
    targets: np.ndarray
    coordinate_names: tuple[str, ...]
    target_names: tuple[str, ...]

    @property
    def n_points(self) -> int:
        """Return the number of flow-field samples."""

        return self.coordinates.shape[0]

    @property
    def input_dim(self) -> int:
        """Return the number of coordinate columns."""

        return self.coordinates.shape[1]

    @property
    def target_dim(self) -> int:
        """Return the number of target columns."""

        return self.targets.shape[1]

    def as_torch(self, *, dtype: object | None = None, device: object | None = None):
        """Return coordinates and targets as torch tensors.

        PyTorch is imported lazily so this module can still be imported in
        environments where only NumPy-based data inspection is available.
        """

        import torch

        tensor_dtype = dtype if dtype is not None else torch.float32
        coordinates = torch.as_tensor(
            self.coordinates,
            dtype=tensor_dtype,
            device=device,
        )
        targets = torch.as_tensor(
            self.targets,
            dtype=tensor_dtype,
            device=device,
        )
        return coordinates, targets


def read_flow_field(config: FlowFieldDataConfig) -> FlowFieldData:
    """Read normalized flow-field data from a configured file."""

    if config.file_format != "csv":
        raise ValueError(f"Unsupported flow-field format: {config.file_format}.")

    return read_flow_field_csv(
        path=config.path,
        columns=config.columns,
    )


def read_flow_field_csv(
    path: str | Path,
    columns: ColumnSpec = ColumnSpec(),
) -> FlowFieldData:
    """Read normalized flow-field CSV data.

    The CSV file must contain coordinate columns, by default ``x, y, z, t``, and
    target columns, by default ``u, v, w, rho``. The values are assumed to have
    already been normalized and scaled outside this package.
    """

    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Flow-field data file not found: {csv_path}.")

    coordinate_rows: list[list[float]] = []
    target_rows: list[list[float]] = []

    with csv_path.open(newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None:
            raise ValueError(f"Flow-field CSV has no header: {csv_path}.")

        _validate_columns(csv_path, reader.fieldnames, columns)

        for row in reader:
            coordinate_rows.append([float(row[name]) for name in columns.coordinates])
            target_rows.append([float(row[name]) for name in columns.values])

    if not coordinate_rows:
        raise ValueError(f"Flow-field CSV contains no data rows: {csv_path}.")

    return FlowFieldData(
        coordinates=np.asarray(coordinate_rows, dtype=np.float32),
        targets=np.asarray(target_rows, dtype=np.float32),
        coordinate_names=columns.coordinates,
        target_names=columns.values,
    )


def _validate_columns(
    path: Path,
    fieldnames: list[str],
    columns: ColumnSpec,
) -> None:
    required = set(columns.coordinates) | set(columns.values)
    available = set(fieldnames)
    missing = sorted(required - available)
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}.")
