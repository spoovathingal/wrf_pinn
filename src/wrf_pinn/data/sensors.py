"""Readers for normalized sensor data."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from wrf_pinn.config.sensors_data import SensorColumnConfig, SensorDataConfig


@dataclass(frozen=True)
class SensorData:
    """Normalized sensor coordinates and velocity targets."""

    coordinates: np.ndarray
    targets: np.ndarray
    coordinate_names: tuple[str, str, str, str]
    target_names: tuple[str, str, str]

    @property
    def n_points(self) -> int:
        """Return the number of sensor samples."""

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
        """Return coordinates and targets as torch tensors."""

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


def read_sensor_data(config: SensorDataConfig) -> SensorData:
    """Read normalized sensor data from the configured source."""

    if config.file_format != "csv":
        raise ValueError(f"Unsupported sensor data format: {config.file_format}.")

    return read_sensor_data_csv(
        path=config.path,
        columns=config.columns,
    )


def read_sensor_data_csv(
    path: str | Path,
    columns: SensorColumnConfig = SensorColumnConfig(),
) -> SensorData:
    """Read normalized sensor ``x,y,z,t,u,v,w`` data from a CSV file."""

    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Sensor data file not found: {csv_path}.")

    coordinate_rows: list[list[float]] = []
    target_rows: list[list[float]] = []

    with csv_path.open(newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None:
            raise ValueError(f"Sensor CSV has no header: {csv_path}.")

        _validate_columns(csv_path, reader.fieldnames, columns)

        for row in reader:
            coordinate_rows.append([float(row[name]) for name in columns.coordinates])
            target_rows.append([float(row[name]) for name in columns.values])

    if not coordinate_rows:
        raise ValueError(f"Sensor CSV contains no data rows: {csv_path}.")

    return SensorData(
        coordinates=np.asarray(coordinate_rows, dtype=np.float32),
        targets=np.asarray(target_rows, dtype=np.float32),
        coordinate_names=columns.coordinates,
        target_names=columns.values,
    )


def _validate_columns(
    path: Path,
    fieldnames: list[str],
    columns: SensorColumnConfig,
) -> None:
    required = set(columns.coordinates) | set(columns.values)
    available = set(fieldnames)
    missing = sorted(required - available)
    if missing:
        raise ValueError(f"Missing sensor columns in {path}: {missing}.")
