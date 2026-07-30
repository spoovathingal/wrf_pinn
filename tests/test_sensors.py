"""Value-verification tests for the sensor-data reader.

These confirm that ``read_sensor_data`` loads a normalized CSV into the correct
shape, column metadata, and per-point values. Sensor data has three velocity
targets and no density. Because each value column is a distinct function of the
coordinates (``u=x, v=y, w=z``), the value check also proves the
coordinate-to-value mapping, row order, and column order in one assertion.

The sensor-specific ``time_index`` and ``unique_times`` helpers are covered too.
"""

from __future__ import annotations

import numpy as np
import pytest

from wrf_pinn.config.sensors_data import SensorDataConfig
from wrf_pinn.data.sensors import read_sensor_data


def test_sensor_shape(sensor_csv):
    """Coordinates are (n, 4) and targets are (n, 3) — no density column."""

    path, rows = sensor_csv
    data = read_sensor_data(SensorDataConfig(path=str(path)))

    assert data.coordinates.shape == (len(rows), 4)
    assert data.targets.shape == (len(rows), 3)


def test_sensor_metadata(sensor_csv):
    """Coordinate and target names are read in the expected order."""

    path, rows = sensor_csv
    data = read_sensor_data(SensorDataConfig(path=str(path)))

    assert data.coordinate_names == ("x", "y", "z", "t")
    assert data.target_names == ("u", "v", "w")


def test_sensor_values(sensor_csv, report):
    """Every coordinate and target value matches what was written."""

    path, rows = sensor_csv
    data = read_sensor_data(SensorDataConfig(path=str(path)))

    expected_coordinates = np.array(
        [[row["x"], row["y"], row["z"], row["t"]] for row in rows]
    )
    expected_targets = np.array(
        [[row["u"], row["v"], row["w"]] for row in rows]
    )

    np.testing.assert_allclose(data.coordinates, expected_coordinates)
    np.testing.assert_allclose(data.targets, expected_targets)

    report(
        f"sensor values ({len(rows)} points): u=x, v=y, w=z (no rho)",
        coordinates=data.coordinates,
        targets=data.targets,
    )


def test_sensor_time_index(sensor_csv):
    """``time_index`` points at the time coordinate column (t is column 3)."""

    path, rows = sensor_csv
    data = read_sensor_data(SensorDataConfig(path=str(path)))

    assert data.time_index == 3
    assert data.coordinate_names[data.time_index] == "t"


def test_sensor_unique_times(sensor_csv, report):
    """``unique_times`` returns sorted, deduplicated time values.

    The fixture grid repeats each time across several rows, so this verifies the
    dedup-and-sort logic rather than just echoing the raw column.
    """

    path, rows = sensor_csv
    data = read_sensor_data(SensorDataConfig(path=str(path)))

    expected = np.unique([row["t"] for row in rows])
    np.testing.assert_allclose(data.unique_times, expected)
    # sanity: fewer unique times than rows (i.e. dedup actually happened)
    assert data.unique_times.size < len(rows)

    report(
        f"sensor unique_times (deduped + sorted from {len(rows)} rows)",
        unique_times=data.unique_times,
    )


def test_sensor_missing_column_raises(tmp_path):
    """A CSV missing a required velocity column is rejected."""

    path = tmp_path / "missing_w.csv"
    path.write_text("x,y,z,t,u,v\n0,0,0,0,1,2\n")

    with pytest.raises(ValueError, match="Missing sensor columns"):
        read_sensor_data(SensorDataConfig(path=str(path)))


def test_sensor_empty_file_raises(tmp_path):
    """A header-only CSV with no data rows is rejected."""

    path = tmp_path / "empty.csv"
    path.write_text("x,y,z,t,u,v,w\n")

    with pytest.raises(ValueError, match="no data rows"):
        read_sensor_data(SensorDataConfig(path=str(path)))


def test_sensor_nan_raises(tmp_path):
    """A NaN value is rejected instead of silently poisoning training."""

    path = tmp_path / "nan.csv"
    path.write_text("x,y,z,t,u,v,w\n1,1,1,1,NaN,1,1\n")

    with pytest.raises(ValueError, match="Non-finite"):
        read_sensor_data(SensorDataConfig(path=str(path)))


def test_sensor_inf_raises(tmp_path):
    """An infinite value is rejected."""

    path = tmp_path / "inf.csv"
    path.write_text("x,y,z,t,u,v,w\n1,1,1,1,inf,1,1\n")

    with pytest.raises(ValueError, match="Non-finite"):
        read_sensor_data(SensorDataConfig(path=str(path)))
