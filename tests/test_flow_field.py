"""Value-verification tests for the dense flow-field reader.

These confirm that ``read_flow_field`` loads a normalized CSV into the correct
shape, column metadata, and per-point values. Because each value column is a
distinct function of the coordinates (``u=x, v=y, w=z, rho=t``), the value check
also proves the coordinate-to-value mapping, the row order, and the column order
in a single assertion.
"""

from __future__ import annotations

import numpy as np
import pytest

from wrf_pinn.config.flow_field_data import FlowFieldDataConfig
from wrf_pinn.data.flow_field import read_flow_field


def test_flow_field_shape(flow_csv):
    """Coordinates are (n, 4) and targets are (n, 4)."""

    path, rows = flow_csv
    data = read_flow_field(FlowFieldDataConfig(path=str(path)))

    assert data.coordinates.shape == (len(rows), 4)
    assert data.targets.shape == (len(rows), 4)


def test_flow_field_metadata(flow_csv):
    """Coordinate and target names are read in the expected order.

    These names are consumed downstream (live monitor, results writer), so this
    is a metadata check rather than a check of the coordinate-to-value mapping.
    """

    path, rows = flow_csv
    data = read_flow_field(FlowFieldDataConfig(path=str(path)))

    assert data.coordinate_names == ("x", "y", "z", "t")
    assert data.target_names == ("u", "v", "w", "rho")


def test_flow_field_values(flow_csv):
    """Every coordinate and target value matches what was written.

    Exact-write / immediate-read makes a float32 round trip; ``assert_allclose``
    is used rather than ``==`` following scientific-Python convention.
    """

    path, rows = flow_csv
    data = read_flow_field(FlowFieldDataConfig(path=str(path)))

    expected_coordinates = np.array(
        [[row["x"], row["y"], row["z"], row["t"]] for row in rows]
    )
    expected_targets = np.array(
        [[row["u"], row["v"], row["w"], row["rho"]] for row in rows]
    )

    np.testing.assert_allclose(data.coordinates, expected_coordinates)
    np.testing.assert_allclose(data.targets, expected_targets)


def test_flow_field_missing_column_raises(tmp_path):
    """A CSV missing a required target column is rejected."""

    path = tmp_path / "missing_rho.csv"
    path.write_text("x,y,z,t,u,v,w\n0,0,0,0,1,2,3\n")

    with pytest.raises(ValueError, match="Missing columns"):
        read_flow_field(FlowFieldDataConfig(path=str(path)))


def test_flow_field_empty_file_raises(tmp_path):
    """A header-only CSV with no data rows is rejected."""

    path = tmp_path / "empty.csv"
    path.write_text("x,y,z,t,u,v,w,rho\n")

    with pytest.raises(ValueError, match="no data rows"):
        read_flow_field(FlowFieldDataConfig(path=str(path)))
