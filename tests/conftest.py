"""Shared fixtures and a friendly test-summary hook for the WRF-PINN tests.

Fixtures build **distinguishable** input data in a temporary directory: the
coordinate grid is asymmetric (each axis uses different values, every row is a
unique tuple) and each value column is a distinct function of the coordinates.
This makes a shuffled row, a dropped column, or a swapped axis fail immediately,
which constant data would not catch.

All files are written under pytest's ``tmp_path``, which pytest removes
automatically after each test, so nothing persists in the repo or working tree.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest


def _distinct_grid() -> list[tuple[float, float, float, float]]:
    """Return an asymmetric grid of unique (x, y, z, t) coordinate tuples.

    Each axis uses different values so that swapping two coordinate columns is
    detectable, and every row is unique so row order is verifiable.
    """

    xs = (1.0, 3.0)
    ys = (2.0, 5.0)
    zs = (7.0,)
    ts = (11.0, 13.0)
    return [(x, y, z, t) for x in xs for y in ys for z in zs for t in ts]


def _write_csv(
    path: Path,
    coordinate_columns: tuple[str, ...],
    value_columns: tuple[str, ...],
    rows: list[dict[str, float]],
) -> None:
    """Write ``rows`` to ``path`` with a header of coordinate + value columns."""

    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow((*coordinate_columns, *value_columns))
        for row in rows:
            writer.writerow(
                [row[name] for name in coordinate_columns]
                + [row[name] for name in value_columns]
            )


@pytest.fixture
def report():
    """Return a function that prints the actual verified values for a test.

    Output is captured by pytest by default, so it stays hidden on a plain
    ``pytest -v`` run and appears when output is shown (``pytest -v -s`` or the
    per-test summary from ``pytest -rA``). This surfaces the real numbers a test
    confirmed — not just a description — without cluttering the default output.

    Pass named arrays (or any values) to print them. Arrays are shown in full up
    to ``max_rows`` rows, then truncated with a count so large datasets stay
    readable. Example::

        def test_x(report):
            data = read_...(...)
            report("flow-field", max_rows=6,
                   coordinates=data.coordinates, targets=data.targets)
    """

    def _format(value: object, max_rows: int) -> str:
        array = np.asarray(value)
        if array.ndim >= 1 and array.shape[0] > max_rows:
            shown = np.array2string(array[:max_rows], separator=", ")
            return f"{shown}  … (+{array.shape[0] - max_rows} more rows)"
        return np.array2string(array, separator=", ")

    def _report(label: str, *, max_rows: int = 8, **arrays: object) -> None:
        print(f"\n  verified: {label}")
        for name, value in arrays.items():
            text = _format(value, max_rows).replace("\n", "\n      ")
            print(f"    {name} = {text}")

    return _report


@pytest.fixture
def flow_csv(tmp_path: Path):
    """Write a distinguishable flow-field CSV and return (path, rows).

    Values are derived from coordinates so the coordinate-to-value mapping is
    verifiable: ``u = x``, ``v = y``, ``w = z``, ``rho = t``.
    """

    rows = [
        {"x": x, "y": y, "z": z, "t": t, "u": x, "v": y, "w": z, "rho": t}
        for (x, y, z, t) in _distinct_grid()
    ]
    path = tmp_path / "flow.csv"
    _write_csv(path, ("x", "y", "z", "t"), ("u", "v", "w", "rho"), rows)
    return path, rows


@pytest.fixture
def sensor_csv(tmp_path: Path):
    """Write a distinguishable sensor CSV and return (path, rows).

    Sensor data has three velocity targets and no density. Values are derived
    from coordinates so the mapping is verifiable: ``u = x``, ``v = y``,
    ``w = z``.
    """

    rows = [
        {"x": x, "y": y, "z": z, "t": t, "u": x, "v": y, "w": z}
        for (x, y, z, t) in _distinct_grid()
    ]
    path = tmp_path / "sensor.csv"
    _write_csv(path, ("x", "y", "z", "t"), ("u", "v", "w"), rows)
    return path, rows


# Constant uniform-flow state for the training-case tests. A constant field is
# the physically correct fixture here: it is the case Savio specified, and its
# PDE residual is zero, so a model that learns it drives the loss down.
UNIFORM_STATE = {"u": 5.0, "v": 2.0, "w": 0.0, "rho": 1.225}


def _uniform_grid() -> list[tuple[float, float, float, float]]:
    """Return a small but non-trivial (x, y, z, t) grid for training tests."""

    axis = (0.0, 0.25, 0.5, 0.75, 1.0)
    return [(x, y, z, t) for x in axis for y in axis for z in axis for t in axis]


@pytest.fixture
def uniform_flow_csv(tmp_path: Path):
    """Write a constant uniform flow-field CSV and return (path, rows).

    Every point has the same velocity and density (``UNIFORM_STATE``). Used by
    the training-case tests, which check that a training run on this data
    completes and that the loss decreases.
    """

    rows = [
        {"x": x, "y": y, "z": z, "t": t, **UNIFORM_STATE}
        for (x, y, z, t) in _uniform_grid()
    ]
    path = tmp_path / "uniform_flow.csv"
    _write_csv(path, ("x", "y", "z", "t"), ("u", "v", "w", "rho"), rows)
    return path, rows
