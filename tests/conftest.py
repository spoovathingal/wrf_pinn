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
