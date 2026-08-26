"""Shared fixtures and a friendly test-summary hook for the WRF-PINN tests.

Data enters as ``.npy`` cases (the pre-processor's output) plus a shared
``metadata.json``. Fixtures write these under pytest's ``tmp_path``, which is
removed automatically after each test.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from wrf_pinn.data.case import (
    read_case, CaseMetadata, SRC_INLET, SRC_SIM, SRC_SENSOR,
)

#: Case schema (must match the pre-processor).
CASE_COLUMNS = ("x", "y", "z", "t", "u", "v", "w", "source")

#: Constant uniform-flow state used by the training-case fixtures.
UNIFORM_STATE = (5.0, 2.0, 0.0)   # u, v, w


def _uniform_grid(coordinate_range=(0.0, 1.0), n_per_axis=5):
    lo, hi = coordinate_range
    axis = [lo + (hi - lo) * i / (n_per_axis - 1) for i in range(n_per_axis)]
    return [(x, y, z, t) for x in axis for y in axis for z in axis for t in axis]


def write_case(
    out_dir: Path,
    coordinates,
    state=UNIFORM_STATE,
    source=SRC_SIM,
) -> tuple[Path, CaseMetadata]:
    """Write one ``.npy`` case + ``metadata.json`` and return (npy_path, metadata).

    Every row gets the same ``state`` (u,v,w) and ``source`` tag.
    """

    coords = np.asarray(coordinates, dtype=np.float64)
    n = coords.shape[0]
    rows = np.empty((n, len(CASE_COLUMNS)), dtype=np.float64)
    rows[:, 0:4] = coords
    rows[:, 4:7] = np.asarray(state, dtype=np.float64)
    rows[:, 7] = source

    out_dir.mkdir(parents=True, exist_ok=True)
    npy_path = out_dir / "case.npy"
    np.save(npy_path, rows)
    metadata = {
        "schema": {
            "columns": list(CASE_COLUMNS),
            "source_codes": {SRC_INLET: "inlet", SRC_SIM: "simulation",
                             SRC_SENSOR: "sensor"},
        },
        "normalization": {},
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata))
    return npy_path, CaseMetadata.load(out_dir / "metadata.json")


@pytest.fixture
def uniform_case(tmp_path: Path):
    """A dense uniform-flow case (all rows tagged 'simulation')."""

    npy, meta = write_case(tmp_path / "case", _uniform_grid(), source=SRC_SIM)
    return read_case(npy, meta)


@pytest.fixture
def uniform_case_factory(tmp_path: Path):
    """Factory: build a uniform case for a given state / domain / source tag."""

    counter = {"n": 0}

    def _make(state=UNIFORM_STATE, coordinate_range=(0.0, 1.0),
              n_per_axis=5, source=SRC_SIM):
        counter["n"] += 1
        grid = _uniform_grid(coordinate_range, n_per_axis)
        npy, meta = write_case(tmp_path / f"case_{counter['n']}", grid,
                               state=state, source=source)
        return read_case(npy, meta)

    return _make


@pytest.fixture
def multi_source_case(tmp_path: Path):
    """A case whose rows carry all three source tags (inlet / sim / sensor)."""

    grid = _uniform_grid(n_per_axis=4)
    coords = np.asarray(grid, dtype=np.float64)
    n = coords.shape[0]
    rows = np.empty((n, len(CASE_COLUMNS)), dtype=np.float64)
    rows[:, 0:4] = coords
    rows[:, 4:7] = np.asarray(UNIFORM_STATE, dtype=np.float64)
    rows[:, 7] = np.array([SRC_INLET, SRC_SIM, SRC_SENSOR])[np.arange(n) % 3]

    out = tmp_path / "multi"
    out.mkdir()
    np.save(out / "case.npy", rows)
    (out / "metadata.json").write_text(json.dumps({
        "schema": {"columns": list(CASE_COLUMNS),
                   "source_codes": {SRC_INLET: "inlet", SRC_SIM: "simulation",
                                    SRC_SENSOR: "sensor"}},
        "normalization": {},
    }))
    return read_case(out / "case.npy", CaseMetadata.load(out / "metadata.json"))


#: Widened schema with theta/p' (for the masked-target tests).
CASE_COLUMNS_THETA = ("x", "y", "z", "t", "u", "v", "w", "theta", "p_prime", "source")


@pytest.fixture
def nan_target_case(tmp_path: Path):
    """A widened case where sensor rows carry NaN theta/p' (measured only on sim)."""

    grid = _uniform_grid(n_per_axis=4)
    coords = np.asarray(grid, dtype=np.float64)
    n = coords.shape[0]
    rows = np.empty((n, len(CASE_COLUMNS_THETA)), dtype=np.float64)
    rows[:, 0:4] = coords
    rows[:, 4:7] = np.asarray(UNIFORM_STATE, dtype=np.float64)
    rows[:, 7:9] = (1.5, 20.0)                       # theta, p'
    rows[:, 9] = np.where(np.arange(n) % 2 == 0, SRC_SIM, SRC_SENSOR)
    rows[rows[:, 9] == SRC_SENSOR, 7:9] = np.nan     # sensors lack theta/p'

    out = tmp_path / "nan_target"
    out.mkdir()
    np.save(out / "case.npy", rows)
    (out / "metadata.json").write_text(json.dumps({
        "schema": {"columns": list(CASE_COLUMNS_THETA),
                   "source_codes": {SRC_INLET: "inlet", SRC_SIM: "simulation",
                                    SRC_SENSOR: "sensor"}},
        "normalization": {},
    }))
    return out / "case.npy", CaseMetadata.load(out / "metadata.json")


@pytest.fixture
def bottom_wall_csv(tmp_path: Path):
    """Write a bottom-wall surface CSV (x,y,z at z=0); return the path.

    Boundary wall geometry still comes from a surface file (see boundary.py); it
    is a separate concern from the .npy case data.
    """

    import csv
    axis = (0.0, 0.25, 0.5, 0.75, 1.0)
    rows = [(x, y, 0.0) for x in axis for y in axis]
    path = tmp_path / "bottom_wall.csv"
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(("x", "y", "z"))
        w.writerows(rows)
    return path


@pytest.fixture
def report():
    """Return a function that prints the actual verified values for a test."""

    def _format(value: object, max_rows: int) -> str:
        array = np.asarray(value)
        if array.ndim >= 1 and array.shape[0] > max_rows:
            shown = np.array2string(array[:max_rows], separator=", ")
            return f"{shown}  ... (+{array.shape[0] - max_rows} more rows)"
        return np.array2string(array, separator=", ")

    def _report(label: str, *, max_rows: int = 8, **arrays: object) -> None:
        print(f"\n  verified: {label}")
        for name, value in arrays.items():
            text = _format(value, max_rows).replace("\n", "\n      ")
            print(f"    {name} = {text}")

    return _report
