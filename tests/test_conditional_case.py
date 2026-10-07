"""ConditionalCase dataclass contract: shapes and the torch move. The save/read
round-trip now lives in the preprocessor suite (generate -> read_conditional_case),
since cases are written by the POD encoder, not the dataclass.
"""

from __future__ import annotations

import numpy as np
import pytest

from wrf_pinn.data.conditional_case import (
    COORD_NAMES, FACE_NAMES, STATE_VARS, ConditionalCase,
)

LATENT_DIM = 8


def make_fixture() -> ConditionalCase:
    """A trivially small conditional case with known shapes."""
    nv, nc, nf = len(STATE_VARS), len(COORD_NAMES), len(FACE_NAMES)
    nx, ny, nz = 3, 3, 4
    n_ic = nx * ny * nz
    n_times, face_len, n_terr, n_pts = 3, 4, 6, 7
    rng = np.random.default_rng(0)
    f32 = np.float32
    return ConditionalCase(
        name="synthetic_0",
        z=rng.standard_normal(LATENT_DIM).astype(f32),
        initial_coords=rng.standard_normal((n_ic, nc)).astype(f32),
        initial=rng.standard_normal((nv, nz, nx * ny)).astype(f32),
        boundary=rng.standard_normal((nv, nf, n_times, face_len)).astype(f32),
        boundary_coords=rng.standard_normal((nf, n_times, face_len, nc)).astype(f32),
        surface=rng.standard_normal((n_times, 9, 5)).astype(f32),
        terrain=rng.standard_normal((n_terr, 3)).astype(f32),
        interior=rng.standard_normal((n_pts, nc)).astype(f32),
        targets=rng.standard_normal((n_pts, nv)).astype(f32),
        target_mask=np.ones((n_pts, nv), dtype=f32),
        times=np.arange(n_times, dtype=f32),
    )


def test_shapes_match_contract():
    case = make_fixture()
    nv, nc, nf = len(STATE_VARS), len(COORD_NAMES), len(FACE_NAMES)
    assert case.initial.shape[0] == nv
    assert case.boundary.shape[0] == nv
    assert case.boundary.shape[1] == nf
    assert case.terrain.shape[1] == 3
    assert case.interior.shape[1] == nc
    assert case.targets.shape[1] == nv
    assert case.target_mask.shape == case.targets.shape


def test_as_torch_moves_members():
    torch = pytest.importorskip("torch")
    case = make_fixture().as_torch()
    assert isinstance(case.z, torch.Tensor)
    assert isinstance(case.initial, torch.Tensor)
    assert isinstance(case.boundary, torch.Tensor)
    assert case.targets.dtype == torch.float32
