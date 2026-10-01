"""Freeze the ConditionalCase contract: build a tiny synthetic case, round-trip it
through save/read, and check shapes, masking, and the torch move.

This fixture is the shared truth the pre-processor (writer) and training (reader)
both build against, so it is validated before either end is written.
"""

from __future__ import annotations

import numpy as np
import pytest

from wrf_pinn.data.conditional_case import (
    COORD_NAMES, FACE_NAMES, STATE_VARS,
    ConditionalCase, read_conditional_case,
)


def make_fixture() -> ConditionalCase:
    """A trivially small conditional case with known shapes."""
    nv, nc, nf = len(STATE_VARS), len(COORD_NAMES), len(FACE_NAMES)
    n_ic, n_times, face_len, n_terr, n_pts = 5, 3, 4, 6, 7
    rng = np.random.default_rng(0)
    return ConditionalCase(
        name="synthetic_0",
        initial=rng.standard_normal((n_ic, nc + nv)).astype(np.float32),
        boundary=rng.standard_normal((nf, n_times, face_len, nv)).astype(np.float32),
        terrain=rng.standard_normal((n_terr, 3)).astype(np.float32),
        interior=rng.standard_normal((n_pts, nc)).astype(np.float32),
        targets=rng.standard_normal((n_pts, nv)).astype(np.float32),
        target_mask=np.ones((n_pts, nv), dtype=np.float32),
        times=np.arange(n_times, dtype=np.float32),
    )


def test_roundtrip_preserves_members(tmp_path):
    case = make_fixture()
    path = case.save(tmp_path / "synthetic_0.npz")
    loaded = read_conditional_case(path)

    assert loaded.name == "synthetic_0"
    for member in ("initial", "boundary", "terrain", "interior", "targets", "times"):
        np.testing.assert_allclose(
            getattr(loaded, member), getattr(case, member), rtol=0, atol=0,
            err_msg=f"{member} changed on round-trip",
        )


def test_shapes_match_contract():
    case = make_fixture()
    nv, nc, nf = len(STATE_VARS), len(COORD_NAMES), len(FACE_NAMES)
    assert case.initial.shape[1] == nc + nv
    assert case.boundary.shape[0] == nf
    assert case.boundary.shape[3] == nv
    assert case.terrain.shape[1] == 3
    assert case.interior.shape[1] == nc
    assert case.targets.shape[1] == nv
    assert case.target_mask.shape == case.targets.shape


def test_nan_targets_masked_and_zero_filled(tmp_path):
    case = make_fixture()
    t = np.array(case.targets, dtype=np.float32)
    t[0, 0] = np.nan                 # one unmeasured entry
    object.__setattr__(case, "targets", t)
    path = case.save(tmp_path / "with_nan.npz")
    loaded = read_conditional_case(path)

    assert loaded.target_mask[0, 0] == 0.0           # recorded as unmeasured
    assert loaded.targets[0, 0] == 0.0               # zero-filled
    assert np.isfinite(loaded.targets).all()         # no NaN leaks into training


def test_nonfinite_interior_rejected(tmp_path):
    case = make_fixture()
    xy = np.array(case.interior, dtype=np.float32)
    xy[0, 0] = np.nan
    object.__setattr__(case, "interior", xy)
    path = case.save(tmp_path / "bad.npz")
    with pytest.raises(ValueError, match="Non-finite interior"):
        read_conditional_case(path)


def test_as_torch_moves_members():
    torch = pytest.importorskip("torch")
    case = make_fixture().as_torch()
    assert isinstance(case.initial, torch.Tensor)
    assert isinstance(case.boundary, torch.Tensor)
    assert case.targets.dtype == torch.float32
