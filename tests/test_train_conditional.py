"""MWE check on synthetic cases: the conditional pipeline runs, the loss
decreases, and the latent z actually conditions the solution network.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from wrf_pinn.data.conditional_case import (  # noqa: E402
    COORD_NAMES, FACE_NAMES, STATE_VARS, ConditionalCase,
)
from wrf_pinn.models.conditional import (  # noqa: E402
    ConditionalModel, ConditionalModelConfig,
)
from wrf_pinn.training.train_conditional import (  # noqa: E402
    ConditionalTrainConfig, split_cases, train_conditional,
)

LATENT_DIM = 8


def _make_case(name: str, seed: int) -> ConditionalCase:
    nv, nc, nf = len(STATE_VARS), len(COORD_NAMES), len(FACE_NAMES)
    nx, ny, nz = 3, 3, 4
    n_ic = nx * ny * nz
    n_times, face_len, n_terr, n_pts = 3, 4, 6, 12
    rng = np.random.default_rng(seed)
    f32 = np.float32
    return ConditionalCase(
        name=name,
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


# PDE/surface losses need physical fields; random synthetic data drives the moist
# EOS to NaN, so the data-wiring smoke tests zero those weights.
_DATA_ONLY = dict(weight_pde=0.0, weight_surface=0.0)


def test_pipeline_runs():
    cases = [_make_case(f"c{i}", i) for i in range(10)]
    model = ConditionalModel(LATENT_DIM, ConditionalModelConfig(hidden_layers=2, hidden_width=32))
    cfg = ConditionalTrainConfig(epochs=50, batch_cases=4, device="cpu", log_every=50, **_DATA_ONLY)
    hist = train_conditional(model, cases, cfg)
    assert len(hist.total) == 50
    assert np.isfinite(hist.total).all()


def test_loss_decreases():
    cases = [_make_case(f"c{i}", i) for i in range(10)]
    model = ConditionalModel(LATENT_DIM, ConditionalModelConfig(hidden_layers=2, hidden_width=32))
    cfg = ConditionalTrainConfig(epochs=300, batch_cases=4, device="cpu", log_every=300, **_DATA_ONLY)
    hist = train_conditional(model, cases, cfg)
    assert hist.total[-1] < hist.total[0], "loss did not decrease"


def test_latent_conditions_output():
    """Changing z must change the prediction at fixed coordinates."""
    case = _make_case("c", 0).as_torch()
    model = ConditionalModel(LATENT_DIM, ConditionalModelConfig(hidden_layers=2, hidden_width=32))
    model.eval()
    pred1 = model(case.interior, case.z.unsqueeze(0))
    pred2 = model(case.interior, (case.z + 1.0).unsqueeze(0))
    assert not torch.allclose(pred1, pred2), "latent z ignored by solution network"


def test_split_cases():
    cases = [_make_case(f"c{i}", i) for i in range(10)]
    train, held = split_cases(cases, chi=0.8, seed=0)
    assert len(train) == 8 and len(held) == 2
