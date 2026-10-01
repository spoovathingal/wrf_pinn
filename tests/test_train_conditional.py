"""End-to-end MWE check on synthetic cases: encode -> decode -> losses -> step.

Proves the conditional pipeline runs, the loss decreases, and the encoder is
actually used (swapping phi/psi changes the output with the flatten encoder but not
with the null encoder).
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from wrf_pinn.data.conditional_case import (  # noqa: E402
    COORD_NAMES, FACE_NAMES, STATE_VARS, ConditionalCase,
)
from wrf_pinn.models.conditional import (  # noqa: E402
    ConditionalModel, ConditionalModelConfig, FlattenMLPEncoder, NullEncoder,
)
from wrf_pinn.training.train_conditional import (  # noqa: E402
    ConditionalTrainConfig, split_cases, train_conditional,
)


def _make_case(name: str, seed: int) -> ConditionalCase:
    nv, nc, nf = len(STATE_VARS), len(COORD_NAMES), len(FACE_NAMES)
    n_ic, n_times, face_len, n_terr, n_pts = 5, 3, 4, 6, 12
    rng = np.random.default_rng(seed)
    return ConditionalCase(
        name=name,
        initial=rng.standard_normal((n_ic, nc + nv)).astype(np.float32),
        boundary=rng.standard_normal((nf, n_times, face_len, nv)).astype(np.float32),
        boundary_coords=rng.standard_normal((nf, n_times, face_len, nc)).astype(np.float32),
        terrain=rng.standard_normal((n_terr, 3)).astype(np.float32),
        interior=rng.standard_normal((n_pts, nc)).astype(np.float32),
        targets=rng.standard_normal((n_pts, nv)).astype(np.float32),
        target_mask=np.ones((n_pts, nv), dtype=np.float32),
        times=np.arange(n_times, dtype=np.float32),
    )


def _flatten_in_features() -> int:
    nv, nc, nf = len(STATE_VARS), len(COORD_NAMES), len(FACE_NAMES)
    n_ic, n_times, face_len, n_terr = 5, 3, 4, 6
    return n_ic * (nc + nv) + nf * n_times * face_len * nv + n_terr * 3


def test_pipeline_runs_with_null_encoder():
    cases = [_make_case(f"c{i}", i) for i in range(10)]
    model = ConditionalModel(NullEncoder(), ConditionalModelConfig(hidden_layers=2,
                                                                   hidden_width=32))
    cfg = ConditionalTrainConfig(epochs=50, batch_cases=4, device="cpu", log_every=50)
    hist = train_conditional(model, cases, cfg)
    assert len(hist.total) == 50
    assert np.isfinite(hist.total).all()


def test_loss_decreases_with_flatten_encoder():
    cases = [_make_case(f"c{i}", i) for i in range(10)]
    enc = FlattenMLPEncoder(in_features=_flatten_in_features(), latent_dim=8)
    model = ConditionalModel(enc, ConditionalModelConfig(hidden_layers=2,
                                                         hidden_width=32))
    cfg = ConditionalTrainConfig(epochs=300, batch_cases=4, device="cpu", log_every=300)
    hist = train_conditional(model, cases, cfg)
    assert hist.total[-1] < hist.total[0], "loss did not decrease"


def test_encoder_is_actually_used():
    """With the flatten encoder, changing phi/psi must change the prediction."""
    case = _make_case("c", 0).as_torch()
    enc = FlattenMLPEncoder(in_features=_flatten_in_features(), latent_dim=8)
    model = ConditionalModel(enc, ConditionalModelConfig(hidden_layers=2, hidden_width=32))
    model.eval()

    z1 = model.encode(case.initial, case.boundary, case.terrain)
    pred1 = model(case.interior, z1)

    bumped = case.initial + 1.0
    z2 = model.encode(bumped, case.boundary, case.terrain)
    pred2 = model(case.interior, z2)

    assert not torch.allclose(pred1, pred2), "encoder output ignored by decoder"


def test_null_encoder_ignores_conditions():
    """The null encoder must NOT react to phi/psi (sanity: it's a plain MLP)."""
    case = _make_case("c", 0).as_torch()
    model = ConditionalModel(NullEncoder(), ConditionalModelConfig(hidden_layers=2,
                                                                   hidden_width=32))
    model.eval()
    z = model.encode(case.initial, case.boundary, case.terrain)
    pred1 = model(case.interior, z)
    z2 = model.encode(case.initial + 5.0, case.boundary, case.terrain)
    pred2 = model(case.interior, z2)
    assert torch.allclose(pred1, pred2)


def test_split_cases():
    cases = [_make_case(f"c{i}", i) for i in range(10)]
    train, held = split_cases(cases, chi=0.8, seed=0)
    assert len(train) == 8 and len(held) == 2
