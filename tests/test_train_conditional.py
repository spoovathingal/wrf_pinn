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


def test_checkpoint_save_and_resume(tmp_path):
    """Checkpointing writes a file (creating its dir) and resume continues from it."""
    cases = [_make_case(f"c{i}", i) for i in range(6)]
    # nested path that does not exist yet: exercises _save_checkpoint's mkdir.
    ckpt = tmp_path / "checkpoints" / "run.pt"

    model = ConditionalModel(LATENT_DIM, ConditionalModelConfig(hidden_layers=2, hidden_width=16))
    cfg = ConditionalTrainConfig(epochs=20, batch_cases=3, device="cpu", log_every=20,
                                 checkpoint_path=str(ckpt), checkpoint_every=10, **_DATA_ONLY)
    hist = train_conditional(model, cases, cfg)
    assert ckpt.is_file(), "checkpoint file was not written"
    assert len(hist.total) == 20

    # the saved epoch must be the last multiple of checkpoint_every (or final epoch)
    ck = torch.load(str(ckpt), map_location="cpu")
    assert ck["epoch"] == 20

    # resume into a FRESH model: weights load from the checkpoint, start epoch advances.
    fresh = ConditionalModel(LATENT_DIM, ConditionalModelConfig(hidden_layers=2, hidden_width=16))
    before = [p.clone() for p in fresh.parameters()]
    cfg2 = ConditionalTrainConfig(epochs=25, batch_cases=3, device="cpu", log_every=25,
                                  checkpoint_path=str(ckpt), checkpoint_every=10, **_DATA_ONLY)
    hist2 = train_conditional(fresh, cases, cfg2)
    # resumed at epoch 21 and continued the saved history: 20 restored + 5 new = 25.
    assert len(hist2.total) == 25, f"expected continuous history of 25, got {len(hist2.total)}"
    # the fresh model's weights changed on load (they were restored from the checkpoint)
    after = list(fresh.parameters())
    assert any(not torch.equal(b, a) for b, a in zip(before, after)), "weights not restored on resume"


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


def test_collocation_points_independent_of_cases_per_pass(monkeypatch):
    """The PDE points a step sees must not depend on how its cases are split into passes."""
    import wrf_pinn.training.train_conditional as tc

    cases = [_make_case(f"c{i}", i) for i in range(8)]

    def collocation(cases_per_pass: int) -> torch.Tensor:
        seen = []

        def fake_pde(model, coll, z_rows, cfg, scaling):  # random data NaNs the real PDE
            seen.append(coll.detach().clone())
            return (model(coll, z_rows) * 0.0).sum()

        monkeypatch.setattr(tc, "_pde_loss", fake_pde)
        torch.manual_seed(0)
        model = ConditionalModel(LATENT_DIM, ConditionalModelConfig(hidden_layers=2, hidden_width=16))
        cfg = ConditionalTrainConfig(epochs=3, batch_cases=4, cases_per_pass=cases_per_pass,
                                     n_collocation=16, device="cpu", log_every=100,
                                     weight_surface=0.0)
        train_conditional(model, cases, cfg)
        return torch.cat(seen)

    whole = collocation(0)
    assert torch.equal(whole, collocation(1))
    assert torch.equal(whole, collocation(2))
