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
from dataclasses import replace

from wrf_pinn.config.scaling import ResidualScalingConfig, VariableScale
import wrf_pinn.training.train_conditional as tc

LATENT_DIM = 8


def _make_case(name: str, seed: int) -> ConditionalCase:
    nv, nc, nf = len(STATE_VARS), len(COORD_NAMES), len(FACE_NAMES)
    nx, ny, nz = 3, 3, 4
    n_ic = nx * ny * nz
    n_times, face_len, n_pts = 3, 4, 12
    rng = np.random.default_rng(seed)
    f32 = np.float32
    return ConditionalCase(
        name=name,
        z=rng.standard_normal(LATENT_DIM).astype(f32),
        initial_coords=rng.random((n_ic, nc)).astype(f32),
        initial=rng.standard_normal(
            (nv, nz, nx * ny)
        ).astype(f32),
        boundary=rng.standard_normal(
            (nv, nf, n_times, face_len)
        ).astype(f32),
        boundary_coords=rng.random(
            (nf, n_times, face_len, nc)
        ).astype(f32),
        surface=rng.random((n_times, 9, 5)).astype(f32),
        terrain=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
            ],
            dtype=f32,
            ),
        interior=rng.random((n_pts, nc)).astype(f32),
        targets=rng.standard_normal((n_pts, nv)).astype(f32),
        target_mask=np.ones((n_pts, nv), dtype=f32),
        times=np.arange(n_times, dtype=f32),
    )

def _geometry_fixture():
    # Source-normalized coordinates: x, y, reference z, t.
    rows = np.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.5, 1.0 / 3.0],
            [0.0, 1.0, 1.0, 2.0 / 3.0],
            [1.0, 1.0, 0.25, 1.0],
        ],
        dtype=np.float32,
    )

    case = ConditionalCase(
        name="geometry",
        z=np.zeros(LATENT_DIM, dtype=np.float32),
        initial_coords=rows.copy(),
        initial=np.zeros((len(STATE_VARS), 1, 4), dtype=np.float32),
        boundary_coords=np.tile(rows, (len(FACE_NAMES), 1, 1, 1)),
        boundary=np.zeros(
            (len(STATE_VARS), len(FACE_NAMES), 1, 4),
            dtype=np.float32,
        ),
        surface=np.column_stack(
            (rows[:, [0, 1, 3]], np.zeros((4, 2), dtype=np.float32))
        )[None],
        terrain=np.array(
            [
                [0.0, 0.0, 100.0],
                [100.0, 0.0, 300.0],
                [0.0, 100.0, 400.0],
                [100.0, 100.0, 600.0],
            ],
            dtype=np.float32,
        ),
        interior=rows.copy(),
        targets=np.zeros((4, len(STATE_VARS)), dtype=np.float32),
        target_mask=np.ones((4, len(STATE_VARS)), dtype=np.float32),
        times=np.array([0.0], dtype=np.float32),
    ).as_torch()

    source = ResidualScalingConfig(
        x=VariableScale(scale=100.0),
        y=VariableScale(scale=100.0),
        z=VariableScale(offset=10.0, scale=90.0),
        t=VariableScale(scale=60.0),
    )
    model_scaling = replace(
        source, z=VariableScale(offset=-20.0, scale=1000.0)
    )
    cfg = ConditionalTrainConfig(
        source_scaling=source,
        scaling=model_scaling,
        source_z_mode="fe_terrain_following",
        terrain_top_height=1000.0,
        fe_surface_reference_z=10.0,
        n_collocation=4,
        weight_surface=0.0,
        device="cpu",
    )
    return case, cfg

# PDE/surface losses need physical fields; random synthetic data drives the moist
# EOS to NaN, so the data-wiring smoke tests zero those weights.
_DATA_ONLY = dict(weight_pde=0.0, weight_surface=0.0,
    source_scaling=ResidualScalingConfig())


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
                                 checkpoint_path=str(ckpt), checkpoint_every=10,
                                 events_path=str(tmp_path / "events.jsonl"), **_DATA_ONLY)
    hist = train_conditional(model, cases, cfg)
    assert ckpt.is_file(), "checkpoint file was not written"
    assert len(hist.total) == 20

    # the saved epoch must be the last multiple of checkpoint_every (or final epoch)
    ck = torch.load(str(ckpt), map_location="cpu")
    assert ck["epoch"] == 20
    assert ck["schema_version"] == 3
    assert ck["geometry_version"] == 2

    saved_training = ck["manifest"]["training"]
    assert saved_training["source_z_mode"] == cfg.source_z_mode
    assert saved_training["terrain_top_height"] == cfg.terrain_top_height
    assert saved_training["fe_surface_reference_z"] == cfg.fe_surface_reference_z
    assert ck["manifest"]["model"]["latent_dim"] == LATENT_DIM
    assert ck["manifest"]["model"]["config"]["hidden_width"] == 16
    assert ck["manifest"]["training"]["seed"] == cfg.seed
    import json
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert events[0]["event"] == "run_start"
    assert any(e["event"] == "step" and e["gradient_norm_finite"] for e in events)
    assert events[-1]["event"] == "checkpoint" and events[-1]["epoch"] == 20

    # Changed geometry must not silently resume an existing run.
    for changed in (
        {"terrain_top_height": cfg.terrain_top_height + 100.0},
        {"fe_surface_reference_z": cfg.fe_surface_reference_z + 1.0},
        {"source_z_mode": "height_above_ground"},
    ):
        incompatible = replace(cfg, epochs=21, **changed)
        with pytest.raises(ValueError, match="settings differ"):
            train_conditional(model, cases, incompatible)

    # Previous geometry versions must also be rejected.
    old_path = tmp_path / "old_geometry.pt"
    torch.save({**ck, "geometry_version": 1}, old_path)

    incompatible = replace(
        cfg, epochs=21, checkpoint_path=str(old_path)
    )
    with pytest.raises(ValueError, match="settings differ"):
        train_conditional(model, cases, incompatible)

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

        def fake_pde(model, coll, z_rows, cfg, scaling, fns=None):  # random data NaNs the real PDE
            seen.append(coll.detach().clone())
            return (model(coll, z_rows, cfg.n_collocation) * 0.0).sum()  # z per case

        monkeypatch.setattr(tc, "_pde_loss", fake_pde)
        torch.manual_seed(0)
        model = ConditionalModel(LATENT_DIM, ConditionalModelConfig(hidden_layers=2, hidden_width=16))
        cfg = ConditionalTrainConfig(epochs=3, batch_cases=4, cases_per_pass=cases_per_pass,
                                     n_collocation=16, device="cpu", log_every=100,
                                     source_scaling=ResidualScalingConfig(),weight_surface=0.0)
        train_conditional(model, cases, cfg)
        return torch.cat(seen)

    whole = collocation(0)
    assert torch.equal(whole, collocation(1))
    assert torch.equal(whole, collocation(2))


def test_bf16_data_forwards_stay_close_to_float32():
    """bf16 autocast on the derivative-free forwards changes losses only at bf16 rounding."""
    cases = [_make_case(f"c{i}", i) for i in range(8)]

    def history(bf16: bool):
        torch.manual_seed(0)
        model = ConditionalModel(LATENT_DIM, ConditionalModelConfig(hidden_layers=2, hidden_width=32))
        cfg = ConditionalTrainConfig(epochs=5, batch_cases=4, device="cpu", log_every=100,
                                     bf16_data=bf16, **_DATA_ONLY)
        return np.array(train_conditional(model, cases, cfg).total)

    fp32, bf16 = history(False), history(True)
    assert np.isfinite(bf16).all()
    assert not np.array_equal(fp32, bf16), "bf16 flag had no effect"
    np.testing.assert_allclose(bf16, fp32, rtol=2e-2)


def test_compiled_step_tracks_eager():
    """compile=True gives the same training (to float32 kernel-fusion rounding)."""
    cases = [_make_case(f"c{i}", i) for i in range(8)]

    def history(compile_: bool):
        torch.manual_seed(0)
        model = ConditionalModel(LATENT_DIM, ConditionalModelConfig(hidden_layers=2, hidden_width=32))
        cfg = ConditionalTrainConfig(epochs=4, batch_cases=4, device="cpu", log_every=100,
                                     compile=compile_, **_DATA_ONLY)
        return np.array(train_conditional(model, cases, cfg).total)

    np.testing.assert_allclose(history(True), history(False), rtol=1e-5)


def test_per_case_latent_matches_per_point_rows():
    """ConditionalModel(coords, z, sizes) equals feeding z repeated per point."""
    torch.manual_seed(0)
    model = ConditionalModel(LATENT_DIM, ConditionalModelConfig(hidden_layers=2, hidden_width=16)).double()
    z = torch.randn(3, LATENT_DIM, dtype=torch.float64)
    sizes = torch.tensor([4, 1, 5])
    coords = torch.randn(int(sizes.sum()), 4, dtype=torch.float64)
    rows = z.repeat_interleave(sizes, dim=0)
    torch.testing.assert_close(model(coords, z, sizes), model(coords, rows), rtol=1e-12, atol=1e-12)
    # an int size gives every case the same number of rows
    coords2 = torch.randn(6, 4, dtype=torch.float64)
    torch.testing.assert_close(model(coords2, z, 2), model(coords2, z.repeat_interleave(2, dim=0)),
                               rtol=1e-12, atol=1e-12)
    # broadcast form: one latent for all points
    torch.testing.assert_close(model(coords2, z[:1]), model(coords2, z[:1].expand(6, -1)),
                               rtol=1e-12, atol=1e-12)


def _write_case_npz(path, seed: int) -> None:
    """A synthetic case in the preprocessor's .npz layout (rank-1 POD pieces)."""
    case = _make_case(path.stem, seed)
    nv, nz, nxy = case.initial.shape
    _, nf, nt, fl = case.boundary.shape
    rng = np.random.default_rng(seed)
    f32 = np.float32
    blob = {k: getattr(case, k) for k in ("initial_coords", "boundary_coords", "surface", "terrain",
                                          "interior", "targets", "times")}
    for v in STATE_VARS:
        blob[f"ini_mean_{v}"] = rng.standard_normal(nxy).astype(f32)
        blob[f"ini_modes_{v}"] = rng.standard_normal((nxy, 1)).astype(f32)
        blob[f"ini_coeffs_{v}"] = rng.standard_normal((nz, 1)).astype(f32)
        for f in range(nf):
            blob[f"bnd_mean_{v}_{f}"] = rng.standard_normal(fl).astype(f32)
            blob[f"bnd_modes_{v}_{f}"] = rng.standard_normal((fl, 1)).astype(f32)
            blob[f"bnd_coeffs_{v}_{f}"] = rng.standard_normal((nt, 1)).astype(f32)
    np.savez(path, **blob)


def test_loader_workers_give_identical_training(tmp_path):
    """Reading cases in DataLoader worker processes changes nothing but where they are read."""
    paths = []
    for i in range(8):
        path = tmp_path / f"c{i}.npz"
        _write_case_npz(path, i)
        paths.append(str(path))

    def history(workers: int):
        torch.manual_seed(0)
        from wrf_pinn.data.conditional_case import read_conditional_case
        latent = read_conditional_case(paths[0]).z.shape[0]
        model = ConditionalModel(latent, ConditionalModelConfig(hidden_layers=2, hidden_width=16))
        cfg = ConditionalTrainConfig(epochs=5, batch_cases=4, cases_per_pass=2, device="cpu",
                                     log_every=100, loader_workers=workers, **_DATA_ONLY)
        return train_conditional(model, config=cfg, case_paths=paths).total

    assert history(2) == history(0)


def test_average_across_processes_matches_per_tensor_reduce(monkeypatch):
    """One packed all-reduce gives each gradient and the losses their own average."""
    import wrf_pinn.training.train_conditional as tc

    world = 3
    # stand-in all-reduce: the sum over `world` processes holding the same values
    monkeypatch.setattr(tc.dist, "all_reduce", lambda t: t.mul_(world))
    torch.manual_seed(0)
    model = ConditionalModel(LATENT_DIM, ConditionalModelConfig(hidden_layers=2, hidden_width=8))
    params = list(model.parameters())
    for p in params:
        p.grad = torch.randn_like(p)
    expected = [p.grad.clone() for p in params]     # sum of world copies / world
    losses = torch.tensor([1.0, 2.0, 3.0])
    out = tc._average_across_processes(params, losses, world)
    for p, e in zip(params, expected):
        torch.testing.assert_close(p.grad, e)
    torch.testing.assert_close(out, losses)


def test_average_across_processes_two_real_processes(tmp_path):
    """Packed all-reduce = per-tensor all-reduces, across two real gloo processes."""
    import socket
    import subprocess
    import sys
    from pathlib import Path

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = str(s.getsockname()[1])
    worker = Path(__file__).with_name("_dist_worker.py")
    procs = [subprocess.Popen([sys.executable, str(worker), str(r), "2", port, str(tmp_path)])
             for r in range(2)]
    assert all(p.wait(timeout=120) == 0 for p in procs)
    assert [(tmp_path / f"rank{r}.txt").read_text() for r in range(2)] == ["ok", "ok"]

def test_terrain_observation_coordinates():
    case, cfg = _geometry_fixture()
    batch = tc.stack_cases([case], cfg)

    expected = case.interior.clone()
    expected[:, 2] = torch.tensor([0.129, 0.3585, 0.480, 0.633])

    torch.testing.assert_close(batch.interior, expected)
    torch.testing.assert_close(batch.initial_coords, expected)
    torch.testing.assert_close(
        batch.boundary_coords, expected.repeat(len(FACE_NAMES), 1)
    )

    # Sampling bounds remain in source-normalized coordinates.
    torch.testing.assert_close(batch.box_lo[0], case.interior.amin(dim=0))
    torch.testing.assert_close(batch.box_hi[0], case.interior.amax(dim=0))

def test_terrain_collocation_matches_observations(monkeypatch):
    case, cfg = _geometry_fixture()
    batch = tc.stack_cases([case], cfg)
    model = ConditionalModel(
        LATENT_DIM,
        ConditionalModelConfig(hidden_layers=1, hidden_width=8),
    )
    seen = {}

    def capture_pde(model, coll, z_rows, config, scaling, fns):
        seen["coordinates"] = coll.detach().clone()
        return coll.new_zeros(())

    monkeypatch.setattr(tc, "_pde_loss", capture_pde)

    # This fixture has source-coordinate bounds [0, 1] on every axis.
    # These samples therefore coincide with its observation locations.
    tc._batch_losses(
        model,
        batch,
        cfg,
        unit_coll=case.interior.unsqueeze(0),
    )

    torch.testing.assert_close(seen["coordinates"], batch.interior)

def test_terrain_surface_reference_height(monkeypatch):
    case, cfg = _geometry_fixture()
    batch = tc.stack_cases([case], cfg)
    model = ConditionalModel(
        LATENT_DIM,
        ConditionalModelConfig(hidden_layers=1, hidden_width=8),
    )
    seen = {}

    def capture_surface(**kwargs):
        seen["wall"] = kwargs["coordinates"].detach().clone()
        seen["reference"] = kwargs["reference_coordinates"].detach().clone()
        return {"test": kwargs["state"][:, :1] * 0.0}

    monkeypatch.setattr(
        tc, "no_penetration_z_wall_residuals", capture_surface
    )
    tc._surface_loss(
        model, batch, cfg, cfg.scaling, tc.step_functions(model)
    )

    # Physical wall heights: 100, 300, 400, 600 m.
    torch.testing.assert_close(
        seen["wall"][:, 2],
        torch.tensor([0.120, 0.320, 0.420, 0.620]),
    )

    # Physical first-level heights: 109, 307, 406, 604 m.
    torch.testing.assert_close(
        seen["reference"][:, 2],
        torch.tensor([0.129, 0.327, 0.426, 0.624]),
    )

def test_physical_altitude_is_not_transformed_twice():
    case, cfg = _geometry_fixture()
    expected = tc.stack_cases([case], cfg)

    physical_case = replace(
        case,
        interior=expected.interior,
        initial_coords=expected.initial_coords,
        boundary_coords=expected.boundary_coords.reshape_as(
            case.boundary_coords
        ),
    )
    physical_cfg = replace(
        cfg,
        source_z_mode="physical_altitude",
        source_scaling=cfg.scaling,
    )
    actual = tc.stack_cases([physical_case], physical_cfg)

    torch.testing.assert_close(actual.interior, expected.interior)
    torch.testing.assert_close(actual.initial_coords, expected.initial_coords)
    torch.testing.assert_close(actual.boundary_coords, expected.boundary_coords)

@pytest.mark.parametrize(
    "changes",
    [
        {"terrain_top_height": 0.0},
        {"terrain_top_height": float("nan")},
        {"terrain_top_height": float("inf")},
        {"fe_surface_reference_z": 0.0},
        {"fe_surface_reference_z": float("nan")},
        {"fe_surface_reference_z": 1000.0},
        {"source_z_mode": "unknown"},
    ],
)
def test_invalid_terrain_configuration(changes):
    _, cfg = _geometry_fixture()
    with pytest.raises(ValueError):
        replace(cfg, **changes)


def test_terrain_must_be_below_compression_height():
    case, cfg = _geometry_fixture()
    terrain = case.terrain.clone()
    terrain[0, 2] = cfg.terrain_top_height

    with pytest.raises(ValueError, match="terrain must be below"):
        tc.stack_cases([replace(case, terrain=terrain)], cfg)