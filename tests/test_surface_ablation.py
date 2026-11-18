"""Exclude the surface heat-flux objective while retaining the three mechanical terms."""

import importlib.util
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from test_train_conditional import LATENT_DIM, _geometry_fixture, _make_case
from wrf_pinn.config.scaling import ResidualScalingConfig, VariableScale
from wrf_pinn.models.conditional import ConditionalModel, ConditionalModelConfig
from wrf_pinn.training import train_conditional as tc


SELECTION = "all-except-heat-flux"
ACTIVE = {"no_penetration_w", "surface_stress_xz", "surface_stress_yz"}


def test_surface_selection_cli():
    path = Path(__file__).parents[1] / "scripts" / "train_conditional_lcc.py"
    spec = importlib.util.spec_from_file_location("ablation_driver", path)
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    parser = driver._argument_parser()
    for flags in (["--surface-term", SELECTION], ["--no-heat-flux"]):
        args = parser.parse_args(["cases", "--loss-weights", "grad-balance", *flags])
        cfg = tc.ConditionalTrainConfig(surface_term=args.surface_term, loss_weights=args.loss_weights)
        assert cfg.surface_term == SELECTION and set(cfg.surface_terms) == ACTIVE
    assert parser.parse_args(["cases"]).surface_term == "all"
    with pytest.raises(SystemExit):
        parser.parse_args(["cases", "--no-heat-flux", "--surface_heat_flux"])
    assert tc.ConditionalTrainConfig(surface_term="all").surface_terms == tc.SURFACE_RAW_NAMES
    assert tc.ConditionalTrainConfig(surface_term="none").surface_terms == ()
    for name in tc.SURFACE_RAW_NAMES:
        assert tc.ConditionalTrainConfig(surface_term=name).surface_terms == (name,)


def test_heat_flux_excluded_from_surface_value_and_gradients(monkeypatch):
    case, cfg = _geometry_fixture()
    cfg = replace(cfg, surface_term=SELECTION)
    batch = tc.stack_cases([case], cfg)
    model = ConditionalModel(LATENT_DIM, ConditionalModelConfig(hidden_layers=1, hidden_width=8))
    residuals = {"no_penetration_w": torch.tensor(1., requires_grad=True),
                 "surface_stress_xz": torch.tensor(2., requires_grad=True),
                 "surface_stress_yz": torch.tensor(3., requires_grad=True),
                 "surface_heat_flux": torch.tensor(float("nan"), requires_grad=True)}

    def fake_residuals(**kwargs):
        values = {name: r.expand(len(batch.surface), 1) for name, r in residuals.items()}
        kwargs["raw_residuals"].update({name: r.detach() for name, r in values.items()})
        return values

    monkeypatch.setattr(tc, "no_penetration_z_wall_residuals", fake_residuals)
    raw = {}
    loss = tc._surface_loss(model, batch, cfg, cfg.scaling, tc.step_functions(model), raw_mse=raw)
    assert loss.item() == pytest.approx((1 + 4 + 9) / 3)
    assert set(raw) == ACTIVE
    loss.backward()
    for name in ACTIVE:
        assert residuals[name].grad.item() == pytest.approx(2 * residuals[name].item() / 3)
    assert residuals["surface_heat_flux"].grad is None


@pytest.mark.parametrize("weighting", ["fixed", "grad-balance"])
def test_training_logging_and_resume_without_heat_flux(tmp_path, weighting):
    torch.manual_seed(0)
    scaling = replace(ResidualScalingConfig(), theta=VariableScale(offset=300.0))
    cfg = tc.ConditionalTrainConfig(
        epochs=2, batch_cases=2, cases_per_pass=1, n_collocation=2, device="cpu", log_every=1,
        loss_weights=weighting, surface_term=SELECTION, scaling=scaling, source_scaling=scaling,
        checkpoint_path=str(tmp_path / "run.pt"), events_path=str(tmp_path / "events.jsonl"),
    )
    cases = [_make_case(f"c{i}", i) for i in range(2)]
    model_cfg = ConditionalModelConfig(hidden_layers=1, hidden_width=8)
    model = ConditionalModel(LATENT_DIM, model_cfg)
    history = tc.train_conditional(model, cases, cfg)
    assert np.isfinite(history.total).all() and all(value > 0 for value in history.surface)
    checkpoint = torch.load(cfg.checkpoint_path, weights_only=True)
    assert checkpoint["manifest"]["training"]["surface_term"] == SELECTION
    assert "surface" in checkpoint["loss_weights"]["weights"]
    resumed = ConditionalModel(LATENT_DIM, model_cfg)
    history = tc.train_conditional(resumed, cases, replace(cfg, epochs=3))
    assert len(history.total) == 3 and np.isfinite(history.total).all()
    steps = [event for line in Path(cfg.events_path).read_text().splitlines()
             if (event := json.loads(line))["event"] == "step"]
    assert [step["epoch"] for step in steps] == [1, 2, 3]
    for step in steps:
        raw = step["raw_residual_rms"]
        assert raw["surface_heat_flux"] is None
        assert all(raw[name] is not None for name in ACTIVE | set(tc.PDE_RAW_NAMES))
    with pytest.raises(ValueError, match="surface selection or loss weights differ"):
        tc.train_conditional(resumed, cases, replace(cfg, epochs=4, surface_term="all"))
