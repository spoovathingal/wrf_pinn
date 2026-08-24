"""Tests for loss-mode conditions: activation, weights, and loss flow-through.

Two layers are checked:

1. ``ConditionsConfig`` / ``ConditionSpec`` logic in isolation — the
   active/inactive and nonnegative-weight invariants, the derived accessors, and
   the default on/off contract.
2. That the on/off state and weight actually reach the assembled loss: an
   inactive condition contributes zero to the total, and an active condition's
   term scales with its weight.
"""

from __future__ import annotations

import pytest
import torch

from wrf_pinn.config.conditions import (
    DEFAULT_CONDITIONS,
    ConditionSpec,
    ConditionsConfig,
)
from wrf_pinn.training.losses import assemble_pinn_loss


# --------------------------------------------------------------------------- #
# ConditionSpec invariants
# --------------------------------------------------------------------------- #


def test_condition_spec_active_defaults():
    """An active spec with a positive weight is valid."""

    spec = ConditionSpec("pde", active=True, weight=2.5)
    assert spec.active is True
    assert spec.weight == 2.5


def test_condition_spec_inactive_requires_zero_weight():
    """An inactive spec must have weight 0.0."""

    ConditionSpec("boundary", active=False, weight=0.0)  # valid

    with pytest.raises(ValueError, match="weight 0.0 when inactive"):
        ConditionSpec("boundary", active=False, weight=1.0)


def test_condition_spec_rejects_negative_weight():
    """A negative weight is rejected regardless of activation."""

    with pytest.raises(ValueError, match="nonnegative"):
        ConditionSpec("pde", active=True, weight=-0.1)


# --------------------------------------------------------------------------- #
# ConditionsConfig accessors and default contract
# --------------------------------------------------------------------------- #


def test_default_conditions_on_off_contract():
    """The default enables pde and the three data terms; disables boundary."""

    weights = DEFAULT_CONDITIONS.weights()
    assert DEFAULT_CONDITIONS.pde.active and weights["pde"] == 1.0
    for name in ("inlet", "simulation", "sensor"):
        assert getattr(DEFAULT_CONDITIONS, name).active and weights[name] == 1.0
    assert not DEFAULT_CONDITIONS.boundary.active and weights["boundary"] == 0.0


def test_active_returns_only_active_in_order():
    """``active`` returns active specs in canonical order."""

    active_names = [spec.name for spec in DEFAULT_CONDITIONS.active]
    assert active_names == ["pde", "inlet", "simulation", "sensor"]


def test_as_tuple_order_is_canonical():
    """``as_tuple`` returns all specs in canonical order."""

    names = [spec.name for spec in DEFAULT_CONDITIONS.as_tuple()]
    assert names == ["pde", "boundary", "inlet", "simulation", "sensor"]


def test_toggling_a_condition_updates_active_set():
    """Turning a condition on/off is reflected in the active set."""

    config = ConditionsConfig(
        boundary=ConditionSpec("boundary", active=True, weight=1.0),
    )
    assert "boundary" in [spec.name for spec in config.active]


# --------------------------------------------------------------------------- #
# On/off and weight actually reach the assembled loss
# --------------------------------------------------------------------------- #


def _sim_errors(value: float) -> dict[str, torch.Tensor]:
    return {"simulation": torch.full((4, 1), value)}


def _data_only(weight: float) -> ConditionsConfig:
    """Only the simulation data term active, at ``weight``."""

    off = lambda name: ConditionSpec(name, active=False, weight=0.0)
    return ConditionsConfig(
        pde=off("pde"),
        inlet=off("inlet"),
        sensor=off("sensor"),
        simulation=ConditionSpec("simulation", active=True, weight=weight),
    )


def test_inactive_condition_contributes_zero():
    """An inactive condition adds nothing to the total loss."""

    loss = assemble_pinn_loss(
        simulation_errors=_sim_errors(2.0),
        conditions=_data_only(1.0),
    )
    assert float(loss.weighted_terms["pde"]) == 0.0
    assert float(loss.terms["pde"]) == 0.0


def test_weight_scales_the_term():
    """Doubling an active condition's weight doubles its weighted term."""

    errors = _sim_errors(3.0)
    loss_base = assemble_pinn_loss(simulation_errors=errors, conditions=_data_only(1.0))
    loss_scaled = assemble_pinn_loss(simulation_errors=errors, conditions=_data_only(2.0))

    weighted_base = float(loss_base.weighted_terms["simulation"])
    weighted_scaled = float(loss_scaled.weighted_terms["simulation"])
    assert float(loss_base.terms["simulation"]) == pytest.approx(
        float(loss_scaled.terms["simulation"])
    )
    assert weighted_scaled == pytest.approx(2.0 * weighted_base)
