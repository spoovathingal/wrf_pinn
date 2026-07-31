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
    """The default enables pde and flow_field_data; disables the rest."""

    weights = DEFAULT_CONDITIONS.weights()
    assert DEFAULT_CONDITIONS.pde.active and weights["pde"] == 1.0
    assert DEFAULT_CONDITIONS.flow_field_data.active
    assert weights["flow_field_data"] == 1.0
    assert not DEFAULT_CONDITIONS.boundary.active and weights["boundary"] == 0.0
    assert not DEFAULT_CONDITIONS.sensor_data.active and weights["sensor_data"] == 0.0


def test_active_returns_only_active_in_order():
    """``active`` returns active specs in canonical order."""

    active_names = [spec.name for spec in DEFAULT_CONDITIONS.active]
    assert active_names == ["pde", "flow_field_data"]


def test_as_tuple_order_is_canonical():
    """``as_tuple`` returns all four specs in canonical order."""

    names = [spec.name for spec in DEFAULT_CONDITIONS.as_tuple()]
    assert names == ["pde", "boundary", "sensor_data", "flow_field_data"]


def test_toggling_a_condition_updates_active_set():
    """Turning a condition on/off is reflected in the active set."""

    config = ConditionsConfig(
        sensor_data=ConditionSpec("sensor_data", active=True, weight=1.0),
    )
    assert "sensor_data" in [spec.name for spec in config.active]


# --------------------------------------------------------------------------- #
# On/off and weight actually reach the assembled loss
# --------------------------------------------------------------------------- #


def _flow_errors(value: float) -> dict[str, torch.Tensor]:
    return {"flow_field": torch.full((4, 1), value)}


def test_inactive_condition_contributes_zero():
    """An inactive condition adds nothing to the total loss."""

    conditions = ConditionsConfig(
        pde=ConditionSpec("pde", active=False, weight=0.0),
        flow_field_data=ConditionSpec("flow_field_data", active=True, weight=1.0),
    )
    loss = assemble_pinn_loss(
        flow_field_data_errors=_flow_errors(2.0),
        conditions=conditions,
    )
    assert float(loss.weighted_terms["pde"]) == 0.0
    assert float(loss.terms["pde"]) == 0.0


def test_weight_scales_the_term():
    """Doubling an active condition's weight doubles its weighted term."""

    base = ConditionsConfig(
        pde=ConditionSpec("pde", active=False, weight=0.0),
        flow_field_data=ConditionSpec("flow_field_data", active=True, weight=1.0),
    )
    scaled = ConditionsConfig(
        pde=ConditionSpec("pde", active=False, weight=0.0),
        flow_field_data=ConditionSpec("flow_field_data", active=True, weight=2.0),
    )
    errors = _flow_errors(3.0)

    loss_base = assemble_pinn_loss(flow_field_data_errors=errors, conditions=base)
    loss_scaled = assemble_pinn_loss(flow_field_data_errors=errors, conditions=scaled)

    weighted_base = float(loss_base.weighted_terms["flow_field_data"])
    weighted_scaled = float(loss_scaled.weighted_terms["flow_field_data"])
    # unweighted term is identical; only the weight differs
    assert float(loss_base.terms["flow_field_data"]) == pytest.approx(
        float(loss_scaled.terms["flow_field_data"])
    )
    assert weighted_scaled == pytest.approx(2.0 * weighted_base)
