"""Direct tests of the residual scaling implementation.

The model works in normalized coordinates, but the PDE residual must be enforced
in physical units. ``ResidualScalingConfig`` records the affine map
``physical = offset + scale * normalized`` per variable, and the residual applies
the chain rule so gradients are taken with respect to physical coordinates:

    d(field)/d(x_physical) = d(field)/d(x_normalized) / scale

These tests pin that math to known analytic answers, independent of training, so
we know the scaling is actually applied (and applied correctly) rather than
inferring it from a converged model.
"""

from __future__ import annotations

import torch
import pytest
from wrf_pinn.config.physics import DEFAULT_PHYSICS
from wrf_pinn.config.scaling import ResidualScalingConfig, VariableScale
from wrf_pinn.physics.residuals_pde import (
    _physical_gradient,
    _to_physical_eddy_viscosity,
)


def test_physical_gradient_applies_coordinate_scale(report):
    """d(x_norm)/d(x_phys) equals 1/scale, not 1.

    With ``x_physical = scale * x_normalized`` and field ``f = x_normalized``,
    the physical-coordinate derivative must be ``1/scale``. This is the check
    that fails if the scaling were ignored (it would give 1.0).
    """

    coordinates = torch.rand(16, 4, requires_grad=True)
    field = coordinates[:, 0:1] * 1.0  # f = x_normalized (graph-connected)

    scaled = _physical_gradient(
        field, coordinates, ResidualScalingConfig(x=VariableScale(0.0, 1000.0))
    )
    identity = _physical_gradient(field, coordinates, ResidualScalingConfig())

    assert torch.allclose(scaled[:, 0], torch.full((16,), 1.0e-3), atol=1e-6)
    assert torch.allclose(identity[:, 0], torch.ones(16), atol=1e-6)
    report(
        "physical gradient applies coordinate scale",
        scaled_dx=scaled[:, 0].detach().mean().reshape(1),
        identity_dx=identity[:, 0].detach().mean().reshape(1),
    )


def test_eddy_viscosity_transform_respects_bounds():
    raw = torch.tensor([[-10.0], [0.0], [10.0]])

    k_m = _to_physical_eddy_viscosity(
        raw,
        DEFAULT_PHYSICS,
    )

    k_m_min = DEFAULT_PHYSICS.constants.eddy_viscosity_min
    k_m_max = DEFAULT_PHYSICS.constants.eddy_viscosity_max
    midpoint = 0.5 * (k_m_min + k_m_max)

    assert (k_m > k_m_min).all()
    assert (k_m < k_m_max).all()
    assert float(k_m[1]) == pytest.approx(midpoint)
    assert k_m[0] < k_m[1] < k_m[2]

