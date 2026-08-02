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

from wrf_pinn.config.scaling import ResidualScalingConfig, VariableScale
from wrf_pinn.physics.residuals_pde import (
    _physical_gradient,
    cartesian_zero_forcing_residuals,
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


def test_uniform_field_zero_residual_under_scaling(report):
    """A spatially constant field has zero residual for any scaling.

    A constant state has zero spatial and temporal gradients, so every residual
    term must vanish regardless of the (nonzero) coordinate or state scales.
    """

    # A zero-weight linear layer yields a constant output that is still connected
    # to the coordinate graph, mimicking a converged constant model state.
    layer = torch.nn.Linear(4, 4)
    with torch.no_grad():
        layer.weight.zero_()
        layer.bias.copy_(torch.tensor([0.3, -0.1, 0.0, 0.5]))

    coordinates = torch.rand(32, 4, requires_grad=True)
    state = layer(coordinates)

    scaling = ResidualScalingConfig(
        x=VariableScale(0.0, 1000.0),
        y=VariableScale(0.0, 950.0),
        z=VariableScale(0.0, 1062.0),
        t=VariableScale(0.0, 600.0),
        u=VariableScale(0.0, 10.0),
        v=VariableScale(0.0, 10.0),
        w=VariableScale(0.0, 1.0),
        rho=VariableScale(0.0, 1.225),
    )
    residuals = cartesian_zero_forcing_residuals(coordinates, state, scaling=scaling)

    max_residual = max(float(r.detach().abs().max()) for r in residuals.values())
    assert max_residual < 1e-5
    report(
        "uniform field, non-identity scaling: residual is zero",
        max_abs_residual=torch.tensor([max_residual]),
    )


def test_scaling_changes_the_residual(report):
    """Non-identity scaling produces a different residual than identity.

    For a non-constant field, the scaled and unscaled residuals must differ,
    confirming the scaling actually participates in the computation rather than
    being silently dropped.
    """

    layer = torch.nn.Linear(4, 4)
    torch.manual_seed(0)
    coordinates = torch.rand(24, 4, requires_grad=True)
    state = layer(coordinates)  # non-constant field

    identity = cartesian_zero_forcing_residuals(
        coordinates, state, scaling=ResidualScalingConfig()
    )
    scaled = cartesian_zero_forcing_residuals(
        coordinates,
        state,
        scaling=ResidualScalingConfig(
            x=VariableScale(0.0, 1000.0),
            z=VariableScale(0.0, 1062.0),
            u=VariableScale(0.0, 10.0),
        ),
    )

    difference = max(
        float((identity[key] - scaled[key]).detach().abs().max()) for key in identity
    )
    assert difference > 1e-3
    report(
        "scaling changes the residual (vs identity)",
        max_abs_difference=torch.tensor([difference]),
    )
