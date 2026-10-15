"""The product-rule PDE residuals equal the one-backward-pass-per-product originals.

``cartesian_zero_forcing_residuals`` builds the gradients of transport products
(rho_d*u, rho_d*u*v, rho_d*u*theta, ...) with the product rule from gradients it
already has, instead of a backward pass through the network per product. These
tests pin that the result is the same math: the full residuals match the original
(autograd of each product, patched back in) to float64 rounding, on a smooth
network with physically plausible outputs, under non-trivial scaling.
"""

from __future__ import annotations

import dataclasses

import pytest

torch = pytest.importorskip("torch")

import wrf_pinn.physics.residuals_pde as rp  # noqa: E402
from wrf_pinn.config.physics import DEFAULT_PHYSICS  # noqa: E402
from wrf_pinn.config.scaling import ResidualScalingConfig, VariableScale  # noqa: E402

# Normalized coords/state in about [-1, 1]; physical = offset + scale * normalized.
SCALING = ResidualScalingConfig(
    x=VariableScale(500.0, 500.0), y=VariableScale(500.0, 500.0),
    z=VariableScale(800.0, 700.0), t=VariableScale(300.0, 300.0),
    u=VariableScale(5.0, 2.0), v=VariableScale(2.0, 2.0), w=VariableScale(0.0, 0.5),
    theta=VariableScale(300.0, 2.0), p_prime=VariableScale(0.0, 50.0),
    q_v=VariableScale(8.0, 1.0), e_sgs=VariableScale(0.5, 0.1),
)
PHYSICS = dataclasses.replace(DEFAULT_PHYSICS, q_v_to_kg_per_kg=1e-3)  # q_v in g/kg


def _inputs(seed: int = 0, n: int = 64):
    torch.manual_seed(seed)
    net = torch.nn.Sequential(torch.nn.Linear(4, 32), torch.nn.Tanh(),
                              torch.nn.Linear(32, 32), torch.nn.Tanh(),
                              torch.nn.Linear(32, PHYSICS.state_dim), torch.nn.Tanh()).double()
    coords = (2 * torch.rand(n, 4, dtype=torch.float64) - 1).requires_grad_(True)
    return net, coords


def _reference_product_gradient(coords):
    """The original: one autograd backward pass per product, same factor order."""
    def product_gradient(*factors):
        field = factors[0][0]
        for f, _ in factors[1:]:
            field = field * f
        return rp._physical_gradient(field, coords, SCALING)
    return product_gradient


def test_product_gradient_matches_autograd():
    net, coords = _inputs()
    out = net(coords)
    fields = [out[:, i:i + 1] for i in range(3)]
    grads = [rp._physical_gradient(f, coords, SCALING) for f in fields]
    pairs = list(zip(fields, grads))
    for k in (1, 2, 3):
        expected = _reference_product_gradient(coords)(*pairs[:k])
        torch.testing.assert_close(rp._product_gradient(*pairs[:k]), expected,
                                   rtol=1e-12, atol=1e-14)


def test_residuals_match_original_formulation(monkeypatch):
    net, coords = _inputs(seed=1)
    state = net(coords)
    new = rp.cartesian_zero_forcing_residuals(coords, state, physics=PHYSICS, scaling=SCALING)

    monkeypatch.setattr(rp, "_product_gradient", _reference_product_gradient(coords))
    old = rp.cartesian_zero_forcing_residuals(coords, state, physics=PHYSICS, scaling=SCALING)

    assert new.keys() == old.keys()
    for name in new:
        assert torch.isfinite(old[name]).all(), f"{name}: reference not finite"
        torch.testing.assert_close(new[name], old[name], rtol=1e-10, atol=1e-12,
                                   msg=lambda m, n=name: f"{n}: {m}")


def test_residual_gradients_match_original_formulation(monkeypatch):
    """Training backpropagates through the residuals: parameter gradients must match too."""
    def param_grads(product_gradient=None):
        net, coords = _inputs(seed=2)
        if product_gradient is not None:
            monkeypatch.setattr(rp, "_product_gradient", product_gradient(coords))
        res = rp.cartesian_zero_forcing_residuals(coords, net(coords), physics=PHYSICS,
                                                  scaling=SCALING)
        loss = sum(r.square().mean() for r in res.values())
        return torch.autograd.grad(loss, list(net.parameters()))

    new = param_grads()
    old = param_grads(_reference_product_gradient)
    for g_new, g_old in zip(new, old):
        torch.testing.assert_close(g_new, g_old, rtol=1e-9, atol=1e-12)


def test_residuals_use_19_backward_passes(monkeypatch):
    """6 primitives + rho_d + 12 SGS stress/flux terms; was 36 before the product rule."""
    calls = []
    original = rp._gradient
    monkeypatch.setattr(rp, "_gradient", lambda f, c: (calls.append(1), original(f, c))[1])
    net, coords = _inputs()
    rp.cartesian_zero_forcing_residuals(coords, net(coords), physics=PHYSICS, scaling=SCALING)
    assert len(calls) == 19
