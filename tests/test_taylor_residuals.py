"""Forward-Taylor PDE residuals equal the nested-autograd originals.

``cartesian_zero_forcing_residuals_taylor`` gets the network's coordinate
derivatives from one forward Taylor pass and differentiates the pointwise physics
with jets, instead of one ``autograd.grad`` per field. These tests pin, in float64
on a ConditionalModel with physically plausible outputs and non-trivial scaling:

- the propagated first and second derivatives equal autograd's;
- all six residuals equal ``cartesian_zero_forcing_residuals``;
- the parameter gradients of a residual loss (what training uses) are equal.
"""

from __future__ import annotations

import dataclasses

import pytest

torch = pytest.importorskip("torch")

from wrf_pinn.config.physics import DEFAULT_PHYSICS  # noqa: E402
from wrf_pinn.config.scaling import ResidualScalingConfig, VariableScale  # noqa: E402
from wrf_pinn.models.conditional import ConditionalModel, ConditionalModelConfig  # noqa: E402
from wrf_pinn.physics.residuals_pde import (  # noqa: E402
    cartesian_zero_forcing_residuals, cartesian_zero_forcing_residuals_taylor,
)
from wrf_pinn.physics.taylor import PAIRS, taylor_forward  # noqa: E402

SCALING = ResidualScalingConfig(
    x=VariableScale(500.0, 500.0), y=VariableScale(500.0, 500.0),
    z=VariableScale(800.0, 700.0), t=VariableScale(300.0, 300.0),
    u=VariableScale(5.0, 2.0), v=VariableScale(2.0, 2.0), w=VariableScale(0.0, 0.5),
    theta=VariableScale(300.0, 2.0), p_prime=VariableScale(0.0, 50.0),
    q_v=VariableScale(8.0, 1.0), e_sgs=VariableScale(0.5, 0.1),
)
PHYSICS = dataclasses.replace(DEFAULT_PHYSICS, q_v_to_kg_per_kg=1e-3)  # q_v in g/kg
LATENT = 5


def _setup(seed: int = 0, n: int = 96, layers: int = 3, width: int = 32, stable: bool = False):
    """A small model and inputs. Random weights give d(theta)/dz < 0 (unstable,
    N^2 <= 0) everywhere; ``stable`` re-aims theta's output weights along the hidden
    units' z-derivative so theta increases with height (N^2 > 0), exercising the
    stable branch of the SGS mixing length."""
    torch.manual_seed(seed)
    model = ConditionalModel(LATENT, ConditionalModelConfig(hidden_layers=layers,
                                                            hidden_width=width)).double()
    coords = 2 * torch.rand(n, 4, dtype=torch.float64) - 1
    z_rows = torch.randn(3, LATENT, dtype=torch.float64).repeat_interleave(n // 3, dim=0)
    if stable:
        hidden = model.net[:-1]
        dz = torch.zeros(1, 4, dtype=torch.float64)
        dz[0, 2] = 1e-4
        x = lambda c: torch.cat([c, z_rows], dim=1)
        with torch.no_grad():
            slope = ((hidden(x(coords + dz)) - hidden(x(coords - dz))) / 2e-4).mean(dim=0)
            model.net[-1].weight[PHYSICS.variable_index("theta")] = 0.5 * slope / slope.norm()
    return model, coords, z_rows


def _stability_fraction(model, coords, z_rows) -> float:
    jet = taylor_forward(model.net, coords, z_rows, SCALING.coordinate_scales())
    return (jet.d[2, :, PHYSICS.variable_index("theta")] > 0).double().mean().item()


def test_taylor_forward_matches_autograd_derivatives():
    model, coords, z_rows = _setup()
    jet = taylor_forward(model.net, coords, z_rows, SCALING.coordinate_scales())

    c = coords.clone().requires_grad_(True)
    out = model(c, z_rows)
    torch.testing.assert_close(jet.v, out, rtol=1e-12, atol=1e-12)
    scales = torch.tensor(SCALING.coordinate_scales(), dtype=torch.float64)
    for ch in range(out.shape[1]):
        g = torch.autograd.grad(out[:, ch].sum(), c, create_graph=True)[0] / scales
        torch.testing.assert_close(jet.d[:, :, ch], g.T, rtol=1e-10, atol=1e-12)
        for k, (i, j) in enumerate(PAIRS):
            gij = torch.autograd.grad(g[:, i].sum(), c, retain_graph=True)[0][:, j] / scales[j]
            torch.testing.assert_close(jet.h[k, :, ch], gij, rtol=1e-9, atol=1e-11)


@pytest.mark.parametrize("stable", [False, True])
def test_taylor_residuals_match_nested_autograd(stable):
    model, coords, z_rows = _setup(seed=1, stable=stable)
    frac = _stability_fraction(model, coords, z_rows)
    assert (frac > 0.9) if stable else (frac < 0.1), f"stable fraction {frac}"
    c = coords.clone().requires_grad_(True)
    old = cartesian_zero_forcing_residuals(c, model(c, z_rows), physics=PHYSICS, scaling=SCALING)
    new = cartesian_zero_forcing_residuals_taylor(model.net, coords, z_rows,
                                                  physics=PHYSICS, scaling=SCALING)
    assert new.keys() == old.keys()
    for name in old:
        assert torch.isfinite(old[name]).all(), f"{name}: reference not finite"
        torch.testing.assert_close(new[name], old[name], rtol=1e-9, atol=1e-10,
                                   msg=lambda m, n=name: f"{n}: {m}")


@pytest.mark.parametrize("stable", [False, True])
def test_taylor_residual_parameter_gradients_match(stable):
    def grads(taylor: bool):
        model, coords, z_rows = _setup(seed=2, stable=stable)
        if taylor:
            res = cartesian_zero_forcing_residuals_taylor(model.net, coords, z_rows,
                                                          physics=PHYSICS, scaling=SCALING)
        else:
            c = coords.clone().requires_grad_(True)
            res = cartesian_zero_forcing_residuals(c, model(c, z_rows), physics=PHYSICS,
                                                   scaling=SCALING)
        loss = sum(r.square().mean() for r in res.values())
        return torch.autograd.grad(loss, list(model.parameters()))

    for g_new, g_old in zip(grads(True), grads(False)):
        torch.testing.assert_close(g_new, g_old, rtol=1e-8, atol=1e-10)


def test_taylor_rejects_unsupported_activation():
    model = ConditionalModel(LATENT, ConditionalModelConfig(hidden_layers=2, hidden_width=8,
                                                            activation="silu"))
    with pytest.raises(TypeError):
        taylor_forward(model.net, torch.zeros(4, 4), torch.zeros(4, LATENT), (1, 1, 1, 1))


def test_surface_residuals_with_taylor_jacobian_match_autograd():
    """The wall residual reads its gradients from a first-order Taylor pass."""
    from wrf_pinn.physics.residuals_boundary import no_penetration_z_wall_residuals

    def run(taylor: bool):
        model, coords, z_rows = _setup(seed=3)
        wall = coords.clone()
        wall[:, 2] = (0.0 - SCALING.z.offset) / SCALING.z.scale          # z = 0 physical
        reference = wall.clone()
        reference[:, 2] = (PHYSICS.constants.surface_reference_height
                           - SCALING.z.offset) / SCALING.z.scale
        gen = torch.Generator().manual_seed(0)
        fric = 0.1 + 0.4 * torch.rand(wall.shape[0], 1, generator=gen, dtype=torch.float64)
        heat = 0.05 * torch.randn(wall.shape[0], 1, generator=gen, dtype=torch.float64)
        if taylor:
            jet = taylor_forward(model.net, wall, z_rows, SCALING.coordinate_scales(), second=False)
            assert jet.h is None
            state, jac = jet.v, jet.d
        else:
            wall = wall.requires_grad_(True)
            state, jac = model(wall, z_rows), None
        res = no_penetration_z_wall_residuals(
            coordinates=wall, state=state, reference_coordinates=reference,
            reference_state=model(reference, z_rows), physics=PHYSICS, scaling=SCALING,
            fric_vel=fric, ht_flux=heat, state_jacobian=jac)
        loss = sum(r.square().mean() for r in res.values())
        return res, torch.autograd.grad(loss, list(model.parameters()))

    (new, g_new), (old, g_old) = run(True), run(False)
    for name in old:
        assert old[name].numel() > 0
        torch.testing.assert_close(new[name], old[name], rtol=1e-10, atol=1e-12)
    for a, b in zip(g_new, g_old):
        torch.testing.assert_close(a, b, rtol=1e-9, atol=1e-12)


def test_compiled_taylor_residuals_match_eager():
    """torch.compile traces the jet residuals as one graph and reproduces them."""
    model, coords, z_rows = _setup(seed=1, stable=True)
    eager = cartesian_zero_forcing_residuals_taylor(model.net, coords, z_rows,
                                                    physics=PHYSICS, scaling=SCALING)
    compiled = torch.compile(cartesian_zero_forcing_residuals_taylor)(
        model.net, coords, z_rows, physics=PHYSICS, scaling=SCALING)
    for name in eager:
        torch.testing.assert_close(compiled[name], eager[name], rtol=1e-12, atol=1e-14)


def test_taylor_per_case_latent_matches_per_point_rows():
    """taylor_forward and the residuals give the same results with z per case."""
    model, coords, z_rows = _setup(seed=4)
    z_case = z_rows[::coords.shape[0] // 3]                       # the 3 distinct cases
    n = coords.shape[0] // 3
    a = taylor_forward(model.net, coords, z_rows, SCALING.coordinate_scales())
    b = taylor_forward(model.net, coords, z_case, SCALING.coordinate_scales(), sizes=n)
    for x, y in ((a.v, b.v), (a.d, b.d), (a.h, b.h)):
        torch.testing.assert_close(x, y, rtol=1e-12, atol=1e-12)
    ra = cartesian_zero_forcing_residuals_taylor(model.net, coords, z_rows, physics=PHYSICS,
                                                 scaling=SCALING)
    rb = cartesian_zero_forcing_residuals_taylor(model.net, coords, z_case, physics=PHYSICS,
                                                 scaling=SCALING, sizes=n)
    for name in ra:
        torch.testing.assert_close(ra[name], rb[name], rtol=1e-10, atol=1e-12)
