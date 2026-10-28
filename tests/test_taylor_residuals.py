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

from dataclasses import replace

import pytest

torch = pytest.importorskip("torch")

from wrf_pinn.config.physics import DEFAULT_PHYSICS  # noqa: E402
from wrf_pinn.config.scaling import ResidualScalingConfig, VariableScale  # noqa: E402
from wrf_pinn.models.conditional import ConditionalModel, ConditionalModelConfig  # noqa: E402
from wrf_pinn.physics.grid_geometry import sgs_filter_width, terrain_height_jet
from wrf_pinn.physics.residuals_boundary import no_penetration_z_wall_residuals
from wrf_pinn.physics.residuals_pde import (  # noqa: E402
    cartesian_zero_forcing_residuals, cartesian_zero_forcing_residuals_taylor,
)
from wrf_pinn.physics.taylor import PAIRS, coordinate_jet, taylor_forward  # noqa: E402

SCALING = ResidualScalingConfig(
    x=VariableScale(500.0, 500.0), y=VariableScale(500.0, 500.0),
    z=VariableScale(700.0, 600.0), t=VariableScale(300.0, 300.0),
    u=VariableScale(5.0, 2.0), v=VariableScale(2.0, 2.0),
    w=VariableScale(0.0, 0.5), theta=VariableScale(300.0, 2.0),
    p_prime=VariableScale(0.0, 50.0), q_v=VariableScale(8.0, 1.0),
    e_sgs=VariableScale(0.5, 0.1),
)
PHYSICS = replace(DEFAULT_PHYSICS, q_v_to_kg_per_kg=1e-3)
H = 4027.5
LATENT = 5
EXPECTED_RESIDUALS = {
    "mass", "x_momentum", "y_momentum", "z_momentum",
    "potential_temperature", "water_vapor",
}

def _setup(seed=0, stable=False):
    torch.manual_seed(seed)
    model = ConditionalModel(LATENT, ConditionalModelConfig(
        hidden_layers=3, hidden_width=32,
        tke_offset=SCALING.e_sgs.offset, tke_scale=SCALING.e_sgs.scale,
        tke_initial_value=0.5,
    )).double()
    coords = 2 * torch.rand(96, 4, dtype=torch.float64) - 1
    z_rows = torch.randn(
        3, LATENT, dtype=torch.float64
    ).repeat_interleave(32, dim=0)
    if stable:
        hidden = model.net[:-1]
        dz = torch.zeros(1, 4, dtype=torch.float64)
        dz[0, 2] = 1e-4
        x = lambda c: torch.cat([c, z_rows], dim=1)
        with torch.no_grad():
            slope = (
                (hidden(x(coords + dz)) - hidden(x(coords - dz))) / 2e-4
            ).mean(0)
            model.net[-1].weight[
                PHYSICS.variable_index("theta")
            ] = 0.5 * slope / slope.norm()
    return model, coords, z_rows


def _terrain(coords, sloped=True):
    """Known physical terrain and its analytic slopes."""
    if not sloped:
        return (
            coords.new_zeros((len(coords), 1)),
            coords.new_zeros((len(coords), 2)),
        )
    x = SCALING.x.offset + SCALING.x.scale * coords[:, 0:1]
    y = SCALING.y.offset + SCALING.y.scale * coords[:, 1:2]
    h = 20.0 + 0.01 * x + 0.02 * y
    slopes = coords.new_tensor([0.01, 0.02]).expand(len(coords), -1)
    return h, slopes


def _width(coords, taylor):
    altitude = SCALING.z.offset + SCALING.z.scale * coords[:, 2:3]
    h, slopes = _terrain(coords)
    if taylor:
        altitude = coordinate_jet(altitude, axis=2)
        h = terrain_height_jet(h, slopes)
    return sgs_filter_width(altitude, h, PHYSICS, H)


def _jet(model, coords, z, **kwargs):
    return taylor_forward(
        model.net, coords, z, SCALING.coordinate_scales(),
        output_config=model.config, **kwargs,
    )


def _pde(model, coords, z, taylor, sizes=None):
    if taylor:
        return cartesian_zero_forcing_residuals_taylor(
            model.net, coords, z, physics=PHYSICS, scaling=SCALING,
            sizes=sizes, output_config=model.config,
            filter_width=_width(coords, True),
        )
    return cartesian_zero_forcing_residuals(
        coords, model(coords, z, sizes),
        physics=PHYSICS, scaling=SCALING,
        filter_width=_width(coords, False),
    )


def test_taylor_forward_matches_autograd_derivatives():
    model, coords, z = _setup()
    jet = _jet(model, coords, z)
    c = coords.clone().requires_grad_(True)
    out = model(c, z)
    torch.testing.assert_close(jet.v, out, rtol=1e-12, atol=1e-12)

    e = (
        SCALING.e_sgs.offset
        + SCALING.e_sgs.scale * out[:, PHYSICS.variable_index("e_sgs")]
    )
    assert (e >= model.config.tke_floor).all()

    scales = out.new_tensor(SCALING.coordinate_scales())
    for ch in range(out.shape[1]):
        g = torch.autograd.grad(
            out[:, ch].sum(), c, create_graph=True
        )[0] / scales
        torch.testing.assert_close(
            jet.d[:, :, ch], g.T, rtol=1e-10, atol=1e-12
        )
        for k, (i, j) in enumerate(PAIRS):
            gij = torch.autograd.grad(
                g[:, i].sum(), c, retain_graph=True
            )[0][:, j] / scales[j]
            torch.testing.assert_close(
                jet.h[k, :, ch], gij, rtol=1e-9, atol=1e-11
            )


@pytest.mark.parametrize("stable", [False, True])
def test_taylor_residuals_match_nested_autograd(stable):
    model, coords, z = _setup(seed=1, stable=stable)
    frac = (
        _jet(model, coords, z).d[2, :, PHYSICS.variable_index("theta")] > 0
    ).double().mean().item()
    assert (frac > 0.9) if stable else (frac < 0.1), f"stable fraction {frac}"

    old = _pde(model, coords.clone().requires_grad_(True), z, False)
    new = _pde(model, coords, z, True)
    assert set(new) == set(old) == EXPECTED_RESIDUALS
    for name in old:
        assert torch.isfinite(old[name]).all()
        torch.testing.assert_close(
            new[name], old[name], rtol=1e-9, atol=1e-10
        )


@pytest.mark.parametrize("stable", [False, True])
def test_taylor_residual_parameter_gradients_match(stable):
    def grads(taylor):
        model, coords, z = _setup(seed=2, stable=stable)
        coords.requires_grad_(not taylor)
        res = _pde(model, coords, z, taylor)
        loss = torch.stack([r.square().mean() for r in res.values()]).mean()
        return torch.autograd.grad(loss, list(model.parameters()))

    for a, b in zip(grads(True), grads(False)):
        torch.testing.assert_close(a, b, rtol=1e-8, atol=1e-10)


def test_taylor_rejects_unsupported_activation():
    model = ConditionalModel(LATENT, ConditionalModelConfig(
        hidden_layers=2, hidden_width=8, activation="silu",
    ))
    with pytest.raises(TypeError, match="Linear/Tanh"):
        _jet(model, torch.zeros(4, 4), torch.zeros(4, LATENT))


def _surface(model, coords, z, taylor, sloped, check=True):
    wall = coords.clone()
    h, slopes = _terrain(wall, sloped)
    wall[:, 2] = (h[:, 0] - SCALING.z.offset) / SCALING.z.scale
    reference = wall.clone()
    reference[:, 2] += (
        PHYSICS.sgs_reference_z[0] * (1 - h[:, 0] / H) / SCALING.z.scale
    )

    gen = torch.Generator().manual_seed(0)
    fric = 0.1 + 0.4 * torch.rand(
        len(wall), 1, generator=gen, dtype=torch.float64
    )
    heat = 0.05 * torch.randn(
        len(wall), 1, generator=gen, dtype=torch.float64
    )
    if taylor:
        jet = _jet(model, wall, z, second=False)
        assert jet.h is None
        state, jac = jet.v, jet.d
    else:
        wall.requires_grad_(True)
        state, jac = model(wall, z), None

    return no_penetration_z_wall_residuals(
        coordinates=wall, state=state, reference_coordinates=reference,
        reference_state=model(reference, z),
        physics=PHYSICS, scaling=SCALING,
        fric_vel=fric, ht_flux=heat,
        state_jacobian=jac, check_values=check,
        terrain_height=h, terrain_slopes=slopes,
        filter_width=sgs_filter_width(h, h, PHYSICS, H),
    )


@pytest.mark.parametrize("sloped", [False, True])
def test_surface_residuals_with_taylor_jacobian_match_autograd(sloped):
    def run(taylor):
        model, coords, z = _setup(seed=3)
        res = _surface(model, coords, z, taylor, sloped)
        loss = sum(r.square().mean() for r in res.values())
        return res, torch.autograd.grad(loss, list(model.parameters()))

    (new, ga), (old, gb) = run(True), run(False)
    for name in old:
        assert old[name].numel() > 0
        torch.testing.assert_close(
            new[name], old[name], rtol=1e-10, atol=1e-12
        )
    for a, b in zip(ga, gb):
        torch.testing.assert_close(a, b, rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize("sloped", [False, True])
def test_surface_residuals_unchecked_match_checked(sloped):
    model, coords, z = _setup(seed=3)
    checked = _surface(model, coords, z, True, sloped, check=True)
    unchecked = _surface(model, coords, z, True, sloped, check=False)
    for name in checked:
        torch.testing.assert_close(
            unchecked[name], checked[name], rtol=0, atol=0
        )


def test_compiled_taylor_residuals_match_eager():
    model, coords, z = _setup(seed=1, stable=True)
    kwargs = dict(
        physics=PHYSICS, scaling=SCALING,
        output_config=model.config, filter_width=_width(coords, True),
    )
    eager = cartesian_zero_forcing_residuals_taylor(
        model.net, coords, z, **kwargs
    )
    compiled = torch.compile(cartesian_zero_forcing_residuals_taylor)(
        model.net, coords, z, **kwargs
    )
    for name in eager:
        torch.testing.assert_close(
            compiled[name], eager[name], rtol=1e-12, atol=1e-14
        )


def test_taylor_per_case_latent_matches_per_point_rows():
    model, coords, z = _setup(seed=4)
    n = len(coords) // 3
    z_case = z[::n]
    a = _jet(model, coords, z)
    b = _jet(model, coords, z_case, sizes=n)
    for x, y in ((a.v, b.v), (a.d, b.d), (a.h, b.h)):
        torch.testing.assert_close(x, y, rtol=1e-12, atol=1e-12)

    ra = _pde(model, coords, z, True)
    rb = _pde(model, coords, z_case, True, sizes=n)
    for name in ra:
        torch.testing.assert_close(
            ra[name], rb[name], rtol=1e-10, atol=1e-12
        )