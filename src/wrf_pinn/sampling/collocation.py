"""Collocation-point generation for PDE residual training.

The objects in ``wrf_pinn.config.sampling`` describe how sampling should be
done. This module performs the runtime work of turning a continuous Cartesian
domain into actual ``(x, y, z, t)`` points where the PDE residual is evaluated.
"""

from __future__ import annotations

import math

import torch

from wrf_pinn.config.domain import CartesianWRFDomain
from wrf_pinn.config.sampling import CollocationSamplingConfig


def sample_collocation_points(
    domain: CartesianWRFDomain,
    sampling: CollocationSamplingConfig,
    *,
    seed: int | None = None,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Generate PDE collocation coordinates inside a Cartesian WRF domain.

    Parameters
    ----------
    domain:
        Continuous Cartesian domain defining the ``x, y, z, t`` coordinate
        bounds.
    sampling:
        Collocation sampling settings, including point count and method.
    seed:
        Optional random seed for reproducible point generation.
    dtype:
        Torch dtype for the returned tensor.
    device:
        Optional torch device for the returned tensor.

    Returns
    -------
    torch.Tensor
        Tensor with shape ``(n_points, 4)`` and column order ``x, y, z, t``.
        The returned tensor does not require gradients by default; the training
        loop should call ``requires_grad_(True)`` before evaluating PDE
        residuals.
    """

    generator = _make_generator(seed=seed, device=device)
    unit_points = _sample_unit_hypercube(
        n_points=sampling.n_points,
        method=sampling.method,
        generator=generator,
        dtype=dtype,
        device=device,
    )

    return _scale_unit_points_to_domain(unit_points, domain)


def _sample_unit_hypercube(
    *,
    n_points: int,
    method: str,
    generator: torch.Generator | None,
    dtype: torch.dtype,
    device: torch.device | str | None,
) -> torch.Tensor:
    """Sample points in the four-dimensional unit hypercube."""

    if method == "random_uniform":
        return torch.rand(
            (n_points, 4),
            generator=generator,
            dtype=dtype,
            device=device,
        )

    if method == "latin_hypercube":
        return _latin_hypercube(
            n_points=n_points,
            generator=generator,
            dtype=dtype,
            device=device,
        )

    if method == "grid":
        return _grid_points(
            n_points=n_points,
            dtype=dtype,
            device=device,
        )

    raise ValueError(f"Unsupported collocation sampling method: {method}.")


def _latin_hypercube(
    *,
    n_points: int,
    generator: torch.Generator | None,
    dtype: torch.dtype,
    device: torch.device | str | None,
) -> torch.Tensor:
    """Generate a simple Latin-hypercube sample in four dimensions."""

    base = torch.arange(n_points, dtype=dtype, device=device).unsqueeze(1)
    offsets = torch.rand(
        (n_points, 4),
        generator=generator,
        dtype=dtype,
        device=device,
    )
    points = (base + offsets) / float(n_points)

    for dimension in range(4):
        permutation = torch.randperm(
            n_points,
            generator=generator,
            device=device,
        )
        points[:, dimension] = points[permutation, dimension]

    return points


def _grid_points(
    *,
    n_points: int,
    dtype: torch.dtype,
    device: torch.device | str | None,
) -> torch.Tensor:
    """Generate an approximately cubic four-dimensional tensor-product grid."""

    points_per_axis = math.ceil(n_points ** 0.25)
    axis = torch.linspace(0.0, 1.0, points_per_axis, dtype=dtype, device=device)
    mesh = torch.meshgrid(axis, axis, axis, axis, indexing="ij")
    points = torch.stack([component.reshape(-1) for component in mesh], dim=1)
    return points[:n_points]


def _scale_unit_points_to_domain(
    unit_points: torch.Tensor,
    domain: CartesianWRFDomain,
) -> torch.Tensor:
    """Scale unit-hypercube coordinates to physical domain bounds."""

    bounds = domain.as_bounds()
    lower = torch.tensor(
        [bounds[name][0] for name in domain.coordinate_names],
        dtype=unit_points.dtype,
        device=unit_points.device,
    )
    upper = torch.tensor(
        [bounds[name][1] for name in domain.coordinate_names],
        dtype=unit_points.dtype,
        device=unit_points.device,
    )
    return lower + unit_points * (upper - lower)


def _make_generator(
    *,
    seed: int | None,
    device: torch.device | str | None,
) -> torch.Generator | None:
    """Create a seeded torch generator when requested."""

    if seed is None:
        return None

    generator_device = "cpu"
    if device is not None:
        resolved_device = torch.device(device)
        if resolved_device.type == "cuda":
            generator_device = "cuda"

    generator = torch.Generator(device=generator_device)
    generator.manual_seed(seed)
    return generator
