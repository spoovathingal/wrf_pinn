"""Local SGS filter width from the native, stretched FE reference grid.

All heights and spacings are physical metres. Jet derivatives are with
respect to physical (x, y, Z, t), not normalized model coordinates.
"""

from functools import lru_cache
import math

import torch

from wrf_pinn.config.physics import PhysicsConfig
from wrf_pinn.physics.taylor import Jet, where


@lru_cache(maxsize=32)
def _reference_profile(reference_z, edge_policy, device, dtype):
    """Cache native centre heights and their effective vertical spacings."""
    if edge_policy != "linear_bottom_quadratic_top":
        raise ValueError("Unsupported SGS vertical edge policy.")
    if len(reference_z) < 3:
        raise ValueError(
            "Local SGS width requires at least three reference levels."
        )

    dz = (
        (reference_z[1] - reference_z[0],)
        + tuple(
            (reference_z[k + 1] - reference_z[k - 1]) / 2.0
            for k in range(1, len(reference_z) - 1)
        )
        + (
            (
                3.0 * reference_z[-1]
                - 4.0 * reference_z[-2]
                + reference_z[-3]
            ) / 2.0,
        )
    )

    return (
        torch.tensor(reference_z, device=device, dtype=dtype),
        torch.tensor(dz, device=device, dtype=dtype),
    )


def terrain_height_jet(height, slopes):
    """Build a stationary terrain Jet from h and physical (dh/dx, dh/dy)."""
    if height.ndim != 2 or height.shape[1] != 1:
        raise ValueError("height must have shape (N, 1).")
    if slopes.shape != (height.shape[0], 2):
        raise ValueError("slopes must have shape (N, 2).")
    if slopes.device != height.device or slopes.dtype != height.dtype:
        raise ValueError(
            "Terrain heights and slopes must share device and dtype."
        )

    zero = torch.zeros_like(height)
    derivatives = torch.stack(
        (slopes[:, 0:1], slopes[:, 1:2], zero, zero),
        dim=0,
    )
    return Jet(height, derivatives)


def sgs_filter_width(
    altitude,
    terrain_height,
    physics: PhysicsConfig,
    terrain_top_height: float,
    *,
    check_values: bool = True,
):
    """Return Delta as a Tensor or a first-order Jet.

    altitude and terrain_height must both be (N, 1) Tensors or both be
    Jets containing (N, 1) values. For autograd, terrain_height must be
    evaluated from the same differentiable coordinates as altitude.

    Spacing is linearly interpolated between reference-level centres.
    Below the first centre it is held constant for the surface closure.
    Checked calls reject heights above the available profile, allowing
    only a small floating-point tolerance. Unchecked calls hold the last
    spacing constant there; callers must validate their domain first.

    check_values=True performs device-to-host checks. Training may use
    False only after validating its terrain and vertical bounds.
    """
    is_jet = isinstance(altitude, Jet)
    if is_jet != isinstance(terrain_height, Jet):
        raise TypeError(
            "altitude and terrain_height must both be Tensors or Jets."
        )

    z = altitude.v if is_jet else altitude
    h = terrain_height.v if is_jet else terrain_height

    if z.ndim != 2 or z.shape[1] != 1 or h.shape != z.shape:
        raise ValueError(
            "Altitude and terrain height must have shape (N, 1)."
        )
    if (
        not z.is_floating_point()
        or z.dtype != h.dtype
        or z.device != h.device
    ):
        raise ValueError(
            "Heights must share a floating-point dtype and device."
        )

    if physics.sgs_filter_width_mode == "uniform":
        value = torch.full_like(z, physics.uniform_filter_width)
        return (
            Jet(value, torch.zeros_like(altitude.d))
            if is_jet
            else value
        )

    if physics.sgs_filter_width_mode != "fe_cell_volume":
        raise ValueError("Unsupported SGS filter-width mode.")
    if (
        not math.isfinite(terrain_top_height)
        or terrain_top_height <= 0.0
    ):
        raise ValueError(
            "terrain_top_height must be finite and positive."
        )

    if check_values:
        if not bool(torch.isfinite(z).all() & torch.isfinite(h).all()):
            raise ValueError(
                "Altitude and terrain height must be finite."
            )
        if bool((h >= terrain_top_height).any()):
            raise ValueError(
                "Terrain must be below terrain_top_height."
            )

    compression = 1.0 - terrain_height / terrain_top_height
    reference_height = (altitude - terrain_height) / compression
    query = reference_height.v if is_jet else reference_height

    levels, spacing = _reference_profile(
        physics.sgs_reference_z,
        physics.sgs_vertical_edge_policy,
        z.device,
        z.dtype,
    )

    if check_values:
        tolerance = (
            16.0
            * torch.finfo(z.dtype).eps
            * max(1.0, abs(physics.sgs_reference_z[-1]))
        )
        if bool((query > levels[-1] + tolerance).any()):
            raise ValueError(
                "Query exceeds the available SGS reference profile."
            )

    # A single interval per point; interpolation carries the height derivative.
    index = torch.searchsorted(
        levels,
        query.contiguous(),
        right=True,
    ).clamp(1, levels.numel() - 1) - 1

    slope = (
        (spacing[index + 1] - spacing[index])
        / (levels[index + 1] - levels[index])
    )
    dz_reference = (
        spacing[index]
        + slope * (reference_height - levels[index])
    )

    # Endpoint extension has zero reference-height derivative.
    dz_reference = where(
        query <= levels[0], spacing[0], dz_reference
    )
    dz_reference = where(
        query >= levels[-1], spacing[-1], dz_reference
    )

    cell_volume = (
        physics.sgs_dx
        * physics.sgs_dy
        * compression
        * dz_reference
    )
    return cell_volume.pow(1.0 / 3.0)