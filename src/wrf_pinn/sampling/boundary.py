"""Boundary-condition sampling utilities.

This module owns the runtime construction of model-ready wall boundary
coordinates. It reads wall surface geometry, selects surface points according
to ``config.sampling``, extracts training times from available data, and returns
``x,y,z,t`` coordinates. It does not enforce boundary physics; that belongs to
``physics.residuals_boundary``.
"""

from __future__ import annotations

import torch

from wrf_pinn.config.boundary_data import NoSlipWallConfig
from wrf_pinn.config.sampling import BoundarySamplingConfig
from wrf_pinn.data.boundary import read_wall_surface_geometry
from wrf_pinn.data.boundary import WallSurfaceGeometry


def sample_wall_boundary_points(
    boundary: NoSlipWallConfig,
    sampling: BoundarySamplingConfig,
    *,
    case: "object | None" = None,
    seed: int | None = None,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Sample no-slip wall boundary coordinates for training.

    This is the high-level boundary sampler used by the training loop. It reads
    the configured wall surface, samples ``x,y,z`` points from that surface,
    extracts unique time values from the available training data, and returns
    the Cartesian product as model-ready ``x,y,z,t`` coordinates.

    The returned tensor is the Cartesian product of sampled surface points and
    unique time values:

    ``(x_i, y_i, z_i) x (t_j) -> (x_i, y_i, z_i, t_j)``.

    Parameters
    ----------
    boundary:
        No-slip wall configuration. Its ``surface`` field identifies the wall
        geometry CSV.
    sampling:
        Boundary sampling settings for selecting surface points.
    flow_field_data:
        Optional flow-field data. If provided, its time coordinate is used as
        the boundary training time source.
    sensor_data:
        Optional sensor data. Used as the boundary time source when
        ``flow_field_data`` is not provided.
    seed:
        Optional seed for reproducible random surface-row selection.
    dtype:
        Torch dtype for the returned tensor.
    device:
        Optional torch device for the returned tensor.

    Returns
    -------
    torch.Tensor
        Boundary coordinates with shape ``(n_surface_samples * n_times, 4)``
        and column order ``x,y,z,t``.
    """

    surface = read_wall_surface_geometry(boundary.surface)
    times = _training_times_from_case(case)
    return sample_wall_boundary_points_from_surface(
        surface=surface,
        times=times,
        sampling=sampling,
        seed=seed,
        dtype=dtype,
        device=device,
    )


def sample_wall_boundary_points_from_surface(
    surface: WallSurfaceGeometry,
    times: object,
    sampling: BoundarySamplingConfig,
    *,
    seed: int | None = None,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Sample wall ``x,y,z`` points and attach provided time values."""

    surface_points = sample_wall_surface_points(
        surface=surface,
        sampling=sampling,
        seed=seed,
        dtype=dtype,
        device=device,
    )
    unique_times = _as_unique_time_tensor(times, dtype=dtype, device=device)
    return _cartesian_product_surface_times(surface_points, unique_times)


def sample_wall_surface_points(
    surface: WallSurfaceGeometry,
    sampling: BoundarySamplingConfig,
    *,
    seed: int | None = None,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Sample ``x,y,z`` points from a no-slip wall surface.

    Parameters
    ----------
    surface:
        Wall surface geometry loaded from boundary data. Coordinates are
        expected to have shape ``(n_surface_points, 3)``.
    sampling:
        Boundary sampling settings. Supported methods are ``"all"`` and
        ``"random"``.
    seed:
        Optional seed for reproducible random row selection.
    dtype:
        Torch dtype for the returned tensor.
    device:
        Optional torch device for the returned tensor.

    Returns
    -------
    torch.Tensor
        Sampled wall surface coordinates with shape ``(n_samples, 3)`` and
        column order ``x,y,z``.
    """

    coordinates = surface.as_torch(dtype=dtype, device=device)
    _validate_surface_coordinates(coordinates)

    if sampling.method == "all":
        return coordinates

    if sampling.method == "random":
        return _sample_random_rows(
            coordinates,
            n_points=sampling.n_points,
            seed=seed,
            device=device,
        )

    raise ValueError(
        "Unsupported boundary sampling method for wall surface points: "
        f"{sampling.method}. Supported methods are 'all' and 'random'."
    )


def _sample_random_rows(
    coordinates: torch.Tensor,
    *,
    n_points: int | None,
    seed: int | None,
    device: torch.device | str | None,
) -> torch.Tensor:
    """Randomly sample rows from wall surface coordinates."""

    if n_points is None:
        raise ValueError("Boundary random sampling requires n_points.")

    n_available = coordinates.shape[0]
    if n_points > n_available:
        raise ValueError(
            "Boundary random sampling cannot request more points than available; "
            f"requested {n_points}, available {n_available}."
        )

    generator = _make_generator(seed=seed, device=device)
    indices = torch.randperm(
        n_available,
        generator=generator,
        device=coordinates.device,
    )[:n_points]
    return coordinates[indices]


def _validate_surface_coordinates(coordinates: torch.Tensor) -> None:
    """Validate sampled wall surface coordinate tensor shape."""

    if coordinates.ndim != 2 or coordinates.shape[1] != 3:
        raise ValueError(
            "Wall surface coordinates must have shape (n_surface_points, 3); "
            f"got {tuple(coordinates.shape)}."
        )

    if coordinates.shape[0] < 1:
        raise ValueError("Wall surface geometry must contain at least one point.")


def _training_times_from_case(case: object | None) -> object:
    """Return the boundary time source from the case's coordinate times."""

    if case is None:
        raise ValueError(
            "Boundary sampling requires a case to provide training time values."
        )
    return _unique_times_from_coordinates(
        coordinates=case.coordinates,
        coordinate_names=case.coordinate_names,
        data_name="case",
    )


def _unique_times_from_coordinates(
    *,
    coordinates: object,
    coordinate_names: tuple[str, ...],
    data_name: str,
) -> torch.Tensor:
    """Extract sorted unique time values from coordinate arrays."""

    if "t" not in coordinate_names:
        raise ValueError(f"{data_name} coordinate_names must contain 't'.")

    time_index = coordinate_names.index("t")
    coordinate_tensor = torch.as_tensor(coordinates)
    if coordinate_tensor.ndim != 2 or coordinate_tensor.shape[1] <= time_index:
        raise ValueError(
            f"{data_name} coordinates do not contain a valid time column."
        )

    return torch.unique(coordinate_tensor[:, time_index], sorted=True)


def _as_unique_time_tensor(
    times: object,
    *,
    dtype: torch.dtype,
    device: torch.device | str | None,
) -> torch.Tensor:
    """Convert provided times to a sorted unique 1D tensor."""

    time_values = torch.as_tensor(times, dtype=dtype, device=device).reshape(-1)
    if time_values.numel() < 1:
        raise ValueError("At least one time value is required for boundary sampling.")

    return torch.unique(time_values, sorted=True)


def _cartesian_product_surface_times(
    surface_points: torch.Tensor,
    times: torch.Tensor,
) -> torch.Tensor:
    """Return Cartesian product of ``x,y,z`` surface points and time values."""

    n_surface = surface_points.shape[0]
    n_times = times.shape[0]

    repeated_surface = surface_points.repeat_interleave(n_times, dim=0)
    tiled_times = times.repeat(n_surface).unsqueeze(1)
    return torch.cat((repeated_surface, tiled_times), dim=1)


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
