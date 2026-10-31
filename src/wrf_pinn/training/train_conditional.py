"""Train one shared ConditionalModel across a batch of cases on the five losses."""

from __future__ import annotations

import sys
import math
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, fields, asdict, replace
from functools import partial
from functools import lru_cache
from pathlib import Path

import torch
import torch.distributed as dist
from torch.profiler import record_function
from torch import nn

from wrf_pinn.data.conditional_case import (
    ConditionalCase, COORD_NAMES, STATE_VARS, read_conditional_case,prepare_terrain, terrain_at,
)
from wrf_pinn.config.physics import DEFAULT_PHYSICS, PhysicsConfig
from wrf_pinn.config.scaling import DEFAULT_RESIDUAL_SCALING, ResidualScalingConfig
from wrf_pinn.physics.residuals_pde import (
    cartesian_zero_forcing_residuals, cartesian_zero_forcing_residuals_taylor,
)
from wrf_pinn.physics.residuals_boundary import no_penetration_z_wall_residuals
from wrf_pinn.physics.taylor import Jet, taylor_forward
from wrf_pinn.training.provenance import run_manifest, write_event

#: the five losses of the conditional PINN, in order.
LOSS_NAMES: tuple[str, ...] = ("data", "initial", "flow_boundary", "pde", "surface")


@dataclass
class ConditionalHistory:
    """Per-epoch loss history (one list per loss name)."""

    total: list[float] = field(default_factory=list)
    data: list[float] = field(default_factory=list)
    initial: list[float] = field(default_factory=list)
    flow_boundary: list[float] = field(default_factory=list)
    pde: list[float] = field(default_factory=list)
    surface: list[float] = field(default_factory=list)


@dataclass(frozen=True)
class ConditionalTrainConfig:
    epochs: int = 2000
    batch_cases: int = 4          # cases drawn per step (per process when distributed)
    #: cases per forward/backward pass; gradients accumulate over the step's
    #: passes, so the math matches one pass over batch_cases. 0 = whole batch.
    cases_per_pass: int = 0
    readers: int = 8              # threads reading cases in lazy mode
    learning_rate: float = 1e-3
    #: cosine-decay the learning rate to this value by the last epoch. None = constant.
    learning_rate_min: "float | None" = None
    weight_data: float = 1.0
    weight_initial: float = 1.0
    weight_flow_boundary: float = 1.0
    weight_pde: float = 1.0
    weight_surface: float = 1.0
    # Relative TKE weight within the PDE loss.
    tke_weight: float = 1.0
    # Conservative TKE residual scale [kg m^-1 s^-3].
    tke_residual_scale: float = 1.0
    n_collocation: int = 2048     # PDE collocation points per case per step
    n_wall: int = 512             # surface/wall points per case per step
    grad_clip_norm: float = 1.0   # clip grads (PINN PDE gradients can spike)
    #: affine scaling (physical = offset + scale*normalized) for the residuals,
    #: built from the pre-processor's normalization recipe. Identity if None.
    scaling: "ResidualScalingConfig | None" = None
    #: SOURCE normalization: saved NPZ coordinates -> source units.
    source_scaling: "ResidualScalingConfig | None" = None

    #: fe_terrain_following: source z is the stretched reference height,
    #: before terrain compression.
    #: reference_plus_terrain: legacy approximation Z = h + source z.
    #: height_above_ground: source z is actual local height above ground.
    #: physical_altitude: source z is already physical altitude.
    source_z_mode: str = "fe_terrain_following"
    #: FE terrain-compression height, in physical metres.
    #: 4027.5 is codex inferred value for CSTL dataset.
    terrain_top_height: float = 4027.5

    #: First FE reference level before terrain compression, in metres.
    #: Used to locate the surface reference winds and density.
    #: None retains the existing fixed-AGL reference-height behavior.
    fe_surface_reference_z: float | None = 10.125386238098145

    #: physics for the residuals; carries the q_v unit conversion.
    physics: PhysicsConfig = DEFAULT_PHYSICS
    #: flux (fricVel, htFlux) offset/scale from the recipe, to un-normalize
    #: case.surface back to physical for the surface-layer residual.
    flux_offset: tuple[float, float] = (0.0, 0.0)
    flux_scale: tuple[float, float] = (1.0, 1.0)
    surface_stress_scale: float = 3.8273
    surface_heat_flux_scale: float = 1.0
    log_every: int = 100
    device: str = "auto"
    seed: int = 0
    #: write a resumable checkpoint here every ``checkpoint_every`` epochs (plus a
    #: step-tagged copy); resume from it on start if it exists. None disables.
    checkpoint_path: "str | None" = None
    checkpoint_every: int = 100
    #: when True, time forward/backward/step per epoch (CUDA-synced) for profiling.
    profile: bool = False
    #: run the derivative-free forwards (data, initial, flow-boundary points) under
    #: bf16 autocast: their matmuls use bf16 tensor cores. The PDE and surface paths,
    #: which differentiate the network, stay in float32. Off by default.
    bf16_data: bool = False
    #: lazy mode: read and stack cases in this many DataLoader worker processes
    #: instead of threads in the training process, so case decoding does not compete
    #: with the training loop for the GIL. 0 = reader threads (``readers``).
    loader_workers: int = 0
    #: torch.compile the network forward, the first-order Taylor pass and the PDE
    #: residuals (fuses their many small elementwise kernels). The first steps pay
    #: the compile time. Off by default.
    compile: bool = False
    #: Optional JSONL optimization events; checkpoint provenance is always saved.
    events_path: "str | None" = None
    #: conditioning z read from each case: 'pod' coeffs or 'raw' fields (read_conditional_case).
    latent: str = "pod"
    #: predict the change from the initial field: state = initial + (t - t0)/T * network.
    #: The initial field and its derivatives come from each case's ini_raw on the grid,
    #: so PDE points sit on grid nodes (at random times). Off: the network is the state.
    predict_change: bool = False
    #: metres per stored terrain x, y unit; must match the x, y conversion in source_scaling.
    terrain_xy_scale: float = 1.0
    def __post_init__(self) -> None:
        if not math.isfinite(self.tke_weight) or self.tke_weight < 0.0:
            raise ValueError("tke_weight must be finite and nonnegative.")
        if not math.isfinite(self.tke_residual_scale) or self.tke_residual_scale <= 0.0:
            raise ValueError("tke_residual_scale must be finite and positive.")
        valid_modes = ("fe_terrain_following", "reference_plus_terrain",
        "height_above_ground", "physical_altitude")
        if self.source_z_mode not in valid_modes:
            raise ValueError(f"Unsupported source_z_mode: {self.source_z_mode}")

        if (not math.isfinite(self.terrain_top_height) or self.terrain_top_height <= 0.0):
            raise ValueError("terrain_top_height must be finite and positive.")

        if self.fe_surface_reference_z is not None:
            if (not math.isfinite(self.fe_surface_reference_z)
                or not 0.0 < self.fe_surface_reference_z < self.terrain_top_height):
                raise ValueError(
                    "fe_surface_reference_z must be finite and between "
                    "zero and terrain_top_height.")

def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _initial_targets(case: ConditionalCase) -> torch.Tensor:
    """Flatten the decoded initial field to (n_ic, n_state), matching initial_coords."""
    fields_ = [case.initial[v].transpose(0, 1).reshape(-1) for v in range(case.initial.shape[0])]
    return torch.stack(fields_, dim=1)


def _boundary_targets(case: ConditionalCase) -> torch.Tensor:
    """Flatten the decoded boundary field to (n_points, n_state), matching boundary_coords."""
    ns = case.boundary.shape[0]
    return torch.stack([case.boundary[v].reshape(-1) for v in range(ns)], dim=1)


@dataclass
class CaseBatch:
    """Several cases' points concatenated, so each loss is one forward pass. The
    ``*_sizes`` give each case's point count, in case order."""

    z: torch.Tensor
    interior: torch.Tensor
    targets: torch.Tensor
    target_mask: torch.Tensor
    initial_coords: torch.Tensor
    initial_targets: torch.Tensor
    boundary_coords: torch.Tensor
    boundary_targets: torch.Tensor
    surface: torch.Tensor
    terrain_x: torch.Tensor
    terrain_y: torch.Tensor
    terrain_h: torch.Tensor
    surface_geometry: torch.Tensor  # physical h, dh/dx, dh/dy per surface row
    box_lo: torch.Tensor          # (k, n_coord) interior bounding box, for collocation
    box_hi: torch.Tensor
    interior_sizes: torch.Tensor
    initial_sizes: torch.Tensor
    boundary_sizes: torch.Tensor
    surface_sizes: torch.Tensor
    # predict_change only (empty otherwise): initial field at the grid nodes, its spatial
    # first (3) and second (6, PAIRS order) derivatives, node model coords (x, y, z), each
    # point set's node index, and per case t0 and window length in model-normalized time.
    base_v: torch.Tensor
    base_d: torch.Tensor
    base_h: torch.Tensor
    node_coords: torch.Tensor
    node_sizes: torch.Tensor
    interior_node: torch.Tensor
    initial_node: torch.Tensor
    boundary_node: torch.Tensor
    surface_node: torch.Tensor
    t0: torch.Tensor
    span: torch.Tensor

    @property
    def k(self) -> int:
        return self.z.shape[0]

    def to(self, device: torch.device) -> "CaseBatch":
        return CaseBatch(**{f.name: getattr(self, f.name).to(device, non_blocking=True)
                            for f in fields(self)})

    def pin(self) -> "CaseBatch":
        return CaseBatch(**{f.name: getattr(self, f.name).pin_memory() for f in fields(self)})

    pin_memory = pin  # what DataLoader(pin_memory=True) calls on custom batch types

def _source_height_to_altitude(z_source, h, cfg):
    """Convert source height in metres to physical altitude in metres."""
    if cfg.source_z_mode == "fe_terrain_following":
        return h + z_source * (1.0 - h / cfg.terrain_top_height)

    if cfg.source_z_mode in ("reference_plus_terrain", "height_above_ground"):
        return h + z_source

    if cfg.source_z_mode == "physical_altitude":
        return z_source
    
    raise ValueError(f"Unsupported source_z_mode: {cfg.source_z_mode}")

def _node_index(points: torch.Tensor, axes: list[torch.Tensor]) -> torch.Tensor:
    """Flat index (first axis slowest) of the grid node at each point; ``points`` are
    (n, len(axes)) source-normalized coords that must sit on the node ``axes``."""
    idx = torch.zeros(points.shape[0], dtype=torch.long)
    for a, ax in enumerate(axes):
        i = torch.searchsorted((ax[1:] + ax[:-1]) / 2, points[:, a].contiguous())
        if (ax[i] - points[:, a]).abs().max() > 1e-4 * (ax[-1] - ax[0]):
            raise ValueError("predict_change: points are not on the case grid.")
        idx = idx * len(ax) + i
    return idx


def _grid_derivatives(f: torch.Tensor, X: torch.Tensor, Y: torch.Tensor, Z: torch.Tensor
                      ) -> tuple[torch.Tensor, torch.Tensor]:
    """Spatial first (3, ...) and second (6, ..., PAIRS order) derivatives of grid fields
    f (nx, ny, nz, c) by central differences, in physical coordinates of a terrain-following
    grid: node (i, j, k) sits at X(i), Y(j), Z(i, j, k)."""
    g = lambda a, d: torch.gradient(a, dim=d)[0]
    xi, yj = g(X, 0)[..., None], g(Y, 1)[..., None]
    zi, zj, zk = (g(Z, d)[..., None] for d in range(3))

    def grad(a):
        az = g(a, 2) / zk
        return (g(a, 0) - zi * az) / xi, (g(a, 1) - zj * az) / yj, az

    fx, fy, fz = grad(f)
    (fxx, fxy, fxz), (_, fyy, fyz), (_, _, fzz) = grad(fx), grad(fy), grad(fz)
    return torch.stack([fx, fy, fz]), torch.stack([fxx, fyy, fzz, fxy, fxz, fyz])


def _with_initial(net: Jet, s: torch.Tensor, ds: torch.Tensor, v0: torch.Tensor,
                  d0: torch.Tensor, h0: "torch.Tensor | None" = None) -> Jet:
    """Jet of initial + s * net with s = (t - t0)/T: the initial field is steady, so d/dt
    is s * dnet/dt + ds/dt * net. ``s``, ``ds`` are (n, 1); v0 (n, c), d0 (3, n, c), h0 (6, n, c)."""
    d = torch.cat([d0 + s * net.d[:3], (s * net.d[3] + ds * net.v)[None]])
    return Jet(v0 + s * net.v, d, None if net.h is None else h0 + s * net.h)


def _initial_grids(cases, initial, interior, model_off, model_scale) -> dict:
    """The predict_change members of a CaseBatch (see CaseBatch). ``initial``/``interior``
    are the cases' model-normalized initial and interior coords."""
    out = {k: [] for k in ("base_v", "base_d", "base_h", "node_coords", "interior_node",
                           "initial_node", "boundary_node", "surface_node", "t0", "span")}
    n_seen = 0
    for c, ini, inner in zip(cases, initial, interior):
        if c.initial_raw is None:
            raise ValueError(f"{c.name}: predict_change needs ini_raw (a --keep-raw case).")
        src = c.initial_coords
        axes = [torch.unique(src[:, a]) for a in range(3)]
        nx, ny, nz = (len(a) for a in axes)
        if not torch.equal(_node_index(src[:, :3], axes), torch.arange(len(src))):
            raise ValueError(f"{c.name}: initial points are not the grid in (x, y, z) order.")
        n_state = c.initial_raw.shape[0]
        f = c.initial_raw.permute(2, 1, 0).reshape(nx, ny, nz, n_state)
        phys = (ini[:, :3] * model_scale[:3] + model_off[:3]).reshape(nx, ny, nz, 3)
        d, h = _grid_derivatives(f, phys[..., 0], phys[..., 1], phys[..., 2])
        out["base_v"].append(f.reshape(-1, n_state))
        out["base_d"].append(d.reshape(3, -1, n_state))
        out["base_h"].append(h.reshape(6, -1, n_state))
        out["node_coords"].append(ini[:, :3])
        out["interior_node"].append(_node_index(c.interior[:, :3], axes) + n_seen)
        out["initial_node"].append(torch.arange(len(src)) + n_seen)
        out["boundary_node"].append(_node_index(c.boundary_coords.reshape(-1, 4)[:, :3], axes) + n_seen)
        surface_xy = c.surface.reshape(-1, c.surface.shape[-1])[:, :2]
        out["surface_node"].append(_node_index(surface_xy, axes[:2]) * nz + n_seen)
        out["t0"].append(ini[0, 3])
        out["span"].append(inner[:, 3].max() - ini[0, 3])
        n_seen += len(src)
    batch = {k: torch.cat(v, dim=1 if k in ("base_d", "base_h") else 0) for k, v in out.items()
             if k not in ("t0", "span")}
    batch["t0"], batch["span"] = torch.stack(out["t0"]), torch.stack(out["span"])
    batch["node_sizes"] = torch.tensor([len(c.initial_coords) for c in cases], dtype=torch.long)
    return batch


def stack_cases(cases: list[ConditionalCase], cfg: ConditionalTrainConfig) -> CaseBatch:
    """Concatenate torch cases (on one device) into a CaseBatch."""
    """Prepare model coordinates and terrain on CPU."""
    if cfg.source_scaling is None:
        raise ValueError("Set source_scaling from the dataset normalization recipe.")
    scaling = cfg.scaling if cfg.scaling is not None else DEFAULT_RESIDUAL_SCALING
    source_off, source_scale = _coordinate_affine(cfg.source_scaling, cases[0].interior)
    model_off, model_scale = _coordinate_affine(scaling, cases[0].interior)

    xy = torch.tensor([cfg.terrain_xy_scale, cfg.terrain_xy_scale, 1.0])
    grids = [prepare_terrain(torch.as_tensor(c.terrain, dtype=torch.float32) * xy) for c in cases]
    if (cfg.source_z_mode == "fe_terrain_following" or cfg.fe_surface_reference_z is not None):
        for case, (_, _, heights) in zip(cases, grids):
            if torch.any(heights >= cfg.terrain_top_height):
                raise ValueError(f"{case.name}: terrain must be below terrain_top_height.")
    cat = lambda values: torch.cat(values, dim=0)
    sizes = lambda values: torch.tensor([len(v) for v in values], dtype=torch.long)

    def convert(coords, grid):
        physical = coords * source_scale + source_off
        if cfg.source_z_mode != "physical_altitude":
            h = terrain_at(physical[:, 0], physical[:, 1], *grid)[0]
            physical[:, 2] = _source_height_to_altitude(physical[:, 2], h, cfg)
        return (physical - model_off) / model_scale

    interior, initial, boundary, surface, geometry = [], [], [], [], []

    for c, grid in zip(cases, grids):
        interior.append( convert(c.interior, grid) )
        initial.append( convert(c.initial_coords, grid) )
        boundary.append( convert(c.boundary_coords.reshape(-1, 4), grid) )

        rows = c.surface.reshape(-1, c.surface.shape[-1])
        physical_xyt = (rows[:, :3] * source_scale[[0, 1, 3]] + source_off[[0, 1, 3]])
        h, hx, hy = terrain_at(physical_xyt[:, 0], physical_xyt[:, 1], *grid)
        model_xyt = (physical_xyt - model_off[[0, 1, 3]]) / model_scale[[0, 1, 3]]

        surface.append( torch.cat( (model_xyt, rows[:, 3:]) , dim=1) )
        geometry.append( torch.stack( (h, hx, hy) , dim=1) )

        # Future physical-altitude mode uses the above-ground part
        # of the case's physical bounding box for PDE sampling.
        if cfg.source_z_mode == "physical_altitude":
            p = c.interior * source_scale + source_off
            lo, hi = p.amin(dim=0), p.amax(dim=0)
            xs, ys, _ = grid
            xs = torch.unique(  torch.cat( (xs[(xs >= lo[0]) & (xs <= hi[0])], lo[0:1], hi[0:1]) )  )
            ys = torch.unique(  torch.cat( (ys[(ys >= lo[1]) & (ys <= hi[1])], lo[1:2], hi[1:2]) )  )
            xx, yy = torch.meshgrid(xs, ys, indexing="ij")
            if terrain_at(xx, yy, *grid)[0].max() >= hi[2]:
                raise ValueError(f"{c.name}: physical case top must be above terrain.")

    if cfg.predict_change:
        change = _initial_grids(cases, initial, interior, model_off, model_scale)
    else:
        none = torch.zeros(0)
        change = dict(base_v=none, base_d=none, base_h=none, node_coords=none,
                      node_sizes=none.long(), interior_node=none.long(), initial_node=none.long(),
                      boundary_node=none.long(), surface_node=none.long(), t0=none, span=none)
    return CaseBatch(
        **change,
        z=torch.stack([c.z for c in cases]),
        interior=cat(interior),
        targets=cat([c.targets for c in cases]), target_mask=cat([c.target_mask for c in cases]),
        # predict_change: the exact initial field (its prediction at t0), not the POD decode
        initial_coords=cat(initial), initial_targets=cat([_initial_targets(
            replace(c, initial=c.initial_raw) if cfg.predict_change else c) for c in cases]),
        boundary_coords=cat(boundary), boundary_targets=cat([_boundary_targets(c) for c in cases]),
        surface=cat(surface), terrain_x=torch.stack([g[0] for g in grids]),
        terrain_y=torch.stack([g[1] for g in grids]), terrain_h=torch.stack([g[2] for g in grids]),
        surface_geometry=cat(geometry),
        # SOURCE-normalized bounds, deliberately kept unconverted.
        box_lo=torch.stack([c.interior.amin(dim=0) for c in cases]), box_hi=torch.stack([c.interior.amax(dim=0) for c in cases]),
        interior_sizes=sizes(interior), initial_sizes=sizes(initial), boundary_sizes=sizes(boundary), surface_sizes=sizes(surface),
    )

def _per_case_mean(values: torch.Tensor, sizes: torch.Tensor, k: int) -> torch.Tensor:
    """Mean of per-point values within each case -> (k,)."""
    seg = torch.repeat_interleave(torch.arange(k, device=values.device), sizes,
                                  output_size=values.shape[0])
    sums = torch.zeros(k, device=values.device, dtype=values.dtype).index_add_(0, seg, values)
    return sums / sizes.to(values.dtype)


@lru_cache(maxsize=32)
def _device_constant(values: tuple[float, ...], device: torch.device,
                     dtype: torch.dtype) -> torch.Tensor:
    """A small constant tensor on ``device``, built once: a per-pass host-to-device
    copy made the CPU wait for the GPU. Callers must not modify it in place."""
    return torch.tensor(values, device=device, dtype=dtype)

def _coordinate_affine(scaling, like):
    offsets = tuple(getattr(scaling, name).offset for name in COORD_NAMES)
    return (_device_constant(offsets, like.device, like.dtype), _device_constant(
            tuple(scaling.coordinate_scales()), like.device, like.dtype),
    )

@dataclass(frozen=True)
class StepFunctions:
    """The callables one training step uses, eager or compiled (``step_functions``)."""

    forward: Callable                # (coords, z_rows) -> state
    taylor: Callable                 # physics.taylor.taylor_forward
    pde_residuals: Callable          # cartesian_zero_forcing_residuals_taylor


def step_functions(model: nn.Module, compile: bool = False) -> StepFunctions:
    """Eager step functions, or torch.compile'd ones sharing the model's parameters."""
    fns = StepFunctions(model, taylor_forward, cartesian_zero_forcing_residuals_taylor)
    if not compile:
        return fns
    return StepFunctions(*(torch.compile(f) for f in (fns.forward, fns.taylor, fns.pde_residuals)))


def _taylor_supported(model: nn.Module) -> bool:
    """A tanh ConditionalModel: its coordinate derivatives can be carried forward
    (``physics.taylor``) instead of taken with nested autograd."""
    return getattr(getattr(model, "config", None), "activation", None) == "tanh"


def _surface_loss(model, b: CaseBatch, cfg, scaling, fns: StepFunctions) -> torch.Tensor:
    """Surface-layer residual on each case's bottom face, using its stored fluxes.

    Wall coords come from the surface rows (x,y,t normalized; z at the floor);
    fluxes are un-normalized to physical for the no-penetration-z residual."""
    surf = b.surface
    dev, dt = surf.device, surf.dtype
    h = b.surface_geometry[:, 0:1]
    z_floor = (h[:, 0] - scaling.z.offset) / scaling.z.scale
    wall = torch.stack( (surf[:, 0], surf[:, 1], z_floor, surf[:, 2]), dim=1)
    if _taylor_supported(model):  # wall gradients from one first-order forward pass
        jet = fns.taylor(model.net, wall, b.z, scaling.coordinate_scales(), second=False,
                         sizes=b.surface_sizes)
        if cfg.predict_change:   # initial field at the ground: extend down from the first level
            k0 = b.surface_node
            dz = h[:, 0] - (b.node_coords[k0, 2] * scaling.z.scale + scaling.z.offset)
            s = _ramp(b, wall, b.surface_sizes)
            seg = torch.repeat_interleave(torch.arange(b.k, device=dev), b.surface_sizes,
                                          output_size=wall.shape[0])
            ds = (1.0 / (b.span[seg] * scaling.coordinate_scales()[3]))[:, None]
            jet = _with_initial(jet, s, ds, b.base_v[k0] + dz[:, None] * b.base_d[2, k0],
                                b.base_d[:, k0])
        wall_state, state_jacobian = jet.v, jet.d
    else:
        wall = wall.requires_grad_(True)
        wall_state, state_jacobian = model(wall, b.z, b.surface_sizes), None

    reference = wall.detach().clone()
    if cfg.fe_surface_reference_z is not None:
        reference_agl = (cfg.fe_surface_reference_z * (1.0 - h[:, 0] / cfg.terrain_top_height))  # physical AGL at the FE reference level)
    else:
        reference_agl = cfg.physics.constants.surface_reference_height

    reference_altitude = h[:, 0] + reference_agl
    reference[:, 2] = (reference_altitude - scaling.z.offset) / scaling.z.scale

    reference_state = fns.forward(reference, b.z, b.surface_sizes)
    if cfg.predict_change:       # the reference level is the first grid level
        k0 = b.surface_node
        if (reference[:, 2] - b.node_coords[k0, 2]).abs().max() > 1e-4:
            raise ValueError("predict_change: surface reference level is not the first grid level.")
        reference_state = b.base_v[k0] + _ramp(b, reference, b.surface_sizes) * reference_state

    f_off = _device_constant(tuple(cfg.flux_offset), dev, dt)
    f_scale = _device_constant(tuple(cfg.flux_scale), dev, dt)
    phys = surf[:, 3:] * f_scale + f_off                             # un-normalize fluxes
    res = no_penetration_z_wall_residuals(
        coordinates=wall, state=wall_state,
        reference_coordinates=reference, reference_state=reference_state,
        physics=cfg.physics, scaling=scaling,
        fric_vel=phys[:, 0:1], ht_flux=phys[:, 1:2],
        surface_stress_scale=cfg.surface_stress_scale,
        surface_heat_flux_scale=cfg.surface_heat_flux_scale,
        state_jacobian=state_jacobian, check_values=False,
        terrain_height=h, terrain_slopes=b.surface_geometry[:, 1:3])  # wall rows are built at z=0 above
    per_eq = torch.stack([_per_case_mean(r.square().reshape(-1), b.surface_sizes, b.k)
                          for r in res.values()])                    # (n_eq, k)
    return per_eq.mean(dim=0).mean()


def _pde_loss(model, coll: torch.Tensor, z: torch.Tensor, cfg, scaling,
              fns: StepFunctions, out: "Jet | None" = None) -> torch.Tensor:
    """Mean squared PDE residual over equal-sized per-case collocation sets.

    ``z`` holds one latent row per case; each case owns ``cfg.n_collocation``
    consecutive rows of ``coll``. A tanh ConditionalModel takes the forward-Taylor
    path (derivatives carried through one forward pass); anything else falls back
    to nested autograd. ``out``: the state jet, when the caller built it (predict_change)."""
    if _taylor_supported(model):
        res = fns.pde_residuals(model.net, coll, z, physics=cfg.physics, scaling=scaling,
                                sizes=cfg.n_collocation, out=out)
    else:
        coll = coll.requires_grad_(True)
        res = cartesian_zero_forcing_residuals(coll, model(coll, z, cfg.n_collocation),
                                               physics=cfg.physics, scaling=scaling)
    base_terms = [
        residual.square().mean() for name, residual in res.items() if name != "sgs_tke"
    ]
    if cfg.tke_weight == 0.0:
        return torch.stack(base_terms).mean()

    tke_mse = (res["sgs_tke"] / cfg.tke_residual_scale).square().mean()
    return (torch.stack(base_terms).sum() + cfg.tke_weight * tke_mse
    ) / (len(base_terms) + cfg.tke_weight)


def _forward_maybe_bf16(forward: Callable, coords: torch.Tensor, z: torch.Tensor,
                        sizes: torch.Tensor, bf16: bool) -> torch.Tensor:
    """forward(coords, z, sizes), under bf16 autocast when ``bf16``; float32 out."""
    with torch.autocast(device_type=coords.device.type, dtype=torch.bfloat16, enabled=bf16):
        out = forward(coords, z, sizes)
    return out.float()


def _ramp(b: CaseBatch, coords: torch.Tensor, sizes: torch.Tensor) -> torch.Tensor:
    """(t - t0) / T per point, (n, 1), in each case's window (predict_change)."""
    seg = torch.repeat_interleave(torch.arange(b.k, device=coords.device), sizes,
                                  output_size=coords.shape[0])
    return ((coords[:, 3] - b.t0[seg]) / b.span[seg])[:, None]


def _pde_loss_on_nodes(model, b: CaseBatch, u: torch.Tensor, cfg, scaling,
                       fns: StepFunctions) -> torch.Tensor:
    """PDE loss of initial + ramp * network (predict_change) at grid nodes, where the
    initial field's derivatives are known. ``u`` (k, n_collocation, 4) uniform draws:
    column 0 picks each point's node, column 3 its time in the window."""
    k, n = u.shape[:2]
    first = torch.cumsum(b.node_sizes, 0) - b.node_sizes
    node = (first[:, None] + (u[..., 0] * b.node_sizes[:, None]).long()
            .clamp_max(b.node_sizes[:, None] - 1)).reshape(-1)
    s = u[..., 3].reshape(-1, 1)                                  # ramp = the time draw
    t = b.t0[:, None] + u[..., 3] * b.span[:, None]
    coll = torch.cat([b.node_coords[node], t.reshape(-1, 1)], dim=1)
    scales = scaling.coordinate_scales()
    ds = (1.0 / (b.span * scales[3]))[:, None].expand(k, n).reshape(-1, 1)
    net = fns.taylor(model.net, coll, b.z, scales, sizes=cfg.n_collocation)
    out = _with_initial(net, s, ds, b.base_v[node], b.base_d[:, node], b.base_h[:, node])
    return _pde_loss(model, coll, b.z, cfg, scaling, fns, out=out)


def _batch_losses(model: nn.Module, b: CaseBatch, cfg: "ConditionalTrainConfig",
                  unit_coll: "torch.Tensor | None" = None,
                  fns: "StepFunctions | None" = None) -> dict[str, torch.Tensor]:
    """The five losses, each averaged over the batch's cases (one forward per loss).

    ``unit_coll`` is (k, n_collocation, n_coord) uniform [0, 1) draws placing each
    case's PDE points in its box; drawn here when None. Each loss runs inside a
    ``loss/<name>`` profiler range, so a trace attributes the step's time to it.
    The latent is passed per case (``b.z`` with each point set's sizes), never
    expanded to one row per point."""
    n_obs = len(STATE_VARS)
    k = b.k
    if getattr(model, "encoder", None) is not None:   # encode once; every loss shares it
        b = replace(b, z=model.encode(b.z))
    fns = fns if fns is not None else step_functions(model)
    # derivative-free forwards, optionally in bf16; results back to float32 for the losses
    plain = lambda coords, sizes: _forward_maybe_bf16(
        fns.forward, coords, b.z, sizes, cfg.bf16_data)[:, :n_obs]
    if cfg.predict_change:
        if not _taylor_supported(model):
            raise ValueError("predict_change needs a tanh model (forward-Taylor derivatives).")
        state = lambda coords, sizes, node: (
            b.base_v[node] + _ramp(b, coords, sizes) * plain(coords, sizes))
    else:
        state = lambda coords, sizes, node: plain(coords, sizes)

    with record_function("loss/data"):
        pred = state(b.interior, b.interior_sizes, b.interior_node)
        err2 = (((pred - b.targets) * b.target_mask) ** 2).sum(dim=1)
        # output_size keeps repeat_interleave from reading the GPU sizes (a sync)
        seg = torch.repeat_interleave(torch.arange(k, device=pred.device), b.interior_sizes,
                                      output_size=b.interior.shape[0])
        sse = torch.zeros(k, device=pred.device).index_add_(0, seg, err2)
        n_meas = torch.zeros(k, device=pred.device).index_add_(0, seg, b.target_mask.sum(dim=1))
        data = (sse / n_meas.clamp_min(1.0)).mean()

    with record_function("loss/initial"):
        ic = state(b.initial_coords, b.initial_sizes, b.initial_node)
        initial = (_per_case_mean(((ic - b.initial_targets) ** 2).sum(dim=1), b.initial_sizes, k)
                   / n_obs).mean()

    with record_function("loss/flow_boundary"):
        bc = state(b.boundary_coords, b.boundary_sizes, b.boundary_node)
        flow = (_per_case_mean(((bc - b.boundary_targets) ** 2).sum(dim=1), b.boundary_sizes, k)
                / n_obs).mean()

    scaling = cfg.scaling if cfg.scaling is not None else DEFAULT_RESIDUAL_SCALING
    zero = torch.zeros((), device=pred.device, dtype=pred.dtype)

    if cfg.weight_pde != 0.0:
        with record_function("loss/pde"):
            u = unit_coll if unit_coll is not None else torch.rand(
                (k, cfg.n_collocation, b.box_lo.shape[1]), device=pred.device, dtype=pred.dtype)
            if cfg.predict_change:
                pde = _pde_loss_on_nodes(model, b, u, cfg, scaling, fns)
            else:
                source_off, source_scale = _coordinate_affine(cfg.source_scaling, u)
                model_off, model_scale = _coordinate_affine(scaling, u)

                sampled = (b.box_lo[:, None] + u * (b.box_hi - b.box_lo)[:, None])
                physical = sampled * source_scale + source_off
                h = terrain_at(physical[:, :, 0], physical[:, :, 1], b.terrain_x, b.terrain_y, b.terrain_h,)[0]

                if cfg.source_z_mode == "physical_altitude":
                    z0 = b.box_lo[:, 2:3] * source_scale[2] + source_off[2]
                    z1 = b.box_hi[:, 2:3] * source_scale[2] + source_off[2]
                    lower = torch.maximum(torch.minimum(z0, z1), h)
                    upper = torch.maximum(z0, z1)
                    physical[:, :, 2] = (lower + u[:, :, 2] * (upper - lower))
                else:
                    physical[:, :, 2] = _source_height_to_altitude(physical[:, :, 2], h, cfg)

                coll = ((physical - model_off) / model_scale).reshape(-1, 4)
                pde = _pde_loss(model, coll, b.z, cfg, scaling, fns)
    else:
        pde = zero

    if cfg.weight_surface != 0.0:
        with record_function("loss/surface"):
            surface = _surface_loss(model, b, cfg, scaling, fns)
    else:
        surface = zero

    return {"data": data, "initial": initial, "flow_boundary": flow,
            "pde": pde, "surface": surface}


def _case_losses(model: nn.Module, case: ConditionalCase,
                 cfg: "ConditionalTrainConfig") -> dict[str, torch.Tensor]:
    """The five losses for one case."""
    return _batch_losses(model, stack_cases([case], cfg), cfg)


def _average_across_processes(params: list[nn.Parameter], losses: torch.Tensor,
                              world: int) -> torch.Tensor:
    """Average every gradient and the logged losses over processes in ONE all-reduce.

    All gradients and the losses are packed into one flat buffer, reduced once and
    unpacked: the same sum-then-divide as reducing each tensor alone, but one
    collective call per step instead of one per parameter (each a network round
    trip across nodes). Returns the averaged losses; gradients are updated in place."""
    grads = [p.grad if p.grad is not None else torch.zeros_like(p) for p in params]
    flat = torch.cat([g.reshape(-1) for g in grads] + [losses.reshape(-1)])
    dist.all_reduce(flat)
    flat.div_(world)
    offset = 0
    for p, g in zip(params, grads):
        n = g.numel()
        if p.grad is None:
            p.grad = g
        p.grad.copy_(flat[offset:offset + n].view_as(g))
        offset += n
    return flat[offset:].view_as(losses)


class _CaseFiles(torch.utils.data.Dataset):
    """One case per index, read in a DataLoader worker process."""

    def __init__(self, paths: "list[str | Path]", latent: str = "pod") -> None:
        self.paths, self.latent = list(paths), latent

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int) -> ConditionalCase:
        return read_conditional_case(self.paths[i], self.latent).as_torch(device="cpu")


class _StepDraws(torch.utils.data.Sampler):
    """Each step's case indices, drawn exactly as the in-process path draws them."""

    def __init__(self, n_cases: int, k: int, seed: int, n_steps: int) -> None:
        self.n_cases, self.k, self.seed, self.n_steps = n_cases, k, seed, n_steps

    def __len__(self) -> int:
        return self.n_steps

    def __iter__(self):
        rng = torch.Generator().manual_seed(self.seed)
        for _ in range(self.n_steps):
            yield torch.randperm(self.n_cases, generator=rng)[:self.k].tolist()


def _collate_passes(picked: "list[ConditionalCase]", per_pass: int,
                    cfg: ConditionalTrainConfig) -> list[CaseBatch]:
    """A step's cases, stacked into its passes (runs in the worker)."""
    return [stack_cases(picked[s:s + per_pass], cfg) for s in range(0, len(picked), per_pass)]


def _world() -> tuple[int, int]:
    """(rank, world_size); (0, 1) when not running distributed."""
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


def train_conditional(
    model: nn.Module,
    cases: "list[ConditionalCase] | None" = None,
    config: ConditionalTrainConfig = ConditionalTrainConfig(),
    *,
    case_paths: "list[str | Path] | None" = None,
    on_step: "Callable[[int], None] | None" = None,
) -> ConditionalHistory:
    """Train the shared model and return the loss history.

    Pass resident ``cases``, or ``case_paths`` to load each batch from disk per
    step. When torch.distributed is initialized, each process trains on the cases
    it was given and gradients are averaged across processes every step.
    ``on_step(epoch)`` runs after each optimizer step, for benchmark timing and
    profiler hooks; it must not touch the model."""
    if not cases and not case_paths:
        raise ValueError("train_conditional needs cases or case_paths.")

    rank, world = _world()
    device = _resolve_device(config.device)
    cuda = device.type == "cuda"
    model.to(device)
    manifest = (run_manifest(model, config, case_paths=case_paths, cases=cases)
                if rank == 0 and (config.checkpoint_path or config.events_path) else None)
    if rank == 0:
        write_event(config.events_path, {"event": "run_start", "manifest": manifest,
                                        "device": str(device), "world_size": world})
    if world > 1:  # every process starts from rank 0's weights
        for p in model.parameters():
            dist.broadcast(p.data, src=0)
    # Collapse check on log steps: share of saturated first-tanh units, and the latent's spread
    # across this step's cases relative to its size (near 0: every case looks the same).
    watch = {"on": False, "sat": [], "lat": []}

    def _saturation(module, inputs, out):
        if watch["on"]:
            watch["sat"].append((out.detach().abs() > 0.99).float().mean())

    def _latent(module, inputs, out):
        if watch["on"]:
            watch["lat"].append(out.detach())
    net = getattr(model, "net", None)
    if isinstance(net, nn.Sequential) and len(net) > 1 and isinstance(net[1], nn.Tanh):
        net[1].register_forward_hook(_saturation)
    if getattr(model, "latent_norm", None) is not None:
        model.latent_norm.register_forward_hook(_latent)
    # Resident cases live in HOST memory; only the per-step batch is moved to the GPU.
    if cases is not None:
        cases = [c.as_torch(device="cpu") for c in cases]
    n_cases = len(case_paths) if case_paths else len(cases)

    readers = ThreadPoolExecutor(max_workers=max(1, config.readers)) if case_paths else None
    per_pass = config.cases_per_pass or config.batch_cases

    def host_batch(idx: list[int]) -> list[CaseBatch]:
        """Build a step's passes on the host (runs on the prefetch thread)."""
        if case_paths is not None:
            load = lambda i: read_conditional_case(case_paths[i], config.latent).as_torch(device="cpu")
            picked = list(readers.map(load, idx))
        else:
            picked = [cases[i] for i in idx]
        passes = [stack_cases(picked[s:s + per_pass], config)
                  for s in range(0, len(picked), per_pass)]
        return [b.pin() for b in passes] if cuda else passes

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    fns = step_functions(model, compile=config.compile)
    rng = torch.Generator().manual_seed(config.seed + rank)
    history = ConditionalHistory()
    weights = {"data": config.weight_data, "initial": config.weight_initial,
               "flow_boundary": config.weight_flow_boundary,
               "pde": config.weight_pde, "surface": config.weight_surface}

    start_epoch = _maybe_resume(model, optimizer, history, config, device)
    k = min(config.batch_cases, n_cases)
    draw = lambda: torch.randperm(n_cases, generator=rng)[:k].tolist()

    prof = {"fwd": 0.0, "bwd": 0.0, "step": 0.0} if config.profile else None
    t_start = t_window = time.perf_counter()
    wait_window, steps_window = 0.0, 0

    # Prefetch: the next step's batch is stacked + pinned while the GPU works, on a
    # thread, or in DataLoader worker processes (same draws, same order).
    n_steps = config.epochs - start_epoch + 1
    if case_paths is not None and config.loader_workers > 0:
        loader = torch.utils.data.DataLoader(
            _CaseFiles(case_paths, config.latent), batch_sampler=_StepDraws(n_cases, k, config.seed + rank, n_steps),
            num_workers=config.loader_workers, collate_fn=partial(_collate_passes, per_pass=per_pass, cfg=config),
            pin_memory=cuda, prefetch_factor=2, persistent_workers=False)
        batches = iter(loader)
        next_batch = lambda: next(batches)
        pool = None
    else:
        pool = ThreadPoolExecutor(max_workers=1)
        pending = pool.submit(host_batch, draw())
        next_batch = lambda: pending.result()

    for epoch in range(start_epoch, config.epochs + 1):
        t_wait = time.perf_counter()
        passes = [b.to(device) for b in next_batch()]
        wait_window += time.perf_counter() - t_wait
        if pool is not None and epoch < config.epochs:
            pending = pool.submit(host_batch, draw())

        if prof is not None and cuda:
            torch.cuda.synchronize()
        t = time.perf_counter()
        lr = _learning_rate(epoch, config)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad()
        watch["on"] = rank == 0 and (epoch == 1 or epoch == config.epochs or epoch % config.log_every == 0)
        # Gradient accumulation: each pass's loss is weighted by its share of the
        # batch, so the summed gradients equal one pass over all k cases.
        losses = torch.zeros(1 + len(weights), device=device)
        # One collocation draw per step, sliced per pass, so the PDE points do not
        # depend on cases_per_pass.
        unit_coll = (torch.rand((k, config.n_collocation, passes[0].box_lo.shape[1]),
                                device=device) if config.weight_pde != 0.0 else None)
        first = 0
        for b in passes:
            share = b.k / k
            u = unit_coll[first:first + b.k] if unit_coll is not None else None
            first += b.k
            batch = _batch_losses(model, b, config, u, fns)
            total = sum(weights[name] * batch[name] for name in weights)
            with record_function("step/backward"):
                (share * total).backward()
            losses += share * torch.stack([total.detach()] + [batch[n].detach() for n in weights])
            del batch, total, b  # free this pass's graph before the next
        if prof is not None:
            if cuda:
                torch.cuda.synchronize()
            prof["fwd"] += time.perf_counter() - t; t = time.perf_counter()
        if world > 1:  # average gradients and logged losses across processes
            with record_function("step/all_reduce"):
                losses = _average_across_processes(list(model.parameters()), losses, world)
        losses = losses.cpu().tolist()
        if not all(map(lambda x: x == x and abs(x) != float("inf"), losses)):
            raise FloatingPointError(f"Non-finite loss at epoch {epoch}.")
        # Clip gradients: PDE residual gradients (2nd derivatives via autograd) can
        # spike and freeze training. Standard PINN safeguard.
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
        if prof is not None:
            if cuda:
                torch.cuda.synchronize()
            prof["bwd"] += time.perf_counter() - t; t = time.perf_counter()
        with record_function("step/optimizer"):
            optimizer.step()
        if prof is not None:
            if cuda:
                torch.cuda.synchronize()
            prof["step"] += time.perf_counter() - t
        steps_window += 1

        history.total.append(losses[0])
        for name, value in zip(weights, losses[1:]):
            getattr(history, name).append(value)
        if on_step is not None:
            on_step(epoch)

        if rank == 0 and (epoch == 1 or epoch == config.epochs or epoch % config.log_every == 0):
            if config.events_path:
                norm = float(grad_norm.detach().cpu())
                write_event(config.events_path, {
                    "event": "step", "epoch": epoch,
                    "losses": dict(zip(("total", *weights), losses)),
                    "gradient_norm_before_clip": norm if math.isfinite(norm) else None,
                    "gradient_norm_finite": math.isfinite(norm),
                    "gradient_clip_norm": config.grad_clip_norm,
                })
            now = time.perf_counter()
            s_step = (now - t_window) / steps_window
            eta_h = s_step * (config.epochs - epoch) / 3600
            mem = f" gpu_mem={torch.cuda.max_memory_allocated(device) / 1e9:.1f}GB" if cuda else ""
            parts = " ".join(f"{n}={v:.4e}" for n, v in zip(weights, losses[1:]))
            # stderr + flush so progress is live in the SLURM log, never buffered
            print(f"epoch {epoch}/{config.epochs} total={losses[0]:.4e} {parts} | "
                  f"lr={lr:.2e} {s_step:.2f}s/step wait={wait_window / steps_window * 1e3:.0f}ms/step "
                  f"elapsed={(now - t_start) / 3600:.2f}h eta={eta_h:.2f}h{mem}",
                  file=sys.stderr, flush=True)
            if watch["sat"] or watch["lat"]:
                msg = f"  collapse check: first tanh saturated {torch.stack(watch['sat']).mean():.3f}" if watch["sat"] else "  collapse check:"
                if len(watch["lat"]) > 0:
                    L = torch.cat(watch["lat"])
                    if L.shape[0] > 1:
                        msg += (f", latent spread across {L.shape[0]} cases "
                                f"{((L - L.mean(0)).norm(dim=1).mean() / L.norm(dim=1).mean()):.4f}")
                print(msg, file=sys.stderr, flush=True)
            watch["sat"].clear(); watch["lat"].clear()
            if prof is not None:
                tot = sum(prof.values()) or 1.0
                print(f"  profile/epoch(avg over {epoch}): "
                      f"fwd+bwd={prof['fwd']/epoch*1e3:.1f}ms ({prof['fwd']/tot*100:.0f}%) "
                      f"sync+clip={prof['bwd']/epoch*1e3:.1f}ms ({prof['bwd']/tot*100:.0f}%) "
                      f"step={prof['step']/epoch*1e3:.1f}ms ({prof['step']/tot*100:.0f}%)",
                      file=sys.stderr, flush=True)
            t_window, wait_window, steps_window = now, 0.0, 0

        if rank == 0 and config.checkpoint_path and (epoch % config.checkpoint_every == 0
                                                     or epoch == config.epochs):
            _save_checkpoint(model, optimizer, history, epoch, config, manifest)

    if pool is not None:
        pool.shutdown()
    if readers is not None:
        readers.shutdown()
    return history


def _learning_rate(epoch: int, config: ConditionalTrainConfig) -> float:
    """Constant, or cosine from learning_rate at epoch 1 to learning_rate_min at the last."""
    if config.learning_rate_min is None:
        return config.learning_rate
    frac = (epoch - 1) / max(config.epochs - 1, 1)
    lo, hi = config.learning_rate_min, config.learning_rate
    return lo + 0.5 * (hi - lo) * (1.0 + math.cos(math.pi * frac))


def _save_checkpoint(model, optimizer, history, epoch, config, manifest=None) -> None:
    """Atomically write a resumable checkpoint, plus a step-tagged copy kept for good."""
    path = Path(config.checkpoint_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save({"schema_version": 3, "geometry_version": 2, "manifest": manifest,
                "epoch": epoch, "model": model.state_dict(),
                "optimizer": optimizer.state_dict(), "history": vars(history)}, tmp)
    tagged = path.with_name(f"{path.stem}_step{epoch:06d}{path.suffix}")
    tagged.write_bytes(tmp.read_bytes())
    tmp.replace(path)
    write_event(config.events_path, {"event": "checkpoint", "epoch": epoch, "path": str(path)})
    print(f"checkpoint @ epoch {epoch} -> {path} (+ {tagged.name})", file=sys.stderr, flush=True)


def _maybe_resume(model, optimizer, history, config, device) -> int:
    """Load a checkpoint if present; return the epoch to start from (1 if none)."""
    if not config.checkpoint_path or not Path(config.checkpoint_path).is_file():
        return 1
    ckpt = torch.load(config.checkpoint_path, map_location=device)
    saved = (ckpt.get("manifest") or {}).get("training", {})
    current = asdict(config)
    keys = ("scaling", "source_scaling", "source_z_mode", "terrain_top_height",
    "fe_surface_reference_z", "physics", "tke_weight", "tke_residual_scale", "predict_change",
    "terrain_xy_scale")

    if (ckpt.get("geometry_version") != 2 or any(saved.get(k) != current[k] for k in keys)):
        raise ValueError("Checkpoint geometry/physics/TKE loss settings differ; use a new checkpoint path.")
    
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    for name, vals in ckpt["history"].items():
        getattr(history, name).extend(vals)
    print(f"resumed from {config.checkpoint_path} at epoch {ckpt['epoch']}",
          file=sys.stderr, flush=True)
    return ckpt["epoch"] + 1


def split_cases(cases: list[ConditionalCase], chi: float, seed: int = 0
                ) -> tuple[list[ConditionalCase], list[ConditionalCase]]:
    """Split a case family into a train fraction ``chi`` and a held-out remainder."""
    import numpy as np

    order = np.random.default_rng(seed).permutation(len(cases))
    n_train = max(1, round(len(cases) * chi))
    train_idx = set(order[:n_train].tolist())
    train = [c for i, c in enumerate(cases) if i in train_idx]
    held = [c for i, c in enumerate(cases) if i not in train_idx]
    return train, held
