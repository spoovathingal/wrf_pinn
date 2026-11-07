"""The conditional-PINN case: one sub-domain over one time window, holding its
conditioning fields (initial, boundary, terrain) and interior targets. One .npz
per case plus a shared metadata.json sidecar.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
import torch
import numpy as np

# Variable order in the existing eight-variable dataset.
STORED_STATE_VARS: tuple[str, ...] = ("u", "v", "w", "theta", "p_prime", "q_v", "e_sgs", "rho_d")
# Variables used by the simplified model.
STATE_VARS: tuple[str, ...] = ("u", "v", "w", "theta", "p_prime", "rho_d")
ACTIVE_STATE_INDICES = tuple(STORED_STATE_VARS.index(name) for name in STATE_VARS)

#: Interior sample coordinate columns.
COORD_NAMES: tuple[str, ...] = ("x", "y", "z", "t")
#: The four open side faces of a sub-domain, in a fixed order.
FACE_NAMES: tuple[str, ...] = ("west", "east", "south", "north")

def _select_active_state(values: np.ndarray, axis: int) -> np.ndarray:
    """Select six active fields from the supported stored array layouts."""
    n = values.shape[axis]

    if n == len(STORED_STATE_VARS):
        return np.take(values, ACTIVE_STATE_INDICES, axis=axis)

    if n == len(STATE_VARS):
        return values

    raise ValueError(f"Expected 8 stored or 6 active state variables; got {n}.")

@dataclass(frozen=True)
class ConditionalCase:
    """One sub-domain + time window: conditioning fields and targets, as numpy or
    torch arrays."""

    name: str
    z: object                   # (k,): concatenated POD coeffs; the conditioning latent
    initial_coords: object      # (n_ic, n_coord): where to enforce the initial field
    initial: object             # (n_state, nz, nx*ny): decoded initial field
    boundary: object            # (n_state, n_faces, n_times, face_len): decoded boundary
    boundary_coords: object     # psi coords: (n_faces, n_times, face_len, n_coord)
    surface: object             # (n_times, n_face_pts, 5): x,y,t,fricVel,htFlux (bottom)
    surface_fe_state: object # Physical FE observations at native z-index 0: (n_times, n_surface, 4), columns rho_d, u, v, theta.
    surface_z0m: object # Physical momentum roughness [m], shape (n_surface, 1).
    surface_height: object # Native terrain elevation above sea level [m], shape (n_surface, 1).
    terrain: object             # (n_terr, 3): x, y, elevation (static)
    interior: object            # (n_pts, n_coord): query coords, tau > 0
    targets: object             # (n_pts, n_state): matched interior state
    target_mask: object         # (n_pts, n_state): 1 where measured
    times: object               # (n_times,): window snapshot times
    initial_raw: object = None  # (n_state, nz, nx*ny): exact initial field (--keep-raw cases)
    state_vars: tuple[str, ...] = STATE_VARS
    coord_names: tuple[str, ...] = COORD_NAMES
    face_names: tuple[str, ...] = FACE_NAMES

    @property
    def n_interior(self) -> int:
        return self.interior.shape[0]

    def as_torch(self, *, dtype: object | None = None, device: object | None = None):
        """Return a new ConditionalCase with members as torch tensors on device."""
        import torch

        td = dtype if dtype is not None else torch.float32

        def t(a):
            return torch.as_tensor(a, dtype=td, device=device)

        return ConditionalCase(
            name=self.name,
            z=t(self.z),
            initial_coords=t(self.initial_coords),
            initial=t(self.initial),
            boundary=t(self.boundary),
            boundary_coords=t(self.boundary_coords),
            surface=t(self.surface),
            surface_fe_state=t(self.surface_fe_state),
            surface_z0m=t(self.surface_z0m),
            surface_height=t(self.surface_height),
            terrain=t(self.terrain),
            interior=t(self.interior),
            targets=t(self.targets),
            target_mask=t(self.target_mask),
            times=t(self.times),
            initial_raw=None if self.initial_raw is None else t(self.initial_raw),
            state_vars=self.state_vars,
            coord_names=self.coord_names,
            face_names=self.face_names,
        )


#: Non-POD members written by the generator and read back directly. The initial
#: and boundary fields are reconstructed from their per-subdomain POD pieces;
#: ``interior`` is rebuilt (``interior_from_grid``) when a case omits it.
_CASE_MEMBERS: tuple[str, ...] = (
    "initial_coords", "boundary_coords", "surface", "surface_fe_state", "surface_z0m",
    "surface_height", "terrain", "targets", "times")


def interior_from_grid(initial_coords: np.ndarray, boundary_coords: np.ndarray) -> np.ndarray:
    """Interior query coords: the initial grid at each lead time (lead-major, initial
    point order), the leads' times taken from the boundary coords' snapshot times."""
    t = boundary_coords[0, 1:, 0, 3]
    xyz = np.tile(initial_coords[:, :3], (len(t), 1))
    return np.concatenate([xyz, np.repeat(t, len(initial_coords))[:, None]], 1).astype(np.float32)


def read_conditional_case(npz_path: str | Path, latent: str = "pod",
                          initial_from_targets: bool = False) -> ConditionalCase:
    """Load one .npz case; decode initial/boundary, mask NaN targets. A case written
    without POD (the carver's --no-pod) takes them from its raw fields instead.
    latent picks z: 'pod' (POD coeffs) or 'raw' (the full raw fields, see latent_vector).
    initial_from_targets: ignore any POD and start the window at its first lead
    snapshot, taken raw from ``targets`` (see first_lead_as_initial)."""
    path = Path(npz_path)
    if not path.exists():
        raise FileNotFoundError(f"Conditional case not found: {path}.")

    with np.load(path) as blob:
        missing = [k for k in _CASE_MEMBERS if k not in blob.files]
        if missing:
            raise ValueError(f"Case {path} missing members: {missing}.")
        members = {k: blob[k].astype(np.float32) for k in _CASE_MEMBERS}
        members["interior"] = (blob["interior"].astype(np.float32) if "interior" in blob.files
                               else interior_from_grid(members["initial_coords"],
                                                       members["boundary_coords"]))
        members["targets"] = _select_active_state(members["targets"], axis=-1)
        targets = members["targets"]

        if targets.ndim != 2 or targets.shape[1] != len(STATE_VARS):
            raise ValueError(
                f"Case {path} targets must have {len(STATE_VARS)} columns "
                f"in order {STATE_VARS}; got shape {targets.shape}. "
                "Regenerate the case with dry-air density included."
            )
        if targets.shape[0] != members["interior"].shape[0]:
            raise ValueError(
                f"Case {path} has different numbers of targets and interior coordinates."
            )

        surface = members["surface"]
        fe = members["surface_fe_state"]
        z0m = members["surface_z0m"]

        if surface.ndim != 3 or surface.shape[2] != 5:
            raise ValueError(f"Case {path}: surface must have shape (n_times, n_surface, 5).")

        nt, ns, _ = surface.shape
        if nt == 0 or ns == 0 or members["times"].shape != (nt,):
            raise ValueError(f"Case {path}: invalid surface/time dimensions.")

        if fe.shape != (nt, ns, 4):
            raise ValueError(
                f"Case {path}: surface_fe_state must have shape ({nt}, {ns}, 4).")

        if z0m.shape != (ns, 1):
            raise ValueError(f"Case {path}: surface_z0m must have shape ({ns}, 1).")

        if members["surface_height"].shape != (ns, 1):
            raise ValueError(f"Case {path}: surface_height must have shape ({ns}, 1).")

        for name in ("surface", "surface_fe_state", "surface_z0m", "surface_height"):
            if not np.isfinite(members[name]).all():
                raise ValueError(f"Case {path}: {name} must contain finite values.")

        if np.any(fe[..., [0, 3]] <= 0.0):
            raise ValueError(
                f"Case {path}: physical FE density and theta must be positive.")

        if np.any(z0m <= 0.0):
            raise ValueError(f"Case {path}: surface_z0m must be positive.")

        if not np.all(surface[..., :2] == surface[0:1, :, :2]):
            raise ValueError(
                f"Case {path}: surface x/y point order must stay fixed across times.")

        required_pod = [
            f"ini_{part}_{var}" for var in STATE_VARS
            for part in ("mean", "modes", "coeffs")
        ]
        required_pod += [
            f"bnd_{part}_{var}_{face}" for var in STATE_VARS
            for face in range(len(FACE_NAMES)) for part in ("mean", "modes", "coeffs")
        ]
        missing_pod = [key for key in required_pod if key not in blob.files]
        if initial_from_targets:
            if latent == "pod":
                raise ValueError("initial_from_targets uses no POD; use a raw or encoder latent.")
            members.update(first_lead_as_initial(members, path))
            members["z"] = latent_vector(
                {"bnd_raw": members.pop("bnd_raw"), "ini_raw": members["initial_raw"],
                 "terrain": members["terrain"]}, latent)
            return _finish_case(path, members)
        raw_only = bool(missing_pod) and {"ini_raw", "bnd_raw"} <= set(blob.files)
        if raw_only and latent == "pod":
            raise ValueError(f"Case {path} has no POD arrays; use a raw or encoder latent.")
        if missing_pod and not raw_only:
            raise ValueError(
                f"Case {path} is missing required POD arrays: {missing_pod}. "
                "The case must provide POD arrays for all active fields "
                "or both ini_raw and bnd_raw."
            )

        bnd = raw_boundary(blob) if raw_only else decode_boundary(blob)
        ini = raw_initial(blob) if raw_only else decode_initial(blob)
        members["z"] = latent_vector(blob, latent)
        if "ini_raw" in blob.files:
            members["initial_raw"] = _select_active_state(
                blob["ini_raw"], axis=0).astype(np.float32)

    members["boundary"] = np.stack([bnd[v] for v in STATE_VARS], axis=0).astype(np.float32)
    members["initial"] = np.stack([ini[v] for v in STATE_VARS], axis=0).astype(np.float32)
    return _finish_case(path, members)


def _finish_case(path: Path, members: dict) -> ConditionalCase:
    """Check coordinates, mask NaN targets, build the case."""
    if not np.isfinite(members["interior"]).all():
        raise ValueError(f"Non-finite interior coordinates in {path}.")

    mask = np.isfinite(members["targets"]).astype(np.float32)
    members["targets"] = np.where(mask > 0.0, members["targets"], 0.0).astype(np.float32)
    members["target_mask"] = mask

    return ConditionalCase(name=path.stem, **members)


def decode_boundary(blob) -> dict:
    """Reconstruct the boundary state from a case's per-face POD pieces.

    Returns {var: (n_faces, n_times, face_len)} in normalized units. blob is an
    np.load handle (or dict) of a case .npz written by the preprocessor."""
    out = {}
    face = 0
    nf = sum(1 for k in blob.files if k.startswith(f"bnd_modes_{STATE_VARS[0]}_"))
    for var in STATE_VARS:
        faces = []
        for f in range(nf):
            mean = blob[f"bnd_mean_{var}_{f}"]       # (face_len,)
            modes = blob[f"bnd_modes_{var}_{f}"]     # (face_len, k)
            coeffs = blob[f"bnd_coeffs_{var}_{f}"]   # (n_times, k)
            faces.append(mean + coeffs @ modes.T)    # (n_times, face_len)
        out[var] = np.stack(faces, axis=0)           # (n_faces, n_times, face_len)
    return out


def decode_initial(blob) -> dict:
    """Reconstruct the initial state from a case's spatial POD pieces.

    Returns {var: (nz, nx*ny)} in normalized units (z-levels x horizontal)."""
    out = {}
    for var in STATE_VARS:
        mean = blob[f"ini_mean_{var}"]               # (nx*ny,)
        modes = blob[f"ini_modes_{var}"]             # (nx*ny, k)
        coeffs = blob[f"ini_coeffs_{var}"]           # (nz, k)
        out[var] = mean + coeffs @ modes.T           # (nz, nx*ny)
    return out


def first_lead_as_initial(members: dict, path: Path) -> dict:
    """Members for the same window started at its first lead snapshot, all raw.

    ``interior``/``targets`` must hold the full initial grid at every lead (lead-major,
    initial point order), as the builders write them. The first lead becomes the initial
    field (initial_coords, initial, initial_raw); its boundary faces at every remaining
    time come from the lead snapshots by node lookup (boundary, bnd_raw); the first lead
    leaves interior/targets, and the first snapshot leaves surface, surface_fe_state, times.
    No POD piece is read, so nothing encoded enters the latent or any loss target."""
    ic, interior, targets = members["initial_coords"], members["interior"], members["targets"]
    n, nt = len(ic), len(members["times"])
    if interior.shape[0] != (nt - 1) * n or not np.allclose(
            interior[:, :3], np.tile(ic[:, :3], (nt - 1, 1)), rtol=0, atol=1e-6):
        raise ValueError(f"Case {path}: interior is not the initial grid at each lead time.")
    axes = [np.unique(ic[:, a]) for a in range(3)]
    nx, ny, nz = (len(a) for a in axes)
    if nx * ny * nz != n:
        raise ValueError(f"Case {path}: initial points are not a full grid.")

    def node(xyz):                       # flat initial-grid index (x, y slow; z fast)
        idx = [np.searchsorted(a, xyz[..., k]) for k, a in enumerate(axes)]
        for k, (a, i) in enumerate(zip(axes, idx)):
            if not np.allclose(a[np.minimum(i, len(a) - 1)], xyz[..., k], rtol=0, atol=1e-6):
                raise ValueError(f"Case {path}: boundary points are not on the initial grid.")
        return (idx[0] * ny + idx[1]) * nz + idx[2]

    lead = targets.reshape(nt - 1, n, -1)                        # (lead, point, var)
    bcoords = members["boundary_coords"][:, 1:]                  # faces at the lead times
    bnd_raw = np.stack([lead[k][node(bcoords[:, k, :, :3])]      # (face, lead, face_len, var)
                        for k in range(nt - 1)], axis=1)
    ini_raw = lead[0].reshape(nx * ny, nz, -1).transpose(2, 1, 0)   # (var, nz, nx*ny)
    return {
        "initial_coords": interior[:n], "initial": ini_raw, "initial_raw": ini_raw,
        "boundary_coords": bcoords, "boundary": np.moveaxis(bnd_raw, -1, 0), "bnd_raw": bnd_raw,
        "interior": interior[n:], "targets": targets[n:],
        "surface": members["surface"][1:], "surface_fe_state": members["surface_fe_state"][1:],
        "times": members["times"][1:],
    }


def raw_boundary(blob) -> dict:
    """The boundary state from bnd_raw (n_faces, n_times, face_len, n_state), in
    decode_boundary's layout {var: (n_faces, n_times, face_len)}."""
    b = _select_active_state(blob["bnd_raw"], axis=-1)
    return {var: b[..., j] for j, var in enumerate(STATE_VARS)}


def raw_initial(blob) -> dict:
    """The initial state from ini_raw (n_state, nz, nx*ny), in decode_initial's layout."""
    i = _select_active_state(blob["ini_raw"], axis=0)
    return {var: i[j] for j, var in enumerate(STATE_VARS)}


def terrain_grid(terrain) -> np.ndarray:
    """Terrain rows (x, y, elevation) -> elevations (nx*ny,), x slow like ini_raw."""
    t = np.asarray(terrain, dtype=np.float64)
    return t[np.lexsort((t[:, 1], t[:, 0])), 2]


def latent_vector(blob, latent: str = "pod") -> np.ndarray:
    """Concatenate the case's POD coeffs into the conditioning latent z. Order is
    fixed (boundary faces then initial, by STATE_VARS) so z is consistent across cases.
    latent='raw': the normalized raw fields instead, bnd_raw (n_faces, n_times,
    face_len, n_state) then ini_raw (n_state, nz, nx*ny), flattened (needs --keep-raw).
    latent='raw_terrain': the same, then the ground height (m) on the grid (terrain_grid)."""
    if latent in ("raw", "raw_terrain"):
        parts = [_select_active_state(blob["bnd_raw"], axis=-1).reshape(-1),
            _select_active_state(blob["ini_raw"], axis=0).reshape(-1)]
        if latent == "raw_terrain":
            parts.append(terrain_grid(blob["terrain"]))
        return np.concatenate(parts).astype(np.float32)
    if latent != "pod":
        raise ValueError(f"latent must be 'pod', 'raw' or 'raw_terrain', got {latent!r}.")
    nf = sum(1 for k in blob.files if k.startswith(f"bnd_modes_{STATE_VARS[0]}_"))
    parts = []
    for var in STATE_VARS:
        for f in range(nf):
            parts.append(blob[f"bnd_coeffs_{var}_{f}"].reshape(-1))
    for var in STATE_VARS:
        parts.append(blob[f"ini_coeffs_{var}"].reshape(-1))
    return np.concatenate(parts).astype(np.float32)


@dataclass(frozen=True)
class ConditionalMetadata:
    """Shared sidecar: variable order and the normalization recipe (if any)."""

    state_vars: tuple[str, ...]
    coord_names: tuple[str, ...]
    normalization: dict

    @classmethod
    def load(cls, path: str | Path) -> "ConditionalMetadata":
        raw = json.loads(Path(path).read_text())
        schema = raw.get("schema", {})
        stored_vars = tuple(schema.get("state_vars", ()))

        if stored_vars not in (STORED_STATE_VARS, STATE_VARS):
            raise ValueError(
                f"Metadata {path} must use the supported eight- or "
                f"six-variable order; got {stored_vars}."
            )

        expected_surface = {
            "state_vars": ["rho_d", "u", "v", "theta"],
            "state_units": ["kg/m^3", "m/s", "m/s", "K"],
            "height_units": "m", "z_index": 0, "z0m_units": "m",
        }

        if schema.get("surface_reference") != expected_surface:
            raise ValueError(
                f"Metadata {path} must declare surface_reference={expected_surface}."
            )

        recipe = dict(raw.get("normalization", {}))

        if tuple(recipe.get("state_vars", ())) != stored_vars:
            raise ValueError(
                f"Metadata {path}: schema and normalization state orders differ."
            )

        indices = [stored_vars.index(name) for name in STATE_VARS]

        for key in ("state_offset", "state_scale"):
            values = recipe.get(key, [])

            if len(values) != len(stored_vars):
                raise ValueError( f"Metadata {path}: invalid {key} length." )

            recipe[key] = [values[i] for i in indices]

        recipe["state_vars"] = list(STATE_VARS)

        return cls(
            state_vars=STATE_VARS,
            coord_names=tuple(schema.get("coord_names", COORD_NAMES)),
            normalization=recipe,
        )

def prepare_terrain(terrain):
    """Build sorted terrain axes and heights once on CPU.

    Input columns are physical x, y, elevation in metres.
    Returns x_axis, y_axis, height[x_index, y_index].
    """
    terrain = torch.as_tensor(terrain, device="cpu").detach()
    if not terrain.is_floating_point():
        terrain = terrain.float()

    if (terrain.ndim != 2 or terrain.shape[1] != 3 or not torch.isfinite(terrain).all()):
        raise ValueError("terrain must contain finite (x, y, elevation) rows")

    x_axis = torch.unique(terrain[:, 0], sorted=True)
    y_axis = torch.unique(terrain[:, 1], sorted=True)
    nx, ny = x_axis.numel(), y_axis.numel()

    if nx < 2 or ny < 2:
        raise ValueError("terrain requires at least two x and y values")

    ix = torch.searchsorted(x_axis, terrain[:, 0].contiguous())
    iy = torch.searchsorted(y_axis, terrain[:, 1].contiguous())

    if (len(terrain) != nx * ny or torch.unique(ix * ny + iy).numel() != nx * ny):
        raise ValueError("terrain must be a complete grid without duplicate rows")

    height = terrain.new_empty((nx, ny))
    height[ix, iy] = terrain[:, 2]
    return x_axis, y_axis, height


def terrain_at(xq, yq, x_axis, y_axis, height):
    """Return physical terrain height and slopes dh/dx, dh/dy.

    Single grid: axes (nx,), (ny,), height (nx, ny).
    Batched grids: axes (k, nx), (k, ny), height (k, nx, ny);
    queries must then have shape (k, n).

    Queries and grid tensors must use physical metres and the same device.
    Boundary cells extrapolate linearly beyond the sampled terrain.
    """

    ix = torch.searchsorted(x_axis, xq.contiguous(), right=True).clamp(1, x_axis.shape[-1] - 1) - 1
    iy = torch.searchsorted(y_axis, yq.contiguous(), right=True).clamp(1, y_axis.shape[-1] - 1) - 1

    batch = ()
    if x_axis.ndim == 2:
        batch = (torch.arange(height.shape[0], device=height.device)[:, None],)

    x0 = x_axis[batch + (ix,)]
    x1 = x_axis[batch + (ix + 1,)]
    y0 = y_axis[batch + (iy,)]
    y1 = y_axis[batch + (iy + 1,)]

    h00 = height[batch + (ix, iy)]
    h10 = height[batch + (ix + 1, iy)]
    h01 = height[batch + (ix, iy + 1)]
    h11 = height[batch + (ix + 1, iy + 1)]

    tx = (xq - x0) / (x1 - x0)
    ty = (yq - y0) / (y1 - y0)

    a = h00 + tx * (h10 - h00)
    b = h01 + tx * (h11 - h01)

    h = a + ty * (b - a)
    hx = ( (1 - ty) * (h10 - h00) + ty * (h11 - h01) ) / (x1 - x0)
    hy = (b - a) / (y1 - y0)

    return h, hx, hy