"""Complete 7-variable FastEddy cases (no rho_d) to the 8-variable layout on load.

Dry density follows from the equation of state used by the PDE residuals,
p = (R_d rho_d theta_m / p0^kappa)^gamma, with total pressure p = p_base(altitude) + p'.
p_base is a fixed hydrostatic profile of altitude above sea level, fitted on cases of the
same FastEddy run that store rho_d. The roughness map z0m comes from the run's grid file.

A dataset opts in through its metadata.json: an 8-variable schema plus
``"seven_var_completion": {"file": "<npz next to metadata.json>"}``. The npz holds
altitude_m, p_base_pa (the profile), z0m (ny, nx map indexed by grid y, x),
rho_d_offset, rho_d_scale and terrain_top_height.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from wrf_pinn.config.physics import PhysicsConfig

#: Members a 7-variable case lacks; complete_seven_var supplies them.
COMPLETED_MEMBERS = ("surface_fe_state", "surface_z0m", "surface_height")


@lru_cache(maxsize=4)
def _completion(dataset_dir: str) -> dict:
    meta = json.loads((Path(dataset_dir) / "metadata.json").read_text())
    entry = meta.get("seven_var_completion")
    if entry is None:
        raise ValueError(f"{dataset_dir}/metadata.json has no seven_var_completion entry; "
                         "7-variable cases cannot be read as 8-variable ones.")
    norm = meta["normalization"]
    with np.load(Path(dataset_dir) / entry["file"]) as f:
        c = {k: f[k] for k in f.files}
    c["state_offset"] = np.asarray(norm["state_offset"], dtype=np.float64)
    c["state_scale"] = np.asarray(norm["state_scale"], dtype=np.float64)
    c["coord_offset"] = np.asarray(norm["coord_offset"], dtype=np.float64)
    c["coord_scale"] = np.asarray(norm["coord_scale"], dtype=np.float64)
    return c


def rho_d_from_state(theta, p_prime, q_v_g_per_kg, altitude, c: dict) -> np.ndarray:
    """Dry density (kg/m^3) from physical theta (K), p' (Pa), q_v (g/kg) and altitude (m)."""
    k = PhysicsConfig().constants
    rd, rv, p0, kappa, gamma = (k.dry_air_gas_constant, k.water_vapor_gas_constant,
                                k.reference_pressure, k.kappa, k.gamma)
    p = np.interp(altitude, c["altitude_m"], c["p_base_pa"]) + p_prime
    theta_m = theta * (1.0 + (rv / rd) * q_v_g_per_kg * 1e-3)
    return p ** (1.0 / gamma) * p0**kappa / (rd * theta_m)


def complete_seven_var(path: Path, members: dict, blob) -> dict:
    """rho_d target column and the surface members for one 7-variable case.

    members holds the case's float32 arrays (targets with 7 columns, interior,
    initial_coords, terrain, surface, times); blob is the open .npz (for ini_raw).
    Returns {targets (8 columns), surface_fe_state, surface_z0m, surface_height}."""
    c = _completion(str(Path(path).parent.parent))
    so, ss, co, cs = c["state_offset"], c["state_scale"], c["coord_offset"], c["coord_scale"]
    top = float(c["terrain_top_height"])
    terrain = members["terrain"].astype(np.float64)                 # grid x, grid y, height (m)
    key = lambda x, y: np.rint(x).astype(np.int64) * 1_000_003 + np.rint(y).astype(np.int64)
    order = np.argsort(key(terrain[:, 0], terrain[:, 1]))
    sorted_keys = key(terrain[:, 0], terrain[:, 1])[order]

    def column(x, y):                                               # terrain row of grid x, y
        k = key(x, y)
        i = np.minimum(np.searchsorted(sorted_keys, k), len(sorted_keys) - 1)
        if np.any(sorted_keys[i] != k):
            raise ValueError(f"{path}: points off the case's terrain grid.")
        return order[i]

    def rho_norm(state7, xyz):                                      # normalized 7-var rows, physical xyz
        s = state7.astype(np.float64) * ss[:7] + so[:7]
        h = terrain[column(xyz[:, 0], xyz[:, 1]), 2]
        alt = h + xyz[:, 2] * (1.0 - h / top)
        return rho_d_from_state(s[:, 3], s[:, 4], s[:, 5], alt, c)

    phys = lambda coords: coords[:, :3].astype(np.float64) * cs[:3] + co[:3]
    targets = members["targets"]
    rho = rho_norm(targets, phys(members["interior"]))
    targets8 = np.concatenate([targets, ((rho - so[7]) / ss[7])[:, None]], 1).astype(np.float32)

    # Level-0 state at each snapshot: ini_raw for the first, the targets for the leads.
    ic = phys(members["initial_coords"])
    level0 = np.flatnonzero(np.isclose(ic[:, 2], ic[:, 2].min()))
    n, nt = len(ic), len(members["times"])
    surface = members["surface"]
    sxy = surface[0, :, :2].astype(np.float64) * cs[:2] + co[:2]
    ic_key = key(ic[level0, 0], ic[level0, 1])
    pos = {k: j for j, k in enumerate(ic_key)}
    try:
        rows = level0[[pos[k] for k in key(sxy[:, 0], sxy[:, 1])]]   # initial-grid row per surface point
    except KeyError:
        raise ValueError(f"{path}: surface points off the initial grid.") from None
    raw = blob["ini_raw"]                                           # (7, nz, nx*ny), x slow
    t_cols = np.lexsort((terrain[:, 1], terrain[:, 0]))             # ini_raw column -> terrain row
    col_of_terrain = np.empty(len(t_cols), dtype=np.int64)
    col_of_terrain[t_cols] = np.arange(len(t_cols))
    cols = column(sxy[:, 0], sxy[:, 1])
    first = raw[:, 0, col_of_terrain[cols]].T                                   # (ns, 7)
    snaps = [(first, rho_norm(first, np.c_[sxy, np.full(len(sxy), ic[level0[0], 2])]))]
    snaps += [(targets8[k * n + rows, :7], targets8[k * n + rows, 7] * ss[7] + so[7])
              for k in range(nt - 1)]
    fe = []
    for s7, r in snaps:
        s = s7.astype(np.float64) * ss[:7] + so[:7]
        fe.append(np.stack([r, s[:, 0], s[:, 1], s[:, 3]], 1))
    z0m = c["z0m"][np.rint(sxy[:, 1]).astype(np.int64), np.rint(sxy[:, 0]).astype(np.int64)]
    return {"targets": targets8,
            "surface_fe_state": np.stack(fe).astype(np.float32),
            "surface_z0m": z0m[:, None].astype(np.float32),
            "surface_height": terrain[cols, 2][:, None].astype(np.float32)}
