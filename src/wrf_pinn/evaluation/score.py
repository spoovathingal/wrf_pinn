"""Scorecard of a trained conditional model on held-out cases, in physical units."""

from collections import deque
from concurrent.futures import ThreadPoolExecutor

import torch

from wrf_pinn.data.conditional_case import STATE_VARS, terrain_at
from wrf_pinn.training import train_conditional as tc

HEIGHT_BANDS = (0.0, 100.0, 300.0, 700.0, float("inf"))  # metres above ground
SUMS = ("n", "t", "t2", "p", "p2", "e", "e2")


def score_cases(model, paths, cfg, read, device, readers: int = 8) -> dict:
    model.eval()
    fns = tc.step_functions(model)
    scale = torch.tensor([getattr(cfg.scaling, v).scale for v in STATE_VARS], device=device)
    offset = torch.tensor([getattr(cfg.scaling, v).offset for v in STATE_VARS], device=device)
    bands = torch.tensor(HEIGHT_BANDS[1:-1], device=device)
    sums = torch.zeros(len(SUMS), len(HEIGHT_BANDS) - 1, len(STATE_VARS), dtype=torch.float64, device=device)
    spread, losses, raw = [], {}, {}
    torch.manual_seed(0)  # same PDE points for every model

    with ThreadPoolExecutor(readers) as pool:
        pending = deque(pool.submit(read, p) for p in paths[:readers])  # bounded read-ahead
        for i in range(len(paths)):
            case = pending.popleft().result()
            if i + readers < len(paths):
                pending.append(pool.submit(read, paths[i + readers]))
            b = tc.stack_cases([case.as_torch(device="cpu")], cfg).to(device)

            case_raw = {}
            for name, value in tc._batch_losses(model, b, cfg, fns=fns, raw_mse=case_raw).items():
                losses[name] = losses.get(name, 0.0) + float(value.detach()) / len(paths)
            for name, value in case_raw.items():
                raw[name] = raw.get(name, 0.0) + float(value) / len(paths)

            with torch.no_grad():
                pred = model(b.interior, model.encode(b.z), b.interior_sizes) * scale + offset
                target = b.targets * scale + offset
                mask = b.target_mask.bool()
                band = torch.bucketize(_height_above_ground(b, cfg), bands)
                _accumulate(sums, pred, target, mask, band)
                spread.append(_spread_ratio(pred, target, mask))

    return _summary(sums, torch.stack(spread).nanmean(0), losses, raw)


def _height_above_ground(b, cfg) -> torch.Tensor:
    off, scale = tc._coordinate_affine(cfg.scaling, b.interior)
    physical = b.interior * scale + off
    h = terrain_at(physical[None, :, 0], physical[None, :, 1], b.terrain_x, b.terrain_y, b.terrain_h)[0][0]
    return physical[:, 2] - h


def _accumulate(sums, pred, target, mask, band):
    m = mask.double()
    t, p = target.double() * m, pred.double() * m
    e = p - t
    for row, values in enumerate((m, t, t * t, p, p * p, e, e * e)):
        sums[row].index_add_(0, band, values)


def _spread_ratio(pred, target, mask) -> torch.Tensor:
    """Model over simulation standard deviation inside one case, per variable."""
    m = mask.float()
    n = m.sum(0).clamp_min(2)
    std = lambda x: (((x - (x * m).sum(0) / n) * m) ** 2).sum(0).div(n - 1).sqrt()
    return std(pred) / std(target)


def _summary(sums, spread, losses, raw) -> dict:
    def metrics(s):
        n, t, t2, p, p2, e, e2 = s
        sst = t2 - t * t / n
        return {"r2": 1 - e2 / sst, "rmse": (e2 / n).sqrt(), "bias": e / n,
                "spread_pooled": ((p2 - p * p / n) / sst).sqrt()}

    out = {"variables": list(STATE_VARS), "height_bands_m": list(HEIGHT_BANDS),
           "all": {k: v.tolist() for k, v in metrics(sums.sum(1)).items()},
           "by_height": [{k: v.tolist() for k, v in metrics(sums[:, i]).items()}
                         for i in range(sums.shape[1])],
           "points_by_height": sums[0, :, 0].tolist(),
           "spread_within_case": spread.tolist(),
           "losses": losses, "raw_residual_mse": raw}
    return out
