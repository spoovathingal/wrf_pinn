"""Step-time statistics and run-to-run equivalence for training benchmarks.

A benchmark run is a JSON record (written by ``scripts/bench_conditional.py``)
holding the per-step losses and per-step wall times of a short fixed-seed run.
This module summarizes one record and compares two: the losses must agree
within a stated tolerance (pass or fail), and the speed difference is reported
with a bootstrap confidence interval so a small gain is not mistaken for noise.

Depends only on the standard library and NumPy, like ``results_writer``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

#: loss series compared between runs, in record order.
LOSS_KEYS: tuple[str, ...] = ("total", "data", "initial", "flow_boundary", "pde", "surface")

#: record fields that must match for two runs to be comparable.
FINGERPRINT_KEYS: tuple[str, ...] = ("case_list_sha256", "seed", "batch_cases", "world_size",
                                     "hidden_layers", "hidden_width")


def bootstrap_median_ci(values: Sequence[float], n_boot: int = 2000, level: float = 0.95,
                        seed: int = 0) -> tuple[float, float, float]:
    """Median of ``values`` and its bootstrap ``level`` confidence interval."""
    x = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    medians = np.median(rng.choice(x, size=(n_boot, x.size), replace=True), axis=1)
    lo, hi = np.quantile(medians, [(1 - level) / 2, (1 + level) / 2])
    return float(np.median(x)), float(lo), float(hi)


def speedup_ci(reference: Sequence[float], candidate: Sequence[float], n_boot: int = 2000,
               level: float = 0.95, seed: int = 0) -> tuple[float, float, float]:
    """Ratio of median step times (reference / candidate) with a bootstrap CI.

    Each run is resampled independently; > 1 means the candidate is faster."""
    a, b = np.asarray(reference, dtype=float), np.asarray(candidate, dtype=float)
    rng = np.random.default_rng(seed)
    ma = np.median(rng.choice(a, size=(n_boot, a.size), replace=True), axis=1)
    mb = np.median(rng.choice(b, size=(n_boot, b.size), replace=True), axis=1)
    lo, hi = np.quantile(ma / mb, [(1 - level) / 2, (1 + level) / 2])
    return float(np.median(a) / np.median(b)), float(lo), float(hi)


def summarize_steps(step_seconds: Sequence[float]) -> dict[str, float]:
    """Median, bootstrap CI, IQR and mean of the measured step times."""
    x = np.asarray(step_seconds, dtype=float)
    med, lo, hi = bootstrap_median_ci(x)
    q1, q3 = np.quantile(x, [0.25, 0.75])
    return {"n": int(x.size), "median": med, "median_ci95": [lo, hi],
            "iqr": [float(q1), float(q3)], "mean": float(x.mean()), "max": float(x.max())}


def summarize_utilization(samples: Sequence[float], idle_below: float = 10.0) -> dict[str, float]:
    """Mean and median GPU utilization (%) and the fraction of samples near idle."""
    x = np.asarray(samples, dtype=float)
    if x.size == 0:
        return {"n": 0}
    return {"n": int(x.size), "mean": float(x.mean()), "median": float(np.median(x)),
            "frac_idle": float((x < idle_below).mean()), "idle_below_pct": idle_below}


@dataclass
class Comparison:
    """Outcome of comparing a candidate run against a reference run."""

    passed: bool
    rtol: float
    final_rtol: "float | None"
    steps_compared: int
    max_rel_diff: dict[str, float]
    worst_step: dict[str, int]
    final_rel_diff: float
    speedup: tuple[float, float, float]
    problems: list[str] = field(default_factory=list)


def _rel_diff(a: np.ndarray, b: np.ndarray, floor: float) -> np.ndarray:
    return np.abs(a - b) / np.maximum(np.abs(a), floor)


def compare_runs(reference: dict, candidate: dict, rtol: float = 1e-4,
                 final_rtol: "float | None" = None, floor: float = 1e-12) -> Comparison:
    """Compare two benchmark records.

    Passes when every loss series agrees within ``rtol`` (relative, per step) over
    the steps both runs recorded and, if ``final_rtol`` is given, the final total
    loss agrees within it. Runs with different fingerprints never pass."""
    problems = [f"{k} differs: {reference['meta'].get(k)!r} vs {candidate['meta'].get(k)!r}"
                for k in FINGERPRINT_KEYS if reference["meta"].get(k) != candidate["meta"].get(k)]
    n = min(len(reference["losses"]["total"]), len(candidate["losses"]["total"]))
    max_diff, worst = {}, {}
    for key in LOSS_KEYS:
        a = np.asarray(reference["losses"][key][:n], dtype=float)
        b = np.asarray(candidate["losses"][key][:n], dtype=float)
        if not (np.isfinite(a).all() and np.isfinite(b).all()):
            problems.append(f"non-finite {key} loss")
        d = _rel_diff(a, b, floor)
        max_diff[key] = float(np.nanmax(d))
        worst[key] = int(np.nanargmax(d)) + 1
    final = float(_rel_diff(np.asarray(reference["losses"]["total"][n - 1:n]),
                            np.asarray(candidate["losses"]["total"][n - 1:n]), floor)[0])
    tol = final_rtol if final_rtol is not None else rtol
    bad = [k for k, v in max_diff.items() if v > rtol]
    if bad:
        problems.append(f"per-step rel diff above {rtol:g} in: {', '.join(bad)}")
    if final > tol:
        problems.append(f"final total loss rel diff {final:.3g} above {tol:g}")
    speed = speedup_ci(reference["step_seconds"], candidate["step_seconds"])
    return Comparison(passed=not problems, rtol=rtol, final_rtol=final_rtol, steps_compared=n,
                      max_rel_diff=max_diff, worst_step=worst, final_rel_diff=final,
                      speedup=speed, problems=problems)
