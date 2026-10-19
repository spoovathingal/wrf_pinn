"""Benchmark statistics, the equivalence check, and the trainer's on_step hook."""

from __future__ import annotations

import copy

import numpy as np
import pytest

from wrf_pinn.training.benchmark import (
    LOSS_KEYS, bootstrap_median_ci, compare_runs, speedup_ci, summarize_utilization,
)


def _record(seed: int = 0, n: int = 30, step: float = 2.8) -> dict:
    rng = np.random.default_rng(seed)
    losses = {k: list(np.linspace(1.0, 0.5, n) + 0.01 * i) for i, k in enumerate(LOSS_KEYS)}
    meta = {"case_list_sha256": "abc", "seed": 0, "batch_cases": 32, "world_size": 2,
            "hidden_layers": 10, "hidden_width": 512, "git": {"commit": "0" * 40}}
    return {"meta": meta, "losses": losses,
            "step_seconds": list(step + 0.05 * rng.standard_normal(n))}


def test_identical_runs_pass():
    ref = _record()
    result = compare_runs(ref, copy.deepcopy(ref))
    assert result.passed
    assert max(result.max_rel_diff.values()) == 0.0


def test_small_drift_passes_and_large_drift_fails():
    ref = _record()
    near, far = copy.deepcopy(ref), copy.deepcopy(ref)
    near["losses"]["pde"][5] *= 1 + 5e-5
    far["losses"]["pde"][5] *= 1 + 5e-3
    assert compare_runs(ref, near, rtol=1e-4).passed
    bad = compare_runs(ref, far, rtol=1e-4)
    assert not bad.passed
    assert bad.worst_step["pde"] == 6
    # the looser lower-precision tolerance accepts the same drift
    assert compare_runs(ref, far, rtol=1e-2, final_rtol=0.02).passed


def test_final_rtol_checks_last_total():
    ref = _record()
    cand = copy.deepcopy(ref)
    cand["losses"]["total"][-1] *= 1.05
    assert not compare_runs(ref, cand, rtol=1e-1, final_rtol=0.02).passed


def test_mismatched_fingerprint_never_passes():
    ref = _record()
    cand = copy.deepcopy(ref)
    cand["meta"]["case_list_sha256"] = "other"
    result = compare_runs(ref, cand)
    assert not result.passed
    assert any("case_list_sha256" in p for p in result.problems)


def test_non_finite_loss_fails():
    ref = _record()
    cand = copy.deepcopy(ref)
    cand["losses"]["surface"][3] = float("nan")
    assert not compare_runs(ref, cand, rtol=1.0).passed


def test_speedup_ci_brackets_true_ratio():
    slow, fast = _record(seed=1, n=50, step=2.8), _record(seed=2, n=50, step=1.4)
    s, lo, hi = speedup_ci(slow["step_seconds"], fast["step_seconds"])
    assert lo <= 2.0 <= hi
    assert lo <= s <= hi


def test_bootstrap_median_ci_contains_median():
    med, lo, hi = bootstrap_median_ci([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    assert lo <= med <= hi


def test_utilization_idle_fraction():
    stats = summarize_utilization([0, 0, 90, 95, 100])
    assert stats["frac_idle"] == pytest.approx(0.4)
    assert summarize_utilization([]) == {"n": 0}


def test_on_step_called_once_per_step():
    torch = pytest.importorskip("torch")
    from test_train_conditional import LATENT_DIM, _DATA_ONLY, _make_case
    from wrf_pinn.models.conditional import ConditionalModel, ConditionalModelConfig
    from wrf_pinn.training.train_conditional import ConditionalTrainConfig, train_conditional

    cases = [_make_case(f"c{i}", i) for i in range(6)]
    torch.manual_seed(0)
    model = ConditionalModel(LATENT_DIM, ConditionalModelConfig(hidden_layers=2, hidden_width=16))
    cfg = ConditionalTrainConfig(epochs=7, batch_cases=4, device="cpu", log_every=100,
                                 **_DATA_ONLY)
    seen: list[int] = []
    hist = train_conditional(model, cases, cfg, on_step=seen.append)
    assert seen == list(range(1, 8))

    # the hook does not change training
    torch.manual_seed(0)
    model2 = ConditionalModel(LATENT_DIM, ConditionalModelConfig(hidden_layers=2, hidden_width=16))
    assert train_conditional(model2, cases, cfg).total == hist.total
