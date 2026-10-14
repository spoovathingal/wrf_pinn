"""Benchmark a short fixed-seed conditional-PINN run and write a JSON record.

Builds the model and scaling exactly as ``drivers/train_conditional_lcc.py`` does,
trains on a fixed case list for ``--warmup + --steps`` steps (plus ``1 + --trace-steps`` when tracing), and
records per-step losses, per-step wall times, GPU utilization and memory, plus the
environment (git commit, torch/CUDA versions, TF32 settings). The trace steps run
under ``torch.profiler`` after the timed window and are excluded from the timing
statistics, since profiling slows them down.

Compare two records with ``scripts/check_equivalence.py``. Launch under
``torchrun --nproc_per_node=N`` exactly like the training driver.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import logging
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import torch
import torch.distributed as dist

from wrf_pinn.config.physics import DEFAULT_PHYSICS
from wrf_pinn.config.scaling import ResidualScalingConfig, VariableScale
from wrf_pinn.data.conditional_case import read_conditional_case
from wrf_pinn.training.benchmark import summarize_steps, summarize_utilization
from wrf_pinn.models.conditional import ConditionalModel, ConditionalModelConfig
from wrf_pinn.training.train_conditional import ConditionalTrainConfig, train_conditional

log = logging.getLogger("bench_conditional")

#: multiplier from the data's q_v units to kg/kg, for the physics.
Q_V_UNITS = {"kg/kg": 1.0, "g/kg": 1e-3}


def _scaling_from_recipe(recipe: dict) -> ResidualScalingConfig:
    """Affine residual scaling from the recipe (same as the training driver)."""
    coord = dict(zip(recipe["coord_names"], zip(recipe["coord_offset"], recipe["coord_scale"])))
    state = dict(zip(recipe["state_vars"], zip(recipe["state_offset"], recipe["state_scale"])))
    return ResidualScalingConfig(**{
        name: VariableScale(offset=off, scale=(scale or 1.0))
        for name, (off, scale) in {**coord, **state}.items()
    })


def _read_case_list(path: Path, cases_dir: Path) -> list[str]:
    """One case path per line; relative paths resolve against ``cases_dir/train``."""
    lines = [ln.strip() for ln in path.read_text().splitlines()]
    return [ln if os.path.isabs(ln) else str(cases_dir / "train" / ln)
            for ln in lines if ln and not ln.startswith("#")]


def _git_state() -> dict:
    root = Path(__file__).resolve().parents[1]
    run = lambda *a: subprocess.run(["git", "-C", str(root), *a], capture_output=True,
                                    text=True).stdout.strip()
    return {"commit": run("rev-parse", "HEAD"), "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(run("status", "--porcelain", "--untracked-files=no"))}


def _precision_state() -> dict:
    """TF32 and matmul-precision settings in effect, including NVIDIA container overrides."""
    return {
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "env": {k: os.environ.get(k) for k in ("NVIDIA_TF32_OVERRIDE",
                                                "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE")},
    }


class UtilizationSampler:
    """Sample every visible GPU's utilization (%) on a thread, every ``period`` s.

    NVML reports the fraction of its last sample window with a kernel running, so
    readings are coarse; the sampler only runs during the timed steps."""

    def __init__(self, period: float = 0.2) -> None:
        self.period = period
        self.samples: dict[int, list[float]] = {i: [] for i in range(torch.cuda.device_count())}
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self.source = "torch.cuda.utilization"

    def _read(self) -> list[float]:
        try:
            return [float(torch.cuda.utilization(i)) for i in self.samples]
        except Exception:  # no pynvml: fall back to nvidia-smi
            self.source = "nvidia-smi"
            out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu",
                                  "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True).stdout.split()
            return [float(v) for v in out[:len(self.samples)]]

    def _run(self) -> None:
        while not self._stop.wait(self.period):
            for i, v in zip(self.samples, self._read()):
                self.samples[i].append(v)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cases_dir", type=Path, help="cases dir with train/ + metadata.json")
    ap.add_argument("--case-list", type=Path, required=True,
                    help="text file, one case path per line (see scripts/select_cases.py)")
    ap.add_argument("--q-v-units", required=True, choices=sorted(Q_V_UNITS))
    ap.add_argument("--out", type=Path, required=True, help="JSON record to write")
    ap.add_argument("--label", default="", help="free-text name for this run")
    ap.add_argument("--warmup", type=int, default=10, help="untimed steps first")
    ap.add_argument("--steps", type=int, default=50, help="timed steps")
    ap.add_argument("--trace-steps", type=int, default=3, help="profiled steps last (0 = none)")
    ap.add_argument("--trace-dir", type=Path, default=None, help="where to write the trace")
    ap.add_argument("--batch-cases", type=int, default=32, help="cases per step, all processes")
    ap.add_argument("--cases-per-pass", type=int, default=2, help="per process; 0 = whole batch")
    ap.add_argument("--readers", type=int, default=16)
    ap.add_argument("--lazy", action="store_true", help="read each batch from disk per step")
    ap.add_argument("--hidden-layers", type=int, default=10)
    ap.add_argument("--hidden-width", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    device = args.device
    if world > 1:
        local = int(os.environ["LOCAL_RANK"])
        if torch.cuda.is_available():
            torch.cuda.set_device(local)
            dist.init_process_group("nccl")
            device = f"cuda:{local}"
        else:  # CPU smoke tests
            dist.init_process_group("gloo")
            device = "cpu"
        if args.batch_cases % world:
            raise SystemExit(f"--batch-cases {args.batch_cases} must divide by {world} processes")

    logging.basicConfig(level=logging.INFO, stream=sys.stderr,
                        format=f"%(asctime)s rank{rank} %(name)s: %(message)s", datefmt="%H:%M:%S")

    recipe = json.loads((args.cases_dir / "metadata.json").read_text())["normalization"]
    scaling = _scaling_from_recipe(recipe)
    physics = dataclasses.replace(DEFAULT_PHYSICS, q_v_to_kg_per_kg=Q_V_UNITS[args.q_v_units])

    all_paths = _read_case_list(args.case_list, args.cases_dir)
    list_sha = hashlib.sha256("\n".join(Path(p).name for p in all_paths).encode()).hexdigest()
    paths = all_paths[rank::world]            # this process's shard, as in the driver
    first = read_conditional_case(paths[0])
    z_dim, points_per_case = first.z.shape[0], int(first.interior.shape[0])

    torch.manual_seed(args.seed)
    model = ConditionalModel(z_dim, ConditionalModelConfig(hidden_layers=args.hidden_layers,
                                                           hidden_width=args.hidden_width))
    # One untimed step between the timed window and the trace is the profiler's warmup.
    tail = 1 + args.trace_steps if args.trace_steps else 0
    total_steps = args.warmup + args.steps + tail
    cfg = ConditionalTrainConfig(
        epochs=total_steps, batch_cases=args.batch_cases // world, device=device,
        cases_per_pass=args.cases_per_pass, readers=args.readers, log_every=10,
        scaling=scaling, physics=physics, flux_offset=tuple(recipe["flux_offset"]),
        flux_scale=tuple(s or 1.0 for s in recipe["flux_scale"]), seed=args.seed,
    )
    cuda = torch.cuda.is_available() and str(device) != "cpu"

    stamps: list[float] = []
    sampler = UtilizationSampler() if (cuda and rank == 0) else None
    profiler = None
    if rank == 0 and args.trace_steps and cuda:
        trace_dir = args.trace_dir or args.out.with_suffix(".trace")
        trace_dir.mkdir(parents=True, exist_ok=True)
        profiler = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            # profiler step i is epoch i + 1, so this waits through the timed window
            schedule=torch.profiler.schedule(wait=args.warmup + args.steps, warmup=1,
                                             active=args.trace_steps),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(str(trace_dir)),
            with_stack=True)
        profiler.start()

    def on_step(epoch: int) -> None:
        stamps.append(time.perf_counter())
        if sampler is not None and epoch == args.warmup:
            sampler.start()
        if sampler is not None and epoch == args.warmup + args.steps:
            sampler.stop()
        if profiler is not None:
            profiler.step()

    if cuda:
        torch.cuda.reset_peak_memory_stats()
    stamps.append(time.perf_counter())
    if args.lazy:
        hist = train_conditional(model, config=cfg, case_paths=paths, on_step=on_step)
    else:
        hist = train_conditional(model, [read_conditional_case(p) for p in paths], cfg,
                                 on_step=on_step)
    if profiler is not None:
        profiler.stop()

    # stamps[0] is before step 1, so stamps[i] - stamps[i-1] is step i's wall time.
    step_seconds = [b - a for a, b in zip(stamps, stamps[1:])]
    timed = step_seconds[args.warmup:args.warmup + args.steps]
    peak = torch.cuda.max_memory_allocated() / 1e9 if cuda else 0.0
    peaks = [peak]
    if world > 1:
        peaks = [None] * world
        dist.all_gather_object(peaks, peak)

    if rank == 0:
        stats = summarize_steps(timed)
        record = {
            "meta": {
                "label": args.label, "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "host": os.uname().nodename, "git": _git_state(),
                "torch": torch.__version__, "cuda": torch.version.cuda,
                "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
                "precision": _precision_state(), "world_size": world,
                "batch_cases": args.batch_cases, "cases_per_pass": args.cases_per_pass,
                "lazy": args.lazy, "readers": args.readers, "seed": args.seed,
                "hidden_layers": args.hidden_layers, "hidden_width": args.hidden_width,
                "n_params": sum(p.numel() for p in model.parameters()),
                "case_list": str(args.case_list), "case_list_sha256": list_sha,
                "n_cases": len(all_paths), "points_per_case": points_per_case,
                "warmup": args.warmup, "steps": args.steps, "trace_steps": args.trace_steps,
            },
            "losses": {"total": hist.total, "data": hist.data, "initial": hist.initial,
                       "flow_boundary": hist.flow_boundary, "pde": hist.pde,
                       "surface": hist.surface},
            "step_seconds": timed,
            "all_step_seconds": step_seconds,
            "stats": {
                "step": stats,
                "points_per_s": points_per_case * args.batch_cases / stats["median"],
                "gpu_mem_peak_gb": peaks,
                "gpu_util": ({str(i): summarize_utilization(s) for i, s in sampler.samples.items()}
                             | {"source": sampler.source}) if sampler else {},
            },
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(record, indent=1))
        log.info("median %.3fs/step (95%% CI %.3f-%.3f), %.2fM points/s -> %s",
                 stats["median"], *stats["median_ci95"],
                 record["stats"]["points_per_s"] / 1e6, args.out)
        if profiler is not None:
            table = profiler.key_averages().table(sort_by="self_cpu_time_total", row_limit=40)
            args.out.with_suffix(".profile.txt").write_text(table)
    if world > 1:
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
