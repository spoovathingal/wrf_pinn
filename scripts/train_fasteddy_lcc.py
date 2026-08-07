"""Run reduced FastEddy training and local smoke checks."""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from wrf_pinn.config.conditions import ConditionSpec, ConditionsConfig
from wrf_pinn.config.domain import make_cartesian_wrf_domain
from wrf_pinn.config.flow_field_data import FlowFieldDataConfig
from wrf_pinn.config.model import ModelConfig
from wrf_pinn.config.physics import PhysicsConfig
from wrf_pinn.config.sampling import (
    CollocationSamplingConfig,
    SamplingConfig,
)
from wrf_pinn.config.training import OptimizerConfig, TrainingConfig
from wrf_pinn.data.flow_field import read_flow_field
from wrf_pinn.data.scaling import read_residual_scaling_txt
from wrf_pinn.models.mlp import MLP
from wrf_pinn.training.checkpoint import save_checkpoint
from wrf_pinn.training.losses import PDEResidualScales
from wrf_pinn.training.train_pinn import train_pinn


EXPECTED_COLUMNS = (
    "x", "y", "z", "t",
    "u", "v", "w", "theta", "p_prime",
)

FASTEDDY_PDE_SCALES = PDEResidualScales(
    mass=3.13536843744e-3,
    x_momentum=3.35707053782e-2,
    y_momentum=1.99362535127e-2,
    z_momentum=5.58677880249e-1,
    potential_temperature=9.61239013761e-1,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--scaling", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--collocation-points", type=int, default=2_048)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--data-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.steps < 1 or args.collocation_points < 1:
        raise ValueError("Steps and collocation points must be positive.")

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    data_path = Path(args.data)
    scaling_path = Path(args.scaling)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with data_path.open(newline="") as file:
        header = tuple(next(csv.reader(file)))

    if header != EXPECTED_COLUMNS:
        raise ValueError(
            f"Expected columns {EXPECTED_COLUMNS}; got {header}."
        )

    flow_data = read_flow_field(
        FlowFieldDataConfig(path=str(data_path))
    )

    for name, values in (
        ("coordinates", flow_data.coordinates),
        ("targets", flow_data.targets),
    ):
        if not np.isfinite(values).all():
            raise ValueError(f"{name} contain NaN or infinity.")

        if (values < 0.0).any() or (values > 1.0).any():
            raise ValueError(f"{name} are outside [0, 1].")

    scaling = read_residual_scaling_txt(scaling_path)

    model_config = ModelConfig()
    physics = PhysicsConfig()
    model = MLP(model_config)

    conditions = ConditionsConfig(
        pde=ConditionSpec(
            "pde",
            active=not args.data_only,
            weight=0.0 if args.data_only else 1.0,
        ),
    )

    domain = make_cartesian_wrf_domain(
        x_min=0.0,
        x_max=1.0,
        y_min=0.0,
        y_max=1.0,
        z_min=0.0,
        z_max=1.0,
        t_min=0.0,
        t_max=1.0,
    )

    sampling = SamplingConfig(
        collocation=CollocationSamplingConfig(
            n_points=args.collocation_points,
            method="latin_hypercube",
        ),
        seed=args.seed,
    )

    training = TrainingConfig(
        epochs=args.steps,
        log_every=args.log_every,
        device=args.device,
        optimizer=OptimizerConfig(
            name="adam",
            learning_rate=1.0e-3,
        ),
    )

    run_config = {
        "data_path": str(data_path.resolve()),
        "data_rows": flow_data.n_points,
        "scaling_path": str(scaling_path.resolve()),
        "variable_scaling": asdict(scaling),
        "data_only": args.data_only,
        "model": asdict(model_config),
        "physics": asdict(physics),
        "conditions": asdict(conditions),
        "sampling": asdict(sampling),
        "training": asdict(training),
        "pde_residual_scales": asdict(FASTEDDY_PDE_SCALES),
    }

    with (output_dir / "run_config.json").open("w") as file:
        json.dump(run_config, file, indent=2)

    print(f"device={args.device}")
    print(f"data_rows={flow_data.n_points}")

    if args.device == "cuda":
        print(f"gpu={torch.cuda.get_device_name(0)}")

    start = time.perf_counter()

    history = train_pinn(
        model,
        domain=domain,
        flow_field_data=flow_data,
        training=training,
        conditions=conditions,
        sampling=sampling,
        physics=physics,
        scaling=scaling,
        pde_residual_scales=FASTEDDY_PDE_SCALES,
    )

    if args.device == "cuda":
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start

    checkpoint_path = save_checkpoint(
        output_dir / "final_checkpoint.pt",
        model=model,
        history=history,
        metadata=run_config,
    )

    # Minimum checkpoint reload check.
    payload = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    reloaded_model = MLP(model_config)
    reloaded_model.load_state_dict(payload["model_state_dict"])
    print("checkpoint_reload=ok")

    timing = {
        "elapsed_seconds": elapsed,
        "seconds_per_step": elapsed / args.steps,
    }

    if args.device == "cuda":
        timing["cuda_peak_allocated_gb"] = (
            torch.cuda.max_memory_allocated() / 1024**3
        )
        timing["cuda_peak_reserved_gb"] = (
            torch.cuda.max_memory_reserved() / 1024**3
        )

    with (output_dir / "timing.json").open("w") as file:
        json.dump(timing, file, indent=2)

    print(f"elapsed_seconds={elapsed:.3f}")
    print(f"seconds_per_step={elapsed / args.steps:.6f}")

    if args.device == "cuda":
        print(
            "cuda_peak_allocated_gb="
            f"{timing['cuda_peak_allocated_gb']:.3f}"
        )
        print(
            "cuda_peak_reserved_gb="
            f"{timing['cuda_peak_reserved_gb']:.3f}"
        )


if __name__ == "__main__":
    main()