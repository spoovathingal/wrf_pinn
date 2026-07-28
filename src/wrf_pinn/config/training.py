"""Training-loop configuration for WRF PINN experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


OptimizerName = Literal["adam", "adamw", "lbfgs"]
SchedulerName = Literal["none", "cosine", "step", "plateau"]


@dataclass(frozen=True)
class OptimizerConfig:
    """Optimizer settings for PINN training."""

    name: OptimizerName = "adam"
    learning_rate: float = 1.0e-3
    weight_decay: float = 0.0

    def __post_init__(self) -> None:
        if self.learning_rate <= 0.0:
            raise ValueError(
                f"learning_rate must be positive; got {self.learning_rate}."
            )

        if self.weight_decay < 0.0:
            raise ValueError(f"weight_decay must be nonnegative; got {self.weight_decay}.")


@dataclass(frozen=True)
class SchedulerConfig:
    """Learning-rate scheduler settings."""

    name: SchedulerName = "none"
    step_size: int = 1000
    gamma: float = 0.5
    min_learning_rate: float = 1.0e-6

    def __post_init__(self) -> None:
        if self.step_size < 1:
            raise ValueError(f"step_size must be positive; got {self.step_size}.")

        if self.gamma <= 0.0:
            raise ValueError(f"gamma must be positive; got {self.gamma}.")

        if self.min_learning_rate <= 0.0:
            raise ValueError(
                f"min_learning_rate must be positive; got {self.min_learning_rate}."
            )


@dataclass(frozen=True)
class CheckpointConfig:
    """Checkpoint output settings."""

    directory: str = "outputs/checkpoints"
    save_every: int = 1000
    keep_best: bool = True
    keep_last: bool = True

    def __post_init__(self) -> None:
        if not self.directory:
            raise ValueError("checkpoint directory cannot be empty.")

        if self.save_every < 1:
            raise ValueError(f"save_every must be positive; got {self.save_every}.")


@dataclass(frozen=True)
class TrainingConfig:
    """Top-level training configuration."""

    epochs: int = 10000
    log_every: int = 100
    validation_every: int = 500
    gradient_clip_norm: float | None = None
    device: str = "auto"
    optimizer: OptimizerConfig = OptimizerConfig()
    scheduler: SchedulerConfig = SchedulerConfig()
    checkpoints: CheckpointConfig = CheckpointConfig()

    def __post_init__(self) -> None:
        if self.epochs < 1:
            raise ValueError(f"epochs must be positive; got {self.epochs}.")

        if self.log_every < 1:
            raise ValueError(f"log_every must be positive; got {self.log_every}.")

        if self.validation_every < 1:
            raise ValueError(
                f"validation_every must be positive; got {self.validation_every}."
            )

        if self.gradient_clip_norm is not None and self.gradient_clip_norm <= 0.0:
            raise ValueError(
                "gradient_clip_norm must be positive when provided; "
                f"got {self.gradient_clip_norm}."
            )


DEFAULT_TRAINING = TrainingConfig()
