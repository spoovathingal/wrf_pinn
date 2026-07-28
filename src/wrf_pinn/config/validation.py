"""Validation and acceptance-criteria configuration for WRF PINN runs."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class NumericalValidationConfig:
    """Numerical health checks during training and evaluation."""

    fail_on_nan: bool = True
    fail_on_inf: bool = True
    max_loss: float | None = None
    max_gradient_norm: float | None = None

    def __post_init__(self) -> None:
        if self.max_loss is not None and self.max_loss <= 0.0:
            raise ValueError(f"max_loss must be positive; got {self.max_loss}.")

        if self.max_gradient_norm is not None and self.max_gradient_norm <= 0.0:
            raise ValueError(
                "max_gradient_norm must be positive; "
                f"got {self.max_gradient_norm}."
            )


@dataclass(frozen=True)
class PhysicalValidationConfig:
    """Physical sanity checks for predicted atmospheric state."""

    enforce_positive_density: bool = True
    min_density: float = 0.0
    max_wind_speed: float | None = 150.0

    def __post_init__(self) -> None:
        if self.max_wind_speed is not None and self.max_wind_speed <= 0.0:
            raise ValueError(
                f"max_wind_speed must be positive; got {self.max_wind_speed}."
            )


@dataclass(frozen=True)
class ResidualValidationConfig:
    """Optional tolerances for PDE residual diagnostics."""

    max_mass_residual_rmse: float | None = None
    max_momentum_residual_rmse: float | None = None

    def __post_init__(self) -> None:
        if (
            self.max_mass_residual_rmse is not None
            and self.max_mass_residual_rmse <= 0.0
        ):
            raise ValueError(
                "max_mass_residual_rmse must be positive; "
                f"got {self.max_mass_residual_rmse}."
            )

        if (
            self.max_momentum_residual_rmse is not None
            and self.max_momentum_residual_rmse <= 0.0
        ):
            raise ValueError(
                "max_momentum_residual_rmse must be positive; "
                f"got {self.max_momentum_residual_rmse}."
            )


@dataclass(frozen=True)
class DataFitValidationConfig:
    """Optional acceptance thresholds for measured-data fit."""

    max_rmse_by_field: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        invalid = {
            name: threshold
            for name, threshold in self.max_rmse_by_field.items()
            if threshold <= 0.0
        }
        if invalid:
            raise ValueError(f"RMSE thresholds must be positive; got {invalid}.")


@dataclass(frozen=True)
class EarlyStoppingConfig:
    """Early-stopping settings for validation-monitored training."""

    enabled: bool = False
    patience: int = 1000
    min_delta: float = 1.0e-6
    monitor: str = "validation_loss"

    def __post_init__(self) -> None:
        if self.patience < 1:
            raise ValueError(f"patience must be positive; got {self.patience}.")

        if self.min_delta < 0.0:
            raise ValueError(f"min_delta must be nonnegative; got {self.min_delta}.")

        if not self.monitor:
            raise ValueError("monitor cannot be empty.")


@dataclass(frozen=True)
class ValidationConfig:
    """Top-level validation configuration."""

    numerical: NumericalValidationConfig = NumericalValidationConfig()
    physical: PhysicalValidationConfig = PhysicalValidationConfig()
    residual: ResidualValidationConfig = ResidualValidationConfig()
    data_fit: DataFitValidationConfig = DataFitValidationConfig()
    early_stopping: EarlyStoppingConfig = EarlyStoppingConfig()


DEFAULT_VALIDATION = ValidationConfig()
