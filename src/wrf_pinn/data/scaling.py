"""Readers for residual scaling metadata."""

from __future__ import annotations

from pathlib import Path

from wrf_pinn.config.scaling import ResidualScalingConfig, VariableScale


_REQUIRED_NAMES = ("x", "y", "z", "t", "u", "v", "w", "theta", "p_prime")


def read_residual_scaling_txt(path: str | Path) -> ResidualScalingConfig:
    """Read residual scaling metadata from a simple whitespace text file.

    Expected format:

    ``name offset scale``

    Blank lines and lines beginning with ``#`` are ignored. The scaling
    convention is:

    ``physical_value = offset + scale * normalized_value``

    Example
    -------
    ``x   0.0   1000.0``
    ``theta  290.0  30.0``
    ``p_prime -500.0 1000.0``
    """

    metadata_path = Path(path)
    if not metadata_path.exists():
        raise FileNotFoundError(f"Scaling metadata file not found: {metadata_path}.")

    scales: dict[str, VariableScale] = {}

    with metadata_path.open() as file:
        for line_number, raw_line in enumerate(file, start=1):
            line = raw_line.strip()

            if not line or line.startswith("#"):
                continue

            parts = line.split()
            if len(parts) != 3:
                raise ValueError(
                    f"Invalid scaling metadata line {line_number} in "
                    f"{metadata_path}. Expected: name offset scale."
                )

            name, offset_text, scale_text = parts

            if name not in _REQUIRED_NAMES:
                raise ValueError(
                    f"Invalid scaling variable {name!r} on line {line_number} "
                    f"in {metadata_path}. Expected one of {_REQUIRED_NAMES}."
                )

            if name in scales:
                raise ValueError(
                    f"Duplicate scaling variable {name!r} on line {line_number} "
                    f"in {metadata_path}."
                )

            try:
                offset = float(offset_text)
                scale = float(scale_text)
            except ValueError as exc:
                raise ValueError(
                    f"Invalid numeric scaling values on line {line_number} "
                    f"in {metadata_path}: {raw_line.rstrip()!r}."
                ) from exc

            scales[name] = VariableScale(offset=offset, scale=scale)

    missing = [name for name in _REQUIRED_NAMES if name not in scales]
    if missing:
        raise ValueError(
            f"Scaling metadata file {metadata_path} is missing variables: {missing}."
        )

    return ResidualScalingConfig(
        x=scales["x"],
        y=scales["y"],
        z=scales["z"],
        t=scales["t"],
        u=scales["u"],
        v=scales["v"],
        w=scales["w"],
        theta=scales["theta"],
        p_prime=scales["p_prime"],
    )
