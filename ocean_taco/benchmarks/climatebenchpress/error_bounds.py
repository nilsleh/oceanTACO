"""Error-bound helpers for OceanTACO benchmark datasets."""

from __future__ import annotations

from pathlib import Path

from ocean_taco.benchmarks.climatebenchpress.config import DEFAULT_ERROR_MULTIPLIERS
from ocean_taco.benchmarks.climatebenchpress.utils import dump_json


def build_error_bounds(
    variable_name: str,
    strict_abs_error: float,
    multipliers: tuple[float, ...] = DEFAULT_ERROR_MULTIPLIERS,
) -> list[dict[str, dict[str, float | None]]]:
    """Return ClimateBenchPress-style error-bound tiers for one variable."""
    return [
        {
            variable_name: {
                "abs_error": strict_abs_error * multiplier,
                "rel_error": None,
            }
        }
        for multiplier in multipliers
    ]


def write_error_bounds_file(
    output_path: Path,
    variable_name: str,
    strict_abs_error: float,
    multipliers: tuple[float, ...] = DEFAULT_ERROR_MULTIPLIERS,
) -> list[dict[str, dict[str, float | None]]]:
    """Write one error_bounds.json file and return the payload."""
    payload = build_error_bounds(variable_name, strict_abs_error, multipliers)
    dump_json(output_path, payload)
    return payload
