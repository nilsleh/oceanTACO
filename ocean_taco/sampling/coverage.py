"""Offline native-grid coverage evidence for published QuerySets."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

import numpy as np

from ..geobox import PatchSize
from ..registry import get_modality
from ..render import canonicalise_dense, crop_dense
from .ocean_mask import OceanMaskArtifact


@dataclass(frozen=True, slots=True)
class DenseCoverage:
    """Native per-date coverage for one sparse grid.

    ``None`` values represent an unavailable/unmeasurable asset closure; a
    tuple of zeroes represents a successfully measured empty swath.
    """

    valid_cells: int | None
    valid_ocean_cells: int | None
    n_obs_sum: int | None = None

    def __post_init__(self) -> None:
        values = (self.valid_cells, self.valid_ocean_cells, self.n_obs_sum)
        if any(value is None for value in values[:2]):
            if (
                self.valid_cells is not None
                or self.valid_ocean_cells is not None
                or self.n_obs_sum is not None
            ):
                raise ValueError(
                    "An unavailable dense coverage tuple must contain only null metrics."
                )
            return
        if any(value is not None and value < 0 for value in values):
            raise ValueError("Coverage counts must be non-negative.")


def unavailable_dense_coverage() -> DenseCoverage:
    """Return the explicit null tuple for an unavailable source closure."""
    return DenseCoverage(None, None, None)


def native_footprint_counts(
    dataset,
    *,
    token: str,
    patch_size: PatchSize,
    centre_lon: float,
    centre_lat: float,
    ocean_mask: OceanMaskArtifact,
) -> tuple[int, int]:
    """Measure date-invariant native-grid and static-ocean denominators."""
    source = get_modality(token)
    dense = canonicalise_dense(dataset, source, fallback_time=datetime(1970, 1, 1))
    crop = crop_dense(dense, patch_size.footprint(centre_lon, centre_lat))
    lats, lons = np.asarray(crop["lat"].values), np.asarray(crop["lon"].values)
    ocean, in_domain = ocean_mask.nearest_with_domain(lats, lons)
    if not in_domain.all():
        raise ValueError(
            "Coverage denominator footprint leaves the frozen ocean-mask domain."
        )
    return int(lats.size * lons.size), int(ocean.sum())


def measure_dense_coverage(
    dataset,
    *,
    token: str,
    patch_size: PatchSize,
    centre_lon: float,
    centre_lat: float,
    ocean_mask: OceanMaskArtifact,
    n_obs_variable: str | None = None,
) -> DenseCoverage:
    """Measure native decoded validity without invoking a renderer.

    This counts finite decoded source cells.  Static geography supplies only
    the separate intersection count and never clears a valid measurement.
    """
    source = get_modality(token)
    dense = canonicalise_dense(dataset, source, fallback_time=datetime(1970, 1, 1))
    crop = crop_dense(dense, patch_size.footprint(centre_lon, centre_lat))
    source_valid = np.isfinite(np.asarray(crop.values))
    # Each coverage row describes one catalog date.  A malformed multi-time
    # source is still measured deterministically by treating a cell as present
    # if any decoded value for that asset date is finite.
    source_valid_2d = np.any(source_valid, axis=0)
    ocean, in_domain = ocean_mask.nearest_with_domain(
        np.asarray(crop["lat"].values), np.asarray(crop["lon"].values)
    )
    if not in_domain.all():
        raise ValueError(
            "Coverage measurement footprint leaves the frozen ocean-mask domain."
        )
    valid_cells = int(source_valid_2d.sum())
    valid_ocean_cells = int((source_valid_2d & ocean).sum())
    if n_obs_variable is None:
        return DenseCoverage(valid_cells, valid_ocean_cells, None)
    if n_obs_variable not in dataset:
        raise ValueError(
            f"{token} coverage requires product-supplied {n_obs_variable!r}."
        )
    n_obs = dataset[n_obs_variable]
    for dimension in tuple(n_obs.dims):
        if dimension not in {"time", "lat", "lon"}:
            if n_obs.sizes[dimension] != 1:
                raise ValueError(
                    f"{token} {n_obs_variable!r} has unsupported dimension {dimension!r}."
                )
            n_obs = n_obs.isel({dimension: 0}, drop=True)
    if "time" in n_obs.dims:
        n_obs = n_obs.isel(time=0, drop=True)
    n_obs = n_obs.sortby("lat")
    n_obs = n_obs.assign_coords(
        lon=np.where(
            np.asarray(n_obs["lon"].values) == 180.0,
            -180.0,
            ((np.asarray(n_obs["lon"].values) + 180.0) % 360.0) - 180.0,
        )
    ).sortby("lon")
    n_obs_crop = crop_dense(n_obs, patch_size.footprint(centre_lon, centre_lat))
    values = np.asarray(n_obs_crop.values)
    return DenseCoverage(valid_cells, valid_ocean_cells, int(np.nansum(values)))


def measure_argo_profile_count(
    dataset,
    *,
    patch_size: PatchSize,
    centre_lon: float,
    centre_lat: float,
    date: datetime | str,
) -> int:
    """Count unique released Argo profiles in one footprint/day without QC filtering."""
    from ..geobox import _utc_datetime
    from ..render.points import _point_mask

    for field in ("lat", "lon", "time", "PLATFORM_NUMBER", "CYCLE_NUMBER"):
        if field not in dataset:
            raise ValueError(f"Argo coverage asset lacks required field {field!r}.")
    target = _utc_datetime(date)
    end = target + timedelta(days=1)
    footprint = patch_size.footprint(centre_lon, centre_lat)
    lat = np.asarray(dataset["lat"].values)
    lon = np.where(
        np.asarray(dataset["lon"].values) == 180.0,
        -180.0,
        ((np.asarray(dataset["lon"].values) + 180.0) % 360.0) - 180.0,
    )
    from ..geobox import _utc_datetime as parse_time

    times = np.asarray(
        [parse_time(str(value)) for value in np.asarray(dataset["time"].values)]
    )
    selected = (
        np.isfinite(lat)
        & np.isfinite(lon)
        & (lat >= footprint.lat_min)
        & (lat <= footprint.lat_max)
        & _point_mask(lon, footprint)
        & (times >= target)
        & (times < end)
    )
    platform = np.asarray(dataset["PLATFORM_NUMBER"].values).astype(str)
    cycle = np.asarray(dataset["CYCLE_NUMBER"].values).astype(str)
    return len(
        {f"{platform[index]}:{cycle[index]}" for index in np.flatnonzero(selected)}
    )


def build_coverage_table(
    positions: Sequence[Mapping[str, Any]],
    dates: Sequence[str],
    *,
    measure: Callable[
        [Mapping[str, Any], int, str], Mapping[str, DenseCoverage | int | None]
    ],
) -> tuple[dict[str, int | None], ...]:
    """Measure every pair, retaining unavailable, zero, and finite cases.

    ``measure`` is an offline builder callback.  It receives every published
    position/date pair; it must never be used from a rendering worker.
    """
    rows: list[dict[str, int | None]] = []
    for position in positions:
        for date_index, date in enumerate(dates):
            values = dict(measure(position, date_index, date))
            swot = values.get("swot", unavailable_dense_coverage())
            ssh = values.get("ssh", unavailable_dense_coverage())
            argo = values.get("argo", None)
            if not isinstance(swot, DenseCoverage) or not isinstance(
                ssh, DenseCoverage
            ):
                raise ValueError(
                    "Coverage builder must return DenseCoverage values for swot and ssh."
                )
            if argo is not None and (not isinstance(argo, int) or argo < 0):
                raise ValueError(
                    "Argo coverage must be a nullable non-negative profile count."
                )
            rows.append(
                {
                    "position_index": int(position["position_index"]),
                    "date_index": date_index,
                    "swot_valid_cells": swot.valid_cells,
                    "swot_valid_ocean_cells": swot.valid_ocean_cells,
                    "swot_n_obs_sum": swot.n_obs_sum,
                    "ssh_valid_cells": ssh.valid_cells,
                    "ssh_valid_ocean_cells": ssh.valid_ocean_cells,
                    "argo_profile_count": argo,
                }
            )
    return tuple(rows)


__all__ = [
    "DenseCoverage",
    "build_coverage_table",
    "measure_argo_profile_count",
    "measure_dense_coverage",
    "native_footprint_counts",
    "unavailable_dense_coverage",
]
