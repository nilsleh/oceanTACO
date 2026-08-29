"""Ragged Argo point rendering with explicit vertical semantics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..geobox import GeoBox, TimeRange, _utc_datetime, utc_isoformat


def _point_mask(values: np.ndarray, box: GeoBox) -> np.ndarray:
    lon = np.where(values == 180.0, -180.0, ((values + 180.0) % 360.0) - 180.0)
    if box.wraps_antimeridian:
        return (lon >= box.lon_min) | (lon <= box.lon_max)
    return (lon >= box.lon_min) & (lon <= box.lon_max)


@dataclass(frozen=True, slots=True)
class Points:
    """Render Argo points; default to the shallowest usable level per profile.

    ``variable`` exposes the released TEMP/PSAL values (and their recorded
    error fields) without inventing a second, misleadingly independent Argo
    source token.
    """

    pres_range: tuple[float, float] | None = None
    variable: str = "TEMP"

    def __post_init__(self) -> None:
        if self.pres_range is not None:
            low, high = self.pres_range
            if not np.isfinite(low) or not np.isfinite(high) or low > high:
                raise ValueError("pres_range must be an ordered finite (low, high) tuple.")
        if self.variable not in {"TEMP", "PSAL", "TEMP_ERROR", "PSAL_ERROR"}:
            raise ValueError("Points.variable must be TEMP, PSAL, TEMP_ERROR, or PSAL_ERROR.")

    def render(self, dataset, box: GeoBox, *, time: TimeRange, variable: str | None = None, ocean_mask=None, **_: Any) -> dict[str, Any]:
        """Filter Argo records and preserve the selected level's true pressure."""
        variable = self.variable if variable is None else variable
        for name in (variable, "lat", "lon", "time", "PRES", "PLATFORM_NUMBER", "CYCLE_NUMBER"):
            if name not in dataset:
                raise ValueError(f"Argo asset lacks required field {name!r}.")
        data = np.asarray(dataset[variable].values)
        lat = np.asarray(dataset["lat"].values)
        lon = np.where(
            np.asarray(dataset["lon"].values) == 180.0,
            -180.0,
            ((np.asarray(dataset["lon"].values) + 180.0) % 360.0) - 180.0,
        )
        pres = np.asarray(dataset["PRES"].values)
        raw_time = np.asarray(dataset["time"].values)
        times = np.asarray([_utc_datetime(str(value)) for value in raw_time])
        valid = np.isfinite(data) & np.isfinite(lat) & np.isfinite(lon) & np.isfinite(pres)
        valid &= (lat >= box.lat_min) & (lat <= box.lat_max) & _point_mask(lon, box)
        valid &= (times >= time.start) & (times <= time.end)
        point_ocean = None
        point_in_mask_domain = None
        if ocean_mask is not None:
            point_ocean, point_in_mask_domain = ocean_mask.nearest_points_with_domain(lat, lon)
        if self.pres_range is not None:
            valid &= (pres >= self.pres_range[0]) & (pres <= self.pres_range[1])
        platform = np.asarray(dataset["PLATFORM_NUMBER"].values).astype(str)
        cycle = np.asarray(dataset["CYCLE_NUMBER"].values).astype(str)
        indices = np.flatnonzero(valid)
        if self.pres_range is None:
            chosen: dict[str, int] = {}
            for index in indices:
                profile_id = f"{platform[index]}:{cycle[index]}"
                if profile_id not in chosen or pres[index] < pres[chosen[profile_id]]:
                    chosen[profile_id] = index
            indices = np.asarray(sorted(chosen.values()), dtype=int)
        output = {
            "data": np.asarray(data[indices], dtype=np.float32),
            "source_valid": np.ones(indices.size, dtype=bool),
            "support_mask": np.ones(indices.size, dtype=bool),
            "valid_mask": np.ones(indices.size, dtype=bool),
            "lat": np.asarray(lat[indices], dtype=np.float32),
            "lon": np.asarray(lon[indices], dtype=np.float32),
            "pres": np.asarray(pres[indices], dtype=np.float32),
            "time": [utc_isoformat(times[index]) for index in indices],
            "profile_id": np.asarray([f"{platform[index]}:{cycle[index]}" for index in indices], dtype=str),
        }
        if "DIRECTION" in dataset:
            output["direction"] = np.asarray(dataset["DIRECTION"].values)[indices].astype(str)
        if point_ocean is not None:
            output["ocean_mask"] = point_ocean[indices]
            output["in_mask_domain"] = point_in_mask_domain[indices]
        return output

    def empty(self) -> dict[str, Any]:
        """Return a valid zero-point source."""
        return {
            "data": np.empty((0,), dtype=np.float32),
            "source_valid": np.empty((0,), dtype=bool),
            "support_mask": np.empty((0,), dtype=bool),
            "valid_mask": np.empty((0,), dtype=bool),
            "lat": np.empty((0,), dtype=np.float32),
            "lon": np.empty((0,), dtype=np.float32),
            "pres": np.empty((0,), dtype=np.float32),
            "time": [],
            "profile_id": np.empty((0,), dtype=str),
        }
