"""Deterministic constant-physical-spacing position grids."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from math import cos, radians
from typing import Any

import numpy as np

from ..geobox import KM_PER_DEGREE_LATITUDE, PatchSize
from ..manifest import position_id
from .ocean_mask import OceanMaskArtifact


def grid_id(*, spacing_km: float, ocean_mask_id: str) -> str:
    """Return the versioned identity of an equidistant ocean grid."""
    if spacing_km <= 0:
        raise ValueError("spacing_km must be positive.")
    return f"equidistant_ocean/v1:{spacing_km:.12g}:{ocean_mask_id}"


def _nearest_index(axis: np.ndarray, value: float) -> int:
    return int(np.abs(axis - value).argmin())


def _seam_distance_degrees(first: float, second: float) -> float:
    """Return shortest longitude separation in degrees."""
    return abs(((first - second + 180.0) % 360.0) - 180.0)


def _snap_tolerance_km(axis: np.ndarray, scale_km: float) -> float:
    """Return the largest separation error snapping to ``axis`` can introduce.

    Both endpoints of a step are moved to their nearest cell centre, each by at
    most half a cell, so a realised step can fall short of the requested one by
    up to a full cell.
    """
    if axis.size < 2:
        return 0.0
    return float(np.abs(np.diff(axis)).max()) * scale_km


def _row_longitudes(
    mask: OceanMaskArtifact, latitude: float, spacing_km: float
) -> np.ndarray:
    """Return snapped canonical longitudes separated by at least ``spacing``."""
    longitude_step = spacing_km / (
        KM_PER_DEGREE_LATITUDE * max(cos(radians(latitude)), 1e-12)
    )
    scale_km = KM_PER_DEGREE_LATITUDE * max(cos(radians(latitude)), 1e-12)
    tolerance_km = _snap_tolerance_km(mask.lon, scale_km)
    # Partition the circle exactly before snapping. ``arange`` leaves a
    # residual final interval whenever 360 is not a multiple of the requested
    # angular step; dropping its final point then adds that residual to the
    # preceding interval at the antimeridian. An exact partition makes the
    # nominal wrap-around interval identical to every interior interval.
    n_columns = max(1, int(np.floor(360.0 / longitude_step)))
    targets = -180.0 + np.arange(n_columns, dtype=np.float64) * (360.0 / n_columns)
    selected: list[float] = []
    for target in targets:
        candidate = float(mask.lon[_nearest_index(mask.lon, target)])
        if candidate in selected:
            continue
        # Snapping moves each endpoint by up to half a mask cell, so a step
        # requested at exactly ``spacing_km`` can land up to a full cell short
        # of it.  Rejecting on that rounding noise would drop the row entirely
        # and leave a gap of twice the spacing -- far worse than the
        # sub-cell shortfall it avoids.  Only a genuine shortfall, beyond what
        # snapping can explain, is rejected.
        if selected:
            nearest = min(
                _seam_distance_degrees(candidate, other) for other in selected
            )
            if nearest * scale_km + tolerance_km < spacing_km:
                continue
        selected.append(candidate)
    return np.asarray(sorted(selected), dtype=np.float64)


def _row_latitudes(mask: OceanMaskArtifact, spacing_km: float) -> np.ndarray:
    """Return snapped mask rows separated by at least the requested spacing."""
    requested_step = spacing_km / KM_PER_DEGREE_LATITUDE
    targets = np.arange(
        float(mask.lat[0]), float(mask.lat[-1]) + requested_step / 2.0, requested_step
    )
    tolerance_km = _snap_tolerance_km(mask.lat, KM_PER_DEGREE_LATITUDE)
    selected: list[float] = []
    for target in targets:
        candidate = float(mask.lat[_nearest_index(mask.lat, target)])
        # See ``_row_longitudes``: reject only a shortfall that snapping to the
        # mask cannot account for, never mere sub-cell rounding noise.
        if (
            selected
            and (candidate - selected[-1]) * KM_PER_DEGREE_LATITUDE + tolerance_km
            < spacing_km
        ):
            continue
        if not selected or candidate != selected[-1]:
            selected.append(candidate)
    return np.asarray(selected, dtype=np.float64)


def footprint_in_mask_domain(
    mask: OceanMaskArtifact, patch_size: PatchSize, lon: float, lat: float
) -> bool:
    """Whether a full patch has a geographic classification everywhere."""
    footprint = patch_size.footprint(lon, lat)
    return footprint.lat_min >= float(mask.lat[0]) and footprint.lat_max <= float(
        mask.lat[-1]
    )


def build_position_grid(
    mask: OceanMaskArtifact,
    *,
    patch_size: PatchSize,
    spacing_km: float,
    region_mask: Callable[[Any], int] | None = None,
    static_counts: Callable[[float, float], Mapping[str, int]] | None = None,
) -> tuple[dict[str, Any], ...]:
    """Build the canonical, footprint-trimmed grid from a frozen mask.

    This is deliberately a grid construction operation, not a draw.  Every
    retained centre is an exact mask-cell centre and an ocean cell; no latitude
    weights or rejection sampling participate.
    """
    identifier = grid_id(spacing_km=spacing_km, ocean_mask_id=mask.artifact_id)
    rows: list[dict[str, Any]] = []
    for latitude in _row_latitudes(mask, spacing_km):
        for longitude in _row_longitudes(mask, float(latitude), spacing_km):
            lat_index = _nearest_index(mask.lat, float(latitude))
            lon_index = _nearest_index(mask.lon, float(longitude))
            if not bool(mask.ocean_mask[lat_index, lon_index]):
                continue
            if not footprint_in_mask_domain(
                mask, patch_size, float(longitude), float(latitude)
            ):
                continue
            counts = dict(
                static_counts(float(longitude), float(latitude))
                if static_counts
                else {}
            )
            required = {
                "swot_footprint_cells",
                "swot_ocean_cells",
                "ssh_footprint_cells",
                "ssh_ocean_cells",
            }
            unknown = set(counts) - required
            if unknown:
                raise ValueError(
                    f"static_counts returned unsupported keys: {sorted(unknown)}"
                )
            counts = {key: int(counts.get(key, 0)) for key in required}
            if any(value < 0 for value in counts.values()):
                raise ValueError("static_counts must be non-negative.")
            if (
                counts["swot_ocean_cells"] > counts["swot_footprint_cells"]
                or counts["ssh_ocean_cells"] > counts["ssh_footprint_cells"]
            ):
                raise ValueError(
                    "static ocean cell count cannot exceed footprint count."
                )
            rows.append(
                {
                    "position_id": position_id(
                        grid_id=identifier,
                        centre_lon=float(longitude),
                        centre_lat=float(latitude),
                    ),
                    "centre_lon": float(longitude),
                    "centre_lat": float(latitude),
                    "region_mask": int(
                        region_mask(
                            patch_size.footprint(float(longitude), float(latitude))
                        )
                        if region_mask
                        else 0
                    ),
                    **counts,
                }
            )
    rows.sort(key=lambda row: (row["centre_lat"], row["centre_lon"]))
    return tuple({"position_index": index, **row} for index, row in enumerate(rows))


def latitude_band_counts(
    positions: tuple[Mapping[str, Any], ...],
    *,
    bands: tuple[float, ...] = (0.0, 15.0, 30.0, 45.0, 60.0),
) -> dict[str, int]:
    """Return absolute-latitude band counts for area-proportionality reports."""
    if len(bands) < 2 or tuple(sorted(bands)) != bands:
        raise ValueError("bands must be increasing with at least two edges.")
    report = {
        f"{bands[index]:g}-{bands[index + 1]:g}": 0 for index in range(len(bands) - 1)
    }
    for position in positions:
        value = abs(float(position["centre_lat"]))
        for index in range(len(bands) - 1):
            low, high = bands[index], bands[index + 1]
            if low <= value < high or (index == len(bands) - 2 and value == high):
                report[f"{low:g}-{high:g}"] += 1
                break
    if sum(report.values()) != len(positions):
        raise AssertionError(
            "latitude bands do not cover every position; extend the bands or "
            "correct the position grid."
        )
    return report


def patch_iou(
    first: Mapping[str, Any], second: Mapping[str, Any], patch_size: PatchSize
) -> float:
    """Return the deterministic lat/lon-footprint IoU for two positions.

    The released domain uses the same lat/lon trapezoid geometry as rendering.
    The area factor is evaluated at the mean latitude; it cancels in the IoU
    numerator/denominator, leaving a robust seam-aware rectangle calculation.
    """
    first_lon, first_lat = float(first["centre_lon"]), float(first["centre_lat"])
    second_lon, second_lat = float(second["centre_lon"]), float(second["centre_lat"])
    first_width, first_height = patch_size.to_degrees(first_lat)
    second_width, second_height = patch_size.to_degrees(second_lat)
    # Express both intervals around the first centre, so an antimeridian
    # footprint has exactly the same width as any other patch.
    offset = ((second_lon - first_lon + 180.0) % 360.0) - 180.0
    first_x = (-first_width / 2.0, first_width / 2.0)
    second_x = (offset - second_width / 2.0, offset + second_width / 2.0)
    first_y = (first_lat - first_height / 2.0, first_lat + first_height / 2.0)
    second_y = (second_lat - second_height / 2.0, second_lat + second_height / 2.0)
    width = max(0.0, min(first_x[1], second_x[1]) - max(first_x[0], second_x[0]))
    height = max(0.0, min(first_y[1], second_y[1]) - max(first_y[0], second_y[0]))
    overlap = width * height
    first_area, second_area = first_width * first_height, second_width * second_height
    return 0.0 if overlap == 0.0 else overlap / (first_area + second_area - overlap)


def maximum_pair_iou(
    positions: tuple[Mapping[str, Any], ...], patch_size: PatchSize
) -> float:
    """Return the maximum realised overlap for a validation fixture/report."""
    maximum = 0.0
    for left, first in enumerate(positions):
        for second in positions[left + 1 :]:
            maximum = max(maximum, patch_iou(first, second, patch_size))
    return maximum


def area_share_ratios(
    position_counts: Mapping[str, int], eligible_area: Mapping[str, float]
) -> dict[str, float]:
    """Compare realised grid density with independently measured eligible area.

    A returned value of one is exact proportionality.  The caller supplies
    basin/latitude-band area measured from the frozen binary mask, keeping this
    validation free of any hidden weighting scheme.
    """
    if set(position_counts) != set(eligible_area):
        raise ValueError(
            "position_counts and eligible_area must name identical bands or basins."
        )
    total_positions, total_area = (
        sum(position_counts.values()),
        sum(eligible_area.values()),
    )
    if (
        total_positions <= 0
        or total_area <= 0
        or any(value <= 0 for value in eligible_area.values())
    ):
        raise ValueError("Area-share validation needs positive counts and areas.")
    return {
        name: (count / total_positions) / (eligible_area[name] / total_area)
        for name, count in position_counts.items()
    }


__all__ = [
    "area_share_ratios",
    "build_position_grid",
    "footprint_in_mask_domain",
    "grid_id",
    "latitude_band_counts",
    "maximum_pair_iou",
    "patch_iou",
]
