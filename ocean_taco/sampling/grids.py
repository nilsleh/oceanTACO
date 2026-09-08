"""Deterministic constant-physical-spacing position grids."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from hashlib import sha256
from math import cos, radians
from typing import Any

import numpy as np

from ..geobox import KM_PER_DEGREE_LATITUDE, PatchSize
from ..queryset import position_id
from .ocean_mask import OceanMaskArtifact


def grid_id(*, spacing_km: float, ocean_mask_id: str) -> str:
    """Return the versioned identity of an equidistant ocean grid."""
    if spacing_km <= 0:
        raise ValueError("spacing_km must be positive.")
    return f"equidistant_ocean/v1:{spacing_km:.12g}:{ocean_mask_id}"


RANDOM_SAMPLER_METHOD = "stratified_best_candidate_ocean/v1"
POSITION_ELIGIBILITY = "ocean_centre_and_footprint_mask_domain/v1"
MAX_TRAINING_IOU = 0.20
STRATUM_LATITUDE_BINS = 18
STRATUM_LONGITUDE_BINS = 36
CANDIDATE_POOL_SIZE = 16


def random_grid_id(
    *, patch_size: PatchSize, seed: int, position_count: int, ocean_mask_id: str
) -> str:
    """Return the identity of one seeded random ocean-position sample."""
    if position_count < 1:
        raise ValueError("position_count must be positive.")
    return (
        f"{RANDOM_SAMPLER_METHOD}:{patch_size.value:.12g}{patch_size.unit}:"
        f"seed={seed}:count={position_count}:{ocean_mask_id}"
    )


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


def _longitude_spacing_bounds(
    mask: OceanMaskArtifact, latitude: float, spacing_km: float
) -> tuple[float, float]:
    """Return realised-step bounds for one circular, snapped longitude row.

    A global row has an integer number of intervals. Distributing 360 degrees
    exactly therefore adds a small, unavoidable partition adjustment above the
    requested step when floor(360 / requested_step) is not exact. The
    remaining error is ordinary mask-cell snapping.
    """
    scale_km = KM_PER_DEGREE_LATITUDE * max(cos(radians(latitude)), 1e-12)
    requested_degrees = spacing_km / scale_km
    n_columns = max(1, int(np.floor(360.0 / requested_degrees)))
    snap_tolerance_km = _snap_tolerance_km(mask.lon, scale_km)
    return (
        spacing_km - snap_tolerance_km,
        (360.0 / n_columns) * scale_km + snap_tolerance_km,
    )


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


def _eligible_random_centres(
    mask: OceanMaskArtifact, patch_size: PatchSize
) -> tuple[np.ndarray, np.ndarray]:
    """Return sorted mask-cell centres admissible to the random sampler.

    The released contract deliberately requires an ocean *centre* and a
    footprint inside the classified latitude domain. It does not require an
    all-ocean footprint: coastal context is a valid training example.
    """
    latitudes: list[float] = []
    longitudes: list[float] = []
    for lat_index, latitude in enumerate(mask.lat):
        latitude = float(latitude)
        if not footprint_in_mask_domain(mask, patch_size, 0.0, latitude):
            continue
        columns = np.flatnonzero(mask.ocean_mask[lat_index])
        latitudes.extend([latitude] * len(columns))
        longitudes.extend(float(mask.lon[column]) for column in columns)
    return (
        np.asarray(latitudes, dtype=np.float64),
        np.asarray(longitudes, dtype=np.float64),
    )


def coordinate_digest(latitudes: np.ndarray, longitudes: np.ndarray) -> str:
    """Return a stable byte digest for a latitude/longitude sequence."""
    digest = sha256()
    for values in (latitudes, longitudes):
        digest.update(np.ascontiguousarray(values, dtype=np.float64).tobytes())
    return digest.hexdigest()


def _stratum_ids(latitudes: np.ndarray, longitudes: np.ndarray) -> np.ndarray:
    """Assign centres to equal-area latitude/longitude strata."""
    sine_latitude = np.sin(np.radians(latitudes))
    lower, upper = float(sine_latitude.min()), float(sine_latitude.max())
    latitude_bin = np.minimum(
        ((sine_latitude - lower) / (upper - lower) * STRATUM_LATITUDE_BINS).astype(
            np.int64
        ),
        STRATUM_LATITUDE_BINS - 1,
    )
    longitude_bin = np.minimum(
        ((longitudes + 180.0) / 360.0 * STRATUM_LONGITUDE_BINS).astype(np.int64),
        STRATUM_LONGITUDE_BINS - 1,
    )
    return latitude_bin * STRATUM_LONGITUDE_BINS + longitude_bin


def _stratum_quotas(
    strata: np.ndarray, weights: np.ndarray, position_count: int
) -> np.ndarray:
    """Apportion an exact sample count by eligible equal-area ocean support."""
    stratum_count = STRATUM_LATITUDE_BINS * STRATUM_LONGITUDE_BINS
    mass = np.bincount(strata, weights=weights, minlength=stratum_count)
    ideal = position_count * mass / mass.sum()
    quotas = np.floor(ideal).astype(np.int64)
    remainder = position_count - int(quotas.sum())
    active = np.flatnonzero(mass > 0)
    order = active[np.lexsort((active, -(ideal[active] - quotas[active])))]
    quotas[order[:remainder]] += 1
    return quotas


def _overlap_exceeds(
    *,
    longitude: float,
    latitude: float,
    selected: tuple[float, float],
    patch_size: PatchSize,
    maximum_iou: float,
) -> bool:
    """Whether two patch footprints overlap beyond the configured IoU ceiling."""
    selected_longitude, selected_latitude = selected
    width, height = patch_size.to_degrees(latitude)
    selected_width, selected_height = patch_size.to_degrees(selected_latitude)
    longitude_offset = ((selected_longitude - longitude + 180.0) % 360.0) - 180.0
    longitude_overlap = max(
        0.0,
        min(width / 2.0, longitude_offset + selected_width / 2.0)
        - max(-width / 2.0, longitude_offset - selected_width / 2.0),
    )
    latitude_overlap = max(
        0.0,
        min(latitude + height / 2.0, selected_latitude + selected_height / 2.0)
        - max(latitude - height / 2.0, selected_latitude - selected_height / 2.0),
    )
    overlap = longitude_overlap * latitude_overlap
    if overlap == 0.0:
        return False
    return (
        overlap / (width * height + selected_width * selected_height - overlap)
        > maximum_iou
    )


def random_centre_sample(
    mask: OceanMaskArtifact, *, patch_size: PatchSize, seed: int, position_count: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Draw a stratified, best-candidate ocean-centre sample."""
    latitudes, longitudes = _eligible_random_centres(mask, patch_size)
    if not 0 < position_count <= len(latitudes):
        raise ValueError("position_count must lie in [1, eligible centre count].")
    weights = np.cos(np.radians(latitudes))
    strata = _stratum_ids(latitudes, longitudes)
    quotas = _stratum_quotas(strata, weights, position_count)
    generator = np.random.Generator(np.random.PCG64(seed))
    keys = -np.log1p(-generator.random(len(latitudes))) / weights
    stratum_count = STRATUM_LATITUDE_BINS * STRATUM_LONGITUDE_BINS
    stratum_orders = tuple(
        indices[np.argsort(keys[indices], kind="stable")]
        for indices in (np.flatnonzero(strata == value) for value in range(stratum_count))
    )
    cursors = np.zeros(stratum_count, dtype=np.int64)
    buffers: list[list[int]] = [[] for _ in range(stratum_count)]
    latitude_bucket = patch_size.to_degrees(0.0)[1]
    longitude_bucket = max(
        patch_size.to_degrees(float(latitude))[0]
        for latitude in (float(latitudes[0]), float(latitudes[-1]))
    )
    longitude_bucket_count = max(1, int(np.ceil(360.0 / longitude_bucket)))
    buckets: dict[tuple[int, int], list[int]] = {}
    selected_indices: list[int] = []
    selected_coordinates: list[tuple[float, float]] = []

    def bucket_key(latitude: float, longitude: float) -> tuple[int, int]:
        return (
            int(np.floor((latitude - float(latitudes[0])) / latitude_bucket)),
            int(np.floor((longitude + 180.0) / longitude_bucket))
            % longitude_bucket_count,
        )

    def candidate_iou(index: int) -> float:
        latitude, longitude = float(latitudes[index]), float(longitudes[index])
        latitude_key, longitude_key = bucket_key(latitude, longitude)
        candidate = {"centre_lon": longitude, "centre_lat": latitude}
        maximum = 0.0
        for nearby_latitude in range(latitude_key - 1, latitude_key + 2):
            for longitude_offset in (-1, 0, 1):
                for selected_index in buckets.get(
                    (
                        nearby_latitude,
                        (longitude_key + longitude_offset) % longitude_bucket_count,
                    ),
                    (),
                ):
                    selected_longitude, selected_latitude = selected_coordinates[
                        selected_index
                    ]
                    maximum = max(
                        maximum,
                        patch_iou(
                            candidate,
                            {
                                "centre_lon": selected_longitude,
                                "centre_lat": selected_latitude,
                            },
                            patch_size,
                        ),
                    )
        return maximum

    def accept(index: int) -> None:
        latitude, longitude = float(latitudes[index]), float(longitudes[index])
        key = bucket_key(latitude, longitude)
        selected_indices.append(index)
        selected_coordinates.append((longitude, latitude))
        buckets.setdefault(key, []).append(len(selected_coordinates) - 1)

    remaining = quotas.copy()
    allocation_weight = np.maximum(quotas, 1)
    order_lengths = np.asarray([len(order) for order in stratum_orders])
    while int(remaining.sum()):
        active = np.flatnonzero(remaining > 0)
        stratum = int(
            active[np.argmax(remaining[active] / allocation_weight[active])]
        )
        order = stratum_orders[stratum]
        while (
            len(buffers[stratum]) < CANDIDATE_POOL_SIZE
            and cursors[stratum] < len(order)
        ):
            stop = min(int(cursors[stratum]) + CANDIDATE_POOL_SIZE, len(order))
            buffers[stratum].extend(
                int(index) for index in order[int(cursors[stratum]) : stop]
            )
            cursors[stratum] = stop
        scored = [(candidate_iou(index), index) for index in buffers[stratum]]
        valid = [item for item in scored if item[0] <= MAX_TRAINING_IOU]
        if valid:
            _, index = min(valid, key=lambda item: (item[0], item[1]))
            buffers[stratum].remove(index)
            accept(index)
            remaining[stratum] -= 1
            continue
        buffers[stratum] = []
        if cursors[stratum] < len(order):
            continue
        deficit = int(remaining[stratum])
        remaining[stratum] = 0
        destinations = np.flatnonzero(cursors < order_lengths)
        if not len(destinations):
            raise ValueError(
                "The requested position count cannot satisfy the maximum training IoU "
                f"of {MAX_TRAINING_IOU}; accepted {len(selected_indices)} of {position_count}."
            )
        remaining[destinations[np.arange(deficit) % len(destinations)]] += 1
    selected = np.sort(np.asarray(selected_indices, dtype=np.int64))
    return latitudes, longitudes, latitudes[selected], longitudes[selected]


def random_position_sampling_metadata(
    mask: OceanMaskArtifact, *, patch_size: PatchSize, seed: int, position_count: int
) -> dict[str, Any]:
    """Describe one reproducible balanced, low-overlap position sample."""
    latitudes, longitudes, selected_latitudes, selected_longitudes = (
        random_centre_sample(
            mask, patch_size=patch_size, seed=seed, position_count=position_count
        )
    )
    return {
        "method": RANDOM_SAMPLER_METHOD,
        "seed": int(seed),
        "position_count": int(position_count),
        "area_weight": "cosine_latitude/v1",
        "stratification": {
            "latitude_equal_area_bins": STRATUM_LATITUDE_BINS,
            "longitude_bins": STRATUM_LONGITUDE_BINS,
        },
        "candidate_pool_size": CANDIDATE_POOL_SIZE,
        "maximum_pair_iou": MAX_TRAINING_IOU,
        "eligibility": POSITION_ELIGIBILITY,
        "eligible_centre_count": len(latitudes),
        "eligible_centres_sha256": coordinate_digest(latitudes, longitudes),
        "selected_centres_sha256": coordinate_digest(
            selected_latitudes, selected_longitudes
        ),
    }


def build_random_position_sample(
    mask: OceanMaskArtifact,
    *,
    patch_size: PatchSize,
    seed: int,
    position_count: int,
    region_mask: Callable[[Any], int] | None = None,
    static_counts: Callable[[float, float], Mapping[str, int]] | None = None,
) -> tuple[dict[str, Any], ...]:
    """Build one seeded, stratified low-overlap sample of eligible centres."""
    _, _, latitudes, longitudes = random_centre_sample(
        mask, patch_size=patch_size, seed=seed, position_count=position_count
    )
    identifier = random_grid_id(
        patch_size=patch_size,
        seed=seed,
        position_count=position_count,
        ocean_mask_id=mask.artifact_id,
    )
    rows: list[dict[str, Any]] = []
    required = {
        "swot_footprint_cells",
        "swot_ocean_cells",
        "ssh_footprint_cells",
        "ssh_ocean_cells",
    }
    for latitude, longitude in zip(latitudes, longitudes, strict=True):
        latitude, longitude = float(latitude), float(longitude)
        counts = dict(static_counts(longitude, latitude) if static_counts else {})
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
            raise ValueError("static ocean cell count cannot exceed footprint count.")
        rows.append(
            {
                "position_id": position_id(
                    grid_id=identifier, centre_lon=longitude, centre_lat=latitude
                ),
                "centre_lon": longitude,
                "centre_lat": latitude,
                "region_mask": int(
                    region_mask(patch_size.footprint(longitude, latitude))
                    if region_mask
                    else 0
                ),
                **counts,
            }
        )
    rows.sort(key=lambda row: (row["centre_lat"], row["centre_lon"]))
    return tuple({"position_index": index, **row} for index, row in enumerate(rows))


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
