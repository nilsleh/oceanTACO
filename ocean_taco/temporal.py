"""Tolerance-safe merge-within-time then explicit temporal reduction."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from datetime import UTC, datetime
from typing import Literal

import numpy as np

TemporalAggregation = Literal["first", "last", "mean", "stack"]

__all__ = ["TemporalAggregation", "merge_tiles", "merge_then_reduce"]


def _as_utc_datetime64(value: object) -> np.datetime64:
    """Normalise a decoded timestamp to nanosecond UTC precision."""
    if isinstance(value, datetime):
        if value.tzinfo is not None:
            value = value.astimezone(UTC).replace(tzinfo=None)
        timestamp = np.datetime64(value, "ns")
    else:
        timestamp = np.datetime64(value, "ns")
    if np.isnat(timestamp):
        raise ValueError("Tile timestamps must be valid decoded datetimes.")
    return timestamp


def _require_canonical_tile(tile) -> np.datetime64:
    """Validate the one-timestamp dense-grid contract for a source tile."""
    if tuple(tile.dims) != ("time", "lat", "lon"):
        raise ValueError("merge_tiles expects canonical (time, lat, lon) DataArrays.")
    if tile.sizes["time"] != 1:
        raise ValueError("merge_tiles accepts exactly one decoded timestamp per tile.")
    for coordinate in ("time", "lat", "lon"):
        if coordinate not in tile.coords or tile[coordinate].ndim != 1:
            raise ValueError(f"merge_tiles requires a one-dimensional {coordinate!r} coordinate.")
    for coordinate in ("lat", "lon"):
        values = np.asarray(tile[coordinate].values, dtype=np.float64)
        if values.size == 0 or not np.isfinite(values).all() or np.any(np.diff(values) <= 0):
            raise ValueError(f"merge_tiles requires finite, strictly ascending {coordinate!r} coordinates.")
    return _as_utc_datetime64(tile["time"].values[0])


def _cluster_axis(values: Iterable[np.ndarray], tolerance: float) -> np.ndarray:
    arrays = [np.asarray(value, dtype=np.float64) for value in values]
    if not arrays:
        return np.empty((0,), dtype=np.float64)
    combined = np.sort(np.concatenate(arrays))
    if combined.size == 0:
        return combined
    if not np.isfinite(combined).all():
        raise ValueError("Coordinate axes must be finite.")
    clusters = [[combined[0]]]
    for value in combined[1:]:
        # Compare with the cluster origin, not merely the previous value.  The
        # latter would let a chain of near values collapse a genuinely wider
        # grid cell into one coordinate.
        if abs(value - clusters[-1][0]) <= tolerance:
            clusters[-1].append(value)
        else:
            clusters.append([value])
    return np.asarray([np.mean(cluster) for cluster in clusters])


def _cluster_indices(axis: np.ndarray, values: np.ndarray, tolerance: float) -> np.ndarray:
    """Map a source coordinate axis to its unique tolerance cluster."""
    indices = np.abs(axis[:, None] - values[None, :]).argmin(axis=0)
    distances = np.abs(axis[indices] - values)
    if np.any(distances > tolerance):  # Defensive: every value built the axis.
        raise ValueError("A tile coordinate could not be matched within tolerance.")
    if np.unique(indices).size != indices.size:
        raise ValueError("Coordinate tolerance collapses distinct cells within one tile.")
    return indices


def merge_tiles(tiles, *, coordinate_tolerance: float):
    """Merge one-timestamp dense tiles on tolerance-clustered 1-D coordinates.

    Values outside a tile's coordinates are never clipped to an output edge.
    Overlap is averaged only where two source cells are finite; an all-missing
    output remains NaN and therefore retains a false source-valid mask.
    """
    import xarray as xr

    tiles = tuple(tiles)
    if not tiles:
        raise ValueError("merge_tiles requires at least one tile.")
    if coordinate_tolerance <= 0:
        raise ValueError("coordinate_tolerance must be positive.")
    times = [_require_canonical_tile(tile) for tile in tiles]
    if any(timestamp != times[0] for timestamp in times[1:]):
        raise ValueError("merge_tiles can merge only tiles at the same decoded timestamp.")
    lat = _cluster_axis((tile["lat"].values for tile in tiles), coordinate_tolerance)
    lon = _cluster_axis((tile["lon"].values for tile in tiles), coordinate_tolerance)
    canonical = []
    for tile in tiles:
        lat_indices = _cluster_indices(lat, np.asarray(tile["lat"].values, dtype=np.float64), coordinate_tolerance)
        lon_indices = _cluster_indices(lon, np.asarray(tile["lon"].values, dtype=np.float64), coordinate_tolerance)
        aligned = tile.assign_coords(lat=lat[lat_indices], lon=lon[lon_indices]).reindex(lat=lat, lon=lon)
        canonical.append(aligned)
    result = xr.concat(canonical, dim="tile").mean(dim="tile", skipna=True)
    return result.assign_coords(time=[times[0]])


def merge_then_reduce(
    timestamped_tiles: Iterable[tuple[object, object]],
    *,
    aggregation: TemporalAggregation,
    coordinate_tolerance: float,
):
    """Merge region tiles within timestamp, then reduce ordered timestamps."""
    import xarray as xr

    if aggregation not in {"first", "last", "mean", "stack"}:
        raise ValueError("aggregation must be first, last, mean, or stack.")
    grouped = defaultdict(list)
    for timestamp, tile in timestamped_tiles:
        decoded_timestamp = _require_canonical_tile(tile)
        timestamp = _as_utc_datetime64(timestamp)
        if timestamp != decoded_timestamp:
            raise ValueError(
                "timestamped_tiles must be keyed by each tile's decoded time coordinate, not a catalog label."
            )
        grouped[decoded_timestamp].append(tile)
    if not grouped:
        raise ValueError("No timestamped tiles were supplied.")
    merged = []
    for timestamp in sorted(grouped):
        value = merge_tiles(grouped[timestamp], coordinate_tolerance=coordinate_tolerance)
        merged.append(value)
    stacked = xr.concat(merged, dim="time").sortby("time")
    if aggregation == "stack":
        return stacked
    if aggregation == "first":
        return stacked.isel(time=0, drop=True)
    if aggregation == "last":
        return stacked.isel(time=-1, drop=True)
    return stacked.mean(dim="time", skipna=True)
