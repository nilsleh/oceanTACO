"""Native-coordinate GeoBox retrieval for science and paper-facing users."""

from __future__ import annotations

import io
import time
from collections.abc import Iterable
from datetime import date, datetime

from .access import LocalCacheBackend
from .catalog import CatalogConfig, load_catalog
from .geobox import GeoBox, TimeRange
from .registry import get_modality
from .temporal import _cluster_axis, _cluster_indices

__all__ = [
    "CatalogConfig",
    "load_hf_dataset",
    "load_tile_nc",
    "load_bbox_nc",
    "load_bbox_swot_nc",
    "load_multisource_time_series_nc",
]


_REGION_BOUNDS: dict[str, GeoBox] = {
    "SOUTH_PACIFIC_WEST": GeoBox(-180, -90, -90, 0),
    "SOUTH_ATLANTIC": GeoBox(-90, 0, -90, 0),
    "SOUTH_INDIAN": GeoBox(0, 90, -90, 0),
    "SOUTH_PACIFIC_EAST": GeoBox(90, 180, -90, 0),
    "NORTH_PACIFIC_WEST": GeoBox(-180, -90, 0, 90),
    "NORTH_ATLANTIC": GeoBox(-90, 0, 0, 90),
    "NORTH_INDIAN": GeoBox(0, 90, 0, 90),
    "NORTH_PACIFIC_EAST": GeoBox(90, 180, 0, 90),
}

# One cache backend must outlive an individual retrieval call for its LRU to
# have any effect.  The backend itself drops inherited HDF5 handles after a
# process boundary, so this process-local registry remains safe under workers.
_CACHE_BACKENDS: dict[CatalogConfig, LocalCacheBackend] = {}


def _cache_for(config: CatalogConfig) -> LocalCacheBackend | None:
    if config.cache_dir is None:
        return None
    cache = _CACHE_BACKENDS.get(config)
    if cache is None:
        cache = LocalCacheBackend(config.cache_dir, revision=config.revision)
        _CACHE_BACKENDS[config] = cache
    return cache


def load_hf_dataset(config: CatalogConfig | None = None):
    """Load the configured, pinned Core TACO catalog."""
    return load_catalog(config or CatalogConfig())


def _date_string(value: str | date | datetime) -> str:
    if isinstance(value, str):
        return value[:10]
    return value.isoformat()[:10]


def _filename(token: str) -> str:
    return get_modality(token).filename


def _rows(catalog, when: str, box: GeoBox, filename: str):
    """Select Core assets by named-region intersection, not bbox argument order.

    ``tacoreader`` has changed its ``filter_bbox`` positional convention across
    supported releases.  Core's eight named L1 regions are immutable, so
    resolve that tiny spatial index locally and filter the flattened catalog by
    its explicit ``l1:id`` instead of relying on an ambiguous external API.
    """
    frame = catalog.filter_datetime(f"{when}/{when}").flatten()
    tiles = [tile for tile, bounds in _REGION_BOUNDS.items() if _intersects(box, bounds)]
    return frame[
        frame["l2:id"].astype(str).str.endswith(filename)
        & frame["l1:id"].astype(str).isin(tiles)
    ]


def _intersects(left: GeoBox, right: GeoBox) -> bool:
    """Return whether two non-wrapped or segmented boxes share positive area."""
    if left.lat_max <= right.lat_min or left.lat_min >= right.lat_max:
        return False
    return any(
        segment.lon_max > right.lon_min and segment.lon_min < right.lon_max
        for segment in left.segments()
    )


def _tile_from_row(row) -> str:
    for column in ("internal:gdal_vsi", "l2:internal:gdal_vsi", "gdal_vsi", "url", "href"):
        if column in row.index:
            candidate = str(row[column]).replace("/vsicurl/", "")
            return candidate.rstrip("/").split("/")[-2]
    raise ValueError("Catalog row has no URL/VSI location from which to determine a tile.")


def _url_from_row(row) -> str:
    for column in ("internal:gdal_vsi", "l2:internal:gdal_vsi", "gdal_vsi", "url", "href"):
        if column in row.index:
            value = str(row[column])
            if "/vsisubfile/" in value:
                raise ValueError("Mode-1 retrieval requires a complete HTTP asset URL, not a VSI subfile reference.")
            return value.replace("/vsicurl/", "")
    raise ValueError("Catalog row has no URL/HREF column.")


def _download_dataset(row, config: CatalogConfig, cache: LocalCacheBackend | None, when: str, filename: str):
    import requests
    import xarray as xr

    url = _url_from_row(row)

    def fetch() -> bytes:
        error: Exception | None = None
        for attempt in range(config.retries + 1):
            try:
                response = requests.get(url, timeout=config.timeout_seconds, headers={"User-Agent": "ocean-taco/0.1"})
                response.raise_for_status()
                return response.content
            except requests.RequestException as exc:
                error = exc
                if attempt < config.retries:
                    time.sleep(min(0.25 * (2**attempt), 2.0))
        assert error is not None
        raise error

    if cache is not None:
        return cache.open_or_fetch(when, _tile_from_row(row), filename, fetch)
    return xr.open_dataset(io.BytesIO(fetch()), engine="h5netcdf")


def load_tile_nc(
    catalog,
    when: str | date | datetime,
    tile: str,
    token: str,
    *,
    config: CatalogConfig | None = None,
    backend: LocalCacheBackend | None = None,
) -> object | None:
    """Read one named-region source asset using the local immutable cache."""
    if tile not in _REGION_BOUNDS:
        raise ValueError(f"Unknown Core region {tile!r}.")
    config = config or CatalogConfig()
    when_string, filename = _date_string(when), _filename(token)
    frame = catalog.filter_datetime(f"{when_string}/{when_string}").flatten()
    rows = frame[
        frame["l2:id"].astype(str).str.endswith(filename)
        & frame["l1:id"].astype(str).eq(tile)
    ]
    if rows.empty:
        return None
    cache = backend or _cache_for(config)
    return _download_dataset(rows.iloc[0], config, cache, when_string, filename)


def _clean_swot(dataset):
    allowed = {"time", "lat", "lon"}
    non_spatial = [dimension for dimension in dataset.dims if dimension not in allowed]
    to_drop = [name for name, variable in dataset.data_vars.items() if set(variable.dims).intersection(non_spatial)]
    if to_drop:
        dataset = dataset.drop_vars(to_drop)
    if non_spatial:
        dataset = dataset.drop_dims(non_spatial)
    return dataset


def _point_dimension(dataset) -> str:
    """Return the shared one-dimensional Argo record axis."""
    for name in ("lat", "lon", "time"):
        if name not in dataset or dataset[name].ndim != 1:
            raise ValueError("Ragged-point retrieval requires one-dimensional lat, lon, and time fields.")
    dimensions = {dataset[name].dims[0] for name in ("lat", "lon", "time")}
    if len(dimensions) != 1:
        raise ValueError("Ragged-point lat, lon, and time fields must share one record dimension.")
    return dimensions.pop()


def _crop_points(dataset, box: GeoBox):
    """Select points by their own coordinates without broadcasting grid fields."""
    import numpy as np

    dimension = _point_dimension(dataset)
    lat = np.asarray(dataset["lat"].values)
    lon = np.where(
        np.asarray(dataset["lon"].values) == 180.0,
        -180.0,
        ((np.asarray(dataset["lon"].values) + 180.0) % 360.0) - 180.0,
    )
    selected = (lat >= box.lat_min) & (lat <= box.lat_max)
    longitude_selected = np.zeros(lon.shape, dtype=bool)
    for segment in box.segments():
        longitude_selected |= (lon >= segment.lon_min) & (lon <= segment.lon_max)
    return dataset.isel({dimension: selected & longitude_selected})


def _merge_points(datasets):
    """Concatenate ragged source records while preserving their native fields."""
    import xarray as xr

    if not datasets:
        return None
    dimensions = {_point_dimension(dataset) for dataset in datasets}
    if len(dimensions) != 1:
        raise ValueError("Ragged-point assets use incompatible record dimensions.")
    return xr.concat(
        datasets,
        dim=dimensions.pop(),
        data_vars="all",
        coords="minimal",
        compat="override",
        combine_attrs="override",
    )


def _canonicalise_grid_coordinates(dataset):
    """Return a grid on sorted canonical lon/lat axes without changing values."""
    import numpy as np

    for coordinate in ("lat", "lon"):
        if coordinate not in dataset.coords or dataset[coordinate].ndim != 1:
            raise ValueError("GeoBox grid retrieval requires one-dimensional lat/lon coordinates.")
    lat = np.asarray(dataset["lat"].values, dtype=np.float64)
    lon = np.asarray(dataset["lon"].values, dtype=np.float64)
    if not np.isfinite(lat).all() or not np.isfinite(lon).all():
        raise ValueError("GeoBox grid retrieval requires finite lat/lon coordinates.")
    # Coordinate selection is always canonical [-180, 180).  The source may
    # have used [0, 360], but exposing that here would make wrapped GeoBoxes
    # ambiguous and can duplicate the antimeridian.
    lon = ((lon + 180.0) % 360.0) - 180.0
    result = dataset.assign_coords(lat=lat, lon=lon).sortby("lat").sortby("lon")
    for coordinate in ("lat", "lon"):
        values = np.asarray(result[coordinate].values, dtype=np.float64)
        if np.unique(values).size != values.size:
            raise ValueError(f"GeoBox grid retrieval found duplicate {coordinate!r} coordinates.")
    return result


def _merge_grid_tiles(datasets, *, coordinate_tolerance: float):
    """Merge spatial tiles after snapping documented coordinate jitter.

    The source files are authoritative for their values.  This helper only
    canonicalises coordinate labels so xarray does not turn metre-scale float
    jitter at a region boundary into an extra grid cell.
    """
    import numpy as np
    import xarray as xr

    canonical = tuple(_canonicalise_grid_coordinates(dataset) for dataset in datasets)
    if len(canonical) == 1:
        return canonical[0]
    if coordinate_tolerance <= 0:
        raise ValueError("coordinate_tolerance must be positive.")
    lat = _cluster_axis((dataset["lat"].values for dataset in canonical), coordinate_tolerance)
    lon = _cluster_axis((dataset["lon"].values for dataset in canonical), coordinate_tolerance)
    aligned = []
    for dataset in canonical:
        lat_indices = _cluster_indices(lat, np.asarray(dataset["lat"].values, dtype=np.float64), coordinate_tolerance)
        lon_indices = _cluster_indices(lon, np.asarray(dataset["lon"].values, dtype=np.float64), coordinate_tolerance)
        aligned.append(
            dataset.assign_coords(lat=lat[lat_indices], lon=lon[lon_indices])
            .sortby("lat")
            .sortby("lon")
            .reindex(lat=lat, lon=lon)
        )
    try:
        # All tiles are now labelled on the same canonical axes.  ``merge``
        # unions their non-overlapping cells; ``combine_by_coords`` would
        # concatenate the shared seam coordinate a second time.
        return xr.merge(aligned, combine_attrs="override", compat="no_conflicts", join="exact").sortby("lat").sortby("lon")
    except ValueError as error:
        raise ValueError("Region tiles disagree on overlapping grid values after coordinate alignment.") from error


def _crop(dataset, box: GeoBox):
    """Crop every grid variable without leaking a tuple-shaped public API."""
    import xarray as xr

    parts = []
    for segment in box.segments():
        part = dataset.sel(lat=slice(segment.lat_min, segment.lat_max), lon=slice(segment.lon_min, segment.lon_max))
        parts.append(part)
    if not box.wraps_antimeridian:
        return parts[0]
    result = xr.concat(parts, dim="lon")
    return result.assign_coords(lon=box.unwrap_longitudes(result["lon"].values))


def load_bbox_nc(
    catalog,
    when: str | date | datetime,
    box: GeoBox,
    token: str,
    *,
    config: CatalogConfig | None = None,
    backend: LocalCacheBackend | None = None,
) -> object | None:
    """Read and coordinate-merge all source tiles intersecting a GeoBox."""
    config = config or CatalogConfig()
    when_string, filename = _date_string(when), _filename(token)
    source = get_modality(token)
    cache = backend or _cache_for(config)
    datasets = []
    seen_urls: set[str] = set()
    for segment in box.segments():
        rows = _rows(catalog, when_string, segment, filename)
        for _, row in rows.iterrows():
            url = _url_from_row(row)
            if url in seen_urls:
                continue
            seen_urls.add(url)
            dataset = _download_dataset(row, config, cache, when_string, filename)
            datasets.append(_clean_swot(dataset) if token == "l3_swot" else dataset)
    if not datasets:
        return None
    if source.is_points:
        return _merge_points([_crop_points(dataset, box) for dataset in datasets])
    merged = _merge_grid_tiles(datasets, coordinate_tolerance=get_modality(token).regularity_tolerance)
    return _crop(merged, box)


def load_bbox_swot_nc(
    catalog,
    when: str | date | datetime,
    box: GeoBox,
    *,
    config: CatalogConfig | None = None,
    backend: LocalCacheBackend | None = None,
) -> object | None:
    """Read a GeoBox crop of dense, upstream-binned L3 SWOT data."""
    return load_bbox_nc(catalog, when, box, "l3_swot", config=config, backend=backend)


def _days(interval: TimeRange) -> Iterable[date]:
    current = interval.start.date()
    while current <= interval.end.date():
        yield current
        current = current.fromordinal(current.toordinal() + 1)


def _ensure_time_dimension(dataset):
    """Promote a decoded scalar time coordinate; never invent a catalog time."""
    import numpy as np

    if "time" in dataset.dims:
        if "time" not in dataset.coords or dataset["time"].ndim != 1:
            raise ValueError("Grid retrieval requires a one-dimensional decoded time coordinate.")
        return dataset
    if "time" not in dataset.coords or dataset["time"].ndim != 0:
        raise ValueError("Grid retrieval requires a decoded time coordinate; catalog dates are not source times.")
    timestamp = np.datetime64(dataset["time"].values, "ns")
    if np.isnat(timestamp):
        raise ValueError("Grid retrieval found an invalid decoded scalar time coordinate.")
    return dataset.expand_dims(time=[timestamp])


def _select_time_range(dataset, interval: TimeRange):
    """Select the closed request interval using decoded source timestamps."""
    import numpy as np

    data = _ensure_time_dimension(dataset)
    times = np.asarray(data["time"].values, dtype="datetime64[ns]")
    start = np.datetime64(interval.start.replace(tzinfo=None), "ns")
    end = np.datetime64(interval.end.replace(tzinfo=None), "ns")
    return data.isel(time=(times >= start) & (times <= end))


def load_multisource_time_series_nc(
    catalog,
    tokens: Iterable[str],
    box: GeoBox,
    time: TimeRange,
    *,
    config: CatalogConfig | None = None,
    backend: LocalCacheBackend | None = None,
) -> dict[str, object | None]:
    """Retrieve native-coordinate source stacks over an explicit time range."""
    import xarray as xr

    config = config or CatalogConfig()
    cache = backend or _cache_for(config)
    result: dict[str, object | None] = {}
    for token in tokens:
        per_day = [load_bbox_nc(catalog, day, box, token, config=config, backend=cache) for day in _days(time)]
        if get_modality(token).is_points:
            available_points = [dataset for dataset in per_day if dataset is not None]
            result[token] = _merge_points(available_points)
            continue
        available = [
            selected
            for dataset in per_day
            if dataset is not None
            for selected in (_select_time_range(dataset, time),)
            if selected.sizes.get("time", 0) > 0
        ]
        if not available:
            result[token] = None
        else:
            # Source timestamps, rather than catalog row/date order, determine
            # the returned temporal axis.  Per-date spatial merging has already
            # completed inside load_bbox_nc.
            result[token] = xr.concat(
                available,
                dim="time",
                data_vars="all",
                coords="minimal",
                compat="no_conflicts",
                combine_attrs="override",
            ).sortby("time")
    return result
