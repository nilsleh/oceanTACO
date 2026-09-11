"""Native-coordinate GeoBox retrieval for science and paper-facing users."""

from __future__ import annotations

import os
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date, datetime
from functools import lru_cache

from .access import LocalCacheBackend
from .catalog import CatalogConfig, load_catalog
from .geobox import GeoBox, TimeRange
from .registry import get_modality
from .temporal import _cluster_axis, _cluster_indices

__all__ = [
    "REGIONS",
    "REGION_BIT",
    "CatalogConfig",
    "load_bbox_nc",
    "load_bbox_swot_nc",
    "load_hf_dataset",
    "load_multisource_time_series_nc",
    "load_tile_nc",
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

# Public region-mask allocation for QueryFilter(region_mask_any=...).
REGIONS: tuple[str, ...] = tuple(sorted(_REGION_BOUNDS))
REGION_BIT: dict[str, int] = {
    region: 1 << index for index, region in enumerate(REGIONS)
}

# One cache backend must outlive an individual retrieval call for its LRU to
# have any effect.  The backend itself drops inherited HDF5 handles after a
# process boundary, so this process-local registry remains safe under workers.
_CACHE_BACKENDS: dict[CatalogConfig, LocalCacheBackend] = {}


def _cache_for(config: CatalogConfig) -> LocalCacheBackend:
    cache = _CACHE_BACKENDS.get(config)
    if cache is None:
        cache = LocalCacheBackend(config.cache_dir, revision=config.revision, max_open_files=config.max_open_files)
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


@dataclass(frozen=True, slots=True)
class ResolvedAsset:
    """One catalog row reduced to the worker-safe data needed to open it."""

    location: str
    tile: str


def _rows_from_frame(frame, box: GeoBox, filename: str):
    """Select matching rows from an already-filtered catalog date frame."""
    tiles = [tile for tile, bounds in _REGION_BOUNDS.items() if _intersects(box, bounds)]
    l2_ids = frame.get("_oceantaco_l2_id", frame["l2:id"].astype(str))
    l1_ids = frame.get("_oceantaco_l1_id", frame["l1:id"].astype(str))
    return frame[
        l2_ids.str.endswith(filename)
        & l1_ids.isin(tiles)
    ]

def _rows(catalog, when: str, box: GeoBox, filename: str):
    """Select Core assets by named-region intersection, not bbox argument order.

    ``tacoreader`` has changed its ``filter_bbox`` positional convention across
    supported releases.  Core's eight named L1 regions are immutable, so
    resolve that tiny spatial index locally and filter the flattened catalog by
    its explicit ``l1:id`` instead of relying on an ambiguous external API.
    """
    return _rows_from_frame(catalog.filter_datetime(f"{when}/{when}").flatten(), box, filename)


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


def _is_local_location(value: str) -> bool:
    """Return whether a catalog location addresses the local filesystem.

    A local ``.taco`` catalog yields plain absolute paths, so the decision is
    made from the location itself rather than from ``taco_path``: a catalog may
    legitimately mix local and remote assets, and only the value knows which.
    """
    from urllib.parse import urlparse

    if value.startswith("file://"):
        return True
    return urlparse(value).scheme in {"", "file"}


def _local_path(value: str) -> str:
    from urllib.parse import unquote, urlparse

    if value.startswith("file://"):
        return unquote(urlparse(value).path)
    return value


def _hub_relative_path(location: str, config: CatalogConfig) -> str:
    """Return the repo-relative path of a Hub asset URL built by this library.

    ``CatalogConfig.resolved_catalog_url`` composes the very prefix stripped
    here, so the remainder is exactly ``hf_hub_download``'s ``filename`` and no
    path mapping is required.
    """
    prefix = f"https://huggingface.co/datasets/{config.repo_id}/resolve/{config.revision}/"
    if not location.startswith(prefix):
        raise ValueError(
            "Remote retrieval addresses the pinned Hub dataset only; "
            f"cannot resolve {location!r} against {prefix!r}."
        )
    return location[len(prefix) :]


def _open_remote(location: str, config: CatalogConfig, cache: LocalCacheBackend | None):
    """Download one Hub asset and open it, reusing the Hub's own cache.

    ``hf_hub_download`` caches atomically under ``HF_HOME`` and keys on the
    revision, so no second copy is made here; ``cache`` contributes only its
    fork-safe read-handle LRU when one is configured.
    """
    import xarray as xr
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        repo_id=config.repo_id,
        filename=_hub_relative_path(location, config),
        revision=config.revision,
        repo_type="dataset",
        cache_dir=config.cache_dir,
    )
    if cache is not None:
        return cache.open_path(path)
    return xr.open_dataset(path, engine="h5netcdf")


def _download_dataset(row, config: CatalogConfig, cache: LocalCacheBackend | None, when: str, filename: str):
    import xarray as xr

    url = _url_from_row(row)

    if _is_local_location(url):
        # A local asset is already immutable on disk.  Routing it through the
        # fetch cache would copy files into a cache of files.
        path = _local_path(url)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Catalog references a local asset that does not exist: {path}")
        return cache.open_path(path) if cache is not None else xr.open_dataset(path, engine="h5netcdf")

    return _open_remote(url, config, cache)


def _download_location(location: str, config: CatalogConfig, cache: LocalCacheBackend | None):
    """Open a resolved local path or pinned Hub location without a catalog object."""
    import xarray as xr

    if _is_local_location(location):
        path = _local_path(location)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Catalog references a local asset that does not exist: {path}")
        return cache.open_path(path) if cache is not None else xr.open_dataset(path, engine="h5netcdf")

    return _open_remote(location, config, cache)


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
    dimension = dimensions.pop()
    if len(datasets) == 1 and all(dimension in variable.dims for variable in datasets[0].data_vars.values()):
        return datasets[0]
    return xr.concat(
        datasets,
        dim=dimension,
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
    result = dataset.assign_coords(lat=lat, lon=lon)
    for coordinate in ("lat", "lon"):
        if np.any(np.diff(result[coordinate].values) < 0):
            result = result.sortby(coordinate)
    for coordinate in ("lat", "lon"):
        values = np.asarray(result[coordinate].values, dtype=np.float64)
        if np.unique(values).size != values.size:
            raise ValueError(f"GeoBox grid retrieval found duplicate {coordinate!r} coordinates.")
    return result


@lru_cache(maxsize=32)
def _missing_dtype(dtype):
    """Ask xarray how full-axis reindexing promotes this native dtype."""
    import numpy as np
    import xarray as xr

    probe = xr.DataArray(np.empty(0, dtype=dtype), dims="cell", coords={"cell": []})
    return probe.reindex(cell=[0]).dtype


def _merge_grid_tiles(datasets, *, coordinate_tolerance: float, box: GeoBox | None = None, coordinates_normalized: bool = False, read_tile=None):
    """Merge spatial tiles after snapping documented coordinate jitter.

    The source files are authoritative for their values.  This helper only
    canonicalises coordinate labels so xarray does not turn metre-scale float
    jitter at a region boundary into an extra grid cell.
    """
    import numpy as np
    import xarray as xr

    canonical = tuple(datasets) if coordinates_normalized else tuple(_canonicalise_grid_coordinates(dataset) for dataset in datasets)
    if len(canonical) == 1:
        return canonical[0] if box is None else _crop(canonical[0], box)
    if coordinate_tolerance <= 0:
        raise ValueError("coordinate_tolerance must be positive.")
    lat = _cluster_axis((dataset["lat"].values for dataset in canonical), coordinate_tolerance)
    lon = _cluster_axis((dataset["lon"].values for dataset in canonical), coordinate_tolerance)
    # Cluster complete lightweight axes first: cropping before clustering
    # changes cluster means and can change inclusion at a request boundary.
    selected_lat, selected_lon = lat, lon
    if box is not None:
        selected_lat = lat[(lat >= box.lat_min) & (lat <= box.lat_max)]
        keep_lon = np.zeros(lon.shape, dtype=bool)
        for segment in box.segments():
            keep_lon |= (lon >= segment.lon_min) & (lon <= segment.lon_max)
        selected_lon = lon[keep_lon]
    aligned = []
    for tile_index, dataset in enumerate(canonical):
        lat_indices = _cluster_indices(lat, np.asarray(dataset["lat"].values, dtype=np.float64), coordinate_tolerance)
        lon_indices = _cluster_indices(lon, np.asarray(dataset["lon"].values, dtype=np.float64), coordinate_tolerance)
        tile_lat, tile_lon = lat[lat_indices], lon[lon_indices]
        lat_keep = np.flatnonzero(np.isin(tile_lat, selected_lat))
        lon_keep = np.flatnonzero(np.isin(tile_lon, selected_lon))
        # A very small LRU may have evicted this handle during the axes pass.
        # Reopen through the LRU and detach the crop before opening another tile.
        if read_tile is not None:
            dataset = read_tile(tile_index)
        # isel precedes reindex/merge: only the requested cells are decoded.
        tile = dataset.isel(lat=lat_keep, lon=lon_keep).assign_coords(
            lat=tile_lat[lat_keep], lon=tile_lon[lon_keep]
        )
        tile = _materialise_empty_variables(tile)
        if read_tile is not None:
            tile.load()
        if box is not None:
            # Full regional reindexing can promote integers even when every
            # requested cell is present. Preserve that observable native dtype.
            missing = {dim for dim, count in (("lat", lat.size), ("lon", lon.size))
                       if dataset.sizes[dim] < count}
            for name, variable in tile.variables.items():
                if missing.intersection(variable.dims):
                    dtype = _missing_dtype(variable.dtype)
                    if dtype != variable.dtype:
                        tile[name] = tile[name].astype(dtype)
        aligned.append(tile.reindex(lat=selected_lat, lon=selected_lon))
    try:
        # All tiles are now labelled on the same canonical axes.  ``merge``
        # unions their non-overlapping cells; ``combine_by_coords`` would
        # concatenate the shared seam coordinate a second time.
        merged = xr.merge(aligned, combine_attrs="override", compat="no_conflicts", join="exact").sortby("lat").sortby("lon")
        return merged if box is None else _crop(merged, box)
    except ValueError as error:
        raise ValueError("Region tiles disagree on overlapping grid values after coordinate alignment.") from error


def _materialise_empty_variables(dataset):
    """Avoid backend indexing of empty arrays after chained lazy selections."""
    import numpy as np

    empty = [name for name, variable in dataset.variables.items() if variable.size == 0]
    if not empty:
        return dataset
    result = dataset.copy(deep=False)
    for name in empty:
        variable = dataset[name].variable
        result[name] = variable.copy(data=np.empty(variable.shape, dtype=variable.dtype))
    return result


def _crop(dataset, box: GeoBox):
    """Crop every grid variable without leaking a tuple-shaped public API."""
    import xarray as xr

    parts = []
    for segment in box.segments():
        part = dataset.sel(lat=slice(segment.lat_min, segment.lat_max), lon=slice(segment.lon_min, segment.lon_max))
        parts.append(_materialise_empty_variables(part))
    if not box.wraps_antimeridian:
        return parts[0]
    # Preserve xarray's current all-variable concatenation semantics across
    # the antimeridian; spelling this out avoids a future-default change.
    result = xr.concat(parts, dim="lon", data_vars="all")
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
    """Read and coordinate-merge all source tiles intersecting a GeoBox.

    Every intersecting Core region is merged using decoded coordinate labels,
    not argument order, and the result is then cropped to ``box``.  Returns
    ``None`` when no matching asset exists for that date and token.  Set
    ``GeoBox(..., wraps_antimeridian=True)`` only for an intentionally wrapped
    longitude interval; the flag is never inferred.
    """
    config = config or CatalogConfig()
    when_string, filename = _date_string(when), _filename(token)
    cache = backend or _cache_for(config)
    assets = []
    seen_urls: set[str] = set()
    for segment in box.segments():
        rows = _rows(catalog, when_string, segment, filename)
        for _, row in rows.iterrows():
            url = _url_from_row(row)
            if url in seen_urls:
                continue
            seen_urls.add(url)
            assets.append(ResolvedAsset(url, _tile_from_row(row)))
    return _load_planned_bbox_nc(assets, when_string, box, token, config=config, backend=cache)


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


def _select_time_range(dataset, interval: TimeRange, token: str | None = None):
    """Select the closed request interval using decoded source timestamps.

    A requested date selects the whole calendar day, 00:00 to 24:00 UTC.  Daily
    gridded products carry one field per day but disagree about where in the day
    to stamp it: L4 SSH and L4 wind use 00:00, GLORYS and L4 SSS use 12:00, and
    L4 SST and the L3 altimetry products use either depending on the granule.
    A QuerySet anchor is midnight, so a single-day request is zero-width as an
    instant and silently drops every granule not stamped exactly at 00:00 --
    indistinguishable from data that is genuinely absent.  Comparing on the
    calendar day is the resolution these labels actually carry.

    ``point_time`` sources are excluded: an Argo profile's surfacing time is a
    real instant rather than a label for its day, and its own selection path
    handles it.
    """
    import numpy as np

    data = _ensure_time_dimension(dataset)
    times = np.asarray(data["time"].values, dtype="datetime64[ns]")
    start = np.datetime64(interval.start.replace(tzinfo=None), "ns")
    end = np.datetime64(interval.end.replace(tzinfo=None), "ns")
    if token is None or get_modality(token).source_time_kind != "point_time":
        times, start, end = (
            times.astype("datetime64[D]"),
            start.astype("datetime64[D]"),
            end.astype("datetime64[D]"),
        )
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
            for selected in (_select_time_range(dataset, time, token),)
            if selected.sizes.get("time", 0) > 0
        ]
        if not available:
            result[token] = None
        else:
            # Source timestamps, rather than catalog row/date order, determine
            # the returned temporal axis.  Per-date spatial merging has already
            # completed inside load_bbox_nc.
            result[token] = (available[0] if len(available) == 1 and all("time" in variable.dims for variable in available[0].data_vars.values()) else xr.concat(
                available,
                dim="time",
                data_vars="all",
                coords="minimal",
                compat="no_conflicts",
                combine_attrs="override",
            )).sortby("time")
    return result

AssetPlan = dict[tuple[str, str, GeoBox], tuple[ResolvedAsset, ...]]


def plan_multisource_assets(
    catalog,
    requests: Iterable[tuple[str, GeoBox, TimeRange]],
) -> AssetPlan:
    """Resolve logical source requests to serialisable assets in the parent.

    The catalog is queried once per distinct ``(token, day)``.  Individual
    patches then filter that in-memory date frame by their named Core regions.
    The returned plan contains only strings and immutable geographic boxes, so
    it can cross a PyTorch process boundary without carrying DuckDB, obstore,
    or tacoreader state.
    """
    entries = {
        (token, _date_string(day), box)
        for token, box, interval in requests
        for day in _days(interval)
    }
    grouped: dict[tuple[str, str], list[GeoBox]] = {}
    for token, when, box in entries:
        grouped.setdefault((token, when), []).append(box)

    plan: AssetPlan = {}
    for (token, when), boxes in grouped.items():
        frame = catalog.filter_datetime(f"{when}/{when}").flatten().assign(
            _oceantaco_l2_id=lambda value: value["l2:id"].astype(str),
            _oceantaco_l1_id=lambda value: value["l1:id"].astype(str),
        )
        filename = _filename(token)
        for box in boxes:
            assets: list[ResolvedAsset] = []
            seen_locations: set[str] = set()
            for segment in box.segments():
                rows = _rows_from_frame(frame, segment, filename)
                for _, row in rows.iterrows():
                    location = _url_from_row(row)
                    if location in seen_locations:
                        continue
                    seen_locations.add(location)
                    assets.append(ResolvedAsset(location, _tile_from_row(row)))
            plan[(token, when, box)] = tuple(assets)
    return plan


def _load_planned_bbox_nc(
    assets: Iterable[ResolvedAsset],
    when: str,
    box: GeoBox,
    token: str,
    *,
    config: CatalogConfig,
    backend: LocalCacheBackend | None,
    variables: tuple[str, ...] | None = None,
):
    """Fetch and render-ready merge of catalog-free planned asset locations."""
    backend = backend or _cache_for(config)
    source = get_modality(token)
    assets = tuple(assets)

    def read_asset(index):
        dataset = _download_location(assets[index].location, config, backend)
        if not source.is_points:
            dataset = backend.canonical_grid(dataset)
        if variables is not None:
            dataset = dataset[list(variables)]
        return _clean_swot(dataset) if token == "l3_swot" else dataset

    if source.is_points:
        # Crop and detach before the next open, even with a one-file LRU.
        return _merge_points([_crop_points(read_asset(index), box).load() for index in range(len(assets))])
    datasets = [read_asset(index) for index in range(len(assets))]
    if not datasets:
        return None
    return _merge_grid_tiles(
        datasets, coordinate_tolerance=source.regularity_tolerance, box=box, coordinates_normalized=True,
        read_tile=read_asset if len(assets) > backend.max_open_files else None,
    )


def _daily_cache_key(assets, when, box, token, variables):
    source = get_modality(token)
    return (tuple(assets), when, box, source.filename, variables)


def load_planned_multisource_time_series_nc(
    plan: AssetPlan,
    tokens: Iterable[str],
    box: GeoBox,
    time: TimeRange,
    *,
    config: CatalogConfig,
    backend: LocalCacheBackend | None = None,
    variables_by_token: dict[str, tuple[str, ...] | None] | None = None,
    daily_cache: dict | None = None,
) -> dict[str, object | None]:
    """Fetch a planned request without consulting a catalog in this process."""
    import xarray as xr

    result: dict[str, object | None] = {}
    # A call-local cache also coalesces paired variables in shared assets.
    daily_cache = {} if daily_cache is None else daily_cache
    for token in tokens:
        per_day = []
        variables = None if variables_by_token is None else variables_by_token[token]
        for day in _days(time):
            when = _date_string(day)
            assets = plan.get((token, when, box), ())
            key = _daily_cache_key(assets, when, box, token, variables)
            if key not in daily_cache:
                dataset = _load_planned_bbox_nc(
                    assets, when, box, token, config=config, backend=backend, variables=variables
                )
                # Detach only the crop from file handles. Batch caches never
                # retain full regional values or live lazy reads after eviction.
                daily_cache[key] = None if dataset is None else dataset.load()
            per_day.append(daily_cache[key])
        if get_modality(token).is_points:
            result[token] = _merge_points(
                [dataset for dataset in per_day if dataset is not None]
            )
            continue
        available = [
            selected
            for dataset in per_day
            if dataset is not None
            for selected in (_select_time_range(dataset, time, token),)
            if selected.sizes.get("time", 0) > 0
        ]
        if not available:
            result[token] = None
        else:
            result[token] = (available[0] if len(available) == 1 and all("time" in variable.dims for variable in available[0].data_vars.values()) else xr.concat(
                available,
                dim="time",
                data_vars="all",
                coords="minimal",
                compat="no_conflicts",
                combine_attrs="override",
            )).sortby("time")
    return result
