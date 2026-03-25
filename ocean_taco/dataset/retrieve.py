"""HuggingFace / tacoreader access for the OceanTACO dataset."""

import io
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
import tacoreader

import requests
import xarray as xr

HF_DEFAULT_URL = "https://huggingface.co/datasets/nilsleh/OceanTACO/resolve/main/"

# (lon_min, lat_min, lon_max, lat_max) for the 8 named ocean tiles
_REGION_BBOXES = {
    "SOUTH_PACIFIC_WEST": (-180, -90, -90, 0),
    "SOUTH_ATLANTIC": (-90, -90, 0, 0),
    "SOUTH_INDIAN": (0, -90, 90, 0),
    "SOUTH_PACIFIC_EAST": (90, -90, 180, 0),
    "NORTH_PACIFIC_WEST": (-180, 0, -90, 90),
    "NORTH_ATLANTIC": (-90, 0, 0, 90),
    "NORTH_INDIAN": (0, 0, 90, 90),
    "NORTH_PACIFIC_EAST": (90, 0, 180, 90),
}


def load_hf_dataset(url: str = HF_DEFAULT_URL):
    """Load OceanTACO catalog from HuggingFace via tacoreader.

    Args:
        url: Base URL of the HuggingFace dataset repository.

    Returns:
        TacoDataset catalog object.
    """
    tacoreader.use("pandas")
    return tacoreader.load(url)


def query_files(dataset, date: str, bbox: tuple, needed_files: set):
    """Filter catalog to specific files in a bbox on a given date.

    Args:
        dataset: TacoDataset object.
        date: Date string YYYY-MM-DD.
        bbox: Tuple (lon_min, lat_min, lon_max, lat_max).
        needed_files: Set of filenames to keep (e.g. {'l4_ssh.nc'}).

    Returns:
        Flattened DataFrame filtered to requested files.
    """
    flat = (
        dataset.filter_datetime(f"{date}/{date}").filter_bbox(*bbox, level=1).flatten()
    )
    flat = flat[flat["l2:id"].isin(needed_files)]
    return flat


def query_global_glorys(dataset, date: str):
    """Return all GLORYS tiles across all regions for a given date.

    Args:
        dataset: TacoDataset object.
        date: Date string YYYY-MM-DD.

    Returns:
        Flattened DataFrame filtered to glorys.nc files.
    """
    flat = dataset.filter_datetime(f"{date}/{date}").flatten()
    flat = flat[flat["l2:id"].str.contains("glorys.nc")]
    return flat


def fetch_nc(row) -> tuple[str, xr.Dataset]:
    """Download a single NetCDF into memory via HTTP GET.

    Args:
        row: DataFrame row with 'l2:id' and url information.

    Returns:
        Tuple of (filename, xr.Dataset).
    """
    fname = row["l2:id"].split("/")[-1]
    url = None
    for col in ["internal:gdal_vsi", "gdal_vsi", "url", "href"]:
        if col in row.index:
            url = str(row[col]).replace("/vsicurl/", "")
            break
    if url is None:
        raise ValueError(
            f"Cannot find URL column for {fname}. Columns: {list(row.index)}"
        )

    r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    return fname, xr.open_dataset(io.BytesIO(r.content), engine="h5netcdf")


def _tile_bbox(tile: str) -> tuple:
    """Return (lon_min, lat_min, lon_max, lat_max) for a named ocean tile.

    Args:
        tile: One of the 8 named ocean regions.

    Returns:
        Tuple (lon_min, lat_min, lon_max, lat_max).
    """
    if tile not in _REGION_BBOXES:
        raise ValueError(f"Unknown tile '{tile}'. Valid: {list(_REGION_BBOXES)}")
    return _REGION_BBOXES[tile]


def load_tile_nc(
    dataset, date: str, tile: str, data_source: str, cache_dir=None
) -> "xr.Dataset | None":
    """Load one TACO-format tile file, optionally caching to disk.

    Args:
        dataset: TacoDataset from load_hf_dataset().
        date: Date string YYYY-MM-DD.
        tile: Region tile name, e.g. 'NORTH_PACIFIC_EAST'.
        data_source: Source token or filename, e.g. 'l4_ssh' or 'l4_ssh.nc'.
        cache_dir: If provided, cache to {cache_dir}/{date}/{tile}/{resolved_filename}.

    Returns:
        xr.Dataset or None if not found.
    """
    resolved_filename = _normalize_source_filename(data_source)

    if cache_dir:
        cached = Path(cache_dir) / date / tile / resolved_filename
        if cached.exists():
            return xr.open_dataset(cached, engine="h5netcdf")

    bbox = _tile_bbox(tile)
    files = query_files(dataset, date, bbox, {resolved_filename})
    if files.empty:
        return None
    _, ds = fetch_nc(files.iloc[0])

    if cache_dir:
        cached.parent.mkdir(parents=True, exist_ok=True)
        ds.to_netcdf(cached)

    return ds


def load_region_product_nc(
    dataset, date: str, region: str, data_source: str, cache_dir=None
) -> "xr.Dataset | None":
    """Load one formatted-region product file, optionally caching to disk.

    Args:
        dataset: TacoDataset from load_hf_dataset().
        date: Date string YYYY-MM-DD.
        region: Region name, e.g. 'NORTH_ATLANTIC'.
        data_source: Source token or filename as stored in TACO,
            e.g. 'glorys' or 'glorys.nc'.
        cache_dir: If provided, cache to {cache_dir}/{date}/{region}/{resolved_filename}.

    Returns:
        xr.Dataset or None if not found.
    """
    return load_tile_nc(
        dataset=dataset,
        date=date,
        tile=region,
        data_source=data_source,
        cache_dir=cache_dir,
    )


def _tile_name_from_row(row) -> str | None:
    """Extract the tile/region name from a catalog row URL."""
    for col in ["internal:gdal_vsi", "gdal_vsi", "url", "href"]:
        if col in row.index:
            return str(row[col]).split("/")[-2]
    return None


def _normalize_source_filename(data_source: str) -> str:
    """Resolve a data source token to a canonical NetCDF filename.

    Accepts source tokens such as ``l4_sst`` and also values that already end
    with ``.nc``.

    Args:
        data_source: Source token or NetCDF filename.

    Returns:
        Canonical filename ending in ``.nc``.
    """
    value = data_source.strip()
    if not value:
        raise ValueError("data_source cannot be empty.")
    return value if value.endswith(".nc") else f"{value}.nc"


def _source_token(data_source: str) -> str:
    """Normalize a data source to its extension-free token."""
    resolved = _normalize_source_filename(data_source)
    return resolved[:-3]


def _iter_dates(date_start: str, date_end: str) -> list[str]:
    """Build an inclusive list of YYYY-MM-DD dates."""
    start = datetime.strptime(date_start, "%Y-%m-%d").date()
    end = datetime.strptime(date_end, "%Y-%m-%d").date()
    if end < start:
        raise ValueError("date_end must be greater than or equal to date_start.")

    out = []
    cur = start
    while cur <= end:
        out.append(cur.isoformat())
        cur += timedelta(days=1)
    return out


def _ensure_time_dim(ds: xr.Dataset, d: str) -> xr.Dataset:
    """Ensure dataset has a time dimension for stacking.

    If a source does not expose an explicit ``time`` dimension after loading,
    add one using the requested date.
    """
    if "time" in ds.dims:
        return ds
    dt = datetime.strptime(d, "%Y-%m-%d")
    return ds.expand_dims(time=[dt])


def load_multisource_time_series_nc(
    dataset,
    data_sources: list[str],
    date_start: str,
    date_end: str,
    *,
    tile: str | None = None,
    bbox: tuple[float, float, float, float] | None = None,
    cache_dir=None,
) -> dict[str, xr.Dataset | None]:
    """Load multiple sources over a date range and stack each source over time.

    Exactly one location selector must be provided: ``tile`` or ``bbox``.

    Args:
        dataset: TacoDataset from ``load_hf_dataset()``.
        data_sources: Source tokens or filenames, e.g. ``["l4_ssh", "glorys.nc"]``.
        date_start: Inclusive start date ``YYYY-MM-DD``.
        date_end: Inclusive end date ``YYYY-MM-DD``.
        tile: Named OceanTACO region tile.
        bbox: Bounding box ``(lon_min, lat_min, lon_max, lat_max)``.
        cache_dir: Optional local cache directory passed to underlying loaders.

    Returns:
        Dict keyed by source token (e.g. ``"l4_sst"``).
        Values are datasets concatenated over ``time`` or ``None`` when no data
        was found for that source across the requested range.
    """
    if (tile is None) == (bbox is None):
        raise ValueError("Provide exactly one of tile or bbox.")

    dates = _iter_dates(date_start, date_end)
    source_specs = [(_source_token(src), _normalize_source_filename(src)) for src in data_sources]

    results: dict[str, xr.Dataset | None] = {}
    for source_token, source_file in source_specs:
        per_date = []
        for d in dates:
            if tile is not None:
                ds = load_tile_nc(
                    dataset=dataset,
                    date=d,
                    tile=tile,
                    data_source=source_file,
                    cache_dir=cache_dir,
                )
            else:
                ds = load_bbox_nc(
                    dataset=dataset,
                    date=d,
                    bbox=bbox,
                    data_source=source_file,
                    cache_dir=cache_dir,
                )

            if ds is None:
                continue

            per_date.append(_ensure_time_dim(ds, d))

        if not per_date:
            results[source_token] = None
            continue

        if len(per_date) == 1:
            results[source_token] = per_date[0]
        else:
            results[source_token] = xr.concat(
                per_date,
                dim="time",
                data_vars="all",
                coords="minimal",
                compat="override",
                combine_attrs="override",
            ).sortby("time")

    return results


def load_bbox_nc(
    dataset,
    date: str,
    bbox: tuple,
    data_source: str,
    cache_dir=None,
) -> "xr.Dataset | None":
    """Load and merge tiles overlapping bbox for one requested data source.

    Args:
        dataset: TacoDataset from load_hf_dataset().
        date: Date string YYYY-MM-DD.
        bbox: Tuple (lon_min, lat_min, lon_max, lat_max).
        data_source: Source token or file name, e.g. 'l4_ssh' or 'l4_ssh.nc'.
        cache_dir: If provided, cache each tile to
            {cache_dir}/{date}/{tile}/{resolved_filename}.

    Returns:
        Merged xr.Dataset for the requested source, or None if no files found.
    """
    resolved_filename = _normalize_source_filename(data_source)
    files = query_files(dataset, date, bbox, {resolved_filename})
    if files.empty:
        return None

    tile_datasets = []
    for _, row in files.iterrows():
        tile = _tile_name_from_row(row)
        if cache_dir and tile:
            cached = Path(cache_dir) / date / tile / resolved_filename
            if cached.exists():
                tile_datasets.append(xr.open_dataset(cached, engine="h5netcdf"))
                continue

        _, ds = fetch_nc(row)

        if cache_dir and tile:
            cached.parent.mkdir(parents=True, exist_ok=True)
            ds.to_netcdf(cached)

        tile_datasets.append(ds)

    if not tile_datasets:
        return None
    if len(tile_datasets) == 1:
        return tile_datasets[0]
    return xr.combine_by_coords(tile_datasets, combine_attrs="override")


def load_bbox_swot_nc(
    dataset,
    date: str,
    bbox: tuple,
    cache_dir=None,
) -> "xr.Dataset | None":
    """Load and merge L3 SWOT tiles overlapping bbox for a given date.

    SWOT tiles carry per-tile auxiliary variables (``track_ids``,
    ``track_times``) indexed by a ``track`` dimension whose size differs
    between tiles.  These cannot be spatially merged, so they are dropped
    before combining.  All spatial data variables (``ssha_filtered``, etc.)
    are preserved.

    Args:
        dataset: TacoDataset from load_hf_dataset().
        date: Date string YYYY-MM-DD.
        bbox: Tuple (lon_min, lat_min, lon_max, lat_max).
        cache_dir: If provided, cache each tile to {cache_dir}/{date}/{tile}/l3_swot.nc.

    Returns:
        Merged xr.Dataset or None if no files found.
    """
    filename = "l3_swot.nc"
    files = query_files(dataset, date, bbox, {filename})
    if files.empty:
        return None
        
    tile_datasets = []
    for _, row in files.iterrows():
        tile = _tile_name_from_row(row)
        if cache_dir and tile:
            cached = Path(cache_dir) / date / tile / filename
            if cached.exists():
                tile_datasets.append(xr.open_dataset(cached, engine="h5netcdf"))
                continue

        _, ds = fetch_nc(row)

        if cache_dir and tile:
            cached.parent.mkdir(parents=True, exist_ok=True)
            ds.to_netcdf(cached)

        tile_datasets.append(ds)

    if not tile_datasets:
        return None
    if len(tile_datasets) == 1:
        return tile_datasets[0]

    # SWOT tiles carry a per-tile 'track' dimension (number of passes differs
    # by tile). Drop only variables that depend on non-spatial dims (e.g.
    # track), while preserving all gridded data on (time, lat, lon).
    allowed_dims = {"time", "lat", "lon"}

    cleaned = []
    for ds in tile_datasets:
        non_spatial_dims = [d for d in ds.dims if d not in allowed_dims]
        if non_spatial_dims:
            vars_to_drop = [
                v for v in ds.data_vars if set(ds[v].dims).intersection(non_spatial_dims)
            ]
            if vars_to_drop:
                ds = ds.drop_vars(vars_to_drop)
            # Remove remaining non-spatial dimensions and their coords.
            ds = ds.drop_dims(non_spatial_dims)
        cleaned.append(ds)

    return xr.combine_by_coords(cleaned, combine_attrs="override")


def download_files(
    files_df, max_workers: int = 8, as_list: bool = False
) -> dict[str, xr.Dataset] | list[xr.Dataset]:
    """Download all NetCDF rows in parallel using ThreadPoolExecutor.

    Args:
        files_df: DataFrame with file metadata rows.
        max_workers: Thread pool size.
        as_list: If True, return a list of datasets instead of a dict.
            Use for tiles that share the same filename (e.g. glorys.nc).

    Returns:
        Dict mapping filename to xr.Dataset, or list of xr.Dataset
        if as_list is True.
    """
    datasets: dict | list = {} if not as_list else []
    rows = [row for _, row in files_df.iterrows()]
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(fetch_nc, row): row for row in rows}
        for f in as_completed(futures):
            try:
                key, ds = f.result()
                if as_list:
                    datasets.append(ds)
                else:
                    datasets[key] = ds
                print(f"  ✓ {key}")
            except Exception as e:
                row = futures[f]
                print(f"  ✗ {row['l2:id']}: {e}")
    return datasets
