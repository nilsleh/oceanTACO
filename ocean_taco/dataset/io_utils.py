"""Dataloading Utilities."""

import glob
import os
import re
from datetime import datetime, timedelta
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import xarray as xr
from shapely import wkt
from shapely.affinity import translate
from shapely.geometry import box

# ==============================================================================
# DATA LOADING FUNCTIONS
# ==============================================================================


def plot_swot_ssh(ds, query_bbox):
    """Plot SWOT L3 SSH anomaly data (filtered version).
    Handles antimeridian crossing by normalizing longitude values.

    Args:
        ds: xarray Dataset with SWOT L3 data
        query_box: lon_min, lon_max, lat_min, lat_max

    Returns:
        matplotlib figure and axes
    """
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    # Extract data
    lon = ds["lon"].values.copy()
    lat = ds["lat"].values
    ssha_filt = ds["ssha_filtered"].values

    # Fix antimeridian crossing: normalize longitudes to [0, 360] range
    lon[lon < 0] += 360

    # Query bounding box (from SQL query)
    bbox_width = query_bbox[3] - query_bbox[2]
    bbox_height = query_bbox[1] - query_bbox[0]

    # Create figure with cartopy projection
    fig, ax = plt.subplots(
        figsize=(12, 8), subplot_kw={"projection": ccrs.PlateCarree()}
    )

    ax.set_global()
    ax.add_feature(cfeature.LAND, facecolor="lightgray", edgecolor="none", zorder=0)

    im = ax.pcolormesh(
        lon,
        lat,
        ssha_filt,
        cmap="RdBu_r",
        vmin=-0.5,
        vmax=0.5,
        shading="auto",
        transform=ccrs.PlateCarree(),
        zorder=1,
    )

    ax.coastlines(linewidth=0.8, color="black", zorder=3)

    # Add query bounding box
    rect = Rectangle(
        (query_bbox[0], query_bbox[2]),
        bbox_width,
        bbox_height,
        linewidth=0.5,
        edgecolor="red",
        facecolor="none",
        transform=ccrs.PlateCarree(),
        zorder=4,
    )
    ax.add_patch(rect)

    ax.gridlines(draw_labels=True, alpha=0.3, linestyle="--", zorder=2)
    ax.set_xlabel("Longitude (°E)", fontsize=12)
    ax.set_ylabel("Latitude (°N)", fontsize=12)

    plt.colorbar(im, ax=ax, label="SSH Anomaly (m)", shrink=0.7)

    # Title with time coverage
    time_start = ds.attrs.get("time_coverage_start", "Unknown")
    plt.title(
        f"SWOT L3 Sea Surface Height Anomaly - {time_start}",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )

    plt.tight_layout()

    return fig, ax


def load_glorys_data(data_dir: str, date_str: str) -> xr.Dataset | None:
    """Load GLORYS data for a specific date."""
    glorys_files = glob.glob(f"{data_dir}/glorys/**/*{date_str}*.nc", recursive=True)
    if glorys_files:
        return xr.open_dataset(glorys_files[0])
    return None


def load_l4_ssh_data(data_dir: str, date_str: str) -> xr.Dataset | None:
    """Load SSH L4 data for a specific date."""
    l4_ssh_files = glob.glob(f"{data_dir}/l4_ssh/**/*{date_str}*.nc", recursive=True)
    if l4_ssh_files:
        return xr.open_dataset(l4_ssh_files[0])
    return None


def load_l4_sst_data(data_dir: str, date_str: str) -> xr.Dataset | None:
    """Load SST L4 data for a specific date."""
    sst_files = glob.glob(f"{data_dir}/l4_sst/**/*{date_str}*.nc", recursive=True)
    if sst_files:
        return xr.open_dataset(sst_files[0])
    return None


def load_l4_sss_data(data_dir: str, date_str: str) -> xr.Dataset | None:
    """Load SSS L4 data for a specific date."""
    sss_files = glob.glob(f"{data_dir}/l4_sss/**/*{date_str}*.nc", recursive=True)
    if sss_files:
        return xr.open_dataset(sss_files[0])
    return None


def load_l4_wind_data(data_dir, date_str):
    """Load L4 wind data for a given date.

    Args:
        data_dir: Root data directory
        date_str: Date string in YYYYMMDD format

    Returns:
        xarray.Dataset or None if no data found
    """
    file_path = os.path.join(data_dir, "l4_wind", f"l4_wind_daily_{date_str}.nc")
    return xr.open_dataset(file_path)


def load_sst_l3_data(data_dir: str, date_str: str) -> xr.Dataset | None:
    """Load SST L3 data for a specific date."""
    sst_l3_files = glob.glob(
        f"{data_dir}/sst_l3_infrared/**/*{date_str}*.nc", recursive=True
    )
    if sst_l3_files:
        return xr.open_dataset(sst_l3_files[0])
    return None


def load_smos_sss_data(data_dir: str, date_str: str) -> xr.Dataset | None:
    """Load SMOS SSS data for a specific date."""
    smos_files = glob.glob(f"{data_dir}/sss_l3_smos/**/*{date_str}*.nc", recursive=True)
    if smos_files:
        return xr.open_dataset(smos_files[0])
    return None


def _select_highest_version(files: list[str]) -> list[str]:
    """Keep only the highest version per base SWOT filename.
    Expects filenames ending with _PGC0_<NN>.nc (NN = version digits).
    """
    best = {}
    rx = re.compile(r"(.*_PGC0_)(\d+)\.nc$")
    for f in files:
        name = f.split("/")[-1]
        m = rx.match(name)
        if not m:
            # Keep as-is if pattern not matched (treat as unique)
            best[name] = f
            continue
        base, ver = m.group(1), int(m.group(2))
        key = base  # base identifies all versions of same file
        if key not in best or ver > best[key][0]:
            best[key] = (ver, f)
    # Extract file paths
    return sorted([v[1] if isinstance(v, tuple) else v for v in best.values()])


def load_l2_swot_data(
    data_dir: str,
    date_str: str,
    min_date: str = None,
    max_date: str = None,
    return_paths_only=False,
) -> list | None:
    """Load SWOT data for a specific date or date range."""
    if min_date and max_date:
        filtered_files = []
        current_date = datetime.strptime(min_date, "%Y%m%d")
        end_date = datetime.strptime(max_date, "%Y%m%d")

        while current_date <= end_date:
            date_str_search = current_date.strftime("%Y%m%d")
            daily_files = glob.glob(
                f"{data_dir}/l2_swot*/*{date_str_search}*_PGC0_*.nc", recursive=True
            )
            daily_files = _select_highest_version(daily_files)
            filtered_files.extend(daily_files)
            current_date += timedelta(days=1)

        if filtered_files:
            if return_paths_only:
                return filtered_files
            else:
                return [xr.open_dataset(f) for f in filtered_files]
    else:
        swot_files = glob.glob(
            f"{data_dir}/l2_swot/*{date_str}*_PGC0_*.nc", recursive=True
        )
        swot_files = _select_highest_version(swot_files)
        if swot_files:
            if return_paths_only:
                return swot_files
            else:
                return [xr.open_dataset(f) for f in swot_files]
    return None


def load_l3_swot_data(
    data_dir: str,
    date_str: str,
    min_date: str = None,
    max_date: str = None,
    return_paths_only=False,
) -> list | None:
    """Load L3 SWOT data for a specific date or date range."""
    if min_date and max_date:
        filtered_files = []
        current_date = datetime.strptime(min_date, "%Y%m%d")
        end_date = datetime.strptime(max_date, "%Y%m%d")

        while current_date <= end_date:
            date_str_search = current_date.strftime("%Y%m%d")
            daily_files = glob.glob(
                f"{data_dir}/l3_swot/*{date_str_search}*.nc", recursive=True
            )
            filtered_files.extend(daily_files)
            current_date += timedelta(days=1)

        if filtered_files:
            if return_paths_only:
                return filtered_files
            else:
                return [xr.open_dataset(f) for f in filtered_files]
    else:
        l3_swot_files = glob.glob(f"{data_dir}/l3_swot/*{date_str}*.nc", recursive=True)
        if l3_swot_files:
            if return_paths_only:
                return l3_swot_files
            else:
                return [xr.open_dataset(f) for f in l3_swot_files]
    return None


def load_l3_ssh_data(
    data_dir: str,
    date_str: str,
    min_date: str = None,
    max_date: str = None,
    return_paths_only=False,
) -> list | None:
    """Load L3 SSH data for a specific date or date range."""
    if min_date and max_date:
        filtered_files = []
        current_date = datetime.strptime(min_date, "%Y%m%d")
        end_date = datetime.strptime(max_date, "%Y%m%d")

        while current_date <= end_date:
            date_str_search = current_date.strftime("%Y%m%d")
            daily_files = glob.glob(
                f"{data_dir}/l3_ssh/**/*{date_str_search}*.nc", recursive=True
            )
            filtered_files.extend(daily_files)
            current_date += timedelta(days=1)

        if filtered_files:
            if return_paths_only:
                return filtered_files
            else:
                return [xr.open_dataset(f) for f in filtered_files]
    else:
        l3_ssh_files = glob.glob(
            f"{data_dir}/l3_ssh/**/*{date_str}*.nc", recursive=True
        )
        if l3_ssh_files:
            if return_paths_only:
                return l3_ssh_files
            else:
                return [xr.open_dataset(f) for f in l3_ssh_files]
    return None


def load_all_data(data_dir: str, date_str: str) -> dict[str, Any]:
    """Load all data types for a specific date or date range."""
    # make min date one day before the date_str
    date_obj = datetime.strptime(date_str, "%Y%m%d")
    min_date = (date_obj - timedelta(days=1)).strftime("%Y%m%d")
    max_date = (date_obj + timedelta(days=1)).strftime("%Y%m%d")

    data_dict = {
        "glorys": load_glorys_data(data_dir, date_str),
        "swot": load_l3_swot_data(data_dir, date_str, min_date, max_date),
        "l3_ssh": load_l3_ssh_data(data_dir, date_str, min_date, max_date),
        "l4_ssh": load_l4_ssh_data(data_dir, date_str),
        "l4_sst": load_l4_sst_data(data_dir, date_str),
        "l4_sss": load_l4_sss_data(data_dir, date_str),
        "sst_l3": load_sst_l3_data(data_dir, date_str),
        "smos_sss": load_smos_sss_data(data_dir, date_str),
    }

    # Count loaded datasets
    loaded_count = 0
    for k, v in data_dict.items():
        if v is not None:
            if isinstance(v, list):
                loaded_count += len(v)
                if k in ["swot", "l3_ssh"] and min_date and max_date:
                    print(f"✓ {k}: {len(v)} files across date range")
            else:
                loaded_count += 1

    print(f"Successfully loaded {loaded_count} total datasets")
    return data_dict


def read_tif_with_metadata(filepath, bbox: tuple[float, ...] | None = None):
    """Read TIF file and return data, transform, and metadata."""
    with rasterio.open(filepath) as src:
        if bbox:
            # Convert bbox to rasterio format (minx, miny, maxx, maxy)
            window = rasterio.windows.from_bounds(
                bbox[0], bbox[2], bbox[1], bbox[3], src.transform
            )
            data = src.read(1, window=window)  # Read first band with window
            transform = src.window_transform(window)
        else:
            data = src.read(1)  # Read first band
            transform = src.transform

        nodata = src.nodata

        # Get metadata from tags
        tags = src.tags(1)

        # Mask nodata values
        if nodata is not None:
            if np.isnan(nodata):
                data = np.ma.masked_invalid(data)
            else:
                data = np.ma.masked_equal(data, nodata)

        # Generate coordinate arrays from transform
        height, width = data.shape

        # Create coordinate arrays using the transform
        x_coords = np.arange(width) * transform.a + transform.c + transform.a / 2
        y_coords = np.arange(height) * transform.e + transform.f + transform.e / 2

        return data, x_coords, y_coords, tags


def _parse_geom_wkt_from_row(row):
    """Parse geometry from row:
      - Prefer 'istac:spatial' WKT (POLYGON or MULTILINESTRING for SWOT/L3)
      - Fallback to 'stac:centroid' POINT
    Returns shapely geometry or None.
    """
    geom_txt = row.get("istac:spatial")
    if isinstance(geom_txt, str) and geom_txt and geom_txt.lower() != "none":
        try:
            return wkt.loads(geom_txt)
        except Exception:
            pass

    centroid_txt = row.get("stac:centroid")
    if (
        isinstance(centroid_txt, str)
        and centroid_txt
        and centroid_txt.lower() != "none"
    ):
        try:
            return wkt.loads(centroid_txt)
        except Exception:
            pass

    return None


def to_geodf(df: pd.DataFrame) -> gpd.GeoDataFrame:
    """Convert DataFrame to GeoDataFrame (EPSG:4326) using ISTAC geometry when available,
    else centroid POINT. Rows with no geometry become None.
    """
    geom = df.apply(_parse_geom_wkt_from_row, axis=1)
    gdf = gpd.GeoDataFrame(df.copy(), geometry=geom, crs="EPSG:4326")
    return gdf


def _build_bbox_polys(lon_min: float, lon_max: float, lat_min: float, lat_max: float):
    """Create a list of shapely boxes representing the requested region.
    If the bbox crosses the dateline (lon_min > lon_max), split into two boxes.
    """
    if lon_min <= lon_max:
        return [box(lon_min, lat_min, lon_max, lat_max)]
    return [
        box(lon_min, lat_min, 180.0, lat_max),
        box(-180.0, lat_min, lon_max, lat_max),
    ]


def spatial_mask_bbox(
    gdf: gpd.GeoDataFrame,
    lon_min: float,
    lon_max: float,
    lat_min: float,
    lat_max: float,
) -> pd.Series:
    """Spatial intersection test using GeoPandas:
      - Tests original geometry and versions shifted by ±360 degrees in longitude
        to accommodate inputs with 0–360 longitudes.
      - Handles dateline-crossing bboxes by splitting them.

    Returns a boolean mask aligned with gdf.index.
    """
    bboxes = _build_bbox_polys(lon_min, lon_max, lat_min, lat_max)

    # Base geometries
    g0 = gdf.geometry

    # Shifted by ±360 in lon to handle 0–360 vs. -180–180
    g_pos = g0.apply(lambda g: translate(g, xoff=360) if g is not None else None)
    g_neg = g0.apply(lambda g: translate(g, xoff=-360) if g is not None else None)

    mask = pd.Series(False, index=gdf.index)
    for b in bboxes:
        mask |= g0.intersects(b).fillna(False)
        mask |= g_pos.intersects(b).fillna(False)
        mask |= g_neg.intersects(b).fillna(False)

    return mask
