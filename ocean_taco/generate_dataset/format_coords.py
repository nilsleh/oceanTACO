"""Coordinate, time, and region helpers for dataset formatting."""

import numpy as np
import pandas as pd
import xarray as xr
from shapely.geometry import box


def lon_to_180(a):
    """Normalize longitudes to [-180, 180]."""
    a = np.asarray(a, dtype=float)
    out = ((a + 180.0) % 360.0) - 180.0
    out[out == 180.0] = -180.0
    return out


def normalize_coords(ds):
    """Normalize lon to [-180,180] and rename coords to 'lon'/'lat'."""
    lon_key = (
        "longitude"
        if "longitude" in ds.coords
        else ("lon" if "lon" in ds.coords else None)
    )
    lat_key = (
        "latitude"
        if "latitude" in ds.coords
        else ("lat" if "lat" in ds.coords else None)
    )
    if lon_key is None or lat_key is None:
        return ds
    lon_vals = lon_to_180(ds[lon_key].values)
    ds = ds.assign_coords({lon_key: xr.DataArray(lon_vals, dims=ds[lon_key].dims)})
    rename_map = {}
    if lon_key != "lon":
        rename_map[lon_key] = "lon"
    if lat_key != "lat":
        rename_map[lat_key] = "lat"
    if rename_map:
        ds = ds.rename(rename_map)
    if "lon" in ds.coords and ds["lon"].ndim == 1:
        ds = ds.sortby("lon")
    return ds


def timestamp_to_microseconds(ts):
    """Convert timestamp to microseconds since epoch for ISTAC."""
    if ts is None:
        return None
    if isinstance(ts, pd.Timestamp):
        return int(ts.timestamp() * 1_000_000)
    elif isinstance(ts, int | float):
        # Assume already in seconds, convert to microseconds
        return int(ts * 1_000_000)
    return None


def posix_range_from_time(time_vals):
    """Get POSIX timestamp range (in seconds) from time array or scalar."""
    if time_vals is None:
        return None, None

    # Handle scalar timestamp
    if isinstance(time_vals, pd.Timestamp | np.datetime64):
        ts = pd.Timestamp(time_vals)
        if pd.isna(ts):
            return None, None
        return float(ts.timestamp()), float(ts.timestamp())

    # Handle array
    try:
        s = pd.to_datetime(np.asarray(time_vals).flatten(), errors="coerce")
        s = pd.Series(s)
        s = s[~s.isna()]
        if len(s) == 0:
            return None, None
        return float(s.min().timestamp()), float(s.max().timestamp())
    except Exception as e:
        raise ValueError("Failed to parse time values in posix_range_from_time") from e


def compute_resolution(ds, bbox=None):
    """Compute resolution in degrees and km from dataset coordinates.

    Args:
        ds: xarray Dataset with 'lat' and 'lon' coordinates
        bbox: [lon_min, lat_min, lon_max, lat_max] - optional, will be computed from ds if not provided

    Returns:
        dict with 'resolution_deg_lat', 'resolution_deg_lon', 'resolution_km_lat', 'resolution_km_lon'
    """
    # Return None for all resolution fields if no dataset
    none_result = {
        "resolution_deg_lat": None,
        "resolution_deg_lon": None,
        "resolution_km_lat": None,
        "resolution_km_lon": None,
    }

    if ds is None:
        return none_result

    # Extract coordinate arrays
    if "lat" not in ds.coords or "lon" not in ds.coords:
        return none_result

    lat_coord = ds["lat"].values
    lon_coord = ds["lon"].values

    # Handle 1D coordinates (regular grids)
    if lat_coord.ndim == 1 and lon_coord.ndim == 1:
        n_lat = len(lat_coord)
        n_lon = len(lon_coord)
        lat_min, lat_max = float(lat_coord.min()), float(lat_coord.max())
        lon_min, lon_max = float(lon_coord.min()), float(lon_coord.max())
    # Handle 2D coordinates (curvilinear grids)
    elif lat_coord.ndim == 2 and lon_coord.ndim == 2:
        n_lat = lat_coord.shape[0]
        n_lon = lat_coord.shape[1]
        lat_min, lat_max = float(lat_coord.min()), float(lat_coord.max())
        lon_min, lon_max = float(lon_coord.min()), float(lon_coord.max())
    else:
        return none_result

    # Avoid division by zero or invalid grid
    if n_lat <= 1 or n_lon <= 1:
        return none_result

    # Calculate resolution in degrees
    # For cell-centered grids with n points from min to max:
    # resolution = (max - min) / (n - 1)
    lat_extent = abs(lat_max - lat_min)
    lon_extent = abs(lon_max - lon_min)

    resolution_deg_lat = lat_extent / (n_lat - 1)
    resolution_deg_lon = lon_extent / (n_lon - 1)

    # Convert to km
    # 1 degree latitude ≈ 110.574 km (constant)
    resolution_km_lat = resolution_deg_lat * 110.574

    # 1 degree longitude varies with latitude: ~111.32 * cos(lat) km
    # Use the mean latitude for conversion
    mean_lat = (lat_min + lat_max) / 2
    resolution_km_lon = resolution_deg_lon * 111.32 * np.cos(np.radians(abs(mean_lat)))

    return {
        "resolution_deg_lat": float(resolution_deg_lat),
        "resolution_deg_lon": float(resolution_deg_lon),
        "resolution_km_lat": float(resolution_km_lat),
        "resolution_km_lon": float(resolution_km_lon),
    }


def point_in_region(lon, lat, bounds):
    """Check if point is in region."""
    lon_min, lon_max = bounds["lon"]
    lat_min, lat_max = bounds["lat"]
    return (lat_min <= lat < lat_max) and (lon_min <= lon < lon_max)


def split_gridded_into_regions(ds, regions):
    """Split a gridded dataset into regional subsets."""
    results = {}
    lons = ds["lon"].values
    lats = ds["lat"].values

    for region_name, bounds in regions.items():
        lon_min, lon_max = bounds["lon"]
        lat_min, lat_max = bounds["lat"]

        lon_mask = (lons >= lon_min) & (lons < lon_max)
        lat_mask = (lats >= lat_min) & (lats < lat_max)

        if not (lon_mask.any() and lat_mask.any()):
            results[region_name] = {
                "dataset": None,
                "bbox": None,
                "geometry": None,
                "time_range": (None, None),
                "intersects": False,
            }
            continue

        regional_ds = ds.isel(lon=lon_mask, lat=lat_mask)
        reg_lons, reg_lats = regional_ds["lon"].values, regional_ds["lat"].values
        bbox = [
            float(reg_lons.min()),
            float(reg_lats.min()),
            float(reg_lons.max()),
            float(reg_lats.max()),
        ]
        time_range = (
            posix_range_from_time(regional_ds["time"].values)
            if "time" in regional_ds.coords
            else (None, None)
        )

        results[region_name] = {
            "dataset": regional_ds,
            "bbox": bbox,
            "geometry": box(*bbox).wkb,
            "time_range": time_range,
            "intersects": True,
        }

    return results
