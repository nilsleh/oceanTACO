"""Swath and track gridding helpers for dataset formatting."""

from typing import Literal

import numpy as np
from scipy.stats import binned_statistic_2d


def create_regional_grid(bounds, resolution_km=2.0):
    """Create a regular grid for a region at specified resolution.

    Uses WGS 84 ellipsoid parameters (EPSG:4326) for degree-to-km conversion:
      - Semi-major axis a = 6378.137 km
      - Semi-minor axis b = 6356.752 km
      - Meridian arc (lat): pi * b / 180 ≈ 110.574 km/deg
      - Prime-vertical radius of curvature (lon) at mean_lat:
            N(lat) = a / sqrt(1 - e² sin²(lat))
            km/deg_lon = (pi / 180) * N(lat) * cos(lat)
    """
    # WGS 84 ellipsoid constants
    _WGS84_A = 6378.137  # semi-major axis, km
    _WGS84_B = 6356.752314245  # semi-minor axis, km
    _WGS84_E2 = 1 - (_WGS84_B / _WGS84_A) ** 2  # first eccentricity squared

    lon_min, lon_max = bounds["lon"]
    lat_min, lat_max = bounds["lat"]
    mean_lat_rad = np.radians((lat_min + lat_max) / 2)

    # Meridional arc: km per degree latitude (WGS 84)
    km_per_deg_lat = (np.pi / 180) * _WGS84_A * (1 - _WGS84_E2) / (
        1 - _WGS84_E2 * np.sin(mean_lat_rad) ** 2
    ) ** 1.5

    # Prime-vertical radius of curvature: km per degree longitude at mean_lat (WGS 84)
    N = _WGS84_A / np.sqrt(1 - _WGS84_E2 * np.sin(mean_lat_rad) ** 2)
    km_per_deg_lon = (np.pi / 180) * N * np.cos(mean_lat_rad)

    lat_res = resolution_km / km_per_deg_lat
    lon_res = resolution_km / km_per_deg_lon

    target_lons = np.arange(lon_min, lon_max, lon_res)
    target_lats = np.arange(lat_min, lat_max, lat_res)
    lon_edges = np.concatenate([target_lons - lon_res / 2, [target_lons[-1] + lon_res / 2]])
    lat_edges = np.concatenate([target_lats - lat_res / 2, [target_lats[-1] + lat_res / 2]])
    return target_lons, target_lats, lon_edges, lat_edges


def swath_intersects_region(lons, lats, bounds):
    """Check if swath data intersects a region."""
    lon_min, lon_max = bounds["lon"]
    lat_min, lat_max = bounds["lat"]
    if np.nanmax(lats) < lat_min or np.nanmin(lats) >= lat_max:
        return False
    if np.nanmax(lons) < lon_min or np.nanmin(lons) >= lon_max:
        return False
    return True


def bin_swath_to_grid(
    lons, lats, values, target_lons, target_lats, radius_of_influence=4000
):
    """Grid swath data using Binning + Gaussian Smoothing (Fast approximation of Gaussian resampling)."""
    from scipy.ndimage import gaussian_filter

    # Flatten input arrays
    lons_flat, lats_flat, vals_flat = lons.flatten(), lats.flatten(), values.flatten()

    # Filter invalid data
    valid = ~np.isnan(vals_flat) & ~np.isnan(lons_flat) & ~np.isnan(lats_flat)
    if valid.sum() == 0:
        return np.full((len(target_lats), len(target_lons)), np.nan)

    # Reconstruct edges from target centers (assuming regular grid)
    lon_res = (target_lons[-1] - target_lons[0]) / (len(target_lons) - 1)
    lat_res = (target_lats[-1] - target_lats[0]) / (len(target_lats) - 1)

    lon_edges = np.concatenate(
        [target_lons - lon_res / 2, [target_lons[-1] + lon_res / 2]]
    )
    lat_edges = np.concatenate(
        [target_lats - lat_res / 2, [target_lats[-1] + lat_res / 2]]
    )

    # 1. Fast Binning
    # Returns the mean value of points falling into each cell
    grid_data, _, _, _ = binned_statistic_2d(
        lons_flat[valid],
        lats_flat[valid],
        vals_flat[valid],
        statistic="mean",
        bins=[lon_edges, lat_edges],
    )
    grid_data = grid_data.T  # binned_statistic_2d returns (nx, ny)

    # 2. NaN-aware Gaussian Smoothing
    # Calculate sigma in pixels
    # radius_of_influence is in meters. Convert to pixels.
    # Approximation: 1 degree ~ 111km.
    lat_mean = np.mean(target_lats)
    meters_per_deg_lat = 111320
    meters_per_deg_lon = 111320 * np.cos(np.deg2rad(lat_mean))
    # Geometric mean of resolution in meters
    avg_res_m = np.sqrt(
        (lon_res * meters_per_deg_lon) ** 2 + (lat_res * meters_per_deg_lat) ** 2
    )

    # Sigma for gaussian filter
    # Matches PyResample logic: sigma = radius / 2
    sigma_pixels = (radius_of_influence / 2.0) / avg_res_m

    # Lower threshold to allow minimal smoothing for gap-filling
    if sigma_pixels < 1e-3:
        return grid_data

    # --- NaN-aware Gaussian Filter ---
    # Convolution: V_out = (V * K) / (M * K)
    # where V is values (0 for nan), M is mask (1 for valid), K is kernel

    mask = ~np.isnan(grid_data)
    data_filled = grid_data.copy()
    data_filled[~mask] = 0

    smoothed_data = gaussian_filter(
        data_filled, sigma=sigma_pixels, mode="constant", cval=0, truncate=4.0
    )
    smoothed_mask = gaussian_filter(
        mask.astype(float), sigma=sigma_pixels, mode="constant", cval=0, truncate=4.0
    )

    with np.errstate(invalid="ignore", divide="ignore"):
        result = smoothed_data / smoothed_mask

    # Mask out areas with too little data contribution (equivalent to radius check)
    # 1e-2 is an arbitrary low threshold to cut off the tails
    result[smoothed_mask < 1e-2] = np.nan

    return result


def bin_swath_to_grid_conservative(
    lons: np.ndarray,
    lats: np.ndarray,
    values: np.ndarray,
    target_lons: np.ndarray,
    target_lats: np.ndarray,
    method: Literal["mean", "median", "nearest"] = "mean",
    min_samples: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Conservative regridding of swath data to regular grid.

    Preserves native resolution by simple binning without smoothing.
    No gap-filling - cells without observations remain NaN.

    Parameters
    ----------
    lons, lats : array-like
        Coordinates of swath observations (can be 1D or 2D)
    values : array-like
        Data values at each observation point
    target_lons, target_lats : 1D arrays
        Regular grid cell centers
    method : str
        Aggregation method: "mean", "median", or "nearest"
    min_samples : int
        Minimum number of samples required per cell (default=1)

    Returns:
    -------
    grid_data : 2D array
        Regridded values, shape (len(target_lats), len(target_lons))
    grid_counts : 2D array
        Number of observations per cell
    """
    # Flatten inputs
    lons_flat = np.asarray(lons).flatten()
    lats_flat = np.asarray(lats).flatten()
    vals_flat = np.asarray(values).flatten()

    # Filter invalid data
    valid = np.isfinite(vals_flat) & np.isfinite(lons_flat) & np.isfinite(lats_flat)
    if valid.sum() == 0:
        empty = np.full((len(target_lats), len(target_lons)), np.nan)
        return empty, np.zeros_like(empty, dtype=np.int32)

    lons_v, lats_v, vals_v = lons_flat[valid], lats_flat[valid], vals_flat[valid]

    # Compute grid edges from centers
    lon_res = np.median(np.diff(target_lons))
    lat_res = np.median(np.diff(target_lats))

    lon_edges = np.concatenate(
        [
            [target_lons[0] - lon_res / 2],
            (target_lons[:-1] + target_lons[1:]) / 2,
            [target_lons[-1] + lon_res / 2],
        ]
    )
    lat_edges = np.concatenate(
        [
            [target_lats[0] - lat_res / 2],
            (target_lats[:-1] + target_lats[1:]) / 2,
            [target_lats[-1] + lat_res / 2],
        ]
    )

    # Bin the data
    if method == "nearest":
        statistic = "mean"
    else:
        statistic = method

    grid_data, _, _, _ = binned_statistic_2d(
        lons_v,
        lats_v,
        vals_v,
        statistic=statistic,
        bins=[lon_edges, lat_edges],
        expand_binnumbers=True,
    )
    grid_data = grid_data.T  # Shape: (nlat, nlon)

    # Count observations per cell
    grid_counts, _, _, _ = binned_statistic_2d(
        lons_v, lats_v, vals_v, statistic="count", bins=[lon_edges, lat_edges]
    )
    grid_counts = grid_counts.T.astype(np.int32)

    # Apply minimum sample threshold
    grid_data[grid_counts < min_samples] = np.nan

    return grid_data, grid_counts


def process_swot_track_to_grid(
    track_lons: np.ndarray,
    track_lats: np.ndarray,
    track_data: np.ndarray,
    target_lons: np.ndarray,
    target_lats: np.ndarray,
    resolution_km: float = 2.0,
) -> dict:
    """Process a single SWOT track onto a regular grid.

    This is a thin wrapper that ensures the grid resolution matches
    the native SWOT resolution to avoid aliasing.

    Parameters
    ----------
    track_lons, track_lats : 2D arrays
        Native SWOT coordinates (along-track × cross-track)
    track_data : 2D array
        SSH or other variable
    target_lons, target_lats : 1D arrays
        Target regular grid
    resolution_km : float
        Target grid resolution (should match native ~2km)

    Returns:
    -------
    dict with 'data', 'counts', 'lons', 'lats'
    """
    # Verify grid resolution is appropriate
    lon_res_deg = np.median(np.diff(target_lons))
    lat_res_deg = np.median(np.diff(target_lats))

    # Approximate resolution in km at mid-latitude
    mid_lat = np.mean(target_lats)
    lon_res_km = lon_res_deg * 111.32 * np.cos(np.radians(mid_lat))
    lat_res_km = lat_res_deg * 111.32
    avg_res_km = np.sqrt(lon_res_km * lat_res_km)

    if avg_res_km > resolution_km * 1.5:
        import warnings

        warnings.warn(
            f"Target grid resolution ({avg_res_km:.1f} km) is coarser than "
            f"native SWOT resolution ({resolution_km} km). This will cause aliasing."
        )

    grid_data, grid_counts = bin_swath_to_grid_conservative(
        track_lons,
        track_lats,
        track_data,
        target_lons,
        target_lats,
        method="mean",
        min_samples=1,
    )

    return {
        "data": grid_data,
        "counts": grid_counts,
        "lons": target_lons,
        "lats": target_lats,
    }
