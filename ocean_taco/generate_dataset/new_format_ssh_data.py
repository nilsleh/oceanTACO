#!/usr/bin/env python3
"""Format SSH data for TACO dataset creation.

Processes ocean data products and splits them into 8 spatial regions.
SWOT swath data is gridded to regular 2km resolution.
"""

import argparse
import glob
import logging
import multiprocessing as mp
import os
from datetime import datetime, timedelta
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import binned_statistic_2d
from shapely.geometry import box

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# =============================================================================
# Spatial Regions - 8 equal rectangles with descriptive names
# =============================================================================

SPATIAL_REGIONS = {
    'SOUTH_PACIFIC_WEST': {'lat': (-90, 0), 'lon': (-180, -90)},
    'SOUTH_ATLANTIC':     {'lat': (-90, 0), 'lon': (-90, 0)},
    'SOUTH_INDIAN':       {'lat': (-90, 0), 'lon': (0, 90)},
    'SOUTH_PACIFIC_EAST': {'lat': (-90, 0), 'lon': (90, 180)},
    'NORTH_PACIFIC_WEST': {'lat': (0, 90), 'lon': (-180, -90)},
    'NORTH_ATLANTIC':     {'lat': (0, 90), 'lon': (-90, 0)},
    'NORTH_INDIAN':       {'lat': (0, 90), 'lon': (0, 90)},
    'NORTH_PACIFIC_EAST': {'lat': (0, 90), 'lon': (90, 180)},
}

NETCDF_ENCODINGS = {
    "glorys": {"zlib": True, "complevel": 4},
    "l4_ssh": {"zlib": True, "complevel": 4},
    "l4_sst": {"zlib": True, "complevel": 4},
    "l4_sss": {"zlib": True, "complevel": 4},
    "l4_wind": {"zlib": True, "complevel": 4},
    "l3_ssh": {"zlib": True, "complevel": 4},
    "l3_sst": {"zlib": True, "complevel": 4},
    "l3_swot": {"zlib": True, "complevel": 4},
    "l3_sss_smos_asc": {"zlib": True, "complevel": 4},
    "l3_sss_smos_desc": {"zlib": True, "complevel": 4},
    "argo": {"zlib": True, "complevel": 4},
}

def get_variable_encoding(var_name):
    """Get NetCDF compression and packing encoding for a variable."""
    # Base encoding with high compression
    base = {"zlib": True, "complevel": 5, "_FillValue": -32767}
    
    # Packing: int16 = 2 bytes vs float32 = 4 bytes (50% size reduction)
    v = var_name.lower()
    
    # EXCLUSIONS: Do NOT pack these variables (keep as float32 or int per netcdf defaults)
    # Time deltas, angles, masks, counts that don't fit
    if any(x in v for x in ['time', 'date', 'mask', 'quality', 'flag', 'status', 'source', 'angle']):
         # If it looks like a flag/status, maybe int16 is fine, but safe default is float32 for safety
         if any(x in v for x in ['flag', 'status', 'source', 'count', 'number']):
             return {**base, "dtype": "int16"} 
         return {**base, "dtype": "float32"}

    # SSH / SLA / MDT / ZOS / ADT (Meters)
    # Range typically -2m to 2m, but MDT can be -4m in some places (geoid-related).
    # Previous scale 0.0001 -> range +/- 3.2m (Too small for MDT=-3.99)
    # New scale 0.0005 -> range +/- 16.38m. Precision 0.5 mm (needed for SSHA ~10m).
    if any(x in v for x in ['ssh', 'sla', 'mdt', 'zos', 'adt']):
        return {**base, "dtype": "int16", "scale_factor": 0.0005, "add_offset": 0.0}
        
    # SST / Thetao (Degrees Celsius)
    # Range typically -2 to 35 C.
    # Precision: 0.001 C
    if any(x in v for x in ['sst', 'thetao', 'temperature']):
        return {**base, "dtype": "int16", "scale_factor": 0.001, "add_offset": 20.0}
        
    # SSS / Salinity (PSU)
    # Some products can exceed 45 PSU (observed up to ~62.7 in SMOS fields).
    # Precision: 0.001 PSU
    # Use exact match or start/end to avoid catching 'so' within 'source' or 'solar'.
    # 'sos' = L4 SSS Copernicus variable; 'salinity' in v catches 'sea_surface_salinity' (L3 SMOS).
    if v in ["sss", "so", "sos", "salinity"] or "salinity" in v or v.startswith("sss_") or v.endswith("_sss"):
        return {**base, "dtype": "int16", "scale_factor": 0.002, "add_offset": 30.0}
        
    # Ocean Velocities (m/s)
    # Range -5 to 5 m/s is sufficient for ocean currents (uo, vo)
    # Precision: 0.001 m/s
    if any(x in v for x in ['uo', 'vo', 'current']):
        return {**base, "dtype": "int16", "scale_factor": 0.001, "add_offset": 0.0}

    # Wind Speed (m/s)
    # Hurricanes can exceed 80 m/s; scale_factor=0.01 gives ±327.6 m/s range, 0.01 m/s precision.
    if any(x in v for x in ['wind', 'speed']):
         return {**base, "dtype": "int16", "scale_factor": 0.01, "add_offset": 0.0}

    # Metadata / counts (keep as is or specific integer types)
    if 'n_obs' in v or 'count' in v:
        # NO scale_factor/add_offset for actual integers to prevents float promotion/casting errors
        return {**base, "dtype": "int16", "_FillValue": -1}
    if 'track' in v or 'overlap' in v or 'mask' in v or 'num' in v:
         # NO scale_factor/add_offset for actual integers
         return {**base, "dtype": "int8", "_FillValue": -1}
         
    # Default for unknown floats
    return {**base, "dtype": "float32"} # Fallback to float32 with compression


L3_SWOT_VARS = ("ssha_filtered", "ssha_unfiltered", "mdt")
PRIMARY_L3_SSH_VAR = "sla_filtered"
PRIMARY_L3_SWOT_VAR = "ssha_filtered"

# =============================================================================
# Coordinate Helpers
# =============================================================================

def lon_to_180(a):
    """Normalize longitudes to [-180, 180]."""
    a = np.asarray(a, dtype=float)
    out = ((a + 180.0) % 360.0) - 180.0
    out[out == 180.0] = -180.0
    return out


def normalize_coords(ds):
    """Normalize lon to [-180,180] and rename coords to 'lon'/'lat'."""
    lon_key = "longitude" if "longitude" in ds.coords else ("lon" if "lon" in ds.coords else None)
    lat_key = "latitude" if "latitude" in ds.coords else ("lat" if "lat" in ds.coords else None)
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


def check_encoding_safety(ds, encoding_dict):
    """Check if data fits within the encoding range to avoid silent clipping."""
    for var_name, enc in encoding_dict.items():
        if var_name not in ds:
            continue
            
        # Only check integer packed variables
        if enc.get("dtype") != "int16":
            continue
            
        data = ds[var_name].values
        if np.isnan(data).all():
            continue
            
        scale = enc.get("scale_factor", 1.0)
        offset = enc.get("add_offset", 0.0)
        
        # Calculate representable range
        # int16 range is -32768 to 32767. 
        # _FillValue is usually -32767 or similar, so usable range is approx +/- 32000 steps
        
        # Max pos val: 32767 * scale + offset
        # Min neg val: -32767 * scale + offset (-32768 often reserved)
        
        max_rep = 32760 * scale + offset
        min_rep = -32760 * scale + offset
        
        act_max = np.nanmax(data)
        act_min = np.nanmin(data)
        
        if act_max > max_rep or act_min < min_rep:
            logging.warning(f"⚠️  Variable '{var_name}' range [{act_min:.2f}, {act_max:.2f}] exceeds encoding limits [{min_rep:.2f}, {max_rep:.2f}]. Data will be clipped!")
 

def clear_encoding(ds):
    """Clear all encoding from dataset."""
    ds.encoding.clear()
    for var in list(ds.data_vars) + list(ds.coords):
        if hasattr(ds[var], "encoding"):
            ds[var].encoding.clear()
    return ds


def timestamp_to_microseconds(ts):
    """Convert timestamp to microseconds since epoch for ISTAC."""
    if ts is None:
        return None
    if isinstance(ts, pd.Timestamp):
        return int(ts.timestamp() * 1_000_000)
    elif isinstance(ts, (int, float)):
        # Assume already in seconds, convert to microseconds
        return int(ts * 1_000_000)
    return None


def posix_range_from_time(time_vals):
    """Get POSIX timestamp range (in seconds) from time array or scalar."""
    if time_vals is None:
        return None, None
    
    # Handle scalar timestamp
    if isinstance(time_vals, (pd.Timestamp, np.datetime64)):
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
    except Exception:
        return None, None


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
        'resolution_deg_lat': None,
        'resolution_deg_lon': None,
        'resolution_km_lat': None,
        'resolution_km_lon': None,
    }
    
    if ds is None:
        return none_result
    
    # Extract coordinate arrays
    if 'lat' not in ds.coords or 'lon' not in ds.coords:
        return none_result
    
    lat_coord = ds['lat'].values
    lon_coord = ds['lon'].values
    
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
        'resolution_deg_lat': float(resolution_deg_lat),
        'resolution_deg_lon': float(resolution_deg_lon),
        'resolution_km_lat': float(resolution_km_lat),
        'resolution_km_lon': float(resolution_km_lon),
    }


# =============================================================================
# Region Splitting
# =============================================================================

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
                "dataset": None, "bbox": None, "geometry": None,
                "time_range": (None, None), "intersects": False
            }
            continue
        
        regional_ds = ds.isel(lon=lon_mask, lat=lat_mask)
        reg_lons, reg_lats = regional_ds["lon"].values, regional_ds["lat"].values
        bbox = [float(reg_lons.min()), float(reg_lats.min()), 
                float(reg_lons.max()), float(reg_lats.max())]
        time_range = posix_range_from_time(regional_ds["time"].values) if "time" in regional_ds.coords else (None, None)
        
        results[region_name] = {
            "dataset": regional_ds,
            "bbox": bbox,
            "geometry": box(*bbox).wkb,
            "time_range": time_range,
            "intersects": True,
        }
    
    return results


# =============================================================================
# SWOT Gridding
# =============================================================================

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

    lon_min, lon_max = bounds['lon']
    lat_min, lat_max = bounds['lat']
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


def bin_swath_to_grid(lons, lats, values, target_lons, target_lats, radius_of_influence=4000):
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
    
    lon_edges = np.concatenate([target_lons - lon_res/2, [target_lons[-1] + lon_res/2]])
    lat_edges = np.concatenate([target_lats - lat_res/2, [target_lats[-1] + lat_res/2]])

    # 1. Fast Binning
    # Returns the mean value of points falling into each cell
    grid_data, _, _, _ = binned_statistic_2d(
        lons_flat[valid], lats_flat[valid], vals_flat[valid],
        statistic='mean', bins=[lon_edges, lat_edges]
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
    avg_res_m = np.sqrt((lon_res * meters_per_deg_lon)**2 + (lat_res * meters_per_deg_lat)**2)
    
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
    
    smoothed_data = gaussian_filter(data_filled, sigma=sigma_pixels, mode='constant', cval=0, truncate=4.0)
    smoothed_mask = gaussian_filter(mask.astype(float), sigma=sigma_pixels, mode='constant', cval=0, truncate=4.0)
    
    with np.errstate(invalid='ignore', divide='ignore'):
        result = smoothed_data / smoothed_mask
    
    # Mask out areas with too little data contribution (equivalent to radius check)
    # 1e-2 is an arbitrary low threshold to cut off the tails
    result[smoothed_mask < 1e-2] = np.nan
    
    return result




# =============================================================================
# Data Loading Functions
# =============================================================================

def load_glorys_data(data_dir, date_str):
    """Load GLORYS data."""
    year, month = date_str[:4], date_str[4:6]
    # /p/project1/hai_uqmethodbox/data/new_ssh_dataset/glorys/GLOBAL_MULTIYEAR_PHY_001_030/
    glorys_dir = os.path.join(data_dir, "glorys", "GLOBAL_MULTIYEAR_PHY_001_030",
                              "cmems_mod_glo_phy_my_0.083deg_P1D-m_202311", year, month)
    pattern = os.path.join(glorys_dir, f"mercatorglorys12v1_gl12_mean_{date_str}_R*.nc")
    files = glob.glob(pattern)
    return xr.open_dataset(files[0]) if files else None


def load_l4_ssh_data(data_dir, date_str):
    """Load L4 SSH data."""
    date_obj = datetime.strptime(date_str, "%Y%m%d")
    year, month = date_str[:4], date_str[4:6]
    # if date_obj < datetime(2025, 5, 1):
    #     subdir = "SEALEVEL_GLO_PHY_CLIMATE_L4_MY_008_057/c3s_obs-sl_glo_phy-ssh_my_twosat-l4-duacs-0.25deg_P1D_202411"
    #     pattern = f"dt_global_twosat_phy_l4_{date_str}_vDT*.nc"
    # else:
    #     subdir = "SEALEVEL_GLO_PHY_L4_NRT_008_046/cmems_obs-sl_glo_phy-ssh_nrt_allsat-l4-duacs-0.125deg_P1D_202506"
    #     pattern = f"nrt_global_allsat_phy_l4_{date_str}_*.nc"
    # /p/project1/hai_uqmethodbox/data/new_ssh_dataset/l4_ssh/SEALEVEL_GLO_PHY_CLIMATE_L4_MY_008_057/c3s_obs-sl_glo_phy-ssh_my_twosat-l4-duacs-0.25deg_P1D_202411
    # /p/project1/hai_uqmethodbox/data/new_ssh_dataset/l4_ssh/SEALEVEL_GLO_PHY_CLIMATE_L4_MY_008_057/c3s_obs-sl_glo_phy-ssh_my_twosat-l4-duacs-0.25deg_P1D_202411/2023/06/dt_global_twosat_phy_l4_20230602_vDT2024.nc
    subdir = "SEALEVEL_GLO_PHY_L4_MY_008_047/cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P1D_202411"
    pattern = f"dt_global_allsat_phy_l4_{date_str}_*.nc"
    l4_dir = os.path.join(data_dir, "l4_ssh", subdir, year, month)
    files = glob.glob(os.path.join(l4_dir, pattern))
    return xr.open_dataset(files[0]) if files else None


def load_l4_sst_data(data_dir, date_str):
    """Load L4 SST data from REP or NRT product depending on date.

    Download splits at 2024-01-16:
      REP: METOFFICE-GLO-SST-L4-REP-OBS-SST  (product SST_GLO_SST_L4_REP_OBSERVATIONS_010_011)
      NRT: METOFFICE-GLO-SST-L4-NRT-OBS-SST-V2 (product SST_GLO_SST_L4_NRT_OBSERVATIONS_010_001)
    Copernicusmarine creates:
      l4_sst/<PRODUCT>/<DATASET_ID>_<ver>/YYYY/MM/<file>.nc
    """
    year, month = date_str[:4], date_str[4:6]
    date_obj = datetime.strptime(date_str, '%Y%m%d')
    nrt_start = datetime(2024, 1, 16)

    if date_obj < nrt_start:
        # REP product
        product = 'SST_GLO_SST_L4_REP_OBSERVATIONS_010_011'
        dataset_glob = 'METOFFICE-GLO-SST-L4-REP-OBS-SST_*'
        fname_pattern = f'{date_str}*-UKMO-L4_GHRSST-SSTfnd-OSTIA-GLOB*REP*.nc'
    else:
        # NRT product
        product = 'SST_GLO_SST_L4_NRT_OBSERVATIONS_010_001'
        dataset_glob = 'METOFFICE-GLO-SST-L4-NRT-OBS-SST*_*'
        fname_pattern = f'{date_str}*-UKMO-L4_GHRSST-SSTfnd-OSTIA-GLOB*.nc'

    dataset_dirs = glob.glob(os.path.join(data_dir, 'l4_sst', product, dataset_glob))
    if not dataset_dirs:
        return None
    dataset_dir = sorted(dataset_dirs)[-1]
    l4_dir = os.path.join(dataset_dir, year, month)
    files = glob.glob(os.path.join(l4_dir, fname_pattern))
    return xr.open_dataset(files[0]) if files else None


def load_l4_sss_data(data_dir, date_str):
    """Load L4 SSS data from MY/NRT products with robust fallback.

    In practice, MY covers historical years (including 2021-2022 in this
    dataset), while NRT may start later. To avoid silent coverage gaps, this
    loader checks both products and returns the first match for the requested
    date.
    """
    year, month = date_str[:4], date_str[4:6]
    date_obj = datetime.strptime(date_str, '%Y%m%d')

    product = 'MULTIOBS_GLO_PHY_S_SURFACE_MYNRT_015_013'
    fname_pattern = f'*daily_{date_str}T*.nc'

    # Prefer MY for historical range, NRT for latest dates, but always fallback.
    if date_obj <= datetime(2022, 12, 31):
        dataset_globs = [
            'cmems_obs-mob_glo_phy-sss_my_multi_P1D_*',
            'cmems_obs-mob_glo_phy-sss_nrt_multi_P1D_*',
        ]
    else:
        dataset_globs = [
            'cmems_obs-mob_glo_phy-sss_nrt_multi_P1D_*',
            'cmems_obs-mob_glo_phy-sss_my_multi_P1D_*',
        ]

    for dataset_glob in dataset_globs:
        dataset_dirs = glob.glob(os.path.join(data_dir, 'l4_sss', product, dataset_glob))
        if not dataset_dirs:
            continue
        for dataset_dir in sorted(dataset_dirs, reverse=True):
            l4_dir = os.path.join(dataset_dir, year, month)
            files = sorted(glob.glob(os.path.join(l4_dir, fname_pattern)))
            if files:
                return xr.open_dataset(files[0])

    return None


def load_l4_wind_data(data_dir, date_str):
    """Load L4 wind data."""
    fpath = os.path.join(data_dir, "l4_wind", f"l4_wind_daily_{date_str}.nc")
    return xr.open_dataset(fpath) if os.path.exists(fpath) else None


def load_l3_sst_data(data_dir, date_str):
    """Load L3 SST data from the MY ODYSSEA product.

    Download uses dataset_id='cmems_obs-sst_glo_phy_my_l3s_P1D-m' which
    belongs to product SST_GLO_PHY_L3S_MY_010_039.  Copernicusmarine creates:
      l3_sst/SST_GLO_PHY_L3S_MY_010_039/cmems_obs-sst_glo_phy_my_l3s_P1D-m_<ver>/YYYY/MM/<file>.nc
    The version suffix (e.g. _202311) is resolved via glob so it doesn't
    need to be hardcoded.
    """
    year, month = date_str[:4], date_str[4:6]
    # Glob for the version-suffixed dataset directory
    dataset_dirs = glob.glob(os.path.join(
        data_dir, 'l3_sst', 'SST_GLO_PHY_L3S_MY_010_039',
        'cmems_obs-sst_glo_phy_my_l3s_P1D-m_*'))
    if not dataset_dirs:
        return None
    # Expect exactly one match; take the latest alphabetically if multiple
    dataset_dir = sorted(dataset_dirs)[-1]
    l3_dir = os.path.join(dataset_dir, year, month)
    pattern = os.path.join(l3_dir, f'{date_str}*-IFR-L3S_GHRSST-*.nc')
    files = glob.glob(pattern)
    return xr.open_dataset(files[0]) if files else None


def load_l3_swot_data(data_dir, date_str, return_paths_only=False):
    """Load L3 SWOT data files."""
    pattern = os.path.join(data_dir, "l3_swot", f"SWOT_L3_LR_SSH_Basic_*_{date_str}T*_v*.nc")
    files = glob.glob(pattern)
    if return_paths_only:
        return files if files else None
    return [xr.open_dataset(f) for f in files] if files else None


def load_l3_ssh_data(data_dir, date_str, return_paths_only=False):
    """Load L3 SSH along-track data files."""
    pattern = os.path.join(data_dir, "l3_ssh", "**", f"*{date_str}*.nc")
    candidates = glob.glob(pattern, recursive=True)
    
    files = []
    if candidates:
        import re
        # Strict matching: date_str must be the measurement date 
        # (first of the two 8-digit sequences at the end)
        # Filename format: dt_global_{sat}_phy_l3_1hz_{MEASUREMENT_DATE}_{PRODUCTION_DATE}.nc
        # We ensure date_str is followed by _\d{8}.nc
        date_pattern = re.compile(rf".*_{date_str}_\d{{8}}\.nc$")
        files = [f for f in candidates if date_pattern.match(f)]

    if return_paths_only:
        return files if files else None
    return [xr.open_dataset(f) for f in files] if files else None


def load_l3_sss_smos_data(data_dir, date_str):
    """Load L3 SMOS SSS data, separated by ascending and descending passes.
    
    Returns:
        Dictionary with 'asc' and 'desc' keys, each containing list of file paths
    """
    year = date_str[:4]
    asc_dir = os.path.join(data_dir, "l3_sss_smos", "MULTIOBS_GLO_PHY_SSS_L3_MYNRT_015_014",
                           "cmems_obs-mob_glo_phy-sss_mynrt_smos-asc_P1D_202411", year)
    des_dir = os.path.join(data_dir, "l3_sss_smos", "MULTIOBS_GLO_PHY_SSS_L3_MYNRT_015_014",
                           "cmems_obs-mob_glo_phy-sss_mynrt_smos-des_P1D_202411", year)
    
    asc_files = glob.glob(os.path.join(asc_dir, f"*{date_str}*.nc"))
    desc_files = glob.glob(os.path.join(des_dir, f"*{date_str}*.nc"))
    
    return {
        'asc': asc_files if asc_files else None,
        'desc': desc_files if desc_files else None
    }


def load_argo_data(data_dir, date_str):
    """Load Argo data."""
    fpath = os.path.join(data_dir, "argo", f"argo_{date_str}.nc")
    return xr.open_dataset(fpath) if os.path.exists(fpath) else None


# =============================================================================
# Processing Functions
# =============================================================================

def make_record(outfile, output_dir, base_timestamp, data_source, variable, 
                region_name, bbox, geometry_wkb, time_range, sensor, 
                dataset=None, **extra):
    """Create a standardized record dict with microsecond timestamps and resolution info.
    
    Args:
        dataset: Optional xarray Dataset to compute resolution from
        **extra: Additional fields to include in the record
    """
    # Convert time_range to microseconds for ISTAC
    if time_range and time_range[0]:
        istac_start = int(time_range[0] * 1_000_000)  # seconds to microseconds
        istac_end = int(time_range[1] * 1_000_000)
    else:
        istac_start = int(base_timestamp.timestamp() * 1_000_000)
        istac_end = int((base_timestamp + pd.Timedelta(days=1)).timestamp() * 1_000_000)
    
    # Compute resolution if dataset is provided
    # Pass bbox for validation but compute_resolution will use dataset coordinates
    resolution_info = compute_resolution(dataset, bbox)
    
    # Debug logging if resolution is None for gridded data
    if dataset is not None and resolution_info['resolution_deg_lat'] is None:
        logging.debug(f"Resolution is None for {data_source} {region_name}: "
                     f"dataset={'None' if dataset is None else 'exists'}, "
                     f"bbox={bbox}, "
                     f"has_lat={'lat' in dataset.coords if dataset is not None else 'N/A'}, "
                     f"has_lon={'lon' in dataset.coords if dataset is not None else 'N/A'}")
    
    record = {
        "relative_path": os.path.relpath(outfile, output_dir),
        "timestamp_file": base_timestamp,
        "timestamp_data": base_timestamp,
        "data_source": data_source,
        "variable": variable,
        "filename": os.path.basename(outfile),
        "_istac_spatial_wkb": geometry_wkb,
        "_istac_time_start": istac_start,
        "_istac_time_end": istac_end,
        "sensor": sensor,
        "region": region_name,
        "bbox": bbox,
        "intersects": True,
        "resolution_deg_lat": resolution_info['resolution_deg_lat'],
        "resolution_deg_lon": resolution_info['resolution_deg_lon'],
        "resolution_km_lat": resolution_info['resolution_km_lat'],
        "resolution_km_lon": resolution_info['resolution_km_lon'],
    }
    record.update(extra)
    return record



def process_glorys_data(ds, date_str, output_dir):
    """Process GLORYS: extract variables at specific depths, split by region."""
    if ds is None:
        return 0, []
    
    if "time" in ds.dims:
        ds = ds.isel(time=slice(0, 1))
    ds = normalize_coords(ds)
    
    variables = {
        "ssh": ("zos", None),
        "mdt": ("mdt", None),
        "sst": ("thetao", 0),
        "sss": ("so", 0),
        "uo": ("uo", 10),
        "vo": ("vo", 10),
    }
    
    regional_data = split_gridded_into_regions(ds, SPATIAL_REGIONS)
    
    out_dir = Path(output_dir) / "glorys"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    records = []
    base_timestamp = pd.Timestamp(datetime.strptime(date_str, "%Y%m%d"))
    
    for region_name, region_info in regional_data.items():
        if not region_info["intersects"]:
            continue
        
        regional_ds = region_info["dataset"]
        
        data_vars = {}
        encoding = {}
        
        for var_name, (var_key, depth_idx) in variables.items():
            if var_key not in regional_ds.data_vars:
                continue
            
            var_da = regional_ds[var_key]
            if depth_idx is not None and "depth" in var_da.dims:
                var_da = var_da.isel(depth=depth_idx)
            
            # FIX: Drop depth coordinate to prevent merge conflicts
            # because different variables are sliced at different depths.
            if "depth" in var_da.coords:
                var_da = var_da.drop_vars("depth")
            # Do NOT squeeze – preserves the size-1 time dimension.
            
            data_vars[var_key] = var_da
            encoding[var_key] = get_variable_encoding(var_key)

        if not data_vars:
            continue

        coords = {"lat": regional_ds["lat"], "lon": regional_ds["lon"]}
        if "time" in regional_ds.coords:
            coords["time"] = regional_ds["time"]
        ds_out = xr.Dataset(
            data_vars,
            coords=coords,
            attrs=regional_ds.attrs
        )
        
        ds_out = clear_encoding(ds_out)
        check_encoding_safety(ds_out, encoding)
        
        outfile = out_dir / f"glorys_{region_name}_{date_str}.nc"
        ds_out.to_netcdf(outfile, encoding=encoding, engine="h5netcdf")
        
        records.append(make_record(
            outfile, output_dir, base_timestamp, "glorys", "all_phys",
            region_name, region_info["bbox"], region_info["geometry"],
            region_info["time_range"], "GLORYS", dataset=ds_out
        ))
    
    logging.info(f"✓ GLORYS: {len(records)} regional files")
    return len(records), records

def process_and_split(ds, date_str, output_dir, modality, keep_vars=None, sensor=None):
    """Generic processing for gridded data."""
    if ds is None:
        return 0, []
    if "time" in ds.dims:
        ds = ds.isel(time=slice(0, 1))
    ds = normalize_coords(ds)
    if keep_vars:
        available = [v for v in keep_vars if v in ds.data_vars]
        if available:
            ds = ds[available]
            
    # Convert Kelvin to Celsius for Temperature variables (SST)
    for v in ds.data_vars:
        # Check if variable name suggests temperature
        is_temp_var = any(x in v.lower() for x in ["temperature", "sst", "thetao", "sea_surface_temperature", "analysed_sst"])
        # Check if units suggest Kelvin (or just check range)
        # Avoid non-temperature vars like 'sst_dtime' or 'sources_of_sst' by name too
        is_metadata = any(x in v.lower() for x in ["dtime", "source", "mask", "flag", "count", "number"])
        
        if is_temp_var and not is_metadata:
            # Check mean value to distinguish K vs C
            # Using dropna=True to handle NaNs
            valid_data = ds[v].values.flatten()
            valid_data = valid_data[~np.isnan(valid_data)]
            
            if len(valid_data) > 0 and valid_data.mean() > 200:
                logging.info(f" Converting {v} from Kelvin to Celsius (mean={valid_data.mean():.1f})")
                ds[v] = ds[v] - 273.15
                ds[v].attrs["units"] = "degree_Celsius"
    
    regional_data = split_gridded_into_regions(ds, SPATIAL_REGIONS)
    out_dir = Path(output_dir) / modality
    out_dir.mkdir(parents=True, exist_ok=True)
    
    records = []
    base_timestamp = pd.Timestamp(datetime.strptime(date_str, "%Y%m%d"))
    
    for region_name, region_info in regional_data.items():
        if not region_info["intersects"]:
            continue
        
        regional_ds = clear_encoding(region_info["dataset"])
        outfile = out_dir / f"{modality}_{region_name}_{date_str}.nc"
        
        # Determine encoding and check safety
        encoding = {v: get_variable_encoding(v) for v in regional_ds.data_vars}
        check_encoding_safety(regional_ds, encoding)
        
        regional_ds.to_netcdf(outfile, encoding=encoding, engine="h5netcdf")
        
        records.append(make_record(
            outfile, output_dir, base_timestamp, modality, modality.split("_")[-1],
            region_name, region_info["bbox"], region_info["geometry"],
            region_info["time_range"], sensor or modality.upper(), dataset=regional_ds
        ))
    
    logging.info(f"✓ {modality}: {len(records)} regional files")
    return len(records), records


import numpy as np
from scipy.stats import binned_statistic_2d
from typing import Literal


def bin_swath_to_grid_conservative(
    lons: np.ndarray,
    lats: np.ndarray,
    values: np.ndarray,
    target_lons: np.ndarray,
    target_lats: np.ndarray,
    method: Literal['mean', 'median', 'nearest'] = 'mean',
    min_samples: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Conservative regridding of swath data to regular grid.

    Preserves native resolution by simple binning without smoothing.
    No gap-filling — cells without observations remain NaN.

    Parameters:
        lons: Coordinates of swath observations (1D or 2D).
        lats: Coordinates of swath observations (1D or 2D).
        values: Data values at each observation point.
        target_lons: Regular grid cell centres (1D).
        target_lats: Regular grid cell centres (1D).
        method: Aggregation method — 'mean', 'median', or 'nearest'.
        min_samples: Minimum observations required per cell.

    Returns:
        grid_data: Binned mean, shape (nlat, nlon). NaN where n < min_samples.
        grid_counts: Observation count per cell, shape (nlat, nlon).
        grid_std: Std dev of the mean (σ/√n) per cell. NaN where n < 2.
            Useful for identifying problem cells with high dispersion.
        grid_mean_lon: Mean longitude of observations in each cell.
            Will differ from the cell-centre lon when obs are not uniformly
            distributed inside the bin.
        grid_mean_lat: Mean latitude of observations in each cell.
    """
    lons_flat = np.asarray(lons).flatten()
    lats_flat = np.asarray(lats).flatten()
    vals_flat = np.asarray(values).flatten()

    shape = (len(target_lats), len(target_lons))
    empty = np.full(shape, np.nan)

    valid = np.isfinite(vals_flat) & np.isfinite(lons_flat) & np.isfinite(lats_flat)
    if valid.sum() == 0:
        return empty, np.zeros(shape, dtype=np.int32), empty, empty, empty

    lons_v, lats_v, vals_v = lons_flat[valid], lats_flat[valid], vals_flat[valid]

    # Compute grid edges from centres
    lon_res = np.median(np.diff(target_lons))
    lat_res = np.median(np.diff(target_lats))
    lon_edges = np.concatenate([
        [target_lons[0] - lon_res / 2],
        (target_lons[:-1] + target_lons[1:]) / 2,
        [target_lons[-1] + lon_res / 2],
    ])
    lat_edges = np.concatenate([
        [target_lats[0] - lat_res / 2],
        (target_lats[:-1] + target_lats[1:]) / 2,
        [target_lats[-1] + lat_res / 2],
    ])

    statistic = 'mean' if method == 'nearest' else method

    grid_data, _, _, _ = binned_statistic_2d(
        lons_v, lats_v, vals_v, statistic=statistic, bins=[lon_edges, lat_edges]
    )
    grid_data = grid_data.T  # (nlat, nlon)

    grid_counts, _, _, _ = binned_statistic_2d(
        lons_v, lats_v, vals_v, statistic='count', bins=[lon_edges, lat_edges]
    )
    grid_counts = grid_counts.T.astype(np.int32)

    # Std dev of the mean: σ/√n  (requires n ≥ 2; NaN for singletons)
    grid_std2, _, _, _ = binned_statistic_2d(
        lons_v, lats_v, vals_v, statistic='std', bins=[lon_edges, lat_edges]
    )
    grid_std2 = grid_std2.T  # population std from scipy
    with np.errstate(invalid='ignore', divide='ignore'):
        grid_sem = grid_std2 / np.sqrt(np.maximum(grid_counts, 1))
    grid_sem[grid_counts < 2] = np.nan

    # Mean lon/lat of observations — will differ from cell centre when obs
    # are clustered (e.g. along-track swath crossing a corner of the cell).
    grid_mean_lon, _, _, _ = binned_statistic_2d(
        lons_v, lats_v, lons_v, statistic='mean', bins=[lon_edges, lat_edges]
    )
    grid_mean_lon = grid_mean_lon.T

    grid_mean_lat, _, _, _ = binned_statistic_2d(
        lons_v, lats_v, lats_v, statistic='mean', bins=[lon_edges, lat_edges]
    )
    grid_mean_lat = grid_mean_lat.T

    # Apply minimum-sample threshold to the primary field
    grid_data[grid_counts < min_samples] = np.nan

    return grid_data, grid_counts, grid_sem, grid_mean_lon, grid_mean_lat


def process_swot_track_to_grid(
    track_lons: np.ndarray,
    track_lats: np.ndarray,
    track_data: np.ndarray,
    target_lons: np.ndarray,
    target_lats: np.ndarray,
    resolution_km: float = 2.0,
) -> dict:
    """
    Process a single SWOT track onto a regular grid.
    
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
        
    Returns
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
        track_lons, track_lats, track_data,
        target_lons, target_lats,
        method="mean",
        min_samples=1
    )
    
    return {
        "data": grid_data,
        "counts": grid_counts,
        "lons": target_lons,
        "lats": target_lats,
    }


# def process_swot_daily_gridded(data_dir, date_str, output_dir, resolution_km=2.0, overlap_method="mean"):
#     """Process L3 SWOT files, gridding them per region with track masking support.
    
#     Creates gridded data with additional track identification layers:
#     - primary_track: Index of the first track to contribute to each cell (-1 = no data)
#     - is_overlap: Boolean mask indicating cells where multiple tracks contributed
#     - track_ids: List of track identifiers (filenames)
#     - track_times: Timestamps for each track
    
#     This allows masking out specific tracks for validation purposes.

#     Process L3 SWOT files with conservative regridding.
    
#     Key improvements over v1:
#     - No Gaussian smoothing (preserves spectral content)
#     - No artificial gap-filling (NaN where no data)
#     - Proper handling of track overlaps
#     - Resolution-matched gridding to preserve 2km features
    
#     Parameters
#     ----------
#     overlap_method : str
#         How to handle overlapping tracks: "mean", "first", "last"
#     """
#     files = load_l3_swot_data(data_dir, date_str, return_paths_only=True)
#     if not files:
#         return 0, []
    
#     out_dir = Path(output_dir) / "l3_swot"
#     out_dir.mkdir(parents=True, exist_ok=True)
#     logging.info(f"Processing {len(files)} L3_SWOT files for {date_str}")
    
#     records = []
#     base_timestamp = pd.Timestamp(datetime.strptime(date_str, "%Y%m%d"))
#     expected_swot_vars = set(L3_SWOT_VARS)
    
#     for region_name, bounds in SPATIAL_REGIONS.items():
#         target_lons, target_lats, lon_edges, lat_edges = create_regional_grid(
#             bounds, resolution_km
#         )
#         grid_shape = (len(target_lats), len(target_lons))
        
#         # Accumulators for observation-weighted mean and pooled variance.
#         # For each variable we track:
#         #   accumulated_sum[v]   = Σ (cell_mean * cell_count)   → used for final mean
#         #   accumulated_ss[v]    = Σ (cell_std * cell_count)²   → used for pooled variance
#         #                          (parallel/pooled variance formula)
#         #   accumulated_count[v] = Σ cell_count                 → denominator
#         # For mean obs position we use the same weighted scheme on lon/lat directly.
#         accumulated_sum = {}
#         accumulated_ss = {}   # sum of (sem * count)^2 for pooled std-of-mean
#         accumulated_count = {}
#         # lon/lat accumulators share the same count as the first variable binned
#         # per cell — we track them once via the coordinate arrays directly.
#         accumulated_lon_sum = np.zeros(grid_shape, dtype=np.float64)
#         accumulated_lat_sum = np.zeros(grid_shape, dtype=np.float64)
#         accumulated_coord_count = np.zeros(grid_shape, dtype=np.int32)

#         # Track metadata
#         primary_track = np.full(grid_shape, -1, dtype=np.int16)
#         track_count_per_cell = np.zeros(grid_shape, dtype=np.int16)
#         track_ids = []
#         track_times = []
#         time_min, time_max = None, None
#         intersected_tracks = 0
#         schema_missing_tracks = 0
#         non_binning_tracks = 0

#         regional_track_idx = 0

#         for fpath in files:
#             try:
#                 ds = xr.open_dataset(fpath)
#             except Exception:
#                 continue

#             # Get coordinates
#             swath_lons = ds['longitude'].values if 'longitude' in ds.coords else ds['lon'].values
#             swath_lats = ds['latitude'].values if 'latitude' in ds.coords else ds['lat'].values
#             swath_lons = lon_to_180(swath_lons)

#             if not swath_intersects_region(swath_lons, swath_lats, bounds):
#                 ds.close()
#                 continue
#             intersected_tracks += 1

#             available_swot_vars = [v for v in L3_SWOT_VARS if v in ds.data_vars]
#             if not available_swot_vars:
#                 schema_missing_tracks += 1
#                 ds.close()
#                 continue

#             # Track time
#             track_time = None
#             if 'time' in ds.coords:
#                 t_vals = pd.to_datetime(
#                     np.asarray(ds['time'].values).flatten(),
#                     errors='coerce',
#                 )
#                 t_vals = pd.Series(t_vals).dropna()
#                 if len(t_vals) > 0:
#                     track_time = t_vals.mean()
#                     if time_min is None or t_vals.min() < time_min:
#                         time_min = t_vals.min()
#                     if time_max is None or t_vals.max() > time_max:
#                         time_max = t_vals.max()

#             # Compute ADT on raw swath pixels before binning so NaN masks are
#             # consistent: adt is NaN exactly where ssha is NaN.
#             swath_var_arrays = {v: ds[v].values for v in available_swot_vars}
#             if 'ssha_filtered' in swath_var_arrays and 'mdt' in swath_var_arrays:
#                 swath_var_arrays['adt_filtered'] = (
#                     swath_var_arrays['ssha_filtered'] + swath_var_arrays['mdt']
#                 )
#             if 'ssha_unfiltered' in swath_var_arrays and 'mdt' in swath_var_arrays:
#                 swath_var_arrays['adt_unfiltered'] = (
#                     swath_var_arrays['ssha_unfiltered'] + swath_var_arrays['mdt']
#                 )

#             track_contributed_mask = np.zeros(grid_shape, dtype=bool)
#             coord_updated_this_track = False

#             for var_name, swath_data in swath_var_arrays.items():
#                 grid_data, grid_counts, grid_sem, grid_mean_lon, grid_mean_lat = (
#                     bin_swath_to_grid_conservative(
#                         swath_lons, swath_lats, swath_data,
#                         target_lons, target_lats,
#                         method='mean',
#                         min_samples=1,
#                     )
#                 )

#                 valid_mask = grid_counts > 0
#                 if not valid_mask.any():
#                     continue

#                 track_contributed_mask |= valid_mask

#                 if var_name not in accumulated_sum:
#                     accumulated_sum[var_name] = np.zeros(grid_shape, dtype=np.float64)
#                     accumulated_ss[var_name] = np.zeros(grid_shape, dtype=np.float64)
#                     accumulated_count[var_name] = np.zeros(grid_shape, dtype=np.int32)

#                 # Observation-weighted mean accumulation
#                 accumulated_sum[var_name][valid_mask] += (
#                     grid_data[valid_mask] * grid_counts[valid_mask]
#                 )
#                 # Pooled variance: add within-bin variance * (n-1) + between-step
#                 # Simplified: accumulate (sem * count)^2 per step; divide at end.
#                 sem_valid = np.where(np.isfinite(grid_sem), grid_sem, 0.0)
#                 accumulated_ss[var_name][valid_mask] += (
#                     sem_valid[valid_mask] * grid_counts[valid_mask]
#                 ) ** 2
#                 accumulated_count[var_name][valid_mask] += grid_counts[valid_mask]

#                 # Accumulate mean obs position once per track (use first variable)
#                 if not coord_updated_this_track:
#                     lon_valid = np.isfinite(grid_mean_lon) & valid_mask
#                     lat_valid = np.isfinite(grid_mean_lat) & valid_mask
#                     accumulated_lon_sum[lon_valid] += (
#                         grid_mean_lon[lon_valid] * grid_counts[lon_valid]
#                     )
#                     accumulated_lat_sum[lat_valid] += (
#                         grid_mean_lat[lat_valid] * grid_counts[lat_valid]
#                     )
#                     accumulated_coord_count[valid_mask] += grid_counts[valid_mask]
#                     coord_updated_this_track = True

#             if not track_contributed_mask.any():
#                 non_binning_tracks += 1
#                 ds.close()
#                 continue

#             track_ids.append(os.path.basename(fpath))
#             track_times.append(track_time.isoformat() if track_time else '')

#             # Update track info only for cells this track contributed to
#             new_cells = track_contributed_mask & (primary_track == -1)
#             primary_track[new_cells] = regional_track_idx
#             track_count_per_cell[track_contributed_mask] = np.minimum(
#                 track_count_per_cell[track_contributed_mask] + 1,
#                 np.iinfo(np.int16).max,
#             )
#             regional_track_idx += 1

#             ds.close()

#         if not accumulated_sum:
#             if intersected_tracks > 0:
#                 logging.warning(
#                     f'L3_SWOT {date_str} {region_name}: no valid gridded output '
#                     f'(intersected_tracks={intersected_tracks}, '
#                     f'schema_missing_tracks={schema_missing_tracks}, '
#                     f'non_binning_tracks={non_binning_tracks}, '
#                     f'expected_vars={sorted(expected_swot_vars)})'
#                 )
#             continue

#         if schema_missing_tracks > 0:
#             logging.warning(
#                 f'L3_SWOT {date_str} {region_name}: {schema_missing_tracks}/{intersected_tracks} '
#                 f'intersecting tracks missing expected variables {sorted(expected_swot_vars)}'
#             )

#         # Compute final observation-weighted mean, std of mean, and mean obs position
#         data_vars = {}
#         total_obs = np.zeros(grid_shape, dtype=np.int32)

#         for var_name, data_sum in accumulated_sum.items():
#             count = accumulated_count[var_name]
#             total_obs = np.maximum(total_obs, count)

#             with np.errstate(invalid='ignore', divide='ignore'):
#                 averaged = data_sum / np.maximum(count, 1)
#             averaged[count == 0] = np.nan
#             data_vars[var_name] = (['time', 'lat', 'lon'], averaged[np.newaxis, ...].astype(np.float32))

#             # Pooled std of the mean: sqrt(Σ(sem*n)² / (Σn)²)
#             with np.errstate(invalid='ignore', divide='ignore'):
#                 pooled_sem = np.sqrt(accumulated_ss[var_name]) / np.maximum(count, 1)
#             pooled_sem[count < 2] = np.nan
#             data_vars[f'{var_name}_sem'] = (
#                 ['time', 'lat', 'lon'], pooled_sem[np.newaxis, ...].astype(np.float32)
#             )

#         # Mean observation position — will differ from grid-cell centre for
#         # swath data that crosses a cell corner or edge.
#         coord_count = np.maximum(accumulated_coord_count, 1)
#         with np.errstate(invalid='ignore', divide='ignore'):
#             mean_obs_lon = accumulated_lon_sum / coord_count
#             mean_obs_lat = accumulated_lat_sum / coord_count
#         mean_obs_lon[accumulated_coord_count == 0] = np.nan
#         mean_obs_lat[accumulated_coord_count == 0] = np.nan
#         data_vars['obs_mean_lon'] = (['time', 'lat', 'lon'], mean_obs_lon[np.newaxis, ...].astype(np.float32))
#         data_vars['obs_mean_lat'] = (['time', 'lat', 'lon'], mean_obs_lat[np.newaxis, ...].astype(np.float32))

#         # Observation count (total native observations, not just tracks)
#         data_vars['n_obs'] = (['time', 'lat', 'lon'], total_obs[np.newaxis, ...])

#         # Track masking layers
#         data_vars['primary_track'] = (['time', 'lat', 'lon'], primary_track[np.newaxis, ...])
#         data_vars['is_overlap'] = (['time', 'lat', 'lon'], (track_count_per_cell > 1).astype(np.int8)[np.newaxis, ...])

#         # Create output dataset
#         ds_out = xr.Dataset(
#             data_vars,
#             coords={
#                 'time': np.array([base_timestamp.to_datetime64()]),
#                 'lat': target_lats,
#                 'lon': target_lons,
#                 'track': np.arange(len(track_ids)),
#             },
#             attrs={
#                 'source': 'L3 SWOT conservative regrid',
#                 'resolution_km': resolution_km,
#                 'n_tracks': len(track_ids),
#                 'processing': 'bin_mean_no_smoothing',
#                 'date': date_str,
#                 # CF-compliant CRS declaration (EPSG:4326 / WGS 84)
#                 'crs': 'EPSG:4326',
#                 'geospatial_lat_units': 'degrees_north',
#                 'geospatial_lon_units': 'degrees_east',
#             },
#         )
#         ds_out['lat'].attrs.update({'units': 'degrees_north', 'standard_name': 'latitude', 'long_name': 'latitude'})
#         ds_out['lon'].attrs.update({'units': 'degrees_east', 'standard_name': 'longitude', 'long_name': 'longitude'})

#         if track_ids:
#             ds_out['track_ids'] = (['track'], track_ids)
#             ds_out['track_times'] = (['track'], track_times)

#         # Save
#         outfile = out_dir / f'l3_swot_{region_name}_{date_str}.nc'

#         encoding = {}
#         for var_name in list(accumulated_sum.keys()) + [
#             f'{v}_sem' for v in accumulated_sum
#         ] + ['obs_mean_lon', 'obs_mean_lat']:
#             encoding[var_name] = {
#                 'dtype': 'float32',
#                 'zlib': True,
#                 'complevel': 4,
#                 '_FillValue': np.float32(np.nan),
#             }
#         encoding['n_obs'] = {'dtype': 'int32', 'zlib': True, '_FillValue': -1}
#         encoding['primary_track'] = {'dtype': 'int16', 'zlib': True, '_FillValue': -1}
#         encoding['is_overlap'] = {'dtype': 'int8', 'zlib': True}
        
#         ds_out.to_netcdf(outfile, encoding=encoding, engine="h5netcdf")
        
#         # Record metadata
#         bbox = [
#             float(target_lons.min()), float(target_lats.min()),
#             float(target_lons.max()), float(target_lats.max())
#         ]
#         time_range = (
#             (time_min.timestamp(), time_max.timestamp()) 
#             if time_min else (None, None)
#         )
        
#         records.append(make_record(
#             outfile, output_dir, base_timestamp, "l3_swot", "ssh",
#             region_name, bbox, box(*bbox).wkb, time_range, "SWOT",
#             dataset=ds_out, gridded=True, n_tracks=len(track_ids)
#         ))
    
#     logging.info(f"✓ L3_SWOT: {len(records)} regional files (conservative regrid)")
#     return len(records), records

"""Optimized SWOT processing functions — drop-in replacements.

Replace `bin_swath_to_grid_conservative` and `process_swot_daily_gridded`
in new_format_ssh_data.py with these functions. No other changes needed.

Optimizations applied:
  A. Loop restructured: file -> region (each file opened once, not 8x)
  B. Single bin-assignment via np.searchsorted + np.add.at (replaces
     25 x binned_statistic_2d calls per file x region with 1 searchsorted)
  C. Pre-clip swath points to region bbox before binning

Scientific correctness notes:
  - SEM is computed as per-track SS (within-track variance) accumulated
    across tracks, matching the original code exactly.  Global pooling
    of sumsq would include between-track variance, changing the meaning.
  - Last-bin-closed-right convention matches scipy's binned_statistic_2d.
"""


def bin_swath_to_grid_fast(
    lons: np.ndarray,
    lats: np.ndarray,
    var_arrays: dict[str, np.ndarray],
    lon_edges: np.ndarray,
    lat_edges: np.ndarray,
    n_lon: int,
    n_lat: int,
    min_samples: int = 1,
) -> dict:
    """Bin multiple swath variables onto a regular grid in a single pass.

    Computes bin assignments once via searchsorted, then accumulates
    sum, sumsq, count for all variables using np.add.at.  This replaces
    N_vars x 5 calls to scipy.binned_statistic_2d with one searchsorted
    pass + cheap scatter-adds.

    Args:
        lons: Flattened, pre-clipped longitude array (only points inside region).
        lats: Flattened, pre-clipped latitude array.
        var_arrays: Mapping of variable name to flattened value array
            (same length as lons/lats, NaN for invalid).
        lon_edges: Bin edges for longitude (length n_lon + 1).
        lat_edges: Bin edges for latitude (length n_lat + 1).
        n_lon: Number of longitude grid cells.
        n_lat: Number of latitude grid cells.
        min_samples: Minimum observations per cell for valid output.

    Returns:
        Dict with keys per variable ('sum', 'ss', 'count') and
        shared ('lon_sum', 'lat_sum', 'coord_count', 'any_valid').
        'ss' is the within-track sum of squared deviations per cell,
        NOT raw sumsq — this preserves the original SEM semantics
        when accumulated across tracks.

    """
    grid_shape = (n_lat, n_lon)
    n_cells = n_lat * n_lon

    # Filter NaN/inf coordinates before bin assignment — the original
    # bin_swath_to_grid_conservative does this via:
    #   valid = isfinite(vals) & isfinite(lons) & isfinite(lats)
    # Pre-clipping (lons >= min & lons < max) already removes NaN coords
    # because NaN comparisons return False. But for defensive correctness
    # (e.g. if called without pre-clipping), filter explicitly.
    coord_finite = np.isfinite(lons) & np.isfinite(lats)
    if not coord_finite.any():
        return {'any_valid': False}

    # Bin assignment: searchsorted gives index such that
    # edges[i-1] <= x < edges[i], so subtract 1 for 0-based cell index.
    lon_idx = np.searchsorted(lon_edges, lons, side='right') - 1
    lat_idx = np.searchsorted(lat_edges, lats, side='right') - 1

    # Match scipy convention: last bin is closed on the right, i.e.
    # points exactly on the upper edge belong to the last bin.
    # searchsorted('right') puts these at index n (out of range);
    # clamp them back to n-1. Safe because NaN coords were filtered above.
    np.clip(lon_idx, -1, n_lon - 1, out=lon_idx)
    np.clip(lat_idx, -1, n_lat - 1, out=lat_idx)

    # Mask to points that fall within grid bounds AND have finite coords
    in_bounds = (
        coord_finite &
        (lon_idx >= 0) & (lon_idx < n_lon) &
        (lat_idx >= 0) & (lat_idx < n_lat)
    )

    if not in_bounds.any():
        return {'any_valid': False}

    li = lon_idx[in_bounds]
    la = lat_idx[in_bounds]
    cell_idx = la * n_lon + li

    # Coordinate accumulators (shared across variables)
    lon_sum = np.zeros(n_cells, dtype=np.float64)
    lat_sum = np.zeros(n_cells, dtype=np.float64)
    coord_count = np.zeros(n_cells, dtype=np.int32)

    lons_ib = lons[in_bounds]
    lats_ib = lats[in_bounds]

    result = {'any_valid': False}

    coord_done = False

    for var_name, vals_full in var_arrays.items():
        vals = vals_full[in_bounds]
        finite = np.isfinite(vals)
        if not finite.any():
            continue

        ci = cell_idx[finite]
        vi = vals[finite]

        var_sum = np.zeros(n_cells, dtype=np.float64)
        var_sumsq = np.zeros(n_cells, dtype=np.float64)
        var_count = np.zeros(n_cells, dtype=np.int32)

        np.add.at(var_sum, ci, vi)
        np.add.at(var_sumsq, ci, vi * vi)
        np.add.at(var_count, ci, 1)

        # Compute per-track SS (sum of squared deviations) for this track.
        # SS = Σ(x - mean)² = sumsq - sum²/n
        # This must be computed PER TRACK before accumulation across tracks,
        # because the original code accumulates per-track (sem*n)² = std²*n = SS.
        # Accumulating raw sumsq globally and computing SS at the end would
        # include between-track variance (ANOVA decomposition), changing the
        # scientific meaning of the SEM field.
        var_sum_2d = var_sum.reshape(grid_shape)
        var_count_2d = var_count.reshape(grid_shape)
        with np.errstate(invalid='ignore', divide='ignore'):
            var_ss = var_sumsq.reshape(grid_shape) - (
                var_sum_2d ** 2 / np.maximum(var_count_2d, 1)
            )
        var_ss[var_count_2d < 2] = 0.0
        np.maximum(var_ss, 0.0, out=var_ss)

        result[var_name] = {
            'sum': var_sum_2d,
            'ss': var_ss,
            'count': var_count_2d,
        }
        result['any_valid'] = True

        # Accumulate coordinate means once (using first valid variable's mask)
        if not coord_done:
            lons_finite = lons_ib[finite]
            lats_finite = lats_ib[finite]
            np.add.at(lon_sum, ci, lons_finite)
            np.add.at(lat_sum, ci, lats_finite)
            np.add.at(coord_count, ci, 1)
            coord_done = True

    result['lon_sum'] = lon_sum.reshape(grid_shape)
    result['lat_sum'] = lat_sum.reshape(grid_shape)
    result['coord_count'] = coord_count.reshape(grid_shape)

    return result


def process_swot_daily_gridded(data_dir, date_str, output_dir, resolution_km=2.0):
    """Process L3 SWOT files, gridding them per region with track masking support.

    Optimised version: opens each file once and dispatches to all
    intersecting regions (instead of opening each file 8 times).  Uses
    searchsorted-based binning instead of repeated binned_statistic_2d.

    Creates gridded data with additional track identification layers:
      - n_tracks: Number of primary-field tracks contributing to each cell
      - primary_track: Index of the first track to contribute to each cell
      - is_overlap: Boolean mask where multiple tracks contributed
      - track_ids / track_times: Per-track metadata

    Args:
        data_dir: Root data directory.
        date_str: Date string in YYYYMMDD format.
        output_dir: Output directory root.
        resolution_km: Target grid resolution in km.

    Returns:
        Tuple of (n_records, records_list).

    """
    files = load_l3_swot_data(data_dir, date_str, return_paths_only=True)
    if not files:
        return 0, []

    out_dir = Path(output_dir) / 'l3_swot'
    out_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f'Processing {len(files)} L3_SWOT files for {date_str}')

    base_timestamp = pd.Timestamp(datetime.strptime(date_str, '%Y%m%d'))
    expected_swot_vars = set(L3_SWOT_VARS)

    # Pre-compute regional grids (cheap, do once)
    region_grids = {}
    for region_name, bounds in SPATIAL_REGIONS.items():
        target_lons, target_lats, lon_edges, lat_edges = create_regional_grid(
            bounds, resolution_km
        )
        n_lat, n_lon = len(target_lats), len(target_lons)
        grid_shape = (n_lat, n_lon)
        region_grids[region_name] = {
            'bounds': bounds,
            'target_lons': target_lons,
            'target_lats': target_lats,
            'lon_edges': lon_edges,
            'lat_edges': lat_edges,
            'n_lon': n_lon,
            'n_lat': n_lat,
            'grid_shape': grid_shape,
            # Accumulators
            'accumulated_sum': {},
            'accumulated_count': {},
            # Observation-weighted inter-track dispersion: Σ_k n_k z̄_k² and
            # the number of contributing tracks K per cell for each variable.
            'accumulated_track_sq': {},
            'accumulated_track_count': {},
            'accumulated_lon_sum': np.zeros(grid_shape, dtype=np.float64),
            'accumulated_lat_sum': np.zeros(grid_shape, dtype=np.float64),
            'accumulated_coord_count': np.zeros(grid_shape, dtype=np.int32),
            # Track metadata
            'primary_track': np.full(grid_shape, -1, dtype=np.int16),
            'primary_track_count': np.zeros(grid_shape, dtype=np.int16),
            'track_ids': [],
            'track_times': [],
            'time_min': None,
            'time_max': None,
            'regional_track_idx': 0,
            # Diagnostics
            'intersected_tracks': 0,
            'schema_missing_tracks': 0,
            'non_binning_tracks': 0,
        }

    # ---- Main loop: iterate files ONCE, dispatch to all regions ----
    for fpath in files:
        try:
            ds = xr.open_dataset(fpath)
        except Exception:
            continue

        # Read coordinates once
        swath_lons = ds['longitude'].values if 'longitude' in ds.coords else ds['lon'].values
        swath_lats = ds['latitude'].values if 'latitude' in ds.coords else ds['lat'].values
        swath_lons = lon_to_180(swath_lons)

        # Flatten once
        lons_flat = swath_lons.ravel()
        lats_flat = swath_lats.ravel()

        # Check which variables are available
        available_swot_vars = [v for v in L3_SWOT_VARS if v in ds.data_vars]

        # Read variable arrays once, flatten, compute derived fields
        swath_var_arrays = {}
        if available_swot_vars:
            for v in available_swot_vars:
                swath_var_arrays[v] = ds[v].values.ravel()
            if 'ssha_filtered' in swath_var_arrays and 'mdt' in swath_var_arrays:
                swath_var_arrays['adt_filtered'] = (
                    swath_var_arrays['ssha_filtered'] + swath_var_arrays['mdt']
                )
            if 'ssha_unfiltered' in swath_var_arrays and 'mdt' in swath_var_arrays:
                swath_var_arrays['adt_unfiltered'] = (
                    swath_var_arrays['ssha_unfiltered'] + swath_var_arrays['mdt']
                )

        # Parse track time once
        track_time = None
        t_min_file, t_max_file = None, None
        if 'time' in ds.coords:
            t_vals = pd.to_datetime(
                np.asarray(ds['time'].values).flatten(), errors='coerce'
            )
            t_vals = pd.Series(t_vals).dropna()
            if len(t_vals) > 0:
                track_time = t_vals.mean()
                t_min_file = t_vals.min()
                t_max_file = t_vals.max()

        ds.close()

        # Dispatch to each intersecting region
        for region_name, rg in region_grids.items():
            bounds = rg['bounds']

            if not swath_intersects_region(lons_flat, lats_flat, bounds):
                continue
            rg['intersected_tracks'] += 1

            if not available_swot_vars:
                rg['schema_missing_tracks'] += 1
                continue

            # Optimisation C: pre-clip points to region bounding box
            lon_min, lon_max = bounds['lon']
            lat_min, lat_max = bounds['lat']
            region_mask = (
                (lons_flat >= lon_min) & (lons_flat < lon_max) &
                (lats_flat >= lat_min) & (lats_flat < lat_max)
            )
            if not region_mask.any():
                continue

            clipped_lons = lons_flat[region_mask]
            clipped_lats = lats_flat[region_mask]
            clipped_vars = {k: v[region_mask] for k, v in swath_var_arrays.items()}

            # Optimisation B: single-pass binning for all variables
            bin_result = bin_swath_to_grid_fast(
                clipped_lons, clipped_lats, clipped_vars,
                rg['lon_edges'], rg['lat_edges'],
                rg['n_lon'], rg['n_lat'],
                min_samples=1,
            )

            if not bin_result['any_valid']:
                rg['non_binning_tracks'] += 1
                continue

            # Update time range
            if t_min_file is not None:
                if rg['time_min'] is None or t_min_file < rg['time_min']:
                    rg['time_min'] = t_min_file
                if rg['time_max'] is None or t_max_file > rg['time_max']:
                    rg['time_max'] = t_max_file

            # Accumulate per-variable results
            grid_shape = rg['grid_shape']
            track_contributed_mask = np.zeros(grid_shape, dtype=bool)
            primary_valid_mask = np.zeros(grid_shape, dtype=bool)

            for var_name in clipped_vars:
                if var_name not in bin_result:
                    continue
                br = bin_result[var_name]
                valid_mask = br['count'] > 0
                track_contributed_mask |= valid_mask
                if var_name == PRIMARY_L3_SWOT_VAR:
                    primary_valid_mask |= valid_mask

                if var_name not in rg['accumulated_sum']:
                    rg['accumulated_sum'][var_name] = np.zeros(grid_shape, dtype=np.float64)
                    rg['accumulated_count'][var_name] = np.zeros(grid_shape, dtype=np.int32)
                    rg['accumulated_track_sq'][var_name] = np.zeros(grid_shape, dtype=np.float64)
                    rg['accumulated_track_count'][var_name] = np.zeros(grid_shape, dtype=np.int32)

                # Observation-weighted accumulation of per-cell totals.
                rg['accumulated_sum'][var_name][valid_mask] += br['sum'][valid_mask]
                rg['accumulated_count'][var_name][valid_mask] += br['count'][valid_mask]
                # Observation-weighted inter-track dispersion of per-track means:
                # z̄_k = sum/count for this track, so n_k z̄_k² = sum²/count.
                rg['accumulated_track_sq'][var_name][valid_mask] += (
                    br['sum'][valid_mask] ** 2 / br['count'][valid_mask]
                )
                rg['accumulated_track_count'][var_name][valid_mask] += 1

            # Coordinate accumulators
            coord_valid = bin_result['coord_count'] > 0
            rg['accumulated_lon_sum'][coord_valid] += bin_result['lon_sum'][coord_valid]
            rg['accumulated_lat_sum'][coord_valid] += bin_result['lat_sum'][coord_valid]
            rg['accumulated_coord_count'][coord_valid] += bin_result['coord_count'][coord_valid]

            if not track_contributed_mask.any():
                rg['non_binning_tracks'] += 1
                continue

            # Track metadata
            rg['track_ids'].append(os.path.basename(fpath))
            rg['track_times'].append(track_time.isoformat() if track_time else '')

            new_cells = primary_valid_mask & (rg['primary_track'] == -1)
            rg['primary_track'][new_cells] = rg['regional_track_idx']
            rg['primary_track_count'][primary_valid_mask] = np.minimum(
                rg['primary_track_count'][primary_valid_mask] + 1,
                np.iinfo(np.int16).max,
            )
            rg['regional_track_idx'] += 1

    # ---- Finalize and write each region ----
    records = []

    for region_name, rg in region_grids.items():
        grid_shape = rg['grid_shape']
        target_lons = rg['target_lons']
        target_lats = rg['target_lats']

        if not rg['accumulated_sum']:
            if rg['intersected_tracks'] > 0:
                logging.warning(
                    f'L3_SWOT {date_str} {region_name}: no valid gridded output '
                    f'(intersected_tracks={rg["intersected_tracks"]}, '
                    f'schema_missing_tracks={rg["schema_missing_tracks"]}, '
                    f'non_binning_tracks={rg["non_binning_tracks"]}, '
                    f'expected_vars={sorted(expected_swot_vars)})'
                )
            continue

        if rg['schema_missing_tracks'] > 0:
            logging.warning(
                f'L3_SWOT {date_str} {region_name}: '
                f'{rg["schema_missing_tracks"]}/{rg["intersected_tracks"]} '
                f'intersecting tracks missing expected variables '
                f'{sorted(expected_swot_vars)}'
            )

        # Compute final observation-weighted mean and variable-specific inter-track dispersion
        data_vars = {}

        for var_name, data_sum in rg['accumulated_sum'].items():
            count = rg['accumulated_count'][var_name]

            with np.errstate(invalid='ignore', divide='ignore'):
                averaged = data_sum / np.maximum(count, 1)
            averaged[count == 0] = np.nan
            data_vars[var_name] = (
                ['time', 'lat', 'lon'], averaged[np.newaxis, ...].astype(np.float32)
            )

            # Observation-weighted dispersion of per-track means:
            # Var = Σ_k (n_k * z̄_k^2) / n - z̄^2
            # Defined only where at least two tracks contributed to this variable.
            n_tracks = rg['accumulated_track_count'][var_name]
            with np.errstate(invalid='ignore', divide='ignore'):
                between_var = (
                    rg['accumulated_track_sq'][var_name] / np.maximum(count, 1) - averaged ** 2
                )
            between_var = np.where(between_var > 0, between_var, 0.0)
            intertrack_std = np.sqrt(between_var)
            intertrack_std[n_tracks < 2] = np.nan
            data_vars[f'{var_name}_intertrack_std'] = (
                ['time', 'lat', 'lon'], intertrack_std[np.newaxis, ...].astype(np.float32)
            )

        # Mean observation position
        coord_count = np.maximum(rg['accumulated_coord_count'], 1)
        with np.errstate(invalid='ignore', divide='ignore'):
            mean_obs_lon = rg['accumulated_lon_sum'] / coord_count
            mean_obs_lat = rg['accumulated_lat_sum'] / coord_count
        mean_obs_lon[rg['accumulated_coord_count'] == 0] = np.nan
        mean_obs_lat[rg['accumulated_coord_count'] == 0] = np.nan
        data_vars['obs_mean_lon'] = (
            ['time', 'lat', 'lon'], mean_obs_lon[np.newaxis, ...].astype(np.float32)
        )
        data_vars['obs_mean_lat'] = (
            ['time', 'lat', 'lon'], mean_obs_lat[np.newaxis, ...].astype(np.float32)
        )

        # Primary-variable provenance and track masking layers
        primary_count = rg['accumulated_count'].get(PRIMARY_L3_SWOT_VAR)
        if primary_count is None:
            primary_count = next(iter(rg['accumulated_count'].values()))
        data_vars['n_obs'] = (
            ['time', 'lat', 'lon'], primary_count[np.newaxis, ...].astype(np.int32)
        )
        data_vars['n_tracks'] = (
            ['time', 'lat', 'lon'], rg['primary_track_count'][np.newaxis, ...].astype(np.int16)
        )
        data_vars['primary_track'] = (
            ['time', 'lat', 'lon'], rg['primary_track'][np.newaxis, ...]
        )
        data_vars['is_overlap'] = (
            ['time', 'lat', 'lon'],
            (rg['primary_track_count'] > 1).astype(np.int8)[np.newaxis, ...],
        )

        track_ids = rg['track_ids']
        track_times = rg['track_times']

        ds_out = xr.Dataset(
            data_vars,
            coords={
                'time': np.array([base_timestamp.to_datetime64()]),
                'lat': target_lats,
                'lon': target_lons,
                'track': np.arange(len(track_ids)),
            },
            attrs={
                'source': 'L3 SWOT conservative regrid',
                'resolution_km': resolution_km,
                'n_tracks': len(track_ids),
                'processing': 'bin_mean_no_smoothing',
                'date': date_str,
                'crs': 'EPSG:4326',
                'geospatial_lat_units': 'degrees_north',
                'geospatial_lon_units': 'degrees_east',
            },
        )
        ds_out['lat'].attrs.update({
            'units': 'degrees_north', 'standard_name': 'latitude', 'long_name': 'latitude',
        })
        ds_out['lon'].attrs.update({
            'units': 'degrees_east', 'standard_name': 'longitude', 'long_name': 'longitude',
        })

        if track_ids:
            ds_out['track_ids'] = (['track'], track_ids)
            ds_out['track_times'] = (['track'], track_times)

        # Save
        outfile = out_dir / f'l3_swot_{region_name}_{date_str}.nc'

        encoding = {}
        for var_name in list(rg['accumulated_sum'].keys()) + [
            f'{v}_intertrack_std' for v in rg['accumulated_sum']
        ] + ['obs_mean_lon', 'obs_mean_lat']:
            encoding[var_name] = {
                'dtype': 'float32',
                'zlib': True,
                'complevel': 4,
                '_FillValue': np.float32(np.nan),
            }
        encoding['n_obs'] = {'dtype': 'int32', 'zlib': True, '_FillValue': -1}
        encoding['n_tracks'] = {'dtype': 'int16', 'zlib': True, '_FillValue': -1}
        encoding['primary_track'] = {'dtype': 'int16', 'zlib': True, '_FillValue': -1}
        encoding['is_overlap'] = {'dtype': 'int8', 'zlib': True}

        ds_out.to_netcdf(outfile, encoding=encoding, engine='h5netcdf')

        # Record metadata
        bbox = [
            float(target_lons.min()), float(target_lats.min()),
            float(target_lons.max()), float(target_lats.max()),
        ]
        time_range = (
            (rg['time_min'].timestamp(), rg['time_max'].timestamp())
            if rg['time_min'] else (None, None)
        )

        records.append(make_record(
            outfile, output_dir, base_timestamp, 'l3_swot', 'ssh',
            region_name, bbox, box(*bbox).wkb, time_range, 'SWOT',
            dataset=ds_out, gridded=True, n_tracks=len(track_ids),
        ))

    logging.info(f'✓ L3_SWOT: {len(records)} regional files (conservative regrid)')
    return len(records), records

def process_l3_ssh_data(data_dir, date_str, output_dir, resolution_km=7.0):
    """Process L3 SSH along-track data - conservative gridding per region with track masking.
    
    L3 SSH is along-track altimetry data (1D tracks, not swath like SWOT). 
    We use conservative binning to a ~7km grid without smoothing.
    
    Key differences from SWOT processing:
    - Input is 1D (along-track points), not 2D swath
    - Coarser target resolution (7km vs 2km) due to sparser data
    - No smoothing - preserves native track measurements
    
    Creates gridded data with additional track identification layers:
    - n_tracks: Number of primary-field tracks contributing to each cell
    - primary_track: Index of the first track to contribute to each cell (-1 = no data)
    - is_overlap: Boolean mask indicating cells where multiple tracks contributed
    - track_ids: List of track identifiers (filenames)
    - track_times: Timestamps for each track
    """
    files = load_l3_ssh_data(data_dir, date_str, return_paths_only=True)
    if not files:
        return 0, []
    
    out_dir = Path(output_dir) / "l3_ssh"
    out_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Processing {len(files)} L3_SSH files for {date_str}")
    
    records = []
    base_timestamp = pd.Timestamp(datetime.strptime(date_str, "%Y%m%d"))
    
    for region_name, bounds in SPATIAL_REGIONS.items():
        target_lons, target_lats, lon_edges, lat_edges = create_regional_grid(bounds, resolution_km)
        grid_shape = (len(target_lats), len(target_lons))
        
        # Data accumulation using observation-weighted means.
        # For each variable we track:
        #   accumulated_sum[v]        = Σ (cell_mean * cell_count)
        #   accumulated_count[v]      = Σ cell_count
        #   accumulated_track_sq[v]   = Σ (n_k * z̄_k²)
        #   accumulated_track_count[v]= number of contributing tracks per cell
        # Mean obs position uses shared lon/lat accumulators.
        accumulated_sum = {}
        accumulated_count = {}
        accumulated_track_sq = {}
        accumulated_track_count = {}
        accumulated_lon_sum = np.zeros(grid_shape, dtype=np.float64)
        accumulated_lat_sum = np.zeros(grid_shape, dtype=np.float64)
        accumulated_coord_count = np.zeros(grid_shape, dtype=np.int32)
        time_min, time_max = None, None

        # Track masking: primary_track stores first track index, -1 = no data
        primary_track = np.full(grid_shape, -1, dtype=np.int16)
        primary_track_count = np.zeros(grid_shape, dtype=np.int16)

        # Track metadata
        track_ids = []
        track_times = []
        track_platforms = []
        regional_track_idx = 0

        for fpath in files:
            ds = xr.open_dataset(fpath)

            # Get coordinates - L3 SSH uses longitude/latitude
            if 'longitude' in ds.coords:
                track_lons = lon_to_180(ds['longitude'].values)
                track_lats = ds['latitude'].values
            elif 'lon' in ds.coords:
                track_lons = lon_to_180(ds['lon'].values)
                track_lats = ds['lat'].values
            else:
                ds.close()
                continue

            if not swath_intersects_region(track_lons, track_lats, bounds):
                ds.close()
                continue

            # Get track time
            track_time = None
            if 'time' in ds.coords:
                t_vals = pd.to_datetime(np.asarray(ds['time'].values).flatten(), errors='coerce')
                t_vals = pd.Series(t_vals).dropna()
                if len(t_vals) > 0:
                    track_time = t_vals.mean()
                    if time_min is None or t_vals.min() < time_min:
                        time_min = t_vals.min()
                    if time_max is None or t_vals.max() > time_max:
                        time_max = t_vals.max()

            # Collect raw SLA data
            if 'sla_filtered' in ds:
                sla_raw = ds['sla_filtered'].values
            elif 'sla' in ds:
                sla_raw = ds['sla'].values
            else:
                ds.close()
                continue

            # Build per-pixel variable dict; compute ADT before binning so
            # NaN masks are consistent (adt is NaN wherever sla is NaN).
            swath_vars = {'sla_filtered': sla_raw}
            if 'mdt' in ds.data_vars:
                mdt_raw = ds['mdt'].values
                swath_vars['mdt'] = mdt_raw
                swath_vars['adt'] = sla_raw + mdt_raw

            # Bin all swath variables; retain primary-field provenance separately.
            primary_valid_mask = np.zeros(grid_shape, dtype=bool)
            any_contributed = False
            coord_updated_this_track = False

            for var_name, swath_data in swath_vars.items():
                grid_data, grid_counts, _grid_sem_unused, grid_mean_lon, grid_mean_lat = (
                    bin_swath_to_grid_conservative(
                        track_lons, track_lats, swath_data,
                        target_lons, target_lats,
                        method='mean',
                        min_samples=1,
                    )
                )
                valid_mask = grid_counts > 0
                if not valid_mask.any():
                    continue
                any_contributed = True
                if var_name == PRIMARY_L3_SSH_VAR:
                    primary_valid_mask |= valid_mask

                if var_name not in accumulated_sum:
                    accumulated_sum[var_name] = np.zeros(grid_shape, dtype=np.float64)
                    accumulated_count[var_name] = np.zeros(grid_shape, dtype=np.int32)
                    accumulated_track_sq[var_name] = np.zeros(grid_shape, dtype=np.float64)
                    accumulated_track_count[var_name] = np.zeros(grid_shape, dtype=np.int32)

                accumulated_sum[var_name][valid_mask] += (
                    grid_data[valid_mask] * grid_counts[valid_mask]
                )
                accumulated_count[var_name][valid_mask] += grid_counts[valid_mask]
                # Observation-weighted inter-track dispersion of per-track means.
                accumulated_track_sq[var_name][valid_mask] += (
                    grid_data[valid_mask] ** 2 * grid_counts[valid_mask]
                )
                accumulated_track_count[var_name][valid_mask] += 1

                # Accumulate mean obs position once per track (use first variable)
                if not coord_updated_this_track:
                    lon_valid = np.isfinite(grid_mean_lon) & valid_mask
                    lat_valid = np.isfinite(grid_mean_lat) & valid_mask
                    accumulated_lon_sum[lon_valid] += (
                        grid_mean_lon[lon_valid] * grid_counts[lon_valid]
                    )
                    accumulated_lat_sum[lat_valid] += (
                        grid_mean_lat[lat_valid] * grid_counts[lat_valid]
                    )
                    accumulated_coord_count[valid_mask] += grid_counts[valid_mask]
                    coord_updated_this_track = True

            if not any_contributed:
                ds.close()
                continue

            # Track contributed to this region
            track_ids.append(os.path.basename(fpath))
            track_times.append(track_time.isoformat() if track_time else '')
            track_platforms.append(ds.attrs.get('platform', 'unknown'))

            # Update primary-variable provenance only where the primary SSH field exists.
            new_cells = primary_valid_mask & (primary_track == -1)
            primary_track[new_cells] = regional_track_idx
            primary_track_count[primary_valid_mask] = np.minimum(
                primary_track_count[primary_valid_mask] + 1,
                np.iinfo(np.int16).max,
            )

            regional_track_idx += 1
            ds.close()

        if not accumulated_sum:
            continue

        # Compute observation-weighted average, variable-specific inter-track
        # dispersion, and mean obs position.
        data_vars = {}
        for var_name, data_sum in accumulated_sum.items():
            count = accumulated_count[var_name]
            with np.errstate(invalid='ignore', divide='ignore'):
                averaged = data_sum / np.maximum(count, 1)
            averaged[count == 0] = np.nan
            data_vars[var_name] = (
                ['time', 'lat', 'lon'], averaged[np.newaxis, ...].astype(np.float32)
            )

            # Observation-weighted dispersion of per-track means:
            # Var = Σ_k (n_k * z̄_k^2) / n - z̄^2
            # Defined only where at least two tracks contributed to this variable.
            n_tracks = accumulated_track_count[var_name]
            with np.errstate(invalid='ignore', divide='ignore'):
                between_var = (
                    accumulated_track_sq[var_name] / np.maximum(count, 1) - averaged ** 2
                )
            between_var = np.where(between_var > 0, between_var, 0.0)
            intertrack_std = np.sqrt(between_var)
            intertrack_std[n_tracks < 2] = np.nan
            data_vars[f'{var_name}_intertrack_std'] = (
                ['time', 'lat', 'lon'], intertrack_std[np.newaxis, ...].astype(np.float32)
            )

        # Mean observation position — will differ from grid-cell centre when
        # along-track points cluster near one edge of the bin.
        coord_count = np.maximum(accumulated_coord_count, 1)
        with np.errstate(invalid='ignore', divide='ignore'):
            mean_obs_lon = accumulated_lon_sum / coord_count
            mean_obs_lat = accumulated_lat_sum / coord_count
        mean_obs_lon[accumulated_coord_count == 0] = np.nan
        mean_obs_lat[accumulated_coord_count == 0] = np.nan
        data_vars['obs_mean_lon'] = (['time', 'lat', 'lon'], mean_obs_lon[np.newaxis, ...].astype(np.float32))
        data_vars['obs_mean_lat'] = (['time', 'lat', 'lon'], mean_obs_lat[np.newaxis, ...].astype(np.float32))

        # Use the primary SSH field to define n_obs, n_tracks, and overlap provenance.
        sla_count = accumulated_count.get(
            PRIMARY_L3_SSH_VAR, next(iter(accumulated_count.values()))
        )
        data_vars['n_obs'] = (['time', 'lat', 'lon'], sla_count[np.newaxis, ...].astype(np.int32))
        data_vars['n_tracks'] = (
            ['time', 'lat', 'lon'], primary_track_count[np.newaxis, ...].astype(np.int16)
        )
        data_vars['primary_track'] = (['time', 'lat', 'lon'], primary_track[np.newaxis, ...])
        data_vars['is_overlap'] = (
            ['time', 'lat', 'lon'], (primary_track_count > 1).astype(np.int8)[np.newaxis, ...]
        )

        # Create dataset with track metadata
        ds_out = xr.Dataset(
            data_vars,
            coords={
                'time': np.array([base_timestamp.to_datetime64()]),
                'lat': target_lats,
                'lon': target_lons,
                'track': np.arange(len(track_ids)),
            },
            attrs={
                'source': 'L3 SSH conservative regrid',
                'resolution_km': resolution_km,
                'n_tracks': len(track_ids),
                'processing': 'bin_mean_no_smoothing',
                'date': date_str,
                # CF-compliant CRS declaration (EPSG:4326 / WGS 84)
                'crs': 'EPSG:4326',
                'geospatial_lat_units': 'degrees_north',
                'geospatial_lon_units': 'degrees_east',
            },
        )
        ds_out['lat'].attrs.update({'units': 'degrees_north', 'standard_name': 'latitude', 'long_name': 'latitude'})
        ds_out['lon'].attrs.update({'units': 'degrees_east', 'standard_name': 'longitude', 'long_name': 'longitude'})

        # Add track metadata as data variables
        if track_ids:
            ds_out['track_ids'] = (['track'], track_ids)
            ds_out['track_times'] = (['track'], track_times)
            ds_out['track_platforms'] = (['track'], track_platforms)

        outfile = out_dir / f'l3_ssh_{region_name}_{date_str}.nc'

        # Set up encoding for all accumulated variables
        encoding = {}
        for var_name in list(accumulated_sum.keys()) + [
            f'{v}_intertrack_std' for v in accumulated_sum
        ] + ['obs_mean_lon', 'obs_mean_lat']:
            encoding[var_name] = {
                'dtype': 'float32',
                'zlib': True,
                'complevel': 4,
                '_FillValue': np.float32(np.nan),
            }
        encoding['n_obs'] = {'dtype': 'int32', 'zlib': True, '_FillValue': -1}
        encoding['n_tracks'] = {'dtype': 'int16', 'zlib': True, '_FillValue': -1}
        encoding['primary_track'] = {'dtype': 'int16', 'zlib': True, '_FillValue': -1}
        encoding['is_overlap'] = {'dtype': 'int8', 'zlib': True}
        
        ds_out.to_netcdf(outfile, encoding=encoding, engine="h5netcdf")
        
        bbox = [float(target_lons.min()), float(target_lats.min()),
                float(target_lons.max()), float(target_lats.max())]
        time_range = (time_min.timestamp(), time_max.timestamp()) if time_min else (None, None)
        
        sensor_str = ", ".join(sorted(set(track_platforms))) if track_platforms else "L3_SSH"
        
        records.append(make_record(
            outfile, output_dir, base_timestamp, "l3_ssh", "ssh",
            region_name, bbox, box(*bbox).wkb, time_range, sensor_str, 
            dataset=ds_out, gridded=True, n_tracks=len(track_ids)
        ))
    
    logging.info(f"✓ L3_SSH: {len(records)} regional files (conservative regrid)")
    return len(records), records


import matplotlib.pyplot as plt


def filter_and_plot_satellite_data(ds: xr.Dataset, satellites_to_keep: list[str], var_name="sla_filtered"):
    """
    Filters dataset to keep only specific satellites and plots the result.
    
    Args:
        ds: xarray Dataset containing 'track_platforms' and 'primary_track'
        satellites_to_keep: List of platform names (e.g. ['Sentinel-3A', 'Jason-3'])
        var_name: The variable to display (default: 'sla_filtered')
    """
    # 1. Identify track indices to keep
    #    track_platforms is a list of strings matching the track order
    platforms = ds["track_platforms"].values
    print("Available platforms in dataset:", platforms)
    keep_indices = [i for i, p in enumerate(platforms) if p in satellites_to_keep]
    
    # 2. Create the mask
    #    primary_track is the grid of track indices (-1 is empty)
    #    We mask cells where the track index is VALID but NOT in our keep list
    track_grid = ds["primary_track"].values
    mask_to_remove = (track_grid != -1) & (~np.isin(track_grid, keep_indices))
    
    # 3. Apply mask to data
    data_filtered = ds[var_name].copy()
    data_filtered.values[mask_to_remove] = np.nan
    
    # 4. Display the result
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    ds[var_name].plot(cmap='viridis')
    plt.title(f"Original Data\n({len(platforms)} tracks)")
    
    plt.subplot(1, 2, 2)
    data_filtered.plot(cmap='viridis')
    plt.title(f"Filtered Data\nIncluded: {', '.join(satellites_to_keep)}")
    
    plt.tight_layout()
    plt.savefig("filtered_satellite_data.png")
    plt.show()
    
    return data_filtered

def process_argo_data(data_dir, date_str, output_dir):
    """Process Argo float data - split points into regions."""
    ds = load_argo_data(data_dir, date_str)
    if ds is None:
        return 0, []
    
    try:
        if len(ds.N_POINTS) == 0:
            ds.close()
            return 0, []
        
        # Normalize coordinates
        if "LONGITUDE" in ds:
            lon_vals = lon_to_180(ds["LONGITUDE"].values)
            ds = ds.assign_coords(lon=("N_POINTS", lon_vals))
            if "LONGITUDE" in ds.coords:
                ds = ds.drop_vars("LONGITUDE")
        if "LATITUDE" in ds:
            ds = ds.rename({"LATITUDE": "lat"})
        
        lons = ds["lon"].values
        lats = ds["lat"].values
        
        out_dir = Path(output_dir) / "argo"
        out_dir.mkdir(parents=True, exist_ok=True)
        
        records = []
        base_timestamp = pd.Timestamp(datetime.strptime(date_str, "%Y%m%d"))
        
        for region_name, bounds in SPATIAL_REGIONS.items():
            mask = np.array([point_in_region(lon, lat, bounds) for lon, lat in zip(lons, lats)])
            
            if not mask.any():
                continue
            
            regional_ds = ds.isel(N_POINTS=mask)
            outfile = out_dir / f"argo_{region_name}_{date_str}.nc"
            regional_ds = clear_encoding(regional_ds)
            regional_ds.to_netcdf(outfile, engine="h5netcdf")
            
            reg_lons = regional_ds["lon"].values
            reg_lats = regional_ds["lat"].values
            bbox = [float(reg_lons.min()), float(reg_lats.min()),
                    float(reg_lons.max()), float(reg_lats.max())]
            
            time_range = posix_range_from_time(regional_ds["TIME"].values) if "TIME" in regional_ds else (None, None)
            
            # Note: Argo is point data, not gridded, so resolution doesn't apply
            records.append(make_record(
                outfile, output_dir, base_timestamp, "argo", "profiles",
                region_name, bbox, box(*bbox).wkb, time_range, "ARGO",
                dataset=None,  # No dataset for resolution calculation (point data)
                n_profiles=int(mask.sum())
            ))
        
        ds.close()
        logging.info(f"✓ Argo: {len(records)} regional files")
        return len(records), records
    
    except Exception as e:
        logging.warning(f"Error processing ARGO for {date_str}: {e}")
        return 0, []


def process_l3_sss_smos_data(data_dir, date_str, output_dir):
    """Process L3 SMOS SSS data - handle ascending and descending passes separately.
    
    SMOS has both ascending and descending satellite passes, which have different
    observation characteristics (time of day, viewing angle, etc.). These are 
    processed as separate data products to preserve this distinction.
    """
    smos_files = load_l3_sss_smos_data(data_dir, date_str)
    if not smos_files or (smos_files['asc'] is None and smos_files['desc'] is None):
        return 0, []
    
    all_records = []
    
    # Process ascending pass
    if smos_files['asc'] is not None:
        for fpath in smos_files['asc']:
            try:
                ds = xr.open_dataset(fpath)
                if "time" in ds.dims:
                    ds = ds.isel(time=slice(0, 1))
                ds = normalize_coords(ds)
                
                regional_data = split_gridded_into_regions(ds, SPATIAL_REGIONS)
                out_dir = Path(output_dir) / "l3_sss_smos_asc"
                out_dir.mkdir(parents=True, exist_ok=True)
                
                base_timestamp = pd.Timestamp(datetime.strptime(date_str, "%Y%m%d"))
                
                for region_name, region_info in regional_data.items():
                    if not region_info["intersects"]:
                        continue
                    
                    regional_ds = clear_encoding(region_info["dataset"])
                    outfile = out_dir / f"l3_sss_smos_asc_{region_name}_{date_str}.nc"
                    regional_ds.to_netcdf(outfile, encoding={v: NETCDF_ENCODINGS.get("l3_sss_smos_asc", {}) 
                                                              for v in regional_ds.data_vars}, engine="h5netcdf")
                    
                    all_records.append(make_record(
                        outfile, output_dir, base_timestamp, "l3_sss_smos_asc", "sss",
                        region_name, region_info["bbox"], region_info["geometry"],
                        region_info["time_range"], "SMOS_ASC", dataset=regional_ds,
                        satellite_pass="ascending"
                    ))
                
                ds.close()
            except Exception as e:
                logging.warning(f"  Error processing SMOS ascending file {fpath}: {e}")
    
    # Process descending pass
    if smos_files['desc'] is not None:
        for fpath in smos_files['desc']:
            try:
                ds = xr.open_dataset(fpath)
                if "time" in ds.dims:
                    ds = ds.isel(time=slice(0, 1))
                ds = normalize_coords(ds)
                
                regional_data = split_gridded_into_regions(ds, SPATIAL_REGIONS)
                out_dir = Path(output_dir) / "l3_sss_smos_desc"
                out_dir.mkdir(parents=True, exist_ok=True)
                
                base_timestamp = pd.Timestamp(datetime.strptime(date_str, "%Y%m%d"))
                
                for region_name, region_info in regional_data.items():
                    if not region_info["intersects"]:
                        continue
                    
                    regional_ds = clear_encoding(region_info["dataset"])
                    outfile = out_dir / f"l3_sss_smos_desc_{region_name}_{date_str}.nc"
                    encoding = {v: get_variable_encoding(v) for v in regional_ds.data_vars}
                    check_encoding_safety(regional_ds, encoding)

                    regional_ds.to_netcdf(
                        outfile,
                        encoding=encoding,
                        engine="h5netcdf",
                    )
                    
                    all_records.append(make_record(
                        outfile, output_dir, base_timestamp, "l3_sss_smos_desc", "sss",
                        region_name, region_info["bbox"], region_info["geometry"],
                        region_info["time_range"], "SMOS_DESC", dataset=regional_ds,
                        satellite_pass="descending"
                    ))
                
                ds.close()
            except Exception as e:
                logging.warning(f"  Error processing SMOS descending file {fpath}: {e}")
    
    logging.info(f"✓ L3_SSS_SMOS: {len(all_records)} regional files (asc + desc)")
    return len(all_records), all_records


# =============================================================================
# Track Masking Utilities
# =============================================================================

def mask_tracks_from_grid(data: np.ndarray, primary_track: np.ndarray, 
                          is_overlap: np.ndarray, exclude_track_indices: list,
                          preserve_overlaps: bool = False) -> np.ndarray:
    """Apply track-based masking to gridded data.
    
    Args:
        data: The gridded data array (lat, lon)
        primary_track: Array of primary track indices (-1 = no data)
        is_overlap: Boolean array indicating cells with multiple tracks
        exclude_track_indices: List of track indices to exclude
        preserve_overlaps: If True, don't mask cells where other tracks also contributed
    
    Returns:
        Masked copy of data with excluded track cells set to NaN
    """
    masked_data = data.copy()
    
    # Cells owned by excluded tracks
    exclude_mask = np.isin(primary_track, exclude_track_indices)
    
    if preserve_overlaps:
        # Don't mask cells that have overlap (other tracks also contributed)
        exclude_mask = exclude_mask & ~is_overlap.astype(bool)
    
    masked_data[exclude_mask] = np.nan
    return masked_data


def get_track_info_from_netcdf(nc_path: str) -> dict:
    """Extract track metadata from a gridded NetCDF file.
    
    Args:
        nc_path: Path to the gridded NetCDF file
    
    Returns:
        Dictionary with track_ids, track_times, n_tracks, and coverage statistics
    """
    ds = xr.open_dataset(nc_path)
    
    info = {
        "n_tracks": ds.attrs.get("n_tracks", 0),
        "track_ids": [],
        "track_times": [],
        "track_platforms": [],
        "track_coverage": {},  # track_idx -> number of cells
    }
    
    if "track_ids" in ds:
        info["track_ids"] = ds["track_ids"].values.tolist()
    if "track_times" in ds:
        info["track_times"] = ds["track_times"].values.tolist()
    if "track_platforms" in ds:
        info["track_platforms"] = ds["track_platforms"].values.tolist()
    
    # Compute coverage per track
    if "primary_track" in ds:
        primary_track = ds["primary_track"].values
        for track_idx in range(info["n_tracks"]):
            info["track_coverage"][track_idx] = int(np.sum(primary_track == track_idx))
    
    ds.close()
    return info


def apply_track_mask_to_netcdf(nc_path: str, exclude_track_indices: list,
                                preserve_overlaps: bool = False,
                                output_path: str = None) -> xr.Dataset:
    """Load a gridded NetCDF and apply track masking.
    
    Args:
        nc_path: Path to the gridded NetCDF file
        exclude_track_indices: List of track indices to exclude
        preserve_overlaps: If True, preserve data in overlapping cells
        output_path: If provided, save masked dataset to this path
    
    Returns:
        xarray Dataset with masked data
    """
    ds = xr.open_dataset(nc_path)
    
    if "primary_track" not in ds or "is_overlap" not in ds:
        logging.warning(f"No track masking info in {nc_path}")
        return ds
    
    primary_track = ds["primary_track"].values
    is_overlap = ds["is_overlap"].values
    
    # Identify data variables to mask (exclude metadata variables)
    skip_vars = {"primary_track", "is_overlap", "n_obs", "n_tracks", "track_ids", "track_times"}
    data_vars = [v for v in ds.data_vars if v not in skip_vars]
    
    # Apply masking to each data variable
    for var in data_vars:
        if set(ds[var].dims) >= {"lat", "lon"}:
            # Support both (lat, lon) and (time, lat, lon) layouts
            has_time = "time" in ds[var].dims
            arr = ds[var].values
            # Squeeze time for masking, then restore
            data_2d = arr[0] if has_time else arr
            masked = mask_tracks_from_grid(
                data_2d, primary_track, is_overlap,
                exclude_track_indices, preserve_overlaps
            )
            if has_time:
                ds[var] = (list(ds[var].dims), masked[np.newaxis, ...])
            else:
                ds[var] = (["lat", "lon"], masked)
    
    if output_path:
        ds.to_netcdf(output_path, engine="h5netcdf")
        logging.info(f"Saved masked data to {output_path}")
    
    return ds


def split_tracks_for_validation(nc_path: str, val_fraction: float = 0.2,
                                 random_seed: int = 42) -> tuple[list, list]:
    """Split tracks into training and validation sets.
    
    Args:
        nc_path: Path to the gridded NetCDF file
        val_fraction: Fraction of tracks to hold out for validation
        random_seed: Random seed for reproducibility
    
    Returns:
        Tuple of (train_track_indices, val_track_indices)
    """
    info = get_track_info_from_netcdf(nc_path)
    n_tracks = info["n_tracks"]
    
    if n_tracks == 0:
        return [], []
    
    np.random.seed(random_seed)
    all_indices = np.arange(n_tracks)
    np.random.shuffle(all_indices)
    
    n_val = max(1, int(n_tracks * val_fraction))
    val_indices = all_indices[:n_val].tolist()
    train_indices = all_indices[n_val:].tolist()
    
    return train_indices, val_indices


def process_date(date_str, data_dir, output_dir, include_l3_swot=True, include_l3_ssh=True, include_argo=True, only_vars=None):
    """Process all data sources for a single date."""
    logging.info(f"Processing date: {date_str}")
    all_records = []
    
    processors = [
        ("glorys", lambda: process_glorys_data(load_glorys_data(data_dir, date_str), date_str, output_dir)),
        ("l4_ssh", lambda: process_and_split(load_l4_ssh_data(data_dir, date_str), date_str, output_dir, "l4_ssh", sensor="L4_SSH")),
        ("l4_sst", lambda: process_and_split(load_l4_sst_data(data_dir, date_str), date_str, output_dir, "l4_sst", sensor="L4_SST")),
        ("l4_sss", lambda: process_and_split(load_l4_sss_data(data_dir, date_str), date_str, output_dir, "l4_sss", sensor="L4_SSS")),
        ("l4_wind", lambda: process_and_split(load_l4_wind_data(data_dir, date_str), date_str, output_dir, "l4_wind", sensor="L4_WIND")),
        ("l3_sst", lambda: process_and_split(load_l3_sst_data(data_dir, date_str), date_str, output_dir, "l3_sst", sensor="L3_SST")),
        ("l3_sss_smos", lambda: process_l3_sss_smos_data(data_dir, date_str, output_dir)),
    ]
    

    if include_l3_swot:
        processors.append(("l3_swot", lambda: process_swot_daily_gridded(data_dir, date_str, output_dir)))
    if include_l3_ssh:
        processors.append(("l3_ssh", lambda: process_l3_ssh_data(data_dir, date_str, output_dir)))
    if include_argo:
        processors.append(("argo", lambda: process_argo_data(data_dir, date_str, output_dir)))
    
    if only_vars:
        processors = [p for p in processors if p[0] in only_vars]

    for name, processor in processors:
        try:
            count, records = processor()
            # Check for silent failures on critical datasets
            if count == 0 and name in ["glorys", "l4_ssh"]:
                logging.warning(f"  [CRITICAL MISSING] No records created for {name} on {date_str}. Check source files!")
            elif count == 0:
                 logging.debug(f"  No records for {name} on {date_str}")
            all_records.extend(records)
        except Exception as e:
            logging.warning(f"  Error processing {name}: {e}")
            import traceback
            traceback.print_exc()
    
    return len(all_records), all_records


def generate_date_list(date_min, date_max):
    """Generate list of date strings."""
    start = datetime.strptime(date_min, "%Y-%m-%d")
    end = datetime.strptime(date_max, "%Y-%m-%d")
    dates = []
    cur = start
    while cur <= end:
        dates.append(cur.strftime("%Y%m%d"))
        cur += timedelta(days=1)
    return dates


def create_inventory(records, output_path):
    """Create file inventory and save to parquet."""
    if not records:
        return None
    df = pd.DataFrame(records)
    df = df.sort_values(["timestamp_file", "data_source"]).reset_index(drop=True)
    df["date_str"] = df["timestamp_file"].dt.strftime("%Y%m%d")
    df.to_parquet(output_path, index=False)
    logging.info(f"✓ Inventory saved: {output_path} ({len(df)} files)")
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date-min", default="2024-01-01")
    parser.add_argument("--date-max", default="2024-01-04")
    parser.add_argument("--data-dir", default="./ssh_state_data")
    parser.add_argument("--output-dir", default="./formatted_ssh_data")
    parser.add_argument("--inventory-path", default="file_inventory.parquet")
    parser.add_argument("--processes", "-p", type=int, default=2)
    parser.add_argument("--include-l3-swot", action="store_true", default=True)
    parser.add_argument("--include-l3-ssh", action="store_true", default=True)
    parser.add_argument("--include-argo", action="store_true", default=True)
    parser.add_argument("--only-vars", nargs="+", help="Only process specific variables (e.g. l4_ssh)")
    parser.add_argument("--update-existing-inventory", action="store_true", help="Update existing inventory file instead of overwriting")
    args = parser.parse_args()
    
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    # Pre-create all subdirectories before spawning workers to avoid race conditions
    # on parallel/network filesystems (Lustre/GPFS) where mkdir visibility is not instant.
    subdirs = ["glorys", "l4_ssh", "l4_sst", "l4_sss", "l4_wind",
               "l3_sst", "l3_sss_smos_asc", "l3_sss_smos_desc",
               "l3_swot", "l3_ssh", "argo"]
    for subdir in subdirs:
        Path(args.output_dir, subdir).mkdir(parents=True, exist_ok=True)

    date_list = generate_date_list(args.date_min, args.date_max)
    logging.info(f"Processing {len(date_list)} dates with {args.processes} workers")

    process_func = partial(process_date, data_dir=args.data_dir, output_dir=args.output_dir,
                          include_l3_swot=args.include_l3_swot, include_l3_ssh=args.include_l3_ssh,
                          include_argo=args.include_argo, only_vars=args.only_vars)

    if args.processes > 1:
        with mp.Pool(processes=args.processes) as pool:
            results = pool.map(process_func, date_list)
    else:
        results = [process_func(d) for d in date_list]
    
    all_records = [rec for _, recs in results for rec in recs]
    
    if args.update_existing_inventory:
        inventory_path = os.path.join(args.output_dir, args.inventory_path)
        if os.path.exists(inventory_path):
            logging.info(f"Updating existing inventory at {inventory_path}")
            df_existing = pd.read_parquet(inventory_path)
            
            # Determine which data_source values were actually processed.
            # Important: when --only-vars is used, do NOT append optional
            # datasets from include_* flags, otherwise unrelated sources can be
            # removed from the existing inventory.
            if args.only_vars:
                processed_vars = list(args.only_vars)
            else:
                processed_vars = [
                    "glorys", "l4_ssh", "l4_sst", "l4_sss", "l4_wind", "l3_sst", "l3_sss_smos"
                ]
                if args.include_l3_swot:
                    processed_vars.append("l3_swot")
                if args.include_l3_ssh:
                    processed_vars.append("l3_ssh")
                if args.include_argo:
                    processed_vars.append("argo")

            # Expand logical aliases to concrete data_source values.
            expanded_processed_vars = []
            for src in processed_vars:
                if src == "l3_sss_smos":
                    expanded_processed_vars.extend(["l3_sss_smos_asc", "l3_sss_smos_desc"])
                else:
                    expanded_processed_vars.append(src)
            processed_vars = sorted(set(expanded_processed_vars))
            
            date_strs = set(date_list)
            
            # Remove old records for valid dates and variables
            mask_vars = df_existing['data_source'].isin(processed_vars)
            mask_dates = df_existing['date_str'].isin(list(date_strs))
            mask_remove = mask_vars & mask_dates
            
            logging.info(f"Removing {mask_remove.sum()} old records matching processed dates/vars.")
            df_existing = df_existing[~mask_remove]
            
            if all_records:
                new_df = pd.DataFrame(all_records)
                # Ensure date_str column exists in new_df for consistency if needed for next steps (create_inventory adds it anyway)
                 # But we convert to dicts, so concat works on dfs.
                combined_df = pd.concat([df_existing, new_df], ignore_index=True)
                all_records = combined_df.to_dict('records')
            else:
                 all_records = df_existing.to_dict('records')
        else:
            logging.warning(f"Inventory {inventory_path} not found for update, creating new.")

    create_inventory(all_records, os.path.join(args.output_dir, args.inventory_path))
    logging.info("✓ Done!")


if __name__ == "__main__":
    main()



# python sea_state_ds/new_format_ssh_data.py --date-min 2023-03-29   --date-max 2023-04-15   --data-dir /p/project1/hai_uqmethodbox/data/new_ssh_dataset   --output-dir /p/project1/hai_uqmethodbox/data/new_ssh_dataset_formatted_region   --inventory-path /p/project1/hai_uqmethodbox/data/new_ssh_dataset_formatted_region/file_collection_swot_period.parquet


# python sea_state_ds/new_format_ssh_data.py --date-min 2023-03-29   --date-max 2023-04-15   --data-dir /p/project1/hai_uqmethodbox/data/new_ssh_dataset   --output-dir /p/project1/hai_uqmethodbox/data/new_ssh_dataset_formatted_region   --inventory-path /p/project1/hai_uqmethodbox/data/new_ssh_dataset_formatted_region/file_collection_swot_period.parquet
    