"""Dataset utilities."""

from enum import Enum

import numpy as np
import xarray as xr

# # Add near the top after imports
# class TemporalAggregation(Enum):
#     """Temporal aggregation strategies for time-range queries."""

#     NONE = "none"  # Keep all timesteps as separate dimension
#     MEAN = "mean"  # Average over time
#     LAST = "last"  # Take only last timestep
#     FIRST = "first"  # Take only first timestep


# def _lon_mask_180(lon: xr.DataArray | np.ndarray, lon_min: float, lon_max: float):
#     """Dateline-aware mask for longitudes in [-180, 180]."""
#     lon_vals = lon if isinstance(lon, xr.DataArray) else xr.DataArray(lon)
#     if lon_min <= lon_max:
#         return (lon_vals >= lon_min) & (lon_vals <= lon_max)
#     return (lon_vals >= lon_min) | (lon_vals <= lon_max)


# def _alongtrack_intersects_bbox(
#     df: pd.DataFrame, bbox: tuple[float, float, float, float]
# ) -> pd.DataFrame:
#     """More precise filter for alongtrack data: check if track actually passes through bbox.

#     Opens each file and checks if any points fall within the bbox.
#     More expensive than envelope check but necessary for thin alongtrack data.

#     Args:
#         df: DataFrame with L3 alongtrack file metadata
#         bbox: (lon_min, lon_max, lat_min, lat_max)

#     Returns:
#         Filtered DataFrame with only files that have points in bbox
#     """
#     lon_min, lon_max, lat_min, lat_max = bbox

#     valid_rows = []
#     for idx, row in df.iterrows():
#         file_path = row["internal:gdal_vsi"]

#         try:
#             # Quick check: open file and sample coordinates
#             byte_stream = _retrieve_byte_stream_static(file_path)
#             with xr.open_dataset(byte_stream, engine="h5netcdf") as ds:
#                 if "lat" not in ds.coords or "lon" not in ds.coords:
#                     continue

#                 lat = ds["lat"].values
#                 lon = ds["lon"].values

#                 # Check if any points fall in bbox
#                 if lon_min <= lon_max:
#                     lon_mask = (lon >= lon_min) & (lon <= lon_max)
#                 else:
#                     lon_mask = (lon >= lon_min) | (lon <= lon_max)

#                 lat_mask = (lat >= lat_min) & (lat <= lat_max)

#                 if np.any(lon_mask & lat_mask):
#                     valid_rows.append(idx)
#         except Exception:
#             continue

#     return df.loc[valid_rows]


# def _retrieve_byte_stream_static(file_path: str) -> io.BytesIO:
#     """Static version of byte stream retrieval for use in filtering."""
#     pattern = r"(\d+)_(\d+),(.+)"
#     match = re.search(pattern, file_path)
#     if not match:
#         raise ValueError(f"Invalid file path format: {file_path}")

#     offset = int(match.group(1))
#     size = int(match.group(2))
#     file_name = match.group(3)

#     with open(file_name, "rb") as f:
#         f.seek(offset)
#         data = f.read(size)

#     return io.BytesIO(data)


# # def select_bbox_swot_swath(
# #     ds: xr.Dataset, bbox: tuple[float, float, float, float]
# # ) -> xr.Dataset:
# #     """Subset SWOT L2/L3 swath data by bounding box.

# #     Strategy:
# #     1. Create spatial mask where (lon, lat) fall within bbox
# #     2. Keep only rows (num_lines) that have at least one valid pixel in bbox
# #     3. Mask out pixels outside bbox (set to NaN)
# #     4. Drop completely empty rows

# #     Args:
# #         ds: xarray Dataset with 2D (num_lines, num_pixels) lon/lat coords
# #         bbox: (lon_min, lon_max, lat_min, lat_max)

# #     Returns:
# #         Subsetted dataset with only relevant rows and masked pixels
# #     """
# #     lon_min, lon_max, lat_min, lat_max = bbox
# #     lon = ds["lon"]
# #     lat = ds["lat"]

# #     # Require 2D swath coordinates
# #     if lon.ndim != 2 or lat.ndim != 2:
# #         return ds

# #     dim_rows, dim_cols = lon.dims  # e.g., ("num_lines", "num_pixels")

# #     # Step 1: Create spatial mask for pixels in bbox
# #     lon_in = (lon >= lon_min) & (lon <= lon_max)
# #     lat_in = (lat >= lat_min) & (lat <= lat_max)
# #     spatial_mask = lon_in & lat_in  # Shape: (num_lines, num_pixels)

# #     # Step 2: Keep only rows that have at least one pixel in bbox
# #     rows_with_data = spatial_mask.any(dim=dim_cols)
# #     if not bool(rows_with_data.any()):
# #         # No rows intersect bbox
# #         return ds.isel({dim_rows: slice(0, 0)})

# #     # Get row indices to keep
# #     valid_row_indices = np.where(rows_with_data.values)[0]
# #     ds_subset = ds.isel({dim_rows: valid_row_indices})
# #     mask_subset = spatial_mask.isel({dim_rows: valid_row_indices})

# #     # Step 3: Mask out pixels outside bbox (set to NaN)
# #     for var_name in ds_subset.data_vars:
# #         if (
# #             dim_rows in ds_subset[var_name].dims
# #             and dim_cols in ds_subset[var_name].dims
# #         ):
# #             ds_subset[var_name] = ds_subset[var_name].where(mask_subset)

# #     # Mask coordinates too
# #     ds_subset["lon"] = ds_subset["lon"].where(mask_subset)
# #     ds_subset["lat"] = ds_subset["lat"].where(mask_subset)

# #     # Step 4: Drop rows where ALL pixels are NaN in the primary variable
# #     # Use first 2D data variable to determine empty rows
# #     data_vars_2d = [
# #         name
# #         for name in ds_subset.data_vars
# #         if dim_rows in ds_subset[name].dims and dim_cols in ds_subset[name].dims
# #     ]

# #     if data_vars_2d:
# #         primary_var = data_vars_2d[0]  # e.g., "ssha_filtered"
# #         rows_with_valid_data = ds_subset[primary_var].notnull().any(dim=dim_cols)

# #         if bool(rows_with_valid_data.any()):
# #             valid_indices = np.where(rows_with_valid_data.values)[0]
# #             ds_subset = ds_subset.isel({dim_rows: valid_indices})
# #         else:
# #             # All rows are NaN
# #             return ds.isel({dim_rows: slice(0, 0)})

# # return ds_subset


def select_bbox_swot_swath(
    ds: xr.Dataset, bbox: tuple[float, float, float, float]
) -> xr.Dataset:
    """Extracts the full-width swath lines that intersect a bounding box.

    Preserves the (num_lines, 69) structure to facilitate stacking.
    Pixels outside the bounding box are masked to NaN.

    Args:
        ds: Xarray Dataset with 2D coords
        bbox: (lon_min, lon_max, lat_min, lat_max)

    Returns:
        Dataset sliced along 'num_lines' only.
    """
    lon_min, lon_max, lat_min, lat_max = bbox

    # --- 1. HANDLE LONGITUDE WRAPPING (CRITICAL) ---
    # Detect if dataset is 0-360 but bbox is -180/180
    ds_lons = ds.lon.values
    if ds_lons.max() > 180 and lon_min < 0:
        # Wrap bbox negative lons to 0-360
        lon_min = lon_min % 360
        lon_max = lon_max % 360

    # --- 2. COMPUTE INTERSECTION MASK ---
    # Determine which pixels fall inside the box
    mask = (
        (ds.lat.values >= lat_min)
        & (ds.lat.values <= lat_max)
        & (ds_lons >= lon_min)
        & (ds_lons <= lon_max)
    )

    # If no intersection, return empty dataset (or None)
    if not np.any(mask):
        return ds.isel(num_lines=slice(0, 0))  # Return empty slice with correct dims

    # --- 3. IDENTIFY RELEVANT LINES ---
    # We want to keep any LINE that has at least one valid pixel.
    # Reduce mask along column axis (axis 1 = num_pixels)
    lines_with_data = np.any(mask, axis=1)

    # Find the start and stop indices for the lines
    # valid_line_indices is an array of indices where the line touched the box
    valid_line_indices = np.where(lines_with_data)[0]

    l_min = valid_line_indices.min()
    l_max = valid_line_indices.max()

    # --- 4. SLICE (LINES ONLY) ---
    # We slice num_lines, but we DO NOT slice num_pixels.
    # This ensures the output width remains 69.
    ds_subset = ds.isel(num_lines=slice(l_min, l_max + 1))

    # Slice the mask to match the new num_lines size
    mask_subset = mask[l_min : l_max + 1, :]

    # --- 5. APPLY MASK (OPTIONAL BUT RECOMMENDED) ---
    # This sets pixels OUTSIDE the bbox to NaN, but keeps the 69-pixel width.
    # If you want to keep the "context" (valid data outside the box on those lines),
    # you can skip this step.

    # Convert numpy mask to DataArray for xarray alignment
    mask_da = xr.DataArray(
        mask_subset,
        dims=("num_lines", "num_pixels"),
        coords={"lat": ds_subset.lat, "lon": ds_subset.lon},  # Optional but safe
    )

    ds_masked = ds_subset.where(mask_da)

    return ds_masked


# # TODO do we need to check on the side of the geometry that is encoded into taco
def select_bbox_alongtrack(
    ds: xr.Dataset, bbox: tuple[float, float, float, float]
) -> xr.Dataset:
    """Keep only observations within bbox = (lon_min, lon_max, lat_min, lat_max).

    Applies strict coordinate clipping after filtering.
    """
    lon_min, lon_max, lat_min, lat_max = bbox
    lon = ds["lon"]
    lat = ds["lat"]

    # Dateline-crossing bbox if lon_min > lon_max
    if lon_min <= lon_max:
        lon_mask = (lon >= lon_min) & (lon <= lon_max)
    else:
        lon_mask = (lon >= lon_min) | (lon <= lon_max)

    lat_mask = (lat >= lat_min) & (lat <= lat_max)
    mask = lon_mask & lat_mask

    if not bool(mask.any()):
        try:
            return ds.isel({ds.dims[0]: slice(0, 0)})
        except Exception:
            first_dim = next(iter(ds.dims))
            return ds.isel({first_dim: slice(0, 0)})

    result = ds.where(mask, drop=True)

    # Strict clipping to bbox bounds
    if len(result["lon"]) > 0:
        lat_vals = result["lat"].values.copy()
        lon_vals = result["lon"].values.copy()

        lat_finite = np.isfinite(lat_vals)
        lon_finite = np.isfinite(lon_vals)

        # Clip latitude
        lat_vals[lat_finite] = np.clip(lat_vals[lat_finite], lat_min, lat_max)

        # Clip longitude
        if lon_min <= lon_max:
            lon_vals[lon_finite] = np.clip(lon_vals[lon_finite], lon_min, lon_max)
        else:
            # Dateline crossing
            left_side = lon_finite & (lon_vals >= lon_min)
            right_side = lon_finite & (lon_vals <= lon_max)

            lon_vals[left_side] = np.clip(lon_vals[left_side], lon_min, 180)
            lon_vals[right_side] = np.clip(lon_vals[right_side], -180, lon_max)

        # Update coordinates
        result["lat"].values[:] = lat_vals
        result["lon"].values[:] = lon_vals

    return result


# def select_bbox_grid_1d(
#     ds: xr.Dataset, bbox: tuple[float, float, float, float]
# ) -> xr.Dataset:
#     """Subset gridded data with 1D lat/lon coords by bbox. Handles antimeridian bboxes.

#     Args:
#         ds: Dataset with 1D 'lat' and 'lon' coordinates
#         bbox: (lon_min, lon_max, lat_min, lat_max)

#     Returns:
#         Subsetted dataset
#     """
#     lon_min, lon_max, lat_min, lat_max = bbox

#     if "lat" not in ds.coords or "lon" not in ds.coords:
#         raise ValueError("Dataset must have 'lat' and 'lon' coordinates")

#     lat = ds.coords["lat"]
#     lon = ds.coords["lon"]

#     lat_mask = (lat >= lat_min) & (lat <= lat_max)

#     if lon_min <= lon_max:
#         lon_mask = (lon >= lon_min) & (lon <= lon_max)
#     else:
#         lon_mask = (lon >= lon_min) | (lon <= lon_max)

#     if not bool(lat_mask.any()) or not bool(lon_mask.any()):
#         # Return empty dataset with same structure
#         return ds.isel(lat=slice(0, 0), lon=slice(0, 0))

#     ds_subset = ds.where(lat_mask, drop=True).where(lon_mask, drop=True)

#     return ds_subset


# def _numeric_envelope_prefilter(
#     df: pd.DataFrame, wkb_col: str, bbox: tuple[float, float, float, float]
# ) -> pd.DataFrame:
#     """Pre-filter DataFrame rows by spatial envelope intersection with bbox.

#     Assumes all coordinates use -180 to 180 longitude convention.

#     Args:
#         df: DataFrame with WKB geometry column
#         wkb_col: Name of column containing WKB geometries
#         bbox: (lon_min, lon_max, lat_min, lat_max) in -180 to 180 convention

#     Returns:
#         Filtered DataFrame with rows whose envelopes intersect bbox
#     """
#     from shapely import from_wkb

#     lon_min, lon_max, lat_min, lat_max = bbox
#     xs_min, xs_max, ys_min, ys_max = [], [], [], []

#     for wkt in df[wkb_col].values:
#         g = from_wkb(wkt)
#         xmin, ymin, xmax, ymax = g.bounds

#         xs_min.append(xmin)
#         xs_max.append(xmax)
#         ys_min.append(ymin)
#         ys_max.append(ymax)

#     df2 = df.assign(_xmin=xs_min, _xmax=xs_max, _ymin=ys_min, _ymax=ys_max)

#     # Check for dateline-crossing bbox (lon_min > lon_max)
#     if lon_min <= lon_max:
#         # Standard bbox (doesn't cross dateline)
#         lon_intersect = (df2._xmax >= lon_min) & (df2._xmin <= lon_max)
#     else:
#         # Dateline-crossing bbox: envelope intersects if it touches either side
#         # Left side: envelope extends into [lon_min, 180]
#         # Right side: envelope extends into [-180, lon_max]
#         left_side = (df2._xmax >= lon_min) | (df2._xmin >= lon_min)
#         right_side = (df2._xmin <= lon_max) | (df2._xmax <= lon_max)
#         lon_intersect = left_side | right_side

#     lat_intersect = (df2._ymax >= lat_min) & (df2._ymin <= lat_max)
#     mask = lon_intersect & lat_intersect

#     return df2.loc[mask].drop(columns=["_xmin", "_xmax", "_ymin", "_ymax"])


# def project_points_to_pixel_grid(
#     lon: np.ndarray,
#     lat: np.ndarray,
#     val: np.ndarray,
#     bbox: tuple[float, float, float, float],
#     nlon: int,
#     nlat: int,
#     agg: str = "mean",
#     dateline_wrap: bool = False,
# ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
#     """Bin (lon, lat, val) onto a fixed pixel grid defined by bbox + (nlon, nlat).
#     No interpolation; pure aggregation. Empty cells remain NaN.

#     Returns:
#         Tuple of (lat_centers, lon_centers, grid, counts), all as numpy arrays
#         - lat_centers: shape (nlat,)
#         - lon_centers: shape (nlon,)
#         - grid: shape (nlat, nlon)
#         - counts: shape (nlat, nlon)
#     """
#     lon = np.asarray(lon).ravel()
#     lat = np.asarray(lat).ravel()
#     val = np.asarray(val).ravel()

#     mask = np.isfinite(lon) & np.isfinite(lat) & np.isfinite(val)
#     lon, lat, val = lon[mask], lat[mask], val[mask]

#     lon_min, lon_max, lat_min, lat_max = bbox

#     if (lon_min > lon_max) and not dateline_wrap:
#         raise ValueError(
#             "bbox crosses dateline (lon_min > lon_max); split bbox or set dateline_wrap=True."
#         )

#     if dateline_wrap and lon_min > lon_max:
#         lon_wrapped = lon.copy()
#         lon_wrapped[lon < lon_min] += 360.0
#         lon_max_adj = lon_max + 360.0
#         inside = (
#             (lon_wrapped >= lon_min)
#             & (lon_wrapped <= lon_max_adj)
#             & (lat >= lat_min)
#             & (lat <= lat_max)
#         )
#         lon_use = lon_wrapped[inside]
#         lat_use = lat[inside]
#         val_use = val[inside]
#     else:
#         inside = (
#             (lon >= lon_min) & (lon <= lon_max) & (lat >= lat_min) & (lat <= lat_max)
#         )
#         lon_use = lon[inside]
#         lat_use = lat[inside]
#         val_use = val[inside]

#     dlon = (lon_max - lon_min) / nlon
#     dlat = (lat_max - lat_min) / nlat
#     lon_centers = lon_min + (0.5 + np.arange(nlon)) * dlon
#     lat_centers = lat_min + (0.5 + np.arange(nlat)) * dlat

#     grid = np.full((nlat, nlon), np.nan, dtype=float)
#     counts = np.zeros((nlat, nlon), dtype=int)

#     if lon_use.size == 0:
#         return lat_centers, lon_centers, grid, counts

#     ilon = np.floor((lon_use - lon_min) / dlon).astype(int)
#     ilat = np.floor((lat_use - lat_min) / dlat).astype(int)
#     ilon = np.clip(ilon, 0, nlon - 1)
#     ilat = np.clip(ilat, 0, nlat - 1)

#     if agg in {"mean", "sum"}:
#         sum_grid = np.zeros((nlat, nlon), dtype=float)
#         np.add.at(sum_grid, (ilat, ilon), val_use)
#         np.add.at(counts, (ilat, ilon), 1)
#         if agg == "sum":
#             grid = sum_grid
#             grid[counts == 0] = np.nan
#         else:
#             nonzero = counts > 0
#             grid[nonzero] = sum_grid[nonzero] / counts[nonzero]
#     elif agg == "count":
#         np.add.at(counts, (ilat, ilon), 1)
#         grid = counts.astype(float)
#         grid[counts == 0] = np.nan
#     else:
#         raise ValueError(f"Unknown agg='{agg}'")

#     return lat_centers, lon_centers, grid, counts

"""Utility functions for OceanTACO dataset."""

import xarray as xr


class TemporalAggregation(Enum):
    """Temporal aggregation strategies for multi-timestep queries."""

    NONE = "none"
    MEAN = "mean"
    FIRST = "first"
    LAST = "last"


# 8 equal rectangles spanning the globe

SPATIAL_REGIONS = {
    "SOUTH_PACIFIC_WEST": {"lat": (-90, 0), "lon": (-180, -90)},
    "SOUTH_ATLANTIC": {"lat": (-90, 0), "lon": (-90, 0)},
    "SOUTH_INDIAN": {"lat": (-90, 0), "lon": (0, 90)},
    "SOUTH_PACIFIC_EAST": {"lat": (-90, 0), "lon": (90, 180)},
    "NORTH_PACIFIC_WEST": {"lat": (0, 90), "lon": (-180, -90)},
    "NORTH_ATLANTIC": {"lat": (0, 90), "lon": (-90, 0)},
    "NORTH_INDIAN": {"lat": (0, 90), "lon": (0, 90)},
    "NORTH_PACIFIC_EAST": {"lat": (0, 90), "lon": (90, 180)},
}


def get_regions_for_bbox(bbox: tuple[float, float, float, float]) -> list[str]:
    """Get list of regions that intersect a bounding box.

    Args:
        bbox: (lon_min, lon_max, lat_min, lat_max)

    Returns:
        List of region names that intersect the bbox
    """
    lon_min, lon_max, lat_min, lat_max = bbox
    intersecting = []

    for name, bounds in SPATIAL_REGIONS.items():
        reg_lon_min, reg_lon_max = bounds["lon"]
        reg_lat_min, reg_lat_max = bounds["lat"]

        # Check latitude overlap
        if lat_max <= reg_lat_min or lat_min >= reg_lat_max:
            continue

        # Check longitude overlap
        if lon_max <= reg_lon_min or lon_min >= reg_lon_max:
            continue

        intersecting.append(name)

    return intersecting


def select_bbox_gridded(
    ds: xr.Dataset, bbox: tuple[float, float, float, float]
) -> xr.Dataset:
    """Select data within bounding box for gridded datasets.

    Args:
        ds: xarray Dataset with 1D 'lat' and 'lon' coordinates
        bbox: (lon_min, lon_max, lat_min, lat_max)

    Returns:
        Subset of dataset within bbox
    """
    lon_min, lon_max, lat_min, lat_max = bbox

    lons = ds["lon"].values
    lats = ds["lat"].values
    lon_mask = (lons >= lon_min) & (lons <= lon_max)
    lat_mask = (lats >= lat_min) & (lats <= lat_max)

    return ds.isel(lon=lon_mask, lat=lat_mask)


def select_bbox_points(
    ds: xr.Dataset,
    bbox: tuple[float, float, float, float],
    lon_var: str = "lon",
    lat_var: str = "lat",
    dim: str = "N_POINTS",
) -> xr.Dataset:
    """Select point data within bounding box (e.g., Argo floats)."""
    lon_min, lon_max, lat_min, lat_max = bbox

    lons = ds[lon_var].values
    lats = ds[lat_var].values

    lon_mask = (lons >= lon_min) & (lons <= lon_max)
    lat_mask = (lats >= lat_min) & (lats <= lat_max)
    mask = lon_mask & lat_mask

    return ds.isel({dim: mask})


def get_colormap_params(var_name: str) -> dict:
    """Get visualization parameters for a variable."""
    var_lower = var_name.lower()

    if "ssh" in var_lower or "swot" in var_lower or "sla" in var_lower:
        return {"vmin": -0.5, "vmax": 0.5, "cmap": "RdBu_r", "label": "SSH (m)"}
    elif "sst" in var_lower or "temp" in var_lower:
        return {"vmin": 0, "vmax": 30, "cmap": "RdYlBu_r", "label": "SST (°C)"}
    elif "sss" in var_lower or "sal" in var_lower:
        return {"vmin": 32, "vmax": 38, "cmap": "viridis", "label": "SSS (PSU)"}
    elif "wind" in var_lower:
        return {"vmin": -15, "vmax": 15, "cmap": "coolwarm", "label": "Wind (m/s)"}
    elif "uo" in var_lower or "vo" in var_lower:
        return {"vmin": -1, "vmax": 1, "cmap": "coolwarm", "label": "Current (m/s)"}
    else:
        return {"vmin": None, "vmax": None, "cmap": "viridis", "label": "Value"}
