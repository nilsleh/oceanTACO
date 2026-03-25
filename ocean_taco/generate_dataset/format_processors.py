"""Processing and track-masking logic for the formatting pipeline."""

import logging
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from shapely.geometry import box

from ocean_taco.generate_dataset.format_constants import L3_SWOT_VARS, SPATIAL_REGIONS
from ocean_taco.generate_dataset.format_coords import (
    compute_resolution,
    lon_to_180,
    normalize_coords,
    point_in_region,
    posix_range_from_time,
    split_gridded_into_regions,
)
from ocean_taco.generate_dataset.format_encoding import (
    check_encoding_safety,
    clear_encoding,
    get_variable_encoding,
)
from ocean_taco.generate_dataset.format_gridding import (
    bin_swath_to_grid_conservative,
    create_regional_grid,
    swath_intersects_region,
)
from ocean_taco.generate_dataset.format_loaders import (
    load_argo_data,
    load_l3_ssh_data,
    load_l3_sss_smos_data,
    load_l3_swot_data,
)


def _extract_track_time_stats(ds: xr.Dataset, coord_name: str = "time"):
    """Extract representative track time and min/max bounds from a time coordinate."""
    if coord_name not in ds.coords:
        return None, None, None

    t_vals = pd.to_datetime(np.asarray(ds[coord_name].values).flatten(), errors="coerce")
    t_vals = pd.Series(t_vals).dropna()
    if len(t_vals) == 0:
        return None, None, None

    return t_vals.mean(), t_vals.min(), t_vals.max()


def make_record(
    outfile,
    output_dir,
    base_timestamp,
    data_source,
    variable,
    region_name,
    bbox,
    geometry_wkb,
    time_range,
    sensor,
    dataset=None,
    **extra,
):
    """Create a standardized record dict with microsecond timestamps and resolution info.

    Args:
        outfile: Absolute output file path.
        output_dir: Root output directory used to compute relative paths.
        base_timestamp: Timestamp associated with the record date.
        data_source: Modality/source identifier (for example ``l4_ssh``).
        variable: Logical variable tag for the saved output.
        region_name: Spatial region name.
        bbox: Region bounding box as ``[lon_min, lat_min, lon_max, lat_max]``.
        geometry_wkb: Spatial geometry encoded as WKB.
        time_range: Optional tuple of POSIX seconds ``(start, end)``.
        sensor: Sensor/platform description.
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
    if dataset is not None and resolution_info["resolution_deg_lat"] is None:
        logging.debug(
            f"Resolution is None for {data_source} {region_name}: "
            f"dataset={'None' if dataset is None else 'exists'}, "
            f"bbox={bbox}, "
            f"has_lat={'lat' in dataset.coords if dataset is not None else 'N/A'}, "
            f"has_lon={'lon' in dataset.coords if dataset is not None else 'N/A'}"
        )

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
        "resolution_deg_lat": resolution_info["resolution_deg_lat"],
        "resolution_deg_lon": resolution_info["resolution_deg_lon"],
        "resolution_km_lat": resolution_info["resolution_km_lat"],
        "resolution_km_lon": resolution_info["resolution_km_lon"],
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

            var_da = var_da.squeeze()

            data_vars[var_key] = var_da
            encoding[var_key] = get_variable_encoding(var_key)

        if not data_vars:
            continue

        ds_out = xr.Dataset(
            data_vars,
            coords={"lat": regional_ds["lat"], "lon": regional_ds["lon"]},
            attrs=regional_ds.attrs,
        )

        ds_out = clear_encoding(ds_out)
        check_encoding_safety(ds_out, encoding)

        outfile = out_dir / f"glorys_{region_name}_{date_str}.nc"
        ds_out.to_netcdf(outfile, encoding=encoding, engine="h5netcdf")

        records.append(
            make_record(
                outfile,
                output_dir,
                base_timestamp,
                "glorys",
                "all_phys",
                region_name,
                region_info["bbox"],
                region_info["geometry"],
                region_info["time_range"],
                "GLORYS",
                dataset=ds_out,
            )
        )

    logging.info(f"✓ GLORYS: {len(records)} regional files")
    return len(records), records


def process_and_split(ds, date_str, output_dir, modality, keep_vars=None, sensor=None):
    """Generic processing for gridded data."""
    if ds is None:
        return 0, []
    if "time" in ds.dims:
        ds = ds.isel(time=0)
    ds = normalize_coords(ds)
    if keep_vars:
        available = [v for v in keep_vars if v in ds.data_vars]
        if available:
            ds = ds[available]

    # Convert Kelvin to Celsius for Temperature variables (SST)
    for v in ds.data_vars:
        # Check if variable name suggests temperature
        is_temp_var = any(
            x in v.lower()
            for x in [
                "temperature",
                "sst",
                "thetao",
                "sea_surface_temperature",
                "analysed_sst",
            ]
        )
        # Check if units suggest Kelvin (or just check range)
        # Avoid non-temperature vars like 'sst_dtime' or 'sources_of_sst' by name too
        is_metadata = any(
            x in v.lower()
            for x in ["dtime", "source", "mask", "flag", "count", "number"]
        )

        if is_temp_var and not is_metadata:
            # Check mean value to distinguish K vs C
            # Using dropna=True to handle NaNs
            valid_data = ds[v].values.flatten()
            valid_data = valid_data[~np.isnan(valid_data)]

            if len(valid_data) > 0 and valid_data.mean() > 200:
                logging.info(
                    f" Converting {v} from Kelvin to Celsius (mean={valid_data.mean():.1f})"
                )
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

        records.append(
            make_record(
                outfile,
                output_dir,
                base_timestamp,
                modality,
                modality.split("_")[-1],
                region_name,
                region_info["bbox"],
                region_info["geometry"],
                region_info["time_range"],
                sensor or modality.upper(),
                dataset=regional_ds,
            )
        )

    logging.info(f"✓ {modality}: {len(records)} regional files")
    return len(records), records




def process_swot_daily_gridded(
    data_dir, date_str, output_dir, resolution_km=2.0, overlap_method="mean"
):
    """Process L3 SWOT files, gridding them per region with track masking support.

    Creates gridded data with additional track identification layers:
    - primary_track: Index of the first track to contribute to each cell (-1 = no data)
    - is_overlap: Boolean mask indicating cells where multiple tracks contributed
    - track_ids: List of track identifiers (filenames)
    - track_times: Timestamps for each track

    This allows masking out specific tracks for validation purposes.

    Process L3 SWOT files with conservative regridding.

    Key improvements over v1:
    - No Gaussian smoothing (preserves spectral content)
    - No artificial gap-filling (NaN where no data)
    - Proper handling of track overlaps
    - Resolution-matched gridding to preserve 2km features

    Parameters
    ----------
    overlap_method : str
        How to handle overlapping tracks: "mean", "first", "last"
    """
    files = load_l3_swot_data(data_dir, date_str, return_paths_only=True)
    if not files:
        return 0, []

    out_dir = Path(output_dir) / "l3_swot"
    out_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Processing {len(files)} L3_SWOT files for {date_str}")

    records = []
    base_timestamp = pd.Timestamp(datetime.strptime(date_str, "%Y%m%d"))

    for region_name, bounds in SPATIAL_REGIONS.items():
        target_lons, target_lats, lon_edges, lat_edges = create_regional_grid(
            bounds, resolution_km
        )
        grid_shape = (len(target_lats), len(target_lons))

        # Accumulate data using Welford's online algorithm for numerical stability
        # This computes running mean without storing all values
        accumulated_sum = {}
        accumulated_count = {}

        # Track metadata
        primary_track = np.full(grid_shape, -1, dtype=np.int16)
        track_count_per_cell = np.zeros(grid_shape, dtype=np.int16)
        track_ids = []
        track_times = []
        time_min, time_max = None, None

        regional_track_idx = 0

        for fpath in files:
            try:
                ds = xr.open_dataset(fpath)
            except Exception:
                continue

            # Get coordinates
            swath_lons = (
                ds["longitude"].values if "longitude" in ds.coords else ds["lon"].values
            )
            swath_lats = (
                ds["latitude"].values if "latitude" in ds.coords else ds["lat"].values
            )
            swath_lons = lon_to_180(swath_lons)

            if not swath_intersects_region(swath_lons, swath_lats, bounds):
                ds.close()
                continue

            # Track time
            track_time, track_time_min, track_time_max = _extract_track_time_stats(ds)
            if track_time_min is not None:
                if time_min is None or track_time_min < time_min:
                    time_min = track_time_min
                if time_max is None or track_time_max > time_max:
                    time_max = track_time_max

            track_ids.append(os.path.basename(fpath))
            track_times.append(track_time.isoformat() if track_time else "")

            track_contributed = False

            for var_name in L3_SWOT_VARS:
                if var_name not in ds:
                    continue

                # Conservative binning - no smoothing
                grid_data, grid_counts = bin_swath_to_grid_conservative(
                    swath_lons,
                    swath_lats,
                    ds[var_name].values,
                    target_lons,
                    target_lats,
                    method="mean",
                    min_samples=1,
                )

                valid_mask = grid_counts > 0

                if not valid_mask.any():
                    continue

                track_contributed = True

                # Initialize accumulators if needed
                if var_name not in accumulated_sum:
                    accumulated_sum[var_name] = np.zeros(grid_shape, dtype=np.float64)
                    accumulated_count[var_name] = np.zeros(grid_shape, dtype=np.int32)

                # Accumulate (weighted by count for proper averaging)
                accumulated_sum[var_name][valid_mask] += (
                    grid_data[valid_mask] * grid_counts[valid_mask]
                )
                accumulated_count[var_name][valid_mask] += grid_counts[valid_mask]

            if track_contributed:
                # Update track info for cells newly covered
                any_var_mask = np.zeros(grid_shape, dtype=bool)
                for var_name in accumulated_count:
                    any_var_mask |= accumulated_count[var_name] > 0

                new_cells = any_var_mask & (primary_track == -1)
                primary_track[new_cells] = regional_track_idx
                track_count_per_cell[any_var_mask] = np.minimum(
                    track_count_per_cell[any_var_mask] + 1, np.iinfo(np.int16).max
                )
                regional_track_idx += 1

            ds.close()

        if not accumulated_sum:
            continue

        # Compute final averages
        data_vars = {}
        total_obs = np.zeros(grid_shape, dtype=np.int32)

        for var_name, data_sum in accumulated_sum.items():
            count = accumulated_count[var_name]
            total_obs = np.maximum(total_obs, count)

            with np.errstate(invalid="ignore", divide="ignore"):
                averaged = data_sum / np.maximum(count, 1)
            averaged[count == 0] = np.nan

            data_vars[var_name] = (["lat", "lon"], averaged.astype(np.float32))

        # Observation count (total native observations, not just tracks)
        data_vars["n_obs"] = (["lat", "lon"], total_obs)

        # Track masking layers
        data_vars["primary_track"] = (["lat", "lon"], primary_track)
        data_vars["is_overlap"] = (
            ["lat", "lon"],
            (track_count_per_cell > 1).astype(np.int8),
        )

        # Create output dataset
        ds_out = xr.Dataset(
            data_vars,
            coords={
                "lat": target_lats,
                "lon": target_lons,
                "track": np.arange(len(track_ids)),
            },
            attrs={
                "source": "L3 SWOT conservative regrid",
                "resolution_km": resolution_km,
                "n_tracks": len(track_ids),
                "processing": "bin_mean_no_smoothing",
                "date": date_str,
            },
        )

        if track_ids:
            ds_out["track_ids"] = (["track"], track_ids)
            ds_out["track_times"] = (["track"], track_times)

        # Save
        outfile = out_dir / f"l3_swot_{region_name}_{date_str}.nc"

        encoding = {}
        for var_name in accumulated_sum.keys():
            encoding[var_name] = {
                "dtype": "float32",
                "zlib": True,
                "complevel": 4,
                "_FillValue": np.float32(np.nan),
            }
        encoding["n_obs"] = {"dtype": "int32", "zlib": True, "_FillValue": -1}
        encoding["primary_track"] = {"dtype": "int16", "zlib": True, "_FillValue": -1}
        encoding["is_overlap"] = {"dtype": "int8", "zlib": True}

        ds_out.to_netcdf(outfile, encoding=encoding, engine="h5netcdf")

        # Record metadata
        bbox = [
            float(target_lons.min()),
            float(target_lats.min()),
            float(target_lons.max()),
            float(target_lats.max()),
        ]
        time_range = (
            (time_min.timestamp(), time_max.timestamp()) if time_min else (None, None)
        )

        records.append(
            make_record(
                outfile,
                output_dir,
                base_timestamp,
                "l3_swot",
                "ssh",
                region_name,
                bbox,
                box(*bbox).wkb,
                time_range,
                "SWOT",
                dataset=ds_out,
                gridded=True,
                n_tracks=len(track_ids),
            )
        )

    logging.info(f"✓ L3_SWOT: {len(records)} regional files (conservative regrid)")
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
        target_lons, target_lats, lon_edges, lat_edges = create_regional_grid(
            bounds, resolution_km
        )
        grid_shape = (len(target_lats), len(target_lons))

        # Data accumulation using observation-weighted approach
        # We accumulate (sum of values * counts) and (total counts) separately
        # to properly handle multiple observations per cell
        accumulated_sum = np.zeros(grid_shape, dtype=np.float64)
        accumulated_count = np.zeros(grid_shape, dtype=np.int32)
        time_min, time_max = None, None

        # Track masking: primary_track stores first track index, -1 = no data
        primary_track = np.full(grid_shape, -1, dtype=np.int16)
        # Track count per cell to determine overlaps
        track_count_per_cell = np.zeros(grid_shape, dtype=np.int16)

        # Track metadata
        track_ids = []
        track_times = []
        track_platforms = []
        regional_track_idx = 0

        for fpath in files:
            try:
                ds = xr.open_dataset(fpath)
            except Exception:
                continue

            # Get coordinates - L3 SSH uses longitude/latitude
            if "longitude" in ds.coords:
                track_lons = lon_to_180(ds["longitude"].values)
                track_lats = ds["latitude"].values
            elif "lon" in ds.coords:
                track_lons = lon_to_180(ds["lon"].values)
                track_lats = ds["lat"].values
            else:
                ds.close()
                continue

            if not swath_intersects_region(track_lons, track_lats, bounds):
                ds.close()
                continue

            # Get track time
            track_time, track_time_min, track_time_max = _extract_track_time_stats(ds)
            if track_time_min is not None:
                if time_min is None or track_time_min < time_min:
                    time_min = track_time_min
                if time_max is None or track_time_max > time_max:
                    time_max = track_time_max

            # Get SLA data
            if "sla_filtered" in ds:
                sla_data = ds["sla_filtered"].values
            elif "sla" in ds:
                sla_data = ds["sla"].values
            else:
                ds.close()
                continue

            # Conservative binning - no smoothing
            grid_data, grid_counts = bin_swath_to_grid_conservative(
                track_lons,
                track_lats,
                sla_data,
                target_lons,
                target_lats,
                method="mean",
                min_samples=1,
            )

            valid_mask = grid_counts > 0

            if not valid_mask.any():
                ds.close()
                continue

            # Track contributed to this region
            track_ids.append(os.path.basename(fpath))
            track_times.append(track_time.isoformat() if track_time else "")
            track_platforms.append(ds.attrs.get("platform", "unknown"))

            # Accumulate data (weighted by observation count for proper averaging)
            accumulated_sum[valid_mask] += (
                grid_data[valid_mask] * grid_counts[valid_mask]
            )
            accumulated_count[valid_mask] += grid_counts[valid_mask]

            # Update primary track for cells that don't have one yet
            new_cells = valid_mask & (primary_track == -1)
            primary_track[new_cells] = regional_track_idx

            # Increment track count for overlap detection
            track_count_per_cell[valid_mask] += 1

            regional_track_idx += 1
            ds.close()

        if accumulated_count.sum() == 0:
            continue

        # Compute observation-weighted average
        with np.errstate(invalid="ignore", divide="ignore"):
            sla_averaged = accumulated_sum / np.maximum(accumulated_count, 1)
        sla_averaged[accumulated_count == 0] = np.nan

        # Build data variables
        data_vars = {
            "sla_filtered": (["lat", "lon"], sla_averaged.astype(np.float32)),
            "n_obs": (["lat", "lon"], accumulated_count.astype(np.int32)),
            "primary_track": (["lat", "lon"], primary_track),
            "is_overlap": (["lat", "lon"], (track_count_per_cell > 1).astype(np.int8)),
        }

        # Create dataset with track metadata
        ds_out = xr.Dataset(
            data_vars,
            coords={
                "lat": target_lats,
                "lon": target_lons,
                "track": np.arange(len(track_ids)),
            },
            attrs={
                "source": "L3 SSH conservative regrid",
                "resolution_km": resolution_km,
                "n_tracks": len(track_ids),
                "processing": "bin_mean_no_smoothing",
                "date": date_str,
            },
        )

        # Add track metadata as data variables
        if track_ids:
            ds_out["track_ids"] = (["track"], track_ids)
            ds_out["track_times"] = (["track"], track_times)
            ds_out["track_platforms"] = (["track"], track_platforms)

        outfile = out_dir / f"l3_ssh_{region_name}_{date_str}.nc"

        # Set up encoding
        encoding = {
            "sla_filtered": {
                "dtype": "float32",
                "zlib": True,
                "complevel": 4,
                "_FillValue": np.float32(np.nan),
            },
            "n_obs": {"dtype": "int32", "zlib": True, "_FillValue": -1},
            "primary_track": {"dtype": "int16", "zlib": True, "_FillValue": -1},
            "is_overlap": {"dtype": "int8", "zlib": True},
        }

        ds_out.to_netcdf(outfile, encoding=encoding, engine="h5netcdf")

        bbox = [
            float(target_lons.min()),
            float(target_lats.min()),
            float(target_lons.max()),
            float(target_lats.max()),
        ]
        time_range = (
            (time_min.timestamp(), time_max.timestamp()) if time_min else (None, None)
        )

        sensor_str = (
            ", ".join(sorted(set(track_platforms))) if track_platforms else "L3_SSH"
        )

        records.append(
            make_record(
                outfile,
                output_dir,
                base_timestamp,
                "l3_ssh",
                "ssh",
                region_name,
                bbox,
                box(*bbox).wkb,
                time_range,
                sensor_str,
                dataset=ds_out,
                gridded=True,
                n_tracks=len(track_ids),
            )
        )

    logging.info(f"✓ L3_SSH: {len(records)} regional files (conservative regrid)")
    return len(records), records




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
            mask = np.array(
                [point_in_region(lon, lat, bounds) for lon, lat in zip(lons, lats)]
            )

            if not mask.any():
                continue

            regional_ds = ds.isel(N_POINTS=mask)
            outfile = out_dir / f"argo_{region_name}_{date_str}.nc"
            regional_ds = clear_encoding(regional_ds)
            regional_ds.to_netcdf(outfile, engine="h5netcdf")

            reg_lons = regional_ds["lon"].values
            reg_lats = regional_ds["lat"].values
            bbox = [
                float(reg_lons.min()),
                float(reg_lats.min()),
                float(reg_lons.max()),
                float(reg_lats.max()),
            ]

            time_range = (
                posix_range_from_time(regional_ds["TIME"].values)
                if "TIME" in regional_ds
                else (None, None)
            )

            # Note: Argo is point data, not gridded, so resolution doesn't apply
            records.append(
                make_record(
                    outfile,
                    output_dir,
                    base_timestamp,
                    "argo",
                    "profiles",
                    region_name,
                    bbox,
                    box(*bbox).wkb,
                    time_range,
                    "ARGO",
                    dataset=None,  # No dataset for resolution calculation (point data)
                    n_profiles=int(mask.sum()),
                )
            )

        ds.close()
        logging.info(f"✓ Argo: {len(records)} regional files")
        return len(records), records

    except Exception as e:
        logging.warning(f"Error processing ARGO for {date_str}: {e}")
        return 0, []


def _process_smos_pass(file_paths, date_str, output_dir, pass_type):
    """Process one SMOS pass type (ascending or descending)."""
    pass_config = {
        "asc": {
            "out_subdir": "l3_sss_smos_asc",
            "data_source": "l3_sss_smos_asc",
            "sensor": "SMOS_ASC",
            "satellite_pass": "ascending",
        },
        "desc": {
            "out_subdir": "l3_sss_smos_desc",
            "data_source": "l3_sss_smos_desc",
            "sensor": "SMOS_DESC",
            "satellite_pass": "descending",
        },
    }
    cfg = pass_config[pass_type]

    records = []
    for fpath in file_paths or []:
        try:
            ds = xr.open_dataset(fpath)
            if "time" in ds.dims:
                ds = ds.isel(time=0)
            ds = normalize_coords(ds)

            regional_data = split_gridded_into_regions(ds, SPATIAL_REGIONS)
            out_dir = Path(output_dir) / cfg["out_subdir"]
            out_dir.mkdir(parents=True, exist_ok=True)

            base_timestamp = pd.Timestamp(datetime.strptime(date_str, "%Y%m%d"))

            for region_name, region_info in regional_data.items():
                if not region_info["intersects"]:
                    continue

                regional_ds = clear_encoding(region_info["dataset"])
                outfile = out_dir / f"{cfg['data_source']}_{region_name}_{date_str}.nc"

                encoding = {v: get_variable_encoding(v) for v in regional_ds.data_vars}
                check_encoding_safety(regional_ds, encoding)

                regional_ds.to_netcdf(
                    outfile,
                    encoding=encoding,
                    engine="h5netcdf",
                )

                records.append(
                    make_record(
                        outfile,
                        output_dir,
                        base_timestamp,
                        cfg["data_source"],
                        "sss",
                        region_name,
                        region_info["bbox"],
                        region_info["geometry"],
                        region_info["time_range"],
                        cfg["sensor"],
                        dataset=regional_ds,
                        satellite_pass=cfg["satellite_pass"],
                    )
                )

            ds.close()
        except Exception as e:
            logging.warning(f"  Error processing SMOS {pass_type} file {fpath}: {e}")

    return records


def process_l3_sss_smos_data(data_dir, date_str, output_dir):
    """Process L3 SMOS SSS data - handle ascending and descending passes separately.

    SMOS has both ascending and descending satellite passes, which have different
    observation characteristics (time of day, viewing angle, etc.). These are
    processed as separate data products to preserve this distinction.
    """
    smos_files = load_l3_sss_smos_data(data_dir, date_str)
    if not smos_files or (smos_files["asc"] is None and smos_files["desc"] is None):
        return 0, []

    all_records = []
    all_records.extend(_process_smos_pass(smos_files["asc"], date_str, output_dir, "asc"))
    all_records.extend(
        _process_smos_pass(smos_files["desc"], date_str, output_dir, "desc")
    )

    logging.info(f"✓ L3_SSS_SMOS: {len(all_records)} regional files (asc + desc)")
    return len(all_records), all_records


# =============================================================================
# Track Masking Utilities
# =============================================================================


def mask_tracks_from_grid(
    data: np.ndarray,
    primary_track: np.ndarray,
    is_overlap: np.ndarray,
    exclude_track_indices: list,
    preserve_overlaps: bool = False,
) -> np.ndarray:
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


def apply_track_mask_to_netcdf(
    nc_path: str,
    exclude_track_indices: list,
    preserve_overlaps: bool = False,
    output_path: str = None,
) -> xr.Dataset:
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
    skip_vars = {"primary_track", "is_overlap", "n_obs", "track_ids", "track_times"}
    data_vars = [v for v in ds.data_vars if v not in skip_vars]

    # Apply masking to each data variable
    for var in data_vars:
        if ds[var].dims == ("lat", "lon"):
            masked = mask_tracks_from_grid(
                ds[var].values,
                primary_track,
                is_overlap,
                exclude_track_indices,
                preserve_overlaps,
            )
            ds[var] = (["lat", "lon"], masked)

    if output_path:
        ds.to_netcdf(output_path, engine="h5netcdf")
        logging.info(f"Saved masked data to {output_path}")

    return ds


def split_tracks_for_validation(
    nc_path: str, val_fraction: float = 0.2, random_seed: int = 42
) -> tuple[list, list]:
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


