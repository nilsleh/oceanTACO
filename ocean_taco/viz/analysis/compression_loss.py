#!/usr/bin/env python3
"""Evaluate compression loss in OceanTACO dataset."""

import argparse
import csv
import glob
import importlib.util
import logging
import os
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from ocean_taco.generate_dataset import format_constants as pipeline_constants
from ocean_taco.generate_dataset import format_coords as pipeline_coords
from ocean_taco.generate_dataset import format_gridding as pipeline_gridding
from ocean_taco.generate_dataset import format_loaders as pipeline_loaders

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

ALL_MODALITIES = [
    "glorys",
    "l4_ssh",
    "l4_sst",
    "l4_sss",
    "l4_wind",
    "l3_sst",
    "l3_ssh",
    "l3_swot",
    "l3_sss_smos_asc",
    "l3_sss_smos_desc",
]

MODALITY_ALIASES = {
    "l3_sss": ["l3_sss_smos_asc", "l3_sss_smos_desc"],
    "l3_sss_smos": ["l3_sss_smos_asc", "l3_sss_smos_desc"],
    "all": ALL_MODALITIES,
}

SKIP_COMPARE_VARS = {
    "n_obs",
    "primary_track",
    "is_overlap",
    "track_ids",
    "track_times",
    "sst_dtime",
    "dtime",
}

REPRESENTATIVE_VARIABLES = {
    "glorys": "zos",
    "l4_ssh": "sla",
    "l4_sst": "analysed_sst",
    "l4_sss": "so",
    "l4_wind": "wind_speed",
    "l3_sst": "sea_surface_temperature",
    "l3_ssh": "sla_filtered",
    "l3_swot": "ssha_filtered",
    "l3_sss_smos_asc": "Sea_Surface_Salinity",
    "l3_sss_smos_desc": "Sea_Surface_Salinity",
}

# (modality, variable) -> (display_modality, display_variable, unit)
# Variable names must match what appears in the CSV `variable` column (i.e. actual NetCDF var names).
_VARIABLE_DISPLAY: dict[tuple[str, str], tuple[str, str, str]] = {
    ("glorys", "zos"):    ("GLORYS", "SSH",  "m"),
    ("glorys", "thetao"): ("GLORYS", "SST",  r"$^\circ$C"),
    ("glorys", "so"):     ("GLORYS", "SSS",  "PSU"),
    ("glorys", "uo"):     ("GLORYS", "uo",   r"m\,s$^{-1}$"),
    ("glorys", "vo"):     ("GLORYS", "vo",   r"m\,s$^{-1}$"),
    ("l4_ssh",  "sla"):                              ("L4 SSH",             "SLA",  "m"),
    ("l4_sst",  "analysed_sst"):                     ("L4 SST",             "SST",  r"$^\circ$C"),
    ("l4_sss",  "sos"):                              ("L4 SSS",             "SSS",  "PSU"),
    ("l4_sss",  "so"):                               ("L4 SSS",             "SSS",  "PSU"),
    ("l4_wind", "eastward_wind"):                    ("L4 Wind",            "Wind", r"m\,s$^{-1}$"),
    ("l4_wind", "wind_speed"):                       ("L4 Wind",            "Wind", r"m\,s$^{-1}$"),
    ("l3_sst",  "adjusted_sea_surface_temperature"): ("L3 SST",             "SST",  r"$^\circ$C"),
    ("l3_sst",  "sea_surface_temperature"):          ("L3 SST",             "SST",  r"$^\circ$C"),
    ("l3_ssh",  "sla_filtered"):                     ("L3 SSH",             "SLA",  "m"),
    ("l3_swot", "ssha_filtered"):                    ("L3 SWOT",            "SSHA", "m"),
    ("l3_sss_smos_asc",  "Sea_Surface_Salinity"):    ("L3 SSS SMOS Asc",    "SSS",  "PSU"),
    ("l3_sss_smos_desc", "Sea_Surface_Salinity"):    ("L3 SSS SMOS Desc",   "SSS",  "PSU"),
}

# Canonical display order for the LaTeX table (GLORYS first, then L4, then L3).
# Uses the first matching variable name — duplicates for l4_sss/l4_wind/l3_sst are handled
# by checking whichever key is present in the aggregated groups dict.
_TABLE_ROW_ORDER: list[tuple[str, str]] = [
    ("glorys", "zos"),   ("glorys", "thetao"), ("glorys", "so"),
    ("glorys", "uo"),    ("glorys", "vo"),
    ("l4_ssh",  "sla"),
    ("l4_sst",  "analysed_sst"),
    ("l4_sss",  "sos"),  ("l4_sss", "so"),
    ("l4_wind", "eastward_wind"), ("l4_wind", "wind_speed"),
    ("l3_sst",  "adjusted_sea_surface_temperature"), ("l3_sst", "sea_surface_temperature"),
    ("l3_ssh",  "sla_filtered"),
    ("l3_swot", "ssha_filtered"),
    ("l3_sss_smos_asc",  "Sea_Surface_Salinity"),
    ("l3_sss_smos_desc", "Sea_Surface_Salinity"),
]


def find_formatted_file(base_dir, modality, region, date_str):
    """Locate the formatted file based on modality specific naming/folder structure."""
    # standard structure: modality/modality_REGION_DATE.nc
    p = Path(base_dir) / modality / f"{modality}_{region}_{date_str}.nc"
    if p.exists():
        return p
    # Check if it might be in a region subdir (some versions might do this)
    p = Path(base_dir) / modality / region / f"{modality}_{region}_{date_str}.nc"
    if p.exists():
        return p

    return None


def reconstruct_l3_data(modality, region_name, date_str, data_dir, ref_ds):
    """Reconstruct the gridded float32 data from L3 source files for comparison.
    This mimics the binning process in the pipeline but keeps precision.
    """
    logging.info(f"Reconstructing {modality} float32 grid for validation...")

    # 1. Setup Grid Edges from ref_ds
    lats = ref_ds.lat.values
    lons = ref_ds.lon.values

    grid_shape = (len(lats), len(lons))
    bounds = pipeline_constants.SPATIAL_REGIONS[region_name]

    # 2. Identify files and vars
    files = []
    vars_to_process = []

    if modality == "l3_swot":
        files = pipeline_loaders.load_l3_swot_data(data_dir, date_str, return_paths_only=True)
        vars_to_process = ["ssha_filtered", "mdt"]

    elif modality == "l3_ssh":
        files = pipeline_loaders.load_l3_ssh_data(data_dir, date_str, return_paths_only=True)
        vars_to_process = ["sla_filtered"]

    logging.debug(f"[{modality}/{date_str}] Found {len(files)} source file(s) for reconstruction.")
    if not files:
        logging.warning(f"[{modality}/{date_str}] No source files found for reconstruction.")
        return None

    # 3. Init accumulators
    accumulated = {}
    for v in vars_to_process:
        accumulated[v] = {
            "sum": np.zeros(grid_shape, dtype=np.float64),
            "count": np.zeros(grid_shape, dtype=int),
        }

    # 4. Process files
    n_intersecting = 0
    for fpath in files:
        with xr.open_dataset(fpath) as ds:
            # Coordinate handling
            track_lons = None
            track_lats = None

            # Try standard names
            if "longitude" in ds.coords:
                track_lons = pipeline_coords.lon_to_180(ds["longitude"].values)
            elif "lon" in ds.coords:
                track_lons = pipeline_coords.lon_to_180(ds["lon"].values)

            if "latitude" in ds.coords:
                track_lats = ds["latitude"].values
            elif "lat" in ds.coords:
                track_lats = ds["lat"].values

            if track_lons is None or track_lats is None:
                logging.warning(f"[{modality}/{date_str}] {fpath}: no lon/lat coords found, skipping.")
                continue

            # Intersection check
            if not pipeline_gridding.swath_intersects_region(track_lons, track_lats, bounds):
                continue
            n_intersecting += 1
            logging.debug(
                f"[{modality}/{date_str}] {Path(fpath).name}: intersects region. "
                f"Data vars: {list(ds.data_vars)}"
            )

            # Variable processing
            for var_name in vars_to_process:
                # Map requested var to source var if needed
                source_var = var_name
                if modality == "l3_ssh" and var_name == "sla_filtered":
                    if "sla_filtered" not in ds:
                        if "sla" in ds:
                            source_var = "sla"
                            logging.warning(
                                f"[{modality}/{date_str}] {Path(fpath).name}: 'sla_filtered' not found, "
                                f"falling back to 'sla'."
                            )
                        elif "ssha" in ds:
                            source_var = "ssha"
                            logging.warning(
                                f"[{modality}/{date_str}] {Path(fpath).name}: 'sla_filtered' not found, "
                                f"falling back to 'ssha'."
                            )
                        else:
                            logging.warning(
                                f"[{modality}/{date_str}] {Path(fpath).name}: 'sla_filtered' not found "
                                f"and no fallback variable available. Available vars: {list(ds.data_vars)}. Skipping."
                            )
                            continue

                if source_var not in ds:
                    logging.warning(
                        f"[{modality}/{date_str}] {Path(fpath).name}: var '{source_var}' not in dataset "
                        f"(available: {list(ds.data_vars)}), skipping."
                    )
                    continue

                val_data = ds[source_var].values
                if val_data.shape != track_lons.shape:
                    logging.warning(
                        f"[{modality}/{date_str}] {Path(fpath).name}: var '{source_var}' shape {val_data.shape} "
                        f"!= coords shape {track_lons.shape}, skipping."
                    )
                    continue
                n_finite_raw = int(np.isfinite(val_data).sum())

                # Binning
                binned = pipeline_gridding.bin_swath_to_grid(
                    track_lons, track_lats, val_data, lons, lats
                )

                valid = ~np.isnan(binned)
                n_binned = int(valid.sum())
                logging.debug(
                    f"[{modality}/{date_str}] {Path(fpath).name}: var='{var_name}' "
                    f"raw_finite={n_finite_raw} binned_cells={n_binned}"
                )
                accumulated[var_name]["sum"][valid] += binned[valid]
                accumulated[var_name]["count"][valid] += 1


    logging.debug(f"[{modality}/{date_str}] {n_intersecting}/{len(files)} file(s) intersected region {region_name}.")

    # 5. Averaging
    data_vars = {}
    all_nan = True
    for v in vars_to_process:
        s = accumulated[v]["sum"]
        c = accumulated[v]["count"]
        mask = c > 0
        if mask.any():
            all_nan = False
        mean_val = np.full(grid_shape, np.nan, dtype=np.float32)
        mean_val[mask] = s[mask] / c[mask]
        data_vars[v] = (("lat", "lon"), mean_val)

    if all_nan:
        logging.warning(
            f"[{modality}/{date_str}] Reconstruction produced all-NaN grid for region {region_name} "
            f"({n_intersecting}/{len(files)} intersecting files). Returning None."
        )
        return None

    return xr.Dataset(data_vars, coords={"lat": lats, "lon": lons})


def parse_date_range(date_str=None, min_date=None, max_date=None):
    """Return list of YYYYMMDD dates to process."""
    if min_date or max_date:
        if not (min_date and max_date):
            raise ValueError(
                "Both --min-date and --max-date must be provided together."
            )
        start = datetime.strptime(min_date, "%Y%m%d")
        end = datetime.strptime(max_date, "%Y%m%d")
        if end < start:
            raise ValueError("--max-date must be >= --min-date")
        dates = []
        current = start
        while current <= end:
            dates.append(current.strftime("%Y%m%d"))
            current += timedelta(days=1)
        return dates

    if date_str is None:
        raise ValueError(
            "Either --date or both --min-date/--max-date must be provided."
        )
    return [date_str]


def parse_modalities(modality_tokens):
    """Parse modality tokens (supports comma-separated and aliases)."""
    if not modality_tokens:
        return ALL_MODALITIES.copy()

    expanded = []
    for token in modality_tokens:
        for part in token.split(","):
            key = part.strip()
            if not key:
                continue
            expanded.extend(MODALITY_ALIASES.get(key, [key]))

    if not expanded:
        return ALL_MODALITIES.copy()

    # de-duplicate while preserving order
    deduped = list(dict.fromkeys(expanded))

    unknown = [m for m in deduped if m not in ALL_MODALITIES]
    if unknown:
        raise ValueError(
            f"Unknown modality/modalities: {unknown}. Supported: {ALL_MODALITIES}"
        )

    return deduped


def get_var_mapping_for_modality(modality):
    """Return mapping of formatted variable name -> original variable name."""
    if modality == "l3_swot":
        return {"ssha_filtered": "ssha_filtered", "mdt": "mdt"}
    if modality == "l3_ssh":
        return {"sla_filtered": "sla_filtered"}
    if modality == "glorys":
        return {"zos": "zos", "thetao": "thetao", "so": "so", "uo": "uo", "vo": "vo"}
    if modality == "l4_ssh":
        return {"sla": "sla"}
    if modality == "l4_sst":
        return {"analysed_sst": "analysed_sst"}
    if modality == "l4_sss":
        return {
            "so": "so",
            "sss": "sss",
            "sos": "sos",
            "sea_surface_salinity": "sea_surface_salinity",
        }
    if modality == "l4_wind":
        return {
            "eastward_wind": "eastward_wind",
            "northward_wind": "northward_wind",
            "wind_speed": "wind_speed",
            "u10": "u10",
            "v10": "v10",
        }
    if modality == "l3_sst":
        return {
            "sea_surface_temperature": "sea_surface_temperature",
            "adjusted_sea_surface_temperature": "adjusted_sea_surface_temperature",
            "analysed_sst": "analysed_sst",
            "sst": "sst",
        }
    if modality in ["l3_sss_smos_asc", "l3_sss_smos_desc"]:
        return {
            "Sea_Surface_Salinity": "Sea_Surface_Salinity",
            "Sea_Surface_Salinity_Rain_Corrected": "Sea_Surface_Salinity_Rain_Corrected",
            "Sea_Surface_Salinity_Error": "Sea_Surface_Salinity_Error",
        }
    return {}


def load_glorys_data_safe(data_dir, date_str):
    """Load GLORYS without invoking debug breakpoints in upstream script."""
    year, month = date_str[:4], date_str[4:6]
    pattern = os.path.join(
        data_dir,
        "glorys",
        "GLOBAL_MULTIYEAR_PHY_001_030",
        "cmems_mod_glo_phy_my_0.083deg_P1D-m_202311",
        year,
        month,
        f"mercatorglorys12v1_gl12_mean_{date_str}_R*.nc",
    )
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(
            f"[glorys/{date_str}] No GLORYS file found at expected path: {pattern}"
        )
    return xr.open_dataset(files[0])


def load_l3_sss_smos_pass_dataset(data_dir, date_str, modality):
    """Load one SMOS pass dataset (ascending or descending)."""
    smos = pipeline_loaders.load_l3_sss_smos_data(data_dir, date_str)
    if not smos:
        logging.warning(f"[{modality}/{date_str}] load_l3_sss_smos_data returned no data.")
        return None
    key = "asc" if modality.endswith("_asc") else "desc"
    files = smos.get(key)
    if not files:
        logging.warning(f"[{modality}/{date_str}] No files found for SMOS pass key='{key}'.")
        return None
    ds = xr.open_dataset(files[0])
    if "time" in ds.dims and ds.sizes.get("time", 0) == 1:
        ds = ds.isel(time=0)
    return pipeline_coords.normalize_coords(ds)


def infer_variable_pairs(ds_orig, ds_fmt, modality):
    """Infer compatible variable pairs between original and formatted datasets."""
    preferred = list(get_var_mapping_for_modality(modality).keys())
    preferred_set = set(preferred)

    candidates = []
    for var in ds_fmt.data_vars:
        # Restrict to expected science variables for this modality.
        # This avoids comparing metadata/ancillary vars (e.g. masks, quality flags)
        # that are not part of compression-loss reporting.
        if preferred_set and var not in preferred_set:
            continue
        if var in SKIP_COMPARE_VARS:
            continue
        if var not in ds_orig:
            continue
        da_fmt = ds_fmt[var]
        da_orig = ds_orig[var]

        if not hasattr(da_fmt, "dtype") or not hasattr(da_orig, "dtype"):
            continue

        # Exclude timedelta and non-numeric variables from compression-loss stats.
        if np.issubdtype(da_fmt.dtype, np.timedelta64) or np.issubdtype(
            da_orig.dtype, np.timedelta64
        ):
            continue

        if not np.issubdtype(da_fmt.dtype, np.number):
            continue
        if not np.issubdtype(da_orig.dtype, np.number):
            continue
        if not (
            {"lat", "lon"}.issubset(set(da_fmt.dims))
            and {"lat", "lon"}.issubset(set(da_orig.dims))
        ):
            continue
        candidates.append((var, var))

    if not candidates:
        return []

    order = {name: idx for idx, name in enumerate(preferred)}
    candidates.sort(key=lambda pair: (order.get(pair[0], 999), pair[0]))
    return candidates


def _convert_kelvin_like_temperature_vars(ds: xr.Dataset) -> xr.Dataset:
    """Match formatter behavior: convert Kelvin SST-like fields to Celsius."""
    for var_name in ds.data_vars:
        lower = var_name.lower()
        is_temp_var = any(
            token in lower
            for token in [
                "temperature",
                "sst",
                "thetao",
                "sea_surface_temperature",
                "analysed_sst",
            ]
        )
        is_metadata = any(
            token in lower
            for token in ["dtime", "source", "mask", "flag", "count", "number"]
        )
        if not is_temp_var or is_metadata:
            continue

        values = ds[var_name].values
        valid = values[np.isfinite(values)]
        if valid.size and float(np.mean(valid)) > 200:
            ds[var_name] = ds[var_name] - 273.15
            ds[var_name].attrs["units"] = "degree_Celsius"
    return ds


def build_pre_encoding_reference_dataset(modality, ds_orig, region):
    """Recreate formatter pre-encoding regional dataset for fair compression comparison."""
    if ds_orig is None:
        return None

    bounds = pipeline.SPATIAL_REGIONS[region]

    # GLORYS has modality-specific variable/depth extraction logic.
    if modality == "glorys":
        ds = ds_orig
        if "time" in ds.dims:
            ds = ds.isel(time=slice(0, 1))
        ds = pipeline.normalize_coords(ds)

        variables = {
            "zos": ("zos", None),
            "thetao": ("thetao", 0),
            "so": ("so", 0),
            "uo": ("uo", 10),
            "vo": ("vo", 10),
        }

        regional = pipeline.split_gridded_into_regions(ds, {region: bounds}).get(region)
        if regional is None or not regional.get("intersects", False):
            return None

        regional_ds = regional["dataset"]
        data_vars = {}
        for _, (source_var, depth_idx) in variables.items():
            if source_var not in regional_ds.data_vars:
                continue
            da = regional_ds[source_var]
            if depth_idx is not None and "depth" in da.dims:
                da = da.isel(depth=depth_idx)
            if "depth" in da.coords:
                da = da.drop_vars("depth")
            data_vars[source_var] = da

        if not data_vars:
            return None

        coords = {"lat": regional_ds["lat"], "lon": regional_ds["lon"]}
        if "time" in regional_ds.coords:
            coords["time"] = regional_ds["time"]

        return xr.Dataset(data_vars, coords=coords, attrs=regional_ds.attrs)

    ds = ds_orig
    if "time" in ds.dims:
        ds = ds.isel(time=slice(0, 1))
    ds = pipeline.normalize_coords(ds)

    keep_vars = list(get_var_mapping_for_modality(modality).keys())
    if keep_vars:
        available = [v for v in keep_vars if v in ds.data_vars]
        if available:
            ds = ds[available]

    ds = _convert_kelvin_like_temperature_vars(ds)

    regional = pipeline.split_gridded_into_regions(ds, {region: bounds}).get(region)
    if regional is None or not regional.get("intersects", False):
        return None

    return pipeline.clear_encoding(regional["dataset"])


def choose_representative_variable(modality, variables):
    """Choose one representative variable for plotting for a modality."""
    if not variables:
        return None
    preferred = REPRESENTATIVE_VARIABLES.get(modality)
    if preferred in variables:
        return preferred
    preferred_order = list(get_var_mapping_for_modality(modality).keys())
    for cand in preferred_order:
        if cand in variables:
            return cand
    return sorted(variables)[0]


def extract_storage_metadata(ds_fmt_raw, var_name):
    """Return storage metadata for a formatted variable (raw on-disk dtype)."""
    if var_name not in ds_fmt_raw:
        return None
    da_raw = ds_fmt_raw[var_name]
    raw_dtype = np.dtype(da_raw.dtype)
    n_elements = int(da_raw.size)
    stored_itemsize = int(raw_dtype.itemsize)
    stored_bytes = int(n_elements * stored_itemsize)
    float32_bytes = int(n_elements * np.dtype(np.float32).itemsize)

    if float32_bytes > 0:
        memory_gain_ratio = float(float32_bytes / max(stored_bytes, 1))
        memory_gain_pct = float((1.0 - (stored_bytes / float32_bytes)) * 100.0)
    else:
        memory_gain_ratio = np.nan
        memory_gain_pct = np.nan

    return {
        "n_elements": n_elements,
        "stored_dtype": str(raw_dtype),
        "stored_itemsize": stored_itemsize,
        "stored_bytes_uncompressed": stored_bytes,
        "float32_bytes": float32_bytes,
        "memory_gain_ratio": memory_gain_ratio,
        "memory_gain_pct": memory_gain_pct,
        "is_int16_packed": bool(raw_dtype == np.dtype(np.int16)),
    }


def _weighted_mean(values, weights):
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not np.any(valid):
        return np.nan
    return float(np.average(values[valid], weights=weights[valid]))


def reduce_to_lat_lon_2d(da, name="variable"):
    """Reduce a DataArray to 2D (lat, lon) by selecting first index on extra dims."""
    if "lat" not in da.dims or "lon" not in da.dims:
        raise ValueError(
            f"{name} does not contain both lat/lon dims. Found dims={da.dims}"
        )

    extra_dims = [d for d in da.dims if d not in ("lat", "lon")]
    for dim in extra_dims:
        da = da.isel({dim: 0})

    da = da.squeeze(drop=True)

    if set(da.dims) != {"lat", "lon"}:
        raise ValueError(
            f"{name} could not be reduced to 2D lat/lon. Remaining dims={da.dims}"
        )

    if da.dims != ("lat", "lon"):
        da = da.transpose("lat", "lon")

    return da


def plot_aggregate_information_loss(rows, output_dir, prefix):
    """Create one aggregate information-loss figure over full run."""
    grouped = {}
    for row in rows:
        modality = row["modality"]
        grouped.setdefault(modality, []).append(row)

    summary = []
    for modality, items in grouped.items():
        variables = {r["variable"] for r in items}
        rep_var = choose_representative_variable(modality, variables)
        if rep_var is None:
            continue
        selected = [r for r in items if r["variable"] == rep_var]
        if not selected:
            continue
        w = [r["n_valid"] for r in selected]
        summary.append(
            {
                "label": f"{modality}:{rep_var}",
                "MAE": _weighted_mean([r["MAE"] for r in selected], w),
                "RMSE": _weighted_mean([r["RMSE"] for r in selected], w),
                "P99_Error_Report": _weighted_mean(
                    [r.get("P99_Error_Report", r["P99_Error"]) for r in selected], w
                ),
                "Bias": _weighted_mean([r["Bias"] for r in selected], w),
            }
        )

    if not summary:
        logging.warning("No rows available for aggregate information-loss plot.")
        return

    summary = sorted(summary, key=lambda x: x["label"])
    labels = [s["label"] for s in summary]
    x = np.arange(len(labels))

    fig, axes = plt.subplots(2, 2, figsize=(18, 10), sharex=True)
    metrics = [
        ("MAE", axes[0, 0]),
        ("RMSE", axes[0, 1]),
        ("P99_Error_Report", axes[1, 0]),
        ("Bias", axes[1, 1]),
    ]

    for metric, ax in metrics:
        vals = [s[metric] for s in summary]
        ax.bar(x, vals)
        ax.set_title(metric)
        ax.grid(True, alpha=0.3, axis="y")
        if metric == "Bias":
            ax.axhline(0.0, color="red", linestyle="--", alpha=0.5)

    for ax in axes[1, :]:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=12)
    
    # also increase size of y-axis ticks
    for ax in axes.flatten():
        ax.tick_params(axis="y", labelsize=12)

    fig.suptitle("Aggregate Compression Information Loss (weighted by n_valid)")
    fig.tight_layout()

    out_file = Path(output_dir) / f"{prefix}_aggregate_information_loss.png"
    fig.savefig(out_file, dpi=150)
    plt.close(fig)
    logging.info(f"Saved aggregate information-loss plot to {out_file}")


def plot_aggregate_memory_improvement(rows, output_dir, prefix):
    """Create one aggregate memory-improvement figure over full run."""
    grouped = {}
    for row in rows:
        modality = row["modality"]
        grouped.setdefault(modality, []).append(row)

    summary = []
    for modality, items in grouped.items():
        variables = {r["variable"] for r in items}
        rep_var = choose_representative_variable(modality, variables)
        if rep_var is None:
            continue

        selected = [
            r
            for r in items
            if r["variable"] == rep_var and r.get("is_int16_packed", False)
        ]
        if not selected:
            continue

        total_float32 = sum(r["float32_bytes"] for r in selected)
        total_stored = sum(r["stored_bytes_uncompressed"] for r in selected)
        if total_float32 <= 0:
            continue

        gain_pct = float((1.0 - (total_stored / total_float32)) * 100.0)
        gain_ratio = float(total_float32 / max(total_stored, 1))
        summary.append(
            {
                "label": f"{modality}:{rep_var}",
                "gain_pct": gain_pct,
                "gain_ratio": gain_ratio,
            }
        )

    if not summary:
        logging.warning(
            "No int16-packed rows available for aggregate memory-improvement plot."
        )
        return

    summary = sorted(summary, key=lambda x: x["label"])
    labels = [s["label"] for s in summary]
    gain_pct = [s["gain_pct"] for s in summary]
    gain_ratio = [s["gain_ratio"] for s in summary]
    x = np.arange(len(labels))

    fig, ax1 = plt.subplots(figsize=(16, 7))
    bars = ax1.bar(
        x, gain_pct, color="tab:green", alpha=0.85, label="Memory reduction (%)"
    )
    ax1.set_ylabel("Memory reduction vs float32 (%)")
    ax1.set_ylim(bottom=0)
    ax1.grid(True, alpha=0.3, axis="y")

    for bar, ratio in zip(bars, gain_ratio):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{ratio:.2f}x",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=35, ha="right", fontsize=12)
    ax1.set_title(
        "Aggregate Memory Improvement per Modality Representative Variable (int16-packed only)"
    )
    fig.tight_layout()

    out_file = Path(output_dir) / f"{prefix}_aggregate_memory_improvement.png"
    fig.savefig(out_file, dpi=150)
    plt.close(fig)
    logging.info(f"Saved aggregate memory-improvement plot to {out_file}")


def plot_timeseries_summary(rows, output_dir, prefix):
    """Plot MAE and RMSE over dates, one line per modality with variance band."""
    if not rows:
        return

    # Group by (modality, date, variable) -> collect values
    grouped = defaultdict(lambda: defaultdict(list))  # [modality][date] -> list of (MAE, RMSE)
    modality_vars = defaultdict(set)

    for row in rows:
        modality = row["modality"]
        date = row["date"]
        modality_vars[modality].add(row["variable"])
        grouped[modality][date].append({"MAE": row["MAE"], "RMSE": row["RMSE"]})

    modalities = sorted(grouped.keys())
    if not modalities:
        return

    # Collect all unique dates
    all_dates_str = sorted({row["date"] for row in rows})
    all_dates = [datetime.strptime(d, "%Y%m%d") for d in all_dates_str]

    colors = plt.cm.tab10(np.linspace(0, 1, max(len(modalities), 1)))

    fig, (ax_mae, ax_rmse) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    for color, modality in zip(colors, modalities):
        date_data = grouped[modality]
        dates = []
        mae_means, mae_stds = [], []
        rmse_means, rmse_stds = [], []

        for date_str, date_dt in zip(all_dates_str, all_dates):
            entries = date_data.get(date_str)
            if not entries:
                continue
            dates.append(date_dt)
            maes = [e["MAE"] for e in entries]
            rmses = [e["RMSE"] for e in entries]
            mae_means.append(np.mean(maes))
            mae_stds.append(np.std(maes) if len(maes) > 1 else 0.0)
            rmse_means.append(np.mean(rmses))
            rmse_stds.append(np.std(rmses) if len(rmses) > 1 else 0.0)

        if not dates:
            continue

        dates = np.array(dates)
        mae_means = np.array(mae_means)
        mae_stds = np.array(mae_stds)
        rmse_means = np.array(rmse_means)
        rmse_stds = np.array(rmse_stds)

        ax_mae.plot(dates, mae_means, marker="o", markersize=3, color=color, label=modality)
        if mae_stds.any():
            ax_mae.fill_between(dates, mae_means - mae_stds, mae_means + mae_stds, alpha=0.2, color=color)

        ax_rmse.plot(dates, rmse_means, marker="o", markersize=3, color=color, label=modality)
        if rmse_stds.any():
            ax_rmse.fill_between(dates, rmse_means - rmse_stds, rmse_means + rmse_stds, alpha=0.2, color=color)

    ax_mae.set_ylabel("MAE")
    ax_mae.set_title("MAE over Time (band = std across variables)")
    ax_mae.grid(True, alpha=0.3)
    ax_mae.legend(fontsize=8, ncol=2)

    ax_rmse.set_ylabel("RMSE")
    ax_rmse.set_title("RMSE over Time (band = std across variables)")
    ax_rmse.grid(True, alpha=0.3)
    ax_rmse.legend(fontsize=8, ncol=2)

    fig.autofmt_xdate()
    for ax in [ax_mae, ax_rmse]:
        ax.tick_params(axis="x", labelsize=12)
    fig.tight_layout()

    out_file = Path(output_dir) / f"{prefix}_timeseries_summary.png"
    fig.savefig(out_file, dpi=150)
    plt.close(fig)
    logging.info(f"Saved timeseries summary plot to {out_file}")


def save_aggregate_plots(rows, output_dir, prefix):
    """Save aggregate run-level plots."""
    plot_aggregate_information_loss(rows, output_dir, prefix)
    plot_aggregate_memory_improvement(rows, output_dir, prefix)
    plot_timeseries_summary(rows, output_dir, prefix)


def load_original_dataset(modality, data_dir, date_str):
    """Load original source dataset for non-L3 modalities."""
    if modality == "glorys":
        ds_orig = load_glorys_data_safe(data_dir, date_str)
    elif modality == "l4_ssh":
        ds_orig = pipeline_loaders.load_l4_ssh_data(data_dir, date_str)
    elif modality == "l4_sst":
        ds_orig = pipeline_loaders.load_l4_sst_data(data_dir, date_str)
    elif modality == "l4_sss":
        ds_orig = pipeline_loaders.load_l4_sss_data(data_dir, date_str)
    elif modality == "l4_wind":
        ds_orig = pipeline_loaders.load_l4_wind_data(data_dir, date_str)
    elif modality == "l3_sst":
        ds_orig = pipeline_loaders.load_l3_sst_data(data_dir, date_str)
    elif modality in ["l3_sss_smos_asc", "l3_sss_smos_desc"]:
        ds_orig = load_l3_sss_smos_pass_dataset(data_dir, date_str, modality)
    else:
        return None

    if ds_orig is not None:
        ds_orig = pipeline_coords.normalize_coords(ds_orig)
    return ds_orig


def process_date_modality_task(task):
    """Worker for one (date, modality) pair. Returns list of result rows."""
    date_str = task["date"]
    modality = task["modality"]
    region = task["region"]
    data_dir = task["data_dir"]
    formatted_dir = task["formatted_dir"]
    output_dir = task["output_dir"]
    save_plots = task["save_plots"]

    is_l3_track = modality in ["l3_swot", "l3_ssh"]
    rows = []

    ds_orig = None
    if not is_l3_track:
        ds_source = load_original_dataset(modality, data_dir, date_str)
        if ds_source is None:
            logging.warning(f"[{modality}/{date_str}] Could not load original source dataset. Skipping.")
            return rows
        ds_orig = build_pre_encoding_reference_dataset(modality, ds_source, region)
        if ds_orig is None:
            logging.warning(
                f"[{modality}/{date_str}] Could not build pre-encoding regional reference for region={region}. Skipping."
            )
            return rows

    fpath = find_formatted_file(formatted_dir, modality, region, date_str)
    if not fpath:
        logging.warning(
            f"[{modality}/{date_str}] No formatted file found for region={region} in {formatted_dir}. Skipping."
        )
        return rows

    with (
        xr.open_dataset(fpath, decode_timedelta=False) as ds_fmt,
        xr.open_dataset(fpath, decode_cf=False, mask_and_scale=False) as ds_fmt_raw,
    ):
        if is_l3_track:
            ds_orig = reconstruct_l3_data(modality, region, date_str, data_dir, ds_fmt)
            if ds_orig is None:
                return rows

        var_pairs = infer_variable_pairs(ds_orig, ds_fmt, modality)
        if not var_pairs:
            logging.warning(
                f"[{modality}/{date_str}] No compatible variable pairs found between original and formatted datasets. "
                f"Original vars: {list(ds_orig.data_vars)}, Formatted vars: {list(ds_fmt.data_vars)}"
            )
        for fmt_var, orig_var in var_pairs:
            stats = compute_loss_stats(ds_orig[orig_var], ds_fmt[fmt_var], fmt_var)
            if stats is None:
                logging.warning(
                    f"[{modality}/{date_str}/{region}] compute_loss_stats returned None for var='{fmt_var}' "
                    f"(no valid overlapping pixels). Skipping."
                )
                continue

            storage = extract_storage_metadata(ds_fmt_raw, fmt_var)
            if storage is None:
                logging.warning(
                    f"[{modality}/{date_str}/{region}] extract_storage_metadata returned None for var='{fmt_var}' "
                    f"(not present in raw file). Skipping."
                )
                continue

            row = {
                "date": date_str,
                "region": region,
                "modality": modality,
                "variable": fmt_var,
                "formatted_file": str(fpath),
                "representative_variable": REPRESENTATIVE_VARIABLES.get(modality),
                **stats,
                **storage,
            }
            rows.append(row)

            if save_plots:
                plot_loss_diagnostics(
                    ds_orig[orig_var],
                    ds_fmt[fmt_var],
                    fmt_var,
                    stats,
                    region,
                    modality,
                    output_dir,
                    date_str=date_str,
                )

    return rows


def compute_loss_stats(data_orig, data_fmt, var_name):
    """Compute compression error statistics and return them as dict."""
    if "time" in data_orig.dims:
        if data_orig.sizes["time"] == 1:
            data_orig = data_orig.squeeze("time")
        else:
            data_orig = data_orig.isel(time=0)

    if "time" in data_orig.coords:
        data_orig = data_orig.drop_vars("time")

    data_orig = reduce_to_lat_lon_2d(data_orig, name=f"original:{var_name}")
    data_fmt = reduce_to_lat_lon_2d(data_fmt, name=f"formatted:{var_name}")

    data_orig_interp = data_orig.interp(
        lat=data_fmt.lat.values, lon=data_fmt.lon.values, method="nearest"
    )

    v_orig = data_orig_interp.values
    v_fmt = data_fmt.values

    mask = ~np.isnan(v_orig) & ~np.isnan(v_fmt)
    if not mask.any():
        return None

    v_orig = v_orig[mask]
    v_fmt = v_fmt[mask]

    mean_orig = np.mean(v_orig)
    mean_fmt = np.mean(v_fmt)
    if mean_orig > 200 and mean_fmt < 100:
        logging.info(
            f"Converting Original {var_name} from Kelvin to Celsius for comparison."
        )
        v_orig = v_orig - 273.15

    error = v_orig - v_fmt
    abs_error = np.abs(error)
    nonzero_mask = abs_error > 0
    nonzero_fraction = float(np.mean(nonzero_mask))
    p99_nonzero = (
        float(np.percentile(abs_error[nonzero_mask], 99))
        if np.any(nonzero_mask)
        else 0.0
    )

    stats = {
        "n_valid": int(v_orig.size),
        "MAE": float(np.mean(abs_error)),
        "RMSE": float(np.sqrt(np.mean(error**2))),
        "Max_Abs_Error": float(np.max(abs_error)),
        "P99_Error": float(np.percentile(abs_error, 99)),
        "P99_Error_NonZero": p99_nonzero,
        "Error_NonZero_Fraction": nonzero_fraction,
        "Bias": float(np.mean(error)),
        "Orig_Min": float(np.min(v_orig)),
        "Orig_Max": float(np.max(v_orig)),
        "Fmt_Min": float(np.min(v_fmt)),
        "Fmt_Max": float(np.max(v_fmt)),
    }

    # For very sparse non-zero errors, the global P99 can collapse to zero even with
    # non-zero RMSE. Preserve raw P99 and expose a report-safe percentile.
    use_nonzero_p99 = bool(
        stats["P99_Error"] < stats["RMSE"]
        and stats["P99_Error_NonZero"] > 0.0
        and stats["Error_NonZero_Fraction"] < 0.1
    )
    stats["P99_Error_Report"] = (
        stats["P99_Error_NonZero"] if use_nonzero_p99 else stats["P99_Error"]
    )
    stats["P99_Uses_NonZero"] = use_nonzero_p99

    if stats["P99_Error"] < stats["RMSE"]:
        if stats["P99_Error"] == 0.0 and stats["RMSE"] > 0.0:
            logging.warning(
                "%s: P99_Error is zero while RMSE is non-zero (RMSE=%.3e). "
                "This indicates sparse non-zero errors (nonzero_fraction=%.6f, P99_nonzero=%.3e).",
                var_name,
                stats["RMSE"],
                stats["Error_NonZero_Fraction"],
                stats["P99_Error_NonZero"],
            )
        else:
            logging.warning(
                "Sanity check failed for %s: P99_Error (%.3e) < RMSE (%.3e), n_valid=%d, "
                "nonzero_fraction=%.6f, P99_nonzero=%.3e",
                var_name,
                stats["P99_Error"],
                stats["RMSE"],
                stats["n_valid"],
                stats["Error_NonZero_Fraction"],
                stats["P99_Error_NonZero"],
            )

    return stats


def plot_loss_diagnostics(
    data_orig, data_fmt, var_name, stats, region, modality, output_dir, date_str
):
    """Generate diagnostic comparison plot for one variable/date."""
    if "time" in data_orig.dims:
        if data_orig.sizes["time"] == 1:
            data_orig = data_orig.squeeze("time")
        else:
            data_orig = data_orig.isel(time=0)

    if "time" in data_orig.coords:
        data_orig = data_orig.drop_vars("time")

    data_orig = reduce_to_lat_lon_2d(data_orig, name=f"original:{var_name}")
    data_fmt = reduce_to_lat_lon_2d(data_fmt, name=f"formatted:{var_name}")

    data_orig_interp = data_orig.interp(
        lat=data_fmt.lat.values, lon=data_fmt.lon.values, method="nearest"
    )

    map_orig = (
        data_orig_interp.isel(time=0)
        if "time" in data_orig_interp.dims
        else data_orig_interp
    )
    map_fmt = data_fmt.isel(time=0) if "time" in data_fmt.dims else data_fmt

    if map_orig.mean() > 200 and map_fmt.mean() < 100:
        map_orig = map_orig - 273.15

    map_diff = map_orig - map_fmt

    v_orig = data_orig_interp.values
    v_fmt = data_fmt.values
    mask = ~np.isnan(v_orig) & ~np.isnan(v_fmt)
    if not mask.any():
        return

    v_orig = v_orig[mask]
    v_fmt = v_fmt[mask]
    error = v_orig - v_fmt

    fig = plt.figure(figsize=(20, 10))
    gs = fig.add_gridspec(2, 3)

    ax1 = fig.add_subplot(gs[0, 0])
    p1 = map_orig.plot(ax=ax1, cmap="viridis", cbar_kwargs={"label": "Distribution"})
    p1.colorbar.set_label("Distribution", fontsize=12)
    ax1.set_title(f"Original (Float32)\n{var_name}", fontsize=16)

    ax2 = fig.add_subplot(gs[0, 1])
    p2 = map_fmt.plot(ax=ax2, cmap="viridis", cbar_kwargs={"label": "Distribution"})
    p2.colorbar.set_label("Distribution", fontsize=12)
    ax2.set_title(f"Compressed (Int16)\n{var_name}", fontsize=16)

    ax3 = fig.add_subplot(gs[0, 2])
    p3 = map_diff.plot(ax=ax3, cmap="RdBu_r", cbar_kwargs={"label": "Error"})
    p3.colorbar.set_label("Error", fontsize=12)
    ax3.set_title(
        f"Difference (Orig - Comp)\nMAE: {stats['MAE']:.2e}, Max: {stats['Max_Abs_Error']:.2e}", fontsize=18
    )

    ax4 = fig.add_subplot(gs[1, :2])
    ax4.hist(error.flatten(), bins=100, log=True)

    ax4.set_title("Error Distribution (Log Count)", fontsize=18)
    ax4.set_xlabel("Error Value", fontsize=16)
    ax4.set_ylabel("Log Count", fontsize=16)
    ax4.grid(True, alpha=0.3)

    ax5 = fig.add_subplot(gs[1, 2])
    n_points = 10000
    if len(v_orig) > n_points:
        idx = np.random.choice(len(v_orig), n_points, replace=False)
        s_orig = v_orig[idx]
        s_fmt = v_fmt[idx]
    else:
        s_orig = v_orig
        s_fmt = v_fmt

    ax5.scatter(s_orig, s_fmt, alpha=0.1, s=1)
    min_val = min(s_orig.min(), s_fmt.min())
    max_val = max(s_orig.max(), s_fmt.max())
    ax5.plot([min_val, max_val], [min_val, max_val], "r--", alpha=0.5)
    ax5.set_xlabel("Original", fontsize=16)
    ax5.set_ylabel("Compressed", fontsize=16)
    ax5.set_title("Correlation", fontsize=18)

    for ax in [ax1, ax2, ax3, ax4, ax5]:
        ax.tick_params(axis="x", labelsize=14)
        ax.tick_params(axis="y", labelsize=14)

    plt.tight_layout()

    out_file = (
        Path(output_dir)
        / f"loss_analysis_{modality}_{region}_{date_str}_{var_name}.png"
    )
    plt.savefig(out_file, dpi=150)
    plt.close()
    logging.info(f"Saved plot to {out_file}")


def compute_and_plot_loss(data_orig, data_fmt, var_name, region, modality, output_dir):
    """Compute error stats and generate plot."""
    stats = compute_loss_stats(data_orig, data_fmt, var_name)

    if stats is None:
        logging.warning("No valid overlap found between datasets.")
        return None

    logging.info(f"--- Stats for {modality} {var_name} ({region}) ---")
    for k, v in stats.items():
        if isinstance(v, float):
            logging.info(f"{k}: {v:.6f}")
        else:
            logging.info(f"{k}: {v}")

    plot_loss_diagnostics(
        data_orig,
        data_fmt,
        var_name,
        stats,
        region,
        modality,
        output_dir,
        date_str="single",
    )
    return stats


def save_latex_table(results_file: str | Path, output_path: str | Path) -> None:
    """Aggregate compression-loss CSV and write an ESSD-style LaTeX table.

    Rows are grouped by (modality, variable). Statistics (RMSE, Bias, P99_Error)
    are summarised as mean ± std across all dates and regions in the CSV.
    """
    import statistics as _stats

    groups: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(
        lambda: {"RMSE": [], "Bias": [], "P99_Error": []}
    )
    with open(results_file, newline="") as f:
        for row in csv.DictReader(f):
            key = (row["modality"], row["variable"])
            for col in ("RMSE", "Bias"):
                try:
                    groups[key][col].append(float(row[col]))
                except (ValueError, KeyError):
                    pass
            p99_col = "P99_Error_Report" if "P99_Error_Report" in row else "P99_Error"
            try:
                groups[key]["P99_Error"].append(float(row[p99_col]))
            except (ValueError, KeyError):
                pass

    def fmt(values: list[float]) -> str:
        if not values:
            return "---"
        m = _stats.mean(values)
        s = _stats.stdev(values) if len(values) > 1 else 0.0
        if 0 < abs(m) < 0.001:
            return rf"${m:.2e} \pm {s:.1e}$"
        return rf"${m:.4f} \pm {s:.4f}$"

    lines = [
        r"\begin{table*}[t]",
        r"\caption{Compression-loss statistics for each data source after \texttt{int16}+\texttt{zlib}",
        r"encoding. For gridded products, errors are computed against a pre-encoding regional reference produced by the same formatting pipeline; for L3 track products, source swaths are conservatively binned to the target grid (no smoothing) and compared on overlapping valid cells only.",
        r"Values report mean\,$\pm$\,std across all dates and ocean regions; for sparse-error cases, the reported P99 uses the non-zero error tail (while raw P99 is retained in the CSV).}",
        r"\label{tab:compression_errors}",
        r"\begin{tabular}{llcrrr}",
        r"\tophline",
        r"Modality & Variable & Unit & RMSE (mean$\pm$std) & Bias (mean$\pm$std) & P99 Error (mean$\pm$std) \\",
        r"\middlehline",
    ]

    seen_modalities: set[str] = set()
    emitted_keys: set[tuple[str, str]] = set()
    prev_top_modality: str | None = None  # tracks "glorys" vs non-glorys for mid-rule

    for key in _TABLE_ROW_ORDER:
        if key not in groups:
            continue
        # Skip duplicate (modality, variable) pairs (e.g. l4_sss appears with both "sos" and "so")
        modality = key[0]
        if modality in seen_modalities and modality != "glorys":
            continue
        if key in emitted_keys:
            continue
        emitted_keys.add(key)

        mod_display, var_display, unit = _VARIABLE_DISPLAY.get(key, (key[0], key[1], ""))
        vals = groups[key]

        top_modality = "glorys" if modality == "glorys" else "other"
        if prev_top_modality == "glorys" and top_modality != "glorys":
            lines.append(r"\middlehline")
        prev_top_modality = top_modality

        # Show modality name only for the first row of each modality group
        if mod_display in seen_modalities:
            mod_cell = ""
        else:
            mod_cell = mod_display
            seen_modalities.add(mod_display)

        if modality != "glorys":
            seen_modalities.add(modality)

        rmse_str = fmt(vals["RMSE"])
        bias_str = fmt(vals["Bias"])
        p99_str  = fmt(vals["P99_Error"])
        lines.append(
            rf"{mod_cell} & {var_display} & {unit} & {rmse_str} & {bias_str} & {p99_str} \\"
        )

    lines += [
        r"\bottomhline",
        r"\end{tabular}",
        r"\end{table*}",
    ]

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text("\n".join(lines) + "\n")
    logging.info(f"Saved LaTeX compression-error table to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze compression loss.")
    parser.add_argument("--date", default="20230402", help="Single date YYYYMMDD")
    parser.add_argument(
        "--min-date", default=None, help="Start date YYYYMMDD (inclusive)"
    )
    parser.add_argument(
        "--max-date", default=None, help="End date YYYYMMDD (inclusive)"
    )
    parser.add_argument("--data-dir", required=True, help="Root of ORIGINAL data")
    parser.add_argument("--formatted-dir", required=True, help="Root of FORMATTED data")
    parser.add_argument(
        "--regions",
        nargs="+",
        default=["NORTH_ATLANTIC"],
        metavar="REGION",
        help=(
            "One or more ocean regions to process (space-separated), or 'all'. "
            f"Choices: {list(pipeline_constants.SPATIAL_REGIONS.keys())}"
        ),
    )
    parser.add_argument(
        "--modality",
        nargs="+",
        default=["all"],
        help="One or more modalities (space/comma-separated), e.g. glorys l4_ssh or glorys,l4_ssh",
    )
    parser.add_argument("--output-dir", default=".", help="Where to save plots")
    parser.add_argument(
        "--results-file", default=None, help="CSV output path for per-date metrics"
    )
    parser.add_argument(
        "--save-plots", action="store_true", help="Also save per-date diagnostic plots"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 2) - 1),
        help="Parallel workers across date×modality tasks",
    )
    parser.add_argument(
        "--no-aggregate-plots",
        action="store_true",
        help="Disable aggregate run-level plots",
    )
    parser.add_argument(
        "--aggregate-prefix",
        default="compression_loss",
        help="Filename prefix for aggregate plot outputs",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable DEBUG-level logging for detailed tracing",
    )
    parser.add_argument(
        "--latex-table",
        default=None,
        help="Path to save ESSD-style LaTeX reconstruction-error table (.tex)",
    )
    args = parser.parse_args()

    logging.info(f"Using formatter pipeline from: {_PIPELINE_SOURCE}")

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    date_list = parse_date_range(args.date, args.min_date, args.max_date)
    modality_list = parse_modalities(args.modality)

    raw_regions = args.regions
    if raw_regions == ["all"]:
        region_list = list(pipeline_constants.SPATIAL_REGIONS.keys())
    else:
        region_list = raw_regions
        invalid = [r for r in region_list if r not in pipeline_constants.SPATIAL_REGIONS]
        if invalid:
            raise ValueError(f"Unknown region(s): {invalid}. Supported: {list(pipeline_constants.SPATIAL_REGIONS.keys())}")

    # Validate output dir
    Path(args.output_dir).mkdir(exist_ok=True, parents=True)

    if args.results_file:
        results_path = Path(args.results_file)
    else:
        date_suffix = (
            date_list[0] if len(date_list) == 1 else f"{date_list[0]}_{date_list[-1]}"
        )
        modality_suffix = "all" if len(modality_list) > 1 else modality_list[0]
        region_suffix = "all_regions" if len(region_list) > 1 else region_list[0]
        results_path = (
            Path(args.output_dir)
            / f"compression_loss_metrics_{modality_suffix}_{region_suffix}_{date_suffix}.csv"
        )

    results_path.parent.mkdir(parents=True, exist_ok=True)

    all_results = []

    logging.info(
        f"Processing {len(date_list)} date(s) x {len(modality_list)} modality(ies) x "
        f"{len(region_list)} region(s) with workers={args.workers}"
    )

    tasks = [
        {
            "date": date_str,
            "modality": modality,
            "region": region,
            "data_dir": args.data_dir,
            "formatted_dir": args.formatted_dir,
            "output_dir": args.output_dir,
            "save_plots": args.save_plots,
        }
        for date_str in date_list
        for modality in modality_list
        for region in region_list
    ]

    n_tasks = len(tasks)
    if args.workers <= 1 or n_tasks <= 1:
        for i, task in enumerate(tasks, 1):
            rows = process_date_modality_task(task)
            all_results.extend(rows)
            logging.info(
                f"[{i}/{n_tasks}] Finished {task['modality']} {task['region']} {task['date']}"
                f" -> {len(rows)} metric row(s)"
            )

    else:
        max_workers = min(args.workers, n_tasks)
        completed = 0
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(process_date_modality_task, task): task
                for task in tasks
            }
            for future in as_completed(futures):
                task = futures[future]
                rows = future.result()
                all_results.extend(rows)
                completed += 1
                logging.info(
                    f"[{completed}/{n_tasks}] Finished {task['modality']} {task['region']} {task['date']}"
                    f" -> {len(rows)} metric row(s)"
                )

    if not all_results:
        logging.error("No compression metrics were computed. Check inputs/date range.")
        sys.exit(1)

    fieldnames = [
        "date",
        "region",
        "modality",
        "variable",
        "formatted_file",
        "n_valid",
        "MAE",
        "RMSE",
        "Max_Abs_Error",
        "P99_Error",
        "P99_Error_NonZero",
        "P99_Error_Report",
        "P99_Uses_NonZero",
        "Error_NonZero_Fraction",
        "Bias",
        "Orig_Min",
        "Orig_Max",
        "Fmt_Min",
        "Fmt_Max",
        "representative_variable",
        "n_elements",
        "stored_dtype",
        "stored_itemsize",
        "stored_bytes_uncompressed",
        "float32_bytes",
        "memory_gain_ratio",
        "memory_gain_pct",
        "is_int16_packed",
    ]
    with open(results_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)

    logging.info(f"Saved {len(all_results)} metric rows to {results_path}")

    if args.latex_table:
        save_latex_table(results_path, args.latex_table)

    if not args.no_aggregate_plots:
        save_aggregate_plots(all_results, args.output_dir, args.aggregate_prefix)


if __name__ == "__main__":
    main()
