#!/usr/bin/env python3
"""Plot Argo validation results."""

import argparse
import concurrent.futures
import glob
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path

import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.colors as mcolors
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy import stats as scipy_stats
from tqdm.auto import tqdm

from ocean_taco.dataset.retrieve import (
    HF_DEFAULT_URL,
    load_hf_dataset,
    load_region_product_nc,
)


def _configure_cartopy_dir(path: str):
    """Configure cartopy data directory."""
    p = Path(path).expanduser()
    p.mkdir(parents=True, exist_ok=True)
    os.environ["CARTOPY_USER_DIR"] = str(p)
    cartopy.config["data_dir"] = str(p)


_configure_cartopy_dir("./.cartopy")

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


# ═══════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════

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

# Short display names for region plots
REGION_SHORT = {
    "SOUTH_PACIFIC_WEST": "S.Pac W",
    "SOUTH_ATLANTIC": "S.Atl",
    "SOUTH_INDIAN": "S.Ind",
    "SOUTH_PACIFIC_EAST": "S.Pac E",
    "NORTH_PACIFIC_WEST": "N.Pac W",
    "NORTH_ATLANTIC": "N.Atl",
    "NORTH_INDIAN": "N.Ind",
    "NORTH_PACIFIC_EAST": "N.Pac E",
}

PRODUCT_SPECS = {
    "glorys": {
        "sst": {
            "subdir": "glorys/",
            "filename": "glorys_{region}_{date}.nc",
            "var_candidates": ["thetao", "sst", "temperature"],
            "label": "GLORYS12",
            "color": "#3498db",
            "marker": "o",
        },
        "sss": {
            "subdir": "glorys",
            "filename": "glorys_{region}_{date}.nc",
            "var_candidates": ["so", "sss", "salinity"],
            "label": "GLORYS12",
            "color": "#3498db",
            "marker": "o",
        },
    },
    "l4": {
        "sst": {
            "subdir": "l4_sst",
            "filename": "l4_sst_{region}_{date}.nc",
            "var_candidates": [
                "analysed_sst",
                "sea_surface_temperature",
                "sst",
                "temperature",
            ],
            "label": "L4 Obs",
            "color": "#e74c3c",
            "marker": "s",
        },
        "sss": {
            "subdir": "l4_sss",
            "filename": "l4_sss_{region}_{date}.nc",
            "var_candidates": ["sss", "so", "salinity", "sea_surface_salinity"],
            "label": "L4 Obs",
            "color": "#e74c3c",
            "marker": "s",
        },
    },
}

SEASON_MAP = {
    12: "DJF",
    1: "DJF",
    2: "DJF",
    3: "MAM",
    4: "MAM",
    5: "MAM",
    6: "JJA",
    7: "JJA",
    8: "JJA",
    9: "SON",
    10: "SON",
    11: "SON",
}
SEASON_ORDER = ["DJF", "MAM", "JJA", "SON"]


# ═══════════════════════════════════════════════════════════════════════
# Utilities
# ═══════════════════════════════════════════════════════════════════════


def lon_to_180(a):
    a = np.asarray(a, dtype=float)
    out = ((a + 180.0) % 360.0) - 180.0
    out[out == 180.0] = -180.0
    return out


def identify_region(lon, lat):
    for name, bounds in SPATIAL_REGIONS.items():
        if (
            bounds["lat"][0] <= lat < bounds["lat"][1]
            and bounds["lon"][0] <= lon < bounds["lon"][1]
        ):
            return name
    return None


def generate_date_list(date_min, date_max):
    start = datetime.strptime(date_min, "%Y-%m-%d")
    end = datetime.strptime(date_max, "%Y-%m-%d")
    dates, cur = [], start
    while cur <= end:
        dates.append(cur.strftime("%Y%m%d"))
        cur += timedelta(days=1)
    return dates


def _find_key(ds, candidates):
    all_keys = set(ds.coords) | set(ds.data_vars)
    for c in candidates:
        if c in all_keys:
            return c
    return None


def _extract_single_file_task(task):
    fpath, idx_arr, lat_arr, lon_arr, var_candidates, tol = task
    values = np.full(len(idx_arr), np.nan)
    if not os.path.exists(fpath):
        return idx_arr, values

    ds = None
    try:
        ds = xr.open_dataset(fpath)
        var_name = None
        for vc in var_candidates:
            if vc in ds.data_vars:
                var_name = vc
                break
        if var_name is None:
            return idx_arr, values

        for i, (lat_v, lon_v) in enumerate(zip(lat_arr, lon_arr)):
            try:
                val = (
                    ds[var_name]
                    .sel(
                        lat=float(lat_v),
                        lon=float(lon_v),
                        method="nearest",
                        tolerance=tol,
                    )
                    .values
                )
                val = float(np.squeeze(val))
                if np.isfinite(val):
                    values[i] = val
            except (KeyError, IndexError, ValueError):
                continue
    except Exception:
        return idx_arr, values
    finally:
        if ds is not None:
            ds.close()

    return idx_arr, values


# ═══════════════════════════════════════════════════════════════════════
# Step 1 — Load Argo surface observations
# ═══════════════════════════════════════════════════════════════════════


def load_argo_surface(taco_dir, date_str, pressure_max=5.0):
    """Load Argo from TACO-processed regional files, extract P < pressure_max."""
    argo_dir = os.path.join(taco_dir, "argo")
    pattern = os.path.join(argo_dir, f"argo_*_{date_str}.nc")
    files = sorted(glob.glob(pattern))
    rows = []

    for fpath in files:
        basename = os.path.basename(fpath)
        parts = basename.replace(".nc", "").split("_")
        region = "_".join(parts[1:-1])

        try:
            ds = xr.open_dataset(fpath)
        except Exception:
            continue

        lon_key = _find_key(ds, ["lon", "LONGITUDE", "longitude"])
        lat_key = _find_key(ds, ["lat", "LATITUDE", "latitude"])
        time_key = _find_key(ds, ["TIME", "time", "JULD"])
        pres_key = _find_key(ds, ["PRES", "pressure", "PRES_ADJUSTED"])
        temp_key = _find_key(
            ds,
            ["TEMP", "temperature", "TEMP_ADJUSTED", "sea_water_temperature", "thetao"],
        )
        psal_key = _find_key(
            ds, ["PSAL", "salinity", "PSAL_ADJUSTED", "sea_water_salinity", "so"]
        )

        if lon_key is None or lat_key is None:
            ds.close()
            continue

        lons = lon_to_180(np.asarray(ds[lon_key].values).flatten())
        lats = np.asarray(ds[lat_key].values).flatten()
        if time_key and time_key in ds:
            times = pd.to_datetime(
                np.asarray(ds[time_key].values).flatten(), errors="coerce"
            )
        else:
            times = np.full(
                len(lons), pd.Timestamp(datetime.strptime(date_str, "%Y%m%d"))
            )

        if pres_key and pres_key in ds:
            pres = np.asarray(ds[pres_key].values)
            if pres.ndim == 2:
                for i in range(pres.shape[0]):
                    smask = pres[i, :] < pressure_max
                    if not np.any(smask):
                        continue
                    j = np.nanargmin(np.where(smask, pres[i, :], np.inf))
                    row = {
                        "lat": float(lats[i]),
                        "lon": float(lons[i]),
                        "time": times[i] if i < len(times) else pd.NaT,
                        "region": region,
                        "date": date_str,
                    }
                    if temp_key and temp_key in ds:
                        v = float(ds[temp_key].values[i, j])
                        row["temp"] = v if np.isfinite(v) else np.nan
                    if psal_key and psal_key in ds:
                        v = float(ds[psal_key].values[i, j])
                        row["psal"] = v if np.isfinite(v) else np.nan
                    rows.append(row)
            elif pres.ndim == 1:
                for i in np.where(pres < pressure_max)[0]:
                    row = {
                        "lat": float(lats[i]),
                        "lon": float(lons[i]),
                        "time": times[i] if i < len(times) else pd.NaT,
                        "region": region,
                        "date": date_str,
                    }
                    if temp_key and temp_key in ds:
                        v = float(ds[temp_key].values[i])
                        row["temp"] = v if np.isfinite(v) else np.nan
                    if psal_key and psal_key in ds:
                        v = float(ds[psal_key].values[i])
                        row["psal"] = v if np.isfinite(v) else np.nan
                    rows.append(row)
        else:
            # No pressure dim — treat all as surface
            for i in range(len(lons)):
                row = {
                    "lat": float(lats[i]),
                    "lon": float(lons[i]),
                    "time": times[i] if i < len(times) else pd.NaT,
                    "region": region,
                    "date": date_str,
                }
                if temp_key and temp_key in ds:
                    vals = np.asarray(ds[temp_key].values).flatten()
                    row["temp"] = (
                        float(vals[i])
                        if i < len(vals) and np.isfinite(vals[i])
                        else np.nan
                    )
                if psal_key and psal_key in ds:
                    vals = np.asarray(ds[psal_key].values).flatten()
                    row["psal"] = (
                        float(vals[i])
                        if i < len(vals) and np.isfinite(vals[i])
                        else np.nan
                    )
                rows.append(row)
        ds.close()

    if not rows:
        return pd.DataFrame(
            columns=["lat", "lon", "time", "temp", "psal", "region", "date"]
        )
    df = pd.DataFrame(rows)
    for col in ["temp", "psal"]:
        if col not in df.columns:
            df[col] = np.nan
    return df


# ═══════════════════════════════════════════════════════════════════════
# Step 2 — Extract gridded product values at Argo locations
# ═══════════════════════════════════════════════════════════════════════


def extract_product_at_argo(
    taco_dir, argo_df, product_name, variable, tolerance_deg=0.5, n_workers=1
):
    """Extract nearest pixel from a gridded product at each Argo location.

    When n_workers > 1, extraction is parallelized across source files.
    """
    spec = PRODUCT_SPECS[product_name][variable]
    result = np.full(len(argo_df), np.nan)
    file_to_indices = {}
    lats = np.asarray(argo_df["lat"].values)
    lons = np.asarray(argo_df["lon"].values)
    regions = np.asarray(argo_df["region"].values)
    dates = np.asarray(argo_df["date"].values)

    for idx in range(len(argo_df)):
        subdir = spec["subdir"].format(region=regions[idx])
        fname = spec["filename"].format(region=regions[idx], date=dates[idx])
        fpath = os.path.join(taco_dir, subdir, fname)
        file_to_indices.setdefault(fpath, []).append(idx)

    tasks = []
    for fpath, idx_list in file_to_indices.items():
        idx_arr = np.asarray(idx_list, dtype=int)
        tasks.append(
            (
                fpath,
                idx_arr,
                lats[idx_arr],
                lons[idx_arr],
                tuple(spec["var_candidates"]),
                tolerance_deg,
            )
        )

    pbar_desc = f"Extract {product_name} {variable}"
    pbar_total = len(tasks)

    if n_workers is None or n_workers <= 1:
        with tqdm(total=pbar_total, desc=pbar_desc, unit="file") as pbar:
            for task in tasks:
                idx_arr, vals = _extract_single_file_task(task)
                result[idx_arr] = vals
                pbar.update(1)
    else:
        max_workers = max(1, int(n_workers))
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as ex:
            futures = [ex.submit(_extract_single_file_task, task) for task in tasks]
            with tqdm(total=pbar_total, desc=pbar_desc, unit="file") as pbar:
                for fut in concurrent.futures.as_completed(futures):
                    idx_arr, vals = fut.result()
                    result[idx_arr] = vals
                    pbar.update(1)

    return result


# ═══════════════════════════════════════════════════════════════════════
# Step 3 — Skill Metrics
# ═══════════════════════════════════════════════════════════════════════


def compute_metrics(obs, model):
    mask = np.isfinite(obs) & np.isfinite(model)
    obs, model = np.asarray(obs)[mask], np.asarray(model)[mask]
    n = len(obs)
    if n < 3:
        return None
    bias = float(np.mean(model - obs))
    rmse = float(np.sqrt(np.mean((model - obs) ** 2)))
    crmse = float(np.sqrt(np.mean(((model - model.mean()) - (obs - obs.mean())) ** 2)))
    std_o, std_m = float(np.std(obs, ddof=1)), float(np.std(model, ddof=1))
    corr, pval = scipy_stats.pearsonr(obs, model)
    return {
        "n": n,
        "bias": bias,
        "rmse": rmse,
        "crmse": crmse,
        "correlation": float(corr),
        "p_value": float(pval),
        "std_obs": std_o,
        "std_model": std_m,
        "std_ratio": std_m / std_o if std_o > 0 else np.nan,
        "norm_bias": bias / std_o if std_o > 0 else np.nan,
        "norm_crmse": crmse / std_o if std_o > 0 else np.nan,
    }


# ═══════════════════════════════════════════════════════════════════════
# Plotting helper — map axis
# ═══════════════════════════════════════════════════════════════════════


def make_map_ax(fig, subplot_spec, projection=None):
    """Create a cartopy or plain axis with land/coastlines."""
    proj = projection or ccrs.Robinson()
    ax = fig.add_subplot(subplot_spec, projection=proj)
    ax.set_global()
    ax.add_feature(cfeature.LAND, facecolor="#f0ece3", edgecolor="none")
    ax.add_feature(cfeature.COASTLINE, linewidth=0.3, color="#888888")
    ax.gridlines(linewidth=0.15, alpha=0.3)
    return ax


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 1 — Aggregated Spatial Bias + Hovmöller + Latitude Profile
#
# This is the "hero" figure: it shows spatial bias accumulated over the
# entire validation period, then reveals how that bias varies in time.
#
# Layout (per variable, per product):
#   ┌────────────────────────────────────┐
#   │  Top:  2°×2° binned mean bias      │
#   │        (full-period aggregate)      │
#   ├──────────────────┬─────────────────┤
#   │  Left: Hovmöller │ Right: Lat bias │
#   │  (lat × month)   │ profile ± spread│
#   └──────────────────┴─────────────────┘
# ═══════════════════════════════════════════════════════════════════════


def plot_aggregated_bias_with_hovmoller(df, variable, output_dir, dpi=300):
    """For each product (GLORYS, L4): one composite figure with
    Top:   Period-aggregated 2°×2° spatial bias map
    Bot-L: Hovmöller diagram (latitude × year-month)
    Bot-R: Latitude profile (full period mean ± monthly spread)
    """
    argo_col = "temp" if variable == "sst" else "psal"
    unit = "°C" if variable == "sst" else "PSU"
    var_label = variable.upper()

    # We need a proper datetime for monthly grouping
    df = df.copy()
    df["datetime"] = pd.to_datetime(df["date"], format="%Y%m%d", errors="coerce")
    df["year_month"] = df["datetime"].dt.to_period("M")

    for pname in ["glorys", "l4"]:
        col = f"{pname}_{variable}"
        if col not in df.columns:
            continue
        spec = PRODUCT_SPECS[pname][variable]

        mask = df[argo_col].notna() & df[col].notna()
        sub = df[mask].copy()
        if len(sub) < 20:
            logging.warning(
                f"  Skipping {pname} {var_label}: only {len(sub)} matchups."
            )
            continue

        sub["bias"] = sub[col] - sub[argo_col]
        bias_lim = np.percentile(np.abs(sub["bias"].dropna()), 95)
        bias_lim = max(bias_lim, 0.05)
        bias_norm = mcolors.TwoSlopeNorm(vmin=-bias_lim, vcenter=0, vmax=bias_lim)
        bias_cmap = "RdBu_r"

        global_metrics = compute_metrics(sub[argo_col].values, sub[col].values)
        n_total = len(sub)
        period_str = (
            f"{sub['datetime'].min().strftime('%Y-%m-%d')} → "
            f"{sub['datetime'].max().strftime('%Y-%m-%d')}"
        )

        # ── Figure layout ────────────────────────────────────────────
        fig = plt.figure(figsize=(14, 11))
        gs = gridspec.GridSpec(
            2,
            2,
            height_ratios=[1.2, 1.0],
            width_ratios=[1.4, 1.0],
            hspace=0.28,
            wspace=0.20,
            left=0.06,
            right=0.94,
            top=0.91,
            bottom=0.06,
        )

        # ── Top row: spatial map spanning both columns ───────────────
        gs_top = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs[0, :])
        ax_map = make_map_ax(fig, gs_top[0])

        # Bin into 2°×2° cells
        lon_bins = np.arange(-180, 182, 2)
        lat_bins = np.arange(-90, 92, 2)
        lon_c = (lon_bins[:-1] + lon_bins[1:]) / 2
        lat_c = (lat_bins[:-1] + lat_bins[1:]) / 2

        sub_b = sub.copy()
        sub_b["lon_bin"] = pd.cut(sub_b["lon"], bins=lon_bins, labels=lon_c)
        sub_b["lat_bin"] = pd.cut(sub_b["lat"], bins=lat_bins, labels=lat_c)
        binned = sub_b.groupby(["lat_bin", "lon_bin"], observed=True)["bias"]
        mean_bias = binned.mean()
        count_per_bin = binned.count()

        grid_bias = np.full((len(lat_c), len(lon_c)), np.nan)
        grid_count = np.full((len(lat_c), len(lon_c)), 0)
        for (lat_v, lon_v), val in mean_bias.items():
            li = np.searchsorted(lat_c, float(lat_v))
            lj = np.searchsorted(lon_c, float(lon_v))
            if 0 <= li < len(lat_c) and 0 <= lj < len(lon_c):
                grid_bias[li, lj] = val
                grid_count[li, lj] = count_per_bin.get((lat_v, lon_v), 0)

        # Mask cells with very few obs (< 3) to avoid noisy outliers
        grid_bias[grid_count < 3] = np.nan

        transform = ccrs.PlateCarree()
        kw = {"transform": transform} if transform else {}
        pcm = ax_map.pcolormesh(
            lon_c,
            lat_c,
            grid_bias,
            cmap=bias_cmap,
            norm=bias_norm,
            rasterized=True,
            **kw,
        )
        cb = fig.colorbar(
            pcm,
            ax=ax_map,
            shrink=0.65,
            pad=0.02,
            orientation="horizontal",
            extend="both",
        )
        cb.set_label(f"Mean Bias ({unit})", fontsize=9)
        cb.ax.tick_params(labelsize=7)

        m_str = ""
        if global_metrics:
            m_str = (
                f"Bias={global_metrics['bias']:+.3f} {unit}   "
                f"RMSE={global_metrics['rmse']:.3f} {unit}   "
                f"R={global_metrics['correlation']:.3f}"
            )
        ax_map.set_title(
            f"{spec['label']} {var_label} Aggregated Bias (2°×2°)\n"
            f"{period_str}   •   n={n_total:,}   •   {m_str}",
            fontsize=12,
        )

        # ── Bottom-left: Hovmöller (latitude × year-month) ──────────
        ax_hov = fig.add_subplot(gs[1, 0])

        lat_edges_hov = np.arange(-90, 92, 5)
        lat_mids_hov = (lat_edges_hov[:-1] + lat_edges_hov[1:]) / 2
        sub_h = sub.copy()
        sub_h["lat_band"] = pd.cut(
            sub_h["lat"], bins=lat_edges_hov, labels=lat_mids_hov
        )

        months = sorted(sub_h["year_month"].dropna().unique())
        if len(months) < 2:
            ax_hov.text(
                0.5,
                0.5,
                "Need ≥2 months for Hovmöller",
                transform=ax_hov.transAxes,
                ha="center",
                fontsize=10,
            )
        else:
            hov_grid = np.full((len(lat_mids_hov), len(months)), np.nan)
            month_labels = []
            for j, ym in enumerate(months):
                month_data = sub_h[sub_h["year_month"] == ym]
                grouped = month_data.groupby("lat_band", observed=True)["bias"]
                for lat_v, val in grouped.mean().items():
                    cnt = grouped.count().get(lat_v, 0)
                    if cnt >= 3:
                        li = np.searchsorted(lat_mids_hov, float(lat_v))
                        if 0 <= li < len(lat_mids_hov):
                            hov_grid[li, j] = val
                month_labels.append(str(ym))

            pcm_h = ax_hov.pcolormesh(
                np.arange(len(months) + 1) - 0.5,
                lat_edges_hov,
                hov_grid,
                cmap=bias_cmap,
                norm=bias_norm,
                rasterized=True,
            )

            ax_hov.set_xticks(np.arange(len(months)))
            # Show labels sparsely for readability
            skip = max(1, len(months) // 12)
            labels = [
                month_labels[i] if i % skip == 0 else "" for i in range(len(months))
            ]
            ax_hov.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)

            ax_hov.set_ylabel("Latitude (°)", fontsize=9)
            ax_hov.set_xlabel("Month", fontsize=9)
            cb_h = fig.colorbar(pcm_h, ax=ax_hov, shrink=0.8, pad=0.02)
            cb_h.set_label(f"Bias ({unit})", fontsize=8)
            cb_h.ax.tick_params(labelsize=7)

        ax_hov.set_title(f"Hovmöller — {var_label} Bias (5° lat × month)", fontsize=12)

        # ── Bottom-right: Latitude profile with temporal spread ──────
        ax_lat = fig.add_subplot(gs[1, 1])

        # Full-period mean ± std
        full_grouped = sub_h.groupby("lat_band", observed=True)["bias"]
        full_mean = full_grouped.mean()
        full_std = full_grouped.std()
        full_count = full_grouped.count()
        valid = full_count >= 3
        x_lat = full_mean.index.astype(float)[valid]
        y_mean = full_mean.values[valid]
        y_std = full_std.values[valid]

        ax_lat.fill_betweenx(
            x_lat,
            y_mean - y_std,
            y_mean + y_std,
            alpha=0.15,
            color=spec["color"],
            label="±1σ (temporal)",
        )
        ax_lat.plot(
            y_mean,
            x_lat,
            color=spec["color"],
            linewidth=2,
            label=f"Mean (n={n_total:,})",
        )

        # Overlay individual monthly profiles as thin lines
        if len(months) >= 2:
            for ym in months:
                month_data = sub_h[sub_h["year_month"] == ym]
                mg = month_data.groupby("lat_band", observed=True)["bias"]
                mm = mg.mean()
                mc = mg.count()
                ok = mc >= 3
                if ok.sum() >= 2:
                    ax_lat.plot(
                        mm.values[ok],
                        mm.index.astype(float)[ok],
                        color=spec["color"],
                        alpha=0.12,
                        linewidth=0.5,
                    )

        ax_lat.axvline(0, color="black", linewidth=0.5, linestyle="--")
        ax_lat.set_xlabel(f"Mean Bias ({unit})", fontsize=9)
        ax_lat.set_ylabel("Latitude (°)", fontsize=9)
        ax_lat.set_ylim(-90, 90)
        ax_lat.set_title("Latitude Bias Profile", fontsize=12)
        ax_lat.legend(fontsize=8, loc="best")
        ax_lat.grid(True, alpha=0.2)

        fig.suptitle(
            f"Spatial Bias Analysis for {spec['label']} {var_label} vs. Argo",
            fontsize=13,
            y=0.97,
        )

        out_path = os.path.join(output_dir, f"fig1_spatial_bias_{pname}_{variable}.pdf")
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
        logging.info(f"✓ {out_path}")
        plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 2 — Regional Time Series (monthly RMSE + bias per region)
# ═══════════════════════════════════════════════════════════════════════


def plot_regional_timeseries(df, variable, output_dir, dpi=300):
    """8-panel figure (one per region): monthly mean bias with ±1σ spread
    for both GLORYS and L4. Shows temporal stability and variability.
    """
    argo_col = "temp" if variable == "sst" else "psal"
    unit = "°C" if variable == "sst" else "PSU"
    var_label = variable.upper()

    df = df.copy()
    df["datetime"] = pd.to_datetime(df["date"], format="%Y%m%d", errors="coerce")
    df["year_month"] = df["datetime"].dt.to_period("M")

    ordered_regions = [
        "NORTH_PACIFIC_WEST",
        "NORTH_ATLANTIC",
        "NORTH_INDIAN",
        "NORTH_PACIFIC_EAST",
        "SOUTH_PACIFIC_WEST",
        "SOUTH_ATLANTIC",
        "SOUTH_INDIAN",
        "SOUTH_PACIFIC_EAST",
    ]
    regions_present = set(df["region"].unique())
    if not regions_present:
        return

    n_cols = 4
    n_rows = 2
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(4.6 * n_cols, 3.6 * n_rows), sharex=True
    )
    axes = np.atleast_2d(axes)

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=PRODUCT_SPECS["glorys"][variable]["color"],
            lw=2,
            linestyle="-",
            label="GLORYS Bias",
        ),
        Patch(
            facecolor=PRODUCT_SPECS["glorys"][variable]["color"],
            alpha=0.18,
            edgecolor="none",
            label="GLORYS ±1σ",
        ),
        Line2D(
            [0],
            [0],
            color=PRODUCT_SPECS["l4"][variable]["color"],
            lw=2,
            linestyle="-",
            label="L4 Bias",
        ),
        Patch(
            facecolor=PRODUCT_SPECS["l4"][variable]["color"],
            alpha=0.18,
            edgecolor="none",
            label="L4 ±1σ",
        ),
    ]

    for idx, rname in enumerate(ordered_regions):
        row_i, col_i = divmod(idx, n_cols)
        ax = axes[row_i, col_i]

        if rname not in regions_present:
            ax.text(
                0.5,
                0.5,
                "No data",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=11,
            )
            ax.set_title(rname.replace("_", " ").title(), fontsize=12)
            ax.grid(True, alpha=0.15)
            continue

        rsub = df[df["region"] == rname].copy()

        for pname in ["glorys", "l4"]:
            col = f"{pname}_{variable}"
            if col not in rsub.columns:
                continue
            spec = PRODUCT_SPECS[pname][variable]

            mask = rsub[argo_col].notna() & rsub[col].notna()
            vsub = rsub[mask].copy()
            if len(vsub) < 5:
                continue

            vsub["bias"] = vsub[col] - vsub[argo_col]
            monthly = vsub.groupby("year_month", observed=True).agg(
                bias_mean=("bias", "mean"),
                bias_std=("bias", "std"),
                count=("bias", "count"),
            )
            monthly = monthly[monthly["count"] >= 3]
            if monthly.empty:
                continue

            x = monthly.index.to_timestamp()
            std = monthly["bias_std"].fillna(0.0)
            ax.fill_between(
                x,
                monthly["bias_mean"] - std,
                monthly["bias_mean"] + std,
                color=spec["color"],
                alpha=0.18,
                linewidth=0,
            )
            ax.plot(
                x,
                monthly["bias_mean"],
                color=spec["color"],
                linewidth=1.2,
                label=f"{spec['label']} bias",
            )

        ax.axhline(0, color="black", linewidth=0.4, linestyle="-")
        ax.set_title(rname.replace("_", " ").title(), fontsize=12)
        ax.tick_params(labelsize=9)
        ax.grid(True, alpha=0.15)
        if col_i == 0:
            ax.set_ylabel(f"Bias ({unit})", fontsize=10)

    for ax in axes[-1, :]:
        if ax.get_visible():
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
            ax.tick_params(axis="x", rotation=45, labelsize=10)

    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=4,
        bbox_to_anchor=(0.5, 0.01),
        frameon=True,
        fontsize=12,
        columnspacing=2.0,
        handlelength=3.0,
    )

    fig.suptitle(
        f"Monthly {var_label} Bias by Region — Gridded Products vs. Argo", fontsize=16
    )
    fig.tight_layout(rect=[0, 0.08, 1, 0.94])

    out_path = os.path.join(output_dir, f"fig2_regional_timeseries_{variable}.pdf")
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    logging.info(f"✓ {out_path}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 3 — Seasonal Spatial Bias Maps (DJF / MAM / JJA / SON)
# ═══════════════════════════════════════════════════════════════════════


def plot_seasonal_bias(df, variable, output_dir, dpi=300):
    """4-panel seasonal map per product: 2°×2° binned mean bias
    for DJF, MAM, JJA, SON. Shows how bias pattern changes with season.
    """
    argo_col = "temp" if variable == "sst" else "psal"
    unit = "°C" if variable == "sst" else "PSU"
    var_label = variable.upper()

    df = df.copy()
    df["datetime"] = pd.to_datetime(df["date"], format="%Y%m%d", errors="coerce")
    df["month"] = df["datetime"].dt.month
    df["season"] = df["month"].map(SEASON_MAP)

    lon_bins = np.arange(-180, 182, 2)
    lat_bins = np.arange(-90, 92, 2)
    lon_c = (lon_bins[:-1] + lon_bins[1:]) / 2
    lat_c = (lat_bins[:-1] + lat_bins[1:]) / 2

    for pname in ["glorys", "l4"]:
        col = f"{pname}_{variable}"
        if col not in df.columns:
            continue
        spec = PRODUCT_SPECS[pname][variable]

        mask = df[argo_col].notna() & df[col].notna()
        sub = df[mask].copy()
        sub["bias"] = sub[col] - sub[argo_col]

        if len(sub) < 20:
            continue

        bias_lim = np.percentile(np.abs(sub["bias"].dropna()), 95)
        bias_lim = max(bias_lim, 0.05)
        bias_norm = mcolors.TwoSlopeNorm(vmin=-bias_lim, vcenter=0, vmax=bias_lim)

        fig = plt.figure(figsize=(14, 8))
        gs_s = gridspec.GridSpec(
            2, 2, hspace=0.25, wspace=0.10, left=0.04, right=0.96, top=0.90, bottom=0.05
        )

        for si, season in enumerate(SEASON_ORDER):
            ssub = sub[sub["season"] == season]
            ax = make_map_ax(fig, gs_s[si // 2, si % 2])

            if len(ssub) < 10:
                ax.set_title(
                    f"{season}  (n={len(ssub)}, insufficient data)", fontsize=10
                )
                continue

            ssub_b = ssub.copy()
            ssub_b["lon_bin"] = pd.cut(ssub_b["lon"], bins=lon_bins, labels=lon_c)
            ssub_b["lat_bin"] = pd.cut(ssub_b["lat"], bins=lat_bins, labels=lat_c)
            bg = ssub_b.groupby(["lat_bin", "lon_bin"], observed=True)["bias"]
            bm = bg.mean()
            bc = bg.count()

            grid = np.full((len(lat_c), len(lon_c)), np.nan)
            for (lv, lnv), val in bm.items():
                cnt = bc.get((lv, lnv), 0)
                if cnt < 3:
                    continue
                li = np.searchsorted(lat_c, float(lv))
                lj = np.searchsorted(lon_c, float(lnv))
                if 0 <= li < len(lat_c) and 0 <= lj < len(lon_c):
                    grid[li, lj] = val

            kw = {"transform": ccrs.PlateCarree()}
            pcm = ax.pcolormesh(
                lon_c, lat_c, grid, cmap="RdBu_r", norm=bias_norm, rasterized=True, **kw
            )

            season_metrics = compute_metrics(ssub[argo_col].values, ssub[col].values)
            m_str = ""
            if season_metrics:
                m_str = (
                    f"  Bias={season_metrics['bias']:+.3f}  "
                    f"RMSE={season_metrics['rmse']:.3f}"
                )
            ax.set_title(f"{season}  (n={len(ssub):,}){m_str}", fontsize=13)

        # Shared colorbar
        cbar_ax = fig.add_axes([0.15, 0.01, 0.70, 0.018])
        sm = plt.cm.ScalarMappable(cmap="RdBu_r", norm=bias_norm)
        sm.set_array([])
        cb = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal", extend="both")
        cb.set_label(f"Mean Bias ({unit})", fontsize=9)
        cb.ax.tick_params(labelsize=7)

        fig.suptitle(
            f"Seasonal {var_label} Bias — {spec['label']} vs. Argo (2°×2°)", fontsize=14
        )

        out_path = os.path.join(
            output_dir, f"fig3_seasonal_bias_{pname}_{variable}.pdf"
        )
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
        logging.info(f"✓ {out_path}")
        plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 4 — Density Scatter Comparison
# ═══════════════════════════════════════════════════════════════════════


def plot_scatter_comparison(df, variable, output_dir, dpi=300):
    argo_col = "temp" if variable == "sst" else "psal"
    unit = "°C" if variable == "sst" else "PSU"
    var_label = variable.upper()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    for ax, pname in zip(axes, ["glorys", "l4"]):
        col = f"{pname}_{variable}"
        if col not in df.columns:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center")
            continue
        spec = PRODUCT_SPECS[pname][variable]
        mask = df[argo_col].notna() & df[col].notna()
        obs, mod = df.loc[mask, argo_col].values, df.loc[mask, col].values
        if len(obs) < 3:
            continue

        h, xe, ye = np.histogram2d(obs, mod, bins=80)
        h[h == 0] = np.nan
        ax.pcolormesh(xe, ye, h.T, cmap="YlOrRd", rasterized=True, alpha=0.8)
        lo, hi = min(obs.min(), mod.min()), max(obs.max(), mod.max())
        mg = (hi - lo) * 0.03
        ax.plot([lo - mg, hi + mg], [lo - mg, hi + mg], "k--", linewidth=0.8)
        slope, intercept, r_val, _, _ = scipy_stats.linregress(obs, mod)
        xfit = np.linspace(lo, hi, 100)
        ax.plot(
            xfit,
            slope * xfit + intercept,
            color=spec["color"],
            linewidth=1.2,
            label=f"R²={r_val**2:.3f}",
        )
        m = compute_metrics(obs, mod)
        if m:
            ax.text(
                0.03,
                0.97,
                f"Bias={m['bias']:+.3f}\nRMSE={m['rmse']:.3f}\nn={m['n']:,}",
                transform=ax.transAxes,
                va="top",
                fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
            )
        ax.set_xlabel(f"Argo {var_label} ({unit})", fontsize=10)
        ax.set_ylabel(f"{spec['label']} {var_label} ({unit})", fontsize=10)
        ax.set_title(f"{spec['label']} vs. Argo", fontsize=12)
        ax.legend(fontsize=8, loc="lower right")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.2)

    fig.suptitle(f"{var_label} — Gridded Products vs. Argo", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out_path = os.path.join(output_dir, f"fig4_scatter_{variable}.pdf")
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    logging.info(f"✓ {out_path}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 5 — Taylor + Target Diagrams
# ═══════════════════════════════════════════════════════════════════════


class TaylorDiagram:
    def __init__(self, fig=None, rect=111, std_range=(0, 1.8), fig_kw=None):
        if fig is None:
            fig = plt.figure(**(fig_kw or {"figsize": (8, 7)}))
        self.fig = fig
        corr_ticks = np.array(
            [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1]
        )
        self.ax = fig.add_subplot(rect, projection="polar")
        self.ax.set_thetamin(0)
        self.ax.set_thetamax(90)
        self.ax.set_theta_direction(-1)
        self.ax.set_theta_offset(0)
        self.ax.set_thetagrids(
            np.degrees(np.arccos(corr_ticks)), labels=[f"{c}" for c in corr_ticks]
        )
        self.ax.set_rlabel_position(0)
        self.ax.set_rlim(std_range)
        self.ax.plot(0, 1.0, "ko", ms=8, label="Reference (Argo)")
        self._rmse_contours(std_range)

    def _rmse_contours(self, sr, n=5):
        th = np.linspace(0, np.pi / 2, 200)
        for rv in np.linspace(0.25, sr[1], n):
            r_v = []
            for t in th:
                d = 4 * np.cos(t) ** 2 - 4 * (1 - rv**2)
                r_v.append((2 * np.cos(t) + np.sqrt(d)) / 2 if d >= 0 else np.nan)
            r_a = np.array(r_v)
            ok = np.isfinite(r_a) & (r_a <= sr[1]) & (r_a >= 0)
            self.ax.plot(th[ok], r_a[ok], ":", color="gray", lw=0.6, alpha=0.5)

    def add_point(self, std_ratio, corr, **kw):
        self.ax.scatter(
            np.arccos(np.clip(corr, -1, 1)),
            std_ratio,
            edgecolors="black",
            linewidths=0.5,
            zorder=5,
            s=80,
            **kw,
        )


def plot_taylor_and_target(all_metrics, output_dir, dpi=300):
    # Taylor
    td = TaylorDiagram(fig_kw={"figsize": (8, 7)})
    for key, m in all_metrics.items():
        if m is None:
            continue
        td.add_point(
            m["std_ratio"],
            m["correlation"],
            label=f"{key} (R={m['correlation']:.3f})",
            marker=m["marker"],
            c=m["color"],
        )
    td.ax.legend(loc="upper left", fontsize=7, framealpha=0.9)
    td.ax.set_title("Taylor Diagram — Gridded Products vs. Argo", fontsize=13, pad=20)
    td.fig.savefig(
        os.path.join(output_dir, "fig5a_taylor.pdf"), dpi=dpi, bbox_inches="tight"
    )
    plt.close(td.fig)

    # Target
    fig, ax = plt.subplots(figsize=(7, 7))
    for key, m in all_metrics.items():
        if m is None:
            continue
        sign = 1.0 if m["std_model"] >= m["std_obs"] else -1.0
        ax.scatter(
            sign * m["norm_crmse"],
            m["norm_bias"],
            marker=m["marker"],
            c=m["color"],
            s=120,
            edgecolors="black",
            linewidths=0.5,
            zorder=5,
            label=f"{key} (RMSE={m['rmse']:.3f}, n={m['n']:,})",
        )
        ax.annotate(
            key,
            (sign * m["norm_crmse"], m["norm_bias"]),
            textcoords="offset points",
            xytext=(8, 8),
            fontsize=8,
        )
    for r in [0.25, 0.5, 0.75, 1.0, 1.25]:
        ax.add_patch(
            plt.Circle(
                (0, 0), r, fill=False, color="gray", linestyle="--", lw=0.5, alpha=0.5
            )
        )
    ax.axhline(0, color="k", lw=0.5)
    ax.axvline(0, color="k", lw=0.5)
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect("equal")
    ax.set_xlabel("Norm. CRMSE (sign = variability diff.)", fontsize=10)
    ax.set_ylabel("Norm. Mean Bias", fontsize=10)
    ax.set_title("Target Diagram — Gridded Products vs. Argo", fontsize=13)
    ax.legend(loc="upper left", fontsize=7)
    ax.grid(True, alpha=0.3)
    fig.savefig(
        os.path.join(output_dir, "fig5b_target.pdf"), dpi=dpi, bbox_inches="tight"
    )
    logging.info("✓ Taylor + Target diagrams saved")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 6 — Per-Region Box Plots
# ═══════════════════════════════════════════════════════════════════════


def plot_regional_boxplots(df, variable, output_dir, dpi=300):
    argo_col = "temp" if variable == "sst" else "psal"
    unit = "°C" if variable == "sst" else "PSU"
    var_label = variable.upper()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, pname in zip(axes, ["glorys", "l4"]):
        col = f"{pname}_{variable}"
        if col not in df.columns:
            continue
        spec = PRODUCT_SPECS[pname][variable]
        mask = df[argo_col].notna() & df[col].notna()
        sub = df[mask].copy()
        sub["bias"] = sub[col] - sub[argo_col]

        rdata, rlabels = [], []
        for rn in SPATIAL_REGIONS:
            rd = sub[sub["region"] == rn]["bias"].dropna()
            if len(rd) >= 3:
                rdata.append(rd.values)
                rlabels.append(REGION_SHORT.get(rn, rn))
        if not rdata:
            continue

        bp = ax.boxplot(
            rdata,
            labels=rlabels,
            patch_artist=True,
            widths=0.6,
            flierprops=dict(marker=".", ms=2, alpha=0.3),
        )
        for p in bp["boxes"]:
            p.set_facecolor(spec["color"])
            p.set_alpha(0.4)
        ax.axhline(0, color="k", lw=0.5, ls="--")
        ax.set_title(f"{spec['label']}", fontsize=12)
        ax.set_ylabel(f"Bias ({unit})", fontsize=9)
        ax.tick_params(axis="x", labelsize=7, rotation=45)
        ax.grid(True, alpha=0.2, axis="y")

    fig.suptitle(f"{var_label} Bias by Region", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out_path = os.path.join(output_dir, f"fig6_regional_boxplot_{variable}.pdf")
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    logging.info(f"✓ {out_path}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════
# Main Pipeline
# ═══════════════════════════════════════════════════════════════════════


def _prefetch_argo_cache(hf_dataset, date_list, regions, cache_dir, max_workers=8):
    """Download all needed files into a local cache with formatted-region naming.

    Saves files as:
      {cache_dir}/argo/argo_{region}_{date}.nc
      {cache_dir}/glorys/glorys_{region}_{date}.nc
      {cache_dir}/l4_sst/l4_sst_{region}_{date}.nc
      {cache_dir}/l4_sss/l4_sss_{region}_{date}.nc
    """
    cache = Path(cache_dir)

    # catalog_filename -> (local_subdir, local_fname_template)
    products = [
        ("argo", "argo", "argo_{region}_{date}.nc"),
        ("glorys", "glorys", "glorys_{region}_{date}.nc"),
        ("l4_sst", "l4_sst", "l4_sst_{region}_{date}.nc"),
        ("l4_sss", "l4_sss", "l4_sss_{region}_{date}.nc"),
    ]

    def _fetch(date_str, region, catalog_fname, local_subdir, local_fname_tmpl):
        dest = cache / local_subdir / local_fname_tmpl.format(region=region, date=date_str)
        if dest.exists():
            return
        date_dash = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
        ds = load_region_product_nc(hf_dataset, date_dash, region, catalog_fname)
        if ds is not None:
            dest.parent.mkdir(parents=True, exist_ok=True)
            ds.to_netcdf(dest)

    tasks = [
        (date_str, region, cat_fn, sub, tmpl)
        for date_str in date_list
        for region in regions
        for cat_fn, sub, tmpl in products
    ]
    logging.info(f"  Pre-downloading {len(tasks)} files to {cache_dir} ...")
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futs = [pool.submit(_fetch, *t) for t in tasks]
        for f in futs:
            try:
                f.result()
            except Exception as e:
                logging.warning(f"  Prefetch warning: {e}")
    logging.info("  Pre-download complete.")


def main():
    parser = argparse.ArgumentParser(
        description="Validate GLORYS and L4 SST/SSS against Argo floats.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--taco-dir", default=None, help="Local TACO data directory.")
    parser.add_argument(
        "--hf-url",
        default=HF_DEFAULT_URL,
        help="HuggingFace dataset URL (used when --taco-dir is not set).",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Local cache directory for remote downloads (required with --hf-url).",
    )
    parser.add_argument("--date-min", default="2023-04-01")
    parser.add_argument("--date-max", default="2025-08-02")
    parser.add_argument("--pressure-max", type=float, default=5.0)
    parser.add_argument("--tolerance-deg", type=float, default=0.5)
    parser.add_argument(
        "--extract-workers",
        type=int,
        default=8,
        help="Number of worker processes for extraction (1=serial).",
    )
    parser.add_argument("--output-dir", default="./validation_results")
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    date_list = generate_date_list(args.date_min, args.date_max)
    logging.info(f"Period: {args.date_min} -> {args.date_max} ({len(date_list)} days)")

    # Resolve taco_dir: local path or download to cache
    if args.taco_dir:
        taco_dir = args.taco_dir
    else:
        if not args.cache_dir:
            raise ValueError("Provide --cache-dir when using --hf-url remote access.")
        hf_dataset = load_hf_dataset(args.hf_url)
        _prefetch_argo_cache(
            hf_dataset,
            date_list,
            list(SPATIAL_REGIONS.keys()),
            args.cache_dir,
            max_workers=args.extract_workers,
        )
        taco_dir = args.cache_dir

    # ── Step 1: Collect Argo ─────────────────────────────────────────
    all_argo = []
    for i, date_str in enumerate(date_list):
        df = load_argo_surface(taco_dir, date_str, args.pressure_max)
        if not df.empty:
            all_argo.append(df)
        if (i + 1) % 30 == 0:
            logging.info(
                f"  Processed {i + 1}/{len(date_list)} days, "
                f"{sum(len(d) for d in all_argo)} obs so far"
            )

    if not all_argo:
        logging.error("No Argo observations found.")
        sys.exit(1)

    argo_df = pd.concat(all_argo, ignore_index=True)
    logging.info(f"Total Argo surface obs: {len(argo_df):,}")
    logging.info(
        f"  SST: {argo_df['temp'].notna().sum():,}   "
        f"SSS: {argo_df['psal'].notna().sum():,}"
    )
    argo_df.to_parquet(output_dir / "argo_surface_obs.parquet", index=False)

    # ── Step 2: Match to gridded products ────────────────────────────
    for pname in ["glorys", "l4"]:
        for variable in ["sst"]:  # "sss"
            col = f"{pname}_{variable}"
            logging.info(f"  Extracting {pname} {variable} …")
            argo_df[col] = extract_product_at_argo(
                taco_dir,
                argo_df,
                pname,
                variable,
                args.tolerance_deg,
                n_workers=args.extract_workers,
            )
            logging.info(f"    → {argo_df[col].notna().sum():,} matched")

    argo_df.to_parquet(output_dir / "matched_all.parquet", index=False)

    # ── Step 3: Compute & print global metrics ───────────────────────
    # all_taylor = {}
    # for variable in ["sst", "sss"]:
    #     argo_col = "temp" if variable == "sst" else "psal"
    #     unit = "°C" if variable == "sst" else "PSU"
    #     vl = variable.upper()
    #     print(f"\n{'═' * 65}")
    #     print(f"  {vl} VALIDATION METRICS (full period)")
    #     print(f"{'═' * 65}")
    #     for pname in ["glorys", "l4"]:
    #         col = f"{pname}_{variable}"
    #         spec = PRODUCT_SPECS[pname][variable]
    #         mask = argo_df[argo_col].notna() & argo_df[col].notna()
    #         m = compute_metrics(argo_df.loc[mask, argo_col].values,
    #                             argo_df.loc[mask, col].values)
    #         if m is None:
    #             print(f"  {spec['label']:12s} {vl}:  insufficient data")
    #             continue
    #         key = f"{spec['label']} {vl}"
    #         m["color"] = spec["color"]; m["marker"] = spec["marker"]
    #         all_taylor[key] = m
    #         print(f"  {spec['label']:12s} {vl}:  "
    #               f"Bias={m['bias']:+.4f} {unit}  "
    #               f"RMSE={m['rmse']:.4f} {unit}  "
    #               f"R={m['correlation']:.4f}  n={m['n']:,}")

    # ── Step 4: Generate all figures ─────────────────────────────────
    for variable in ["sst", "sss"]:
        logging.info(f"\nGenerating {variable.upper()} figures …")
        # plot_aggregated_bias_with_hovmoller(argo_df, variable, str(output_dir), dpi=args.dpi)
        plot_regional_timeseries(argo_df, variable, str(output_dir), dpi=args.dpi)
        # plot_seasonal_bias(argo_df, variable, str(output_dir), dpi=args.dpi)
        # plot_scatter_comparison(argo_df, variable, str(output_dir), dpi=args.dpi)
        # plot_regional_boxplots(argo_df, variable, str(output_dir), dpi=args.dpi)

    # if any(v is not None for v in all_taylor.values()):
    #     plot_taylor_and_target(all_taylor, str(output_dir), dpi=args.dpi)

    # ── Summary ──────────────────────────────────────────────────────
    print(f"\n{'═' * 65}")
    print(f"  OUTPUT FILES ({output_dir})")
    print(f"{'═' * 65}")
    for f in sorted(output_dir.glob("*")):
        print(f"  {f.name:50s}  {f.stat().st_size / 1024:8.1f} KB")
    print(f"{'═' * 65}")


if __name__ == "__main__":
    main()
