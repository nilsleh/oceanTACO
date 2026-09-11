#!/usr/bin/env python3
"""Plot SSH comparison figures."""

import argparse
import logging
import os
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

ORDERED_REGIONS = [
    "NORTH_PACIFIC_WEST",
    "NORTH_ATLANTIC",
    "NORTH_INDIAN",
    "NORTH_PACIFIC_EAST",
    "SOUTH_PACIFIC_WEST",
    "SOUTH_ATLANTIC",
    "SOUTH_INDIAN",
    "SOUTH_PACIFIC_EAST",
]

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

OBS_PRODUCTS = {
    "swot": {
        "subdir": "l3_swot",
        "filename": "l3_swot_{region}_{date}.nc",
        "catalog_id": "l3_swot",
        "var_candidates": ["ssha_filtered", "ssha_unfiltered"],
        "label": "SWOT L3",
        "color": "#e74c3c",
        "marker": "D",
        "res_km": 2.0,
    },
    "l3_ssh": {
        "subdir": "l3_ssh",
        "filename": "l3_ssh_{region}_{date}.nc",
        "catalog_id": "l3_ssh",
        "var_candidates": ["sla_filtered", "sla"],
        "label": "L3 Altimetry",
        "color": "#2ecc71",
        "marker": "^",
        "res_km": 7.0,
    },
}

REF_PRODUCTS = {
    "glorys": {
        "subdir": "glorys/{region}",
        "filename": "glorys_ssh_{region}_{date}.nc",
        "catalog_id": "glorys",
        "var_candidates": ["zos", "ssh"],
        "label": "GLORYS12",
        "color": "#3498db",
        "marker": "o",
    },
    "l4_ssh": {
        "subdir": "l4_ssh",
        "filename": "l4_ssh_{region}_{date}.nc",
        "catalog_id": "l4_ssh",
        "var_candidates": ["adt", "sla", "ssh"],
        "label": "DUACS L4",
        "color": "#9b59b6",
        "marker": "s",
    },
}

# Fixed spatial grid for accumulation
BIN_DEG = 2
LON_EDGES = np.arange(-180, 180 + BIN_DEG, BIN_DEG)
LAT_EDGES = np.arange(-90, 90 + BIN_DEG, BIN_DEG)
LON_C = (LON_EDGES[:-1] + LON_EDGES[1:]) / 2
LAT_C = (LAT_EDGES[:-1] + LAT_EDGES[1:]) / 2
N_LON_BINS = len(LON_C)
N_LAT_BINS = len(LAT_C)

# Latitude bands for profiles
LAT_BAND_EDGES = np.arange(-90, 92, 5)
LAT_BAND_MIDS = (LAT_BAND_EDGES[:-1] + LAT_BAND_EDGES[1:]) / 2
N_LAT_BANDS = len(LAT_BAND_MIDS)


# ═══════════════════════════════════════════════════════════════════════
# Accumulator — the key to performance
#
# Instead of building million-row DataFrames, we maintain fixed-size
# numpy arrays and accumulate running sums.  At the end we divide to
# get means, compute RMSE from sum-of-squares, etc.
# ═══════════════════════════════════════════════════════════════════════


class BinnedAccumulator:
    """Accumulates (diff, obs, ref) statistics into:
      - 2D spatial bins (lat × lon) — for maps
      - 1D lat-band bins × monthly time bins — for Hovmöller
      - per-region × monthly — for time-series plots
      - global running sums — for Taylor/Target

    All updates are vectorised numpy operations on the valid-pixel arrays.
    """

    def __init__(self, month_list):
        """Parameters
        ----------
        month_list : list of str
            Pre-computed list of YYYY-MM strings covering the full period.
        """
        self.month_list = month_list
        self.month_to_idx = {m: i for i, m in enumerate(month_list)}
        n_months = len(month_list)

        # ── Spatial bins (full period) ──
        self.spatial_sum = np.zeros((N_LAT_BINS, N_LON_BINS), dtype=np.float64)
        self.spatial_sq = np.zeros((N_LAT_BINS, N_LON_BINS), dtype=np.float64)
        self.spatial_cnt = np.zeros((N_LAT_BINS, N_LON_BINS), dtype=np.int64)

        # ── Seasonal spatial bins ──
        self.season_sum = {
            s: np.zeros((N_LAT_BINS, N_LON_BINS), dtype=np.float64)
            for s in SEASON_ORDER
        }
        self.season_sq = {
            s: np.zeros((N_LAT_BINS, N_LON_BINS), dtype=np.float64)
            for s in SEASON_ORDER
        }
        self.season_cnt = {
            s: np.zeros((N_LAT_BINS, N_LON_BINS), dtype=np.int64) for s in SEASON_ORDER
        }

        # ── Hovmöller (lat-band × month) ──
        self.hov_sum = np.zeros((N_LAT_BANDS, n_months), dtype=np.float64)
        self.hov_sq = np.zeros((N_LAT_BANDS, n_months), dtype=np.float64)
        self.hov_cnt = np.zeros((N_LAT_BANDS, n_months), dtype=np.int64)

        # ── Per-region × month (for time series) ──
        self.region_sum = {
            r: np.zeros(n_months, dtype=np.float64) for r in SPATIAL_REGIONS
        }
        self.region_sq = {
            r: np.zeros(n_months, dtype=np.float64) for r in SPATIAL_REGIONS
        }
        self.region_cnt = {
            r: np.zeros(n_months, dtype=np.int64) for r in SPATIAL_REGIONS
        }

        # ── Global running sums for overall metrics ──
        self.global_obs_sum = 0.0
        self.global_obs_sq = 0.0
        self.global_ref_sum = 0.0
        self.global_ref_sq = 0.0
        self.global_diff_sum = 0.0
        self.global_diff_sq = 0.0
        self.global_cross = 0.0  # sum(obs * ref) for correlation
        self.global_n = 0

    def update(self, lons, lats, obs_vals, ref_vals, diffs, region, date_str):
        """Ingest one comparison result (arrays of valid pixels).
        All inputs are 1-D numpy arrays of the same length.
        """
        n = len(diffs)
        if n == 0:
            return

        month_str = f"{date_str[:4]}-{date_str[4:6]}"
        mi = self.month_to_idx.get(month_str)
        month_num = int(date_str[4:6])
        season = SEASON_MAP.get(month_num)

        # ── Spatial bins ──
        li = np.searchsorted(LAT_EDGES, lats, side="right") - 1
        lj = np.searchsorted(LON_EDGES, lons, side="right") - 1
        # Clip to valid range
        li = np.clip(li, 0, N_LAT_BINS - 1)
        lj = np.clip(lj, 0, N_LON_BINS - 1)

        # Use np.add.at for unbuffered accumulation (handles duplicate indices)
        np.add.at(self.spatial_sum, (li, lj), diffs)
        np.add.at(self.spatial_sq, (li, lj), diffs**2)
        np.add.at(self.spatial_cnt, (li, lj), 1)

        if season:
            np.add.at(self.season_sum[season], (li, lj), diffs)
            np.add.at(self.season_sq[season], (li, lj), diffs**2)
            np.add.at(self.season_cnt[season], (li, lj), 1)

        # ── Hovmöller ──
        if mi is not None:
            lb = np.searchsorted(LAT_BAND_EDGES, lats, side="right") - 1
            lb = np.clip(lb, 0, N_LAT_BANDS - 1)
            np.add.at(self.hov_sum, (lb, mi), diffs)
            np.add.at(self.hov_sq, (lb, mi), diffs**2)
            np.add.at(self.hov_cnt, (lb, mi), 1)

        # ── Region × month ──
        if mi is not None and region in self.region_sum:
            self.region_sum[region][mi] += diffs.sum()
            self.region_sq[region][mi] += (diffs**2).sum()
            self.region_cnt[region][mi] += n

        # ── Global ──
        self.global_obs_sum += obs_vals.sum()
        self.global_obs_sq += (obs_vals**2).sum()
        self.global_ref_sum += ref_vals.sum()
        self.global_ref_sq += (ref_vals**2).sum()
        self.global_diff_sum += diffs.sum()
        self.global_diff_sq += (diffs**2).sum()
        self.global_cross += (obs_vals * ref_vals).sum()
        self.global_n += n

    # ── Derived quantities ──

    def spatial_mean_bias(self, min_count=5):
        with np.errstate(divide="ignore", invalid="ignore"):
            m = self.spatial_sum / np.maximum(self.spatial_cnt, 1)
        m[self.spatial_cnt < min_count] = np.nan
        return m

    def spatial_rmse(self, min_count=5):
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.sqrt(self.spatial_sq / np.maximum(self.spatial_cnt, 1))
        r[self.spatial_cnt < min_count] = np.nan
        return r

    def season_mean_bias(self, season, min_count=5):
        with np.errstate(divide="ignore", invalid="ignore"):
            m = self.season_sum[season] / np.maximum(self.season_cnt[season], 1)
        m[self.season_cnt[season] < min_count] = np.nan
        return m

    def hovmoller_mean(self, min_count=5):
        with np.errstate(divide="ignore", invalid="ignore"):
            m = self.hov_sum / np.maximum(self.hov_cnt, 1)
        m[self.hov_cnt < min_count] = np.nan
        return m

    def region_monthly_stats(self, region, min_count=5):
        """Returns (months, bias, rmse, count) arrays."""
        cnt = self.region_cnt[region]
        with np.errstate(divide="ignore", invalid="ignore"):
            bias = self.region_sum[region] / np.maximum(cnt, 1)
            rmse = np.sqrt(self.region_sq[region] / np.maximum(cnt, 1))
        mask = cnt >= min_count
        return mask, bias, rmse, cnt

    def global_metrics(self):
        """Compute global bias, RMSE, correlation, std_ratio."""
        n = self.global_n
        if n < 10:
            return None
        mean_o = self.global_obs_sum / n
        mean_r = self.global_ref_sum / n
        var_o = self.global_obs_sq / n - mean_o**2
        var_r = self.global_ref_sq / n - mean_r**2
        std_o = np.sqrt(max(var_o, 0))
        std_r = np.sqrt(max(var_r, 0))
        bias = self.global_diff_sum / n
        rmse = np.sqrt(self.global_diff_sq / n)
        # Pearson: r = (E[XY] - E[X]E[Y]) / (σ_X σ_Y)
        cov = self.global_cross / n - mean_o * mean_r
        corr = cov / (std_o * std_r) if (std_o > 0 and std_r > 0) else 0.0
        crmse = np.sqrt(max(rmse**2 - bias**2, 0))
        return {
            "n": n,
            "bias": float(bias),
            "rmse": float(rmse),
            "crmse": float(crmse),
            "correlation": float(np.clip(corr, -1, 1)),
            "std_obs": float(std_o),
            "std_model": float(std_r),
            "std_ratio": float(std_r / std_o) if std_o > 0 else np.nan,
            "norm_bias": float(bias / std_o) if std_o > 0 else np.nan,
            "norm_crmse": float(crmse / std_o) if std_o > 0 else np.nan,
        }

    def lat_profile(self, min_count=5):
        """Full-period mean bias and std per latitude band."""
        cnt = self.hov_cnt.sum(axis=1)  # sum over all months
        s = self.hov_sum.sum(axis=1)
        sq = self.hov_sq.sum(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            mean = s / np.maximum(cnt, 1)
            var = sq / np.maximum(cnt, 1) - mean**2
        std = np.sqrt(np.maximum(var, 0))
        mask = cnt >= min_count
        return LAT_BAND_MIDS, mean, std, cnt, mask


# ═══════════════════════════════════════════════════════════════════════
# File I/O (cached per day)
# ═══════════════════════════════════════════════════════════════════════


def generate_date_list(date_min, date_max):
    start = datetime.strptime(date_min, "%Y-%m-%d")
    end = datetime.strptime(date_max, "%Y-%m-%d")
    dates, cur = [], start
    while cur <= end:
        dates.append(cur.strftime("%Y%m%d"))
        cur += timedelta(days=1)
    return dates


def generate_month_list(date_min, date_max):
    """Generate list of YYYY-MM strings covering the period."""
    start = datetime.strptime(date_min, "%Y-%m-%d").replace(day=1)
    end = datetime.strptime(date_max, "%Y-%m-%d").replace(day=1)
    months = []
    cur = start
    while cur <= end:
        months.append(cur.strftime("%Y-%m"))
        if cur.month == 12:
            cur = cur.replace(year=cur.year + 1, month=1)
        else:
            cur = cur.replace(month=cur.month + 1)
    return months


class DayFileCache:
    """Cache open datasets for a single day. Closes all on flush.

    Supports both local file access (taco_dir) and remote HuggingFace access
    (dataset_hf + cache_dir).
    """

    def __init__(self, dataset_hf=None, cache_dir=None):
        self._cache = {}
        self._dataset_hf = dataset_hf
        self._cache_dir = cache_dir

    def get(self, taco_dir, spec, region, date_str):
        """Returns (ds, var_name) or (None, None). Cached per (catalog_id, region, date)."""
        if self._dataset_hf is not None:
            cache_key = (spec.get("catalog_id", spec["filename"]), region, date_str)
            if cache_key in self._cache:
                return self._cache[cache_key]

            date_dash = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
            ds = load_region_product_nc(
                self._dataset_hf, date_dash, region,
                spec["catalog_id"], self._cache_dir
            )
            if ds is None:
                self._cache[cache_key] = (None, None)
                return None, None
        else:
            subdir = spec["subdir"].format(region=region)
            fname = spec["filename"].format(region=region, date=date_str)
            fpath = os.path.join(taco_dir, subdir, fname)
            cache_key = fpath

            if cache_key in self._cache:
                return self._cache[cache_key]

            if not os.path.exists(fpath):
                self._cache[cache_key] = (None, None)
                return None, None

            try:
                ds = xr.open_dataset(fpath)
            except Exception:
                self._cache[cache_key] = (None, None)
                return None, None

        var_name = None
        for vc in spec["var_candidates"]:
            if vc in ds.data_vars:
                var_name = vc
                break
        if var_name is None:
            for v in ds.data_vars:
                if any(k in v.lower() for k in ("ssh", "sla", "adt", "zos", "ssha")):
                    var_name = v
                    break
        if var_name is None:
            ds.close()
            self._cache[cache_key] = (None, None)
            return None, None

        self._cache[cache_key] = (ds, var_name)
        return ds, var_name

    def flush(self):
        for ds, _ in self._cache.values():
            if ds is not None:
                ds.close()
        self._cache.clear()


# ═══════════════════════════════════════════════════════════════════════
# Core comparison — vectorised, no per-pixel DataFrame
# ═══════════════════════════════════════════════════════════════════════


def compare_and_accumulate(
    cache, taco_dir, obs_key, ref_key, region, date_str, accum, subsample=1
):
    """Compare obs vs ref for one day/region.  Feeds results directly
    into the BinnedAccumulator — no intermediate DataFrame.

    Returns the number of valid comparison pixels (for logging).
    """
    obs_spec = OBS_PRODUCTS[obs_key]
    ref_spec = REF_PRODUCTS[ref_key]

    obs_ds, obs_var = cache.get(taco_dir, obs_spec, region, date_str)
    if obs_ds is None:
        return 0
    ref_ds, ref_var = cache.get(taco_dir, ref_spec, region, date_str)
    if ref_ds is None:
        return 0

    # ── Get observation field ────────────────────────────────────────
    obs_data = obs_ds[obs_var].squeeze().values.astype(np.float64)
    obs_lon = obs_ds["lon"].values
    obs_lat = obs_ds["lat"].values

    obs_valid = np.isfinite(obs_data)
    if "n_obs" in obs_ds:
        obs_valid = obs_valid & (obs_ds["n_obs"].values > 0)
    if obs_valid.sum() < 10:
        return 0

    # ── Interpolate reference to obs grid (vectorised) ───────────────
    # Use xarray.interp for clean, fast bilinear interpolation.
    ref_da = ref_ds[ref_var].squeeze()

    try:
        ref_interp_da = ref_da.interp(
            {
                "lat": xr.DataArray(obs_lat, dims="lat_obs"),
                "lon": xr.DataArray(obs_lon, dims="lon_obs"),
            },
            method="linear",
        )
        ref_interp = ref_interp_da.values.astype(np.float64)
    except Exception:
        return 0

    if ref_interp.shape != obs_data.shape:
        return 0

    # ── Mask and anomaly alignment ───────────────────────────────────
    both_valid = obs_valid & np.isfinite(ref_interp)
    n_valid = both_valid.sum()
    if n_valid < 10:
        return 0

    # Subtract co-located spatial means (removes reference surface diff)
    obs_mean = obs_data[both_valid].mean()
    ref_mean = ref_interp[both_valid].mean()
    obs_anom = obs_data - obs_mean
    ref_anom = ref_interp - ref_mean
    diff = ref_anom - obs_anom

    # ── Extract valid pixels ─────────────────────────────────────────
    if obs_lon.ndim == 1 and obs_lat.ndim == 1:
        lat_grid, lon_grid = np.meshgrid(obs_lat, obs_lon, indexing="ij")
    else:
        lat_grid, lon_grid = obs_lat, obs_lon

    lats_v = lat_grid[both_valid]
    lons_v = lon_grid[both_valid]
    obs_v = obs_anom[both_valid]
    ref_v = ref_anom[both_valid]
    diff_v = diff[both_valid]

    # ── Subsample (AFTER masking — no wasted computation) ────────────
    if subsample > 1 and len(diff_v) > subsample * 100:
        idx = np.arange(0, len(diff_v), subsample)
        lats_v, lons_v = lats_v[idx], lons_v[idx]
        obs_v, ref_v, diff_v = obs_v[idx], ref_v[idx], diff_v[idx]

    # ── Feed into accumulator ────────────────────────────────────────
    accum.update(lons_v, lats_v, obs_v, ref_v, diff_v, region, date_str)

    return len(diff_v)


# ═══════════════════════════════════════════════════════════════════════
# Spectral analysis (SWOT only — kept separate for full resolution)
# ═══════════════════════════════════════════════════════════════════════


def compute_difference_spectrum(cache, taco_dir, ref_key, region, date_str):
    """Compute zonal PSD of obs, ref-interpolated, and difference fields."""
    obs_spec = OBS_PRODUCTS["swot"]
    ref_spec = REF_PRODUCTS[ref_key]

    obs_ds, obs_var = cache.get(taco_dir, obs_spec, region, date_str)
    if obs_ds is None:
        return None
    ref_ds, ref_var = cache.get(taco_dir, ref_spec, region, date_str)
    if ref_ds is None:
        return None

    obs_data = obs_ds[obs_var].squeeze().values.astype(np.float64)
    obs_lon = obs_ds["lon"].values
    obs_lat = obs_ds["lat"].values

    obs_valid = np.isfinite(obs_data)
    if "n_obs" in obs_ds:
        obs_valid = obs_valid & (obs_ds["n_obs"].values > 0)
    if obs_valid.sum() < 100:
        return None

    ref_da = ref_ds[ref_var].squeeze()
    try:
        ref_interp = ref_da.interp(
            {
                "lat": xr.DataArray(obs_lat, dims="y"),
                "lon": xr.DataArray(obs_lon, dims="x"),
            },
            method="linear",
        ).values.astype(np.float64)
    except Exception:
        return None

    both = obs_valid & np.isfinite(ref_interp)
    if both.sum() < 100:
        return None

    obs_anom = obs_data - obs_data[both].mean()
    ref_anom = ref_interp - ref_interp[both].mean()
    diff_field = ref_anom - obs_anom

    # Zonal spectra
    mid_lat = np.mean(obs_lat)
    dx_deg = (
        float(np.abs(np.median(np.diff(obs_lon))))
        if obs_lon.ndim == 1
        else float(np.abs(np.median(np.diff(obs_lon, axis=1))))
    )
    dx_km = dx_deg * 111.32 * np.cos(np.radians(mid_lat))
    if dx_km <= 0:
        return None

    n_lon = obs_data.shape[1]

    def _row_spectra(field):
        specs = []
        for i in range(field.shape[0]):
            row = field[i, :]
            valid = np.isfinite(row)
            if valid.sum() < n_lon * 0.5:
                continue
            if not valid.all():
                x = np.arange(n_lon)
                row = np.interp(x, x[valid], row[valid])
            row -= np.polyval(np.polyfit(np.arange(n_lon), row, 1), np.arange(n_lon))
            fft_v = np.fft.rfft(row * np.hanning(n_lon))
            specs.append((2.0 / (n_lon * dx_km)) * np.abs(fft_v) ** 2)
        return np.mean(specs, axis=0) if specs else None

    psd_obs = _row_spectra(obs_anom)
    if psd_obs is None:
        return None
    psd_ref = _row_spectra(ref_anom)
    psd_diff = _row_spectra(diff_field)

    freqs = np.fft.rfftfreq(n_lon, d=dx_km)[1:]
    with np.errstate(divide="ignore"):
        wl = 1.0 / freqs

    return {
        "wavelength_km": wl,
        "psd_obs": psd_obs[1:],
        "psd_ref": psd_ref[1:] if psd_ref is not None else None,
        "psd_diff": psd_diff[1:] if psd_diff is not None else None,
    }


# ═══════════════════════════════════════════════════════════════════════
# Plotting Helpers
# ═══════════════════════════════════════════════════════════════════════


def make_map_ax(fig, subplot_spec, projection=None):
    proj = projection or ccrs.Robinson()
    ax = fig.add_subplot(subplot_spec, projection=proj)
    ax.set_global()
    ax.add_feature(cfeature.LAND, facecolor="#f0ece3", edgecolor="none")
    ax.add_feature(cfeature.COASTLINE, linewidth=0.3, color="#888888")
    ax.gridlines(linewidth=0.15, alpha=0.3)
    return ax


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 1 — Aggregated Spatial Diff + Hovmöller + Lat Profile
# ═══════════════════════════════════════════════════════════════════════


def plot_fig1(accum, obs_key, ref_key, month_list, output_dir, dpi=300):
    obs_spec, ref_spec = OBS_PRODUCTS[obs_key], REF_PRODUCTS[ref_key]
    gm = accum.global_metrics()

    bias_map = accum.spatial_mean_bias()
    bias_lim = np.nanpercentile(np.abs(bias_map[np.isfinite(bias_map)]), 95)
    bias_lim = max(bias_lim, 0.005)
    bnorm = mcolors.TwoSlopeNorm(vmin=-bias_lim, vcenter=0, vmax=bias_lim)

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

    # Top: spatial map
    ax_map = make_map_ax(fig, gridspec.GridSpecFromSubplotSpec(1, 1, gs[0, :])[0])
    kw = {"transform": ccrs.PlateCarree()}
    pcm = ax_map.pcolormesh(
        LON_C, LAT_C, bias_map, cmap="RdBu_r", norm=bnorm, rasterized=True, **kw
    )
    cb = fig.colorbar(
        pcm, ax=ax_map, shrink=0.65, pad=0.02, orientation="horizontal", extend="both"
    )
    cb.set_label("Mean Difference (m)", fontsize=9)
    ms = ""
    if gm:
        ms = (
            f"Bias={gm['bias']:+.5f} m   RMSE={gm['rmse']:.5f} m   "
            f"R={gm['correlation']:.3f}   n={gm['n']:,}"
        )
    ax_map.set_title(
        f"{ref_spec['label']} − {obs_spec['label']} SSH — "
        f"Period Aggregate (2°×2°)\n{ms}",
        fontsize=11,
        fontweight="bold",
    )

    # Bot-left: Hovmöller
    ax_hov = fig.add_subplot(gs[1, 0])
    hov = accum.hovmoller_mean()
    if len(month_list) >= 2:
        ax_hov.pcolormesh(
            np.arange(len(month_list) + 1) - 0.5,
            LAT_BAND_EDGES,
            hov,
            cmap="RdBu_r",
            norm=bnorm,
            rasterized=True,
        )
        skip = max(1, len(month_list) // 12)
        ax_hov.set_xticks(np.arange(len(month_list)))
        ax_hov.set_xticklabels(
            [month_list[i] if i % skip == 0 else "" for i in range(len(month_list))],
            rotation=45,
            ha="right",
            fontsize=7,
        )
    ax_hov.set_ylabel("Latitude (°)", fontsize=9)
    ax_hov.set_xlabel("Month", fontsize=9)
    ax_hov.set_title(
        "Hovmöller — SSH Diff (5° lat × month)", fontsize=10, fontweight="bold"
    )

    # Bot-right: latitude profile
    ax_lat = fig.add_subplot(gs[1, 1])
    lat_mids, mean, std, cnt, mask = accum.lat_profile()
    ax_lat.fill_betweenx(
        lat_mids[mask],
        mean[mask] - std[mask],
        mean[mask] + std[mask],
        alpha=0.15,
        color=ref_spec["color"],
    )
    ax_lat.plot(
        mean[mask],
        lat_mids[mask],
        color=ref_spec["color"],
        lw=2,
        label=f"Mean (n={accum.global_n:,})",
    )
    # Monthly ghost lines
    for mi in range(len(month_list)):
        col_cnt = accum.hov_cnt[:, mi]
        col_s = accum.hov_sum[:, mi]
        with np.errstate(divide="ignore", invalid="ignore"):
            col_m = col_s / np.maximum(col_cnt, 1)
        ok = col_cnt >= 5
        if ok.sum() >= 2:
            ax_lat.plot(
                col_m[ok], lat_mids[ok], color=ref_spec["color"], alpha=0.08, lw=0.5
            )
    ax_lat.axvline(0, color="k", lw=0.5, ls="--")
    ax_lat.set_xlabel("Mean Diff (m)", fontsize=9)
    ax_lat.set_ylabel("Latitude (°)", fontsize=9)
    ax_lat.set_ylim(-90, 90)
    ax_lat.legend(fontsize=8)
    ax_lat.grid(True, alpha=0.2)
    ax_lat.set_title("Latitude Profile ±1σ", fontsize=10, fontweight="bold")

    fig.suptitle(
        f"SSH Difference — {ref_spec['label']} vs. {obs_spec['label']}",
        fontsize=13,
        fontweight="bold",
        y=0.97,
    )
    tag = f"{ref_key}_vs_{obs_key}"
    fig.savefig(
        os.path.join(output_dir, f"fig1_spatial_diff_{tag}.pdf"),
        dpi=dpi,
        bbox_inches="tight",
        facecolor="white",
    )
    logging.info(f"✓ fig1_spatial_diff_{tag}.pdf")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 2 — Regional Time Series (matching user's style)
# ═══════════════════════════════════════════════════════════════════════


def plot_fig2(accumulators, obs_key, month_list, output_dir, dpi=300):
    """8-panel figure: one per region, both ref products overlaid.
    Matches the user's requested style with shared legend at bottom.
    """
    obs_spec = OBS_PRODUCTS[obs_key]
    month_timestamps = pd.to_datetime(month_list, format="%Y-%m")

    fig, axes = plt.subplots(2, 4, figsize=(4.6 * 4, 3.6 * 2), sharex=True)
    axes = np.atleast_2d(axes)

    legend_handles = []
    for ref_key, ref_spec in REF_PRODUCTS.items():
        legend_handles.extend(
            [
                Line2D(
                    [0],
                    [0],
                    color=ref_spec["color"],
                    lw=2,
                    ls="-",
                    label=f"{ref_spec['label']} Bias",
                ),
                Line2D(
                    [0],
                    [0],
                    color=ref_spec["color"],
                    lw=2,
                    ls="--",
                    label=f"{ref_spec['label']} RMSE",
                ),
            ]
        )

    for idx, rname in enumerate(ORDERED_REGIONS):
        ri, ci = divmod(idx, 4)
        ax = axes[ri, ci]

        has_data = False
        for ref_key, accum in accumulators.items():
            ref_spec = REF_PRODUCTS[ref_key]
            mask, bias, rmse, cnt = accum.region_monthly_stats(rname)
            if not mask.any():
                continue
            has_data = True
            x = month_timestamps[mask]
            ax.plot(x, bias[mask], color=ref_spec["color"], lw=1.2)
            ax.fill_between(
                x, -rmse[mask], rmse[mask], color=ref_spec["color"], alpha=0.08
            )
            ax.plot(x, rmse[mask], color=ref_spec["color"], lw=0.8, ls="--", alpha=0.6)

        if not has_data:
            ax.text(
                0.5,
                0.5,
                "No data",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=11,
            )

        ax.axhline(0, color="k", lw=0.4)
        ax.set_title(rname.replace("_", " ").title(), fontsize=12)
        ax.tick_params(labelsize=9)
        ax.grid(True, alpha=0.15)
        if ci == 0:
            ax.set_ylabel("Diff (m)", fontsize=10)

    for ax in axes[-1, :]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.tick_params(axis="x", rotation=45, labelsize=9)

    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=len(legend_handles),
        bbox_to_anchor=(0.5, 0.01),
        frameon=True,
        fontsize=11,
        columnspacing=1.8,
        handlelength=2.8,
    )

    fig.suptitle(
        f"Monthly SSH Bias & RMSE by Region — Products vs. {obs_spec['label']}",
        fontsize=16,
    )
    fig.tight_layout(rect=[0, 0.08, 1, 0.94])
    fig.savefig(
        os.path.join(output_dir, f"fig2_regional_ts_{obs_key}.pdf"),
        dpi=dpi,
        bbox_inches="tight",
        facecolor="white",
    )
    logging.info(f"✓ fig2_regional_ts_{obs_key}.pdf")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 3 — Seasonal Spatial Difference Maps
# ═══════════════════════════════════════════════════════════════════════


def plot_fig3(accum, obs_key, ref_key, output_dir, dpi=300):
    obs_spec, ref_spec = OBS_PRODUCTS[obs_key], REF_PRODUCTS[ref_key]
    gm = accum.global_metrics()

    # Get bias limits from full-period map
    full_map = accum.spatial_mean_bias()
    bias_lim = np.nanpercentile(np.abs(full_map[np.isfinite(full_map)]), 95)
    bias_lim = max(bias_lim, 0.005)
    bnorm = mcolors.TwoSlopeNorm(vmin=-bias_lim, vcenter=0, vmax=bias_lim)

    fig = plt.figure(figsize=(14, 8))
    gs_s = gridspec.GridSpec(
        2, 2, hspace=0.25, wspace=0.10, left=0.04, right=0.96, top=0.90, bottom=0.07
    )

    for si, season in enumerate(SEASON_ORDER):
        ax = make_map_ax(fig, gs_s[si // 2, si % 2])
        grid = accum.season_mean_bias(season)
        n_valid = (accum.season_cnt[season] >= 5).sum()
        kw = {"transform": ccrs.PlateCarree()}
        ax.pcolormesh(
            LON_C, LAT_C, grid, cmap="RdBu_r", norm=bnorm, rasterized=True, **kw
        )

        total_n = accum.season_cnt[season].sum()
        ax.set_title(
            f"{season}  (n={total_n:,}, {n_valid} bins)", fontsize=10, fontweight="bold"
        )

    cbar_ax = fig.add_axes([0.15, 0.02, 0.70, 0.018])
    sm = plt.cm.ScalarMappable(cmap="RdBu_r", norm=bnorm)
    sm.set_array([])
    fig.colorbar(sm, cax=cbar_ax, orientation="horizontal", extend="both").set_label(
        "Mean Difference (m)", fontsize=9
    )

    tag = f"{ref_key}_vs_{obs_key}"
    fig.suptitle(
        f"Seasonal SSH Diff — {ref_spec['label']} vs. {obs_spec['label']}",
        fontsize=13,
        fontweight="bold",
    )
    fig.savefig(
        os.path.join(output_dir, f"fig3_seasonal_{tag}.pdf"),
        dpi=dpi,
        bbox_inches="tight",
        facecolor="white",
    )
    logging.info(f"✓ fig3_seasonal_{tag}.pdf")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 4 — Wavenumber Spectra (SWOT)
# ═══════════════════════════════════════════════════════════════════════


def plot_fig4(all_spectra, output_dir, dpi=300):
    if not all_spectra:
        return
    fig, axes = plt.subplots(1, len(all_spectra), figsize=(7 * len(all_spectra), 6))
    if len(all_spectra) == 1:
        axes = [axes]
    for ax, (ref_key, slist) in zip(axes, all_spectra.items()):
        ref_spec = REF_PRODUCTS[ref_key]
        wl = slist[0]["wavelength_km"]
        psd_o = np.nanmean([s["psd_obs"] for s in slist], axis=0)
        psd_r_l = [s["psd_ref"] for s in slist if s["psd_ref"] is not None]
        psd_d_l = [s["psd_diff"] for s in slist if s["psd_diff"] is not None]

        ax.loglog(wl, psd_o, color=OBS_PRODUCTS["swot"]["color"], lw=1.8, label="SWOT")
        if psd_r_l:
            ax.loglog(
                wl,
                np.nanmean(psd_r_l, axis=0),
                color=ref_spec["color"],
                lw=1.8,
                label=f"{ref_spec['label']} (interp.)",
            )
        if psd_d_l:
            psd_d = np.nanmean(psd_d_l, axis=0)
            ax.loglog(wl, psd_d, color="#333", lw=1.2, ls="--", label="Difference")
            # Effective resolution: where diff/obs > 0.5
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = psd_d / psd_o
            cross = np.where(ratio > 0.5)[0]
            if len(cross):
                eff = wl[cross[0]]
                ax.axvline(eff, color="#e67e22", ls=":", lw=1.2, alpha=0.7)
                ax.text(
                    eff * 1.1,
                    ax.get_ylim()[0] * 5 if ax.get_ylim()[0] > 0 else 1e-6,
                    f"Eff. res.\n~{eff:.0f} km",
                    fontsize=8,
                    color="#e67e22",
                )

        ax.set_xlabel("Wavelength (km)")
        ax.set_ylabel("PSD (m²/cpkm)")
        ax.set_title(f"SWOT vs. {ref_spec['label']}", fontsize=11, fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2, which="both")
        ax.set_xlim(5, 2000)

    fig.suptitle("Scale-Dependent SSH Analysis", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(
        os.path.join(output_dir, "fig4_spectra_swot.pdf"),
        dpi=dpi,
        bbox_inches="tight",
        facecolor="white",
    )
    logging.info("✓ fig4_spectra_swot.pdf")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 5 — Taylor + Target
# ═══════════════════════════════════════════════════════════════════════


class TaylorDiagram:
    def __init__(self, fig=None, rect=111, std_range=(0, 1.8), fig_kw=None):
        if fig is None:
            fig = plt.figure(**(fig_kw or {"figsize": (8, 7)}))
        self.fig = fig
        ct = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1])
        self.ax = fig.add_subplot(rect, projection="polar")
        self.ax.set_thetamin(0)
        self.ax.set_thetamax(90)
        self.ax.set_theta_direction(-1)
        self.ax.set_theta_offset(0)
        self.ax.set_thetagrids(np.degrees(np.arccos(ct)), labels=[f"{c}" for c in ct])
        self.ax.set_rlabel_position(0)
        self.ax.set_rlim(std_range)
        self.ax.plot(0, 1.0, "ko", ms=8, label="Reference (obs)")
        th = np.linspace(0, np.pi / 2, 200)
        for rv in np.linspace(0.25, std_range[1], 5):
            r_v = [
                (2 * np.cos(t) + np.sqrt(d)) / 2
                if (d := 4 * np.cos(t) ** 2 - 4 * (1 - rv**2)) >= 0
                else np.nan
                for t in th
            ]
            r_a = np.array(r_v)
            ok = np.isfinite(r_a) & (r_a <= std_range[1]) & (r_a >= 0)
            self.ax.plot(th[ok], r_a[ok], ":", color="gray", lw=0.6, alpha=0.5)

    def add_point(self, std_ratio, corr, **kw):
        self.ax.scatter(
            np.arccos(np.clip(corr, -1, 1)),
            std_ratio,
            edgecolors="k",
            linewidths=0.5,
            zorder=5,
            s=80,
            **kw,
        )


def plot_fig5(all_metrics, output_dir, dpi=300):
    td = TaylorDiagram(fig_kw={"figsize": (8, 7)})
    for key, m in all_metrics.items():
        if m is None:
            continue
        td.add_point(
            m["std_ratio"],
            m["correlation"],
            label=f"{key} (R={m['correlation']:.3f})",
            marker=m.get("marker", "o"),
            c=m.get("color", "gray"),
        )
    td.ax.legend(loc="upper left", fontsize=7, framealpha=0.9)
    td.ax.set_title(
        "Taylor Diagram — SSH Products vs. Observations",
        fontsize=12,
        fontweight="bold",
        pad=20,
    )
    td.fig.savefig(
        os.path.join(output_dir, "fig5a_taylor.pdf"), dpi=dpi, bbox_inches="tight"
    )
    plt.close(td.fig)

    fig, ax = plt.subplots(figsize=(7, 7))
    for key, m in all_metrics.items():
        if m is None:
            continue
        sign = 1.0 if m["std_model"] >= m["std_obs"] else -1.0
        ax.scatter(
            sign * m["norm_crmse"],
            m["norm_bias"],
            marker=m.get("marker", "o"),
            c=m.get("color", "gray"),
            s=120,
            edgecolors="k",
            lw=0.5,
            zorder=5,
            label=f"{key} (RMSE={m['rmse']:.4f}m)",
        )
    for r in [0.25, 0.5, 0.75, 1.0]:
        ax.add_patch(
            plt.Circle((0, 0), r, fill=False, color="gray", ls="--", lw=0.5, alpha=0.5)
        )
    ax.axhline(0, color="k", lw=0.5)
    ax.axvline(0, color="k", lw=0.5)
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect("equal")
    ax.set_xlabel("Norm. CRMSE")
    ax.set_ylabel("Norm. Bias")
    ax.set_title("Target Diagram — SSH", fontsize=12, fontweight="bold")
    ax.legend(loc="upper left", fontsize=7)
    ax.grid(True, alpha=0.3)
    fig.savefig(
        os.path.join(output_dir, "fig5b_target.pdf"), dpi=dpi, bbox_inches="tight"
    )
    logging.info("✓ Taylor + Target diagrams")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════
# Main Pipeline
# ═══════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(
        description="Validate GLORYS/L4 SSH against L3 altimetry and SWOT.",
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
        help="Local cache directory for remote downloads.",
    )
    parser.add_argument("--date-min", default="2023-04-01")
    parser.add_argument("--date-max", default="2024-03-31")
    parser.add_argument("--output-dir", default="./ssh_validation_results")
    parser.add_argument(
        "--subsample",
        type=int,
        default=5,
        help="Keep every Nth pixel for SWOT accumulation.",
    )
    parser.add_argument(
        "--spectral-regions",
        nargs="*",
        default=["NORTH_ATLANTIC"],
        help="Regions for spectral analysis (expensive). Use 'all' for all regions.",
    )
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    date_list = generate_date_list(args.date_min, args.date_max)
    month_list = generate_month_list(args.date_min, args.date_max)
    logging.info(
        f"Period: {args.date_min} -> {args.date_max} "
        f"({len(date_list)} days, {len(month_list)} months)"
    )

    # Resolve data source: local path or remote HuggingFace
    if args.taco_dir:
        taco_dir = args.taco_dir
        dataset_hf = None
        cache_dir = None
    else:
        taco_dir = None
        dataset_hf = load_hf_dataset(args.hf_url)
        cache_dir = args.cache_dir

    spectral_regions = (
        list(SPATIAL_REGIONS.keys())
        if args.spectral_regions == ["all"]
        else args.spectral_regions
    )

    # ── Create accumulators ──────────────────────────────────────────
    # One accumulator per (obs_product, ref_product) pair
    accumulators = {}
    for obs_key in OBS_PRODUCTS:
        accumulators[obs_key] = {}
        for ref_key in REF_PRODUCTS:
            accumulators[obs_key][ref_key] = BinnedAccumulator(month_list)

    spectra = {rk: [] for rk in REF_PRODUCTS}

    # ── Main loop ────────────────────────────────────────────────────
    total_pixels = 0
    for di, date_str in enumerate(date_list):
        cache = DayFileCache(dataset_hf=dataset_hf, cache_dir=cache_dir)

        for region in SPATIAL_REGIONS:
            for obs_key in OBS_PRODUCTS:
                subsample = args.subsample if obs_key == "swot" else 1
                for ref_key in REF_PRODUCTS:
                    n = compare_and_accumulate(
                        cache,
                        taco_dir,
                        obs_key,
                        ref_key,
                        region,
                        date_str,
                        accumulators[obs_key][ref_key],
                        subsample=subsample,
                    )
                    total_pixels += n

            # Spectra (SWOT only, selected regions)
            if region in spectral_regions:
                for ref_key in REF_PRODUCTS:
                    sp = compute_difference_spectrum(
                        cache, taco_dir, ref_key, region, date_str
                    )
                    if sp is not None:
                        spectra[ref_key].append(sp)

        cache.flush()

        if (di + 1) % 30 == 0:
            logging.info(
                f"  Day {di + 1}/{len(date_list)}: "
                f"{total_pixels:,} comparison pixels accumulated"
            )

    # ── Print global metrics ─────────────────────────────────────────
    all_taylor = {}
    for obs_key in OBS_PRODUCTS:
        obs_label = OBS_PRODUCTS[obs_key]["label"]
        print(f"\n{'═' * 65}")
        print(f"  SSH VALIDATION vs. {obs_label}")
        print(f"{'═' * 65}")
        for ref_key in REF_PRODUCTS:
            accum = accumulators[obs_key][ref_key]
            m = accum.global_metrics()
            ref_spec = REF_PRODUCTS[ref_key]
            if m is None:
                print(f"  {ref_spec['label']:12s}: no data")
                continue
            key = f"{ref_spec['label']} vs {obs_label}"
            m["color"] = ref_spec["color"]
            m["marker"] = OBS_PRODUCTS[obs_key]["marker"]
            all_taylor[key] = m
            print(
                f"  {ref_spec['label']:12s}: Bias={m['bias']:+.5f} m  "
                f"RMSE={m['rmse']:.5f} m  R={m['correlation']:.4f}  "
                f"n={m['n']:,}"
            )

    # ── Generate figures ─────────────────────────────────────────────
    for obs_key in OBS_PRODUCTS:
        for ref_key in REF_PRODUCTS:
            accum = accumulators[obs_key][ref_key]
            if accum.global_n < 50:
                continue
            plot_fig1(accum, obs_key, ref_key, month_list, str(output_dir), args.dpi)
            plot_fig3(accum, obs_key, ref_key, str(output_dir), args.dpi)

        # Fig 2: regional time series (all ref products overlaid)
        ref_accums = {
            rk: accumulators[obs_key][rk]
            for rk in REF_PRODUCTS
            if accumulators[obs_key][rk].global_n > 0
        }
        if ref_accums:
            plot_fig2(ref_accums, obs_key, month_list, str(output_dir), args.dpi)

    spec_with_data = {rk: sl for rk, sl in spectra.items() if sl}
    if spec_with_data:
        plot_fig4(spec_with_data, str(output_dir), args.dpi)

    if any(v is not None for v in all_taylor.values()):
        plot_fig5(all_taylor, str(output_dir), args.dpi)

    # ── Summary ──────────────────────────────────────────────────────
    print(f"\n{'═' * 65}")
    print(f"  OUTPUT FILES ({output_dir})")
    print(f"{'═' * 65}")
    for f in sorted(output_dir.glob("*")):
        print(f"  {f.name:55s}  {f.stat().st_size / 1024:8.1f} KB")
    print(f"{'═' * 65}")
    print(f"  Total comparison pixels processed: {total_pixels:,}")
    print(f"{'═' * 65}")


if __name__ == "__main__":
    main()
