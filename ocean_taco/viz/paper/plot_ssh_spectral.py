#!/usr/bin/env python3
"""Plot spectral analysis of SSH products."""

import argparse
import os
import warnings
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path

import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.lines import Line2D
from scipy.interpolate import RegularGridInterpolator

from ocean_taco.dataset.retrieve import HF_DEFAULT_URL, load_hf_dataset, load_tile_nc


def _configure_cartopy_dir(path: str):
    """Configure cartopy data directory."""
    p = Path(path).expanduser()
    p.mkdir(parents=True, exist_ok=True)
    os.environ["CARTOPY_USER_DIR"] = str(p)
    cartopy.config["data_dir"] = str(p)


_configure_cartopy_dir("./.cartopy")


warnings.filterwarnings("ignore", category=RuntimeWarning)

# =============================================================================
# Configuration
# =============================================================================

LON_MIN, LON_MAX = 125, 165
LAT_MIN, LAT_MAX = 20, 50

START_DATE = datetime(2025, 3, 1)
N_DAYS = 30

TILE = "NORTH_PACIFIC_EAST"

PRODUCTS = {
    "l3_swot": ("l3_swot", "ssha_filtered", 2.0, "L3 SWOT (2 km)", "#e63946"),
    "l3_ssh": ("l3_ssh", "sla_filtered", 7.0, "L3 along-track (7 km)", "#f4a261"),
    "glorys": ("glorys", "zos", 9.3, "GLORYS-12 (9.3 km)", "#2a9d8f"),
    "l4_ssh": ("l4_ssh", "sla", 13.3, "L4 DUACS (13.3 km)", "#264653"),
}

MIN_SEGMENT_KM = 300
MIN_SEGMENT_PTS = 50
MIN_OCEAN_FRAC = 0.85


# =============================================================================
# I/O helpers
# =============================================================================


def _load_from_cache(cache_dir: Path, dt: datetime, data_source: str) -> xr.Dataset | None:
    date_str = dt.strftime("%Y-%m-%d")
    path = cache_dir / date_str / TILE / f"{data_source}.nc"
    if not path.exists():
        return None
    return xr.open_dataset(path, engine="h5netcdf")


def crop_to_region(ds: xr.Dataset, var: str) -> xr.DataArray:
    lats = ds["lat"].values
    lat_sl = slice(LAT_MAX, LAT_MIN) if lats[0] > lats[-1] else slice(LAT_MIN, LAT_MAX)
    da = ds[var].sel(lon=slice(LON_MIN, LON_MAX), lat=lat_sl)
    if "time" in da.dims:
        da = da.isel(time=0)
    if "depth" in da.dims:
        da = da.isel(depth=0)
    return da.squeeze()


def build_interpolator(da: xr.DataArray):
    lons = da["lon"].values
    lats = da["lat"].values
    vals = da.values.astype(float)
    vals -= np.nanmean(vals)
    if lats[0] > lats[-1]:
        lats = lats[::-1]
        vals = vals[::-1, :]
    vals_filled = vals.copy()
    vals_filled[np.isnan(vals_filled)] = 0.0
    return RegularGridInterpolator(
        (lats, lons),
        vals_filled,
        method="linear",
        bounds_error=False,
        fill_value=np.nan,
    )


# =============================================================================
# Track extraction
# =============================================================================


def haversine_km(lon1, lat1, lon2, lat2):
    R = 6371.0
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2) ** 2
    )
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def extract_track_segments(ds: xr.Dataset, var: str, dx_km: float):
    """Extract 1D segments from gridded L3 product using primary_track.
    Returns list of (values_1d, dx_km) tuples.
    """
    da = crop_to_region(ds, var)
    lons = da["lon"].values
    lats = da["lat"].values

    if "primary_track" in ds:
        track_da = crop_to_region(ds, "primary_track")
        track_grid = track_da.values
    else:
        track_grid = None

    vals = da.values.copy()
    vals = vals - np.nanmean(vals)
    segments = []

    if track_grid is not None:
        unique_tracks = np.unique(track_grid[~np.isnan(track_grid) & (track_grid >= 0)])
        for tid in unique_tracks:
            tid = int(tid)
            mask = track_grid == tid
            ys, xs = np.where(mask & ~np.isnan(vals))
            if len(ys) < MIN_SEGMENT_PTS:
                continue

            order = np.argsort(lats[ys])
            ys, xs = ys[order], xs[order]
            seg_vals = vals[ys, xs]
            seg_lons = lons[xs]
            seg_lats = lats[ys]

            dists = np.zeros(len(seg_vals))
            for i in range(1, len(seg_vals)):
                dists[i] = dists[i - 1] + haversine_km(
                    seg_lons[i - 1], seg_lats[i - 1], seg_lons[i], seg_lats[i]
                )

            if dists[-1] < MIN_SEGMENT_KM:
                continue

            steps = np.diff(dists)
            if len(steps) == 0:
                continue
            med_step = np.median(steps[steps > 0]) if np.any(steps > 0) else dx_km
            gap_idx = np.where(steps > 3 * med_step)[0]
            split_points = [0] + list(gap_idx + 1) + [len(seg_vals)]

            for a, b in zip(split_points[:-1], split_points[1:]):
                sub = seg_vals[a:b]
                sub_dist = dists[min(b, len(dists)) - 1] - dists[a]
                n_valid = np.sum(~np.isnan(sub))
                if (
                    sub_dist >= MIN_SEGMENT_KM
                    and len(sub) >= MIN_SEGMENT_PTS
                    and n_valid / len(sub) > 0.9
                ):
                    good = ~np.isnan(sub)
                    sub_clean = np.interp(
                        np.arange(len(sub)), np.where(good)[0], sub[good]
                    )
                    avg_dx = sub_dist / (len(sub_clean) - 1)
                    segments.append((sub_clean, avg_dx))
    else:
        for j in range(vals.shape[1]):
            col = vals[:, j]
            valid_mask = ~np.isnan(col)
            if np.sum(valid_mask) < MIN_SEGMENT_PTS:
                continue
            indices = np.where(valid_mask)[0]
            diffs = np.diff(indices)
            breaks = np.where(diffs > 3)[0]
            split_points = [0] + list(breaks + 1) + [len(indices)]
            for a, b in zip(split_points[:-1], split_points[1:]):
                idx = indices[a:b]
                sub = col[idx]
                seg_lats_sub = lats[idx]
                total_km = haversine_km(
                    lons[j], seg_lats_sub[0], lons[j], seg_lats_sub[-1]
                )
                if total_km >= MIN_SEGMENT_KM and len(sub) >= MIN_SEGMENT_PTS:
                    avg_dx = total_km / (len(sub) - 1)
                    segments.append((sub, avg_dx))

    return segments


# =============================================================================
# Spectral analysis — 1D
# =============================================================================


def psd_1d(values: np.ndarray, dx_km: float):
    """1D PSD with Parseval-consistent normalization.
    Returns (wavenumbers [cpkm], psd [m²/cpkm]) or (None, None) on failure.
    """
    n = len(values)
    if n < MIN_SEGMENT_PTS:
        return None, None

    x = np.arange(n, dtype=float)
    coeffs = np.polyfit(x, values, 1)
    detrended = values - np.polyval(coeffs, x)

    window = np.hanning(n)
    window_power = np.mean(window**2)
    if window_power < 1e-15:
        return None, None

    windowed = detrended * window
    if np.all(windowed == 0) or np.any(np.isnan(windowed)):
        return None, None

    L = n * dx_km
    fft_vals = np.fft.rfft(windowed)
    psd = (np.abs(fft_vals) ** 2) / (n**2 * window_power) * L
    psd[1:-1] *= 2

    freqs = np.fft.rfftfreq(n, d=dx_km)
    k = freqs[1:]
    p = psd[1:]
    valid = (k > 0) & (p > 0) & np.isfinite(p)
    if np.sum(valid) < 5:
        return None, None

    return k[valid], p[valid]


# =============================================================================
# Spectral analysis — 2D isotropic
# =============================================================================


def psd_2d_isotropic(field: np.ndarray, dx_km: float, dy_km: float):
    ny, nx = field.shape
    yy, xx = np.mgrid[0:ny, 0:nx].astype(float)
    A = np.column_stack([xx.ravel(), yy.ravel(), np.ones(nx * ny)])
    coeffs, _, _, _ = np.linalg.lstsq(A, field.ravel(), rcond=None)
    plane = coeffs[0] * xx + coeffs[1] * yy + coeffs[2]
    detrended = field - plane

    win_y = np.hanning(ny)
    win_x = np.hanning(nx)
    window = np.outer(win_y, win_x)
    windowed = detrended * window
    window_power = np.mean(window**2)
    if window_power < 1e-15:
        return None, None

    Lx = nx * dx_km
    Ly = ny * dy_km

    fft2 = np.fft.fft2(windowed)
    power2d = (np.abs(fft2) ** 2) / ((nx * ny) ** 2) * (Lx * Ly) / window_power

    kx = np.fft.fftfreq(nx, d=dx_km)
    ky = np.fft.fftfreq(ny, d=dy_km)
    KX, KY = np.meshgrid(kx, ky)
    K_iso = np.sqrt(KX**2 + KY**2)

    k_max = min(1 / (2 * dx_km), 1 / (2 * dy_km))
    dk = max(1 / Lx, 1 / Ly)
    k_edges = np.arange(dk / 2, k_max + dk, dk)
    k_centers = k_edges[:-1] + dk / 2

    psd_iso = np.zeros(len(k_centers))
    for i in range(len(k_centers)):
        ring = (K_iso >= k_edges[i]) & (K_iso < k_edges[i + 1])
        if np.any(ring):
            psd_iso[i] = np.mean(power2d[ring]) * 2 * np.pi * k_centers[i]

    valid = psd_iso > 0
    if np.sum(valid) < 5:
        return None, None
    return k_centers[valid], psd_iso[valid]


def compute_gridded_psd(da: xr.DataArray, dx_km: float, dy_km: float):
    field = da.values.astype(float)
    if np.sum(~np.isnan(field)) / field.size < MIN_OCEAN_FRAC:
        return None, None
    field[np.isnan(field)] = np.nanmean(field)
    return psd_2d_isotropic(field, dx_km, dy_km)


# =============================================================================
# Averaging helper
# =============================================================================


def average_psds(psd_list):
    if not psd_list:
        return None, None

    k_min = np.max([k.min() for k, _ in psd_list])
    k_max = np.min([k.max() for k, _ in psd_list])
    if k_min >= k_max:
        all_k = np.concatenate([k for k, _ in psd_list])
        k_min = np.min(all_k[all_k > 0])
        k_max = np.max(all_k)

    k_bins = np.logspace(np.log10(k_min), np.log10(k_max), 200)
    log_psds = []
    for k, psd in psd_list:
        valid = (k > 0) & (psd > 0)
        if np.sum(valid) < 5:
            continue
        interp = np.interp(
            np.log10(k_bins),
            np.log10(k[valid]),
            np.log10(psd[valid]),
            left=np.nan,
            right=np.nan,
        )
        log_psds.append(interp)

    if not log_psds:
        return None, None
    log_psds = np.array(log_psds)
    mean = np.nanmean(log_psds, axis=0)
    valid = ~np.isnan(mean)
    return k_bins[valid], 10 ** mean[valid]


# =============================================================================
# Single-day processing (parallelized)
# =============================================================================


def _run_psd_on_ds(ds, key, var, dx_km):
    """Compute PSD for one dataset (shared by cached and remote paths)."""
    psd_list = []
    try:
        if key in ("l3_swot", "l3_ssh"):
            segments = extract_track_segments(ds, var, dx_km)
            for seg_vals, seg_dx in segments:
                k, psd = psd_1d(seg_vals, seg_dx)
                if k is not None:
                    psd_list.append((k, psd))
        else:
            da = crop_to_region(ds, var)
            mid_lat = float(da["lat"].mean())
            dx_actual = dx_km * np.cos(np.radians(mid_lat))
            k, psd = compute_gridded_psd(da, dx_actual, dx_km)
            if k is not None:
                psd_list.append((k, psd))
    except Exception:
        pass
    return psd_list


def process_single_day_cached(dt_str: str, cache_dir_str: str):
    """Process one day reading from local cache (safe for ProcessPoolExecutor)."""
    dt = datetime.strptime(dt_str, "%Y-%m-%d")
    cache_dir = Path(cache_dir_str)
    psd_results = {key: [] for key in PRODUCTS}

    for key, (fname, var, dx_km, label, color) in PRODUCTS.items():
        ds = _load_from_cache(cache_dir, dt, fname)
        if ds is None:
            continue
        psd_results[key].extend(_run_psd_on_ds(ds, key, var, dx_km))
        ds.close()

    return psd_results


def process_single_day_remote(dt: datetime, dataset_hf):
    """Process one day loading directly from HuggingFace (sequential only)."""
    psd_results = {key: [] for key in PRODUCTS}

    for key, (fname, var, dx_km, label, color) in PRODUCTS.items():
        date_str = dt.strftime("%Y-%m-%d")
        ds = load_tile_nc(dataset_hf, date_str, TILE, fname)
        if ds is None:
            continue
        psd_results[key].extend(_run_psd_on_ds(ds, key, var, dx_km))
        ds.close()

    return psd_results


# =============================================================================
# Main computation
# =============================================================================


def compute_all(
    dataset_hf,
    cache_dir=None,
    max_workers: int = 8,
    snapshot_date: datetime | None = None,
):
    dates = [START_DATE + timedelta(days=i) for i in range(N_DAYS)]

    all_psds = {key: [] for key in PRODUCTS}
    days_with_data = {key: 0 for key in PRODUCTS}

    if cache_dir:
        cache_path = Path(cache_dir)
        # Pre-download all files in parallel with threads
        print(f"  Pre-downloading {N_DAYS} × {len(PRODUCTS)} files ...")

        def _prefetch(dt, fname):
            date_str = dt.strftime("%Y-%m-%d")
            try:
                load_tile_nc(dataset_hf, date_str, TILE, fname, cache_dir)
            except Exception as e:
                print(f"\n  Warning fetching {fname} {date_str}: {e}")

        with ThreadPoolExecutor(max_workers=max_workers) as tpool:
            futs = [
                tpool.submit(_prefetch, dt, fname)
                for dt in dates
                for fname, _, _, _, _ in PRODUCTS.values()
            ]
            for f in futs:
                f.result()

        print(f"  Dispatching {N_DAYS} days across {max_workers} workers ...")
        date_strs = [dt.strftime("%Y-%m-%d") for dt in dates]
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(process_single_day_cached, ds, str(cache_path)): ds
                for ds in date_strs
            }
            done = 0
            for future in as_completed(futures):
                done += 1
                print(f"\r  Completed {done}/{N_DAYS} days", end="", flush=True)
                try:
                    psd_results = future.result()
                    for key in PRODUCTS:
                        if psd_results[key]:
                            all_psds[key].extend(psd_results[key])
                            days_with_data[key] += 1
                except Exception as e:
                    print(f"\n  Warning: {e}")
    else:
        print(f"  Processing {N_DAYS} days sequentially (use --cache-dir for parallel) ...")
        for di, dt in enumerate(dates):
            print(f"\r  Processing day {di + 1}/{N_DAYS}", end="", flush=True)
            psd_results = process_single_day_remote(dt, dataset_hf)
            for key in PRODUCTS:
                if psd_results[key]:
                    all_psds[key].extend(psd_results[key])
                    days_with_data[key] += 1

    print("\n\n  PSD data per product:")
    for key in PRODUCTS:
        print(
            f"    {PRODUCTS[key][3]:30s}: {days_with_data[key]:3d} days, "
            f"{len(all_psds[key]):5d} spectra"
        )

    # Snapshot
    if snapshot_date is None:
        print("\n  Auto-selecting snapshot date ...")
        for dt in dates:
            date_str = dt.strftime("%Y-%m-%d")
            if cache_dir:
                ds_swot = _load_from_cache(Path(cache_dir), dt, "l3_swot")
            else:
                ds_swot = load_tile_nc(dataset_hf, date_str, TILE, "l3_swot")
            if ds_swot is not None:
                segs = extract_track_segments(ds_swot, "ssha_filtered", 2.0)
                ds_swot.close()
                if segs:
                    snapshot_date = dt
                    break
        if snapshot_date is None:
            snapshot_date = dates[0]
    else:
        print("\n  Using specified snapshot date ...")

    snapshot_data = {}
    for key, (fname, var, dx_km, label, color) in PRODUCTS.items():
        snap_date_str = snapshot_date.strftime("%Y-%m-%d")
        if cache_dir:
            ds = _load_from_cache(Path(cache_dir), snapshot_date, fname)
        else:
            ds = load_tile_nc(dataset_hf, snap_date_str, TILE, fname)
        if ds is not None:
            try:
                snapshot_data[key] = crop_to_region(ds, var)
            except Exception:
                pass
            ds.close()
    print(f"  Snapshot date: {snapshot_date.strftime('%Y-%m-%d')}")

    averaged_psds = {}
    for key in PRODUCTS:
        k, psd = average_psds(all_psds[key])
        if k is not None:
            averaged_psds[key] = (k, psd)

    return averaged_psds, snapshot_data, snapshot_date


# =============================================================================
# Error map
# =============================================================================


def compute_error_map(snapshot_data):
    if "l4_ssh" not in snapshot_data:
        return None
    interp_func = build_interpolator(snapshot_data["l4_ssh"])
    error_maps = {}
    for l3_key in ("l3_ssh", "l3_swot"):
        if l3_key not in snapshot_data:
            continue
        l3_da = snapshot_data[l3_key]
        l3_vals = l3_da.values.astype(float)
        l3_lons = l3_da["lon"].values
        l3_lats = l3_da["lat"].values
        l3_vals -= np.nanmean(l3_vals)

        l3_valid = ~np.isnan(l3_vals)
        lon_grid, lat_grid = np.meshgrid(l3_lons, l3_lats)
        pts = np.column_stack([lat_grid.ravel(), lon_grid.ravel()])
        l4_on_l3 = interp_func(pts).reshape(l3_vals.shape)

        diff = np.full_like(l3_vals, np.nan)
        diff[l3_valid] = l4_on_l3[l3_valid] - l3_vals[l3_valid]
        error_maps[l3_key] = xr.DataArray(diff, coords=l3_da.coords, dims=l3_da.dims)
    return error_maps


# =============================================================================
# Figure
# =============================================================================


def make_figure(averaged_psds, snapshot_data, snapshot_date, output_path):
    """Three-panel figure:
    1. SSH snapshot map
    2. L4 − L3 error map
    3. Wavenumber PSD
    """
    fig = plt.figure(figsize=(17, 5.5))
    gs = gridspec.GridSpec(
        1,
        3,
        width_ratios=[1.2, 1.2, 1.0],
        wspace=0.28,
        left=0.04,
        right=0.97,
        bottom=0.12,
        top=0.85,
    )

    proj = ccrs.Mercator()
    date_str = snapshot_date.strftime("%Y-%m-%d") if snapshot_date else "N/A"

    # =========================================================================
    # Panel 1: SSH snapshot
    # =========================================================================
    ax_map = fig.add_subplot(gs[0], projection=proj)
    ax_map.set_extent([LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], crs=ccrs.PlateCarree())
    ax_map.coastlines(linewidth=0.6, color="#444")
    ax_map.add_feature(cfeature.LAND, facecolor="#e8e8e8", edgecolor="none", zorder=2)
    gl = ax_map.gridlines(draw_labels=True, linewidth=0.3, alpha=0.4)
    gl.right_labels = gl.top_labels = False
    gl.xlabel_style = gl.ylabel_style = {"size": 7}

    vmin, vmax = -0.4, 0.4
    mappable = None

    if "l4_ssh" in snapshot_data:
        da = snapshot_data["l4_ssh"]
        vals = da.values.astype(float)
        vals -= np.nanmean(vals)
        lons2d, lats2d = np.meshgrid(da["lon"].values, da["lat"].values)
        mappable = ax_map.pcolormesh(
            lons2d,
            lats2d,
            vals,
            transform=ccrs.PlateCarree(),
            cmap="RdBu_r",
            vmin=vmin,
            vmax=vmax,
            shading="auto",
            alpha=0.6,
            rasterized=True,
            zorder=1,
        )

    for key, sz in [("l3_ssh", 0.3), ("l3_swot", 0.05)]:
        if key not in snapshot_data:
            continue
        da = snapshot_data[key]
        vals = da.values.astype(float)
        vals -= np.nanmean(vals)
        lons2d, lats2d = np.meshgrid(da["lon"].values, da["lat"].values)
        mask = ~np.isnan(vals)
        if np.any(mask):
            sc = ax_map.scatter(
                lons2d[mask],
                lats2d[mask],
                c=vals[mask],
                s=sz,
                cmap="RdBu_r",
                vmin=vmin,
                vmax=vmax,
                transform=ccrs.PlateCarree(),
                zorder=3,
                rasterized=True,
            )
            if mappable is None:
                mappable = sc

    if mappable is not None:
        cb = plt.colorbar(
            mappable,
            ax=ax_map,
            orientation="horizontal",
            pad=0.06,
            fraction=0.04,
            aspect=30,
        )
        cb.set_label("SSH anomaly (m, mean-removed)", fontsize=8)
        cb.ax.tick_params(labelsize=7)

    ax_map.set_title(f"SSH snapshot — {date_str}", fontsize=10, fontweight="bold")

    legend_items = []
    if "l4_ssh" in snapshot_data:
        legend_items.append(
            Line2D(
                [0], [0], color=PRODUCTS["l4_ssh"][4], lw=6, alpha=0.6, label="L4 DUACS"
            )
        )
    if "l3_ssh" in snapshot_data:
        legend_items.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=PRODUCTS["l3_ssh"][4],
                markersize=4,
                label="L3 along-track",
            )
        )
    if "l3_swot" in snapshot_data:
        legend_items.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=PRODUCTS["l3_swot"][4],
                markersize=4,
                label="L3 SWOT",
            )
        )
    if legend_items:
        ax_map.legend(
            handles=legend_items,
            loc="lower left",
            fontsize=7,
            framealpha=0.9,
            edgecolor="none",
        )

    # =========================================================================
    # Panel 2: L4 − L3 error map
    # =========================================================================
    ax_err = fig.add_subplot(gs[1], projection=proj)
    ax_err.set_extent([LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], crs=ccrs.PlateCarree())
    ax_err.coastlines(linewidth=0.6, color="#444")
    ax_err.add_feature(cfeature.LAND, facecolor="#e8e8e8", edgecolor="none", zorder=2)
    gl2 = ax_err.gridlines(draw_labels=True, linewidth=0.3, alpha=0.4)
    gl2.right_labels = gl2.top_labels = False
    gl2.xlabel_style = gl2.ylabel_style = {"size": 7}

    error_maps = compute_error_map(snapshot_data)
    err_mappable = None

    if error_maps:
        for l3_key, sz in [("l3_ssh", 0.5), ("l3_swot", 0.08)]:
            if l3_key not in error_maps:
                continue
            err_da = error_maps[l3_key]
            err_vals = err_da.values
            lons2d, lats2d = np.meshgrid(err_da["lon"].values, err_da["lat"].values)
            mask = ~np.isnan(err_vals)
            if np.any(mask):
                err_mappable = ax_err.scatter(
                    lons2d[mask],
                    lats2d[mask],
                    c=err_vals[mask],
                    s=sz,
                    cmap="PuOr",
                    vmin=-0.15,
                    vmax=0.15,
                    transform=ccrs.PlateCarree(),
                    zorder=3,
                    rasterized=True,
                )

    if err_mappable is not None:
        cb2 = plt.colorbar(
            err_mappable,
            ax=ax_err,
            orientation="horizontal",
            pad=0.06,
            fraction=0.04,
            aspect=30,
        )
        cb2.set_label("L4 DUACS − L3 observations (m)", fontsize=8)
        cb2.ax.tick_params(labelsize=7)

        all_errs = []
        for l3_key in error_maps:
            v = error_maps[l3_key].values
            all_errs.append(v[~np.isnan(v)])
        if all_errs:
            all_errs = np.concatenate(all_errs)
            rmse = np.sqrt(np.mean(all_errs**2))
            mae = np.mean(np.abs(all_errs))
            ax_err.text(
                0.02,
                0.97,
                f"RMSE = {rmse:.3f} m\nMAE  = {mae:.3f} m",
                transform=ax_err.transAxes,
                fontsize=7,
                va="top",
                ha="left",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                zorder=10,
            )

    ax_err.set_title(f"L4 − L3 difference — {date_str}", fontsize=10, fontweight="bold")

    err_legend = []
    if error_maps and "l3_ssh" in error_maps:
        err_legend.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="#888",
                markersize=4,
                label="vs L3 along-track",
            )
        )
    if error_maps and "l3_swot" in error_maps:
        err_legend.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="#888",
                markersize=3,
                label="vs L3 SWOT",
            )
        )
    if err_legend:
        ax_err.legend(
            handles=err_legend,
            loc="lower left",
            fontsize=7,
            framealpha=0.9,
            edgecolor="none",
        )

    # =========================================================================
    # Panel 3: PSD
    # =========================================================================
    ax_psd = fig.add_subplot(gs[2])

    for key, (fname, var, dx_km, label, color) in PRODUCTS.items():
        if key not in averaged_psds:
            continue
        k, psd = averaged_psds[key]
        ax_psd.loglog(1.0 / k, psd, color=color, linewidth=1.8, label=label, alpha=0.9)

    # Auto-anchored reference slopes
    if averaged_psds:
        anchor_key = max(averaged_psds, key=lambda k: len(averaged_psds[k][0]))
        k_a, psd_a = averaged_psds[anchor_key]
        idx_200 = np.argmin(np.abs(1.0 / k_a - 200))
        k0 = k_a[idx_200]
        p0 = psd_a[idx_200]
        k_line = np.logspace(np.log10(k_a.min()), np.log10(k_a.max()), 100)
        wl_line = 1.0 / k_line

        for exp, style, txt, offset in [
            (-5, "--", r"$k^{-5}$", 4.0),
            (-11 / 3, ":", r"$k^{-11/3}$", 0.25),
        ]:
            slope = p0 * offset * (k_line / k0) ** exp
            ax_psd.loglog(wl_line, slope, f"k{style}", lw=0.7, alpha=0.35, zorder=0)
            i_lbl = len(k_line) // 4
            ax_psd.text(wl_line[i_lbl], slope[i_lbl] * 1.8, txt, fontsize=7, alpha=0.5)

    ax_psd.set_xlabel("Wavelength (km)", fontsize=9)
    ax_psd.set_ylabel(r"PSD (m$^2$ / cpkm)", fontsize=9)
    ax_psd.set_title("SSH wavenumber spectra", fontsize=10, fontweight="bold")
    ax_psd.legend(fontsize=7, loc="upper left", framealpha=0.9, edgecolor="none")
    ax_psd.set_xlim(10, 1000)
    ax_psd.tick_params(labelsize=8)
    ax_psd.grid(True, which="both", alpha=0.15)
    ax_psd.invert_xaxis()

    ax_k = ax_psd.secondary_xaxis(
        "top", functions=(lambda x: 1.0 / x, lambda k: 1.0 / k)
    )
    ax_k.set_xlabel("Wavenumber (cpkm)", fontsize=8)
    ax_k.tick_params(labelsize=7)

    fig.suptitle(
        f"OceanTACO — SSH spectral analysis across processing levels  ·  "
        f"Kuroshio Current  ·  "
        f"{START_DATE.strftime('%b %Y')}  ({N_DAYS} days)",
        fontsize=12,
        fontweight="bold",
    )

    plt.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"\n  Figure saved to {output_path}")
    plt.show()


# =============================================================================
# Entry point
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="OceanTACO SSH spectral analysis across processing levels"
    )
    parser.add_argument(
        "--hf-url",
        type=str,
        default=HF_DEFAULT_URL,
        help="HuggingFace dataset URL",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Local cache directory (enables parallel processing via pre-download)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="ssh_spectral_analysis.png",
        help="Output figure path",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Start date (YYYY-MM-DD). Default: 2025-03-01",
    )
    parser.add_argument(
        "--n-days", type=int, default=30, help="Number of days to average (default: 30)"
    )
    parser.add_argument(
        "--workers", type=int, default=8, help="Parallel workers (default: 8)"
    )
    parser.add_argument(
        "--snapshot-date",
        type=str,
        default="2025-03-15",
        help="Snapshot date for map panels (YYYY-MM-DD). Auto-selected if omitted.",
    )
    args = parser.parse_args()

    dataset_hf = load_hf_dataset(args.hf_url)

    global START_DATE, N_DAYS
    if args.start_date:
        START_DATE = datetime.strptime(args.start_date, "%Y-%m-%d")
    N_DAYS = args.n_days

    snap_date = None
    if args.snapshot_date:
        snap_date = datetime.strptime(args.snapshot_date, "%Y-%m-%d")

    print("=" * 65)
    print("  OceanTACO — SSH Spectral Analysis")
    print(f"  Region   : Kuroshio ({LON_MIN}–{LON_MAX}°E, {LAT_MIN}–{LAT_MAX}°N)")
    print(f"  Period   : {START_DATE.strftime('%Y-%m-%d')} + {N_DAYS} days")
    print(f"  Tile     : {TILE}")
    print(f"  Workers  : {args.workers}")
    print("=" * 65)

    print("\n[1/2] Computing power spectral densities ...")
    averaged_psds, snapshot_data, snapshot_date = compute_all(
        dataset_hf,
        cache_dir=args.cache_dir,
        max_workers=args.workers,
        snapshot_date=snap_date,
    )

    print("\n[2/2] Generating figure ...")
    make_figure(averaged_psds, snapshot_data, snapshot_date, args.output)

    print("\nDone.")


if __name__ == "__main__":
    main()
