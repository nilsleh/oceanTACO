#!/usr/bin/env python3
"""SSH comparison across processing levels: L3 nadir, L3 SWOT, L4 DUACS.

Two dynamically active regions (Gulf Stream, Kuroshio) are shown side-by-side,
with SSH snapshot maps and averaged wavenumber power spectral density (PSD)
panels that quantify scale-dependent attenuation across processing levels.
"""

import argparse
import csv
import os
from datetime import datetime, timedelta
from pathlib import Path

import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from ocean_taco.dataset.retrieve import (
    HF_DEFAULT_URL,
    load_bbox_nc,
    load_bbox_swot_nc,
    load_hf_dataset,
)


def _configure_cartopy_dir(path: str):
    p = Path(path).expanduser()
    p.mkdir(parents=True, exist_ok=True)
    os.environ["CARTOPY_USER_DIR"] = str(p)
    cartopy.config["data_dir"] = str(p)


_configure_cartopy_dir("./.cartopy")

# =============================================================================
# Configuration
# =============================================================================

REGIONS = {
    "gulf_stream": {
        "bbox": (-80, 25, -40, 45),  # (lon_min, lat_min, lon_max, lat_max)
        "label": "Gulf Stream",
    },
    "kuroshio": {
        "bbox": (130, 25, 170, 45),
        "label": "Kuroshio Current",
    },
}

PRODUCTS = {
    "l3_swot": ("l3_swot.nc", "ssha_filtered", 2.0, "L3 SWOT (2 km)", "#e63946"),
    "l3_ssh": ("l3_ssh.nc", "sla_filtered", 7.0, "L3 along-track (7 km)", "#f4a261"),
    "l4_ssh": ("l4_ssh.nc", "sla", 13.3, "L4 DUACS (13.3 km)", "#264653"),
}

MIN_SEGMENT_KM = 300
MIN_SEGMENT_PTS = 50


# =============================================================================
# I/O helpers
# =============================================================================


def crop_to_region(ds: xr.Dataset, var: str, bbox: tuple) -> xr.DataArray:
    """Crop a dataset variable to the given bounding box and squeeze."""
    lon_min, lat_min, lon_max, lat_max = bbox
    lats = ds["lat"].values
    lat_sl = slice(lat_max, lat_min) if lats[0] > lats[-1] else slice(lat_min, lat_max)
    da = ds[var].sel(lon=slice(lon_min, lon_max), lat=lat_sl)
    if "time" in da.dims:
        da = da.isel(time=0)
    if "depth" in da.dims:
        da = da.isel(depth=0)
    return da.squeeze()


# =============================================================================
# Track extraction
# =============================================================================


def haversine_km(lon1, lat1, lon2, lat2):
    """Compute great-circle distance in kilometres between two points."""
    R = 6371.0
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2) ** 2
    )
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def _choose_track_order(lons, lats, xs, ys, dx_km: float):
    """Choose a stable 1D order for track points using realistic step spacing.

    Some products have track ids but not a guaranteed in-memory point order after
    masking/cropping. We compare sorting by latitude vs. longitude and keep the
    one whose median step is closest to the nominal product spacing.
    """

    def _score(order_idx):
        if len(order_idx) < 2:
            return np.inf
        olon = lons[xs[order_idx]]
        olat = lats[ys[order_idx]]
        steps = haversine_km(olon[:-1], olat[:-1], olon[1:], olat[1:])
        steps = steps[np.isfinite(steps) & (steps > 0)]
        if steps.size == 0:
            return np.inf
        med = float(np.median(steps))
        return abs(med - dx_km)

    order_lat = np.argsort(lats[ys])
    order_lon = np.argsort(lons[xs])

    score_lat = _score(order_lat)
    score_lon = _score(order_lon)
    return order_lat if score_lat <= score_lon else order_lon


def extract_track_segments(ds: xr.Dataset, var: str, dx_km: float, bbox: tuple):
    """Extract 1D segments from L3 product for spectral analysis.

    Returns list of (values_1d, lats_1d, lons_1d, dx_km) tuples.
    """
    da = crop_to_region(ds, var, bbox)
    lons = da["lon"].values
    lats = da["lat"].values

    if "primary_track" in ds:
        track_da = crop_to_region(ds, "primary_track", bbox)
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

            order = _choose_track_order(lons, lats, xs, ys, dx_km)
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
                    segments.append((sub_clean, seg_lats[a:b], seg_lons[a:b], avg_dx))
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
                    segments.append((sub, seg_lats_sub, np.full(len(sub), lons[j]), avg_dx))

    return segments


def sample_l4_along_tracks(l4_da: xr.DataArray, swot_tracks: list) -> list:
    """Sample a gridded L4 DataArray at SWOT track positions.

    Uses bilinear interpolation so that the L4 field is evaluated at the same
    lat/lon positions as the SWOT passes.  This produces 1D profiles suitable
    for psd_1d, making L4 spectrally comparable to L3 along-track products.

    Args:
        l4_da: 2D xr.DataArray with lon/lat coordinates (output of crop_to_region).
        swot_tracks: list of (values, lats, lons, dx_km) from extract_track_segments.

    Returns:
        list of (sampled_1d_values, dx_km) tuples suitable for psd_1d.
    """
    lats_grid = l4_da["lat"].values.astype(float)
    lons_grid = l4_da["lon"].values.astype(float)
    field = l4_da.values.astype(float)

    # Ensure latitude axis is increasing for searchsorted
    if lats_grid[0] > lats_grid[-1]:
        lats_grid = lats_grid[::-1]
        field = field[::-1, :]

    field = np.where(np.isnan(field), np.nanmean(field), field)

    result = []
    for _, seg_lats, seg_lons, seg_dx in swot_tracks:
        # Nearest-neighbour lookup on the regular L4 grid
        lat_idx = np.searchsorted(lats_grid, seg_lats).clip(0, len(lats_grid) - 1)
        lon_idx = np.searchsorted(lons_grid, seg_lons).clip(0, len(lons_grid) - 1)
        sampled = field[lat_idx, lon_idx]
        valid = np.isfinite(sampled)
        if np.sum(valid) < MIN_SEGMENT_PTS:
            continue
        sampled_clean = np.interp(
            np.arange(len(sampled)), np.where(valid)[0], sampled[valid]
        )
        result.append((sampled_clean, seg_dx))
    return result


# =============================================================================
# Spectral analysis — 1D
# =============================================================================


def psd_1d(values: np.ndarray, dx_km: float):
    """1D PSD with Parseval-consistent normalisation.

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
# PSD averaging
# =============================================================================


def average_psds(psd_list):
    """Compute geometric mean PSD across multiple spectra via log-space interpolation."""
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
# Per-day PSD computation
# =============================================================================


def _run_psd_on_ds(ds, key, var, dx_km, bbox, swot_tracks=None):
    psd_list = []
    if key in ("l3_swot", "l3_ssh"):
        segments = extract_track_segments(ds, var, dx_km, bbox)
        for segment in segments:
            k, psd = psd_1d(segment[0], segment[3])
            if k is not None:
                psd_list.append((k, psd))
    elif swot_tracks:
        da = crop_to_region(ds, var, bbox)
        sampled = sample_l4_along_tracks(da, swot_tracks)
        for seg_vals, seg_dx in sampled:
            k, psd = psd_1d(seg_vals, seg_dx)
            if k is not None:
                psd_list.append((k, psd))
    return psd_list


def process_single_day_remote(dt: datetime, dataset_hf):
    """Process one day loading from HuggingFace."""
    results = {rk: {pk: [] for pk in PRODUCTS} for rk in REGIONS}
    date_str = dt.strftime("%Y-%m-%d")

    for region_key, region_cfg in REGIONS.items():
        bbox = region_cfg["bbox"]

        # Load SWOT first to obtain track geometry for L4 sampling
        swot_tracks = []
        ds_swot = load_bbox_swot_nc(dataset_hf, date_str, bbox)
        if ds_swot is not None:
            _, var_swot, dx_swot, _, _ = PRODUCTS["l3_swot"]
            segments = extract_track_segments(ds_swot, var_swot, dx_swot, bbox)
            swot_tracks = segments
            for segment in segments:
                k, psd = psd_1d(segment[0], segment[3])
                if k is not None:
                    results[region_key]["l3_swot"].append((k, psd))
            ds_swot.close()

        for key, (fname, var, dx_km, _label, _color) in PRODUCTS.items():
            if key == "l3_swot":
                continue  # already handled above
            ds = load_bbox_nc(dataset_hf, date_str, bbox, data_source=fname)
            if ds is None:
                continue
            results[region_key][key].extend(
                _run_psd_on_ds(ds, key, var, dx_km, bbox, swot_tracks=swot_tracks)
            )
            ds.close()

    return results


# =============================================================================
# Main computation
# =============================================================================


def compute_all(dataset_hf, start_date: datetime, n_days: int):
    """Download data, compute PSDs for all regions/products, and load snapshot.

    Args:
        dataset_hf: TacoDataset from load_hf_dataset().
        start_date: First date of the averaging window.
        n_days: Number of days to include.

    Returns:
        Tuple of (averaged_psds, snapshot_data, snapshot_date).
    """
    dates = [start_date + timedelta(days=i) for i in range(n_days)]
    all_psds = {rk: {pk: [] for pk in PRODUCTS} for rk in REGIONS}
    days_with_data = {rk: {pk: 0 for pk in PRODUCTS} for rk in REGIONS}

    print(f"  Processing {n_days} days sequentially ...")
    for di, dt in enumerate(dates):
        print(f"\r  Processing day {di + 1}/{n_days}", end="", flush=True)
        day_results = process_single_day_remote(dt, dataset_hf)
        for rk in REGIONS:
            for pk in PRODUCTS:
                if day_results[rk][pk]:
                    all_psds[rk][pk].extend(day_results[rk][pk])
                    days_with_data[rk][pk] += 1

    print("\n\n  Spectra per region / product:")
    for rk in REGIONS:
        print(f"    {REGIONS[rk]['label']}:")
        for pk in PRODUCTS:
            print(
                f"      {PRODUCTS[pk][3]:30s}: {days_with_data[rk][pk]:3d} days, "
                f"{len(all_psds[rk][pk]):5d} spectra"
            )

    # Snapshot: first date with SWOT data in any region
    print("\n  Auto-selecting snapshot date ...")
    snapshot_date = None
    for dt in dates:
        date_str = dt.strftime("%Y-%m-%d")
        for rk in REGIONS:
            bbox = REGIONS[rk]["bbox"]
            ds_swot = load_bbox_swot_nc(dataset_hf, date_str, bbox)
            if ds_swot is not None:
                segs = extract_track_segments(ds_swot, "ssha_filtered", 2.0, bbox)
                ds_swot.close()
                if segs:
                    snapshot_date = dt
                    break
        if snapshot_date is not None:
            break

    if snapshot_date is None:
        snapshot_date = dates[0]
    print(f"  Snapshot date: {snapshot_date.strftime('%Y-%m-%d')}")

    # Load snapshot data
    snapshot_data = {rk: {} for rk in REGIONS}
    snap_str = snapshot_date.strftime("%Y-%m-%d")
    for rk, region_cfg in REGIONS.items():
        bbox = region_cfg["bbox"]
        for key, (fname, var, _dx_km, _label, _color) in PRODUCTS.items():
            if key == "l3_swot":
                ds = load_bbox_swot_nc(dataset_hf, snap_str, bbox)
            else:
                ds = load_bbox_nc(dataset_hf, snap_str, bbox, data_source=fname)
            if ds is not None:
                # Materialize snapshot values before closing the backing NetCDF file.
                snapshot_data[rk][key] = crop_to_region(ds, var, bbox).load()
                ds.close()

    # Average PSDs per region
    averaged_psds = {rk: {} for rk in REGIONS}
    for rk in REGIONS:
        for pk in PRODUCTS:
            k, psd = average_psds(all_psds[rk][pk])
            if k is not None:
                averaged_psds[rk][pk] = (k, psd)

    return averaged_psds, snapshot_data, snapshot_date


# =============================================================================
# Figure helpers
# =============================================================================


def _plot_map_panel(ax, region_data: dict, region_key: str, date_str: str):
    """Draw SSH snapshot for one region. Returns the last mappable for colorbar."""
    region_cfg = REGIONS[region_key]
    lon_min, lat_min, lon_max, lat_max = region_cfg["bbox"]
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    ax.coastlines(linewidth=0.6, color="#444")
    ax.add_feature(cfeature.LAND, facecolor="#e8e8e8", edgecolor="none", zorder=2)
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.4)
    gl.right_labels = gl.top_labels = False
    gl.xlabel_style = gl.ylabel_style = {"size": 13}

    vmin, vmax = -0.8, 0.8
    mappable = None

    if "l4_ssh" in region_data:
        da = region_data["l4_ssh"]
        vals = da.values.astype(float)
        vals -= np.nanmean(vals)
        lons2d, lats2d = np.meshgrid(da["lon"].values, da["lat"].values)
        mappable = ax.pcolormesh(
            lons2d,
            lats2d,
            vals,
            transform=ccrs.PlateCarree(),
            cmap="RdBu_r",
            vmin=vmin,
            vmax=vmax,
            shading="auto",
            alpha=0.45,
            rasterized=True,
            zorder=1,
        )

    # Make sparse along-track samples visually distinct over the L4 background field.
    for key, sz in [("l3_ssh", 1.2), ("l3_swot", 0.55)]:
        if key not in region_data:
            continue
        da = region_data[key]
        vals = da.values.astype(float)
        vals -= np.nanmean(vals)
        lons2d, lats2d = np.meshgrid(da["lon"].values, da["lat"].values)
        mask = ~np.isnan(vals)
        if np.any(mask):
            sc = ax.scatter(
                lons2d[mask],
                lats2d[mask],
                c=vals[mask],
                s=sz,
                cmap="RdBu_r",
                vmin=vmin,
                vmax=vmax,
                alpha=0.9,
                transform=ccrs.PlateCarree(),
                zorder=3,
                rasterized=True,
            )
            if mappable is None:
                mappable = sc

    ax.set_title(
        f"{region_cfg['label']} - {date_str}", fontsize=13, fontweight="bold"
    )

    return mappable


def _plot_psd_panel(
    ax, region_psds: dict, region_key: str, start_date: datetime, end_date: datetime, output_path: str
):
    """Draw averaged PSD for one region (L3 products only)."""
    for key, (_fname, _var, _dx_km, label, color) in PRODUCTS.items():
        if key == "l4_ssh":
            continue
        if key not in region_psds:
            continue
        k, psd = region_psds[key]
        ax.loglog(1.0 / k, psd, color=color, linewidth=1.8, label=label, alpha=0.9)
        
    ax.set_xlabel("Wavelength (km)", fontsize=12)
    ax.set_ylabel(r"PSD (m$^2$ / cpkm)", fontsize=12)
    ax.set_title(
        f"{REGIONS[region_key]['label']} - {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}",
        fontsize=12,
        fontweight="bold",
    )
    ax.legend(fontsize=11, loc="lower left", framealpha=0.9, edgecolor="none")
    ax.set_xlim(10, 1000)
    ax.tick_params(labelsize=13)
    ax.grid(True, which="both", alpha=0.15)
    ax.invert_xaxis()

    ax_k = ax.secondary_xaxis("top", functions=(lambda x: 1.0 / x, lambda wl: 1.0 / wl))
    ax_k.set_xlabel("Wavenumber (cpkm)", fontsize=11)
    ax_k.tick_params(labelsize=12)


# =============================================================================
# Figure
# =============================================================================


def make_figure(
    averaged_psds,
    snapshot_data,
    snapshot_date,
    start_date,
    end_date,
    output_path,
):
    """Two-region SSH processing-level comparison figure.

    Layout: 2 rows × 2 columns
      Row 0: SSH snapshot maps (Gulf Stream | Kuroshio)
      Row 1: Averaged PSD panels (Gulf Stream | Kuroshio)
    """
    fig = plt.figure(figsize=(14, 9))
    gs = gridspec.GridSpec(
        2,
        2,
        height_ratios=[1.5, 1.0],
        hspace=0.50,
        wspace=0.20,
        left=0.05,
        right=0.97,
        bottom=0.08,
        top=0.93,
    )

    proj = ccrs.Mercator()
    date_str = snapshot_date.strftime("%Y-%m-%d") if snapshot_date else "N/A"
    region_keys = list(REGIONS.keys())

    # --- Row 0: SSH snapshot maps ---
    map_axes = []
    last_mappable = None
    for col, region_key in enumerate(region_keys):
        ax = fig.add_subplot(gs[0, col], projection=proj)
        m = _plot_map_panel(ax, snapshot_data.get(region_key, {}), region_key, date_str)
        if m is not None:
            last_mappable = m
        map_axes.append(ax)

    # Shared colorbar below both map panels
    if last_mappable is not None:
        cb = fig.colorbar(
            last_mappable,
            ax=map_axes,
            orientation="horizontal",
            pad=0.09,
            fraction=0.04,
            aspect=40,
            shrink=0.82,
        )
        cb.set_label("SSH anomaly", fontsize=12)
        cb.ax.tick_params(labelsize=11)

    # --- Row 1: PSD panels ---
    for col, region_key in enumerate(region_keys):
        ax = fig.add_subplot(gs[1, col])
        _plot_psd_panel(
            ax, averaged_psds.get(region_key, {}), region_key, start_date, end_date, output_path
        )

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"\n  Figure saved to {output_path}")
    else:
        plt.show()


def save_psd_data_csv(averaged_psds: dict, output_csv: str):
    """Save plotted PSD curves to CSV for downstream analysis.

    Columns:
      region, product, label, wavenumber_cpkm, wavelength_km, psd_m2_per_cpkm
    """
    rows = []
    for region_key in REGIONS:
        region_psds = averaged_psds.get(region_key, {})
        for key, (_fname, _var, _dx_km, label, _color) in PRODUCTS.items():
            if key not in region_psds:
                continue
            k, psd = region_psds[key]
            for ki, pi in zip(k, psd):
                rows.append(
                    {
                        "region": REGIONS[region_key]["label"],
                        "product": key,
                        "label": label,
                        "wavenumber_cpkm": float(ki),
                        "wavelength_km": float(1.0 / ki),
                        "psd_m2_per_cpkm": float(pi),
                    }
                )

    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "region",
        "product",
        "label",
        "wavenumber_cpkm",
        "wavelength_km",
        "psd_m2_per_cpkm",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"  PSD data saved to {out_path}")


# =============================================================================
# Entry point
# =============================================================================


def main():
    """Parse CLI arguments and run the processing-level comparison figure."""
    parser = argparse.ArgumentParser(
        description="OceanTACO SSH comparison across processing levels"
    )
    parser.add_argument(
        "--hf-url", type=str, default=HF_DEFAULT_URL, help="HuggingFace dataset URL"
    )
    parser.add_argument("--output", type=str, default=None, help="Output figure path")
    parser.add_argument(
        "--psd-data-output",
        type=str,
        default=None,
        help="Optional CSV path to save plotted PSD data",
    )
    parser.add_argument(
        "--start-date", type=str, default="2025-03-01", help="Start date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--n-days", type=int, default=30, help="Number of days to average (default: 30)"
    )
    args = parser.parse_args()

    start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
    end_date = start_date + timedelta(days=args.n_days - 1)

    print("=" * 65)
    print("  OceanTACO — SSH Processing Level Comparison")
    for rk, rcfg in REGIONS.items():
        lon_min, lat_min, lon_max, lat_max = rcfg["bbox"]
        print(f"  {rcfg['label']:20s}: {lon_min}–{lon_max}°, {lat_min}–{lat_max}°N")
    print(f"  Period   : {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print("=" * 65)

    dataset_hf = load_hf_dataset(args.hf_url)

    print("\n[1/2] Computing power spectral densities ...")
    averaged_psds, snapshot_data, snapshot_date = compute_all(
        dataset_hf,
        start_date=start_date,
        n_days=args.n_days,
    )

    print("\n[2/2] Generating figure ...")
    make_figure(
        averaged_psds,
        snapshot_data,
        snapshot_date,
        start_date,
        end_date,
        args.output,
    )

    if args.psd_data_output:
        print("\n[3/3] Saving PSD data ...")
        save_psd_data_csv(averaged_psds, args.psd_data_output)

    print("\nDone.")


if __name__ == "__main__":
    main()
