#!/usr/bin/env python3
"""Plot spatial mosaic of OceanTACO data."""

import argparse
import logging
import os
from pathlib import Path

import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from ocean_taco.dataset.retrieve import (
    HF_DEFAULT_URL,
    load_hf_dataset,
    load_region_product_nc,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def _configure_cartopy_dir(path: str):
    """Configure cartopy data directory."""
    p = Path(path).expanduser()
    p.mkdir(parents=True, exist_ok=True)
    os.environ["CARTOPY_USER_DIR"] = str(p)
    cartopy.config["data_dir"] = str(p)


_configure_cartopy_dir("./.cartopy")

# ═══════════════════════════════════════════════════════════════════════
# Region definitions (mirrors new_format_ssh_data.py)
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

# Product loading specifications
PRODUCTS = {
    "glorys": {
        "subdir": "glorys",
        "filename": "glorys_{region}_{date}.nc",
        "var_candidates": ["zos", "ssh"],
        "label": "GLORYS12 Reanalysis",
        "resolution": "~8.3 km (1/12°)",
        "res_km": 8.3,
    },
    "l4_ssh": {
        "subdir": "l4_ssh",
        "filename": "l4_ssh_{region}_{date}.nc",
        "var_candidates": ["adt", "sla", "ssh"],
        "label": "DUACS L4 (allsat)",
        "resolution": "~13.9 km (1/8°)",
        "res_km": 13.9,
    },
    "l3_ssh": {
        "subdir": "l3_ssh",
        "filename": "l3_ssh_{region}_{date}.nc",
        "var_candidates": ["sla_filtered", "sla"],
        "label": "L3 Along-Track SLA",
        "resolution": "~7 km gridded",
        "res_km": 7.0,
    },
    "l3_swot": {
        "subdir": "l3_swot",
        "filename": "l3_swot_{region}_{date}.nc",
        "var_candidates": ["ssha_filtered", "ssha_unfiltered"],
        "label": "SWOT L3 KaRIn",
        "resolution": "~2 km gridded",
        "res_km": 2.0,
    },
}


# ═══════════════════════════════════════════════════════════════════════
# Data Loading
# ═══════════════════════════════════════════════════════════════════════


def load_product_remote(dataset, key, region, date_str, cache_dir=None):
    """Load a single product NetCDF from HuggingFace remote.

    Args:
        dataset: TacoDataset from load_hf_dataset().
        key: Product key (e.g. 'glorys', 'l4_ssh').
        region: Region name.
        date_str: Date string YYYYMMDD.
        cache_dir: Optional local cache directory.

    Returns:
        (xr.Dataset, variable_name) or (None, None).
    """
    spec = PRODUCTS[key]
    date_dash = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
    ds = load_region_product_nc(dataset, date_dash, region, key, cache_dir)

    if ds is None:
        logging.warning(f"  [{key}] Not found in remote catalog")
        return None, None

    for vc in spec["var_candidates"]:
        if vc in ds.data_vars:
            return ds, vc

    # Fallback — first variable with ssh/sla in name
    for v in ds.data_vars:
        if any(k in v.lower() for k in ("ssh", "sla", "adt", "zos")):
            return ds, v

    logging.warning(f"  [{key}] No SSH variable found. Available: {list(ds.data_vars)}")
    ds.close()
    return None, None


def get_coords(ds):
    """Extract 1-D lon/lat arrays from dataset."""
    return ds["lon"].values, ds["lat"].values


def subset_ds(ds, var_name, lon_range, lat_range):
    """Subset a dataset to a bounding box. Returns lon, lat, data arrays."""
    lon, lat = get_coords(ds)

    if lon.ndim == 1:
        lon_mask = (lon >= lon_range[0]) & (lon <= lon_range[1])
        lat_mask = (lat >= lat_range[0]) & (lat <= lat_range[1])
        if not (lon_mask.any() and lat_mask.any()):
            return None, None, None
        data = ds[var_name].values
        if data.ndim > 2:
            data = np.squeeze(data)
        return lon[lon_mask], lat[lat_mask], data[np.ix_(lat_mask, lon_mask)]
    else:
        # 2D coords
        mask = (
            (lon >= lon_range[0])
            & (lon <= lon_range[1])
            & (lat >= lat_range[0])
            & (lat <= lat_range[1])
        )
        if not mask.any():
            return None, None, None
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        data = ds[var_name].values
        if data.ndim > 2:
            data = np.squeeze(data)
        return (
            lon[np.ix_(rows, cols)],
            lat[np.ix_(rows, cols)],
            data[np.ix_(rows, cols)],
        )


# ═══════════════════════════════════════════════════════════════════════
# Auto-detect SWOT data cluster for zoom window
# ═══════════════════════════════════════════════════════════════════════


def find_swot_cluster(ds, var_name, min_size_deg=5.0, pad_deg=2.0):
    """Locate the densest cluster of SWOT data and return a bounding box.

    Strategy: find the connected region with the most non-NaN pixels,
    then pad it to create a visually appealing zoom window.

    Returns (lon_min, lon_max, lat_min, lat_max) or None.
    """
    lon, lat = get_coords(ds)
    data = ds[var_name].values
    if data.ndim > 2:
        data = np.squeeze(data)

    valid = np.isfinite(data)
    if valid.sum() == 0:
        return None

    # Use n_obs if available (more reliable coverage indicator)
    if "n_obs" in ds:
        valid = ds["n_obs"].values > 0

    # Find bounding box of all valid data
    valid_rows = np.any(valid, axis=1)
    valid_cols = np.any(valid, axis=0)

    if not (valid_rows.any() and valid_cols.any()):
        return None

    row_indices = np.where(valid_rows)[0]
    col_indices = np.where(valid_cols)[0]

    # For sparse track data, find the densest vertical strip
    # (SWOT tracks are roughly N-S oriented)
    if valid.sum() / valid.size < 0.05:
        # Very sparse — find the column range with highest density
        col_density = valid.sum(axis=0)  # sum along lat for each lon column
        if col_density.max() == 0:
            return None

        # Sliding window to find densest lon band
        window_size = max(1, int(min_size_deg / np.abs(np.median(np.diff(lon)))))
        if window_size >= len(lon):
            best_start = 0
            best_end = len(lon) - 1
        else:
            convolved = np.convolve(col_density, np.ones(window_size), mode="valid")
            best_start = np.argmax(convolved)
            best_end = best_start + window_size

        # Find lat range within this lon strip
        strip_valid = valid[:, best_start:best_end]
        strip_rows = np.any(strip_valid, axis=1)
        if not strip_rows.any():
            return None
        row_indices = np.where(strip_rows)[0]
        col_indices = np.arange(best_start, best_end)

    lat_min = float(lat[row_indices[0]])
    lat_max = float(lat[row_indices[-1]])
    lon_min = float(lon[col_indices[0]])
    lon_max = float(lon[col_indices[-1]])

    # Ensure minimum size
    lat_span = lat_max - lat_min
    lon_span = lon_max - lon_min
    if lat_span < min_size_deg:
        center = (lat_min + lat_max) / 2
        lat_min = center - min_size_deg / 2
        lat_max = center + min_size_deg / 2
    if lon_span < min_size_deg:
        center = (lon_min + lon_max) / 2
        lon_min = center - min_size_deg / 2
        lon_max = center + min_size_deg / 2

    # Pad
    return (lon_min - pad_deg, lon_max + pad_deg, lat_min - pad_deg, lat_max + pad_deg)


# ═══════════════════════════════════════════════════════════════════════
# Spectral analysis — resolution comparison
# ═══════════════════════════════════════════════════════════════════════


def compute_zonal_spectrum(lon, lat, data):
    """Compute mean zonal power spectral density.

    Returns wavelength_km, psd arrays (averaged over all valid latitude bands).
    """
    if data.ndim != 2 or data.shape[0] < 4 or data.shape[1] < 8:
        return None, None

    # Resolution in km (approximate at mid-latitude)
    mid_lat = np.mean(lat)
    if lon.ndim == 1:
        dx_deg = np.abs(np.median(np.diff(lon)))
    else:
        dx_deg = np.abs(np.median(np.diff(lon, axis=1)))
    dx_km = dx_deg * 111.32 * np.cos(np.radians(mid_lat))

    if dx_km <= 0:
        return None, None

    n_lon = data.shape[1]
    spectra = []

    for i in range(data.shape[0]):
        row = data[i, :]
        valid = np.isfinite(row)
        # Need at least 50% valid data in the row for a clean spectrum
        if valid.sum() < n_lon * 0.5:
            continue
        # Fill small gaps with linear interpolation for FFT
        if not valid.all():
            x = np.arange(n_lon)
            row = np.interp(x, x[valid], row[valid])

        # Detrend
        row = row - np.polyval(np.polyfit(np.arange(n_lon), row, 1), np.arange(n_lon))
        # Hann window
        window = np.hanning(n_lon)
        windowed = row * window

        fft_vals = np.fft.rfft(windowed)
        psd = (2.0 / (n_lon * dx_km)) * np.abs(fft_vals) ** 2
        spectra.append(psd)

    if not spectra:
        return None, None

    mean_psd = np.mean(spectra, axis=0)
    freqs = np.fft.rfftfreq(n_lon, d=dx_km)

    # Skip DC component
    freqs = freqs[1:]
    mean_psd = mean_psd[1:]

    # Convert to wavelength
    with np.errstate(divide="ignore"):
        wavelength_km = 1.0 / freqs

    return wavelength_km, mean_psd


# ═══════════════════════════════════════════════════════════════════════
# Plotting helpers
# ═══════════════════════════════════════════════════════════════════════


def add_map_features(ax, extent, gridline_step=5):
    """Add coastlines, gridlines, land to a cartopy axis."""
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor="#f0ece3", edgecolor="none", zorder=2)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.6, edgecolor="#4a4a4a", zorder=3)
    gl = ax.gridlines(
        draw_labels=True,
        linewidth=0.3,
        color="#999999",
        alpha=0.4,
        linestyle="-",
        zorder=1,
    )
    gl.top_labels = False
    gl.right_labels = False
    gl.xlocator = mticker.MultipleLocator(gridline_step)
    gl.ylocator = mticker.MultipleLocator(gridline_step)
    gl.xlabel_style = {"fontsize": 7, "color": "#555555"}
    gl.ylabel_style = {"fontsize": 7, "color": "#555555"}
    return gl


def plot_ssh_panel(
    ax,
    lon,
    lat,
    data,
    extent,
    title,
    resolution_str,
    norm,
    cmap,
    gridline_step=5,
    show_colorbar=False,
):
    """Plot a single SSH panel on a cartopy axis."""
    add_map_features(ax, extent, gridline_step)

    if lon is not None and data is not None:
        pcm = ax.pcolormesh(
            lon,
            lat,
            data,
            transform=ccrs.PlateCarree(),
            cmap=cmap,
            norm=norm,
            shading="auto",
            rasterized=True,
            zorder=1.5,
        )

        # Coverage stats
        total = data.size
        valid = np.count_nonzero(np.isfinite(data))
        coverage = 100.0 * valid / total if total > 0 else 0

        if lon.ndim == 1:
            grid_str = f"{len(lat)}×{len(lon)}"
        else:
            grid_str = f"{data.shape[0]}×{data.shape[1]}"

        info_text = f"{resolution_str}  •  {grid_str} px  •  {coverage:.0f}% coverage"
    else:
        pcm = None
        info_text = "No data available"

    # Title styling
    ax.set_title(title, fontsize=10, fontweight="bold", color="#222222", pad=4)
    ax.text(
        0.5,
        -0.02,
        info_text,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=7,
        color="#777777",
        fontstyle="italic",
    )

    return pcm


# ═══════════════════════════════════════════════════════════════════════
# Main figure builder
# ═══════════════════════════════════════════════════════════════════════


def plot_resolution_mosaic(
    dataset,
    region,
    date_str,
    cache_dir=None,
    zoom_lon=None,
    zoom_lat=None,
    vmin=-0.15,
    vmax=0.15,
    cmap="RdBu_r",
    output_path="mosaic_ssh_resolution.pdf",
    dpi=300,
):
    """Create the full resolution comparison figure.

    Layout:
        Top-left:     Overview (L4 SSH, full region) with zoom box
        Top-right:    Zoom panels for GLORYS and L4 (stacked vertically)
        Bottom-left:  Zoom panel for SWOT (large, to show detail)
        Bottom-right: Zoom panels for L3 SSH + power spectra
    """
    # ── Load all products ────────────────────────────────────────────
    datasets = {}
    for key in PRODUCTS:
        ds, var = load_product_remote(dataset, key, region, date_str, cache_dir)
        datasets[key] = (ds, var)
        if ds is not None:
            logging.info(f"  Loaded {key}: var={var}, shape={ds[var].squeeze().shape}")

    # ── Determine zoom window ────────────────────────────────────────
    if zoom_lon is not None and zoom_lat is not None:
        zoom_extent = (zoom_lon[0], zoom_lon[1], zoom_lat[0], zoom_lat[1])
        logging.info(f"  Using user-specified zoom: {zoom_extent}")
    else:
        # Auto-detect from SWOT data
        swot_ds, swot_var = datasets.get("l3_swot", (None, None))
        if swot_ds is not None:
            bbox = find_swot_cluster(swot_ds, swot_var, min_size_deg=8.0, pad_deg=1.0)
            if bbox is not None:
                zoom_extent = bbox
                logging.info(f"  Auto-detected SWOT cluster zoom: {zoom_extent}")
            else:
                logging.warning("  No SWOT cluster found, using region center")
                rb = SPATIAL_REGIONS[region]
                cx = (rb["lon"][0] + rb["lon"][1]) / 2
                cy = (rb["lat"][0] + rb["lat"][1]) / 2
                zoom_extent = (cx - 10, cx + 10, cy - 8, cy + 8)
        else:
            rb = SPATIAL_REGIONS[region]
            cx = (rb["lon"][0] + rb["lon"][1]) / 2
            cy = (rb["lat"][0] + rb["lat"][1]) / 2
            zoom_extent = (cx - 10, cx + 10, cy - 8, cy + 8)

    zoom_lon_range = (zoom_extent[0], zoom_extent[1])
    zoom_lat_range = (zoom_extent[2], zoom_extent[3])

    # Full region extent for overview
    rb = SPATIAL_REGIONS[region]
    overview_extent = (rb["lon"][0], rb["lon"][1], rb["lat"][0], rb["lat"][1])

    # ── Subset data to zoom window ───────────────────────────────────
    zoom_data = {}
    for key, (ds, var) in datasets.items():
        if ds is None:
            zoom_data[key] = (None, None, None)
            continue
        lon, lat, data = subset_ds(ds, var, zoom_lon_range, zoom_lat_range)
        if lon is not None:
            # Compute anomaly (subtract spatial mean)
            mean_val = np.nanmean(data)
            if np.isfinite(mean_val):
                data = data - mean_val
            zoom_data[key] = (lon, lat, data)
            logging.info(
                f"  {key} zoom: {data.shape}, "
                f"range [{np.nanmin(data):.3f}, {np.nanmax(data):.3f}]"
            )
        else:
            zoom_data[key] = (None, None, None)

    # ── Colour normalisation ─────────────────────────────────────────
    norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)

    # ── Figure layout ────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 12))

    # Custom GridSpec: 2 rows, 2 columns
    # Left column wider for overview and SWOT detail
    gs = gridspec.GridSpec(
        2,
        2,
        width_ratios=[1.2, 1.0],
        height_ratios=[1.0, 1.0],
        hspace=0.22,
        wspace=0.12,
        left=0.05,
        right=0.92,
        top=0.92,
        bottom=0.08,
    )

    # Right column: split each cell into 2 sub-rows
    gs_right_top = gridspec.GridSpecFromSubplotSpec(
        2, 1, subplot_spec=gs[0, 1], hspace=0.30
    )
    gs_right_bot = gridspec.GridSpecFromSubplotSpec(
        2, 1, subplot_spec=gs[1, 1], hspace=0.30
    )

    proj = ccrs.PlateCarree()

    # ── Panel A: Overview (L4 SSH, full region) ──────────────────────
    ax_overview = fig.add_subplot(gs[0, 0], projection=proj)
    l4_ds, l4_var = datasets.get("l4_ssh", (None, None))
    if l4_ds is not None:
        l4_full_data = l4_ds[l4_var].squeeze().values
        l4_lon, l4_lat = get_coords(l4_ds)
        # Anomaly for overview
        overview_mean = np.nanmean(l4_full_data)
        if np.isfinite(overview_mean):
            l4_overview = l4_full_data - overview_mean
        else:
            l4_overview = l4_full_data
        plot_ssh_panel(
            ax_overview,
            l4_lon,
            l4_lat,
            l4_overview,
            overview_extent,
            "A) Overview — DUACS L4 SSH",
            PRODUCTS["l4_ssh"]["resolution"],
            norm,
            cmap,
            gridline_step=15,
        )
    else:
        add_map_features(ax_overview, overview_extent, gridline_step=15)
        ax_overview.set_title(
            "A) Overview — DUACS L4 SSH\n(data not found)",
            fontsize=10,
            fontweight="bold",
        )

    # Draw zoom box on overview
    rect = mpatches.Rectangle(
        (zoom_extent[0], zoom_extent[2]),
        zoom_extent[1] - zoom_extent[0],
        zoom_extent[3] - zoom_extent[2],
        linewidth=2.0,
        edgecolor="#e74c3c",
        facecolor="none",
        transform=proj,
        zorder=10,
        linestyle="-",
    )
    ax_overview.add_patch(rect)
    ax_overview.text(
        zoom_extent[0],
        zoom_extent[3] + 0.5,
        "ZOOM",
        transform=proj,
        fontsize=8,
        fontweight="bold",
        color="#e74c3c",
        ha="left",
        va="bottom",
        zorder=11,
    )

    # ── Panel B: GLORYS zoom ─────────────────────────────────────────
    ax_glorys = fig.add_subplot(gs_right_top[0], projection=proj)
    lon, lat, data = zoom_data.get("glorys", (None, None, None))
    plot_ssh_panel(
        ax_glorys,
        lon,
        lat,
        data,
        zoom_extent,
        "B) GLORYS Reanalysis",
        PRODUCTS["glorys"]["resolution"],
        norm,
        cmap,
        gridline_step=5,
    )

    # ── Panel C: L4 SSH zoom ─────────────────────────────────────────
    ax_l4 = fig.add_subplot(gs_right_top[1], projection=proj)
    lon, lat, data = zoom_data.get("l4_ssh", (None, None, None))
    plot_ssh_panel(
        ax_l4,
        lon,
        lat,
        data,
        zoom_extent,
        "C) DUACS L4 SSH",
        PRODUCTS["l4_ssh"]["resolution"],
        norm,
        cmap,
        gridline_step=5,
    )

    # ── Panel D: SWOT zoom (large panel) ─────────────────────────────
    ax_swot = fig.add_subplot(gs[1, 0], projection=proj)
    lon, lat, data = zoom_data.get("l3_swot", (None, None, None))
    plot_ssh_panel(
        ax_swot,
        lon,
        lat,
        data,
        zoom_extent,
        "D) SWOT L3 KaRIn SSH",
        PRODUCTS["l3_swot"]["resolution"],
        norm,
        cmap,
        gridline_step=5,
    )

    # ── Panel E: L3 along-track zoom ─────────────────────────────────
    ax_l3 = fig.add_subplot(gs_right_bot[0], projection=proj)
    lon, lat, data = zoom_data.get("l3_ssh", (None, None, None))
    plot_ssh_panel(
        ax_l3,
        lon,
        lat,
        data,
        zoom_extent,
        "E) L3 Along-Track SLA",
        PRODUCTS["l3_ssh"]["resolution"],
        norm,
        cmap,
        gridline_step=5,
    )

    # ── Panel F: Power spectra comparison ────────────────────────────
    ax_spec = fig.add_subplot(gs_right_bot[1])

    spec_colors = {
        "l3_swot": ("#e74c3c", "SWOT (2 km)"),
        "l3_ssh": ("#2ecc71", "L3 SSH (7 km)"),
        "glorys": ("#3498db", "GLORYS (8.3 km)"),
        "l4_ssh": ("#9b59b6", "L4 SSH (13.9 km)"),
    }

    has_spectrum = False
    for key in ["l4_ssh", "glorys", "l3_ssh", "l3_swot"]:
        lon, lat, data = zoom_data.get(key, (None, None, None))
        if data is None:
            continue
        wl, psd = compute_zonal_spectrum(lon, lat, data)
        if wl is not None and len(wl) > 2:
            color, label = spec_colors[key]
            ax_spec.loglog(wl, psd, color=color, label=label, linewidth=1.5, alpha=0.85)
            has_spectrum = True

    if has_spectrum:
        ax_spec.set_xlabel("Wavelength (km)", fontsize=9)
        ax_spec.set_ylabel("PSD (m² / cpkm)", fontsize=9)
        ax_spec.set_title(
            "F) Zonal Power Spectra", fontsize=10, fontweight="bold", pad=4
        )
        ax_spec.legend(fontsize=7, loc="upper right", framealpha=0.9)
        ax_spec.grid(True, alpha=0.2, which="both")
        ax_spec.set_xlim(5, 2000)
        ax_spec.tick_params(labelsize=7)
        # Add effective resolution markers (Nyquist = 2 × grid spacing)
        for res_km, color in [
            (2, "#e74c3c"),
            (7, "#2ecc71"),
            (8.3, "#3498db"),
            (13.9, "#9b59b6"),
        ]:
            ax_spec.axvline(
                2 * res_km, color=color, linestyle=":", alpha=0.4, linewidth=0.8
            )
    else:
        ax_spec.text(
            0.5,
            0.5,
            "Insufficient data\nfor spectral analysis",
            transform=ax_spec.transAxes,
            ha="center",
            va="center",
            fontsize=11,
            color="#999999",
        )
        ax_spec.set_title("F) Zonal Power Spectra", fontsize=10, fontweight="bold")

    # ── Shared colorbar ──────────────────────────────────────────────
    cbar_ax = fig.add_axes([0.93, 0.15, 0.015, 0.70])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation="vertical", extend="both")
    cbar.set_label("SSH Anomaly (m)", fontsize=10, labelpad=8)
    cbar.ax.tick_params(labelsize=8)

    # ── Super title ──────────────────────────────────────────────────
    date_fmt = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
    fig.suptitle(
        f"Multi-Product SSH Resolution Comparison — "
        f"{region.replace('_', ' ')}  •  {date_fmt}",
        fontsize=14,
        fontweight="bold",
        y=0.97,
        color="#222222",
    )

    # ── Save ─────────────────────────────────────────────────────────
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    logging.info(f"✓ Figure saved: {output_path}")
    plt.close(fig)

    # ── Close datasets ───────────────────────────────────────────────
    for ds, _ in datasets.values():
        if ds is not None:
            ds.close()

    # ── Print summary ────────────────────────────────────────────────
    print(f"\n{'═' * 62}")
    print("  MOSAIC SUMMARY")
    print(f"{'═' * 62}")
    print(f"  Region     : {region}")
    print(f"  Date       : {date_fmt}")
    print(
        f"  Zoom       : lon [{zoom_extent[0]:.1f}, {zoom_extent[1]:.1f}]  "
        f"lat [{zoom_extent[2]:.1f}, {zoom_extent[3]:.1f}]"
    )
    print(f"  Output     : {output_path}")
    for key in ["glorys", "l4_ssh", "l3_ssh", "l3_swot"]:
        lon, lat, data = zoom_data.get(key, (None, None, None))
        if data is not None:
            valid_pct = 100.0 * np.count_nonzero(np.isfinite(data)) / data.size
            print(
                f"  {key:10s} : {data.shape[0]:>5d}×{data.shape[1]:<5d}  "
                f"({valid_pct:5.1f}% valid)"
            )
        else:
            print(f"  {key:10s} : —")
    print(f"{'═' * 62}")


# ═══════════════════════════════════════════════════════════════════════
# CLI entry point
# ═══════════════════════════════════════════════════════════════════════


def main():
    """Entry point for SSH resolution mosaic plot."""
    parser = argparse.ArgumentParser(
        description="SSH resolution comparison mosaic (overview + zoom).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--hf-url",
        default=HF_DEFAULT_URL,
        help="HuggingFace dataset URL.",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Local cache directory for downloaded files.",
    )
    parser.add_argument(
        "--region", default="NORTH_ATLANTIC", choices=list(SPATIAL_REGIONS.keys())
    )
    parser.add_argument(
        "--date", default="20230330", help="Date (YYYYMMDD or YYYY-MM-DD)."
    )
    parser.add_argument(
        "--zoom-lon",
        nargs=2,
        type=float,
        default=None,
        help="Zoom longitude range [min max].",
    )
    parser.add_argument(
        "--zoom-lat",
        nargs=2,
        type=float,
        default=None,
        help="Zoom latitude range [min max].",
    )
    parser.add_argument("--vmin", type=float, default=-0.15)
    parser.add_argument("--vmax", type=float, default=0.15)
    parser.add_argument("--cmap", default="RdBu_r")
    parser.add_argument("--output", default=None)
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    date_str = args.date.replace("-", "")
    output_path = args.output or f"mosaic_resolution_{args.region}_{date_str}.pdf"

    dataset = load_hf_dataset(args.hf_url)

    plot_resolution_mosaic(
        dataset=dataset,
        region=args.region,
        date_str=date_str,
        cache_dir=args.cache_dir,
        zoom_lon=args.zoom_lon,
        zoom_lat=args.zoom_lat,
        vmin=args.vmin,
        vmax=args.vmax,
        cmap=args.cmap,
        output_path=output_path,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
