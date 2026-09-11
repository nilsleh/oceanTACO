#!/usr/bin/env python3
"""Plot cross-product of SSH products."""

import argparse
import os
import warnings
from datetime import datetime
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
from scipy.stats import pearsonr

from ocean_taco.dataset.retrieve import HF_DEFAULT_URL, load_hf_dataset, load_tile_nc

warnings.filterwarnings("ignore", category=RuntimeWarning)


def _configure_cartopy_dir(path: str):
    """Configure cartopy data directory."""
    p = Path(path).expanduser()
    p.mkdir(parents=True, exist_ok=True)
    os.environ["CARTOPY_USER_DIR"] = str(p)
    cartopy.config["data_dir"] = str(p)


_configure_cartopy_dir("./.cartopy")


# =============================================================================
# Configuration
# =============================================================================

LON_MIN, LON_MAX = 130, 160
LAT_MIN, LAT_MAX = 25, 50

TILE = "NORTH_PACIFIC_EAST"

# Transect longitude (cuts across the Kuroshio jet)
TRANSECT_LON = 145.0

PRODUCTS = {
    "l3_swot": ("l3_swot", "ssha_filtered", "L3 SWOT (2 km)", "#e63946"),
    "l3_ssh": ("l3_ssh", "sla_filtered", "L3 along-track (7 km)", "#f4a261"),
    "l4_ssh": ("l4_ssh", "sla", "L4 DUACS (13.3 km)", "#264653"),
}


# =============================================================================
# I/O
# =============================================================================


def load_and_crop_remote(
    dataset_hf, dt: datetime, key: str, cache_dir=None
) -> xr.DataArray | None:
    """Load one product from HuggingFace, crop to region, squeeze, mean-remove."""
    fname, var, label, color = PRODUCTS[key]
    date_str = dt.strftime("%Y-%m-%d")
    ds = load_tile_nc(dataset_hf, date_str, TILE, fname, cache_dir)
    if ds is None:
        return None

    lats = ds["lat"].values
    lat_sl = slice(LAT_MAX, LAT_MIN) if lats[0] > lats[-1] else slice(LAT_MIN, LAT_MAX)

    da = ds[var].sel(lon=slice(LON_MIN, LON_MAX), lat=lat_sl)
    if "time" in da.dims:
        da = da.isel(time=0)
    if "depth" in da.dims:
        da = da.isel(depth=0)
    da = da.squeeze()

    # Mean-remove so different references are comparable
    da = da - float(da.mean(skipna=True))

    ds.close()
    return da


# =============================================================================
# Transect extraction
# =============================================================================


def extract_transect(da, transect_lon, tol_deg=0.5):
    """Extract a meridional transect at a given longitude.
    For sparse L3 products, use a longitude band of width ±tol_deg.
    Returns (latitudes, values) with NaN where no data.
    """
    vals = da.values
    lons = da["lon"].values
    lats = da["lat"].values

    # Check data density near transect
    lon_idx = np.argmin(np.abs(lons - transect_lon))
    col = vals[:, lon_idx] if vals.ndim == 2 else vals
    frac_valid = np.sum(~np.isnan(col)) / len(col) if vals.ndim == 2 else 1.0

    if frac_valid > 0.5:
        # Dense product: just take nearest longitude
        if vals.ndim == 2:
            return lats, vals[:, lon_idx]
        else:
            return lats, col

    # Sparse product: average over a longitude band
    lon_mask = (lons >= transect_lon - tol_deg) & (lons <= transect_lon + tol_deg)
    if vals.ndim == 2:
        band = vals[:, lon_mask]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            profile = np.nanmean(band, axis=1)
        return lats, profile

    return lats, col


# =============================================================================
# Scatter data: interpolate gridded product onto L3 positions
# =============================================================================


def scatter_data(da_gridded, da_l3):
    """Interpolate a gridded product onto the non-NaN positions of an L3 product.
    Returns (l3_values, gridded_values) — matched pairs, NaN-free.
    """
    g_lons = da_gridded["lon"].values
    g_lats = da_gridded["lat"].values
    g_vals = da_gridded.values.astype(float)

    # Ensure ascending lat for interpolator
    if g_lats[0] > g_lats[-1]:
        g_lats = g_lats[::-1]
        g_vals = g_vals[::-1, :]

    g_vals_filled = g_vals.copy()
    g_vals_filled[np.isnan(g_vals_filled)] = 0.0

    interp = RegularGridInterpolator(
        (g_lats, g_lons),
        g_vals_filled,
        method="linear",
        bounds_error=False,
        fill_value=np.nan,
    )

    l3_vals = da_l3.values.astype(float)
    l3_lons = da_l3["lon"].values
    l3_lats = da_l3["lat"].values

    lon_grid, lat_grid = np.meshgrid(l3_lons, l3_lats)
    l3_flat = l3_vals.ravel()
    pts = np.column_stack([lat_grid.ravel(), lon_grid.ravel()])

    # Only where L3 has data
    valid = ~np.isnan(l3_flat)
    if np.sum(valid) < 10:
        return None, None

    g_interp = interp(pts[valid])
    l3_valid = l3_flat[valid]

    # Remove any remaining NaN from interpolation edges
    both_good = ~np.isnan(g_interp) & ~np.isnan(l3_valid)
    return l3_valid[both_good], g_interp[both_good]


# =============================================================================
# Figure
# =============================================================================


def make_figure(data: dict, date_str: str, output_path: str):
    fig = plt.figure(figsize=(12, 4.8))
    gs = gridspec.GridSpec(
        1,
        2,
        width_ratios=[1.25, 1.0],
        wspace=0.12,
        left=0.06,
        right=0.99,
        bottom=0.10,
        top=0.88,
    )

    # =========================================================================
    # Panel 1: SSH snapshot map
    # =========================================================================
    ax_map = fig.add_subplot(gs[0], projection=ccrs.Mercator())
    ax_map.set_extent([LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], crs=ccrs.PlateCarree())
    ax_map.coastlines(linewidth=0.6, color="#444")
    ax_map.add_feature(cfeature.LAND, facecolor="#e8e8e8", edgecolor="none", zorder=2)
    gl = ax_map.gridlines(draw_labels=True, linewidth=0.3, alpha=0.4)
    gl.right_labels = gl.top_labels = False
    gl.xlabel_style = gl.ylabel_style = {"size": 11}

    vmin, vmax = -0.4, 0.4
    mappable = None

    # L4 as background
    if "l4_ssh" in data:
        da = data["l4_ssh"]
        lons2d, lats2d = np.meshgrid(da["lon"].values, da["lat"].values)
        mappable = ax_map.pcolormesh(
            lons2d,
            lats2d,
            da.values,
            transform=ccrs.PlateCarree(),
            cmap="RdBu_r",
            vmin=vmin,
            vmax=vmax,
            shading="auto",
            alpha=0.6,
            rasterized=True,
            zorder=1,
        )

    # L3 products overlaid
    for key, sz in [("l3_ssh", 0.3), ("l3_swot", 0.05)]:
        if key not in data:
            continue
        da = data[key]
        vals = da.values
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
            pad=0.05,
            fraction=0.03,
            aspect=35,
        )
        cb.set_label("SSH anomaly (m, mean-removed)", fontsize=11)
        cb.ax.tick_params(labelsize=10)

    ax_map.set_title(f"SSH snapshot on {date_str}", fontsize=14)

    # Legend
    legend_items = []
    if "l4_ssh" in data:
        legend_items.append(
            Line2D(
                [0], [0], color=PRODUCTS["l4_ssh"][3], lw=6, alpha=0.6, label="L4 DUACS"
            )
        )
    if "l3_ssh" in data:
        legend_items.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=PRODUCTS["l3_ssh"][3],
                markersize=4,
                label="L3 along-track",
            )
        )
    if "l3_swot" in data:
        legend_items.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=PRODUCTS["l3_swot"][3],
                markersize=4,
                label="L3 SWOT",
            )
        )
    if legend_items:
        ax_map.legend(
            handles=legend_items,
            loc="lower left",
            fontsize=11,
            framealpha=0.9,
            edgecolor="none",
        )

    # =========================================================================
    # Panel 2: Scatter — L4 vs L3 along-track and L4 vs SWOT
    # =========================================================================
    ax_sc = fig.add_subplot(gs[1])

    scatter_configs = []
    if "l4_ssh" in data and "l3_ssh" in data:
        scatter_configs.append(
            (
                "l4_ssh",
                "l3_ssh",
                PRODUCTS["l3_ssh"][3],  # orange — along-track color
                "L4 vs along-track",
                "o",
            )
        )
    if "l4_ssh" in data and "l3_swot" in data:
        scatter_configs.append(
            (
                "l4_ssh",
                "l3_swot",
                PRODUCTS["l3_swot"][3],  # red — SWOT color
                "L4 vs SWOT",
                "o",
            )
        )

    for gkey, l3key, color, label, marker in scatter_configs:
        l3_vals, g_vals = scatter_data(data[gkey], data[l3key])
        if l3_vals is None:
            continue

        # Subsample for plotting
        n = len(l3_vals)
        if n > 5000:
            idx = np.random.default_rng(42).choice(n, 5000, replace=False)
            l3_plot, g_plot = l3_vals[idx], g_vals[idx]
        else:
            l3_plot, g_plot = l3_vals, g_vals

        r, _ = pearsonr(l3_vals, g_vals)
        rmse = np.sqrt(np.mean((l3_vals - g_vals) ** 2))

        ax_sc.scatter(
            l3_plot,
            g_plot,
            s=1.0,
            alpha=0.3,
            color=color,
            marker=marker,
            rasterized=True,
            label=f"{label}  (r={r:.3f}, RMSE={rmse:.3f} m)",
        )

    # 1:1 line
    lim = max(
        abs(ax_sc.get_xlim()[0]),
        abs(ax_sc.get_xlim()[1]),
        abs(ax_sc.get_ylim()[0]),
        abs(ax_sc.get_ylim()[1]),
    )
    lim = min(lim, 0.6)
    ax_sc.plot([-lim, lim], [-lim, lim], "k--", linewidth=0.8, alpha=0.4, zorder=0)
    ax_sc.set_xlim(-lim, lim)
    ax_sc.set_ylim(-lim, lim)
    ax_sc.set_aspect("auto")

    ax_sc.set_xlabel("L3 observation (m)", fontsize=13)
    ax_sc.set_ylabel("L4 DUACS (m)", fontsize=13)
    ax_sc.set_title("L4 DUACS vs L3 observations", fontsize=14)
    ax_sc.legend(
        fontsize=11,
        loc="upper left",
        framealpha=0.9,
        edgecolor="none",
        markerscale=4,
        handletextpad=0.5,
    )
    ax_sc.grid(True, alpha=0.2)
    ax_sc.tick_params(labelsize=11)

    fig.suptitle(
        f"SSH cross-product comparison  on Kuroshio Current  on {date_str}", fontsize=16
    )
    plt.tight_layout(pad=0.3)
    plt.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"  Figure saved to {output_path}")
    plt.show()


# =============================================================================
# Entry point
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="OceanTACO SSH cross-product comparison"
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
        help="Local cache directory for downloaded files",
    )
    parser.add_argument(
        "--date", type=str, default="2025-03-01", help="Date to plot (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--output", type=str, default="ssh_cross_product.png", help="Output figure path"
    )
    args = parser.parse_args()

    dataset_hf = load_hf_dataset(args.hf_url)
    dt = datetime.strptime(args.date, "%Y-%m-%d")
    date_str = args.date

    print("=" * 55)
    print("  OceanTACO — SSH Cross-Product Comparison")
    print(f"  Date     : {date_str}")
    print(f"  Region   : Kuroshio ({LON_MIN}–{LON_MAX}°E, {LAT_MIN}–{LAT_MAX}°N)")
    print("=" * 55)

    # Load all products for this date
    print("\n  Loading products ...")
    data = {}
    for key in PRODUCTS:
        da = load_and_crop_remote(dataset_hf, dt, key, args.cache_dir)
        if da is not None:
            data[key] = da
            print(f"    {PRODUCTS[key][2]:30s}: loaded")
        else:
            print(f"    {PRODUCTS[key][2]:30s}: not found")

    if not data:
        print("\n  No data found for this date. Try a different --date.")
        return

    print("\n  Generating figure ...")
    make_figure(data, date_str, args.output)
    print("\nDone.")


if __name__ == "__main__":
    main()
