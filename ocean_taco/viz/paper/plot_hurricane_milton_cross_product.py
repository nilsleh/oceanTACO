#!/usr/bin/env python3
"""SSH cross-product comparison for Hurricane Milton (Gulf of Mexico).

Map of L4 SSH + L3 along-track + L3 SWOT overlaid, paired with a scatter
plot showing correlation between L4 DUACS and each L3 product.
Default date: 2024-10-09 (SWOT passes directly over the hurricane eye).
"""

import argparse
import os
import warnings
from pathlib import Path

import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.lines import Line2D
from scipy.interpolate import RegularGridInterpolator
from scipy.stats import pearsonr

from ocean_taco.dataset.retrieve import (
    HF_DEFAULT_URL,
    load_bbox_nc,
    load_bbox_swot_nc,
    load_hf_dataset,
)
from ocean_taco.viz.paper.plot_hurricane_milton import (
    BBOX,
    LAT_MAX,
    LAT_MIN,
    LON_MAX,
    LON_MIN,
    MILTON_EYE,
    VMAX,
    VMIN,
)

warnings.filterwarnings("ignore", category=RuntimeWarning)


def _configure_cartopy_dir(path: str):
    p = Path(path).expanduser()
    p.mkdir(parents=True, exist_ok=True)
    os.environ["CARTOPY_USER_DIR"] = str(p)
    cartopy.config["data_dir"] = str(p)


_configure_cartopy_dir("./.cartopy")

PRODUCTS = {
    "l4_ssh": ("l4_ssh.nc", "sla", "L4 DUACS (13.3 km)", "#264653"),
    "l3_ssh": ("l3_ssh.nc", "sla_filtered", "L3 along-track (7 km)", "#f4a261"),
    "l3_swot": ("l3_swot.nc", "ssha_filtered", "L3 SWOT (2 km)", "#e63946"),
}

CMAP = "RdBu_r"

# =============================================================================
# I/O
# =============================================================================


def load_product(dataset_hf, date_str: str, key: str, cache_dir=None):
    """Load and crop one product to the Gulf bbox. Returns xr.DataArray or None."""
    fname, var, label, _ = PRODUCTS[key]

    if key == "l3_swot":
        ds = load_bbox_swot_nc(dataset_hf, date_str, BBOX, cache_dir)
    else:
        ds = load_bbox_nc(
            dataset_hf,
            date_str,
            BBOX,
            data_source=fname,
            cache_dir=cache_dir,
        )

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
    ds.close()
    return da


# =============================================================================
# Scatter interpolation
# =============================================================================


def scatter_data(da_gridded, da_l3):
    """Interpolate da_gridded onto the non-NaN positions of da_l3.

    Returns (l3_values, gridded_values) — matched pairs, NaN-free.
    """
    g_lons = da_gridded["lon"].values
    g_lats = da_gridded["lat"].values
    g_vals = da_gridded.values.astype(float)

    if g_lats[0] > g_lats[-1]:
        g_lats = g_lats[::-1]
        g_vals = g_vals[::-1, :]

    g_vals_filled = np.where(np.isnan(g_vals), 0.0, g_vals)
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

    valid = ~np.isnan(l3_flat)
    if valid.sum() < 10:
        return None, None

    g_interp = interp(pts[valid])
    l3_valid = l3_flat[valid]

    both_good = ~np.isnan(g_interp) & ~np.isnan(l3_valid)
    return l3_valid[both_good], g_interp[both_good]


# =============================================================================
# Figure
# =============================================================================


def make_figure(data: dict, date_str: str, output_path: str | None = None):
    sns.set_context("paper", font_scale=1.5)
    plt.rcParams["font.family"] = "serif"

    fig = plt.figure(figsize=(13, 5))
    gs = gridspec.GridSpec(
        1, 2,
        width_ratios=[1.3, 1.0],
        wspace=0.14,
        left=0.06,
        right=0.99,
        bottom=0.12,
        top=0.90,
    )

    # =========================================================================
    # Panel 1: map
    # =========================================================================
    ax_map = fig.add_subplot(gs[0], projection=ccrs.Mercator())
    ax_map.set_extent([LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], crs=ccrs.PlateCarree())
    ax_map.coastlines(linewidth=0.6, color="#444")
    ax_map.add_feature(cfeature.LAND, facecolor="#e8e8e8", edgecolor="none", zorder=2)
    gl = ax_map.gridlines(draw_labels=True, linewidth=0.3, alpha=0.4)
    gl.right_labels = gl.top_labels = False
    gl.xlabel_style = gl.ylabel_style = {"size": 10}

    mappable = None

    if "l4_ssh" in data:
        da = data["l4_ssh"]
        lons2d, lats2d = np.meshgrid(da["lon"].values, da["lat"].values)
        mappable = ax_map.pcolormesh(
            lons2d, lats2d, da.values,
            transform=ccrs.PlateCarree(),
            cmap=CMAP, vmin=VMIN, vmax=VMAX,
            shading="auto", alpha=0.4, rasterized=True, zorder=1,
        )

    for key, sz in [("l3_ssh", 3.0), ("l3_swot", 0.5)]:
        if key not in data:
            continue
        da = data[key]
        vals = da.values
        lons2d, lats2d = np.meshgrid(da["lon"].values, da["lat"].values)
        mask = ~np.isnan(vals)
        if mask.any():
            sc = ax_map.scatter(
                lons2d[mask], lats2d[mask], c=vals[mask],
                s=sz, cmap=CMAP, vmin=VMIN, vmax=VMAX,
                transform=ccrs.PlateCarree(), zorder=3, rasterized=True,
            )
            if mappable is None:
                mappable = sc

    # Hurricane eye marker
    if date_str in MILTON_EYE:
        eye_lon, eye_lat = MILTON_EYE[date_str]
        ax_map.scatter(
            [eye_lon], [eye_lat], marker="x", s=80, c="black",
            linewidths=1.8, zorder=5, transform=ccrs.PlateCarree(),
            label="Milton eye",
        )

    if mappable is not None:
        cb = plt.colorbar(
            mappable, ax=ax_map, orientation="horizontal",
            pad=0.05, fraction=0.03, aspect=35,
        )
        cb.set_label("SSH anomaly [m]", fontsize=11)
        cb.ax.tick_params(labelsize=10)

    ax_map.set_title(f"SSH products — {date_str}", fontsize=13)

    # Legend
    legend_items = []
    for key, marker, lw in [
        ("l4_ssh", None, 6),
        ("l3_ssh", "o", None),
        ("l3_swot", "o", None),
    ]:
        if key not in data:
            continue
        _, _, label, color = PRODUCTS[key]
        if lw:
            legend_items.append(Line2D([0], [0], color=color, lw=lw, alpha=0.85, label=label))
        else:
            legend_items.append(
                Line2D([0], [0], marker=marker, color="w",
                       markerfacecolor=color, markersize=5, label=label)
            )
    if date_str in MILTON_EYE:
        legend_items.append(
            Line2D([0], [0], marker="x", color="black", markersize=7,
                   linestyle="none", label="Milton eye")
        )
    if legend_items:
        ax_map.legend(handles=legend_items, loc="lower left",
                      fontsize=10, framealpha=0.9, edgecolor="none")

    # =========================================================================
    # Panel 2: scatter — L4 vs L3 products
    # =========================================================================
    ax_sc = fig.add_subplot(gs[1])

    scatter_configs = []
    if "l4_ssh" in data and "l3_ssh" in data:
        scatter_configs.append(("l3_ssh", PRODUCTS["l3_ssh"][3], "L4 vs L3 along-track"))
    if "l4_ssh" in data and "l3_swot" in data:
        scatter_configs.append(("l3_swot", PRODUCTS["l3_swot"][3], "L4 vs L3 SWOT"))

    for l3key, color, label in scatter_configs:
        l3_vals, g_vals = scatter_data(data["l4_ssh"], data[l3key])
        if l3_vals is None:
            continue

        n = len(l3_vals)
        rng = np.random.default_rng(42)
        if n > 5000:
            idx = rng.choice(n, 5000, replace=False)
            l3_plot, g_plot = l3_vals[idx], g_vals[idx]
        else:
            l3_plot, g_plot = l3_vals, g_vals

        r, _ = pearsonr(l3_vals, g_vals)
        rmse = np.sqrt(np.mean((l3_vals - g_vals) ** 2))

        ax_sc.scatter(
            l3_plot, g_plot, s=1.0, alpha=0.35, color=color,
            rasterized=True,
            label=f"{label}  (r={r:.3f}, RMSE={rmse:.3f} m)",
        )

    lim = min(
        max(abs(v) for v in [*ax_sc.get_xlim(), *ax_sc.get_ylim()]),
        0.8,
    )
    ax_sc.plot([-lim, lim], [-lim, lim], "k--", linewidth=0.8, alpha=0.4, zorder=0)
    ax_sc.set_xlim(-lim, lim)
    ax_sc.set_ylim(-lim, lim)

    ax_sc.set_xlabel("L3 observation [m]", fontsize=13)
    ax_sc.set_ylabel("L4 DUACS [m]", fontsize=13)
    ax_sc.set_title("L4 DUACS vs L3 observations", fontsize=13)
    ax_sc.legend(fontsize=10, loc="upper left", framealpha=0.9,
                 edgecolor="none", markerscale=4, handletextpad=0.5)
    ax_sc.grid(True, alpha=0.2)
    ax_sc.tick_params(labelsize=11)

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"  Figure saved to {output_path}")
    else:
        plt.show()


# =============================================================================
# Entry point
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="SSH cross-product comparison for Hurricane Milton"
    )
    parser.add_argument("--hf-url", type=str, default=HF_DEFAULT_URL)
    parser.add_argument("--cache-dir", type=str, default=None)
    parser.add_argument("--date", type=str, default="2024-10-09",
                        help="Date to plot (YYYY-MM-DD)")
    parser.add_argument("--output", type=str, default="hurricane_milton_cross_product.png")
    args = parser.parse_args()

    print("=" * 55)
    print("  OceanTACO — Hurricane Milton SSH Cross-Product")
    print(f"  Date   : {args.date}")
    print(f"  Region : Gulf of Mexico ({LON_MIN}–{LON_MAX}°W, {LAT_MIN}–{LAT_MAX}°N)")
    print("=" * 55)

    dataset_hf = load_hf_dataset(args.hf_url)

    print("\n  Loading products ...")
    data = {}
    for key, (_, _, label, _) in PRODUCTS.items():
        da = load_product(dataset_hf, args.date, key, args.cache_dir)
        if da is not None:
            data[key] = da
            print(f"    {label:30s}: loaded")
        else:
            print(f"    {label:30s}: not found")

    if not data:
        print("\n  No data found. Try a different --date.")
        return

    print("\n  Generating figure ...")
    make_figure(data, args.date, args.output)
    print("\nDone.")


if __name__ == "__main__":
    main()
