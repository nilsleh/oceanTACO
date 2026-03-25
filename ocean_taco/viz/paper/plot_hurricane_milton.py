#!/usr/bin/env python3
"""Hurricane Milton extreme-event figure.

Shows L4 Wind (quiver), L3 along-track SSH, and L3 SWOT for 3-4 representative
dates during Hurricane Milton (Oct 5-10, 2024) in the Gulf of Mexico.
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

from ocean_taco.dataset.retrieve import (
    HF_DEFAULT_URL,
    load_bbox_nc,
    load_bbox_swot_nc,
    load_hf_dataset,
)

warnings.filterwarnings("ignore", category=RuntimeWarning)


def _configure_cartopy_dir(path: str):
    p = Path(path).expanduser()
    p.mkdir(parents=True, exist_ok=True)
    os.environ["CARTOPY_USER_DIR"] = str(p)
    cartopy.config["data_dir"] = str(p)


_configure_cartopy_dir("./.cartopy")

# =============================================================================
# Configuration
# =============================================================================

# Gulf of Mexico bbox: lon_min, lon_max, lat_min, lat_max
LON_MIN, LON_MAX = -100, -75
LAT_MIN, LAT_MAX = 15, 32

# load_bbox_nc expects (lon_min, lat_min, lon_max, lat_max)
BBOX = (LON_MIN, LAT_MIN, LON_MAX, LAT_MAX)

PRODUCTS = {
    "l4_wind": ("l4_wind.nc", ("eastward_wind", "northward_wind"), "L4 Wind (0.125°)"),
    "l3_ssh": ("l3_ssh.nc", "sla_filtered", "L3 along-track (7 km)"),
    "l3_swot": ("l3_swot.nc", "ssha_filtered", "L3 SWOT (2 km)"),
}

COL_TITLES = ["L4 Wind", "L3 Along-Track", "L3 SWOT"]

DEFAULT_DATES = ["2024-10-05", "2024-10-07", "2024-10-09", "2024-10-10"]

# Approximate Hurricane Milton eye positions (IBTrACS)
MILTON_EYE = {
    "2024-10-04": (-94.70, 20.60),
    "2024-10-05": (-95.20, 21.10),
    "2024-10-06": (-95.40, 22.80),
    "2024-10-07": (-93.40, 22.50),
    "2024-10-08": (-90.40, 21.80),
    "2024-10-09": (-86.90, 23.00),
    "2024-10-10": (-82.70, 27.20),
}

VMIN, VMAX = -0.7, 0.7
CMAP = "RdBu_r"
EXTENT = [LON_MIN, LON_MAX, LAT_MIN, LAT_MAX]


# =============================================================================
# Data loading
# =============================================================================


def load_date(dataset_hf, date_str: str, cache_dir=None) -> dict:
    """Load all three products for a given date. Returns dict of xr.Dataset."""
    data = {}
    for key, (fname, var, label) in PRODUCTS.items():
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
        if ds is not None:
            data[key] = (ds, var)
            print(f"    {label:30s}: loaded")
        else:
            print(f"    {label:30s}: not found")
    return data


def extract_2d(ds, var: str) -> tuple:
    """Return (lons_2d, lats_2d, vals_2d) from an xr.Dataset."""
    lats = ds["lat"].values
    lat_sl = slice(LAT_MAX, LAT_MIN) if lats[0] > lats[-1] else slice(LAT_MIN, LAT_MAX)

    d = ds[var].sel(lon=slice(LON_MIN, LON_MAX), lat=lat_sl)
    if "time" in d.dims:
        d = d.isel(time=0)
    if "depth" in d.dims:
        d = d.isel(depth=0)
    d = d.squeeze()

    vals = d.values.astype(float)

    lon_vals = ds["lon"].sel(lon=slice(LON_MIN, LON_MAX)).values
    lat_vals = ds["lat"].sel(lat=lat_sl).values
    lons_2d, lats_2d = np.meshgrid(lon_vals, lat_vals)

    return lons_2d, lats_2d, vals


def extract_2d_wind(ds) -> tuple:
    """Return (lons_2d, lats_2d, u_mean, v_mean, speed_max) from an L4 wind xr.Dataset."""
    lons_2d, lats_2d, u_mean = extract_2d(ds, "eastward_wind")
    _, _, v_mean = extract_2d(ds, "northward_wind")
    _, _, u_max = extract_2d(ds, "eastward_wind_max")
    _, _, v_max = extract_2d(ds, "northward_wind_max")
    speed_max = np.sqrt(u_max**2 + v_max**2)
    return lons_2d, lats_2d, u_mean, v_mean, speed_max


# =============================================================================
# Panel drawing
# =============================================================================


def draw_panel(ax, date_str: str, key: str, data: dict, mappable_ref: list, wind_mappable_ref: list):
    """Draw one map panel. mappable_ref captures the SSH mappable; wind_mappable_ref captures the wind mappable."""
    ax.set_extent(EXTENT, crs=ccrs.PlateCarree())
    ax.coastlines(linewidth=0.5, color="#444")
    ax.add_feature(cfeature.LAND, facecolor="#e8e8e8", edgecolor="none", zorder=2)
    gl = ax.gridlines(draw_labels=False, linewidth=0.2, alpha=0.3)
    gl.left_labels = False
    gl.right_labels = False
    gl.top_labels = False
    gl.bottom_labels = False

    if key not in data:
        ax.text(
            0.5,
            0.5,
            "No data",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=12,
            color="gray",
        )
        return

    ds, var = data[key]

    if key == "l4_wind":
        lons_2d, lats_2d, u_mean, v_mean, speed_max = extract_2d_wind(ds)
        ny, nx = lons_2d.shape

        # Background: daily max wind speed
        mp = ax.pcolormesh(
            lons_2d, lats_2d, speed_max,
            transform=ccrs.PlateCarree(),
            cmap="BuGn", vmin=0, vmax=40, zorder=1,
        )
        if not wind_mappable_ref:
            wind_mappable_ref.append(mp)

        # Quiver: mean direction (white arrows, no color encoding)
        step_y = max(1, ny // 30)
        step_x = max(1, nx // 30)
        sl = (slice(None, None, step_y), slice(None, None, step_x))
        ax.quiver(
            lons_2d[sl], lats_2d[sl], u_mean[sl], v_mean[sl],
            transform=ccrs.PlateCarree(),
            color="white", scale=300, width=0.003, zorder=3,
        )
    else:
        lons_2d, lats_2d, vals = extract_2d(ds, var)
        s = 0.5 if key == "l3_ssh" else 0.05
        mask = ~np.isnan(vals)
        if np.any(mask):
            mp = ax.scatter(
                lons_2d[mask],
                lats_2d[mask],
                c=vals[mask],
                s=s,
                cmap=CMAP,
                vmin=VMIN,
                vmax=VMAX,
                transform=ccrs.PlateCarree(),
                zorder=3,
                rasterized=True,
            )
        else:
            ax.text(
                0.5,
                0.5,
                "No data",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=9,
                color="gray",
            )
            return
        if not mappable_ref:
            mappable_ref.append(mp)

    # Hurricane track as dashed line
    track_lons = [lon for _, (lon, _) in sorted(MILTON_EYE.items())]
    track_lats = [lat for _, (_, lat) in sorted(MILTON_EYE.items())]
    ax.plot(
        track_lons,
        track_lats,
        color="black",
        linewidth=1.2,
        linestyle="--",
        transform=ccrs.PlateCarree(),
        zorder=4,
    )

    # Hurricane eye marker
    if date_str in MILTON_EYE:
        eye_lon, eye_lat = MILTON_EYE[date_str]
        if LON_MIN <= eye_lon <= LON_MAX and LAT_MIN <= eye_lat <= LAT_MAX:
            ax.scatter(
                [eye_lon],
                [eye_lat],
                marker="x",
                s=60,
                c="black",
                linewidths=1.5,
                zorder=5,
                transform=ccrs.PlateCarree(),
            )


# =============================================================================
# Figure
# =============================================================================


def make_figure(all_data: dict, dates: list, output_path: str):
    sns.set_context("paper", font_scale=1.5)
    plt.rcParams["font.family"] = "serif"

    n_dates = len(dates)
    fig = plt.figure(figsize=(14, 4 * n_dates))

    gs = gridspec.GridSpec(
        n_dates,
        3,
        hspace=0.12,
        wspace=0.06,
        left=0.10,
        right=0.98,
        bottom=0.08,
        top=0.94,
    )

    mappable_ref = []
    wind_mappable_ref = []

    for row_idx, date_str in enumerate(dates):
        data = all_data[date_str]
        keys = ["l4_wind", "l3_ssh", "l3_swot"]

        for col_idx, key in enumerate(keys):
            ax = fig.add_subplot(gs[row_idx, col_idx], projection=ccrs.Mercator())
            draw_panel(ax, date_str, key, data, mappable_ref, wind_mappable_ref)

            # Row label (date) on leftmost panel
            if col_idx == 0:
                ax.text(
                    -0.18,
                    0.5,
                    date_str,
                    transform=ax.transAxes,
                    rotation=90,
                    va="center",
                    ha="right",
                    fontsize=13,
                    fontweight="bold",
                )
                # Add gridline lat labels only on left column
                gl = ax.gridlines(draw_labels=True, linewidth=0.2, alpha=0.3)
                gl.left_labels = True
                gl.right_labels = False
                gl.top_labels = False
                gl.bottom_labels = row_idx == n_dates - 1
                gl.xlabel_style = {"size": 12}
                gl.ylabel_style = {"size": 12}

            # Column header on top row only
            if row_idx == 0:
                ax.set_title(COL_TITLES[col_idx], fontsize=14, pad=4)

            # Bottom row: add lon labels for all panels
            if row_idx == n_dates - 1 and col_idx > 0:
                gl2 = ax.gridlines(draw_labels=True, linewidth=0.2, alpha=0.3)
                gl2.left_labels = False
                gl2.right_labels = False
                gl2.top_labels = False
                gl2.bottom_labels = True
                gl2.xlabel_style = {"size": 12}
                gl2.ylabel_style = {"size": 12}

    # Wind colorbar (left portion)
    if wind_mappable_ref:
        cbar_wind_ax = fig.add_axes([0.10, 0.03, 0.22, 0.018])
        cb_wind = fig.colorbar(wind_mappable_ref[0], cax=cbar_wind_ax, orientation="horizontal")
        cb_wind.set_label("Max daily wind speed [m s\u207b\u00b9]", fontsize=14)
        cb_wind.ax.tick_params(labelsize=12)

    # SSH colorbar (right portion)
    if mappable_ref:
        cbar_ax = fig.add_axes([0.40, 0.03, 0.55, 0.018])
        cb = fig.colorbar(mappable_ref[0], cax=cbar_ax, orientation="horizontal")
        cb.set_label("SSH anomaly [m]", fontsize=14)
        cb.ax.tick_params(labelsize=12)

    # fig.suptitle(
    #     "Hurricane Milton (Oct 2024) — SSH anomaly across products",
    #     fontsize=16,
    #     y=0.97,
    # )

    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"  Figure saved to {output_path}")


# =============================================================================
# Entry point
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Hurricane Milton SSH extreme-event figure"
    )
    parser.add_argument("--hf-url", type=str, default=HF_DEFAULT_URL)
    parser.add_argument("--cache-dir", type=str, default=None)
    parser.add_argument("--output", type=str, default="hurricane_milton.png")
    parser.add_argument(
        "--dates",
        type=str,
        default=",".join(DEFAULT_DATES),
        help="Comma-separated dates YYYY-MM-DD",
    )
    args = parser.parse_args()

    dates = [d.strip() for d in args.dates.split(",")]

    print("=" * 55)
    print("  OceanTACO — Hurricane Milton Wind & SSH Figure")
    print(f"  Dates   : {', '.join(dates)}")
    print(f"  Region  : Gulf of Mexico ({LON_MIN}–{LON_MAX}°W, {LAT_MIN}–{LAT_MAX}°N)")
    print("=" * 55)

    dataset_hf = load_hf_dataset(args.hf_url)

    all_data = {}
    valid_dates = []
    for date_str in dates:
        print(f"\n  Loading {date_str} ...")
        data = load_date(dataset_hf, date_str, args.cache_dir)
        if data:
            all_data[date_str] = data
            valid_dates.append(date_str)
        else:
            print(f"  Skipping {date_str} — no data found.")

    if not valid_dates:
        print("\n  No data found for any date. Exiting.")
        return

    print(f"\n  Generating figure for {len(valid_dates)} date(s) ...")
    make_figure(all_data, valid_dates, args.output)
    print("\nDone.")


if __name__ == "__main__":
    main()
