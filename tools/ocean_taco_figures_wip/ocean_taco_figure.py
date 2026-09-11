"""OceanTACO central dataset figure for ESSD publication."""

import argparse

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from ocean_taco.dataset.retrieve import (
    download_files,
    load_hf_dataset,
    query_files,
    query_global_glorys,
)
from ocean_taco.viz.plot_utils import (
    configure_cartopy_dir,
    extract_field,
    plot_argo_scatter,
    plot_field,
    style_inset,
)

configure_cartopy_dir("./.cartopy")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MM_TO_INCH = 1 / 25.4
FIG_WIDTH_MM = 180
FIG_HEIGHT_MM = 220
FIG_WIDTH = FIG_WIDTH_MM * MM_TO_INCH
FIG_HEIGHT = FIG_HEIGHT_MM * MM_TO_INCH

# Region bounds: (lon_min, lat_min, lon_max, lat_max) for filter_bbox
GULF_STREAM_BBOX = (-70, 30, -40, 45)
KUROSHIO_BBOX = (130, 25, 160, 45)
SOUTH_PACIFIC_BBOX = (-130, -40, -100, -20)

# Variable configurations: (nc_var, cmap, label, (vmin, vmax))
GULF_STREAM_VARS = {
    "l4_ssh.nc": ("sla", "RdBu_r", "SSH — L4 (m)", (-0.5, 0.5)),
    "l3_ssh.nc": ("sla_filtered", "RdBu_r", "SSH — L3 (m)", (-0.5, 0.5)),
    "l3_swot.nc": ("ssha_filtered", "RdBu_r", "SSH — SWOT (m)", (-0.5, 0.5)),
}

KUROSHIO_VARS = {
    "l4_sst.nc": ("analysed_sst", "RdYlBu_r", "SST — L4 (°C)", (5, 28)),
    "l3_sst.nc": (
        "adjusted_sea_surface_temperature",
        "RdYlBu_r",
        "SST — L3 (°C)",
        (5, 28),
    ),
}

SOUTH_PACIFIC_VARS = {
    "l4_sss.nc": ("sos", "viridis", "SSS — L4 (PSU)", (33, 37)),
    "l3_sss_asc.nc": ("Sea_Surface_Salinity", "viridis", "SSS — L3 (PSU)", (33, 37)),
}

# Argo is handled separately as scatter point data
ARGO_FILE = "argo.nc"
ARGO_VAR = "PSAL"
ARGO_LABEL = "SSS — Argo (PSU)"
ARGO_VRANGE = (33, 37)

# Font sizes for journal figure
FONTSIZE_TITLE = 8
FONTSIZE_LABEL = 6
FONTSIZE_TICK = 5
FONTSIZE_CBAR = 5
FONTSIZE_PANEL = 7


# ---------------------------------------------------------------------------
# Main figure
# ---------------------------------------------------------------------------


def make_figure(global_data, gs_data, ks_data, sp_data, date):
    """Compose the full publication figure.

    Layout (3 rows, 10 columns):
        Row 0: Gulf Stream SSH (3 panels, cols 2-7, centered)
        Row 1: Global GLORYS SST (full width, Robinson projection)
        Row 2: Kuroshio SST (2 panels, cols 0-3) | South Pacific SSS (3 panels, cols 4-9)

    Args:
        global_data: List of xr.Dataset for global GLORYS tiles.
        gs_data: Dict {filename: xr.Dataset} for Gulf Stream region.
        ks_data: Dict {filename: xr.Dataset} for Kuroshio region.
        sp_data: Dict {filename: xr.Dataset} for South Pacific region.
        date: Date string for the figure title.

    Returns:
        Matplotlib Figure.
    """
    fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=300)
    plt.rcParams.update(
        {
            "font.size": FONTSIZE_LABEL,
            "font.family": "sans-serif",
            "axes.linewidth": 0.4,
        }
    )

    gs = gridspec.GridSpec(
        3, 10, figure=fig, height_ratios=[1, 1.8, 1], hspace=0.06, wspace=0.12, top=0.95
    )

    # ---- Gulf Stream insets (top row, centered: cols 2-7) ----
    gs_extent = [
        GULF_STREAM_BBOX[0],
        GULF_STREAM_BBOX[2],
        GULF_STREAM_BBOX[1],
        GULF_STREAM_BBOX[3],
    ]

    ax_gs = []
    for i, (fname, (nc_var, cmap, label, (vmin, vmax))) in enumerate(
        GULF_STREAM_VARS.items()
    ):
        ax = fig.add_subplot(gs[0, i * 3 : (i + 1) * 3], projection=ccrs.Mercator())
        style_inset(
            ax,
            gs_extent,
            label,
            left_labels=(i == 0),
            right_labels=False,
            fontsize_label=FONTSIZE_LABEL,
            fontsize_tick=FONTSIZE_TICK,
        )
        if fname in gs_data:
            try:
                lons, lats, vals = extract_field(
                    gs_data[fname],
                    nc_var,
                    (GULF_STREAM_BBOX[0], GULF_STREAM_BBOX[2]),
                    (GULF_STREAM_BBOX[1], GULF_STREAM_BBOX[3]),
                )
                plot_field(
                    ax,
                    lons,
                    lats,
                    vals,
                    cmap,
                    vmin,
                    vmax,
                    add_cbar=True,
                    cbar_label=label,
                    fontsize_cbar=FONTSIZE_CBAR,
                    fontsize_tick=FONTSIZE_TICK,
                )
            except Exception as e:
                ax.text(
                    0.5,
                    0.5,
                    f"Error: {e}",
                    transform=ax.transAxes,
                    ha="center",
                    fontsize=5,
                    color="red",
                )
        else:
            ax.text(
                0.5,
                0.5,
                "no data",
                transform=ax.transAxes,
                ha="center",
                va="center",
                color="gray",
                fontsize=FONTSIZE_LABEL,
            )
        ax_gs.append(ax)

    # ---- Global GLORYS SST (middle row) ----
    ax_global = fig.add_subplot(gs[1, :], projection=ccrs.Robinson())
    ax_global.set_global()
    ax_global.coastlines(linewidth=0.3, color="#555")
    ax_global.add_feature(
        cfeature.LAND, facecolor="#e8e8e8", edgecolor="none", zorder=2
    )
    ax_global.gridlines(linewidth=0.15, alpha=0.3)

    glorys_vmin, glorys_vmax = -2, 30
    mappable = None
    for ds in global_data:
        try:
            d = ds["thetao"]
            if "time" in d.dims:
                d = d.isel(time=0)
            if "depth" in d.dims:
                d = d.isel(depth=0)
            d = d.squeeze()

            vals = d.values.astype(float)
            lons_2d, lats_2d = np.meshgrid(ds["lon"].values, ds["lat"].values)
            mappable = ax_global.pcolormesh(
                lons_2d,
                lats_2d,
                vals,
                transform=ccrs.PlateCarree(),
                cmap="RdYlBu_r",
                vmin=glorys_vmin,
                vmax=glorys_vmax,
                shading="auto",
                rasterized=True,
                zorder=1,
            )
        except Exception as e:
            print(f"  Warning: could not plot GLORYS tile: {e}")

    if mappable:
        cb = plt.colorbar(
            mappable,
            ax=ax_global,
            orientation="horizontal",
            pad=0.03,
            fraction=0.035,
            shrink=0.5,
        )
        cb.set_label("Sea Surface Temperature (°C)", fontsize=FONTSIZE_LABEL)
        cb.ax.tick_params(labelsize=FONTSIZE_TICK)

    # Bounding boxes on global map
    for bbox, color in [
        (GULF_STREAM_BBOX, "#d32f2f"),
        (KUROSHIO_BBOX, "#1565c0"),
        (SOUTH_PACIFIC_BBOX, "#2e7d32"),
    ]:
        rect = mpatches.Rectangle(
            (bbox[0], bbox[1]),
            bbox[2] - bbox[0],
            bbox[3] - bbox[1],
            linewidth=0.8,
            edgecolor=color,
            facecolor="none",
            transform=ccrs.PlateCarree(),
            zorder=5,
        )
        ax_global.add_patch(rect)

    # ---- Kuroshio insets (bottom row, right: cols 6-9) ----
    ks_extent = [KUROSHIO_BBOX[0], KUROSHIO_BBOX[2], KUROSHIO_BBOX[1], KUROSHIO_BBOX[3]]

    n_ks = len(KUROSHIO_VARS)
    ax_ks = []
    for i, (fname, (nc_var, cmap, label, (vmin, vmax))) in enumerate(
        KUROSHIO_VARS.items()
    ):
        ax = fig.add_subplot(
            gs[2, 6 + i * 2 : 6 + (i + 1) * 2], projection=ccrs.Mercator()
        )
        style_inset(
            ax,
            ks_extent,
            label,
            left_labels=False,
            right_labels=(i == n_ks - 1),
            fontsize_label=FONTSIZE_LABEL,
            fontsize_tick=FONTSIZE_TICK,
        )
        if fname in ks_data:
            try:
                lons, lats, vals = extract_field(
                    ks_data[fname],
                    nc_var,
                    (KUROSHIO_BBOX[0], KUROSHIO_BBOX[2]),
                    (KUROSHIO_BBOX[1], KUROSHIO_BBOX[3]),
                )
                plot_field(
                    ax,
                    lons,
                    lats,
                    vals,
                    cmap,
                    vmin,
                    vmax,
                    add_cbar=True,
                    cbar_label=label,
                    fontsize_cbar=FONTSIZE_CBAR,
                    fontsize_tick=FONTSIZE_TICK,
                )
            except Exception as e:
                ax.text(
                    0.5,
                    0.5,
                    f"Error: {e}",
                    transform=ax.transAxes,
                    ha="center",
                    fontsize=5,
                    color="red",
                )
        else:
            ax.text(
                0.5,
                0.5,
                "no data",
                transform=ax.transAxes,
                ha="center",
                va="center",
                color="gray",
                fontsize=FONTSIZE_LABEL,
            )
        ax_ks.append(ax)

    # ---- South Pacific SSS insets (bottom row, left: cols 0-5) ----
    sp_extent = [
        SOUTH_PACIFIC_BBOX[0],
        SOUTH_PACIFIC_BBOX[2],
        SOUTH_PACIFIC_BBOX[1],
        SOUTH_PACIFIC_BBOX[3],
    ]

    sp_panels = list(SOUTH_PACIFIC_VARS.items()) + [
        (ARGO_FILE, (ARGO_VAR, "viridis", ARGO_LABEL, ARGO_VRANGE))
    ]

    ax_sp = []
    for i, (fname, (nc_var, cmap, label, (vmin, vmax))) in enumerate(sp_panels):
        ax = fig.add_subplot(gs[2, i * 2 : (i + 1) * 2], projection=ccrs.Mercator())
        style_inset(
            ax,
            sp_extent,
            label,
            left_labels=(i == 0),
            right_labels=False,
            fontsize_label=FONTSIZE_LABEL,
            fontsize_tick=FONTSIZE_TICK,
        )
        if fname in sp_data:
            try:
                if fname == ARGO_FILE:
                    plot_argo_scatter(
                        ax,
                        sp_data[fname],
                        nc_var,
                        (SOUTH_PACIFIC_BBOX[0], SOUTH_PACIFIC_BBOX[2]),
                        (SOUTH_PACIFIC_BBOX[1], SOUTH_PACIFIC_BBOX[3]),
                        cmap,
                        vmin,
                        vmax,
                        cbar_label=label,
                        fontsize_cbar=FONTSIZE_CBAR,
                        fontsize_tick=FONTSIZE_TICK,
                    )
                else:
                    lons, lats, vals = extract_field(
                        sp_data[fname],
                        nc_var,
                        (SOUTH_PACIFIC_BBOX[0], SOUTH_PACIFIC_BBOX[2]),
                        (SOUTH_PACIFIC_BBOX[1], SOUTH_PACIFIC_BBOX[3]),
                    )
                    plot_field(
                        ax,
                        lons,
                        lats,
                        vals,
                        cmap,
                        vmin,
                        vmax,
                        add_cbar=True,
                        cbar_label=label,
                        fontsize_cbar=FONTSIZE_CBAR,
                        fontsize_tick=FONTSIZE_TICK,
                    )
            except Exception as e:
                ax.text(
                    0.5,
                    0.5,
                    f"Error: {e}",
                    transform=ax.transAxes,
                    ha="center",
                    fontsize=5,
                    color="red",
                )
        else:
            ax.text(
                0.5,
                0.5,
                "no data",
                transform=ax.transAxes,
                ha="center",
                va="center",
                color="gray",
                fontsize=FONTSIZE_LABEL,
            )
        ax_sp.append(ax)

    # ---- Connector lines (leftmost + rightmost only) ----
    fig.canvas.draw()

    for ax_panels, bbox, color, side in [
        (ax_gs, GULF_STREAM_BBOX, "#d32f2f", "top"),
        (ax_ks, KUROSHIO_BBOX, "#1565c0", "bottom"),
        (ax_sp, SOUTH_PACIFIC_BBOX, "#2e7d32", "bottom"),
    ]:
        _draw_connector(fig, ax_global, ax_panels[0], bbox, side, color, anchor="left")
        _draw_connector(
            fig, ax_global, ax_panels[-1], bbox, side, color, anchor="right"
        )

    # fig.suptitle(
    #     f'OceanTACO — Global Sea Surface State  |  {date}',
    #     fontsize=FONTSIZE_TITLE, fontweight='bold', y=0.97,
    # )

    return fig


def _draw_connector(fig, ax_main, ax_inset, bbox, side, color, anchor="left"):
    """Draw a single dashed connector line between a main-map bbox corner and an inset panel edge.

    Args:
        fig: Matplotlib Figure.
        ax_main: Main map GeoAxes.
        ax_inset: Inset GeoAxes.
        bbox: (lon_min, lat_min, lon_max, lat_max).
        side: 'top' or 'bottom'.
        color: Line color.
        anchor: 'left' or 'right' — which side of the bbox/inset to connect.
    """
    lon_min, lat_min, lon_max, lat_max = bbox

    if side == "top":
        main_lat = lat_max
        inset_y_frac = 0  # bottom edge of inset (inset is above)
    else:
        main_lat = lat_min
        inset_y_frac = 1  # top edge of inset (inset is below)

    if anchor == "left":
        main_lon = lon_min
        inset_x_frac = 0
    else:
        main_lon = lon_max
        inset_x_frac = 1

    proj_xy = ax_main.projection.transform_point(main_lon, main_lat, ccrs.PlateCarree())
    disp = ax_main.transData.transform(proj_xy)
    fig_main = fig.transFigure.inverted().transform(disp)

    inset_pos = ax_inset.get_position()
    fig_inset = (
        inset_pos.x0 + inset_x_frac * inset_pos.width,
        inset_pos.y0 + inset_y_frac * inset_pos.height,
    )

    fig.add_artist(
        plt.Line2D(
            [fig_main[0], fig_inset[0]],
            [fig_main[1], fig_inset[1]],
            transform=fig.transFigure,
            color=color,
            linewidth=0.5,
            linestyle="--",
            alpha=0.6,
            clip_on=False,
            zorder=10,
        )
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    """Run the OceanTACO figure CLI."""
    parser = argparse.ArgumentParser(
        description="Generate OceanTACO central dataset figure for ESSD."
    )
    parser.add_argument(
        "--date",
        type=str,
        required=True,
        help="Snapshot date in YYYY-MM-DD format (e.g. 2023-06-07)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output filename (default: oceantaco_figure_<date>.pdf)",
    )
    parser.add_argument(
        "--dpi", type=int, default=300, help="Output resolution in DPI (default: 300)"
    )
    args = parser.parse_args()

    date = args.date
    output = args.output or f"oceantaco_figure_{date}.pdf"

    print("OceanTACO Figure Generator")
    print(f"  Date   : {date}")
    print(f"  Output : {output}\n")

    # Load catalog
    print("Loading OceanTACO catalog ...")
    dataset = load_hf_dataset()

    # Query regional files
    print("\nQuerying Gulf Stream files ...")
    gs_files = query_files(
        dataset, date, GULF_STREAM_BBOX, set(GULF_STREAM_VARS.keys())
    )
    print(f"  Found {len(gs_files)} files")

    print("\nQuerying Kuroshio files ...")
    ks_files = query_files(dataset, date, KUROSHIO_BBOX, set(KUROSHIO_VARS.keys()))
    print(f"  Found {len(ks_files)} files")

    print("\nQuerying global GLORYS files ...")
    global_files = query_global_glorys(dataset, date)
    print(f"  Found {len(global_files)} files")

    print("\nQuerying South Pacific files ...")
    sp_needed = set(SOUTH_PACIFIC_VARS.keys()) | {ARGO_FILE}
    sp_files = query_files(dataset, date, SOUTH_PACIFIC_BBOX, sp_needed)
    print(f"  Found {len(sp_files)} files")

    # Download
    print("\nDownloading Gulf Stream data ...")
    gs_data = download_files(gs_files)

    print("\nDownloading Kuroshio data ...")
    ks_data = download_files(ks_files)

    print("\nDownloading global GLORYS data ...")
    global_data = download_files(global_files, as_list=True)

    print("\nDownloading South Pacific data ...")
    sp_data = download_files(sp_files)

    # Plot
    print("\nComposing figure ...")
    fig = make_figure(global_data, gs_data, ks_data, sp_data, date)

    print(f"\nSaving to {output} ...")
    fig.savefig(output, dpi=args.dpi, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    print("Done.")


if __name__ == "__main__":
    main()
