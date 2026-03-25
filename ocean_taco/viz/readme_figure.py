"""Generate a compact OceanTACO figure for README usage.

The figure is built directly from OceanTACO data sources:
- Global GLORYS SST background
- Regional SSH (L4)
- Regional SWOT SSH (L3)
- Regional SST (L4)
- Regional SSS (L4) with Argo salinity points
"""

import argparse
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

from ocean_taco.dataset.retrieve import (
    download_files,
    load_hf_dataset,
    query_files,
    query_global_glorys,
)
from ocean_taco.viz.plot_utils import configure_cartopy_dir, extract_field, plot_field, style_inset

configure_cartopy_dir("./.cartopy")

HF_URL = "https://huggingface.co/datasets/nilsleh/OceanTACO/resolve/main/"

GULF_STREAM_BBOX = (-70, 30, -40, 45)
KUROSHIO_BBOX = (130, 25, 160, 45)
SOUTH_PACIFIC_BBOX = (-130, -40, -100, -20)

GLOBAL_SST_VMIN = -2
GLOBAL_SST_VMAX = 30

PANEL_SPECS = {
    "ssh_l4": {
        "bbox": GULF_STREAM_BBOX,
        "filename": "l4_ssh.nc",
        "var": "sla",
        "title": "SSH L4",
        "cmap": "RdBu_r",
        "range": (-0.5, 0.5),
    },
    "ssh_swot": {
        "bbox": GULF_STREAM_BBOX,
        "filename": "l3_swot.nc",
        "var": "ssha_filtered",
        "title": "SSH SWOT",
        "cmap": "RdBu_r",
        "range": (-0.5, 0.5),
    },
    "sst_l4": {
        "bbox": KUROSHIO_BBOX,
        "filename": "l4_sst.nc",
        "var": "analysed_sst",
        "title": "SST L4",
        "cmap": "RdYlBu_r",
        "range": (5, 30),
    },
    "sst_l3": {
        "bbox": KUROSHIO_BBOX,
        "filename": "l3_sst.nc",
        "var": "adjusted_sea_surface_temperature",
        "title": "SST L3",
        "cmap": "RdYlBu_r",
        "range": (5, 30),
    },
    "sss_l4": {
        "bbox": SOUTH_PACIFIC_BBOX,
        "filename": "l4_sss.nc",
        "var": "sos",
        "title": "SSS L4 + Argo",
        "cmap": "viridis",
        "range": (33, 37),
    },
    "sss_l3_smos": {
        "bbox": SOUTH_PACIFIC_BBOX,
        "filename": "l3_sss_asc.nc",
        "var": "Sea_Surface_Salinity",
        "title": "SSS L3 SMOS",
        "cmap": "viridis",
        "range": (33, 37),
    },
}


def _load_panel_dataset(dataset, date, spec):
    lon_min, lat_min, lon_max, lat_max = spec["bbox"]
    files = query_files(
        dataset,
        date,
        (lon_min, lat_min, lon_max, lat_max),
        {spec["filename"]},
    )
    if files.empty:
        return None
    loaded = download_files(files, max_workers=4)
    return loaded.get(spec["filename"])


def _load_argo_dataset(dataset, date):
    lon_min, lat_min, lon_max, lat_max = SOUTH_PACIFIC_BBOX
    files = query_files(dataset, date, (lon_min, lat_min, lon_max, lat_max), {"argo.nc"})
    if files.empty:
        return None
    loaded = download_files(files, max_workers=4)
    return loaded.get("argo.nc")


def _draw_global(ax, global_tiles):
    ax.set_global()
    ax.set_facecolor("#f8fbff")
    ax.coastlines(linewidth=0.35, color="#303030")
    ax.add_feature(cfeature.LAND, facecolor="#f0f0f0", edgecolor="none", zorder=2)
    ax.gridlines(linewidth=0.2, alpha=0.25)

    mappable = None
    for ds in global_tiles:
        try:
            field = ds["thetao"]
            if "time" in field.dims:
                field = field.isel(time=0)
            if "depth" in field.dims:
                field = field.isel(depth=0)
            field = field.squeeze()

            vals = field.values.astype(float)
            lons_2d, lats_2d = np.meshgrid(ds["lon"].values, ds["lat"].values)
            mappable = ax.pcolormesh(
                lons_2d,
                lats_2d,
                vals,
                transform=ccrs.PlateCarree(),
                cmap="RdYlBu_r",
                vmin=GLOBAL_SST_VMIN,
                vmax=GLOBAL_SST_VMAX,
                shading="auto",
                rasterized=True,
                zorder=1,
            )
        except Exception as exc:
            print(f"Warning: skipped a global tile: {exc}")

    return mappable


def _plot_panel(ax, ds, spec, side):
    bbox = spec["bbox"]
    extent = [bbox[0], bbox[2], bbox[1], bbox[3]]
    style_inset(
        ax,
        extent,
        spec["title"],
        left_labels=(side == "left"),
        right_labels=(side == "right"),
        fontsize_label=8,
        fontsize_tick=6,
    )

    if ds is None:
        ax.text(
            0.5,
            0.5,
            "data unavailable",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=7,
            color="#666666",
        )
        return

    try:
        lons, lats, vals = extract_field(
            ds,
            spec["var"],
            (bbox[0], bbox[2]),
            (bbox[1], bbox[3]),
        )
        vmin, vmax = spec["range"]
        plot_field(
            ax,
            lons,
            lats,
            vals,
            spec["cmap"],
            vmin,
            vmax,
            add_cbar=False,
        )
    except Exception as exc:
        ax.text(
            0.5,
            0.5,
            f"error: {exc}",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=6,
            color="#b00020",
        )


def _overlay_argo(ax, argo_ds, bbox):
    if argo_ds is None:
        return

    try:
        lon = argo_ds["lon"].values
        lat = argo_ds["lat"].values
        sal = argo_ds["PSAL"].values.astype(float)
        mask = (
            (lon >= bbox[0])
            & (lon <= bbox[2])
            & (lat >= bbox[1])
            & (lat <= bbox[3])
            & np.isfinite(sal)
        )
        ax.scatter(
            lon[mask],
            lat[mask],
            c=sal[mask],
            transform=ccrs.PlateCarree(),
            cmap="viridis",
            vmin=33,
            vmax=37,
            s=14,
            edgecolors="white",
            linewidths=0.15,
            alpha=0.95,
            zorder=4,
            rasterized=True,
        )
    except Exception as exc:
        print(f"Warning: could not overlay Argo: {exc}")


def _draw_bbox(ax, bbox, color):
    lon_min, lat_min, lon_max, lat_max = bbox
    rect = plt.Rectangle(
        (lon_min, lat_min),
        lon_max - lon_min,
        lat_max - lat_min,
        linewidth=1.0,
        edgecolor=color,
        facecolor="none",
        transform=ccrs.PlateCarree(),
        zorder=5,
    )
    ax.add_patch(rect)


def make_figure(global_tiles, panels, argo_ds, date):
    """Compose and return the README figure."""
    fig = plt.figure(figsize=(13.0, 8.8), dpi=300)
    plt.rcParams.update({
        "font.size": 9,
        "font.family": "sans-serif",
        "axes.linewidth": 0.5,
    })

    gs = gridspec.GridSpec(
        3,
        4,
        figure=fig,
        width_ratios=[1.15, 2.3, 2.3, 1.15],
        height_ratios=[1, 1, 1],
        left=0.02,
        right=0.98,
        bottom=0.07,
        top=0.92,
        wspace=0.02,
        hspace=0.05,
    )

    ax_global = fig.add_subplot(gs[:, 1:3], projection=ccrs.Robinson())
    ax_left_top = fig.add_subplot(gs[0, 0], projection=ccrs.Mercator())
    ax_left_mid = fig.add_subplot(gs[1, 0], projection=ccrs.Mercator())
    ax_left_bottom = fig.add_subplot(gs[2, 0], projection=ccrs.Mercator())
    ax_right_top = fig.add_subplot(gs[0, 3], projection=ccrs.Mercator())
    ax_right_mid = fig.add_subplot(gs[1, 3], projection=ccrs.Mercator())
    ax_right_bottom = fig.add_subplot(gs[2, 3], projection=ccrs.Mercator())

    mappable = _draw_global(ax_global, global_tiles)

    _plot_panel(ax_left_top, panels.get("ssh_l4"), PANEL_SPECS["ssh_l4"], "left")
    _plot_panel(
        ax_left_mid,
        panels.get("sss_l3_smos"),
        PANEL_SPECS["sss_l3_smos"],
        "left",
    )
    _plot_panel(
        ax_left_bottom,
        panels.get("sss_l4"),
        PANEL_SPECS["sss_l4"],
        "left",
    )

    _plot_panel(
        ax_right_top,
        panels.get("ssh_swot"),
        PANEL_SPECS["ssh_swot"],
        "right",
    )
    _plot_panel(
        ax_right_mid,
        panels.get("sst_l3"),
        PANEL_SPECS["sst_l3"],
        "right",
    )
    _plot_panel(
        ax_right_bottom,
        panels.get("sst_l4"),
        PANEL_SPECS["sst_l4"],
        "right",
    )

    _overlay_argo(ax_left_bottom, argo_ds, SOUTH_PACIFIC_BBOX)

    _draw_bbox(ax_global, GULF_STREAM_BBOX, "#b71c1c")
    _draw_bbox(ax_global, KUROSHIO_BBOX, "#0d47a1")
    _draw_bbox(ax_global, SOUTH_PACIFIC_BBOX, "#1b5e20")

    fig.suptitle("OceanTACO", fontsize=36, y=0.9)
    if mappable is not None:
        cax = fig.add_axes([0.39, 0.085, 0.22, 0.015])
        cb = plt.colorbar(mappable, cax=cax, orientation="horizontal")
        cb.set_label("Global SST (degC)", fontsize=8)
        cb.ax.tick_params(labelsize=7)

    fig.text(0.02, 0.95, f"Snapshot: {date}", fontsize=16, color="#455a64")
    return fig


def build_readme(
    date,
    output_png,
    output_svg,
    hf_url=HF_URL,
):
    """Generate README assets from OceanTACO remote data."""
    print("Loading OceanTACO catalog...")
    dataset = load_hf_dataset(hf_url)

    print("Querying global GLORYS tiles...")
    global_files = query_global_glorys(dataset, date)
    global_tiles = download_files(global_files, as_list=True)

    print("Loading regional source panels...")
    panels = {}
    for key, spec in PANEL_SPECS.items():
        print(f"  - {key}: {spec['filename']}")
        panels[key] = _load_panel_dataset(dataset, date, spec)

    print("Loading Argo panel overlay...")
    argo_ds = _load_argo_dataset(dataset, date)

    print("Rendering figure...")
    fig = make_figure(global_tiles, panels, argo_ds, date)

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved PNG: {output_png}")

    if output_svg is not None:
        output_svg.parent.mkdir(parents=True, exist_ok=True)
        try:
            fig.savefig(output_svg, bbox_inches="tight", facecolor="white")
            print(f"Saved SVG: {output_svg}")
        except Exception as exc:
            print(f"Warning: SVG export failed ({exc}). PNG output is available.")

    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a README figure that showcases OceanTACO data sources.",
    )
    parser.add_argument(
        "--date",
        type=str,
        default="2025-03-04",
        help="Snapshot date in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--output-png",
        type=Path,
        default=Path("docs/images/oceantaco.png"),
        help="Path to write the PNG asset.",
    )
    parser.add_argument(
        "--output-svg",
        type=Path,
        default=Path("docs/images/oceantaco.svg"),
        help="Path to write the SVG asset. Set to empty string to skip.",
    )
    parser.add_argument(
        "--hf-url",
        type=str,
        default=HF_URL,
        help="OceanTACO HuggingFace base URL.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_svg = args.output_svg
    if str(output_svg).strip() == "":
        output_svg = None

    build_readme(
        date=args.date,
        output_png=args.output_png,
        output_svg=output_svg,
        hf_url=args.hf_url,
    )


if __name__ == "__main__":
    main()
