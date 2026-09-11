"""Generate figures.."""

import os
from pathlib import Path

import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

from ocean_taco.dataset.dataset import SeaTACODataset


def _configure_cartopy_dir(path: str):
    p = Path(path).expanduser()
    p.mkdir(parents=True, exist_ok=True)
    os.environ["CARTOPY_USER_DIR"] = str(p)
    cartopy.config["data_dir"] = str(p)


def create_vertical_slice_visualization(
    dataset: SeaTACODataset,
    bbox: tuple[float, float, float, float],
    time_slice: tuple[pd.Timestamp, pd.Timestamp],
    variables: list[str],
    figsize: tuple[int, int] = (20, 10),
    save_path: str | None = None,
    dpi: int = 300,
    variable_labels: dict[str, str] | None = None,
    colormaps: dict[str, str] | None = None,
    vmin_vmax: dict[str, tuple[float, float]] | None = None,
    show_colorbars: bool = False,
):
    lon_min, lon_max, lat_min, lat_max = bbox
    n_vars = len(variables)

    default_cmaps = {
        "l4_sss": "viridis",
        "l3_sst": "RdYlBu_r",
        "l4_sst": "RdYlBu_r",
        "l4_ssh": "RdBu_r",
        "l3_ssh": "RdBu_r",
        "l2_swot": "RdBu_r",
        "l3_swot": "RdBu_r",
        "glorys_ssh": "RdBu_r",
        "glorys_sst": "RdYlBu_r",
        "glorys_sss": "viridis",
        "argo_sst": "RdYlBu_r",
        "argo_sss": "viridis",
    }
    colormaps = colormaps or default_cmaps
    vmin_vmax = vmin_vmax or {}
    variable_labels = variable_labels or {}

    lon_extent = lon_max - lon_min
    slice_width = lon_extent / n_vars

    fig = plt.figure(figsize=figsize, constrained_layout=False)

    left_margin = 0.02
    right_margin = 0.98
    top_margin = 0.94 if show_colorbars else 0.96

    if show_colorbars:
        bottom_cbar = 0.05
        cbar_height = 0.025
        gap_between_plot_cbar = 0.01
        bottom_plot = bottom_cbar + cbar_height + gap_between_plot_cbar
    else:
        bottom_plot = 0.04

    total_plot_width = right_margin - left_margin
    plot_height = top_margin - bottom_plot

    main_ax = fig.add_axes(
        [left_margin, bottom_plot, total_plot_width, plot_height],
        projection=ccrs.PlateCarree(),
    )
    main_ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())

    main_ax.add_feature(cfeature.OCEAN.with_scale("50m"), facecolor="white", zorder=1)
    main_ax.add_feature(
        cfeature.LAND.with_scale("50m"),
        facecolor="lightgray",
        edgecolor="none",
        zorder=3,
    )
    main_ax.coastlines(resolution="10m", linewidth=0.8, color="black", zorder=5)

    main_ax.spines["geo"].set_visible(False)
    for spine in main_ax.spines.values():
        spine.set_visible(False)
    main_ax.patch.set_visible(False)

    for i, var in enumerate(variables):
        slice_lon_min = lon_min + i * slice_width
        slice_lon_max = lon_min + (i + 1) * slice_width
        slice_bbox = (slice_lon_min, slice_lon_max, lat_min, lat_max)

        print(f"Processing {var}: lon=[{slice_lon_min:.1f}, {slice_lon_max:.1f}]")

        if var in dataset.input_variables:
            sample = dataset.get_region(bbox=slice_bbox, time_slice=time_slice)
            var_data = sample["inputs"][var]
        elif var in dataset.target_variables:
            sample = dataset.get_region(bbox=slice_bbox, time_slice=time_slice)
            var_data = sample["targets"][var]
        else:
            raise ValueError(f"Variable {var} not in input or target variables")

        mappable = _plot_variable_slice(
            ax=main_ax,
            var_data=var_data,
            cmap=colormaps.get(var, "viridis"),
            vmin_vmax=vmin_vmax.get(var, (None, None)),
        )

        if mappable is None:
            center_lon = slice_lon_min + slice_width / 2
            center_lat = lat_min + (lat_max - lat_min) / 2
            main_ax.text(
                center_lon,
                center_lat,
                "No data",
                ha="center",
                va="center",
                transform=ccrs.PlateCarree(),
                fontsize=12,
                zorder=5,
            )
        else:
            clip_rect = Rectangle(
                (slice_lon_min, lat_min),
                slice_width,
                lat_max - lat_min,
                transform=ccrs.PlateCarree(),
                facecolor="none",
                edgecolor="none",
            )
            main_ax.add_patch(clip_rect)
            mappable.set_clip_path(clip_rect)

        if show_colorbars:
            panel_width = total_plot_width / n_vars
            ax_left = left_margin + i * panel_width
            cax = fig.add_axes([ax_left, bottom_cbar, panel_width, cbar_height])

            if mappable is not None:
                cbar = plt.colorbar(mappable, cax=cax, orientation="horizontal")
                label = variable_labels.get(var, var)
                cax.set_xlabel(label, fontsize=8, fontweight="bold")
                cbar.ax.tick_params(labelsize=6)
        else:
            label = variable_labels.get(var, var)
            center_lon = slice_lon_min + slice_width / 2
            main_ax.text(
                center_lon,
                lat_min - 0.02 * (lat_max - lat_min),
                label,
                ha="center",
                va="top",
            )

    # time_str = (
    #     f"{time_slice[0].strftime('%Y-%m-%d')} to {time_slice[1].strftime('%Y-%m-%d')}"
    # )
    # fig.text(
    #     0.5,
    #     0.98,
    #     f"Sea Surface State Dataset: {time_str}",
    #     ha="center",
    #     va="top",
    #     fontsize=13,
    #     fontweight="bold",
    # )

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(
            save_path, dpi=dpi, bbox_inches="tight", pad_inches=0.0, facecolor="white"
        )
        print(f"Saved figure to {save_path}")

    return fig


def _plot_variable_slice(
    ax: plt.Axes,
    var_data: dict,
    cmap: str,
    vmin_vmax: tuple[float | None, float | None],
) -> plt.cm.ScalarMappable | None:
    data = var_data["data"].detach().cpu().numpy()

    if data.size == 0:
        return None

    lats = var_data["coords"][0].detach().cpu().numpy()
    lons = var_data["coords"][1].detach().cpu().numpy()
    vmin, vmax = vmin_vmax

    if data.ndim == 3:
        data = data[-1]

    if data.ndim == 2 and lats.ndim == 2:
        valid_mask = np.isfinite(data) & np.isfinite(lats) & np.isfinite(lons)
        if not valid_mask.any():
            return None
        mappable = ax.scatter(
            lons[valid_mask],
            lats[valid_mask],
            c=data[valid_mask],
            transform=ccrs.PlateCarree(),
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            s=0.5,
            alpha=0.8,
            rasterized=True,
            zorder=4,
        )
    elif data.ndim == 2 and lats.ndim == 1:
        data_masked = np.ma.masked_invalid(data)
        mappable = ax.pcolormesh(
            lons,
            lats,
            data_masked,
            transform=ccrs.PlateCarree(),
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            rasterized=True,
            shading="auto",
            zorder=2,
        )
    elif data.ndim == 1:
        valid_mask = np.isfinite(data) & np.isfinite(lats) & np.isfinite(lons)
        if not valid_mask.any():
            return None
        mappable = ax.scatter(
            lons[valid_mask],
            lats[valid_mask],
            c=data[valid_mask],
            transform=ccrs.PlateCarree(),
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            s=1.5,
            alpha=0.9,
            rasterized=True,
            zorder=4,
        )
    else:
        return None

    return mappable


if __name__ == "__main__":
    _configure_cartopy_dir("./.cartopy")

    taco_path = "/p/scratch/hai_uqmethodbox/data/ssh_dataset_taco/__TACOCAT__"

    all_vars = ["l4_sst", "l4_ssh", "l3_swot", "l3_ssh", "glorys_ssh", "glorys_sst"]
    all_vars = ["glorys_ssh", "l4_ssh", "l3_swot", "l3_ssh"]

    dataset = SeaTACODataset(
        taco_zip_path=taco_path, input_variables=all_vars, target_variables=[]
    )

    print("Creating North Atlantic figure...")
    fig = create_vertical_slice_visualization(
        dataset=dataset,
        bbox=(-90, 0, 10, 60),
        time_slice=(pd.Timestamp("2023-04-01"), pd.Timestamp("2023-04-03")),
        variables=all_vars,
        save_path="manuscript_north_atlantic.png",
        dpi=300,
        show_colorbars=False,
    )
    plt.close(fig)

    print("Creating custom Pacific figure...")
    fig = create_vertical_slice_visualization(
        dataset=dataset,
        bbox=(120, 179, 20, 60),
        time_slice=(pd.Timestamp("2023-04-01"), pd.Timestamp("2023-04-03")),
        variables=all_vars,
        figsize=(22, 10),
        save_path="manuscript_pacific_custom.png",
        dpi=300,
        show_colorbars=False,
    )
    plt.close(fig)

    print("Manuscript figures created successfully!")
