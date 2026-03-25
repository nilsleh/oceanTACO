"""SSH-SST vs SSH-SSS coupling comparison: Gulf Stream vs Indian Ocean.

Generates a 2x4 figure with spatial correlation maps and scatter plots showing
that SST dominates SSH variability in the Gulf Stream while SSS plays a larger
role in the Indian Ocean.
"""

import argparse
from datetime import date, timedelta

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import xarray as xr

from ocean_taco.dataset.retrieve import HF_DEFAULT_URL, load_bbox_nc, load_hf_dataset


def _configure_cartopy_dir(path: str):
    """Configure cartopy data directory."""
    import os
    from pathlib import Path

    import cartopy
    p = Path(path).expanduser()
    p.mkdir(parents=True, exist_ok=True)
    os.environ["CARTOPY_USER_DIR"] = str(p)
    cartopy.config["data_dir"] = str(p)


_configure_cartopy_dir("./.cartopy")

# Plotting style
sns.set_context("paper", font_scale=1.2)
plt.rcParams["font.family"] = "serif"

# Region definitions: (lon_min, lon_max, lat_min, lat_max)
REGIONS = {
    "Gulf Stream": (-80, -40, 10, 50),
    "Indian Ocean": (55, 95, -25, 15),
}


def _september_dates():
    """Return list of date strings for all days in September 2023."""
    start = date(2023, 9, 1)
    return [(start + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(30)]


def load_region_data(dataset_hf, dates, region_label, lon_slice, lat_slice, cache_dir):
    """Load SSH, SST, SSS stacks for a region over multiple dates.

    Args:
        dataset_hf: TacoDataset catalog object.
        dates: List of date strings YYYY-MM-DD.
        region_label: Human-readable region name, e.g. 'Gulf Stream'.
        lon_slice: (lon_min, lon_max) tuple for cropping.
        lat_slice: (lat_min, lat_max) tuple for cropping.
        cache_dir: Optional local cache directory.

    Returns:
        Tuple of (ssh, sst, sss) DataArrays with dims (time, lat, lon),
        or (None, None, None) if no valid days.
    """
    ssh_days, sst_days, sss_days = [], [], []

    lon_min, lon_max = lon_slice
    lat_min, lat_max = lat_slice
    query_bbox = (lon_min, lat_min, lon_max, lat_max)

    for d in dates:
        ds_ssh = load_bbox_nc(
            dataset_hf, d, query_bbox, data_source="l4_ssh", cache_dir=cache_dir
        )
        ds_sst_raw = load_bbox_nc(
            dataset_hf, d, query_bbox, data_source="l4_sst", cache_dir=cache_dir
        )
        ds_sss_raw = load_bbox_nc(
            dataset_hf, d, query_bbox, data_source="l4_sss", cache_dir=cache_dir
        )

        if ds_ssh is None or ds_sst_raw is None or ds_sss_raw is None:
            print(f"  Missing data for {region_label} on {d}, skipping.")
            continue

        try:
            # Interpolate only over lat/lon to avoid time-mismatch NaNs:
            # SSH timestamps are midnight, SST/SSS are noon; interp_like
            # would try to interpolate the time dimension too, producing
            # all-NaN output when the two timestamps differ.
            ds_sst = ds_sst_raw.interp(
                lat=ds_ssh.lat, lon=ds_ssh.lon, method="nearest"
            )
            ds_sss = ds_sss_raw.interp(
                lat=ds_ssh.lat, lon=ds_ssh.lon, method="nearest"
            )

            # Sort coords so slice works regardless of ascending/descending storage
            ds_ssh = ds_ssh.sortby(["lat", "lon"])
            ds_sst = ds_sst.sortby(["lat", "lon"])
            ds_sss = ds_sss.sortby(["lat", "lon"])

            # Crop to sub-bbox
            ds_ssh = ds_ssh.sel(
                lon=slice(*lon_slice), lat=slice(*lat_slice)
            )
            ds_sst = ds_sst.sel(
                lon=slice(*lon_slice), lat=slice(*lat_slice)
            )
            ds_sss = ds_sss.sel(
                lon=slice(*lon_slice), lat=slice(*lat_slice)
            )

            if ds_ssh["lon"].size == 0 or ds_ssh["lat"].size == 0:
                print(f"  Empty spatial slice for {region_label} on {d}, skipping.")
                continue

            ssh_sq = ds_ssh["sla"].squeeze()
            # SST/SSS files use noon timestamps while SSH uses midnight.
            # Reassign to SSH's time so all three stacks share the same time
            # axis; otherwise xr.corr aligns them and finds no overlap → all NaN.
            sst_days.append(
                (ds_sst["analysed_sst"]).squeeze().assign_coords(time=ssh_sq.time)
            )
            sss_days.append(
                ds_sss["sos"].squeeze().assign_coords(time=ssh_sq.time)
            )
            ssh_days.append(ssh_sq)
        except Exception as e:
            print(f"  Error processing {region_label} on {d}: {e}")

    if not ssh_days:
        return None, None, None

    ssh = xr.concat(ssh_days, dim="time")
    sst = xr.concat(sst_days, dim="time")
    sss = xr.concat(sss_days, dim="time")

    return ssh, sst, sss


def _valid_mask(a, b):
    """Pixels with no NaN across all days for two variables."""
    return (~a.isnull().any("time") & ~b.isnull().any("time")).values


def _add_map(ax, lons, lats, r_vals, mask, title):
    """Draw a Cartopy correlation map onto ax."""
    r_plot = np.where(mask, r_vals, np.nan)

    ax.set_extent(
        [lons.min(), lons.max(), lats.min(), lats.max()],
        crs=ccrs.PlateCarree(),
    )
    ax.add_feature(cfeature.LAND, facecolor="#f0ece3", zorder=1)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, zorder=2)

    gl = ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=True,
        linewidth=0.4,
        color="gray",
        alpha=0.5,
        linestyle="--",
    )
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {"size": 7}
    gl.ylabel_style = {"size": 7}

    pcm = ax.pcolormesh(
        lons,
        lats,
        r_plot,
        vmin=-1,
        vmax=1,
        cmap="RdBu_r",
        transform=ccrs.PlateCarree(),
        zorder=0,
    )

    ax.set_title(title, fontsize=10)
    return pcm


def _add_scatter(ax, x_all, y_all, xlabel, ylabel):
    """Draw hexbin scatter onto ax."""
    mask = ~np.isnan(x_all) & ~np.isnan(y_all)
    x_c = x_all[mask]
    y_c = y_all[mask]

    hb = ax.hexbin(
        x_c,
        y_c,
        gridsize=60,
        cmap="YlGnBu",
        mincnt=1,
        bins="log",
        linewidths=0,
    )

    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(True, alpha=0.3)

    return hb


def generate_figure(dataset_hf, dates, cache_dir=None, output="sst_sss_coupling.png"):
    """Build and save the 2x4 SSH-SST/SSS coupling figure.

    Args:
        dataset_hf: TacoDataset catalog object.
        dates: List of date strings YYYY-MM-DD.
        cache_dir: Optional local cache directory.
        output: Output PNG path.
    """
    fig = plt.figure(figsize=(19, 9))

    # 2 rows x 4 cols; cols 0-1 are maps (need projection), cols 2-3 are plain axes
    map_proj = ccrs.PlateCarree()

    axes = []
    for row in range(2):
        row_axes = []
        for col in range(4):
            if col < 2:
                ax = fig.add_subplot(2, 4, row * 4 + col + 1, projection=map_proj)
            else:
                ax = fig.add_subplot(2, 4, row * 4 + col + 1)
            row_axes.append(ax)
        axes.append(row_axes)

    region_names = list(REGIONS.keys())
    last_pcm = None
    last_hb = None

    for row_idx, region_label in enumerate(region_names):
        lon_min, lon_max, lat_min, lat_max = REGIONS[region_label]
        lon_slice = (lon_min, lon_max)
        lat_slice = (lat_min, lat_max)

        print(f"\nLoading data for {region_label}...")
        ssh, sst, sss = load_region_data(
            dataset_hf, dates, region_label, lon_slice, lat_slice, cache_dir
        )

        if ssh is None:
            print(f"  No data loaded for {region_label}, skipping row.")
            for ax in axes[row_idx]:
                ax.text(0.5, 0.5, "No Data", ha="center", transform=ax.transAxes)
            continue

        lons = ssh.lon.values
        lats = ssh.lat.values
        mask_sst = _valid_mask(ssh, sst)
        mask_sss = _valid_mask(ssh, sss)

        # Pixel-wise correlations
        r_sst = xr.corr(ssh, sst, dim="time").values
        r_sss = xr.corr(ssh, sss, dim="time").values

        # Map: SSH-SST
        pcm = _add_map(
            axes[row_idx][0], lons, lats, r_sst, mask_sst, "SSH – SST correlation"
        )
        last_pcm = pcm

        # Map: SSH-SSS
        _add_map(axes[row_idx][1], lons, lats, r_sss, mask_sss, "SSH – SSS correlation")

        # Scatter: SSH-SST
        ssh_sst = ssh.values[:, mask_sst].flatten()
        hb = _add_scatter(
            axes[row_idx][2], ssh_sst, sst.values[:, mask_sst].flatten(), "SLA [m]", "SST [°C]"
        )
        last_hb = hb

        # Scatter: SSH-SSS
        ssh_sss = ssh.values[:, mask_sss].flatten()
        _add_scatter(axes[row_idx][3], ssh_sss, sss.values[:, mask_sss].flatten(), "SLA [m]", "SSS [PSU]")

        if row_idx == len(region_names) - 1:
            axes[row_idx][2].set_xlabel("")
            axes[row_idx][3].set_xlabel("")

        # Add scatter title
        axes[row_idx][2].set_title("SSH vs SST scatter", fontsize=10)
        axes[row_idx][3].set_title("SSH vs SSS scatter", fontsize=10)

        # Row label — anchored to the left map axis so there's no gap
        axes[row_idx][0].annotate(
            region_label,
            xy=(-0.12, 0.5),
            xycoords="axes fraction",
            va="center",
            ha="right",
            fontsize=12,
            fontweight="bold",
            rotation="vertical",
        )

    # Shared colorbars
    if last_pcm is not None:
        map_axes = [axes[r][c] for r in range(2) for c in range(2)]
        cb_map = fig.colorbar(
            last_pcm, ax=map_axes, orientation="horizontal", pad=0.05, shrink=0.8
        )
        cb_map.set_label("Pearson r", fontsize=10)

    if last_hb is not None:
        scatter_axes = [axes[r][c] for r in range(2) for c in range(2, 4)]
        cb_sc = fig.colorbar(
            last_hb, ax=scatter_axes, orientation="horizontal", pad=0.05, shrink=0.8
        )
        cb_sc.set_label("Log count", fontsize=10)

    # plt.suptitle(
    #     "SSH–SST vs SSH–SSS coupling: Gulf Stream vs Indian Ocean (Sep 2023)",
    #     fontsize=13,
    #     y=0.98,
    # )

    plt.savefig(output, dpi=300, bbox_inches="tight")
    print(f"\nSaved figure to {output}")


def main():
    """Entry point for SSH-SST/SSS coupling figure."""
    parser = argparse.ArgumentParser(
        description="SSH-SST vs SSH-SSS coupling comparison figure.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--hf-url", default=HF_DEFAULT_URL, help="HuggingFace dataset URL.")
    parser.add_argument("--cache-dir", default=None, help="Local cache directory.")
    parser.add_argument("--output", default="sst_sss_coupling.png", help="Output PNG path.")
    args = parser.parse_args()

    dates = _september_dates()
    dataset_hf = load_hf_dataset(args.hf_url)
    generate_figure(dataset_hf, dates, cache_dir=args.cache_dir, output=args.output)


if __name__ == "__main__":
    main()
