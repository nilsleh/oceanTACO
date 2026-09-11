"""Plot map and spatial regions."""

import os
from pathlib import Path

import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt


def _configure_cartopy_dir(path: str):
    """Configure cartopy data directory."""
    p = Path(path).expanduser()
    p.mkdir(parents=True, exist_ok=True)
    os.environ["CARTOPY_USER_DIR"] = str(p)
    cartopy.config["data_dir"] = str(p)


_configure_cartopy_dir("./.cartopy")

# Spatial region definitions
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


def plot_minimalist_regions(regions):
    # Robinson projection for a professional global view
    fig = plt.figure(figsize=(14, 8), dpi=300)
    ax = plt.axes(projection=ccrs.Robinson())

    # Map Styling
    ax.set_global()
    ax.add_feature(cfeature.LAND, facecolor="#f7f7f7", edgecolor="none")
    ax.add_feature(cfeature.OCEAN, facecolor="#ffffff")
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, color="#444444")

    # Regional Styling
    # Using a professional "Sky Blue" or "Alice Blue"
    region_color = "#b0e0e6"  # PowderBlue

    for name, bounds in regions.items():
        lon0, lon1 = bounds["lon"]
        lat0, lat1 = bounds["lat"]

        # Rectangle with dashed lines
        rect = mpatches.Rectangle(
            (lon0, lat0),
            lon1 - lon0,
            lat1 - lat0,
            linewidth=1.5,
            edgecolor="#2c3e50",  # Dark slate for contrast
            facecolor=region_color,
            alpha=0.25,
            linestyle="--",  # The requested dashed lines
            transform=ccrs.PlateCarree(),
            zorder=3,
        )
        ax.add_patch(rect)

        # Center Labels
        ax.text(
            (lon0 + lon1) / 2,
            (lat0 + lat1) / 2,
            name.replace("_", "\n"),
            color="#2c3e50",
            weight="bold",
            fontsize=16,
            ha="center",
            va="center",
            transform=ccrs.PlateCarree(),
            zorder=10,
            bbox=dict(
                facecolor="white", alpha=0.8, edgecolor="none", boxstyle="round,pad=0.2"
            ),
        )

    # Gridlines
    gl = ax.gridlines(
        draw_labels=True, linewidth=0.3, color="gray", alpha=0.2, linestyle="-"
    )
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {"size": 14}
    gl.ylabel_style = {"size": 14}

    # plt.title("Spatial Regions", fontsize=16, pad=25, color="#333333")

    plt.tight_layout()
    plt.savefig("spatial_regions.png", bbox_inches="tight")
    plt.show()


plot_minimalist_regions(SPATIAL_REGIONS)
