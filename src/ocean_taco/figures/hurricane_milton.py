"""Publication-style Hurricane Milton source comparison."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from ..catalog import CatalogConfig
from ..geobox import GeoBox
from ..retrieve import load_bbox_nc

LON_MIN, LON_MAX = -100.0, -75.0
LAT_MIN, LAT_MAX = 15.0, 32.0
GULF_BOX = GeoBox(LON_MIN, LON_MAX, LAT_MIN, LAT_MAX)
DEFAULT_DATES = ("2024-10-05", "2024-10-07", "2024-10-09", "2024-10-10")
PRODUCTS = {"l4_wind": "L4 wind", "l3_ssh": "L3 along-track SSH", "l3_swot": "L3 SWOT SSH"}
MILTON_EYE = {
    "2024-10-04": (-94.70, 20.60), "2024-10-05": (-95.20, 21.10), "2024-10-06": (-95.40, 22.80),
    "2024-10-07": (-93.40, 22.50), "2024-10-08": (-90.40, 21.80), "2024-10-09": (-86.90, 23.00),
    "2024-10-10": (-82.70, 27.20),
}


def load_date(catalog, date: str, *, config: CatalogConfig) -> dict[str, object]:
    """Load the three products used in one Milton figure row."""
    return {token: dataset for token in PRODUCTS if (dataset := load_bbox_nc(catalog, date, GULF_BOX, token, config=config)) is not None}


def close_data(rows: Mapping[str, Mapping[str, object]]) -> None:
    """Close every xarray dataset opened by :func:`load_date`."""
    for row in rows.values():
        for dataset in row.values():
            if callable(close := getattr(dataset, "close", None)):
                close()


def _surface(dataset, variable: str):
    field = dataset[variable]
    for dimension in ("time", "depth"):
        if dimension in field.dims:
            field = field.isel({dimension: 0})
    return field.squeeze()


def _decorate(axis, date: str, labels: bool) -> None:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    axis.set_extent((LON_MIN, LON_MAX, LAT_MIN, LAT_MAX), crs=ccrs.PlateCarree())
    axis.add_feature(cfeature.LAND, facecolor="#e8e8e8", edgecolor="none", zorder=4)
    axis.coastlines(linewidth=0.55, color="#444", zorder=5)
    grid = axis.gridlines(draw_labels=labels, linewidth=0.25, alpha=0.4)
    grid.top_labels = grid.right_labels = False
    track = [MILTON_EYE[key] for key in sorted(MILTON_EYE)]
    axis.plot(*zip(*track, strict=True), "k--", linewidth=0.9, transform=ccrs.PlateCarree(), zorder=6)
    if date in MILTON_EYE:
        axis.scatter(*MILTON_EYE[date], marker="x", s=42, color="black", linewidths=1.3, transform=ccrs.PlateCarree(), zorder=7)


def make_figure(rows: Mapping[str, Mapping[str, object]], dates: Sequence[str]):
    """Return the four-date, three-product projected comparison figure."""
    import cartopy.crs as ccrs
    import matplotlib.pyplot as plt
    import numpy as np
    figure, axes = plt.subplots(len(dates), 3, figsize=(13, 3.5 * len(dates)), squeeze=False,
                                subplot_kw={"projection": ccrs.Mercator()}, layout="constrained")
    ssh_artist = wind_artist = None
    for row_index, date in enumerate(dates):
        for column, token in enumerate(PRODUCTS):
            axis = axes[row_index, column]
            _decorate(axis, date, column == 0 or row_index == len(dates) - 1)
            axis.set_title(PRODUCTS[token] if row_index == 0 else "")
            dataset = rows[date].get(token)
            if dataset is None:
                axis.text(0.5, 0.5, "No product for this date", transform=axis.transAxes, ha="center", va="center")
                continue
            if token == "l4_wind":
                u = _surface(dataset, "eastward_wind_max")
                v = _surface(dataset, "northward_wind_max")
                speed = np.hypot(u, v)
                wind_artist = axis.pcolormesh(u.lon, u.lat, speed, transform=ccrs.PlateCarree(), shading="auto", cmap="BuGn", vmin=0, vmax=40, zorder=1)
                step = max(1, min(u.shape) // 28)
                axis.quiver(u.lon.values[::step], u.lat.values[::step], u.values[::step, ::step], v.values[::step, ::step], transform=ccrs.PlateCarree(), color="white", scale=300, width=0.0025, zorder=3)
            else:
                field = _surface(dataset, "sla_filtered" if token == "l3_ssh" else "ssha_filtered")
                lon, lat = np.meshgrid(field.lon.values, field.lat.values)
                valid = np.isfinite(field.values)
                if valid.any():
                    ssh_artist = axis.scatter(lon[valid], lat[valid], c=field.values[valid], s=0.45 if token == "l3_ssh" else 0.08, cmap="RdBu_r", vmin=-0.7, vmax=0.7, transform=ccrs.PlateCarree(), rasterized=True, zorder=3)
        axes[row_index, 0].text(
            -0.08, 0.5, date,
            transform=axes[row_index, 0].transAxes,
            rotation="vertical", ha="center", va="center", fontweight="bold",
        )
    if wind_artist is not None:
        figure.colorbar(wind_artist, ax=axes[:, 0], orientation="horizontal", pad=0.05, label="daily maximum wind speed [m s$^{-1}$]")
    if ssh_artist is not None:
        figure.colorbar(ssh_artist, ax=axes[:, 1:], orientation="horizontal", pad=0.05, label="SSH anomaly [m]")
    return figure

