"""Projected SSH cross-product comparison for Hurricane Milton."""

from __future__ import annotations

from collections.abc import Mapping

from ..catalog import CatalogConfig
from ..retrieve import load_bbox_nc
from .hurricane_milton import GULF_BOX, LAT_MAX, LAT_MIN, LON_MAX, LON_MIN, MILTON_EYE

PRODUCTS = {
    "l4_ssh": ("sla", "L4 DUACS", "#264653"),
    "l3_ssh": ("sla_filtered", "L3 along-track", "#f4a261"),
    "l3_swot": ("ssha_filtered", "L3 SWOT", "#e63946"),
}


def load_products(catalog, date: str, *, config: CatalogConfig) -> dict[str, object]:
    """Load all SSH products, including dense L3 SWOT, for one comparison."""
    return {token: dataset for token in PRODUCTS if (dataset := load_bbox_nc(catalog, date, GULF_BOX, token, config=config)) is not None}


def close_products(data: Mapping[str, object]) -> None:
    """Close datasets opened by :func:`load_products`."""
    for dataset in data.values():
        if callable(close := getattr(dataset, "close", None)):
            close()


def _surface(dataset, variable: str):
    field = dataset[variable]
    for dimension in ("time", "depth"):
        if dimension in field.dims:
            field = field.isel({dimension: 0})
    return field.squeeze()


def _pairs(reference, observation):
    """Interpolate L4 data to valid L3 points without filling missing values."""
    import numpy as np
    from scipy.interpolate import RegularGridInterpolator
    values = np.asarray(reference.values, dtype=float)
    lat = np.asarray(reference.lat.values, dtype=float)
    if lat[0] > lat[-1]:
        lat, values = lat[::-1], values[::-1]
    interpolate = RegularGridInterpolator((lat, reference.lon.values), values, bounds_error=False, fill_value=np.nan)
    lon, latitude = np.meshgrid(observation.lon.values, observation.lat.values)
    observed = np.asarray(observation.values, dtype=float).ravel()
    predicted = interpolate(np.column_stack((latitude.ravel(), lon.ravel())))
    valid = np.isfinite(observed) & np.isfinite(predicted)
    return observed[valid], predicted[valid]


def make_figure(data: Mapping[str, object], date: str):
    """Return projected overlay and deterministic correlation/RMSE comparison."""
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.lines import Line2D
    figure = plt.figure(figsize=(13, 5), layout="constrained")
    map_axis = figure.add_subplot(1, 2, 1, projection=ccrs.Mercator())
    scatter_axis = figure.add_subplot(1, 2, 2)
    map_axis.set_extent((LON_MIN, LON_MAX, LAT_MIN, LAT_MAX), crs=ccrs.PlateCarree())
    map_axis.add_feature(cfeature.LAND, facecolor="#e8e8e8", edgecolor="none", zorder=4)
    map_axis.coastlines(linewidth=0.55, color="#444", zorder=5)
    grid = map_axis.gridlines(draw_labels=True, linewidth=0.25, alpha=0.4)
    grid.top_labels = grid.right_labels = False
    fields = {token: _surface(dataset, PRODUCTS[token][0]) for token, dataset in data.items()}
    artist = None
    if "l4_ssh" in fields:
        field = fields["l4_ssh"]
        artist = map_axis.pcolormesh(field.lon, field.lat, field, transform=ccrs.PlateCarree(), shading="auto",
                                     cmap="RdBu_r", vmin=-0.7, vmax=0.7, alpha=0.45, rasterized=True, zorder=1)
    for token, size in (("l3_ssh", 2.5), ("l3_swot", 0.2)):
        if token not in fields:
            continue
        field = fields[token]
        lon, lat = np.meshgrid(field.lon.values, field.lat.values)
        valid = np.isfinite(field.values)
        if valid.any():
            artist = map_axis.scatter(lon[valid], lat[valid], c=field.values[valid], s=size, cmap="RdBu_r",
                                      vmin=-0.7, vmax=0.7, transform=ccrs.PlateCarree(), rasterized=True, zorder=3)
    if date in MILTON_EYE:
        map_axis.scatter(*MILTON_EYE[date], marker="x", color="black", s=65, linewidths=1.6,
                         transform=ccrs.PlateCarree(), zorder=6)
    handles = [Line2D([0], [0], color=color, lw=5 if token == "l4_ssh" else 0,
                      marker=None if token == "l4_ssh" else "o", markersize=5, label=label)
               for token, (_, label, color) in PRODUCTS.items() if token in fields]
    handles.append(Line2D([0], [0], marker="x", color="black", linestyle="none", label="Milton eye"))
    map_axis.legend(handles=handles, loc="lower left", framealpha=0.9)
    map_axis.set_title(f"SSH products — {date}")
    if artist is not None:
        figure.colorbar(artist, ax=map_axis, orientation="horizontal", pad=0.06, label="SSH anomaly [m]")
    if "l4_ssh" in fields:
        for token in ("l3_ssh", "l3_swot"):
            if token not in fields:
                continue
            observed, predicted = _pairs(fields["l4_ssh"], fields[token])
            if observed.size < 3:
                continue
            rng = np.random.default_rng(42)
            index = rng.choice(observed.size, min(observed.size, 5000), replace=False)
            correlation = np.corrcoef(observed, predicted)[0, 1]
            rmse = np.sqrt(np.mean((observed - predicted) ** 2))
            scatter_axis.scatter(observed[index], predicted[index], s=1, alpha=0.3, color=PRODUCTS[token][2],
                                 rasterized=True, label=f"{PRODUCTS[token][1]} (r={correlation:.3f}, RMSE={rmse:.3f} m)")
    scatter_axis.axline((0, 0), slope=1, color="black", linestyle="--", linewidth=0.8)
    scatter_axis.set(xlim=(-0.8, 0.8), ylim=(-0.8, 0.8), xlabel="L3 observation [m]",
                     ylabel="L4 DUACS [m]", title="L4 versus L3 SSH")
    scatter_axis.grid(alpha=0.2)
    scatter_axis.legend(loc="upper left", markerscale=4)
    return figure

