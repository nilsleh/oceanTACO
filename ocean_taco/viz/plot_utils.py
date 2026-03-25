"""Reusable cartopy plot helpers for OceanTACO visualizations."""

import os
from pathlib import Path

import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


def configure_cartopy_dir(path: str) -> None:
    """Set CARTOPY_USER_DIR env var and cartopy data_dir config.

    Args:
        path: Directory path for cartopy data/natural earth files.
    """
    p = Path(path).expanduser()
    p.mkdir(parents=True, exist_ok=True)
    os.environ["CARTOPY_USER_DIR"] = str(p)
    cartopy.config["data_dir"] = str(p)


def extract_field(
    nc: xr.Dataset,
    nc_var: str,
    lon_range: tuple[float, float],
    lat_range: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Slice a 2D field from an xr.Dataset.

    Squeezes time/depth dimensions and converts Kelvin to Celsius if needed.

    Args:
        nc: xr.Dataset.
        nc_var: Variable name.
        lon_range: (lon_min, lon_max).
        lat_range: (lat_min, lat_max).

    Returns:
        Tuple of (lons_2d, lats_2d, values_2d).
    """
    lats = nc["lat"].values
    lat_min, lat_max = lat_range
    lat_sl = slice(lat_max, lat_min) if lats[0] > lats[-1] else slice(lat_min, lat_max)

    d = nc[nc_var].sel(lon=slice(lon_range[0], lon_range[1]), lat=lat_sl)
    if "time" in d.dims:
        d = d.isel(time=0)
    if "depth" in d.dims:
        d = d.isel(depth=0)
    d = d.squeeze()

    vals = d.values.astype(float)
    if np.nanmax(vals) > 200:
        vals -= 273.15

    lon_vals = nc["lon"].sel(lon=slice(lon_range[0], lon_range[1])).values
    lat_vals = nc["lat"].sel(lat=lat_sl).values
    lons_2d, lats_2d = np.meshgrid(lon_vals, lat_vals)

    return lons_2d, lats_2d, vals


def plot_field(
    ax,
    lons: np.ndarray,
    lats: np.ndarray,
    vals: np.ndarray,
    cmap: str,
    vmin: float,
    vmax: float,
    add_cbar: bool = True,
    cbar_label: str = "",
    fontsize_cbar: int = 5,
    fontsize_tick: int = 5,
):
    """Plot a pcolormesh field on a cartopy GeoAxes.

    Args:
        ax: Cartopy GeoAxes.
        lons: 2D longitude array.
        lats: 2D latitude array.
        vals: 2D data array.
        cmap: Colormap name.
        vmin: Color scale minimum.
        vmax: Color scale maximum.
        add_cbar: Whether to add a colorbar.
        cbar_label: Colorbar label text.
        fontsize_cbar: Font size for colorbar label.
        fontsize_tick: Font size for colorbar tick labels.

    Returns:
        Pcolormesh mappable.
    """
    mp = ax.pcolormesh(
        lons,
        lats,
        vals,
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        shading="auto",
        rasterized=True,
        zorder=1,
    )
    if add_cbar:
        cb = plt.colorbar(
            mp, ax=ax, orientation="horizontal", pad=0.04, fraction=0.046, shrink=0.8
        )
        cb.set_label(cbar_label, fontsize=fontsize_cbar)
        cb.ax.tick_params(labelsize=fontsize_tick)
    return mp


def style_inset(
    ax,
    extent: list[float],
    title: str,
    left_labels: bool = True,
    right_labels: bool = False,
    fontsize_label: int = 6,
    fontsize_tick: int = 5,
) -> None:
    """Apply consistent coast/gridlines/extent styling to an inset map panel.

    Args:
        ax: Cartopy GeoAxes.
        extent: [lon_min, lon_max, lat_min, lat_max].
        title: Panel title string.
        left_labels: Whether to draw latitude labels on the left side.
        right_labels: Whether to draw latitude labels on the right side.
        fontsize_label: Font size for the panel title.
        fontsize_tick: Font size for gridline tick labels.
    """
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.coastlines(linewidth=0.4, color="#333")
    ax.add_feature(cfeature.LAND, facecolor="#e8e8e8", edgecolor="none", zorder=2)
    gl = ax.gridlines(draw_labels=True, linewidth=0.2, alpha=0.3)
    gl.left_labels = left_labels
    gl.right_labels = right_labels
    gl.top_labels = False
    gl.bottom_labels = False
    gl.xlabel_style = {"size": fontsize_tick}
    gl.ylabel_style = {"size": fontsize_tick}
    ax.set_title(title, fontsize=fontsize_label, pad=3)


def plot_argo_scatter(
    ax,
    ds: xr.Dataset,
    var_name: str,
    lon_range: tuple[float, float],
    lat_range: tuple[float, float],
    cmap: str,
    vmin: float,
    vmax: float,
    cbar_label: str = "",
    fontsize_cbar: int = 5,
    fontsize_tick: int = 5,
):
    """Scatter Argo profile point data on a cartopy GeoAxes.

    Args:
        ax: Cartopy GeoAxes.
        ds: xr.Dataset with Argo profiles (flat N_POINTS dimension).
        var_name: Variable name for the salinity field.
        lon_range: (lon_min, lon_max).
        lat_range: (lat_min, lat_max).
        cmap: Colormap name.
        vmin: Color scale minimum.
        vmax: Color scale maximum.
        cbar_label: Colorbar label text.
        fontsize_cbar: Font size for colorbar label.
        fontsize_tick: Font size for colorbar tick labels.

    Returns:
        PathCollection mappable.
    """
    lons = ds["lon"].values
    lats = ds["lat"].values
    vals = ds[var_name].values.astype(float)

    mask = (
        (lons >= lon_range[0])
        & (lons <= lon_range[1])
        & (lats >= lat_range[0])
        & (lats <= lat_range[1])
        & np.isfinite(vals)
    )
    sc = ax.scatter(
        lons[mask],
        lats[mask],
        c=vals[mask],
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        s=3,
        edgecolors="none",
        zorder=3,
        rasterized=True,
    )
    cb = plt.colorbar(
        sc, ax=ax, orientation="horizontal", pad=0.04, fraction=0.046, shrink=0.8
    )
    cb.set_label(cbar_label, fontsize=fontsize_cbar)
    cb.ax.tick_params(labelsize=fontsize_tick)
    return sc
