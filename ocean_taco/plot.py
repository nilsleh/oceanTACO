"""Optional, lazily imported visualisation helpers for rendered samples."""

from __future__ import annotations

from typing import Any

import numpy as np


def _array(value: Any) -> np.ndarray:
    """Convert NumPy or CPU torch values without importing torch eagerly."""
    detach = getattr(value, "detach", None)
    if callable(detach):
        value = detach().cpu().numpy()
    return np.asarray(value)


def plot_ocean_sample(sample: dict[str, Any], token: str, *, time_index: int = 0, component: int = 0, ax=None):
    """Plot one grid or ragged-point source record from a rendered sample.

    Matplotlib is imported only when this optional helper is called.  Vector
    pairs are plotted by component (0=eastward, 1=northward); ragged point
    sources are plotted at their geographical locations.
    """
    try:
        import matplotlib.pyplot as pyplot
    except ImportError as error:  # pragma: no cover - depends on optional extra
        raise ImportError("plot_ocean_sample requires the 'viz' extra (matplotlib).") from error
    if token not in sample:
        raise ValueError(f"Sample has no source token {token!r}.")
    record = sample[token]
    if ax is None:
        _, ax = pyplot.subplots()
    data = _array(record["data"])
    lat, lon = _array(record["lat"]), _array(record["lon"])
    if "pres" in record:
        artist = ax.scatter(lon, lat, c=data)
        ax.set_xlabel("longitude")
        ax.set_ylabel("latitude")
        return artist
    if data.ndim == 4:
        if component not in (0, 1):
            raise ValueError("component must be 0 (eastward) or 1 (northward).")
        image = data[time_index, component]
    elif data.ndim == 3:
        image = data[time_index]
    else:
        raise ValueError("plot_ocean_sample expects a grid (T,H,W) or vector pair (T,2,H,W).")
    if time_index < 0 or time_index >= data.shape[0]:
        raise ValueError("time_index is outside the source record.")
    extent = (float(lon[0]), float(lon[-1]), float(lat[0]), float(lat[-1])) if lat.size and lon.size else None
    artist = ax.imshow(image, origin="lower", extent=extent, aspect="auto")
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    return artist
