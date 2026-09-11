"""Exact native-grid rendering and dense-source canonicalisation."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np

from ..geobox import GeoBox, utc_isoformat
from ..registry import ModalitySpec


def canonicalise_dense(dataset, spec: ModalitySpec, *, fallback_time: datetime | None = None):
    """Return a decoded dense source as sorted ``(time, lat, lon)`` xarray data.

    Only singleton dimensions outside the declared grid contract are removed;
    a newly introduced non-singleton dimension is a hard failure rather than a
    silent, arbitrary selection.
    """
    import xarray as xr

    if spec.primary_variable not in dataset:
        raise ValueError(f"{spec.token} asset lacks primary variable {spec.primary_variable!r}.")
    for coordinate in ("lat", "lon"):
        if coordinate not in dataset.coords:
            raise ValueError(f"{spec.token} asset lacks required {coordinate!r} coordinate.")
        if dataset[coordinate].ndim != 1:
            raise ValueError(f"{spec.token} uses non-1-D {coordinate!r}; 0.1.0 supports regular dense grids only.")

    data = dataset[spec.primary_variable]
    for dimension in tuple(data.dims):
        if dimension not in {"time", "lat", "lon"}:
            if data.sizes[dimension] != 1:
                raise ValueError(f"{spec.token} has unsupported non-singleton dimension {dimension!r}.")
            data = data.isel({dimension: 0}, drop=True)
    if "time" not in data.dims:
        if "time" in dataset.coords and dataset["time"].ndim == 0:
            timestamp = dataset["time"].item()
        elif fallback_time is not None:
            timestamp = np.datetime64(fallback_time.replace(tzinfo=None))
        else:
            raise ValueError(f"{spec.token} has no time dimension and no scalar/fallback time.")
        data = data.expand_dims(time=[timestamp])
    data = data.transpose("time", "lat", "lon")
    data = data.sortby("lat")
    lon = np.asarray(data["lon"].values, dtype=np.float64)
    canonical_lon = np.where(lon == 180.0, -180.0, ((lon + 180.0) % 360.0) - 180.0)
    data = data.assign_coords(lon=canonical_lon).sortby("lon")
    if np.unique(np.asarray(data["lon"].values)).size != data.sizes["lon"]:
        raise ValueError(f"{spec.token} contains duplicate longitudes at the antimeridian.")
    # Source units are an asset-level property, validated once where assets are
    # ingested (``scripts/release/queryset_build.check_units``).  Re-checking
    # here would gate every ``__getitem__`` on metadata this path never reads:
    # no registered source requires a value conversion, and the rendered sample
    # carries arrays, not attrs.  A released asset that legitimately declares no
    # units therefore must not fail a training run.
    return xr.DataArray(
        np.asarray(data.values),
        dims=("time", "lat", "lon"),
        coords={"time": data["time"], "lat": data["lat"], "lon": data["lon"]},
        name=spec.primary_variable,
        attrs={**data.attrs, "units": spec.canonical_unit},
    )


def crop_dense(data, box: GeoBox):
    """Crop a canonical dense array and reassemble wrapped boxes in query order."""
    parts = []
    for segment in box.segments():
        # Coordinate selection retains categorical dtypes and touches only the
        # spatial axes.  ``where(..., drop=True)`` would allocate a broadcast
        # mask and upcast integer flag fields to floating point.
        parts.append(data.sel(lat=slice(segment.lat_min, segment.lat_max), lon=slice(segment.lon_min, segment.lon_max)))
    if box.wraps_antimeridian:
        import xarray as xr

        result = xr.concat(parts, dim="lon")
        result = result.assign_coords(lon=box.unwrap_longitudes(result["lon"].values))
        return result
    return parts[0]


def _times(data) -> list[str]:
    return [utc_isoformat(str(value)) for value in np.asarray(data["time"].values)]


def _mask_payload(data, ocean_mask=None) -> dict[str, Any]:
    # Native rendering is an exact decoded crop.  In particular, do not narrow
    # float64 products or coordinates here: callers use this renderer as their
    # bit-for-bit native-grid reference.
    values = np.asarray(data.values)
    source_valid = np.isfinite(values)
    support_mask = np.ones_like(source_valid, dtype=bool)
    lat, lon = np.asarray(data["lat"].values), np.asarray(data["lon"].values)
    payload: dict[str, Any] = {
        "data": values,
        "source_valid": source_valid,
        "support_mask": support_mask,
        "valid_mask": source_valid & support_mask,
        "lat": lat,
        "lon": lon,
        "times": _times(data),
    }
    if ocean_mask is not None:
        mask, in_mask_domain = ocean_mask.nearest_with_domain(lat, lon)
        payload["ocean_mask"] = mask
        payload["in_mask_domain"] = in_mask_domain
        payload["valid_mask"] = payload["valid_mask"] & mask[None, :, :]
    return payload


class Native:
    """Exact crop renderer; output values are unchanged decoded source values."""

    def render(self, data, box: GeoBox, *, ocean_mask=None, **_: Any) -> dict[str, Any]:
        """Crop source-native data without interpolation."""
        return _mask_payload(crop_dense(data, box), ocean_mask=ocean_mask)

    def empty(self) -> dict[str, Any]:
        """Return a structurally valid missing grid source."""
        return {
            "data": np.empty((0, 0, 0), dtype=np.float32),
            "source_valid": np.empty((0, 0, 0), dtype=bool),
            "support_mask": np.empty((0, 0, 0), dtype=bool),
            "valid_mask": np.empty((0, 0, 0), dtype=bool),
            "lat": np.empty((0,), dtype=np.float32),
            "lon": np.empty((0,), dtype=np.float32),
            "times": [],
        }
