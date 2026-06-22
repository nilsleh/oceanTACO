"""Shared helpers for OceanTACO ClimateBenchPress-style benchmarks."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from ocean_taco.generate_dataset.format_encoding import clear_encoding


def ensure_zarr_available() -> None:
    """Raise a clear error if zarr is not installed."""
    if importlib.util.find_spec("zarr") is None:
        raise RuntimeError(
            "zarr is required for benchmark packaging. Install the benchmark "
            "dependencies before running package_standardized.py."
        )


def coord_hash(values: np.ndarray) -> str:
    """Return a stable hash for one coordinate array."""
    arr = np.asarray(values, dtype=np.float64)
    digest = hashlib.sha256(np.ascontiguousarray(arr).view(np.uint8)).hexdigest()
    return digest


def normalize_export_dataset(ds: xr.Dataset) -> xr.Dataset:
    """Prepare a pre-encoding dataset for benchmark export."""
    ds_out = clear_encoding(ds.copy(deep=True))
    keep_vars = []

    for var_name, data_array in ds_out.data_vars.items():
        dtype = data_array.dtype
        if np.issubdtype(dtype, np.timedelta64) or np.issubdtype(dtype, np.datetime64):
            continue
        if np.issubdtype(dtype, np.number):
            keep_vars.append(var_name)
            if np.issubdtype(dtype, np.floating):
                ds_out[var_name] = data_array.astype(np.float32)

    ds_out = ds_out[keep_vars]
    return ds_out


def load_json(path: Path) -> dict:
    """Load a JSON file from disk."""
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def dump_json(path: Path, payload: dict | list) -> None:
    """Write JSON with stable formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def reduce_to_lat_lon_2d(data_array: xr.DataArray, var_name: str) -> xr.DataArray:
    """Reduce a DataArray to 2D lat/lon by selecting the first index on extras."""
    if "lat" not in data_array.dims or "lon" not in data_array.dims:
        raise ValueError(
            f"{var_name} must contain lat/lon dimensions; found {data_array.dims}"
        )

    for dim_name in [dim for dim in data_array.dims if dim not in ("lat", "lon")]:
        data_array = data_array.isel({dim_name: 0})

    data_array = data_array.transpose("lat", "lon")
    return data_array.astype(np.float32)


def collect_sidecar_paths(root: Path) -> list[Path]:
    """Return all benchmark sidecar metadata paths under one benchmark root."""
    tile_root = root / "tiles"
    if not tile_root.exists():
        return []
    return sorted(tile_root.glob("**/*.json"))


def dataframe_from_records(records: list[dict]) -> pd.DataFrame:
    """Create a deterministic manifest DataFrame."""
    if not records:
        return pd.DataFrame()

    df = pd.DataFrame.from_records(records)
    sort_columns = [col for col in ["dataset_name", "region", "date", "tile_path"] if col in df]
    if sort_columns:
        df = df.sort_values(sort_columns).reset_index(drop=True)
    return df
