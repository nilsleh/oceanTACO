"""Package exported OceanTACO benchmark tiles into standardized.zarr datasets."""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from ocean_taco.benchmarks.climatebenchpress.config import (
    BENCHMARK_MODALITY_CONFIGS,
    DATASET_NAME_TO_MODALITY,
)
from ocean_taco.benchmarks.climatebenchpress.error_bounds import write_error_bounds_file
from ocean_taco.benchmarks.climatebenchpress.export_subset import (
    collect_export_records,
    write_manifest,
)
from ocean_taco.benchmarks.climatebenchpress.utils import (
    dataframe_from_records,
    ensure_zarr_available,
    netcdf_engine,
    reduce_to_lat_lon_2d,
)


def _load_manifest(root: Path) -> pd.DataFrame:
    records = collect_export_records(root)
    df = dataframe_from_records(records)
    if df.empty:
        raise ValueError(f"No benchmark export records found under {root}")
    return df


def _validate_complete_grid(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    regions = sorted(df["region"].unique())
    dates = sorted(df["date"].unique())
    expected = set(product(regions, dates))
    actual = set(zip(df["region"], df["date"]))
    missing = sorted(expected - actual)
    if missing:
        raise ValueError(f"Benchmark subset is missing region/date tiles: {missing}")
    return regions, dates


@dataclass(frozen=True)
class RegionGrid:
    """Coordinate vectors and 2D tile shape for one stable regional grid."""

    lat_values: np.ndarray
    lon_values: np.ndarray
    y_size: int
    x_size: int


def _validate_region_grids(df: pd.DataFrame) -> None:
    for region, group in df.groupby("region", sort=True):
        if group["lat_hash"].nunique() != 1 or group["lon_hash"].nunique() != 1:
            raise ValueError(
                f"All tiles for region '{region}' in one benchmark dataset must share one lat/lon grid."
            )


def _load_region_grids(df: pd.DataFrame, primary_var: str) -> tuple[dict[str, RegionGrid], int, int]:
    region_grids: dict[str, RegionGrid] = {}
    max_y_size = 0
    max_x_size = 0

    for region, group in df.groupby("region", sort=True):
        sample_path = Path(group.iloc[0]["tile_path"])
        with xr.open_dataset(sample_path, engine=netcdf_engine()) as ds_sample:
            if primary_var not in ds_sample:
                raise ValueError(
                    f"Primary variable '{primary_var}' not found in exported tile {sample_path}."
                )
            lat_values = np.asarray(ds_sample["lat"].values)
            lon_values = np.asarray(ds_sample["lon"].values)
            sample_array = reduce_to_lat_lon_2d(ds_sample[primary_var], primary_var)
            y_size, x_size = sample_array.shape

        region_grids[region] = RegionGrid(
            lat_values=lat_values,
            lon_values=lon_values,
            y_size=y_size,
            x_size=x_size,
        )
        max_y_size = max(max_y_size, y_size)
        max_x_size = max(max_x_size, x_size)

    return region_grids, max_y_size, max_x_size


def _build_standardized_dataset(
    df: pd.DataFrame,
    primary_var: str,
    dataset_name: str,
) -> xr.Dataset:
    regions, dates = _validate_complete_grid(df)
    _validate_region_grids(df)
    region_grids, max_y_size, max_x_size = _load_region_grids(df, primary_var)

    data = np.full((len(regions), len(dates), 1, max_y_size, max_x_size), np.nan, dtype=np.float32)
    lat_coord = np.full((len(regions), max_y_size), np.nan, dtype=np.float64)
    lon_coord = np.full((len(regions), max_x_size), np.nan, dtype=np.float64)
    region_index = {region: idx for idx, region in enumerate(regions)}
    date_index = {date: idx for idx, date in enumerate(dates)}

    for region, idx in region_index.items():
        region_grid = region_grids[region]
        lat_coord[idx, : region_grid.y_size] = region_grid.lat_values
        lon_coord[idx, : region_grid.x_size] = region_grid.lon_values

    for row in df.itertuples(index=False):
        with xr.open_dataset(Path(row.tile_path), engine=netcdf_engine()) as ds_tile:
            if primary_var not in ds_tile:
                raise ValueError(
                    f"Primary variable '{primary_var}' not found in exported tile {row.tile_path}."
                )
            tile = reduce_to_lat_lon_2d(ds_tile[primary_var], primary_var).values
            y_size, x_size = tile.shape
            data[region_index[row.region], date_index[row.date], 0, :y_size, :x_size] = tile

    ds = xr.Dataset(
        data_vars={
            primary_var: (("E", "T", "Z", "Y", "X"), data),
        },
        coords={
            "E": np.arange(len(regions), dtype=np.int32),
            "T": np.arange(len(dates), dtype=np.int32),
            "Z": np.array([0], dtype=np.int32),
            "Y": np.arange(max_y_size, dtype=np.int32),
            "X": np.arange(max_x_size, dtype=np.int32),
            "lat": (("E", "Y"), lat_coord),
            "lon": (("E", "X"), lon_coord),
        },
        attrs={
            "dataset_name": dataset_name,
            "grid_layout": "E,T,Z,Y,X",
            "grid_coord_layout": "lat(E,Y),lon(E,X)",
            "grid_padding": "NaN-padded to the largest regional grid",
            "source": "OceanTACO benchmark export",
            # region/date provenance kept as attrs (E/T index order) rather than
            # object-string coordinates, which trigger a VLenUTF8 write failure when
            # CBP re-writes the decompressed zarr.
            "regions": ",".join(str(region) for region in regions),
            "dates": ",".join(str(date) for date in dates),
        },
    )

    # CF axis attributes so ClimateBenchPress' cf-xarray / canon.canonicalize_dataset
    # identify the E,T,Z,Y,X axes unambiguously (the E realization axis in particular
    # is only recognised through its axis attribute).
    # NOTE: lat/lon are 2-D auxiliary coords and are deliberately left without
    # standard_name/units so they do not compete with the Y/X index axes in
    # cf-xarray (which would make ds[var].cf["Y"] ambiguous in compress.py).
    for axis_name in ("E", "T", "Z", "Y", "X"):
        ds[axis_name].attrs["axis"] = axis_name
    ds["E"].attrs["standard_name"] = "realization"
    return ds


def package_dataset(
    benchmark_root: str | Path,
    dataset_name: str,
    overwrite: bool = False,
) -> Path:
    """Create standardized.zarr, error bounds, and packaged manifest for one dataset."""
    ensure_zarr_available()

    root = Path(benchmark_root)
    if dataset_name not in DATASET_NAME_TO_MODALITY:
        raise ValueError(f"Unknown benchmark dataset name: {dataset_name}")

    modality = DATASET_NAME_TO_MODALITY[dataset_name]
    config = BENCHMARK_MODALITY_CONFIGS[modality]
    manifest_df = _load_manifest(root)
    subset = manifest_df[manifest_df["modality"] == modality].copy()
    if subset.empty:
        raise ValueError(f"No exported tiles found for modality '{modality}'.")

    subset = subset.sort_values(["region", "date"]).reset_index(drop=True)
    dataset = _build_standardized_dataset(subset, config.primary_var, dataset_name)

    dataset_root = root / "datasets" / dataset_name
    store_path = dataset_root / "standardized.zarr"
    error_path = root / "datasets-error-bounds" / dataset_name / "error_bounds.json"
    packaged_manifest_path = dataset_root / "packaged_manifest.csv"

    dataset_root.mkdir(parents=True, exist_ok=True)
    if store_path.exists() and overwrite:
        import shutil

        shutil.rmtree(store_path)
    elif store_path.exists() and not overwrite:
        raise FileExistsError(f"{store_path} already exists. Pass overwrite=True to replace it.")

    # Write each array as a single contiguous chunk so the data variable and the
    # T/E coordinates share consistent chunks (CBP's CodecStack rejects datasets
    # with inconsistent chunks along a shared dimension).
    encoding = {var: {"chunks": dataset[var].shape} for var in dataset.data_vars}
    # Force zarr v2 on-disk format. Packaging runs in the OceanTACO generation env (which
    # may have zarr 3.x, defaulting to v3 stores), but the ClimateBenchPress benchmark env
    # pins zarr~=2.18 and can only read v2. The ``zarr_format`` kwarg exists only on newer
    # xarray/zarr-3; on zarr-2 v2 is already the only format, so we pass it conditionally.
    to_zarr_kwargs = {"mode": "w", "encoding": encoding}
    import inspect

    if "zarr_format" in inspect.signature(dataset.to_zarr).parameters:
        to_zarr_kwargs["zarr_format"] = 2
    dataset.to_zarr(store_path, **to_zarr_kwargs)
    subset.to_csv(packaged_manifest_path, index=False)
    write_error_bounds_file(error_path, config.primary_var, config.strict_abs_error)
    write_manifest(root)
    return store_path


def package_all_datasets(
    benchmark_root: str | Path,
    dataset_names: list[str] | None = None,
    overwrite: bool = False,
) -> list[Path]:
    """Package all selected dataset names that have exported tiles on disk.

    Modalities without any exported tiles (e.g. l4_wind when it was deferred) are
    skipped with a warning rather than raising, so the pipeline runs over whatever
    subset was actually exported.
    """
    names = dataset_names or [cfg.dataset_name for cfg in BENCHMARK_MODALITY_CONFIGS.values()]
    manifest = _load_manifest(Path(benchmark_root))
    available_modalities = set(manifest["modality"].unique())

    outputs: list[Path] = []
    for name in names:
        modality = DATASET_NAME_TO_MODALITY[name]
        if modality not in available_modalities:
            logging.warning(
                "Skipping benchmark dataset '%s': no exported tiles for modality '%s'.",
                name,
                modality,
            )
            continue
        outputs.append(package_dataset(benchmark_root, name, overwrite=overwrite))
    return outputs


def main() -> None:
    """Package one or more exported benchmark datasets."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-root", required=True)
    parser.add_argument("--dataset-name", action="append")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    outputs = package_all_datasets(
        args.benchmark_root,
        dataset_names=args.dataset_name,
        overwrite=args.overwrite,
    )
    for output in outputs:
        print(f"Packaged benchmark dataset: {output}")


if __name__ == "__main__":
    main()
