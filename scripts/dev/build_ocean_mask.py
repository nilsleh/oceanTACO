"""Build the frozen Core GHRSST ocean-mask artifact for a release revision."""

from __future__ import annotations

import argparse
from pathlib import Path

import xarray as xr

from ocean_taco.catalog import CORE_DATASET_REVISION, CatalogConfig
from ocean_taco.retrieve import _REGION_BOUNDS, load_hf_dataset, load_tile_nc
from ocean_taco.sampling import build_ocean_mask

MASK_DATE = "2024-06-01"
DEFAULT_OUTPUT = Path("ocean_taco/sampling/data/ocean_mask_0p1deg_60S_60N.npz")


def build(*, output: Path, cache_dir: Path | None = None) -> Path:
    """Read eight pinned GHRSST masks and atomically write one global artifact."""
    config = CatalogConfig(cache_dir=cache_dir)
    catalog = load_hf_dataset(config)
    datasets = []
    asset_ids = []
    try:
        for tile in sorted(_REGION_BOUNDS):
            dataset = load_tile_nc(catalog, MASK_DATE, tile, "l4_sst", config=config)
            if dataset is None:
                raise RuntimeError(f"Pinned Core revision has no {tile}/l4_sst.nc for {MASK_DATE}.")
            datasets.append(dataset[["mask"]])
            asset_ids.append(f"DATA/{MASK_DATE.replace('-', '_')}/{tile}/l4_sst.nc")
        merged = xr.combine_by_coords(datasets, combine_attrs="override")
        artifact = build_ocean_mask(
            merged,
            source_asset_ids=asset_ids,
            source_revision=CORE_DATASET_REVISION,
            date=MASK_DATE,
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        artifact.write(output)
        return output
    finally:
        for dataset in datasets:
            dataset.close()
        close = getattr(catalog, "close", None)
        if close is not None:
            close()


def main() -> None:
    """Build the release artifact from explicit command-line paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cache-dir", type=Path)
    arguments = parser.parse_args()
    output = build(output=arguments.output, cache_dir=arguments.cache_dir)
    print(output)


if __name__ == "__main__":
    main()
