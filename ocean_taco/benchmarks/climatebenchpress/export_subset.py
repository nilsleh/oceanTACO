"""Benchmark export helpers for OceanTACO pre-encoding regional tiles."""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import xarray as xr

from ocean_taco.benchmarks.climatebenchpress.config import (
    BENCHMARK_MODALITY_CONFIGS,
    DEFAULT_BENCHMARK_MODALITIES,
)
from ocean_taco.benchmarks.climatebenchpress.utils import (
    collect_sidecar_paths,
    coord_hash,
    dataframe_from_records,
    dump_json,
    load_json,
    normalize_export_dataset,
)


def _parse_date(date_str: str | None) -> datetime | None:
    if not date_str:
        return None
    if len(date_str) == 8:
        return datetime.strptime(date_str, "%Y%m%d")
    return datetime.strptime(date_str, "%Y-%m-%d")


@dataclass
class BenchmarkTileExporter:
    """Export pre-encoding regional tiles for ClimateBenchPress-style comparison."""

    root: Path
    modalities: set[str]
    regions: set[str] | None = None
    date_min: datetime | None = None
    date_max: datetime | None = None
    overwrite: bool = False

    def should_export(self, modality: str, region_name: str, date_str: str) -> bool:
        """Return True when one tile should be exported."""
        if modality not in self.modalities:
            return False
        if modality not in BENCHMARK_MODALITY_CONFIGS:
            return False
        if self.regions is not None and region_name not in self.regions:
            return False

        current_date = _parse_date(date_str)
        if self.date_min and current_date and current_date < self.date_min:
            return False
        if self.date_max and current_date and current_date > self.date_max:
            return False
        return True

    def export_tile(
        self,
        *,
        dataset: xr.Dataset,
        modality: str,
        region_name: str,
        date_str: str,
        formatted_file: str | Path,
    ) -> Path | None:
        """Write one float32 benchmark tile plus JSON sidecar metadata."""
        if not self.should_export(modality, region_name, date_str):
            return None

        ds_out = normalize_export_dataset(dataset)
        if not ds_out.data_vars:
            logging.info(
                "Skipping benchmark export for %s/%s/%s because the dataset has no numeric vars.",
                modality,
                region_name,
                date_str,
            )
            return None

        tile_path = self.root / "tiles" / modality / region_name / f"{date_str}.nc"
        sidecar_path = tile_path.with_suffix(".json")

        if tile_path.exists() and sidecar_path.exists() and not self.overwrite:
            return tile_path

        tile_path.parent.mkdir(parents=True, exist_ok=True)
        ds_out.to_netcdf(tile_path, engine="h5netcdf")

        metadata = {
            "modality": modality,
            "dataset_name": BENCHMARK_MODALITY_CONFIGS[modality].dataset_name,
            "primary_var": BENCHMARK_MODALITY_CONFIGS[modality].primary_var,
            "error_family": BENCHMARK_MODALITY_CONFIGS[modality].error_family,
            "strict_abs_error": BENCHMARK_MODALITY_CONFIGS[modality].strict_abs_error,
            "region": region_name,
            "date": date_str,
            "tile_path": str(tile_path.resolve()),
            "formatted_file": str(Path(formatted_file).resolve()),
            "variable_names": sorted(ds_out.data_vars),
            "shape_yx": [int(ds_out.sizes["lat"]), int(ds_out.sizes["lon"])],
            "lat_size": int(ds_out.sizes["lat"]),
            "lon_size": int(ds_out.sizes["lon"]),
            "lat_hash": coord_hash(ds_out["lat"].values),
            "lon_hash": coord_hash(ds_out["lon"].values),
            "dtype": "float32",
        }
        dump_json(sidecar_path, metadata)
        return tile_path

    @classmethod
    def from_args(
        cls,
        root: str | Path,
        modalities: list[str] | None = None,
        regions: list[str] | None = None,
        date_min: str | None = None,
        date_max: str | None = None,
        overwrite: bool = False,
    ) -> BenchmarkTileExporter:
        """Build an exporter from CLI-style values."""
        selected_modalities = set(modalities or DEFAULT_BENCHMARK_MODALITIES)
        return cls(
            root=Path(root),
            modalities=selected_modalities,
            regions=set(regions) if regions else None,
            date_min=_parse_date(date_min),
            date_max=_parse_date(date_max),
            overwrite=overwrite,
        )


def collect_export_records(root: str | Path) -> list[dict]:
    """Load all per-tile sidecar records from one benchmark root."""
    root_path = Path(root)
    records = []
    for sidecar_path in collect_sidecar_paths(root_path):
        records.append(load_json(sidecar_path))
    return records


def write_manifest(root: str | Path, output_path: str | Path | None = None) -> Path:
    """Scan sidecars and write one CSV manifest."""
    root_path = Path(root)
    manifest_path = Path(output_path) if output_path else root_path / "manifest.csv"
    df = dataframe_from_records(collect_export_records(root_path))
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(manifest_path, index=False)
    return manifest_path


def main() -> None:
    """Rebuild the benchmark manifest from exported sidecars."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-root", required=True)
    parser.add_argument("--manifest-path")
    args = parser.parse_args()

    manifest_path = write_manifest(args.benchmark_root, args.manifest_path)
    print(f"Wrote benchmark manifest: {manifest_path}")


if __name__ == "__main__":
    main()
