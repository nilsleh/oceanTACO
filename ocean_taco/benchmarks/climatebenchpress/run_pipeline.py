"""Prepare OceanTACO subsets and run the real ClimateBenchPress benchmark.

Orchestrates the reviewer workflow end to end: download (Copernicus Marine regular-grid
subsets) -> format into regional tiles -> export pre-encoding float32 tiles -> package
``standardized.zarr`` + ``error_bounds.json`` -> (optionally) run the genuine
ClimateBenchPress pipeline in-process with the OceanTACO codec registered.

The ClimateBenchPress run requires conda env ``testpy312`` (Python >= 3.12); the download
and formatting steps use the OceanTACO generation stack. Use ``--prepare-only`` to stop
after packaging, or ``--skip-download`` to format/export/package already-downloaded raw
data (the common path when driving everything from ``testpy312``).
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
from functools import partial
from pathlib import Path

from ocean_taco.benchmarks.climatebenchpress.config import (
    BENCHMARK_MODALITY_CONFIGS,
    DEFAULT_BENCHMARK_MODALITIES,
)
from ocean_taco.benchmarks.climatebenchpress.package_standardized import (
    package_all_datasets,
)
from ocean_taco.benchmarks.climatebenchpress.run_cbp_benchmark import run_cbp_benchmark
from ocean_taco.generate_dataset.download import _parse_variable_overrides
from ocean_taco.generate_dataset.download_sources import (
    BENCHMARK_SUBSET_MODALITIES,
    download_benchmark_subset_data,
)
from ocean_taco.generate_dataset.download_tracker import DownloadTracker
from ocean_taco.generate_dataset.format import (
    create_inventory,
    generate_date_list,
    process_date,
)
from ocean_taco.generate_dataset.format_constants import SPATIAL_REGIONS


def _dataset_names_for_sources(sources: list[str]) -> list[str]:
    """Map benchmark source names to packaged dataset names."""
    return [BENCHMARK_MODALITY_CONFIGS[source].dataset_name for source in sources]


def _bbox_for_regions(region_names: list[str]) -> tuple[float, float, float, float]:
    """Return one enclosing bbox for the requested benchmark regions."""
    unknown = sorted(set(region_names) - set(SPATIAL_REGIONS))
    if unknown:
        raise ValueError(f"Unknown benchmark regions: {unknown}")

    lat_mins = [SPATIAL_REGIONS[name]["lat"][0] for name in region_names]
    lat_maxs = [SPATIAL_REGIONS[name]["lat"][1] for name in region_names]
    lon_mins = [SPATIAL_REGIONS[name]["lon"][0] for name in region_names]
    lon_maxs = [SPATIAL_REGIONS[name]["lon"][1] for name in region_names]
    return (min(lon_mins), max(lon_maxs), min(lat_mins), max(lat_maxs))


def _resolve_download_bbox(
    bbox: tuple[float, float, float, float] | None,
    benchmark_regions: list[str] | None,
) -> tuple[float, float, float, float]:
    """Choose the download bbox from explicit input or requested benchmark regions."""
    if bbox is not None:
        return bbox
    if benchmark_regions:
        return _bbox_for_regions(benchmark_regions)
    raise ValueError("Provide either bbox or benchmark_regions for the benchmark download.")


def run_benchmark_pipeline(
    *,
    benchmark_root: str | Path,
    start_date: str,
    end_date: str,
    bbox: tuple[float, float, float, float] | None = None,
    sources: list[str] | None = None,
    benchmark_regions: list[str] | None = None,
    variable_overrides: dict[str, list[str]] | None = None,
    raw_data_dir: str | Path | None = None,
    formatted_dir: str | Path | None = None,
    inventory_path: str = "file_inventory.parquet",
    processes: int = 1,
    skip_download: bool = False,
    overwrite_subsets: bool = False,
    benchmark_overwrite: bool = False,
    overwrite_packaging: bool = False,
    prepare_only: bool = False,
    climatebenchpress_root: str | Path | None = None,
    include_compressors: list[str] | None = None,
    exclude_compressors: list[str] | None = None,
) -> Path | None:
    """Run download, format, export, package, and (optionally) the real CBP benchmark.

    Returns the path to ``metrics/all_results.csv`` when the CBP run executes, else
    ``None`` (prepare-only).
    """
    chosen_sources = sources or list(DEFAULT_BENCHMARK_MODALITIES)
    unsupported = sorted(set(chosen_sources) - set(BENCHMARK_SUBSET_MODALITIES))
    if unsupported:
        raise ValueError(
            "Benchmark pipeline only supports regular-grid subset downloads for: "
            f"{', '.join(BENCHMARK_SUBSET_MODALITIES)}. Unsupported: {unsupported}"
        )

    benchmark_root = Path(benchmark_root)
    raw_root = Path(raw_data_dir) if raw_data_dir else benchmark_root / "raw"
    formatted_root = Path(formatted_dir) if formatted_dir else benchmark_root / "formatted"
    raw_root.mkdir(parents=True, exist_ok=True)
    formatted_root.mkdir(parents=True, exist_ok=True)

    tracker = DownloadTracker(benchmark_root / "logs")

    if not skip_download:
        resolved_bbox = _resolve_download_bbox(bbox, benchmark_regions)
        download_benchmark_subset_data(
            start_date,
            end_date,
            str(raw_root),
            tracker,
            sources=chosen_sources,
            bbox=resolved_bbox,
            variables_by_source=variable_overrides or {},
            dry_run=False,
            overwrite=overwrite_subsets,
        )
        tracker.save_report()

    date_list = generate_date_list(start_date, end_date)
    process_func = partial(
        process_date,
        data_dir=str(raw_root),
        output_dir=str(formatted_root),
        include_l3_swot=False,
        include_l3_ssh=False,
        include_argo=False,
        only_vars=chosen_sources,
        benchmark_root=str(benchmark_root),
        benchmark_modalities=chosen_sources,
        benchmark_regions=benchmark_regions,
        benchmark_date_min=start_date,
        benchmark_date_max=end_date,
        benchmark_overwrite=benchmark_overwrite,
    )

    if processes > 1:
        with mp.Pool(processes=processes) as pool:
            results = pool.map(process_func, date_list)
    else:
        results = [process_func(date_str) for date_str in date_list]

    all_records = [record for _, records in results for record in records]
    create_inventory(all_records, formatted_root / inventory_path)

    dataset_names = _dataset_names_for_sources(chosen_sources)
    package_all_datasets(benchmark_root, dataset_names, overwrite=overwrite_packaging)

    if prepare_only:
        return None

    if climatebenchpress_root is None:
        raise ValueError(
            "climatebenchpress_root is required to run the benchmark; pass it or use "
            "prepare_only=True to stop after packaging."
        )
    return run_cbp_benchmark(
        benchmark_root=benchmark_root,
        climatebenchpress_root=climatebenchpress_root,
        include_compressors=include_compressors,
        exclude_compressors=exclude_compressors,
    )


def main() -> None:
    """CLI entrypoint for the full reviewer-workflow benchmark pipeline."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", required=True)
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument(
        "--bbox",
        nargs=4,
        type=float,
        metavar=("LON_MIN", "LON_MAX", "LAT_MIN", "LAT_MAX"),
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        choices=sorted(DEFAULT_BENCHMARK_MODALITIES),
        default=list(DEFAULT_BENCHMARK_MODALITIES),
    )
    parser.add_argument(
        "--benchmark-regions",
        nargs="+",
        choices=sorted(SPATIAL_REGIONS),
        help="Benchmark only these OceanTACO regions. If --bbox is omitted, use the enclosing bbox.",
    )
    parser.add_argument(
        "--variables",
        nargs="+",
        metavar="SOURCE=VAR1,VAR2",
        help="Optional subset variable overrides.",
    )
    parser.add_argument("--raw-data-dir")
    parser.add_argument("--formatted-dir")
    parser.add_argument("--inventory-path", default="file_inventory.parquet")
    parser.add_argument("--processes", type=int, default=1)
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--overwrite-subsets", action="store_true")
    parser.add_argument("--benchmark-overwrite", action="store_true")
    parser.add_argument("--overwrite-packaging", action="store_true")
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Stop after packaging standardized.zarr + error bounds; do not run CBP.",
    )
    parser.add_argument(
        "--climatebenchpress-root",
        help="Directory containing the pinned compressor/ and data-loader/ checkouts.",
    )
    parser.add_argument("--include-compressors", nargs="+")
    parser.add_argument("--exclude-compressors", nargs="+", default=[])
    args = parser.parse_args()

    result = run_benchmark_pipeline(
        benchmark_root=args.benchmark_root,
        start_date=args.start_date,
        end_date=args.end_date,
        bbox=tuple(args.bbox) if args.bbox is not None else None,
        sources=args.sources,
        benchmark_regions=args.benchmark_regions,
        variable_overrides=_parse_variable_overrides(args.variables),
        raw_data_dir=args.raw_data_dir,
        formatted_dir=args.formatted_dir,
        inventory_path=args.inventory_path,
        processes=args.processes,
        skip_download=args.skip_download,
        overwrite_subsets=args.overwrite_subsets,
        benchmark_overwrite=args.benchmark_overwrite,
        overwrite_packaging=args.overwrite_packaging,
        prepare_only=args.prepare_only,
        climatebenchpress_root=args.climatebenchpress_root,
        include_compressors=args.include_compressors,
        exclude_compressors=args.exclude_compressors,
    )
    if result is not None:
        print(f"Benchmark complete. Concatenated metrics: {result}")


if __name__ == "__main__":
    main()
