#!/usr/bin/env python3
"""Download source data for OceanTACO.

This file is intentionally kept as the compatibility facade and CLI entry point.
Implementation details are split across focused modules to keep core orchestration clear.
"""

import argparse
from pathlib import Path

from ocean_taco.generate_dataset.download_date_filters import (
    _week_ranges,
    create_copernicus_glorys_date_filter,
    create_l3_sst_date_filter,
    create_l4_sst_date_filter,
    create_ssh_date_filter,
    create_sss_date_filter,
    create_sss_smos_date_filter,
    create_wind_date_filter,
    regex_date_filter,
)
from ocean_taco.generate_dataset.download_sources import (
    BENCHMARK_SUBSET_MODALITIES,
    download_argo_data,
    download_benchmark_subset_data,
    download_glorys_data,
    download_l3_ssh_data,
    download_l3_sss_smos_data,
    download_l3_sst_data,
    download_l4_ssh_data,
    download_l4_sss_data,
    download_l4_sst_data,
    download_l4_wind_data,
)
from ocean_taco.generate_dataset.download_swot import (
    SWOT_FTP_ROOTS,
    _download_swot_file,
    build_swot_file_catalog,
    catalog_to_dataframe,
    download_swot_data,
    parallel_swot_download,
)
from ocean_taco.generate_dataset.download_tracker import DownloadTracker


ALL_DOWNLOAD_SOURCES = [
    "glorys",
    "l4_ssh",
    "l3_ssh",
    "l3_sst",
    "l3_sss_smos",
    "l4_sst",
    "l4_sss",
    "l4_wind",
    "argo",
]


def _parse_variable_overrides(items: list[str] | None) -> dict[str, list[str]]:
    """Parse ``source=var1,var2`` entries into a mapping."""
    overrides: dict[str, list[str]] = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(
                f"Invalid variable override '{item}'. Use source=var1,var2."
            )
        source, value = item.split("=", 1)
        values = [token.strip() for token in value.split(",") if token.strip()]
        if not source or not values:
            raise ValueError(
                f"Invalid variable override '{item}'. Use source=var1,var2."
            )
        overrides[source] = values
    return overrides


def main():
    """CLI entry point for running dataset downloads."""
    parser = argparse.ArgumentParser(description="Download SSH State Data")
    parser.add_argument("--start-date", default="2024-01-01")
    parser.add_argument("--end-date", default="2024-01-04")
    parser.add_argument("--output-dir", default="./ssh_state_data")
    parser.add_argument(
        "--log-dir", default=None, help="Directory for logs (default: output-dir/logs)"
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--download", action="store_true")
    parser.add_argument("--weekly-batches", action="store_true")
    parser.add_argument("--aviso-ftp-user", default="")
    parser.add_argument("--aviso-ftp-pass", default="")
    parser.add_argument(
        "--swot-level",
        choices=["l2", "l3"],
        default="l2",
        help="SWOT product level to download",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue downloading other datasets if one fails",
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        choices=ALL_DOWNLOAD_SOURCES,
        help="Restrict downloads to a subset of source families.",
    )
    parser.add_argument(
        "--benchmark-subset",
        action="store_true",
        help="Use copernicusmarine.subset(...) to download a tiny benchmark subset for regular-grid products.",
    )
    parser.add_argument(
        "--bbox",
        nargs=4,
        type=float,
        metavar=("LON_MIN", "LON_MAX", "LAT_MIN", "LAT_MAX"),
        help="Bounding box for benchmark-subset downloads.",
    )
    parser.add_argument(
        "--variables",
        nargs="+",
        metavar="SOURCE=VAR1,VAR2",
        help="Subset variable overrides, for example l4_ssh=sla glorys=zos.",
    )
    parser.add_argument(
        "--overwrite-existing-subsets",
        action="store_true",
        help="Overwrite existing benchmark-subset raw files instead of reusing them.",
    )
    args = parser.parse_args()

    dry_run = args.dry_run or not args.download
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log_dir = Path(args.log_dir) if args.log_dir else output_dir / "logs"
    tracker = DownloadTracker(log_dir)

    tracker.logger.info("=" * 80)
    tracker.logger.info("SSH State Data Download Script")
    tracker.logger.info(f"Date range: {args.start_date} to {args.end_date}")
    tracker.logger.info(f"Output directory: {output_dir}")
    tracker.logger.info(f"Log directory: {log_dir}")
    tracker.logger.info(f"Mode: {'DRY RUN' if dry_run else 'DOWNLOAD'}")
    tracker.logger.info("=" * 80)

    spans = (
        _week_ranges(args.start_date, args.end_date)
        if args.weekly_batches
        else [(args.start_date, args.end_date)]
    )

    selected_sources = args.sources or ALL_DOWNLOAD_SOURCES
    variable_overrides = _parse_variable_overrides(args.variables)

    if args.benchmark_subset:
        if args.bbox is None:
            raise ValueError("--bbox is required when --benchmark-subset is used.")
        unsupported = sorted(set(selected_sources) - set(BENCHMARK_SUBSET_MODALITIES))
        if unsupported:
            raise ValueError(
                "--benchmark-subset only supports regular-grid modalities: "
                f"{', '.join(BENCHMARK_SUBSET_MODALITIES)}. Unsupported: {unsupported}"
            )

    if args.weekly_batches:
        tracker.logger.info(f"Weekly batch mode: {len(spans)} segments")

    download_functions = [
        (
            "GLORYS",
            lambda s, e: download_glorys_data(s, e, str(output_dir), tracker, dry_run),
        ),
        (
            "L4 SSH",
            lambda s, e: download_l4_ssh_data(s, e, str(output_dir), tracker, dry_run),
        ),
        (
            "L3 SSH",
            lambda s, e: download_l3_ssh_data(s, e, str(output_dir), tracker, dry_run),
        ),
        (
            "L3 SST",
            lambda s, e: download_l3_sst_data(s, e, str(output_dir), tracker, dry_run),
        ),
        (
            "L3 SMOS SSS",
            lambda s, e: download_l3_sss_smos_data(s, e, str(output_dir), tracker, dry_run),
        ),
        (
            "L4 SST",
            lambda s, e: download_l4_sst_data(s, e, str(output_dir), tracker, dry_run),
        ),
        (
            "L4 SSS",
            lambda s, e: download_l4_sss_data(s, e, str(output_dir), tracker, dry_run),
        ),
        (
            "L4 Wind",
            lambda s, e: download_l4_wind_data(s, e, str(output_dir), tracker, dry_run),
        ),
        (
            "Argo",
            lambda s, e: download_argo_data(s, e, str(output_dir), tracker, dry_run),
        ),
    ]

    if args.benchmark_subset:
        download_functions = [
            (
                "Benchmark subset",
                lambda s, e: download_benchmark_subset_data(
                    s,
                    e,
                    str(output_dir),
                    tracker,
                    sources=selected_sources,
                    bbox=tuple(args.bbox),
                    variables_by_source=variable_overrides,
                    dry_run=dry_run,
                    overwrite=args.overwrite_existing_subsets,
                ),
            )
        ]
    else:
        filtered_functions = []
        for dataset_name, download_func in download_functions:
            source_token = dataset_name.lower().replace(" ", "_")
            if dataset_name == "GLORYS":
                source_key = "glorys"
            elif dataset_name == "L4 SSH":
                source_key = "l4_ssh"
            elif dataset_name == "L3 SSH":
                source_key = "l3_ssh"
            elif dataset_name == "L3 SST":
                source_key = "l3_sst"
            elif dataset_name == "L3 SMOS SSS":
                source_key = "l3_sss_smos"
            elif dataset_name == "L4 SST":
                source_key = "l4_sst"
            elif dataset_name == "L4 SSS":
                source_key = "l4_sss"
            elif dataset_name == "L4 Wind":
                source_key = "l4_wind"
            elif dataset_name == "Argo":
                source_key = "argo"
            else:
                source_key = source_token
            if source_key in selected_sources:
                filtered_functions.append((dataset_name, download_func))
        download_functions = filtered_functions

    # SWOT requires explicit AVISO FTP credentials.
    if not args.benchmark_subset and args.aviso_ftp_user and args.aviso_ftp_pass:
        swot_label = f"SWOT {args.swot_level.upper()}"
        download_functions.append(
            (
                swot_label,
                lambda s, e: download_swot_data(
                    s,
                    e,
                    str(output_dir),
                    args.aviso_ftp_user,
                    args.aviso_ftp_pass,
                    tracker,
                    swot_level=args.swot_level,
                    force_rebuild_catalog=False,
                    dry_run=dry_run,
                ),
            )
        )
    else:
        tracker.logger.warning(
            "AVISO FTP credentials not provided, skipping SWOT download"
        )

    for i, (start, end) in enumerate(spans, 1):
        if args.weekly_batches:
            tracker.logger.info(f"\nWeek {i}/{len(spans)}: {start} -> {end}")

        for dataset_name, download_func in download_functions:
            try:
                tracker.logger.info(f"\n{'=' * 80}")
                tracker.logger.info(f"Starting: {dataset_name}")
                tracker.logger.info(f"{'=' * 80}")
                download_func(start, end)

            except Exception:
                if args.continue_on_error:
                    tracker.logger.error(
                        f"Failed to download {dataset_name}, continuing..."
                    )
                else:
                    tracker.logger.error(
                        f"Failed to download {dataset_name}, stopping."
                    )
                    tracker.save_report()
                    tracker.print_summary()
                    raise

    report_file = tracker.save_report()
    tracker.print_summary()

    tracker.logger.info("=" * 80)
    tracker.logger.info("Download complete!")
    tracker.logger.info(f"Full report: {report_file}")
    tracker.logger.info("=" * 80)


__all__ = [
    "DownloadTracker",
    "_week_ranges",
    "regex_date_filter",
    "create_l3_sst_date_filter",
    "create_l4_sst_date_filter",
    "create_sss_date_filter",
    "create_copernicus_glorys_date_filter",
    "create_ssh_date_filter",
    "create_wind_date_filter",
    "create_sss_smos_date_filter",
    "SWOT_FTP_ROOTS",
    "build_swot_file_catalog",
    "catalog_to_dataframe",
    "download_swot_data",
    "_download_swot_file",
    "parallel_swot_download",
    "download_glorys_data",
    "download_l4_ssh_data",
    "download_l3_ssh_data",
    "download_l3_sst_data",
    "download_l3_sss_smos_data",
    "download_l4_sst_data",
    "download_l4_sss_data",
    "download_l4_wind_data",
    "download_argo_data",
    "main",
]


if __name__ == "__main__":
    main()
