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
    download_argo_data,
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

    # SWOT requires explicit AVISO FTP credentials.
    if args.aviso_ftp_user and args.aviso_ftp_pass:
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
