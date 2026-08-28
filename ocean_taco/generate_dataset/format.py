#!/usr/bin/env python3
"""Description of file."""

import argparse
import logging
import multiprocessing as mp
import os
from datetime import datetime, timedelta
from functools import partial
from pathlib import Path

import pandas as pd

from ocean_taco.benchmarks.climatebenchpress.config import (
    DEFAULT_BENCHMARK_MODALITIES,
)
from ocean_taco.benchmarks.climatebenchpress.export_subset import (
    BenchmarkTileExporter,
    write_manifest,
)
from ocean_taco.generate_dataset.format_loaders import (
    load_glorys_data,
    load_l3_sst_data,
    load_l4_ssh_data,
    load_l4_sss_data,
    load_l4_sst_data,
    load_l4_wind_data,
)
from ocean_taco.generate_dataset.format_processors import (
    apply_published_metadata_compatibility,
    process_and_split,
    process_argo_data,
    process_glorys_data,
    process_l3_sss_smos_data,
)
from ocean_taco.generate_dataset.new_format_ssh_data import (
    process_l3_ssh_data,
    process_swot_daily_gridded,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def process_date(
    date_str,
    data_dir,
    output_dir,
    include_l3_swot=True,
    include_l3_ssh=True,
    include_argo=True,
    only_vars=None,
    benchmark_root=None,
    benchmark_modalities=None,
    benchmark_regions=None,
    benchmark_date_min=None,
    benchmark_date_max=None,
    benchmark_overwrite=False,
):
    """Process all data sources for a single date."""
    logging.info(f"Processing date: {date_str}")
    all_records = []
    benchmark_exporter = None

    if benchmark_root:
        benchmark_exporter = BenchmarkTileExporter.from_args(
            root=benchmark_root,
            modalities=benchmark_modalities,
            regions=benchmark_regions,
            date_min=benchmark_date_min,
            date_max=benchmark_date_max,
            overwrite=benchmark_overwrite,
        )

    processors = [
        (
            "glorys",
            lambda: process_glorys_data(
                load_glorys_data(data_dir, date_str),
                date_str,
                output_dir,
                benchmark_exporter=benchmark_exporter,
            ),
        ),
        (
            "l4_ssh",
            lambda: process_and_split(
                load_l4_ssh_data(data_dir, date_str),
                date_str,
                output_dir,
                "l4_ssh",
                sensor="L4_SSH",
                benchmark_exporter=benchmark_exporter,
            ),
        ),
        (
            "l4_sst",
            lambda: process_and_split(
                load_l4_sst_data(data_dir, date_str),
                date_str,
                output_dir,
                "l4_sst",
                sensor="L4_SST",
                benchmark_exporter=benchmark_exporter,
            ),
        ),
        (
            "l4_sss",
            lambda: process_and_split(
                load_l4_sss_data(data_dir, date_str),
                date_str,
                output_dir,
                "l4_sss",
                sensor="L4_SSS",
                benchmark_exporter=benchmark_exporter,
            ),
        ),
        (
            "l4_wind",
            lambda: process_and_split(
                load_l4_wind_data(data_dir, date_str),
                date_str,
                output_dir,
                "l4_wind",
                sensor="L4_WIND",
                benchmark_exporter=benchmark_exporter,
            ),
        ),
        (
            "l3_sst",
            lambda: process_and_split(
                load_l3_sst_data(data_dir, date_str),
                date_str,
                output_dir,
                "l3_sst",
                sensor="L3_SST",
                benchmark_exporter=benchmark_exporter,
            ),
        ),
        (
            "l3_sss_smos",
            lambda: process_l3_sss_smos_data(data_dir, date_str, output_dir),
        ),
    ]

    if include_l3_swot:
        processors.append(
            (
                "l3_swot",
                lambda: process_swot_daily_gridded(data_dir, date_str, output_dir),
            )
        )
    if include_l3_ssh:
        processors.append(("l3_ssh", lambda: process_l3_ssh_data(data_dir, date_str, output_dir)))
    if include_argo:
        processors.append(
            ("argo", lambda: process_argo_data(data_dir, date_str, output_dir))
        )

    if only_vars:
        processors = [p for p in processors if p[0] in only_vars]

    for name, processor in processors:
        try:
            count, records = processor()
            # Check for silent failures on critical datasets
            if count == 0 and name in ["glorys", "l4_ssh"]:
                logging.warning(
                    f"  [CRITICAL MISSING] No records created for {name} on {date_str}. Check source files!"
                )
            elif count == 0:
                logging.debug(f"  No records for {name} on {date_str}")
            all_records.extend(records)
        except Exception as e:
            raise RuntimeError(f"Failed while processing {name} for {date_str}") from e

    apply_published_metadata_compatibility(
        all_records, pd.Timestamp(datetime.strptime(date_str, "%Y%m%d"))
    )
    return len(all_records), all_records


def generate_date_list(date_min, date_max):
    """Generate list of date strings."""
    start = datetime.strptime(date_min, "%Y-%m-%d")
    end = datetime.strptime(date_max, "%Y-%m-%d")
    dates = []
    cur = start
    while cur <= end:
        dates.append(cur.strftime("%Y%m%d"))
        cur += timedelta(days=1)
    return dates


def create_inventory(records, output_path):
    """Create file inventory and save to parquet."""
    if not records:
        return None
    df = pd.DataFrame(records)
    df = df.sort_values(["timestamp_file", "data_source"]).reset_index(drop=True)
    df["date_str"] = df["timestamp_file"].dt.strftime("%Y%m%d")
    df.to_parquet(output_path, index=False)
    logging.info(f"✓ Inventory saved: {output_path} ({len(df)} files)")
    return df


def main():
    """Run the date-range formatting pipeline from CLI arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--date-min", default="2024-01-01")
    parser.add_argument("--date-max", default="2024-01-04")
    parser.add_argument("--data-dir", default="./ssh_state_data")
    parser.add_argument("--output-dir", default="./formatted_ssh_data")
    parser.add_argument("--inventory-path", default="file_inventory.parquet")
    parser.add_argument("--processes", "-p", type=int, default=2)
    parser.add_argument(
        "--include-l3-swot", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--include-l3-ssh", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--include-argo", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--only-vars", nargs="+", help="Only process specific variables (e.g. l4_ssh)"
    )
    parser.add_argument(
        "--update-existing-inventory",
        action="store_true",
        help="Update existing inventory file instead of overwriting",
    )
    parser.add_argument(
        "--benchmark-export-root",
        help="Optional root directory for pre-encoding benchmark tile export.",
    )
    parser.add_argument(
        "--benchmark-modalities",
        nargs="+",
        choices=sorted(DEFAULT_BENCHMARK_MODALITIES),
        help="Benchmark-export only these regular-grid modalities.",
    )
    parser.add_argument(
        "--benchmark-regions",
        nargs="+",
        help="Benchmark-export only these region names.",
    )
    parser.add_argument(
        "--benchmark-date-min",
        help="Optional lower benchmark-export date bound (YYYY-MM-DD or YYYYMMDD).",
    )
    parser.add_argument(
        "--benchmark-date-max",
        help="Optional upper benchmark-export date bound (YYYY-MM-DD or YYYYMMDD).",
    )
    parser.add_argument(
        "--benchmark-overwrite",
        action="store_true",
        help="Overwrite existing benchmark-export tiles and metadata.",
    )
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    date_list = generate_date_list(args.date_min, args.date_max)
    logging.info(f"Processing {len(date_list)} dates with {args.processes} workers")

    process_func = partial(
        process_date,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        include_l3_swot=args.include_l3_swot,
        include_l3_ssh=args.include_l3_ssh,
        include_argo=args.include_argo,
        only_vars=args.only_vars,
        benchmark_root=args.benchmark_export_root,
        benchmark_modalities=args.benchmark_modalities,
        benchmark_regions=args.benchmark_regions,
        benchmark_date_min=args.benchmark_date_min,
        benchmark_date_max=args.benchmark_date_max,
        benchmark_overwrite=args.benchmark_overwrite,
    )

    if args.processes > 1:
        with mp.Pool(processes=args.processes) as pool:
            results = pool.map(process_func, date_list)
    else:
        results = [process_func(d) for d in date_list]

    all_records = [rec for _, recs in results for rec in recs]

    if args.update_existing_inventory:
        inventory_path = os.path.join(args.output_dir, args.inventory_path)
        if os.path.exists(inventory_path):
            logging.info(f"Updating existing inventory at {inventory_path}")
            df_existing = pd.read_parquet(inventory_path)

            processed_vars = (
                args.only_vars
                if args.only_vars
                else [
                    "glorys",
                    "l4_ssh",
                    "l4_sst",
                    "l4_sss",
                    "l4_wind",
                    "l3_sst",
                    "l3_sss_smos",
                ]
            )
            if args.only_vars is None:
                if args.include_l3_swot:
                    processed_vars.append("l3_swot")
                if args.include_l3_ssh:
                    processed_vars.append("l3_ssh")
                if args.include_argo:
                    processed_vars.append("argo")

            date_strs = set(date_list)

            mask_vars = df_existing["data_source"].isin(processed_vars)
            mask_dates = df_existing["date_str"].isin(list(date_strs))
            mask_remove = mask_vars & mask_dates

            logging.info(
                f"Removing {mask_remove.sum()} old records matching processed dates/vars."
            )
            df_existing = df_existing[~mask_remove]

            if all_records:
                new_df = pd.DataFrame(all_records)
                combined_df = pd.concat([df_existing, new_df], ignore_index=True)
                all_records = combined_df.to_dict("records")
            else:
                all_records = df_existing.to_dict("records")
        else:
            logging.warning(
                f"Inventory {inventory_path} not found for update, creating new."
            )

    create_inventory(all_records, os.path.join(args.output_dir, args.inventory_path))
    if args.benchmark_export_root:
        manifest_path = write_manifest(args.benchmark_export_root)
        logging.info(f"✓ Benchmark manifest saved: {manifest_path}")
    logging.info("✓ Done!")


if __name__ == "__main__":
    main()


# python ocean_taco/generate_dataset/format.py --date-min 2023-03-29 --date-max 2023-04-15 --data-dir data/new_ssh_dataset --output-dir data/new_ssh_dataset_formatted_region --inventory-path data/new_ssh_dataset_formatted_region/file_collection_swot_period.parquet
