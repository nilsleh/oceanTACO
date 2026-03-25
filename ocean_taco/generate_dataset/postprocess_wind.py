"""Post-process L4 wind data to daily means."""

import argparse
import glob
import os
import re
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

import xarray as xr
from tqdm import tqdm


def process_single_date(date_str, hour_files_dict, out_dir, chunks=None):
    """Process a single date's hourly files to daily mean.

    This function is designed to run in a separate process.

    Args:
        date_str (str): Date string in YYYYMMDD format
        hour_files_dict (dict): Dictionary mapping hour strings to file paths
        out_dir (str): Output directory
        chunks (dict): Dask chunk specification

    Returns:
        tuple: (date_str, success, out_file, error_msg)
    """
    if chunks is None:
        chunks = {"time": 1, "lat": 720, "lon": 1440}

    # Sort by hour and extract file paths
    files = [hour_files_dict[h] for h in sorted(hour_files_dict.keys())]

    if len(files) != 24:
        return (date_str, False, None, f"Expected 24 files, found {len(files)}")

    out_file = os.path.join(out_dir, f"l4_wind_daily_{date_str}.nc")

    if os.path.exists(out_file):
        return (date_str, True, out_file, "already_exists")

    try:
        # Open with Dask chunking
        ds = xr.open_mfdataset(
            files, combine="nested", concat_dim="time", chunks=chunks
        )

        # Resample to daily mean
        u_daily = ds["eastward_wind"].resample(time="1D").mean()
        v_daily = ds["northward_wind"].resample(time="1D").mean()

        # Create output dataset
        ds_out = xr.Dataset(
            {"eastward_wind": u_daily, "northward_wind": v_daily},
            coords={"lat": ds["lat"], "lon": ds["lon"], "time": u_daily["time"]},
            attrs=ds.attrs,
        )

        # Save (triggers Dask computation)
        ds_out.to_netcdf(out_file)

        ds.close()

        return (date_str, True, out_file, None)

    except Exception as e:
        return (date_str, False, None, str(e))


def process_l4_wind_to_daily_from_files(
    hourly_files, out_dir, chunks=None, max_workers=4
):
    """Resample hourly L4 wind files to daily means using parallel processing.

    Args:
        hourly_files (list): List of hourly NetCDF file paths
        out_dir (str): Output directory for daily files
        chunks (dict): Dask chunk specification
        max_workers (int): Number of parallel workers

    Returns:
        dict: Processing statistics
    """
    if chunks is None:
        chunks = {"time": 1, "lat": 720, "lon": 1440}

    # Remove duplicates
    hourly_files = list(dict.fromkeys(hourly_files))

    # Group files by date
    files_by_date = defaultdict(lambda: {})
    for f in hourly_files:
        filename = os.path.basename(f)
        m = re.search(r"PT1H_(\d{8})(\d{2})_", filename)
        if m:
            date_str = m.group(1)
            hour_str = m.group(2)
            files_by_date[date_str][hour_str] = f

    if not files_by_date:
        print("No hourly files found to process")
        return {"processed_dates": 0, "total_hourly_files": 0, "output_files": []}

    print(f"Found {len(files_by_date)} dates with hourly data")
    print(f"Using {max_workers} parallel workers")

    output_files = []
    processed_count = 0
    skipped_count = 0
    failed_count = 0

    # Process dates in parallel
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_date = {}
        for date_str, hour_files_dict in sorted(files_by_date.items()):
            future = executor.submit(
                process_single_date, date_str, hour_files_dict, out_dir, chunks
            )
            future_to_date[future] = date_str

        # Process results as they complete
        with tqdm(total=len(files_by_date), desc="Processing dates") as pbar:
            for future in as_completed(future_to_date):
                date_str = future_to_date[future]
                try:
                    result_date, success, out_file, error_msg = future.result()

                    if success:
                        if error_msg == "already_exists":
                            skipped_count += 1
                            pbar.write(f"  ✓ {date_str}: Already exists")
                        else:
                            output_files.append(out_file)
                            processed_count += 1
                            pbar.write(
                                f"  ✓ {date_str}: Saved {os.path.basename(out_file)}"
                            )
                    else:
                        failed_count += 1
                        pbar.write(f"  ✗ {date_str}: {error_msg}")

                except Exception as e:
                    failed_count += 1
                    pbar.write(f"  ✗ {date_str}: Unexpected error: {e}")

                pbar.update(1)

    print(f"\n{'=' * 60}")
    print(
        f"Processed: {processed_count}, Skipped: {skipped_count}, Failed: {failed_count}"
    )
    print(f"{'=' * 60}")

    return {
        "processed_dates": processed_count,
        "skipped_dates": skipped_count,
        "failed_dates": failed_count,
        "total_hourly_files": sum(
            len(hour_dict) for hour_dict in files_by_date.values()
        ),
        "output_files": output_files,
    }


def process_l4_wind_directory(
    input_dir,
    date_min=None,
    date_max=None,
    remove_hourly=True,
    chunks=None,
    max_workers=4,
):
    """Scan a directory for hourly L4 wind files, resample to daily, and save.

    Args:
        input_dir (str): Directory containing hourly L4 wind NetCDF files
        date_min (str): Optional minimum date 'YYYY-MM-DD' (inclusive)
        date_max (str): Optional maximum date 'YYYY-MM-DD' (inclusive)
        remove_hourly (bool): Whether to delete hourly files after processing
        chunks (dict): Dask chunk specification
        max_workers (int): Number of parallel workers

    Returns:
        dict: Processing statistics
    """
    print(f"Processing L4 wind data in: {input_dir}")
    if date_min:
        print(f"  Date range: {date_min} to {date_max or 'latest'}")

    # Find all NetCDF files
    pattern = os.path.join(
        input_dir,
        "WIND_GLO_PHY_L4_MY_012_006",
        "cmems_obs-wind_glo_phy_my_l4_0.125deg_PT1H_202211",
        "**",
        "**",
        "*.nc",
    )
    all_files = sorted(glob.glob(pattern, recursive=True))

    # Filter for hourly files
    hourly_files = []
    print("Scanning files...")
    for f in tqdm(all_files):
        filename = os.path.basename(f)
        if "daily" in filename.lower():
            continue

        m = re.search(r"PT1H_(\d{8})(\d{2})_", filename)
        if m:
            date_str = m.group(1)
            dt = datetime.strptime(date_str, "%Y%m%d").date()

            if date_min:
                if dt < datetime.strptime(date_min, "%Y-%m-%d").date():
                    continue
            if date_max:
                if dt > datetime.strptime(date_max, "%Y-%m-%d").date():
                    continue

            hourly_files.append(f)

    if not hourly_files:
        print("  No hourly L4 wind files found matching criteria")
        return {"processed_dates": 0, "total_hourly_files": 0, "output_files": []}

    print(f"  Found {len(hourly_files)} hourly files to process")

    # Process files in parallel
    result = process_l4_wind_to_daily_from_files(
        hourly_files,
        input_dir,
        chunks=chunks,
        max_workers=max_workers,
    )

    print("  ✓ Processing complete")
    print(f"    Processed dates: {result['processed_dates']}")
    print(f"    Skipped dates: {result['skipped_dates']}")
    print(f"    Failed dates: {result['failed_dates']}")
    print(f"    Daily files created: {len(result['output_files'])}")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Process hourly L4 wind data to daily means with parallel processing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process with 4 workers (default)
  python post_processs_wind.py --input-dir ./l4_wind

  # Process with 8 workers
  python post_processs_wind.py --input-dir ./l4_wind --workers 8

  # Process specific date range with 6 workers
  python post_processs_wind.py --input-dir ./l4_wind --start-date 2024-01-01 --end-date 2024-01-31 --workers 6

  # Keep original hourly files
  python post_processs_wind.py --input-dir ./l4_wind --no-remove-hourly

  # Custom Dask chunking for memory optimization
  python post_processs_wind.py --input-dir ./l4_wind --time-chunk 24 --lat-chunk 360 --lon-chunk 720 --workers 4
        """,
    )

    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing hourly L4 wind NetCDF files",
    )

    parser.add_argument(
        "--start-date", help="Start date for processing (YYYY-MM-DD, inclusive)"
    )

    parser.add_argument(
        "--end-date", help="End date for processing (YYYY-MM-DD, inclusive)"
    )

    parser.add_argument(
        "--no-remove-hourly",
        action="store_true",
        help="Keep original hourly files after processing",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel workers (default: 4, recommended: 4-8)",
    )

    parser.add_argument(
        "--time-chunk",
        type=int,
        default=1,
        help="Dask chunk size for time dimension (default: 1)",
    )

    parser.add_argument(
        "--lat-chunk",
        type=int,
        default=720,
        help="Dask chunk size for latitude dimension (default: 720)",
    )

    parser.add_argument(
        "--lon-chunk",
        type=int,
        default=1440,
        help="Dask chunk size for longitude dimension (default: 1440)",
    )

    args = parser.parse_args()

    # Validate input directory
    if not os.path.isdir(args.input_dir):
        print(f"Error: Input directory does not exist: {args.input_dir}")
        return 1

    # Validate worker count
    if args.workers < 1 or args.workers > 16:
        print(
            f"Warning: Worker count {args.workers} is outside recommended range (1-16)"
        )

    # Setup Dask chunks
    chunks = {"time": args.time_chunk, "lat": args.lat_chunk, "lon": args.lon_chunk}

    print("=" * 60)
    print("L4 Wind Hourly-to-Daily Processing (Parallel)")
    print("=" * 60)
    print(f"Input directory: {args.input_dir}")
    if args.start_date:
        print(f"Date range: {args.start_date} to {args.end_date or 'latest'}")
    print(f"Remove hourly files: {not args.no_remove_hourly}")
    print(f"Parallel workers: {args.workers}")
    print(f"Dask chunks: {chunks}")
    print("-" * 60)

    start_time = time.time()

    result = process_l4_wind_directory(
        input_dir=args.input_dir,
        date_min=args.start_date,
        date_max=args.end_date,
        remove_hourly=not args.no_remove_hourly,
        chunks=chunks,
        max_workers=args.workers,
    )

    elapsed = time.time() - start_time

    print("=" * 60)
    print("Processing Summary")
    print("=" * 60)
    print(f"Dates processed: {result['processed_dates']}")
    print(f"Dates skipped: {result.get('skipped_dates', 0)}")
    print(f"Dates failed: {result.get('failed_dates', 0)}")
    print(f"Hourly files processed: {result['total_hourly_files']}")
    print(f"Daily files created: {len(result['output_files'])}")
    print(f"Time elapsed: {elapsed:.1f} seconds")
    if result["processed_dates"] > 0:
        print(
            f"Average time per date: {elapsed / result['processed_dates']:.1f} seconds"
        )
    print("=" * 60)

    return 0


if __name__ == "__main__":
    exit(main())
