#!/usr/bin/env python3
"""Build TACO dataset from formatted files."""

import argparse
import gc
import glob
import os
from pathlib import Path

import pandas as pd
import tacoreader
import tacotoolbox
from shapely.geometry import box
from shapely.wkb import dumps as wkb_dumps
from tacotoolbox.sample.extensions.istac import ISTAC
from tqdm import tqdm

tacoreader.use("pandas")

MAX_REGION_SAMPLES = 11

# 8 equal rectangles with descriptive names
SPATIAL_REGIONS = {
    "SOUTH_PACIFIC_WEST": {"lat": (-90, 0), "lon": (-180, -90)},
    "SOUTH_ATLANTIC": {"lat": (-90, 0), "lon": (-90, 0)},
    "SOUTH_INDIAN": {"lat": (-90, 0), "lon": (0, 90)},
    "SOUTH_PACIFIC_EAST": {"lat": (-90, 0), "lon": (90, 180)},
    "NORTH_PACIFIC_WEST": {"lat": (0, 90), "lon": (-180, -90)},
    "NORTH_ATLANTIC": {"lat": (0, 90), "lon": (-90, 0)},
    "NORTH_INDIAN": {"lat": (0, 90), "lon": (0, 90)},
    "NORTH_PACIFIC_EAST": {"lat": (0, 90), "lon": (90, 180)},
}


def generate_sample_id(row: pd.Series) -> str:
    """Generate a unique sample ID based on data source and variable."""
    ds = row["data_source"]

    if ds == "glorys_all_phys":
        return f"glorys_{row['variable']}"
    elif ds == "l3_sss_smos_asc" or ds == "l3_sss_smos_desc":
        fname = row.get("relative_path", "")
        return "l3_sss_asc" if "asc" in fname.lower() else "l3_sss_desc"
    else:
        return ds


def get_modality(data_source: str) -> str:
    """Determine modality from data source."""
    if data_source == "glorys":
        return "model"
    elif data_source == "argo":
        return "in_situ"
    else:
        return "satellite"


def create_sample(row: pd.Series, data_dir: str) -> tacotoolbox.datamodel.Sample:
    """Create a Sample object from an inventory row."""
    file_path = os.path.join(data_dir, row["relative_path"])
    sid = generate_sample_id(row)

    sample = tacotoolbox.datamodel.Sample(id=sid + ".nc", path=file_path, type="FILE")

    meta = {
        "sensor": row.get("sensor", "Unknown"),
        "modality": get_modality(row["data_source"]),
        "data_source": row["data_source"],
        "variable": row.get("variable", "unknown"),
        "res_deg_lat": row["resolution_deg_lat"],
        "res_km_lat": row["resolution_km_lat"],
    }

    sample.extend_with(meta)

    # Add ISTAC spatial-temporal extension
    wkb_geom = row.get("_istac_spatial_wkb")
    if wkb_geom is not None:
        time_start = row.get("_istac_time_start")
        time_end = row.get("_istac_time_end")

        if time_start is not None and time_end is not None:
            # _istac_time_start/end are already in MICROSECONDS from formatting script
            ts_start = int(time_start)
            ts_end = int(time_end)

            istac = ISTAC(
                crs="EPSG:4326", geometry=wkb_geom, time_start=ts_start, time_end=ts_end
            )
            sample.extend_with(istac)
    return sample


def create_region_sample(
    region_name: str, file_samples: list, date
) -> tacotoolbox.datamodel.Sample:
    """Create a region-level sample containing all file samples."""
    region_tortilla = tacotoolbox.datamodel.Tortilla(
        samples=file_samples, strict_schema=False, pad_to=MAX_REGION_SAMPLES
    )

    region_sample = tacotoolbox.datamodel.Sample(
        id=region_name, path=region_tortilla, type="FOLDER"
    )

    # Add region bounding box as ISTAC
    bounds = SPATIAL_REGIONS[region_name]
    lon_min, lon_max = bounds["lon"]
    lat_min, lat_max = bounds["lat"]
    bbox_polygon = box(lon_min, lat_min, lon_max, lat_max)

    # Create proper timestamps in MICROSECONDS
    if isinstance(date, pd.Timestamp):
        date_start = date.tz_localize("UTC") if date.tz is None else date
    else:
        date_start = pd.Timestamp(date, tz="UTC")

    date_end = date_start + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

    # Convert to microseconds
    ts_start_us = int(date_start.timestamp() * 1_000_000)
    ts_end_us = int(date_end.timestamp() * 1_000_000)

    istac = ISTAC(
        crs="EPSG:4326",
        geometry=wkb_dumps(bbox_polygon),
        time_start=ts_start_us,
        time_end=ts_end_us,
    )
    region_sample.extend_with(istac)

    return region_sample


def create_tortilla(
    formatted_data_dir: str,
    df: pd.DataFrame,
    save_path: str,
    include_l3_swot: bool = True,
    include_argo: bool = True,
) -> dict:
    """Create TACO with structure: Date → Region → Files."""
    Path(os.path.dirname(save_path)).mkdir(parents=True, exist_ok=True)

    if not include_l3_swot:
        df = df[df["data_source"] != "l3_swot"].copy()
    if not include_argo:
        df = df[df["data_source"] != "argo"].copy()

    # Ensure timestamp is datetime
    if not pd.api.types.is_datetime64_any_dtype(df["timestamp_file"]):
        df["timestamp_file"] = pd.to_datetime(df["timestamp_file"])

    df["date"] = df["timestamp_file"].dt.date
    unique_dates = sorted(df["date"].unique())

    print(f"\nBuilding TACO with {len(unique_dates)} dates")
    print(f"Data sources: {df['data_source'].unique().tolist()}")
    print(f"Regions: {df['region'].unique().tolist()}")
    print(f"Total files: {len(df)}")

    top_level_samples = []

    for date in tqdm(unique_dates, desc="Building daily tortillas"):
        date_df = df[df["date"] == date]
        regional_samples = []

        # Use fixed list of regions to ensure consistency across dates (PIT requirement)
        for region_name in sorted(SPATIAL_REGIONS.keys()):
            region_df = date_df[date_df["region"] == region_name].drop_duplicates(subset=["relative_path"])

            file_samples = []
            for _, row in region_df.iterrows():
                sample = create_sample(row, formatted_data_dir)
                file_samples.append(sample)

            # Truncate if we exceed MAX_REGION_SAMPLES
            if len(file_samples) > MAX_REGION_SAMPLES:
                print(
                    f"  [Warning] Truncating {len(file_samples)} files to {MAX_REGION_SAMPLES} in {region_name} for date {date}"
                )
                file_samples = file_samples[:MAX_REGION_SAMPLES]

            # Always create region sample, even if empty, to satisfy PIT
            if not file_samples:
                print(f"  [Warning] No files for date {date} in region {region_name}")
            region_sample = create_region_sample(region_name, file_samples, date)
            regional_samples.append(region_sample)

        if regional_samples:
            date_tortilla = tacotoolbox.datamodel.Tortilla(samples=regional_samples)
            date_id = date.strftime("%Y_%m_%d")

            date_sample = tacotoolbox.datamodel.Sample(
                id=date_id, path=date_tortilla, type="FOLDER"
            )

            date_ts = pd.Timestamp(date, tz="UTC")
            date_sample.extend_with({"stac:time_start": date_ts.isoformat()})
            top_level_samples.append(date_sample)

        gc.collect()

    # Create final TACO
    final_tortilla = tacotoolbox.datamodel.Tortilla(samples=top_level_samples)

    taco = tacotoolbox.datamodel.Taco(
        tortilla=final_tortilla,
        id="sea_surface_state_regionalized",
        dataset_version="0.1.0",
        description="Sea Surface State with 8 spatial regions (Date → Region → Files). "
        "SWOT data is gridded to regular resolution. Timestamps in microseconds.",
        licenses=["CC-BY-4.0"],
        keywords=[
            "ocean",
            "sea surface",
            "ssh",
            "sst",
            "sss",
            "swot",
            "glorys",
            "argo",
        ],
        providers=[{"name": "TUM", "roles": ["producer", "licensor"]}],
        tasks=["regression", "generative"],
        extent={
            "spatial": [-180, -90, 180, 90],
            "temporal": [
                df["timestamp_file"].min().date().isoformat(),
                df["timestamp_file"].max().date().isoformat(),
            ],
        },
    )

    if os.path.exists(save_path):
        # if it is a directory, remove DATA and METADATA but preserve .cache
        # (used for HF uploads to allow faster incremental syncs)
        if os.path.isdir(save_path):
            import shutil

            data_dir_path = os.path.join(save_path, "DATA")
            metadata_dir = os.path.join(save_path, "METADATA")
            if os.path.exists(data_dir_path):
                shutil.rmtree(data_dir_path)
            if os.path.exists(metadata_dir):
                shutil.rmtree(metadata_dir)
        else:
            os.remove(save_path)

    print(f"\nWriting TACO to {save_path}...")

    results = tacotoolbox.create(
        taco=taco,
        output=save_path,
        version="2.6",
        output_format="folder",
        compression="zstd",
        split_size="40GB",
        compression_level=22,
        use_dictionary=False,
        write_statistics=False,
        data_page_size=256 * 1024,
        write_batch_size=1024,
        store_schema=False,
        use_content_defined_chunking=True,
        data_page_version="2.0",
    )

    print(f"✓ TACO created: {save_path}")
    return results


def verify_taco(taco_path: str) -> None:
    """Verify the created TACO can be loaded."""
    print("\nVerifying TACO...")

    try:
        dataset = tacoreader.load(taco_path)
    except ValueError:
        parts = sorted(glob.glob(taco_path.replace(".tacozip", "_part*.tacozip")))
        if parts:
            dataset = tacoreader.load(parts)
            from tacotoolbox import create_tacocat

            create_tacocat(tacozips=parts, output_path=os.path.dirname(taco_path))
        else:
            raise

    print("✓ TACO loaded successfully")
    print(f"  Dataset: {dataset}")
    print(f"  First record: {dataset.data.read(0)}")


def print_inventory_summary(df: pd.DataFrame) -> None:
    """Print summary of the inventory DataFrame."""
    print("\n" + "=" * 60)
    print("INVENTORY SUMMARY")
    print("=" * 60)
    print(f"Total files: {len(df)}")
    print(f"Date range: {df['timestamp_file'].min()} to {df['timestamp_file'].max()}")
    print("\nFiles by data source:")
    print(df.groupby("data_source").size().to_string())
    print("\nFiles by region:")
    print(df.groupby("region").size().to_string())
    print("=" * 60)


def analyze_duplicates_upfront(df: pd.DataFrame, report_path: str | None = None) -> pd.DataFrame:
    """Analyze duplicate generated sample IDs across the full inventory upfront.

    Duplicates are detected at the effective tortilla level: date + region + sample_id.
    """
    work_df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(work_df["timestamp_file"]):
        work_df["timestamp_file"] = pd.to_datetime(work_df["timestamp_file"])

    work_df["date"] = work_df["timestamp_file"].dt.date
    work_df["sample_id"] = work_df.apply(lambda row: f"{generate_sample_id(row)}.nc", axis=1)

    dup_mask = work_df.duplicated(subset=["date", "region", "sample_id"], keep=False)
    dup_df = work_df[dup_mask].copy()

    print("\n" + "=" * 60)
    print("UPFRONT DUPLICATE ANALYSIS")
    print("=" * 60)

    if dup_df.empty:
        print("No duplicate generated sample IDs found across date+region.")
        print("=" * 60)
        return dup_df

    collision_counts = (
        dup_df.groupby(["date", "region", "sample_id"]).size().sort_values(ascending=False)
    )

    print(f"Rows involved in duplicates: {len(dup_df)}")
    print(f"Unique date-region-sample_id collisions: {len(collision_counts)}")
    print(f"Duplicate sample IDs: {sorted(dup_df['sample_id'].unique().tolist())}")
    print("\nTop collisions:")
    print(collision_counts.head(20).to_string())

    cols = [
        "date",
        "region",
        "sample_id",
        "data_source",
        "variable",
        "relative_path",
        "timestamp_file",
        "sensor",
    ]
    existing_cols = [c for c in cols if c in dup_df.columns]
    detail_df = dup_df[existing_cols].sort_values(["date", "region", "sample_id", "relative_path"])

    if report_path:
        Path(os.path.dirname(report_path)).mkdir(parents=True, exist_ok=True)
        detail_df.to_parquet(report_path, index=False)
        print(f"Saved upfront duplicate report to: {report_path}")

    print("=" * 60)
    return detail_df


def clean_inventory(df: pd.DataFrame) -> pd.DataFrame:
    """Remove malformed duplicate rows from the inventory.

    For l3_swot, keep only rows that use the canonical uppercase region filename
    pattern and have complete metadata.
    """
    df = df.copy()

    swot_mask = df["data_source"] == "l3_swot"
    swot_df = df[swot_mask].copy()

    if swot_df.empty:
        return df

    filename_series = swot_df["relative_path"].fillna("").astype(str).str.rsplit("/", n=1).str[-1]
    filename_match = filename_series.str.extract(r"^l3_swot_([A-Z_]+)_(\d{8})\.nc$")
    region_from_filename = filename_match[0]

    required_cols = [
        "filename",
        "variable",
        "sensor",
        "_istac_spatial_wkb",
        "_istac_time_start",
        "_istac_time_end",
        "region",
    ]
    has_required_meta = swot_df[required_cols].notna().all(axis=1)
    has_valid_region_name = region_from_filename.isin(SPATIAL_REGIONS.keys())
    region_matches_column = region_from_filename == swot_df["region"].astype(str)

    valid_swot_mask = has_required_meta & has_valid_region_name & region_matches_column
    invalid_swot_count = int((~valid_swot_mask).sum())

    cleaned_df = pd.concat([df[~swot_mask], swot_df[valid_swot_mask]], ignore_index=True)

    # Remove non-canonical l3_ssh rows only when they are part of duplicate
    # date+region+sample_id groups. Canonical file pattern uses uppercase region names.
    if not pd.api.types.is_datetime64_any_dtype(cleaned_df["timestamp_file"]):
        cleaned_df["timestamp_file"] = pd.to_datetime(cleaned_df["timestamp_file"])
    cleaned_df["date"] = cleaned_df["timestamp_file"].dt.date
    cleaned_df["sample_id"] = cleaned_df.apply(lambda row: f"{generate_sample_id(row)}.nc", axis=1)

    l3_ssh_mask = cleaned_df["data_source"] == "l3_ssh"
    l3_ssh_filename = (
        cleaned_df.loc[l3_ssh_mask, "relative_path"]
        .fillna("")
        .astype(str)
        .str.rsplit("/", n=1)
        .str[-1]
    )
    l3_ssh_canonical = l3_ssh_filename.str.match(r"^l3_ssh_[A-Z_]+_\d{8}\.nc$")

    duplicate_group_mask = cleaned_df.duplicated(
        subset=["date", "region", "sample_id"],
        keep=False,
    )

    drop_noncanonical_l3_ssh_mask = pd.Series(False, index=cleaned_df.index)
    drop_noncanonical_l3_ssh_mask.loc[l3_ssh_mask] = (
        duplicate_group_mask.loc[l3_ssh_mask] & (~l3_ssh_canonical)
    )

    removed_noncanonical_l3_ssh = int(drop_noncanonical_l3_ssh_mask.sum())
    if removed_noncanonical_l3_ssh > 0:
        removed_examples = (
            cleaned_df.loc[drop_noncanonical_l3_ssh_mask, "relative_path"].head(5).tolist()
        )
        print(f"Removed non-canonical duplicate l3_ssh rows: {removed_noncanonical_l3_ssh}")
        print(f"  Example removed l3_ssh rows: {removed_examples}")

    cleaned_df = cleaned_df.loc[~drop_noncanonical_l3_ssh_mask].copy()
    cleaned_df = cleaned_df.drop(columns=["date", "sample_id"], errors="ignore")

    before_dedup = len(cleaned_df)
    cleaned_df = cleaned_df.drop_duplicates(subset=["relative_path"], keep="first")
    dedup_removed = before_dedup - len(cleaned_df)

    print(
        "Filtered l3_swot inventory rows: "
        f"removed {invalid_swot_count} malformed rows, "
        f"removed {dedup_removed} duplicate relative_path rows."
    )

    if invalid_swot_count > 0:
        invalid_examples = swot_df.loc[~valid_swot_mask, "relative_path"].head(5).tolist()
        print(f"  Example removed l3_swot rows: {invalid_examples}")

    return cleaned_df


def main():
    """Entry point for building the OceanTACO dataset."""
    parser = argparse.ArgumentParser(
        description="Create spatially regionalized TACO from Sea Surface State data."
    )
    parser.add_argument(
        "--data-dir",
        default="./formatted_ssh_data",
        help="Root directory containing formatted data.",
    )
    parser.add_argument(
        "--output-dir",
        default="./tortilla",
        help="Output directory for the final TACO zip.",
    )
    parser.add_argument(
        "--inventory-path",
        required=True,
        help="Path to the file_inventory.parquet file.",
    )
    parser.add_argument(
        "--include-l3-swot", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--include-argo", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--verify", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--start-date", default=None, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", default=None, help="End date (YYYY-MM-DD)")
    parser.add_argument(
        "--analyze-duplicates-only",
        action="store_true",
        help="Analyze duplicates on the full loaded inventory and exit.",
    )
    parser.add_argument(
        "--duplicate-report-path",
        default=None,
        help="Optional parquet output path for upfront duplicate analysis.",
    )
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print(f"Loading inventory from {args.inventory_path}...")
    df = pd.read_parquet(args.inventory_path)

    if args.analyze_duplicates_only:
        analyze_duplicates_upfront(df, args.duplicate_report_path)
        return

    # Apply date filters if provided
    if args.start_date:
        start_ts = pd.Timestamp(args.start_date)
        print(f"Filtering start date: {start_ts}")
        df = df[df["timestamp_file"] >= start_ts]

    if args.end_date:
        end_ts = pd.Timestamp(args.end_date)
        print(f"Filtering end date: {end_ts}")
        df = df[df["timestamp_file"] <= end_ts]

    print_inventory_summary(df)

    out_zip = os.path.join(args.output_dir, "OceanTACO")

    create_tortilla(
        formatted_data_dir=args.data_dir,
        df=df,
        save_path=out_zip,
        include_l3_swot=args.include_l3_swot,
        include_argo=args.include_argo,
    )

    if args.verify:
        verify_taco(out_zip)

    # # load taco parts _part0001.tacozip etc
    # taco_parts = sorted(glob.glob(os.path.join(args.output_dir, "OceanTACO_part*.tacozip")))
    # if taco_parts:
    #     df = tacoreader.load(taco_parts)
    #
    # from tacotoolbox import create_tacocat
    # create_tacocat(
    #     inputs=taco_parts,
    #     output=args.output_dir,
    #     validate_schema=True,
    # )

    print("\n✓ Done!")


if __name__ == "__main__":
    main()
