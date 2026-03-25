"""Check inventory coverage across spatial regions and time."""

import argparse

import pandas as pd

# 8 equal rectangles
SPATIAL_REGIONS = [
    "SOUTH_PACIFIC_WEST",
    "SOUTH_ATLANTIC",
    "SOUTH_INDIAN",
    "SOUTH_PACIFIC_EAST",
    "NORTH_PACIFIC_WEST",
    "NORTH_ATLANTIC",
    "NORTH_INDIAN",
    "NORTH_PACIFIC_EAST",
]


def check_coverage(inventory_path):
    print(f"Loading inventory: {inventory_path}")
    df = pd.read_parquet(inventory_path)

    # Ensure timestamp is datetime
    if not pd.api.types.is_datetime64_any_dtype(df["timestamp_file"]):
        df["timestamp_file"] = pd.to_datetime(df["timestamp_file"])

    df["date"] = df["timestamp_file"].dt.date
    # Filter out dates beyond 2025-10-01 as per the main script logic
    df = df[df["timestamp_file"] < pd.Timestamp("2025-10-01")]

    unique_dates = sorted(df["date"].unique())

    print(
        f"Checking {len(unique_dates)} dates from {unique_dates[0]} to {unique_dates[-1]}"
    )

    # Expected data sources
    expected_sources = [
        "glorys",
        "l4_ssh",
        "l4_sst",
        "l4_sss",
        "l4_wind",
        "l3_sst",
        "l3_sss_smos_asc",
        "l3_sss_smos_desc",
        "l3_ssh",
        "l3_swot",
        "argo",
    ]

    # Track missing data per source
    missing_data = {source: [] for source in expected_sources}

    for date in unique_dates:
        date_df = df[df["date"] == date]

        # Check global gridded sources (should be present in ALL regions)
        global_sources = ["glorys", "l4_ssh", "l4_sst", "l4_sss", "l4_wind"]

        for source in global_sources:
            # Check if source exists AT ALL for this date
            if not ((date_df["data_source"] == source).any()):
                missing_data[source].append(f"Date: {date} | ENTIRE DATE MISSING")
                continue

            # Check per region
            for region in SPATIAL_REGIONS:
                region_df = date_df[date_df["region"] == region]
                if not ((region_df["data_source"] == source).any()):
                    missing_data[source].append(f"Date: {date} | Region: {region}")

        # Check sparse/swath sources (might not be in every region, but check if date has ANY files)
        sparse_sources = [
            "l3_sst",
            "l3_sss_smos_asc",
            "l3_sss_smos_desc",
            "l3_ssh",
            "l3_swot",
            "argo",
        ]

        for source in sparse_sources:
            if not ((date_df["data_source"] == source).any()):
                missing_data[source].append(f"Date: {date} | NO FILES FOUND GLOBALLY")

    for source in expected_sources:
        print("\n" + "=" * 50)
        print(f"MISSING {source.upper()} COVERAGE")
        print("=" * 50)

        errors = missing_data[source]
        if errors:
            # Cap output if too many errors
            if len(errors) > 20:
                print(f"Total missing entries: {len(errors)}. Showing first 20:")
                for err in errors[:20]:
                    print(err)
                print(f"... and {len(errors) - 20} more.")
            else:
                for err in errors:
                    print(err)
        else:
            print("None! Full coverage (or at least consistent presence).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("inventory_path")
    args = parser.parse_args()
    check_coverage(args.inventory_path)
