"""Description of file."""

import os

import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr


def _bytes_human(n: int) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if n < 1024:
            return f"{n:.2f} {unit}"
        n /= 1024
    return f"{n:.2f} PB"


def _shape_str(shape: tuple[int, ...]) -> str:
    return "(" + ",".join(str(int(s)) for s in shape) + ")"


def inspect_one_date_shapes(
    inventory_path: str,
    base_dir: str | None = None,
    date_str: str | None = None,
    per_file_vars: int = 12,
) -> None:
    """From the inventory, pick a single date and print dataset/variable shapes, dtypes,
    chunking and estimated uncompressed sizes for one representative file per
    (data_source, variable).

    Args:
        inventory_path: Path to Parquet inventory
        base_dir: Base directory that relative_path is relative to. Defaults to inventory's parent
        date_str: Optional YYYYMMDD to fix the date. If None, picks the first (sorted) date in inventory
        per_file_vars: Max variables to describe per file
    """
    try:
        df = pd.read_parquet(inventory_path)
    except Exception as e:
        print(f"Failed to read inventory: {e}")
        return

    if df.empty:
        print("Inventory is empty.")
        return

    # Determine base directory for relative paths
    if base_dir is None:
        base_dir = os.path.dirname(os.path.abspath(inventory_path))

    # Keep NetCDF outputs only
    df = df[df["relative_path"].str.endswith(".nc", na=False)].copy()
    if df.empty:
        print("No NetCDF files in the inventory.")
        return

    # Choose date
    if date_str is None:
        # Pick earliest date deterministically
        date_str = df["date_str"].sort_values().iloc[0]
    print(f"Inspecting shapes for date: {date_str}")

    day = df[df["date_str"] == date_str].copy()
    if day.empty:
        print(f"No records for date {date_str}")
        return

    # One representative file per (data_source, variable)
    reps = (
        day.sort_values(["data_source", "variable", "filename"])
        .groupby(["data_source", "variable"], as_index=False)
        .head(1)
    )

    for _, row in reps.iterrows():
        source = row.get("data_source")
        var = row.get("variable")
        rel = row.get("relative_path")
        fpath = rel if os.path.isabs(rel) else os.path.join(base_dir, rel)

        print(f"\n=== {source} / {var} ===")
        if not os.path.exists(fpath):
            print(f"  ! Missing file: {fpath}")
            continue

        try:
            fsize = os.path.getsize(fpath)
        except Exception:
            fsize = float("nan")

        print(f"  File: {fpath} ({_bytes_human(fsize)})")

        try:
            ds = xr.open_dataset(fpath)
        except Exception as e:
            print(f"  ! Failed to open: {e}")
            continue

        try:
            # Header
            print(f"  {'Var':<24} {'Shape':<18} {'Est. size':>12}")
            print(f"  {'-' * 24} {'-' * 18} {'-' * 12}")
            # print min and max lon and lat to see extent
            try:
                print(
                    f"  {'(lon min,max)':<24} {'':<18} {ds['lon'].min().item():.2f}, {ds['lon'].max().item():.2f}"
                )
                print(
                    f"  {'(lat min,max)':<24} {'':<18} {ds['lat'].min().item():.2f}, {ds['lat'].max().item():.2f}"
                )
            except Exception:
                print(
                    f"  {'(lon min,max)':<24} {'':<18} {ds['LONGITUDE'].min().item():.2f}, {ds['LONGITUDE'].max().item():.2f}"
                )
                print(
                    f"  {'(lat min,max)':<24} {'':<18} {ds['LATITUDE'].min().item():.2f}, {ds['LATITUDE'].max().item():.2f}"
                )

            shown = 0
            total_est = 0
            for v in list(ds.data_vars):
                if shown >= per_file_vars:
                    break
                da = ds[v]
                dtype = da.dtype
                # Estimate uncompressed size when numeric
                est_bytes = None
                try:
                    if hasattr(dtype, "itemsize") and dtype.kind in (
                        "f",
                        "i",
                        "u",
                        "b",
                    ):
                        est_bytes = int(da.size) * int(dtype.itemsize)
                        total_est += est_bytes
                except Exception:
                    est_bytes = None

                est_str = _bytes_human(est_bytes) if est_bytes is not None else "n/a"
                shape_disp = _shape_str(tuple(int(s) for s in da.shape))
                print(f"  {v:<24} {shape_disp:<18} {est_str:>12}")
                shown += 1

            if total_est:
                print(
                    f"  {'Total (listed)':<24} {'':<18} {_bytes_human(total_est):>12}"
                )
        finally:
            try:
                ds.close()
            except Exception:
                pass


def count_swot_l3_files_per_date(json_path):
    """Count the number of SWOT L3 files per date and per cycle from the scraped catalogue.

    Args:
        json_path: Path to swot_l3_file_catalog.json

    Returns:
        df_per_date: DataFrame with columns ['date', 'n_files']
        df_per_cycle: DataFrame with columns ['cycle', 'n_files']
    """
    import json
    from collections import defaultdict

    import pandas as pd

    with open(json_path) as f:
        catalogue = json.load(f)

    # Count files per date
    date_to_count = defaultdict(int)
    cycle_to_count = defaultdict(int)

    for cycle_name, cycle_files in catalogue.get("cycles", {}).items():
        cycle_to_count[cycle_name] = len(cycle_files)

        for entry in cycle_files:
            date_str = entry.get("date")

            if date_str and len(date_str) == 8:
                # Format as YYYY-MM-DD
                formatted_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
                date_to_count[formatted_date] += 1

    # Convert to DataFrames
    df_per_date = pd.DataFrame(
        [
            {"date": date, "n_files": count}
            for date, count in sorted(date_to_count.items())
        ]
    )
    df_per_date["date"] = pd.to_datetime(df_per_date["date"])

    df_per_cycle = pd.DataFrame(
        [
            {"cycle": cycle, "n_files": count}
            for cycle, count in sorted(cycle_to_count.items())
        ]
    )

    # Print summaries
    print("\n=== SWOT L3 File Counts ===")
    print("\nFiles per date:")
    print(
        f"  Date range: {df_per_date['date'].min().date()} to {df_per_date['date'].max().date()}"
    )
    print(f"  Total dates: {len(df_per_date)}")
    print(f"  Total files: {df_per_date['n_files'].sum()}")

    print("\nFiles per cycle:")
    print(f"  Total cycles: {len(df_per_cycle)}")
    print(f"  Total files: {df_per_cycle['n_files'].sum()}")

    return df_per_date, df_per_cycle


def find_missing_l3_swot_in_catalogue(json_path, missing_dates_txt):
    import json

    import pandas as pd

    with open(json_path) as f:
        catalogue = json.load(f)

    # Build a mapping: date (YYYY-MM-DD) -> list of (filename, cycle)
    date_to_files = {}
    for cycle_name, cycle_files in catalogue.get("cycles", {}).items():
        for entry in cycle_files:
            date_str = entry.get("date")
            if date_str and len(date_str) == 8:
                formatted = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
                date_to_files.setdefault(formatted, []).append(
                    {"filename": entry.get("filename"), "cycle": cycle_name}
                )

    missing_df = pd.read_csv(missing_dates_txt)
    missing_l3_swot = missing_df[missing_df["data_source"] == "l3_swot"]
    missing_dates = set(missing_l3_swot["missing_date"].astype(str))

    found_in_catalogue = sorted(missing_dates & set(date_to_files.keys()))
    still_missing = sorted(missing_dates - set(date_to_files.keys()))

    print(f"Total missing L3 SWOT dates in inventory: {len(missing_dates)}")
    print(f"Dates found in catalogue: {len(found_in_catalogue)}")
    print(f"Dates still missing: {len(still_missing)}")

    print("\nDates found in catalogue but missing in inventory:")
    for d in found_in_catalogue:
        for fileinfo in date_to_files[d]:
            print(
                f"  {d} | cycle: {fileinfo['cycle']} | filename: {fileinfo['filename']}"
            )

    print("\nDates truly missing (not in catalogue):")
    for d in still_missing:
        print(f"  {d}")

    return found_in_catalogue, still_missing


def check_gaps_and_plot(df, missing_dates_txt="missing_dates_per_modality.txt"):
    """Check for gaps in data coverage and save missing dates per modality.

    Returns:
        results_df: DataFrame with coverage statistics per (data_source, variable)
        missing_dates_dict: Dictionary mapping (data_source, variable) to list of missing dates
    """
    # Ensure timestamp is datetime
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp_file"])

    # Store missing dates for inspection
    missing_dates_records = []
    missing_dates_dict = {}

    # Group by data_source and variable
    results = []
    for (ds, var), group in df.groupby(["data_source", "variable"]):
        times = group["timestamp"].dt.normalize().sort_values()
        min_time = times.min()
        max_time = times.max()
        # Expected daily range
        expected = pd.date_range(min_time, max_time, freq="D")
        # Count files per date
        files_per_date = times.value_counts().reindex(expected, fill_value=0)
        n_missing = (files_per_date == 0).sum()
        has_gap = n_missing > 0

        # Save missing dates for this modality
        missing_dates = expected[files_per_date == 0]
        if len(missing_dates) > 0:
            missing_dates_dict[(ds, var)] = missing_dates.tolist()
            for d in missing_dates:
                missing_dates_records.append(f"{ds},{var},{d.date()}")

        results.append(
            {
                "data_source": ds,
                "variable": var,
                "min": min_time,
                "max": max_time,
                "n_expected": len(expected),
                "n_dates_with_file": (files_per_date > 0).sum(),
                "has_gap": has_gap,
                "n_missing_dates": n_missing,
                "max_files_per_date": files_per_date.max(),
                "min_files_per_date": files_per_date[files_per_date > 0].min()
                if (files_per_date > 0).any()
                else 0,
            }
        )

    results_df = pd.DataFrame(results)
    print("\n=== Coverage Summary ===")
    print(results_df.to_string())

    # Write missing dates to text file
    if missing_dates_records:
        with open(missing_dates_txt, "w") as f:
            f.write("data_source,variable,missing_date\n")
            for line in missing_dates_records:
                f.write(line + "\n")
        print(f"\n✓ Missing dates written to {missing_dates_txt}")
        print(f"  Total missing dates: {len(missing_dates_records)}")
    else:
        print("\n✓ No missing dates found - complete coverage!")

    return results_df, missing_dates_dict


def create_timeline_plot(df, missing_dates_dict=None):
    """Create timeline plot showing data coverage with gaps/holes.

    Args:
        df: DataFrame with coverage statistics (from check_gaps_and_plot)
        missing_dates_dict: Dictionary mapping (data_source, variable) to missing dates

    Returns:
        fig: Matplotlib figure
    """
    import matplotlib.dates as mdates

    df = df.copy()
    df["min"] = pd.to_datetime(df["min"])
    df["max"] = pd.to_datetime(df["max"])

    # IMPROVEMENT 1: Merge GLORYS variables into single row
    # Group GLORYS entries together - they have same availability
    df_glorys = df[df["data_source"] == "glorys"]
    df_other = df[df["data_source"] != "glorys"]

    if not df_glorys.empty:
        # Take the first GLORYS entry and merge all variables
        glorys_merged = df_glorys.iloc[0:1].copy()
        glorys_merged.loc[glorys_merged.index[0], "label"] = "glorys (all variables)"
        glorys_merged.loc[glorys_merged.index[0], "variable"] = "all"

        # Merge missing dates from all GLORYS variables
        glorys_missing = []
        for _, row in df_glorys.iterrows():
            key = (row["data_source"], row["variable"])
            if missing_dates_dict and key in missing_dates_dict:
                glorys_missing.extend(missing_dates_dict[key])

        # Update missing dates dict with merged entry
        if missing_dates_dict is not None and glorys_missing:
            missing_dates_dict[("glorys", "all")] = list(set(glorys_missing))
            glorys_merged.loc[glorys_merged.index[0], "has_gap"] = True
        else:
            glorys_merged.loc[glorys_merged.index[0], "has_gap"] = False

        # Combine back
        df = pd.concat([df_other, glorys_merged], ignore_index=True)

    # Create a label that combines data_source and variable (for non-glorys entries)
    mask_no_label = (
        ~df["data_source"].eq("glorys") | df.get("label", pd.Series()).isna()
    )
    df.loc[mask_no_label, "label"] = (
        df.loc[mask_no_label, "data_source"] + " - " + df.loc[mask_no_label, "variable"]
    )

    # IMPROVEMENT 2: Sort by data source type (L4 -> L3 -> L2 -> ARGO -> GLORYS)
    def sort_key(row):
        ds = row["data_source"]
        # Define custom sort order
        if ds.startswith("l4"):
            return (0, ds, str(row["variable"]))  # L4 first
        elif ds.startswith("l3"):
            return (1, ds, str(row["variable"]))  # L3 second
        elif ds.startswith("l2"):
            return (2, ds, str(row["variable"]))  # L2 third
        elif ds == "argo":
            return (3, ds, str(row["variable"]))  # ARGO fourth
        elif ds == "glorys":
            return (4, ds, str(row["variable"]))  # GLORYS last
        else:
            return (5, ds, str(row["variable"]))  # Others at the end

    df["_sort_key"] = df.apply(sort_key, axis=1)
    df = df.sort_values("_sort_key").drop(columns=["_sort_key"])

    # IMPROVEMENT 3: Get global min/max dates for x-axis limits
    global_min = df["min"].min()
    global_max = df["max"].max()

    # Create figure
    fig, ax = plt.subplots(figsize=(14, max(8, len(df) * 0.4)))

    # Plot each data source timeline
    for i, row in enumerate(df.itertuples()):
        y_pos = i

        # Determine if this modality has gaps
        key = (row.data_source, row.variable)
        missing_dates = missing_dates_dict.get(key, []) if missing_dates_dict else []
        has_gaps = len(missing_dates) > 0

        if not has_gaps:
            # Solid line for complete coverage
            ax.plot(
                [row.min, row.max],
                [y_pos, y_pos],
                color="tab:blue",
                linewidth=8,
                solid_capstyle="round",
                alpha=0.8,
            )
        else:
            # Create segments for available dates
            # Build list of all dates in range
            all_dates = pd.date_range(row.min, row.max, freq="D")
            missing_set = set(pd.to_datetime(missing_dates).normalize())
            available_dates = sorted(set(all_dates) - missing_set)

            if not available_dates:
                # No data at all - show as empty/dotted line
                ax.plot(
                    [row.min, row.max],
                    [y_pos, y_pos],
                    color="lightgray",
                    linewidth=2,
                    linestyle=":",
                    alpha=0.5,
                )
            else:
                # Find continuous segments of available data
                segments = []
                current_segment = [available_dates[0]]

                for j in range(1, len(available_dates)):
                    if (available_dates[j] - available_dates[j - 1]).days == 1:
                        current_segment.append(available_dates[j])
                    else:
                        # Gap found - save current segment and start new one
                        if len(current_segment) > 0:
                            segments.append((current_segment[0], current_segment[-1]))
                        current_segment = [available_dates[j]]

                # Add last segment
                if len(current_segment) > 0:
                    segments.append((current_segment[0], current_segment[-1]))

                # Plot each segment
                for start, end in segments:
                    ax.plot(
                        [start, end],
                        [y_pos, y_pos],
                        color="tab:orange",
                        linewidth=8,
                        solid_capstyle="round",
                        alpha=0.8,
                    )

                # Mark gaps with red markers
                for missing_date in missing_dates:
                    ax.plot(
                        missing_date,
                        y_pos,
                        marker="x",
                        color="red",
                        markersize=6,
                        alpha=0.6,
                    )

    # Label the y-axis with dataset names
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(
        df["label"].tolist(), fontsize=15
    )  # Convert to list to avoid NaN issues
    ax.invert_yaxis()  # so top = first item

    # IMPROVEMENT 3: Set x-axis limits to global with two day buffer
    ax.set_xlim(global_min - pd.Timedelta(days=5), global_max + pd.Timedelta(days=5))

    # Format x-axis
    ax.set_xlabel("Date", fontsize=11)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right", fontsize=14)

    ax.grid(axis="x", linestyle="--", alpha=0.3)

    # Add statistics text
    total_missing = (
        sum(len(dates) for dates in missing_dates_dict.values())
        if missing_dates_dict
        else 0
    )
    total_modalities_with_gaps = sum(1 for row in df.itertuples() if row.has_gap)

    stats_text = f"Total modalities: {len(df)}\n"
    stats_text += f"With gaps: {total_modalities_with_gaps}\n"
    stats_text += f"Total missing dates: {total_missing}"

    plt.tight_layout()

    return fig


if __name__ == "__main__":
    # Paths
    catalogue_path = "/p/scratch/hai_uqmethodbox/data/new_ssh_dataset_formatted/file_collection_swot_period.parquet"
    json_path = (
        "/p/scratch/hai_uqmethodbox/data/new_ssh_dataset/swot_l3_file_catalog.json"
    )
    text_output_path = "nils/oceanTACO/missing_dates_per_modality.txt"

    # Load and analyze
    print("Loading inventory...")
    df_inventory = pd.read_parquet(catalogue_path)

    print("\nChecking for gaps...")
    df_coverage, missing_dates_dict = check_gaps_and_plot(
        df_inventory, missing_dates_txt=text_output_path
    )

    # Create timeline plot with gaps visualized
    print("\nCreating timeline plot...")
    fig = create_timeline_plot(df_coverage, missing_dates_dict)
    fig.savefig("timeline_plot.pdf", dpi=150, bbox_inches="tight")
    print("✓ Saved timeline_plot.pdf")

    # Check L3 SWOT against catalogue
    print("\nChecking L3 SWOT against catalogue...")
    found, missing = find_missing_l3_swot_in_catalogue(json_path, text_output_path)

    # swot counts
    df_dates, df_cycles = count_swot_l3_files_per_date(json_path)

    print("\n=== Analysis Complete ===")
    print(f"Coverage summary saved to: {text_output_path}")
    print("Timeline plot saved to: timeline_plot.pdf")
