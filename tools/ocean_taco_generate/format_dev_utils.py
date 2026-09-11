"""Development-only helpers for format pipeline inspection."""

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


def filter_and_plot_satellite_data(
    ds: xr.Dataset, satellites_to_keep: list[str], var_name="sla_filtered"
):
    """Filters dataset to keep only specific satellites and plots the result.

    Args:
        ds: xarray Dataset containing 'track_platforms' and 'primary_track'
        satellites_to_keep: List of platform names (e.g. ['Sentinel-3A', 'Jason-3'])
        var_name: The variable to display (default: 'sla_filtered')
    """
    # 1. Identify track indices to keep
    #    track_platforms is a list of strings matching the track order
    platforms = ds["track_platforms"].values
    print("Available platforms in dataset:", platforms)
    keep_indices = [i for i, p in enumerate(platforms) if p in satellites_to_keep]

    # 2. Create the mask
    #    primary_track is the grid of track indices (-1 is empty)
    #    We mask cells where the track index is VALID but NOT in our keep list
    track_grid = ds["primary_track"].values
    mask_to_remove = (track_grid != -1) & (~np.isin(track_grid, keep_indices))

    # 3. Apply mask to data
    data_filtered = ds[var_name].copy()
    data_filtered.values[mask_to_remove] = np.nan

    # 4. Display the result
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    ds[var_name].plot(cmap="viridis")
    plt.title(f"Original Data\n({len(platforms)} tracks)")

    plt.subplot(1, 2, 2)
    data_filtered.plot(cmap="viridis")
    plt.title(f"Filtered Data\nIncluded: {', '.join(satellites_to_keep)}")

    plt.tight_layout()
    plt.savefig("filtered_satellite_data.png")
    plt.show()

    return data_filtered
