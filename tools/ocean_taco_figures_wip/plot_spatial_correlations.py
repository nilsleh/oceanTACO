"""Plot spatial correlations between SSH and SST."""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from ocean_taco.dataset.retrieve import (
    HF_DEFAULT_URL,
    load_hf_dataset,
    load_region_product_nc,
)

# Set plotting style
sns.set_context("paper", font_scale=1.2)
plt.rcParams["font.family"] = "serif"

VAR_MAP = {"l4_ssh": "sla", "l4_sst": "analysed_sst"}
REGIONS = [
    "NORTH_PACIFIC_WEST",
    "NORTH_ATLANTIC",
    "NORTH_INDIAN",
    "NORTH_PACIFIC_EAST",
    "SOUTH_PACIFIC_WEST",
    "SOUTH_ATLANTIC",
    "SOUTH_INDIAN",
    "SOUTH_PACIFIC_EAST",
]


def generate_aggregated_plot(
    dataset_hf, date_list, cache_dir=None, output_name="seasonal_coupling.png"
):
    """Plot aggregated SSH-SST correlation across regions and multiple dates."""
    fig, axes = plt.subplots(2, 4, figsize=(20, 10), sharex=True, sharey=True)
    axes = axes.flatten()

    print(f"Aggregating data from {len(date_list)} dates...")

    for i, reg in enumerate(REGIONS):
        ax = axes[i]
        all_x, all_y = [], []

        for date_str in date_list:
            date_dash = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
            ds_ssh = load_region_product_nc(dataset_hf, date_dash, reg, "l4_ssh", cache_dir)
            ds_sst_raw = load_region_product_nc(dataset_hf, date_dash, reg, "l4_sst", cache_dir)

            if ds_ssh is not None and ds_sst_raw is not None:
                try:
                    ds_sst = ds_sst_raw.interp_like(ds_ssh, method="nearest")

                    # Subsample 1% of pixels to keep memory manageable while maintaining distribution
                    x_val = ds_ssh.sla.values.flatten()
                    y_val = ds_sst.analysed_sst.values.flatten() - 273.15

                    mask = ~np.isnan(x_val) & ~np.isnan(y_val)
                    # Random subsampling
                    indices = np.random.choice(
                        np.where(mask)[0], size=int(mask.sum() * 0.01), replace=False
                    )

                    all_x.append(x_val[indices])
                    all_y.append(y_val[indices])
                except Exception as e:
                    print(f"Skipping {reg} {date_str}: {e}")

        if not all_x:
            ax.text(0.5, 0.5, "No Data", ha="center")
            continue

        # Concatenate all dates
        x_final = np.concatenate(all_x)
        y_final = np.concatenate(all_y)

        # Calculate Aggregated Correlation
        r = np.corrcoef(x_final, y_final)[0, 1]

        # Plot Hexbin
        hb = ax.hexbin(
            x_final,
            y_final,
            gridsize=50,
            cmap="YlGnBu",
            mincnt=1,
            bins="log",
            linewidths=0,
        )

        # Add Regression
        m, b = np.polyfit(x_final, y_final, 1)
        ax.plot(
            x_final, m * x_final + b, color="red", lw=1.5, label=f"Global r={r:.2f}"
        )

        ax.set_title(reg.replace("_", " "), fontsize=12, fontweight="bold")
        ax.legend(loc="upper left", fontsize=9)
        ax.grid(True, alpha=0.3)

        # Set limits to standard ocean ranges (optional)
        ax.set_xlim(-0.5, 0.5)  # SSH range
        ax.set_ylim(0, 32)  # SST range

    # Global Labels
    fig.text(0.5, 0.04, "Sea Level Anomaly (SLA) [m]", ha="center", fontsize=14)
    fig.text(
        0.08,
        0.5,
        "Sea Surface Temperature (SST) [°C]",
        va="center",
        rotation="vertical",
        fontsize=14,
    )

    cb = fig.colorbar(
        hb, ax=axes.ravel().tolist(), orientation="vertical", shrink=0.8, pad=0.02
    )
    cb.set_label("Pixel Count (Log Scale)")

    plt.savefig(output_name, dpi=300, bbox_inches="tight")
    print(f"Saved aggregated plot to {output_name}")


def main():
    """Entry point for spatial correlations plot."""
    parser = argparse.ArgumentParser(
        description="Spatial SSH-SST correlations aggregated over multiple dates.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--hf-url", default=HF_DEFAULT_URL, help="HuggingFace dataset URL.")
    parser.add_argument("--cache-dir", default=None, help="Local cache directory.")
    parser.add_argument(
        "--dates",
        nargs="+",
        default=[
            "20230115", "20230215", "20230315", "20230415",
            "20230515", "20230615", "20230715", "20230815",
            "20230915", "20231015", "20231115", "20231215",
        ],
        help="List of dates (YYYYMMDD) to aggregate.",
    )
    parser.add_argument("--output", default="seasonal_coupling.png")
    args = parser.parse_args()

    dataset_hf = load_hf_dataset(args.hf_url)
    generate_aggregated_plot(
        dataset_hf,
        args.dates,
        cache_dir=args.cache_dir,
        output_name=args.output,
    )


if __name__ == "__main__":
    main()
