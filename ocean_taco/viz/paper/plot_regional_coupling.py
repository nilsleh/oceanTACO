"""Plot regional coupling between SSH and other modalities."""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from ocean_taco.dataset.retrieve import (
    HF_DEFAULT_URL,
    load_hf_dataset,
    load_region_product_nc,
)

# Set plotting style for academic publication
sns.set_context("paper", font_scale=1.2)
plt.rcParams["font.family"] = "serif"

VAR_MAP = {
    "l4_ssh": "sla",  # Target X-axis
    "l4_sst": "analysed_sst",  # Option A for Y-axis
    "l4_sss": "sos",  # Option B for Y-axis
}

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


def generate_plot(
    dataset_hf, date_str, y_var="l4_sst", cache_dir=None, output_name="coupling_analysis.png"
):
    """Plot regional coupling between SSH and SST/SSS across all regions."""
    fig, axes = plt.subplots(2, 4, figsize=(20, 10), sharex=True)
    axes = axes.flatten()

    y_var_name = VAR_MAP[y_var]
    date_dash = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"

    print(f"Starting analysis for {y_var} vs L4_SSH...")

    for i, reg in enumerate(REGIONS):
        ax = axes[i]

        # Load Files
        ds_ssh = load_region_product_nc(dataset_hf, date_dash, reg, "l4_ssh", cache_dir)
        ds_y_raw = load_region_product_nc(dataset_hf, date_dash, reg, y_var, cache_dir)

        if ds_ssh is None or ds_y_raw is None:
            ax.text(0.5, 0.5, f"Data Missing:\n{reg}", ha="center")
            continue

        # Load and align
        ds_y = ds_y_raw.interp_like(ds_ssh, method="nearest")

        # Flatten and clean data
        x_data = ds_ssh[VAR_MAP["l4_ssh"]].values.flatten()
        y_data = ds_y[y_var_name].values.flatten()

        # Convert Kelvin to Celsius if SST
        if y_var == "l4_sst":
            y_data = y_data - 273.15
            ylabel = "SST [°C]"
        else:
            ylabel = "Salinity [PSU]"

        mask = ~np.isnan(x_data) & ~np.isnan(y_data)
        x_clean = x_data[mask]
        y_clean = y_data[mask]

        # Calculate Correlation Coefficient
        r = np.corrcoef(x_clean, y_clean)[0, 1]

        # Plot Density using Hexbin (Faster and clearer for large datasets)
        hb = ax.hexbin(
            x_clean,
            y_clean,
            gridsize=50,
            cmap="YlGnBu",
            mincnt=1,
            bins="log",
            edgecolors="none",
        )

        # Add a Linear Regression Line
        m, b = np.polyfit(x_clean, y_clean, 1)
        ax.plot(x_clean, m * x_clean + b, color="red", lw=1.5, label=f"r = {r:.2f}")

        # Formatting
        ax.set_title(reg.replace("_", " "), fontsize=14, fontweight="bold")
        ax.legend(loc="upper left", frameon=True, fontsize=10)

        if i >= 4:
            ax.set_xlabel("SSH / SLA [m]")
        if i % 4 == 0:
            ax.set_ylabel(ylabel)

    # Add a global colorbar
    cb = fig.colorbar(
        hb, ax=axes.ravel().tolist(), orientation="vertical", shrink=0.8, pad=0.02
    )
    cb.set_label("Log10(Pixel Count)")

    plt.suptitle(
        f"Regional Multi-Modal Coupling: SSH vs {y_var.split('_')[1].upper()} ({date_str})",
        fontsize=20,
        y=0.98,
    )

    plt.savefig(output_name, dpi=300, bbox_inches="tight")
    plt.show()


def main():
    """Entry point for regional coupling plot."""
    parser = argparse.ArgumentParser(
        description="Regional coupling between SSH and SST/SSS.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--hf-url", default=HF_DEFAULT_URL, help="HuggingFace dataset URL.")
    parser.add_argument("--cache-dir", default=None, help="Local cache directory.")
    parser.add_argument("--date", default="20230330", help="Date (YYYYMMDD).")
    parser.add_argument("--y-var", default="l4_sst", choices=["l4_sst", "l4_sss"])
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    output = args.output or f"coupling_{args.y_var}_{args.date}.png"
    dataset_hf = load_hf_dataset(args.hf_url)
    generate_plot(
        dataset_hf,
        args.date,
        y_var=args.y_var,
        cache_dir=args.cache_dir,
        output_name=output,
    )


if __name__ == "__main__":
    main()
