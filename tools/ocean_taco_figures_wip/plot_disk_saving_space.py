#!/usr/bin/env python3
"""Plot disk space savings for OceanTACO dataset."""

import argparse
import logging
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timedelta
from functools import partial
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


# =============================================================================
# Constants
# =============================================================================

REGIONS = [
    "SOUTH_PACIFIC_WEST",
    "SOUTH_ATLANTIC",
    "SOUTH_INDIAN",
    "SOUTH_PACIFIC_EAST",
    "NORTH_PACIFIC_WEST",
    "NORTH_ATLANTIC",
    "NORTH_INDIAN",
    "NORTH_PACIFIC_EAST",
]

ALL_MODALITIES = [
    "glorys",
    "l4_ssh",
    "l4_sst",
    "l4_sss",
    "l4_wind",
    "l3_sst",
    "l3_swot",
    "l3_ssh",
    "l3_sss_smos_asc",
    "l3_sss_smos_desc",
    "argo",
]

MODALITY_PRIMARY_VAR = {
    "glorys": "zos",
    "l4_ssh": "sla",
    "l4_sst": "analysed_sst",
    "l4_sss": "sos",
    "l4_wind": "eastward_wind",
    "l3_sst": "adjusted_sea_surface_temperature",
    "l3_swot": "ssha_filtered",
    "l3_ssh": "sla_filtered",
    "l3_sss_smos_asc": "Sea_Surface_Salinity",
    "l3_sss_smos_desc": "Sea_Surface_Salinity",
    "argo": "TEMP",
}

MODALITY_DISPLAY = {
    "glorys": "GLORYS",
    "l4_ssh": "L4 SSH",
    "l4_sst": "L4 SST",
    "l4_sss": "L4 SSS",
    "l4_wind": "L4 Wind",
    "l3_sst": "L3 SST",
    "l3_swot": "L3 SWOT",
    "l3_ssh": "L3 SSH",
    "l3_sss_smos_asc": "L3 SSS\nSMOS Asc",
    "l3_sss_smos_desc": "L3 SSS\nSMOS Desc",
    "argo": "Argo",
}

# Modalities where raw source files are individual satellite tracks, making a
# direct size comparison unreliable. These fall back to a float32 estimate.
FLOAT32_ESTIMATE_MODALITIES = {"l3_swot", "l3_ssh"}

# Maps modality name → subdirectory name under --raw-dir.
# Most modalities match their name; SMOS asc/desc share one directory.
RAW_MOD_DIR = {
    "glorys": "glorys",
    "l4_ssh": "l4_ssh",
    "l4_sst": "l4_sst",
    "l4_sss": "l4_sss",
    "l4_wind": "l4_wind",
    "l3_sst": "l3_sst",
    "l3_sss_smos_asc": "l3_sss_smos",
    "l3_sss_smos_desc": "l3_sss_smos",
    "argo": "argo",
}


# =============================================================================
# Size computation
# =============================================================================


def compressed_bytes_for_processed_file(fpath: str, primary_var: str) -> tuple[int, int]:
    """Return (float32_estimate_bytes, compressed_bytes) for one variable in one processed file.

    The float32 estimate is used as the uncompressed baseline only for modalities
    in FLOAT32_ESTIMATE_MODALITIES (SWOT, L3 SSH) where raw source files are not
    directly comparable (individual satellite tracks).

    Args:
        fpath: Path to the processed NetCDF4/HDF5 file.
        primary_var: Name of the variable to measure.

    Returns:
        Tuple of (float32_estimate_bytes, compressed_bytes).
    """
    with h5py.File(fpath, "r") as f:
        if primary_var not in f:
            # raise KeyError(f"Variable '{primary_var}' not found in {fpath}")
            print(f"Warning: Variable '{primary_var}' not found in {fpath}. Skipping.")
            print("date: ", fpath.split("_")[-1].split(".")[0])
            return 0, 0
        ds = f[primary_var]
        float32_estimate = int(np.prod(ds.shape)) * 4
        compressed = ds.id.get_storage_size()
        return float32_estimate, compressed


def raw_uncompressed_bytes(raw_dir: str, mod: str, date_str: str, primary_var: str) -> int:
    """Return actual uncompressed bytes of primary_var across raw source files for one date.

    Reads dtype and shape from the raw file via h5py (no data loading).
    Uses dtype.itemsize rather than assuming float32, so float64 raw files
    are measured correctly.

    For SMOS ascending/descending, files are disambiguated by "asc"/"desc" in
    the filename (case-insensitive). Falls back to 0 if no files are found.

    Args:
        raw_dir: Root directory of raw downloaded files.
        mod: Modality key (used to look up RAW_MOD_DIR).
        date_str: Date string in YYYYMMDD format.
        primary_var: Variable name to measure inside the raw file.

    Returns:
        Total uncompressed bytes across all matching raw files.
    """
    subdir = RAW_MOD_DIR.get(mod)
    if subdir is None:
        raise ValueError(f"No RAW_MOD_DIR entry for modality '{mod}'")

    mod_dir = Path(raw_dir) / subdir
    if not mod_dir.is_dir():
        raise FileNotFoundError(f"Raw modality directory not found: {mod_dir}")

    candidates = list(mod_dir.glob(f"**/*{date_str}*.nc"))

    # For SMOS, keep only files matching the pass direction.
    # The asc/desc distinction lives in the directory name (smos-asc / smos-des),
    # not the individual filenames, so check the full path.
    if mod == "l3_sss_smos_asc":
        candidates = [f for f in candidates if "smos-asc" in str(f).lower()]
    elif mod == "l3_sss_smos_desc":
        candidates = [f for f in candidates if "smos-des" in str(f).lower()]

    total = 0

    if not candidates:
        print(f"Warning: No raw files found for modality '{mod}' on date {date_str} in {mod_dir}")
    for fpath in candidates:
        with h5py.File(fpath, "r") as f:
            if primary_var not in f:
                raise KeyError(f"Variable '{primary_var}' not found in {fpath}")
            ds = f[primary_var]
            total += ds.dtype.itemsize * int(np.prod(ds.shape))

    return total


def compute_sizes_for_date(
    output_dir: str, date_str: str, raw_dir: str | None = None
) -> tuple[dict[str, int], dict[str, int]]:
    """Compute uncompressed and compressed sizes per modality for one date.

    Compressed size: h5py get_storage_size() on processed region files.
    Uncompressed size:
      - Raw source files (dtype.itemsize * size) when raw_dir is given and the
        modality is not in FLOAT32_ESTIMATE_MODALITIES.
      - Float32 estimate (n_elements * 4) for SWOT/L3_SSH and when raw_dir
        is not provided.
    """
    uncomp: dict[str, int] = {}
    comp: dict[str, int] = {}

    for mod in ALL_MODALITIES:
        primary_var = MODALITY_PRIMARY_VAR[mod]
        total_float32 = 0
        total_comp = 0

        for region in REGIONS:
            fpath = os.path.join(output_dir, mod, f"{mod}_{region}_{date_str}.nc")
            if os.path.exists(fpath):
                f32, c = compressed_bytes_for_processed_file(fpath, primary_var)
                total_float32 += f32
                total_comp += c

        if raw_dir and mod not in FLOAT32_ESTIMATE_MODALITIES:
            uncomp[mod] = raw_uncompressed_bytes(raw_dir, mod, date_str, primary_var)
        else:
            uncomp[mod] = total_float32

        comp[mod] = total_comp

    return uncomp, comp


def compute_all_sizes(
    output_dir: str,
    date_list: list[str],
    raw_dir: str | None = None,
    max_workers: int | None = None,
) -> tuple[dict[str, int], dict[str, int]]:
    """Aggregate sizes across all dates using parallel workers."""
    uncomp_totals: dict[str, int] = defaultdict(int)
    comp_totals: dict[str, int] = defaultdict(int)

    fn = partial(compute_sizes_for_date, output_dir, raw_dir=raw_dir)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = executor.map(fn, date_list)
        for uncomp, comp in tqdm(results, total=len(date_list), desc="Scanning dates"):
            for mod in ALL_MODALITIES:
                uncomp_totals[mod] += uncomp.get(mod, 0)
                comp_totals[mod] += comp.get(mod, 0)

    return dict(uncomp_totals), dict(comp_totals)


# =============================================================================
# Helpers
# =============================================================================


def generate_date_list(date_min: str, date_max: str) -> list[str]:
    start = datetime.strptime(date_min, "%Y-%m-%d")
    end = datetime.strptime(date_max, "%Y-%m-%d")
    dates = []
    cur = start
    while cur <= end:
        dates.append(cur.strftime("%Y%m%d"))
        cur += timedelta(days=1)
    return dates


def _format_size(nbytes: float) -> str:
    if nbytes >= 1e9:
        return f"{nbytes / 1e9:.2f} GB"
    elif nbytes >= 1e6:
        return f"{nbytes / 1e6:.1f} MB"
    elif nbytes >= 1e3:
        return f"{nbytes / 1e3:.0f} KB"
    return f"{nbytes:.0f} B"


# =============================================================================
# Plotting
# =============================================================================


def plot_disk_savings(
    uncomp_sizes: dict[str, int],
    comp_sizes: dict[str, int],
    n_days: int,
    output: str | None = None,
    figsize: tuple = (7, 4),
    dpi: int = 300,
) -> None:
    """Bar chart showing the compression ratio (float32 uncompressed / int16+zlib on disk)."""
    sns.set_context("paper", font_scale=1.2)
    plt.rcParams["font.family"] = "serif"

    modalities = [
        m
        for m in ALL_MODALITIES
        if uncomp_sizes.get(m, 0) > 0 and comp_sizes.get(m, 0) > 0
    ]
    if not modalities:
        raise ValueError("No data found for any modality. Aborting plot.")

    labels = [MODALITY_DISPLAY.get(m, m) for m in modalities]
    ratios = np.array([uncomp_sizes[m] / comp_sizes[m] for m in modalities])

    total_uncomp = sum(uncomp_sizes.get(m, 0) for m in modalities)
    total_comp = sum(comp_sizes.get(m, 0) for m in modalities)
    total_ratio = total_uncomp / total_comp if total_comp > 0 else 0

    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(len(modalities))
    colors = plt.cm.viridis(np.linspace(0.25, 0.85, len(modalities)))

    bars = ax.bar(x, ratios, width=0.6, color=colors, edgecolor="white", linewidth=0.5)

    for bar, r in zip(bars, ratios):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.15,
            f"{r:.1f}×",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            color="#333",
        )

    ax.set_ylabel("Compression ratio")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9, rotation=45, ha="right", rotation_mode="anchor")

    ax.yaxis.grid(True, alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)
    ax.set_ylim(0, max(ratios) * 1.25)

    fig.tight_layout()

    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=dpi, bbox_inches="tight")
        logging.info(f"Saved plot to {output}")
        plt.close(fig)
    else:
        plt.show()


# =============================================================================
# LaTeX table
# =============================================================================


def save_latex_table(
    uncomp_sizes: dict[str, int],
    comp_sizes: dict[str, int],
    n_days: int,
    path: str,
) -> None:
    modalities = [
        m
        for m in ALL_MODALITIES
        if uncomp_sizes.get(m, 0) > 0 or comp_sizes.get(m, 0) > 0
    ]

    lines = [
        r"\begin{table}[t]",
        r"\caption{Storage compression ratios per data source. Compressed sizes reflect the",
        r"on-disk \texttt{int16}+\texttt{zlib} storage of the processed region files.}",
        r"\label{tab:disk_savings}",
        r"\begin{tabular}{llrrr}",
        r"\tophline",
        r"Modality & Primary variable & Uncompressed (MB\,day$^{-1}$) & Compressed (MB\,day$^{-1}$) & Ratio \\",
        r"\middlehline",
    ]

    total_uncomp = 0
    total_comp = 0

    for m in modalities:
        u = uncomp_sizes.get(m, 0)
        c = comp_sizes.get(m, 0)
        total_uncomp += u
        total_comp += c

        u_mb = u / n_days / 1e6
        c_mb = c / n_days / 1e6
        ratio = u / c if c > 0 else float("inf")
        display = MODALITY_DISPLAY.get(m, m).replace("\n", " ")
        if m in FLOAT32_ESTIMATE_MODALITIES:
            display += r"$^{a}$"
        var = MODALITY_PRIMARY_VAR[m].replace("_", r"\_")
        lines.append(
            rf"{display} & \texttt{{{var}}} & {u_mb:.1f} & {c_mb:.1f} & {ratio:.1f}$\times$ \\"
        )

    lines.append(r"\middlehline")

    total_u_mb = total_uncomp / n_days / 1e6
    total_c_mb = total_comp / n_days / 1e6
    total_ratio = total_uncomp / total_comp if total_comp > 0 else float("inf")
    lines.append(
        rf"\textbf{{TOTAL}} & & \textbf{{{total_u_mb:.1f}}} & \textbf{{{total_c_mb:.1f}}} & \textbf{{{total_ratio:.1f}$\times$}} \\"
    )

    lines += [
        r"\bottomhline",
        r"\end{tabular}",
        r"\belowtable{$^{a}$ Uncompressed size estimated as \texttt{float32} ($4\,\text{bytes per element}$)"
        r" because raw L3 SWOT and L3 SSH source files contain overlapping swath segments that are"
        r" not directly comparable to the gridded region files. All other modalities use the actual"
        r" raw-file size at native floating-point precision (\texttt{dtype.itemsize} $\times$ number of elements).}",
        r"\end{table}",
    ]

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n")
    logging.info(f"Saved LaTeX table to {path}")

# =============================================================================
# CLI
# =============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure int16+zlib compression savings on processed ocean data."
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Root directory of processed/formatted output data.",
    )
    parser.add_argument(
        "--raw-dir",
        default=None,
        help=(
            "Root directory of raw downloaded source files. When provided, uncompressed "
            "sizes are read from actual raw files (dtype.itemsize × shape) instead of "
            "a float32 estimate. SWOT and L3 SSH always use the float32 estimate."
        ),
    )
    parser.add_argument("--date-min", required=True, help="Start date (YYYY-MM-DD).")
    parser.add_argument("--date-max", required=True, help="End date (YYYY-MM-DD).")
    parser.add_argument(
        "--output",
        default="disk_savings.png",
        help="Path to save the bar chart (default: disk_savings.png).",
    )
    parser.add_argument(
        "--latex-table",
        default=None,
        help="Path to save a booktabs LaTeX table (.tex). Skipped if not provided.",
    )
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 2) - 1),
        help="Number of parallel worker processes (default: cpu_count - 1).",
    )
    args = parser.parse_args()

    date_list = generate_date_list(args.date_min, args.date_max)
    logging.info(
        f"Scanning {len(date_list)} dates from {args.date_min} to {args.date_max} "
        f"with {args.workers} workers"
    )

    if args.raw_dir:
        logging.info(f"Using raw source files from {args.raw_dir} for uncompressed baseline.")
    else:
        logging.info("No --raw-dir given; using float32 estimate for uncompressed baseline.")

    uncomp_sizes, comp_sizes = compute_all_sizes(
        args.output_dir, date_list, raw_dir=args.raw_dir, max_workers=args.workers
    )

    plot_disk_savings(
        uncomp_sizes,
        comp_sizes,
        n_days=len(date_list),
        output=args.output,
        dpi=args.dpi,
    )

    if args.latex_table:
        save_latex_table(uncomp_sizes, comp_sizes, n_days=len(date_list), path=args.latex_table)


if __name__ == "__main__":
    main()
