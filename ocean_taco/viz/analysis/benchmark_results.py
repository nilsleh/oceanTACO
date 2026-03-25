#!/usr/bin/env python3
"""Visualize benchmark results."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def apply_style(dpi: int = 300) -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "legend.fontsize": 9,
            "figure.dpi": dpi,
            "savefig.dpi": dpi,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.08,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "axes.grid": True,
            "grid.alpha": 0.30,
            "grid.linewidth": 0.5,
            "lines.linewidth": 1.8,
            "lines.markersize": 7,
            "axes.spines.top": True,
            "axes.spines.right": True,
        }
    )


# Tol bright palette (colourblind-safe)
C_BLUE = "#4477AA"
C_CYAN = "#66CCEE"
C_GREEN = "#228833"
C_YELLOW = "#CCBB44"
C_RED = "#EE6677"
C_PURPLE = "#AA3377"
C_GREY = "#BBBBBB"
COLORS = [C_BLUE, C_RED, C_GREEN, C_PURPLE, C_CYAN, C_YELLOW]
COLOR_PRIMARY = COLORS[0]
COLOR_REFERENCE = C_GREY


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def add_p5_p95_band(ax, x, means, p5s, p95s, color, label=None):
    """Plot mean line with [p5, p95] shaded band."""
    ax.fill_between(x, p5s, p95s, alpha=0.18, color=color, linewidth=0)
    ax.plot(
        x,
        means,
        "o-",
        color=color,
        label=label,
        markeredgecolor="white",
        markeredgewidth=0.6,
    )


def palette_cycle(n: int, start: int = 0) -> list[str]:
    """Return n colors from the global palette in consistent order."""
    return [COLORS[(start + i) % len(COLORS)] for i in range(n)]


# ---------------------------------------------------------------------------
# Individual figure functions
# ---------------------------------------------------------------------------
def plot_exp1_spatial(data: dict, out_dir: Path, fmt: str) -> None:
    """Experiment 1: Spatial Extent Scaling."""
    exp = data["experiment_1_spatial"]
    patch_px = [exp[k]["patch_size_px"] for k in exp]
    extent_deg = [exp[k]["extent_deg"] for k in exp]
    mean_ms = [exp[k]["time_mean_s"] * 1000 for k in exp]
    p5_ms = [exp[k]["time_p5_s"] * 1000 for k in exp]
    p95_ms = [exp[k]["time_p95_s"] * 1000 for k in exp]
    tput = [exp[k]["throughput_queries_per_s"] for k in exp]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 3.0))

    # (a) Latency vs patch size
    add_p5_p95_band(ax1, patch_px, mean_ms, p5_ms, p95_ms, COLOR_PRIMARY)
    ax1.set_xlabel("Patch size (pixels per side)")
    ax1.set_ylabel("Retrieval latency (ms)")
    ax1.set_xticks(patch_px)
    ax1.set_xticklabels([f"{p}\u00d7{p}" for p in patch_px])

    ax1_top = ax1.twiny()
    ax1_top.set_xlim(ax1.get_xlim())
    ax1_top.set_xticks(patch_px)
    ax1_top.set_xticklabels(
        [f"{e:.1f}\u00b0" for e in extent_deg], fontsize=8.5, color="#555555"
    )
    ax1_top.tick_params(direction="in", length=3, width=0.5)
    ax1_top.set_xlabel(
        "Spatial extent (degrees)", fontsize=9, color="#555555", labelpad=6
    )

    ax1.annotate(
        "p5\u2013p95 range",
        xy=(patch_px[1], p95_ms[1]),
        xytext=(patch_px[1] + 20, p95_ms[1] + 20),
        fontsize=7.5,
        color="#666666",
        arrowprops=dict(arrowstyle="-", color="#999999", lw=0.6),
    )

    # (b) Throughput
    ax2.bar(
        range(len(patch_px)),
        tput,
        color=palette_cycle(len(patch_px)),
        edgecolor="white",
        linewidth=0.6,
        width=0.55,
    )
    for i, v in enumerate(tput):
        ax2.text(
            i,
            v + 0.3,
            f"{v:.1f}",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="medium",
        )
    ax2.set_xlabel("Patch size")
    ax2.set_ylabel("Throughput (queries s$^{-1}$)")
    ax2.set_xticks(range(len(patch_px)))
    ax2.set_xticklabels([f"{p}\u00d7{p}" for p in patch_px])
    ax2.set_ylim(0, max(tput) * 1.20)

    fig.suptitle(
        "Experiment 1 \u2014 Spatial Extent Scaling",
        fontsize=12,
        fontweight="bold",
        y=0.92,
    )
    # ax1.text(-0.12, 1.05, "(a)", fontsize=11, fontweight="bold",
    #          transform=ax1.transAxes)
    # ax2.text(-0.12, 1.05, "(b)", fontsize=11, fontweight="bold",
    #          transform=ax2.transAxes)
    fig.tight_layout(w_pad=1.0)

    path = out_dir / f"fig_exp1_spatial.{fmt}"
    fig.savefig(path)
    plt.close(fig)
    print(f"  \u2713 {path.name}")


def plot_exp2_temporal(data: dict, out_dir: Path, fmt: str) -> None:
    """Experiment 2: Temporal Depth Scaling."""
    exp = data["experiment_2_temporal"]
    tw_days = [exp[k]["time_window_days"] for k in exp]
    mean_ms = [exp[k]["time_mean_s"] * 1000 for k in exp]
    p5_ms = [exp[k]["time_p5_s"] * 1000 for k in exp]
    p95_ms = [exp[k]["time_p95_s"] * 1000 for k in exp]
    tput = [exp[k]["throughput_queries_per_s"] for k in exp]
    files = [exp[k]["files_mean"] for k in exp]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 3.0))

    # (a) Latency vs temporal window
    add_p5_p95_band(ax1, tw_days, mean_ms, p5_ms, p95_ms, COLOR_PRIMARY)
    ax1.set_xlabel("Temporal window (days)")
    ax1.set_ylabel("Retrieval latency (ms)")
    ax1.set_xticks(tw_days)
    ax1.set_xticklabels([str(t) for t in tw_days])

    for td, m, nf in zip(tw_days, mean_ms, files):
        ax1.annotate(
            f"{int(nf)} files",
            xy=(td, m),
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=7.5,
            color="#555555",
            ha="left",
        )

    ideal = [mean_ms[0] * (f / files[0]) for f in files]
    ax1.plot(
        tw_days,
        ideal,
        "--",
        color=COLOR_REFERENCE,
        linewidth=1.2,
        zorder=1,
        label="Linear reference",
    )
    ax1.legend(loc="upper left", frameon=True, framealpha=0.85, edgecolor="none")

    # (b) Throughput
    ax2.bar(
        range(len(tw_days)),
        tput,
        color=palette_cycle(len(tw_days)),
        edgecolor="white",
        linewidth=0.6,
        width=0.55,
    )
    for i, v in enumerate(tput):
        ax2.text(
            i,
            v + 0.15,
            f"{v:.1f}",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="medium",
        )
    ax2.set_xlabel("Temporal window (days)")
    ax2.set_ylabel("Throughput (queries s$^{-1}$)")
    ax2.set_xticks(range(len(tw_days)))
    ax2.set_xticklabels([str(t) for t in tw_days])
    ax2.set_ylim(0, max(tput) * 1.25)

    fig.suptitle(
        "Experiment 2 \u2014 Temporal Depth Scaling",
        fontsize=12,
        fontweight="bold",
        y=0.92,
    )
    # ax1.text(-0.12, 1.05, "(a)", fontsize=11, fontweight="bold",
    #          transform=ax1.transAxes)
    # ax2.text(-0.12, 1.05, "(b)", fontsize=11, fontweight="bold",
    #          transform=ax2.transAxes)
    fig.tight_layout(w_pad=1.0)

    path = out_dir / f"fig_exp2_temporal.{fmt}"
    fig.savefig(path)
    plt.close(fig)
    print(f"  \u2713 {path.name}")


def plot_exp3_variables(data: dict, out_dir: Path, fmt: str) -> None:
    """Experiment 3: Variable Count Scaling."""
    exp = data["experiment_3_variables"]
    n_vars = [exp[k]["n_variables"] for k in exp]
    mean_ms = [exp[k]["time_mean_s"] * 1000 for k in exp]
    p5_ms = [exp[k]["time_p5_s"] * 1000 for k in exp]
    p95_ms = [exp[k]["time_p95_s"] * 1000 for k in exp]
    tput = [exp[k]["throughput_queries_per_s"] for k in exp]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 3.0))

    # (a) Latency vs variable count
    add_p5_p95_band(ax1, n_vars, mean_ms, p5_ms, p95_ms, COLOR_PRIMARY)
    ideal = [mean_ms[0] * (nv / n_vars[0]) for nv in n_vars]
    ax1.plot(
        n_vars,
        ideal,
        "--",
        color=COLOR_REFERENCE,
        linewidth=1.2,
        zorder=1,
        label="Linear reference",
    )
    ax1.set_xlabel("Number of ocean variables")
    ax1.set_ylabel("Retrieval latency (ms)")
    ax1.set_xticks(n_vars)
    var_desc = {1: "SST", 3: "SST, SSH, L4-SST", 6: "All 6 vars"}
    ax1.set_xticklabels([f"{n}\n({var_desc.get(n, '')})" for n in n_vars], fontsize=8.5)
    ax1.legend(loc="upper left", frameon=True, framealpha=0.85, edgecolor="none")

    # (b) Throughput
    ax2.bar(
        range(len(n_vars)),
        tput,
        color=palette_cycle(len(n_vars)),
        edgecolor="white",
        linewidth=0.6,
        width=0.55,
    )
    for i, v in enumerate(tput):
        ax2.text(
            i,
            v + 1.0,
            f"{v:.1f}",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="medium",
        )
    ax2.set_xlabel("Number of ocean variables")
    ax2.set_ylabel("Throughput (queries s$^{-1}$)")
    ax2.set_xticks(range(len(n_vars)))
    ax2.set_xticklabels([str(n) for n in n_vars])
    ax2.set_ylim(0, max(tput) * 1.20)

    fig.suptitle(
        "Experiment 3 \u2014 Variable Count Scaling",
        fontsize=12,
        fontweight="bold",
        y=0.92,
    )
    # ax1.text(-0.12, 1.05, "(a)", fontsize=11, fontweight="bold",
    #          transform=ax1.transAxes)
    # ax2.text(-0.12, 1.05, "(b)", fontsize=11, fontweight="bold",
    #  transform=ax2.transAxes)
    fig.tight_layout(w_pad=1.0)

    path = out_dir / f"fig_exp3_variables.{fmt}"
    fig.savefig(path)
    plt.close(fig)
    print(f"  \u2713 {path.name}")


def plot_exp4_dataloader(data: dict, out_dir: Path, fmt: str) -> None:
    """Experiment 4: DataLoader Throughput Scaling."""
    exp = data["experiment_4_dataloader"]
    workers = [exp[k]["num_workers"] for k in exp]
    sps = [exp[k]["samples_per_s"] for k in exp]
    wall = [exp[k]["wall_time_s"] for k in exp]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 3.0))

    # (a) Throughput vs workers
    ax1.plot(
        workers,
        sps,
        "o-",
        color=COLOR_PRIMARY,
        markeredgecolor="white",
        markeredgewidth=0.6,
        label="Measured",
    )

    ideal_linear = [sps[0] * (max(w, 1) / 1) if w > 0 else sps[0] for w in workers]
    # ax1.plot(workers, ideal_linear, "--", color=C_GREY, linewidth=1.2,
    #          label="Ideal linear speedup")

    ax1.annotate(
        f"{sps[-1]:.0f} samples s$^{{-1}}$",
        xy=(workers[-1], sps[-1]),
        xytext=(-55, 12),
        textcoords="offset points",
        fontsize=8,
        fontweight="medium",
        color=COLOR_PRIMARY,
        arrowprops=dict(arrowstyle="-", color=COLOR_PRIMARY, lw=0.7),
    )

    ax1.set_xlabel("Number of DataLoader workers")
    ax1.set_ylabel("Throughput (samples s$^{-1}$)")
    ax1.set_xticks(workers)
    ax1.legend(loc="upper left", frameon=True, framealpha=0.85, edgecolor="none")
    ax1.set_ylim(0, max(max(sps), max(ideal_linear)) * 1.0)

    # (b) Wall-clock time
    bars = ax2.bar(
        range(len(workers)),
        wall,
        color=palette_cycle(len(workers)),
        edgecolor="white",
        linewidth=0.6,
        width=0.55,
    )
    for i, v in enumerate(wall):
        ax2.text(
            i,
            v + 0.05,
            f"{v:.1f} s",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="medium",
        )
    ax2.set_xlabel("Number of DataLoader workers")
    ax2.set_ylabel("Wall-clock time (s)")
    ax2.set_xticks(range(len(workers)))
    ax2.set_xticklabels([str(w) for w in workers])
    ax2.set_ylim(0, max(wall) * 1.22)

    for i in range(1, len(workers)):
        speedup = wall[0] / wall[i]
        ax2.text(
            i,
            wall[i] / 2,
            f"{speedup:.1f}\u00d7",
            ha="center",
            va="center",
            fontsize=8,
            color="white",
            fontweight="bold",
        )

    fig.suptitle(
        "Experiment 4 \u2014 DataLoader Throughput Scaling",
        fontsize=12,
        fontweight="bold",
        y=0.92,
    )
    # ax1.text(-0.12, 1.05, "(a)", fontsize=11, fontweight="bold",
    #          transform=ax1.transAxes)
    # ax2.text(-0.12, 1.05, "(b)", fontsize=11, fontweight="bold",
    #          transform=ax2.transAxes)
    fig.tight_layout(w_pad=1.0)

    path = out_dir / f"fig_exp4_dataloader.{fmt}"
    fig.savefig(path)
    plt.close(fig)
    print(f"  \u2713 {path.name}")


# ---------------------------------------------------------------------------
# Dispatcher: maps JSON keys → plot functions
# ---------------------------------------------------------------------------
PLOTTERS = {
    "experiment_1_spatial": plot_exp1_spatial,
    "experiment_2_temporal": plot_exp2_temporal,
    "experiment_3_variables": plot_exp3_variables,
    "experiment_4_dataloader": plot_exp4_dataloader,
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot publication-quality figures from OceanTACO benchmark results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--results_json",
        type=str,
        help="Path to benchmark_results.json produced by benchmark_retrieval.py.",
    )
    parser.add_argument(
        "--format",
        "-f",
        type=str,
        default="pdf",
        choices=["pdf", "png", "svg", "eps"],
        help="Output figure format.",
    )
    parser.add_argument(
        "--output_dir",
        "-o",
        type=str,
        default=None,
        help="Directory to save figures. Default: same directory as the JSON file.",
    )
    parser.add_argument(
        "--dpi", type=int, default=300, help="Resolution for raster formats (png)."
    )
    parser.add_argument(
        "--experiments",
        type=int,
        nargs="+",
        default=None,
        help="Which experiments to plot (1 2 3 4). Default: all found in JSON.",
    )
    args = parser.parse_args()

    # --- Resolve paths -------------------------------------------------------
    json_path = Path(args.results_json).resolve()
    if not json_path.is_file():
        print(f"ERROR: File not found: {json_path}", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.output_dir).resolve() if args.output_dir else json_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Load data -----------------------------------------------------------
    with open(json_path) as f:
        data = json.load(f)

    # --- Apply style ---------------------------------------------------------
    apply_style(dpi=args.dpi)

    # --- Determine which experiments to plot ---------------------------------
    if args.experiments is not None:
        exp_keys = [f"experiment_{i}_" for i in args.experiments]
        keys_to_plot = [
            k for k in PLOTTERS if any(k.startswith(prefix) for prefix in exp_keys)
        ]
    else:
        keys_to_plot = [k for k in PLOTTERS if k in data]

    if not keys_to_plot:
        print("WARNING: No matching experiments found in JSON.", file=sys.stderr)
        sys.exit(1)

    # --- Plot ----------------------------------------------------------------
    print(f"Plotting {len(keys_to_plot)} figure(s) → {out_dir}/")
    for key in keys_to_plot:
        PLOTTERS[key](data, out_dir, args.format)

    print(f"\nDone. {len(keys_to_plot)} figure(s) saved to {out_dir}/")


if __name__ == "__main__":
    main()
