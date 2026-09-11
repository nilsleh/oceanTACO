"""Visualization and Benchmarking for OceanTACO Query Generation."""

from __future__ import annotations

import os
import time
from collections import defaultdict
from pathlib import Path

import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle

from ocean_taco.dataset.dataset import OceanTACODataset, Query, collate_ocean_samples
from ocean_taco.dataset.queries import (
    PatchSize,
    QueryGenerator,
    TrainingQueryConfig,
    generate_training_queries,
)


def _configure_cartopy_dir(path: str):
    """Configure cartopy data directory."""
    p = Path(path).expanduser()
    p.mkdir(parents=True, exist_ok=True)
    os.environ["CARTOPY_USER_DIR"] = str(p)
    cartopy.config["data_dir"] = str(p)


# =============================================================================
# Query Visualization
# =============================================================================


def visualize_queries(
    queries: list[Query],
    title: str = "Query Distribution",
    figsize: tuple[int, int] = (16, 10),
    max_show: int = 500,
    color_by: str = "time",  # "time", "density", "index"
    alpha: float = 0.3,
    show_land_mask: bool = True,
    land_mask: np.ndarray | None = None,
    save_path: str | Path | None = None,
):
    """Visualize query bboxes on a map with temporal distribution.

    Args:
        queries: List of Query objects
        title: Plot title
        figsize: Figure size
        max_show: Maximum queries to display (samples if more)
        color_by: How to color boxes - "time", "density", "index"
        alpha: Box transparency
        show_land_mask: Show land mask background
        land_mask: LandMask instance (for density calculation)
        save_path: Path to save figure
    """
    if len(queries) > max_show:
        indices = np.random.choice(len(queries), max_show, replace=False)
        queries_show = [queries[i] for i in sorted(indices)]
        print(f"Showing {max_show} of {len(queries)} queries")
    else:
        queries_show = queries

    # Create figure with map and histograms
    fig = plt.figure(figsize=figsize)

    # Layout: main map + side panels
    gs = fig.add_gridspec(
        2, 3, width_ratios=[3, 1, 1], height_ratios=[3, 1], hspace=0.05, wspace=0.05
    )

    ax_map = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())
    ax_lon = fig.add_subplot(gs[1, 0])
    ax_lat = fig.add_subplot(gs[0, 1])
    ax_time = fig.add_subplot(gs[0, 2])
    ax_info = fig.add_subplot(gs[1, 1:])

    # --- Main Map ---
    ax_map.set_global()
    ax_map.coastlines(linewidth=0.5)
    ax_map.add_feature(cfeature.LAND, facecolor="lightgray", alpha=0.5)
    ax_map.add_feature(cfeature.OCEAN, facecolor="lightblue", alpha=0.3)
    ax_map.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5)

    # Color mapping
    if color_by == "time":
        dates = [pd.Timestamp(q.time_start) for q in queries_show]
        date_nums = [(d - min(dates)).days for d in dates]
        norm = Normalize(vmin=0, vmax=max(date_nums) if date_nums else 1)
        cmap = cm.viridis
        colors = [cmap(norm(d)) for d in date_nums]
    elif color_by == "index":
        norm = Normalize(vmin=0, vmax=len(queries_show))
        cmap = cm.viridis
        colors = [cmap(norm(i)) for i in range(len(queries_show))]
    else:  # density - all same color
        colors = ["blue"] * len(queries_show)

    # Draw query boxes
    for q, color in zip(queries_show, colors):
        lon_min, lon_max, lat_min, lat_max = q.bbox
        width = lon_max - lon_min
        height = lat_max - lat_min
        rect = Rectangle(
            (lon_min, lat_min),
            width,
            height,
            linewidth=0.5,
            edgecolor=color,
            facecolor=color,
            alpha=alpha,
            transform=ccrs.PlateCarree(),
        )
        ax_map.add_patch(rect)

    ax_map.set_title(
        f"{title}\n({len(queries_show)} queries shown)", fontsize=12, fontweight="bold"
    )

    # --- Longitude Distribution ---
    lon_centers = [(q.bbox[0] + q.bbox[1]) / 2 for q in queries]
    ax_lon.hist(lon_centers, bins=36, color="steelblue", edgecolor="white", alpha=0.7)
    ax_lon.set_xlabel("Longitude")
    ax_lon.set_ylabel("Count")
    ax_lon.set_xlim(-180, 180)

    # --- Latitude Distribution ---
    lat_centers = [(q.bbox[2] + q.bbox[3]) / 2 for q in queries]
    ax_lat.hist(
        lat_centers,
        bins=18,
        orientation="horizontal",
        color="steelblue",
        edgecolor="white",
        alpha=0.7,
    )
    ax_lat.set_ylabel("Latitude")
    ax_lat.set_xlabel("Count")
    ax_lat.set_ylim(-90, 90)

    # --- Temporal Distribution ---
    dates = [pd.Timestamp(q.time_start) for q in queries]
    ax_time.hist(
        dates,
        bins=min(50, len(set(dates))),
        color="coral",
        edgecolor="white",
        alpha=0.7,
        orientation="horizontal",
    )
    ax_time.set_ylabel("Date")
    ax_time.set_xlabel("Count")
    ax_time.tick_params(axis="y", labelrotation=45)

    # --- Info Panel ---
    ax_info.axis("off")

    # Compute statistics
    lon_sizes = [q.bbox[1] - q.bbox[0] for q in queries]
    lat_sizes = [q.bbox[3] - q.bbox[2] for q in queries]

    if land_mask:
        land_fracs = [
            land_mask.get_land_fraction(q.bbox)
            for q in queries[: min(1000, len(queries))]
        ]
        land_info = (
            f"Land fraction: {np.mean(land_fracs):.2f} ± {np.std(land_fracs):.2f}"
        )
    else:
        land_info = ""

    info_text = (
        f"Total queries: {len(queries)}\n"
        f"Date range: {min(dates).date()} to {max(dates).date()}\n"
        f"Patch size: {np.mean(lon_sizes):.2f}° × {np.mean(lat_sizes):.2f}°\n"
        f"Lon range: [{min(lon_centers):.1f}, {max(lon_centers):.1f}]\n"
        f"Lat range: [{min(lat_centers):.1f}, {max(lat_centers):.1f}]\n"
        f"{land_info}"
    )
    ax_info.text(
        0.1,
        0.5,
        info_text,
        transform=ax_info.transAxes,
        fontsize=10,
        verticalalignment="center",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    else:
        plt.show()

    plt.close(fig)


def visualize_query_grid(
    queries: list[Query],
    title: str = "Evaluation Grid",
    figsize: tuple[int, int] = (12, 8),
    date_to_show: str | None = None,
    save_path: str | Path | None = None,
):
    """Visualize evaluation grid queries (single date slice).

    Args:
        queries: List of Query objects
        title: Plot title
        figsize: Figure size
        date_to_show: Specific date to show (None = first date)
        save_path: Path to save figure
    """
    # Filter to single date if specified
    if date_to_show:
        queries_show = [q for q in queries if q.time_start == date_to_show]
    else:
        first_date = queries[0].time_start
        queries_show = [q for q in queries if q.time_start == first_date]

    # Get extent
    all_lons = [q.bbox[0] for q in queries_show] + [q.bbox[1] for q in queries_show]
    all_lats = [q.bbox[2] for q in queries_show] + [q.bbox[3] for q in queries_show]
    extent = [
        min(all_lons) - 1,
        max(all_lons) + 1,
        min(all_lats) - 1,
        max(all_lats) + 1,
    ]

    fig, ax = plt.subplots(
        figsize=figsize, subplot_kw={"projection": ccrs.PlateCarree()}
    )

    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.coastlines(linewidth=0.5)
    ax.add_feature(cfeature.LAND, facecolor="lightgray", alpha=0.5)
    ax.add_feature(cfeature.OCEAN, facecolor="lightblue", alpha=0.3)
    ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5)

    # Draw grid cells
    for i, q in enumerate(queries_show):
        lon_min, lon_max, lat_min, lat_max = q.bbox
        rect = Rectangle(
            (lon_min, lat_min),
            lon_max - lon_min,
            lat_max - lat_min,
            linewidth=1,
            edgecolor="red",
            facecolor="coral",
            alpha=0.3,
            transform=ccrs.PlateCarree(),
        )
        ax.add_patch(rect)

    date_shown = date_to_show or queries_show[0].time_start
    ax.set_title(
        f"{title}\nDate: {date_shown} | {len(queries_show)} patches",
        fontsize=12,
        fontweight="bold",
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    else:
        plt.show()

    plt.close(fig)


def compare_query_configs(
    configs: list[dict],
    land_mask_path: str | None = None,
    n_samples: int = 500,
    figsize: tuple[int, int] = (16, 5),
    save_path: str | Path | None = None,
):
    """Compare different query configurations side by side.

    Args:
        configs: List of config dicts with 'name' and config parameters
        land_mask_path: Path to land mask
        n_samples: Queries per config
        figsize: Figure size
        save_path: Path to save figure

    Example:
        compare_query_configs([
            {'name': '5° patches', 'patch_size': 5, 'max_land_fraction': 0.3},
            {'name': '10° patches', 'patch_size': 10, 'max_land_fraction': 0.3},
            {'name': '10° high land', 'patch_size': 10, 'max_land_fraction': 0.7},
        ])
    """
    n_configs = len(configs)
    fig, axes = plt.subplots(
        1, n_configs, figsize=figsize, subplot_kw={"projection": ccrs.PlateCarree()}
    )
    if n_configs == 1:
        axes = [axes]

    generator = QueryGenerator(land_mask_path=land_mask_path)

    for ax, cfg in zip(axes, configs):
        name = cfg.pop("name", "Config")

        # Build config
        patch_size = cfg.get("patch_size", 10)
        if isinstance(patch_size, (int, float)):
            patch_size = PatchSize(patch_size, "deg")

        config = TrainingQueryConfig(
            patch_size=patch_size,
            bbox_constraint=cfg.get("bbox_constraint", (-180, 180, -60, 60)),
            date_range=cfg.get("date_range", ("2023-01-01", "2023-12-31")),
            max_land_fraction=cfg.get("max_land_fraction", 0.3),
            require_variables=cfg.get("require_variables", []),
            validate_availability=False,
            n_queries=n_samples,
            seed=cfg.get("seed", 42),
        )

        queries = generator.generate_training_queries(config, verbose=False)

        # Plot
        ax.set_global()
        ax.coastlines(linewidth=0.5)
        ax.add_feature(cfeature.LAND, facecolor="lightgray", alpha=0.5)
        ax.gridlines(linewidth=0.3, alpha=0.5)

        for q in queries[:200]:
            lon_min, lon_max, lat_min, lat_max = q.bbox
            rect = Rectangle(
                (lon_min, lat_min),
                lon_max - lon_min,
                lat_max - lat_min,
                linewidth=0.3,
                edgecolor="blue",
                facecolor="blue",
                alpha=0.2,
                transform=ccrs.PlateCarree(),
            )
            ax.add_patch(rect)

        land_fracs = [generator.land_mask.get_land_fraction(q.bbox) for q in queries]
        ax.set_title(
            f"{name}\nn={len(queries)}, land={np.mean(land_fracs):.2f}", fontsize=10
        )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    else:
        plt.show()

    plt.close(fig)


# =============================================================================
# DataLoader Benchmarking
# =============================================================================


def benchmark_dataloader(
    dataset: OceanTACODataset,
    batch_size: int = 8,
    num_batches: int = 20,
    num_workers: int = 0,
    warmup_batches: int = 2,
    verbose: bool = True,
) -> dict:
    """Benchmark DataLoader performance.

    Args:
        dataset: OceanTACODataset instance
        batch_size: Batch size
        num_batches: Number of batches to time
        num_workers: DataLoader workers
        warmup_batches: Warmup batches (not timed)
        verbose: Print progress

    Returns:
        Dict with timing statistics
    """
    from torch.utils.data import DataLoader

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_ocean_samples,
        pin_memory=True if num_workers > 0 else False,
    )

    times = []
    batch_sizes = []
    data_shapes = defaultdict(list)

    if verbose:
        print(
            f"Benchmarking: batch_size={batch_size}, workers={num_workers}, "
            f"batches={num_batches}"
        )

    loader_iter = iter(loader)

    for i in range(warmup_batches + num_batches):
        try:
            t_start = time.perf_counter()
            batch = next(loader_iter)
            t_end = time.perf_counter()
        except StopIteration:
            if verbose:
                print("Reached end of dataset during benchmarking.")
            break

        # Check for empty data
        for kind in ["inputs", "targets"]:
            for var, tensor in batch[kind].items():
                if tensor is None:
                    raise RuntimeError(
                        f"Found None tensor for {kind} '{var}' in batch {i}"
                    )
                if tensor.numel() == 0:
                    raise RuntimeError(
                        f"Found empty tensor for {kind} '{var}' in batch {i}"
                    )

                if i >= warmup_batches:
                    data_shapes[f"{kind}_{var}"].append(tensor.shape)

        if i >= warmup_batches:
            times.append(t_end - t_start)
            batch_sizes.append(len(batch["metadata"]["bboxes"]))

        if verbose and (i + 1) % 5 == 0:
            print(f"  Batch {i + 1}/{warmup_batches + num_batches}")

    # Compute statistics
    times = np.array(times)
    total_samples = sum(batch_sizes)

    results = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "num_batches": len(times),
        "total_samples": total_samples,
        "total_time_s": times.sum(),
        "mean_batch_time_s": times.mean(),
        "std_batch_time_s": times.std(),
        "min_batch_time_s": times.min(),
        "max_batch_time_s": times.max(),
        "samples_per_second": total_samples / times.sum(),
        "batches_per_second": len(times) / times.sum(),
    }

    # Data shapes summary
    for key, shapes in data_shapes.items():
        if shapes:
            results[f"{key}_shape"] = shapes[0]

    if verbose:
        print(f"\n{'=' * 50}")
        print("Results:")
        print(f"  Total time: {results['total_time_s']:.2f}s")
        print(
            f"  Mean batch time: {results['mean_batch_time_s'] * 1000:.1f}ms "
            f"± {results['std_batch_time_s'] * 1000:.1f}ms"
        )
        print(f"  Throughput: {results['samples_per_second']:.1f} samples/s")
        print(f"  Throughput: {results['batches_per_second']:.2f} batches/s")
        print(f"{'=' * 50}")

    return results


# =============================================================================
# Quick Test Functions
# =============================================================================


def quick_test_queries(
    n_queries: int = 100,
    patch_size_deg: float = 10.0,
    land_mask_path: str | None = None,
    save_path: str | None = None,
):
    """Quick test: generate and visualize random queries.benchmark_worker_scaling

    Example:
        quick_test_queries(n_queries=200, patch_size_deg=5.0)
    """
    queries = generate_training_queries(
        n_queries=n_queries,
        patch_size=PatchSize(patch_size_deg, "deg"),
        date_range=("2023-04-01", "2023-04-02"),
        bbox_constraint=(-180, 180, -60, 60),
        max_land_fraction=0.3,
        seed=42,
        land_mask_path=land_mask_path,
    )
    visualize_queries(
        queries, title=f"Test: {patch_size_deg}° patches", save_path=save_path
    )
    return queries


def quick_test_dataloader(
    taco_path: str, n_queries: int = 50, batch_size: int = 4, num_batches: int = 10
):
    """Quick test: benchmark dataloader with generated queries.

    Example:
        quick_test_dataloader("OceanTACO.tacozip", n_queries=50)
    """
    # Generate simple queries (no validation)
    queries = generate_training_queries(
        n_queries=n_queries,
        patch_size=PatchSize(10, "deg"),
        date_range=("2023-04-01", "2023-04-02"),
        max_land_fraction=0.1,
        seed=42,
    )

    # Create dataset
    dataset = OceanTACODataset(
        taco_path=taco_path,
        queries=queries,
        input_variables=["l4_ssh"],
        target_variables=["l3_swot"],
    )

    # Benchmark
    results = benchmark_dataloader(
        dataset, batch_size=batch_size, num_batches=num_batches, num_workers=2
    )

    return results


if __name__ == "__main__":
    _configure_cartopy_dir("./.cartopy")
    # Demo: generate and visualize queries
    print("Running quick query visualization test...")
    # quick_test_queries(n_queries=200, patch_size_deg=16.0, save_path="query_test.png")
    quick_test_dataloader(
        taco_path="data/new_ssh_dataset_taco/OceanTACO.tacozip",
        n_queries=200,
        batch_size=8,
        num_batches=50,
    )
