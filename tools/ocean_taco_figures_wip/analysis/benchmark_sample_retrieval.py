"""Benchmark sample retrieval performance."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Project imports — adjust paths / PYTHONPATH as needed
# ---------------------------------------------------------------------------
from ocean_taco.dataset.dataset import OceanTACODatasetV2, collate_ocean_samples
from ocean_taco.dataset.queries import (
    PatchSize,
    Query,
    QueryDataset,
    QueryGenerator,
    QuerySampler,
)

# ============================================================================
# Configuration
# ============================================================================

# Pre-defined geographic regions (lon_min, lon_max, lat_min, lat_max)
REGIONS = {
    "north_atlantic": (-60, -10, 20, 50),
    "tropical_pacific": (-180, -120, -15, 15),
    "global": (-180, 180, -60, 60),
}

# Variable sets for Experiment 3
VARIABLE_SETS = {
    "1var": ["glorys_sst"],
    "3var": ["glorys_sst", "l4_sst", "l3_ssh"],
    "6var": ["glorys_sst", "l4_sst", "l3_ssh", "l3_sst", "l4_wind", "l3_swot"],
}

# Default date range
DEFAULT_DATE_RANGE = ("2023-03-29", "2023-08-30")


# ============================================================================
# Helpers
# ============================================================================


def pixel_to_degrees(n_pixels: int, resolution_deg: float) -> float:
    """Convert a pixel count to a spatial extent in degrees."""
    return n_pixels * resolution_deg


def extract_centers(queries: list[Query]) -> list[dict[str, Any]]:
    """Extract (center_lon, center_lat, date) from Query objects.

    This lets us regenerate queries at different spatial / temporal extents
    while keeping the same ocean locations.
    """
    centers = []
    for q in queries:
        centers.append(
            {
                "center_lon": (q.lon_min + q.lon_max) / 2.0,
                "center_lat": (q.lat_min + q.lat_max) / 2.0,
                "date": str(q.time_start),
            }
        )
    return centers


def rebuild_queries(
    centers: list[dict[str, Any]], patch_size_deg: float, time_window_days: int = 1
) -> list[Query]:
    """Rebuild Query objects from centers at a new patch size / time window."""
    queries = []
    half = patch_size_deg / 2.0
    for c in centers:
        lon_min = max(-180.0, c["center_lon"] - half)
        lon_max = min(180.0, c["center_lon"] + half)
        lat_min = max(-90.0, c["center_lat"] - half)
        lat_max = min(90.0, c["center_lat"] + half)

        t_start = pd.Timestamp(c["date"])
        t_end = t_start + pd.Timedelta(days=time_window_days - 1)

        queries.append(
            Query(
                lon_min=lon_min,
                lon_max=lon_max,
                lat_min=lat_min,
                lat_max=lat_max,
                time_start=t_start.strftime("%Y-%m-%d"),
                time_end=t_end.strftime("%Y-%m-%d"),
            )
        )
    return queries


@dataclass
class TimingResult:
    """Timing for a single query retrieval."""

    query_idx: int
    elapsed_s: float
    n_files: int
    bbox: tuple[float, float, float, float]
    time_range: tuple[str, str]
    error: str | None = None


def time_single_query(
    dataset: OceanTACODatasetV2, query: Query, query_idx: int
) -> TimingResult:
    """Time a single dataset[query.to_geoslice()] call."""
    geoslice = query.to_geoslice()
    # try:
    t0 = time.perf_counter()
    sample = dataset[geoslice]
    elapsed = time.perf_counter() - t0

    return TimingResult(
        query_idx=query_idx,
        elapsed_s=elapsed,
        n_files=sample["metadata"]["n_files"],
        bbox=sample["metadata"]["bbox"],
        time_range=sample["metadata"]["time_range"],
    )
    # except Exception as e:
    #     return TimingResult(
    #         query_idx=query_idx,
    #         elapsed_s=float("nan"),
    #         n_files=0,
    #         bbox=query.bbox,
    #         time_range=(str(query.time_start), str(query.time_end)),
    #         error=str(e),
    #     )


def compute_stats(timings: list[TimingResult]) -> dict[str, Any]:
    """Compute summary statistics from a list of TimingResults."""
    valid = [t for t in timings if t.error is None]
    errors = [t for t in timings if t.error is not None]
    if not valid:
        return {
            "n_total": len(timings),
            "n_errors": len(errors),
            "error": "all queries failed",
            "first_errors": [e.error for e in errors[:3]],
        }

    times = np.array([t.elapsed_s for t in valid])
    n_files = np.array([t.n_files for t in valid])

    return {
        "n_total": len(timings),
        "n_valid": len(valid),
        "n_errors": len(errors),
        "time_median_s": float(np.median(times)),
        "time_mean_s": float(np.mean(times)),
        "time_std_s": float(np.std(times)),
        "time_p5_s": float(np.percentile(times, 5)),
        "time_p95_s": float(np.percentile(times, 95)),
        "time_min_s": float(np.min(times)),
        "time_max_s": float(np.max(times)),
        "files_median": float(np.median(n_files)),
        "files_mean": float(np.mean(n_files)),
        "throughput_queries_per_s": float(len(valid) / np.sum(times)),
    }


def print_row(label: str, stats: dict[str, Any]) -> None:
    """Pretty-print one row of benchmark results."""
    if "error" in stats:
        print(f"  {label:<30s}  ALL FAILED — {stats.get('first_errors', ['?'])}")
        return
    print(
        f"  {label:<30s}  "
        f"median={stats['time_median_s'] * 1000:8.1f}ms  "
        f"p95={stats['time_p95_s'] * 1000:8.1f}ms  "
        f"mean={stats['time_mean_s'] * 1000:8.1f}ms  "
        f"std={stats['time_std_s'] * 1000:8.1f}ms  "
        f"q/s={stats['throughput_queries_per_s']:6.2f}  "
        f"files={stats['files_mean']:5.1f}  "
        f"ok={stats['n_valid']}/{stats['n_total']}"
    )


# ============================================================================
# Query generation (uses QueryGenerator from new_query_generator.py)
# ============================================================================


def generate_base_queries(
    n_queries: int,
    largest_patch_deg: float,
    longest_time_window: int,
    bbox_constraint: tuple[float, float, float, float],
    date_range: tuple[str, str],
    land_mask_path: str | None,
    seed: int,
) -> tuple[list[Query], list[dict[str, Any]]]:
    """Generate a base set of valid ocean queries at the largest extent needed.

    Uses QueryGenerator.generate_training_queries so that all queries are
    guaranteed to be over ocean (land mask filtering) and spatially spread.

    Returns:
        (queries, centers) where centers can be reused at smaller extents.
    """
    generator = QueryGenerator(land_mask_path)

    # Use PatchSize in degrees
    patch = PatchSize(value=largest_patch_deg, unit="deg")

    queries = generator.generate_training_queries(
        n_queries=n_queries,
        patch_size=patch,
        date_range=date_range,
        bbox_constraint=bbox_constraint,
        time_window_days=longest_time_window,
        max_land_fraction=0.3,
        seed=seed,
        oversample_factor=3.0,
        verbose=True,
    )

    centers = extract_centers(queries)
    print(f"  Generated {len(queries)} base queries, extracted {len(centers)} centers.")
    return queries, centers


# ============================================================================
# Experiment runners
# ============================================================================


def run_experiment_1(
    taco_path: str,
    centers: list[dict[str, Any]],
    target_resolution: float,
    variables: list[str],
    warmup: int,
    patch_sizes_px: list[int] = [64, 128, 256],
) -> dict[str, Any]:
    """Experiment 1: Spatial extent scaling."""
    print("\n" + "=" * 72)
    print("EXPERIMENT 1: Spatial Extent Scaling")
    print(f"  Variables:          {variables}")
    print(
        f"  Target resolution:  {target_resolution}° ({target_resolution * 111:.1f} km)"
    )
    print("  Temporal window:    1 day")
    print(f"  Patch sizes (px):   {patch_sizes_px}")
    print("=" * 72)

    results = {}

    for px in patch_sizes_px:
        extent_deg = pixel_to_degrees(px, target_resolution)
        label = f"{px}x{px}px ({extent_deg:.2f}°)"
        print(f"\n  → {label}")

        dataset = OceanTACODatasetV2(
            taco_path=taco_path,
            input_variables=variables,
            target_variables=[],
            target_resolution=target_resolution,
            temporal_agg="mean",
        )

        queries = rebuild_queries(centers, extent_deg, time_window_days=1)

        # Warmup (discard timings)
        for i in range(min(warmup, len(queries))):
            _ = dataset[queries[i].to_geoslice()]

        # Timed runs
        timings = []
        for i in tqdm(
            range(warmup, len(queries)),
            desc=f"  Timing queries for {label}",
            unit="query",
        ):
            timings.append(time_single_query(dataset, queries[i], i))

        stats = compute_stats(timings)
        stats["patch_size_px"] = px
        stats["extent_deg"] = extent_deg
        print_row(label, stats)
        results[f"{px}x{px}"] = stats
        del dataset

    return results


def run_experiment_2(
    taco_path: str,
    centers: list[dict[str, Any]],
    target_resolution: float,
    variables: list[str],
    warmup: int,
    patch_size_px: int = 128,
    time_windows: list[int] = [1, 5, 15, 30],
) -> dict[str, Any]:
    """Experiment 2: Temporal depth scaling (temporal_agg='stack')."""
    extent_deg = pixel_to_degrees(patch_size_px, target_resolution)

    print("\n" + "=" * 72)
    print("EXPERIMENT 2: Temporal Depth Scaling")
    print(f"  Variables:          {variables}")
    print(
        f"  Patch size:         {patch_size_px}x{patch_size_px}px ({extent_deg:.2f}°)"
    )
    print(f"  Temporal windows:   {time_windows} days")
    print("=" * 72)

    results = {}

    for tw in time_windows:
        label = f"T={tw} day{'s' if tw > 1 else ''}"
        print(f"\n  → {label}")

        dataset = OceanTACODatasetV2(
            taco_path=taco_path,
            input_variables=variables,
            target_variables=[],
            target_resolution=target_resolution,
            temporal_agg="stack",
        )

        queries = rebuild_queries(centers, extent_deg, time_window_days=tw)

        for i in range(min(warmup, len(queries))):
            _ = dataset[queries[i].to_geoslice()]

        timings = []
        for i in tqdm(
            range(warmup, len(queries)),
            desc=f"  Timing queries for {label}",
            unit="query",
        ):
            timings.append(time_single_query(dataset, queries[i], i))

        stats = compute_stats(timings)
        stats["time_window_days"] = tw
        stats["patch_size_px"] = patch_size_px
        print_row(label, stats)
        results[f"T{tw}"] = stats
        del dataset

    return results


def run_experiment_3(
    taco_path: str,
    centers: list[dict[str, Any]],
    target_resolution: float,
    warmup: int,
    patch_size_px: int = 128,
    variable_sets: dict[str, list[str]] | None = None,
) -> dict[str, Any]:
    """Experiment 3: Variable count scaling."""
    if variable_sets is None:
        variable_sets = VARIABLE_SETS

    extent_deg = pixel_to_degrees(patch_size_px, target_resolution)

    print("\n" + "=" * 72)
    print("EXPERIMENT 3: Variable Count Scaling")
    print(
        f"  Patch size:         {patch_size_px}x{patch_size_px}px ({extent_deg:.2f}°)"
    )
    print("  Temporal window:    1 day")
    print(f"  Variable sets:      {list(variable_sets.keys())}")
    print("=" * 72)

    results = {}

    for set_name, variables in variable_sets.items():
        label = f"{set_name} ({len(variables)} vars)"
        print(f"\n  → {label}: {variables}")

        dataset = OceanTACODatasetV2(
            taco_path=taco_path,
            input_variables=variables,
            target_variables=[],
            target_resolution=target_resolution,
            temporal_agg="mean",
        )

        queries = rebuild_queries(centers, extent_deg, time_window_days=1)

        for i in range(min(warmup, len(queries))):
            _ = dataset[queries[i].to_geoslice()]

        timings = []
        for i in tqdm(
            range(warmup, len(queries)),
            desc=f"  Querying variable set '{set_name}'",
            unit="query",
        ):
            timings.append(time_single_query(dataset, queries[i], i))

        stats = compute_stats(timings)
        stats["n_variables"] = len(variables)
        stats["variable_set"] = set_name
        stats["variables"] = variables
        print_row(label, stats)
        results[set_name] = stats
        del dataset

    return results


def run_experiment_4(
    taco_path: str,
    centers: list[dict[str, Any]],
    target_resolution: float,
    variables: list[str],
    warmup: int,
    patch_size_px: int = 128,
    worker_counts: list[int] = [0, 2, 4, 8],
    batch_size: int = 4,
    max_batches: int = 20,
) -> dict[str, Any]:
    """Experiment 4: DataLoader throughput (end-to-end).

    Uses QueryDataset and QuerySampler from new_query_generator.py
    with the dataset's collate_ocean_samples.
    """
    extent_deg = pixel_to_degrees(patch_size_px, target_resolution)

    print("\n" + "=" * 72)
    print("EXPERIMENT 4: DataLoader Throughput")
    print(f"  Variables:          {variables}")
    print(
        f"  Patch size:         {patch_size_px}x{patch_size_px}px ({extent_deg:.2f}°)"
    )
    print(f"  Batch size:         {batch_size}")
    print(f"  Max batches:        {max_batches}")
    print(f"  Worker counts:      {worker_counts}")
    print("=" * 72)

    queries = rebuild_queries(centers, extent_deg, time_window_days=1)

    results = {}

    for nw in worker_counts:
        label = f"workers={nw}"
        print(f"\n  → {label}")

        dataset = OceanTACODatasetV2(
            taco_path=taco_path,
            input_variables=variables,
            target_variables=[],
            target_resolution=target_resolution,
            temporal_agg="mean",
        )

        # Use QueryDataset + QuerySampler from new_query_generator
        qds = QueryDataset(queries=queries, dataset=dataset)
        sampler = QuerySampler(queries=queries, shuffle=False)

        loader = DataLoader(
            qds,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=nw,
            collate_fn=collate_ocean_samples,
            pin_memory=False,
            prefetch_factor=2 if nw > 0 else None,
            persistent_workers=nw > 0,
        )

        # Warmup
        loader_iter = iter(loader)
        for _ in range(min(warmup, max_batches)):
            _ = next(loader_iter)

        # Timed iteration
        loader_iter = iter(loader)
        batch_times = []
        n_samples_total = 0

        t_wall_start = time.perf_counter()
        for _ in tqdm(
            range(max_batches),
            desc=f"  Iterating DataLoader with {nw} workers",
            unit="batch",
        ):
            t0 = time.perf_counter()
            batch = next(loader_iter)
            elapsed = time.perf_counter() - t0
            batch_times.append(elapsed)
            n_samples_total += len(batch["metadata"]["bboxes"])
        t_wall_total = time.perf_counter() - t_wall_start

        bt = np.array(batch_times) if batch_times else np.array([float("nan")])
        stats = {
            "num_workers": nw,
            "batch_size": batch_size,
            "n_batches": len(batch_times),
            "n_samples": n_samples_total,
            "wall_time_s": float(t_wall_total),
            "samples_per_s": float(n_samples_total / t_wall_total)
            if t_wall_total > 0
            else 0,
            "batch_time_median_s": float(np.median(bt)),
            "batch_time_p95_s": float(np.percentile(bt, 95)),
            "batch_time_mean_s": float(np.mean(bt)),
        }

        print(
            f"  {label:<30s}  "
            f"samples/s={stats['samples_per_s']:6.2f}  "
            f"batch_median={stats['batch_time_median_s'] * 1000:8.1f}ms  "
            f"batch_p95={stats['batch_time_p95_s'] * 1000:8.1f}ms  "
            f"wall={stats['wall_time_s']:.1f}s  "
            f"batches={stats['n_batches']}"
        )

        results[f"w{nw}"] = stats
        del dataset, loader, qds

    return results


# ============================================================================
# Main
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark retrieval speed for OceanTACO dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--taco_path", type=str, required=True, help="Path to the TACO dataset."
    )
    parser.add_argument(
        "--target_resolution",
        type=float,
        default=1 / 12,
        help="Target resolution in degrees (e.g. 0.083 for ~1/12°).",
    )
    parser.add_argument(
        "--n_queries", type=int, default=50, help="Number of query centers to generate."
    )
    parser.add_argument(
        "--warmup", type=int, default=3, help="Number of warmup queries to discard."
    )
    parser.add_argument(
        "--region",
        type=str,
        default="north_atlantic",
        choices=list(REGIONS.keys()),
        help="Geographic region for query sampling.",
    )
    parser.add_argument(
        "--date_start",
        type=str,
        default="2025-05-01",
        help="Start of date range for queries.",
    )
    parser.add_argument(
        "--date_end",
        type=str,
        default="2025-08-01",
        help="End of date range for queries.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./benchmark_results",
        help="Directory for output JSON results.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility."
    )
    parser.add_argument(
        "--land_mask_path",
        type=str,
        default=None,
        help="Path to .npy land mask (generated if absent).",
    )
    parser.add_argument(
        "--experiments",
        type=str,
        nargs="+",
        default=["1", "2", "3", "4"],
        help="Which experiments to run (1 2 3 4).",
    )
    parser.add_argument(
        "--variables",
        type=str,
        nargs="+",
        default=["glorys_sst", "l4_sst", "l3_ssh", "l3_swot", "l4_wind"],
        help="Default variable set for experiments 1, 2, 4.",
    )
    # Experiment-specific overrides
    parser.add_argument(
        "--patch_sizes_px",
        type=int,
        nargs="+",
        default=[64, 128, 256],
        help="Patch sizes in pixels for Experiment 1.",
    )
    parser.add_argument(
        "--time_windows",
        type=int,
        nargs="+",
        default=[1, 5, 15],
        help="Temporal windows in days for Experiment 2.",
    )
    parser.add_argument(
        "--worker_counts",
        type=int,
        nargs="+",
        default=[0, 2, 4, 8, 16],
        help="Number of DataLoader workers for Experiment 4.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=4, help="Batch size for Experiment 4."
    )
    parser.add_argument(
        "--max_batches",
        type=int,
        default=10,
        help="Max batches to iterate in Experiment 4.",
    )

    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    region_bbox = REGIONS[args.region]
    date_range = (args.date_start, args.date_end)

    # Compute the largest spatial extent we'll need (for base query generation)
    max_px = max(args.patch_sizes_px)
    largest_extent_deg = pixel_to_degrees(max_px, args.target_resolution)
    longest_tw = max(args.time_windows) if "2" in args.experiments else 1

    print("=" * 72)
    print("OceanTACO Retrieval Benchmark")
    print("=" * 72)
    print(f"  TACO path:          {args.taco_path}")
    print(
        f"  Target resolution:  {args.target_resolution:.4f}° "
        f"(~{args.target_resolution * 111:.1f} km)"
    )
    print(f"  Region:             {args.region} {region_bbox}")
    print(f"  Date range:         {date_range}")
    print(f"  N queries:          {args.n_queries}")
    print(f"  Warmup:             {args.warmup}")
    print(f"  Seed:               {args.seed}")
    print(f"  Experiments:        {args.experiments}")
    print(f"  Largest patch:      {max_px}px = {largest_extent_deg:.2f}°")
    print(f"  Longest time win:   {longest_tw} days")

    # ------------------------------------------------------------------
    # Generate base queries using QueryGenerator (valid ocean locations)
    # ------------------------------------------------------------------
    print("\n--- Generating base queries via QueryGenerator ---")
    base_queries, centers = generate_base_queries(
        n_queries=args.n_queries,
        largest_patch_deg=largest_extent_deg,
        longest_time_window=longest_tw,
        bbox_constraint=region_bbox,
        date_range=date_range,
        land_mask_path=args.land_mask_path,
        seed=args.seed,
    )

    if len(centers) == 0:
        print(
            "ERROR: No valid queries generated. Check region / date range / land mask."
        )
        return

    all_results: dict[str, Any] = {
        "config": {
            "taco_path": args.taco_path,
            "target_resolution_deg": args.target_resolution,
            "target_resolution_km": args.target_resolution * 111,
            "region": args.region,
            "region_bbox": region_bbox,
            "date_range": date_range,
            "n_queries_requested": args.n_queries,
            "n_queries_generated": len(centers),
            "warmup": args.warmup,
            "seed": args.seed,
            "timestamp": datetime.now().isoformat(),
        }
    }

    # ----- Experiment 1: Spatial extent -----
    if "1" in args.experiments:
        all_results["experiment_1_spatial"] = run_experiment_1(
            taco_path=args.taco_path,
            centers=centers,
            target_resolution=args.target_resolution,
            variables=args.variables,
            warmup=args.warmup,
            patch_sizes_px=args.patch_sizes_px,
        )

    # ----- Experiment 2: Temporal depth -----
    if "2" in args.experiments:
        all_results["experiment_2_temporal"] = run_experiment_2(
            taco_path=args.taco_path,
            centers=centers,
            target_resolution=args.target_resolution,
            variables=args.variables,
            warmup=args.warmup,
            time_windows=args.time_windows,
        )

    # ----- Experiment 3: Variable count -----
    if "3" in args.experiments:
        all_results["experiment_3_variables"] = run_experiment_3(
            taco_path=args.taco_path,
            centers=centers,
            target_resolution=args.target_resolution,
            warmup=args.warmup,
        )

    # ----- Experiment 4: DataLoader throughput -----
    if "4" in args.experiments:
        all_results["experiment_4_dataloader"] = run_experiment_4(
            taco_path=args.taco_path,
            centers=centers,
            target_resolution=args.target_resolution,
            variables=args.variables,
            warmup=args.warmup,
            worker_counts=args.worker_counts,
            batch_size=args.batch_size,
            max_batches=args.max_batches,
        )

    # ----- Save results -----
    out_path = output_dir / "benchmark_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n{'=' * 72}")
    print(f"Results saved to {out_path}")

    # ----- Summary -----
    print(f"\n{'=' * 72}")
    print("SUMMARY")
    print("=" * 72)
    for exp_key, exp_data in all_results.items():
        if not exp_key.startswith("experiment_"):
            continue
        print(f"\n  {exp_key}:")
        if isinstance(exp_data, dict):
            for setting, stats in exp_data.items():
                if isinstance(stats, dict):
                    if "time_median_s" in stats:
                        print(
                            f"    {setting:<25s}  "
                            f"median={stats['time_median_s'] * 1000:8.1f}ms  "
                            f"p95={stats['time_p95_s'] * 1000:8.1f}ms"
                        )
                    elif "samples_per_s" in stats:
                        print(
                            f"    {setting:<25s}  "
                            f"samples/s={stats['samples_per_s']:6.2f}  "
                            f"wall={stats['wall_time_s']:.1f}s"
                        )
    print("\nDone.")


if __name__ == "__main__":
    main()
