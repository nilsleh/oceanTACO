"""Measure local OceanTACO retrieval, startup, cache reuse, and worker memory.

The existing invocation and defaults are retained::

    python scripts/dev/benchmark_local_draw.py --taco-path /path/to/OceanTACO

Use --queryset for published 128/256/512 km sets, --grid-size 32/64/128/256,
--workers 0/2/4/8, --prefetch-factor 1/2, and --epochs with
--persistent-workers for repeated epochs. --date-end and --shuffle exercise
multi-date draws. Repeat --diagnostic-patch LON LAT KM for explicit interior,
seam, antimeridian, or 64 km patches; those bypass coverage-backed drawing.
Sources may include argo and glorys_currents (the registered vector pair).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import resource
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from ocean_taco import (
    CatalogConfig,
    CoverageRequirement,
    PatchSize,
    PatchSpec,
    QueryFilter,
    QuerySet,
)
from ocean_taco.render import Points, Resample, VectorPair
from ocean_taco.sampling import draw_queryset
from ocean_taco.torch import (
    OceanTACODataset,
    collate_ocean_samples,
    seed_ocean_taco_worker,
)

BATCH_SIZE = 16
WORKERS = 4
GRID_SHAPE = (128, 128)
DEFAULT_SOURCES = ("l3_ssh", "l3_swot")
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_QUERYSET = ROOT / "release/querysets/v2/128-training"


class ProgressBar:
    """A dependency-free, terminal-friendly progress bar."""

    def __init__(self, total: int) -> None:
        self.total = total
        self.start = time.perf_counter()

    def update(self, current: int) -> None:
        """Redraw the bar with the completed batch count."""
        complete = 0 if self.total == 0 else round(28 * current / self.total)
        elapsed = time.perf_counter() - self.start
        rate = 0.0 if elapsed == 0 else current / elapsed
        print(
            f"\rDrawing batches [{'#' * complete}{'-' * (28 - complete)}] "
            f"{current}/{self.total} ({rate:.1f} batches/s)",
            end="",
            file=sys.stderr,
            flush=True,
        )

    def close(self) -> None:
        """Terminate the progress-bar line."""
        print(file=sys.stderr)


class MeasuredDataset(OceanTACODataset):
    """Attach process counters to each batch without altering source samples."""

    def __getitems__(self, indices):
        """Measure worker service time as well as the parent's delivery latency."""
        start = time.perf_counter()
        samples = super().__getitems__(indices)
        backend = self.source_loader._backend
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # Linux reports KiB; macOS reports bytes.
        peak_bytes = peak if sys.platform == "darwin" else peak * 1024
        samples[0]["_benchmark"] = {
            "pid": os.getpid(),
            "file_opens": 0 if backend is None else backend.file_opens,
            "cache_hits": 0 if backend is None else backend.cache_hits,
            "peak_rss_bytes": peak_bytes,
            "service_seconds": time.perf_counter() - start,
        }
        return samples


def measured_collate(samples):
    """Transport metrics beside the public collated batch."""
    metrics = samples[0].pop("_benchmark")
    return collate_ocean_samples(samples), metrics


def parse_args() -> argparse.Namespace:
    """Parse and validate the local-only benchmark configuration."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--taco-path", type=Path, default=os.environ.get("OCEANTACO_LOCAL_PORT")
    )
    parser.add_argument("--queryset", type=Path, default=DEFAULT_QUERYSET)
    parser.add_argument("--date", default="2023-03-29")
    parser.add_argument(
        "--date-end", help="Last anchor date (inclusive); defaults to --date."
    )
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--sources", default=",".join(DEFAULT_SOURCES))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument(
        "--grid-size", type=int, choices=(32, 64, 128, 256), default=128
    )
    parser.add_argument("--workers", type=int, default=WORKERS)
    parser.add_argument("--prefetch-factor", type=int, choices=(1, 2), default=2)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--persistent-workers", action="store_true")
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--max-open-files", type=int, default=16)
    parser.add_argument("--context-start", type=int, default=0)
    parser.add_argument("--context-end", type=int, default=0)
    parser.add_argument(
        "--diagnostic-patch",
        nargs=3,
        type=float,
        action="append",
        metavar=("LON", "LAT", "KM"),
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        help="Write timings and cumulative process counters as JSON.",
    )
    args = parser.parse_args()
    if args.taco_path is None:
        parser.error("--taco-path is required (or set OCEANTACO_LOCAL_PORT).")
    args.taco_path = args.taco_path.expanduser().resolve()
    args.queryset = args.queryset.expanduser().resolve()
    if not args.taco_path.is_dir():
        parser.error(f"local taco path does not exist: {args.taco_path}")
    if not args.diagnostic_patch and not args.queryset.is_dir():
        parser.error(f"QuerySet directory does not exist: {args.queryset}")
    if (
        min(args.samples, args.batch_size, args.epochs, args.max_open_files) <= 0
        or args.workers < 0
    ):
        parser.error(
            "Samples, batch size, epochs, and max open files must be positive; workers must be non-negative."
        )
    if args.context_start > args.context_end:
        parser.error("--context-start must not exceed --context-end.")
    if args.persistent_workers and args.workers == 0:
        parser.error("--persistent-workers requires workers > 0.")
    args.sources = tuple(
        token.strip() for token in args.sources.split(",") if token.strip()
    )
    if not args.sources:
        parser.error("--sources must name at least one source.")
    return args


def coverage_for(sources: tuple[str, ...]) -> tuple[CoverageRequirement, ...]:
    """Require stored evidence for sources whose coverage is in the QuerySet."""
    requirements = {
        "l3_ssh": CoverageRequirement("ssh", "valid_ocean_cells", 1),
        "l3_swot": CoverageRequirement("swot", "valid_ocean_cells", 1),
    }
    return tuple(requirements[token] for token in sources if token in requirements)


def assert_batch_shapes(
    batch: dict[str, Any],
    sources: tuple[str, ...],
    shape=GRID_SHAPE,
    maximum=BATCH_SIZE,
) -> int:
    """Check scalar grids, paired grids, and ragged point batch contracts."""
    size = len(batch["query"])
    assert 1 <= size <= maximum
    for token in sources:
        record, availability = batch[token], batch["availability"][token]
        data = record["data"]
        assert data.shape[0] == size and len(availability) == size
        if "pres" in record:
            assert data.ndim == 2 and record["point_mask"].shape == data.shape
            continue
        paired = "pair_available" in record
        assert data.ndim == (5 if paired else 4), (token, tuple(data.shape))
        assert tuple(data.shape[-2:]) == shape, (token, tuple(data.shape))
        mask_shape = (size, data.shape[1], *shape) if paired else tuple(data.shape)
        for key in ("valid_mask", "source_valid", "support_mask", "support"):
            assert tuple(record[key].shape) == mask_shape, (token, key)
        assert tuple(record["lat"].shape) == (size, shape[0])
        assert tuple(record["lon"].shape) == (size, shape[1])
    return size


def percentiles(values):
    """Return delivery/service latency percentiles in seconds."""
    return {
        f"p{percentile}": float(np.percentile(values, percentile))
        for percentile in (50, 90, 95, 99)
    }


def main() -> None:
    """Draw, validate, and separately time startup and repeated loader epochs."""
    args = parse_args()
    timings = {}
    start = time.perf_counter()
    queryset = None if args.diagnostic_patch else QuerySet.read(args.queryset)
    timings["queryset_read_seconds"] = time.perf_counter() - start
    if queryset is not None:
        assert queryset.header["patch_size"] in [
            {"unit": "km", "value": size} for size in (128.0, 256.0, 512.0)
        ]
    with tempfile.TemporaryDirectory(prefix="oceantaco-throughput-") as temporary:
        start = time.perf_counter()
        if args.diagnostic_patch:
            dates = np.arange(
                np.datetime64(args.date, "D"),
                np.datetime64(args.date_end or args.date, "D") + 1,
            )
            if not dates.size:
                raise ValueError("--date-end must not precede --date.")
            patches = [
                PatchSpec(
                    lon,
                    lat,
                    PatchSize(km, "km"),
                    str(day),
                    args.context_start,
                    args.context_end,
                )
                for lon, lat, km in args.diagnostic_patch
                for day in dates
            ]
            draw = [patches[index % len(patches)] for index in range(args.samples)]
        else:
            draw = draw_queryset(
                queryset,
                requested_row_count=args.samples,
                seed=args.seed,
                record_path=Path(temporary) / "draw.json",
                query_filter=QueryFilter(
                    date_start=args.date,
                    date_end=args.date_end or args.date,
                    coverage=coverage_for(args.sources),
                    context_start_offset_days=args.context_start,
                    context_end_offset_days=args.context_end,
                ),
            )
        timings["drawing_seconds"] = time.perf_counter() - start
        shape = (args.grid_size, args.grid_size)
        renderers = {
            token: Points()
            if token == "argo"
            else VectorPair(Resample(shape, 0.5))
            if token == "glorys_currents"
            else Resample(shape, 0.5)
            for token in args.sources
        }
        start = time.perf_counter()
        dataset = MeasuredDataset(
            queries=draw,
            sources=renderers,
            catalog_config=CatalogConfig(
                taco_path=args.taco_path, max_open_files=args.max_open_files
            ),
        )
        timings["planning_seconds"] = time.perf_counter() - start
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            num_workers=args.workers,
            worker_init_fn=seed_ocean_taco_worker,
            collate_fn=measured_collate,
            shuffle=args.shuffle,
            generator=torch.Generator().manual_seed(args.seed),
            persistent_workers=args.persistent_workers,
            **({"prefetch_factor": args.prefetch_factor} if args.workers else {}),
        )
        expected_batches = math.ceil(len(dataset) / args.batch_size)
        print(
            f"Drawing {len(dataset)} rows with {args.batch_size}-sample batches and {args.workers} workers.\n"
            f"Local data: {args.taco_path}\nSources: {', '.join(args.sources)}\nStartup: {timings}"
        )
        epochs, processes = [], {}
        try:
            for epoch in range(args.epochs):
                progress = ProgressBar(expected_batches)
                start = previous = time.perf_counter()
                samples, first_samples, first_latency = 0, 0, 0.0
                available = {token: 0 for token in args.sources}
                latencies, services = [], []
                previous_opens = sum(item["file_opens"] for item in processes.values())
                previous_hits = sum(item["cache_hits"] for item in processes.values())
                for batch_index, (batch, metrics) in enumerate(loader, start=1):
                    received = time.perf_counter()
                    latencies.append(received - previous)
                    previous = received
                    count = assert_batch_shapes(
                        batch, args.sources, shape, args.batch_size
                    )
                    samples += count
                    if batch_index == 1:
                        first_latency, first_samples = received - start, count
                    for token in args.sources:
                        available[token] += sum(batch["availability"][token])
                    processes[metrics["pid"]] = metrics
                    services.append(metrics["service_seconds"])
                    progress.update(batch_index)
                elapsed = time.perf_counter() - start
                progress.close()
                assert samples == args.samples
                entry = {
                    "epoch": epoch + 1,
                    "samples": samples,
                    "elapsed_seconds": elapsed,
                    "samples_per_second": samples / elapsed,
                    "batches_per_second": expected_batches / elapsed,
                    "first_batch_seconds": first_latency,
                    "steady_samples_per_second": (samples - first_samples)
                    / (previous - start - first_latency)
                    if samples > first_samples
                    else None,
                    "batch_latency_seconds": percentiles(latencies),
                    "worker_service_seconds": percentiles(services),
                    "file_opens": sum(item["file_opens"] for item in processes.values())
                    - previous_opens,
                    "cache_hits": sum(item["cache_hits"] for item in processes.values())
                    - previous_hits,
                    "peak_process_rss_bytes": max(
                        item["peak_rss_bytes"] for item in processes.values()
                    ),
                    "available_samples": available,
                }
                epochs.append(entry)
                print(
                    f"Epoch {epoch + 1}: {samples} samples in {elapsed:.2f}s: {samples / elapsed:.2f} samples/s "
                    f"({expected_batches / elapsed:.2f} batches/s)"
                )
                print(json.dumps(entry, sort_keys=True))
        finally:
            if loader._iterator is not None:
                loader._iterator._shutdown_workers()
            dataset.source_loader.close()
    report = {
        "configuration": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "startup": timings,
        "epochs": epochs,
        "processes": processes,
        "memory_note": "ru_maxrss high-water mark per process (includes inherited parent memory with fork); workers=0 measures the parent.",
    }
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
