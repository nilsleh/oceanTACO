"""Measure local OceanTACO batch-draw throughput for 128 km query rows.

Example:
    python scripts/dev/benchmark_local_draw.py \
        --taco-path /path/to/OceanTACO

The catalog path is required (or may be supplied through OCEANTACO_LOCAL_PORT),
so this script cannot silently fall back to the Hub.  Its defaults draw 128
coverage-backed rows for 2023-03-29 from the released 128 km training QuerySet,
render l3_ssh and l3_swot onto 128 x 128 grids, and load batches of 16 with
four PyTorch workers.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from torch.utils.data import DataLoader

from ocean_taco import CatalogConfig, CoverageRequirement, QueryFilter, QuerySet
from ocean_taco.render import Resample
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
        width = 28
        complete = 0 if self.total == 0 else round(width * current / self.total)
        bar = "#" * complete + "-" * (width - complete)
        elapsed = time.perf_counter() - self.start
        rate = 0.0 if elapsed == 0 else current / elapsed
        print(
            f"\rDrawing batches [{bar}] {current}/{self.total} "
            f"({rate:.1f} samples/s)",
            end="",
            file=sys.stderr,
            flush=True,
        )

    def close(self) -> None:
        """Terminate the progress-bar line."""
        print(file=sys.stderr)


def parse_args() -> argparse.Namespace:
    """Parse and validate the local-only benchmark configuration."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--taco-path",
        type=Path,
        default=os.environ.get("OCEANTACO_LOCAL_PORT"),
        help="Local OceanTACO catalog/data directory (or OCEANTACO_LOCAL_PORT).",
    )
    parser.add_argument(
        "--queryset",
        type=Path,
        default=DEFAULT_QUERYSET,
        help=f"Local 128 km QuerySet directory (default: {DEFAULT_QUERYSET}).",
    )
    parser.add_argument(
        "--date",
        default="2023-03-29",
        help="Single UTC anchor date to draw (default: %(default)s).",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=128,
        help="Rows to draw; must be a positive multiple of 16.",
    )
    parser.add_argument(
        "--sources",
        default=",".join(DEFAULT_SOURCES),
        help="Comma-separated dense source tokens (default: %(default)s).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Draw seed.")
    args = parser.parse_args()
    if args.taco_path is None:
        parser.error("--taco-path is required (or set OCEANTACO_LOCAL_PORT).")
    args.taco_path = args.taco_path.expanduser().resolve()
    args.queryset = args.queryset.expanduser().resolve()
    if not args.taco_path.is_dir():
        parser.error(f"local taco path does not exist: {args.taco_path}")
    if not args.queryset.is_dir():
        parser.error(f"QuerySet directory does not exist: {args.queryset}")
    if args.samples <= 0:
        parser.error("--samples must be positive.")
    if args.samples % BATCH_SIZE:
        parser.error("--samples must be a multiple of 16 so every batch is full.")
    args.sources = tuple(token.strip() for token in args.sources.split(",") if token.strip())
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


def assert_batch_shapes(batch: dict[str, Any], sources: tuple[str, ...]) -> int:
    """Check the fixed-grid portion of the public batch contract."""
    batch_size = len(batch["query"])
    assert 1 <= batch_size <= BATCH_SIZE
    for token in sources:
        record = batch[token]
        data = record["data"]
        assert data.ndim == 4, (token, tuple(data.shape))
        assert data.shape[0] == batch_size, (token, tuple(data.shape))
        assert tuple(data.shape[-2:]) == GRID_SHAPE, (token, tuple(data.shape))
        for key in ("valid_mask", "source_valid", "support_mask", "support"):
            assert tuple(record[key].shape) == tuple(data.shape), (token, key)
        assert tuple(record["lat"].shape) == (batch_size, GRID_SHAPE[0]), token
        assert tuple(record["lon"].shape) == (batch_size, GRID_SHAPE[1]), token
        assert len(batch["availability"][token]) == batch_size, token
    return batch_size


def main() -> None:
    """Draw, validate, and time locally planned OceanTACO batches."""
    args = parse_args()
    queryset = QuerySet.read(args.queryset)
    patch_size = queryset.header["patch_size"]
    assert patch_size == {"unit": "km", "value": 128.0}, patch_size

    query_filter = QueryFilter(
        date_start=args.date,
        date_end=args.date,
        coverage=coverage_for(args.sources),
    )
    with tempfile.TemporaryDirectory(prefix="oceantaco-throughput-") as temporary:
        draw = draw_queryset(
            queryset,
            requested_row_count=args.samples,
            seed=args.seed,
            record_path=Path(temporary) / "draw.json",
            query_filter=query_filter,
        )
        dataset = OceanTACODataset(
            queries=draw,
            sources={token: Resample(GRID_SHAPE, support_threshold=0.5) for token in args.sources},
            catalog_config=CatalogConfig(taco_path=args.taco_path),
        )
        loader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            num_workers=WORKERS,
            worker_init_fn=seed_ocean_taco_worker,
            collate_fn=collate_ocean_samples,
        )
        expected_batches = math.ceil(len(dataset) / BATCH_SIZE)
        print(
            f"Drawing {len(dataset)} rows from {args.queryset.name} with "
            f"{BATCH_SIZE}-sample batches and {WORKERS} workers.\n"
            f"Local data: {args.taco_path}\nSources: {', '.join(args.sources)}"
        )
        progress = ProgressBar(expected_batches)
        start = time.perf_counter()
        samples = 0
        available = {token: 0 for token in args.sources}
        nonempty = {token: 0 for token in args.sources}
        for batch_index, batch in enumerate(loader, start=1):
            batch_size = assert_batch_shapes(batch, args.sources)
            samples += batch_size
            for token in args.sources:
                available[token] += sum(batch["availability"][token])
                nonempty[token] += int(batch[token]["data"].shape[1] > 0)
            progress.update(batch_index)
        elapsed = time.perf_counter() - start
        progress.close()

    assert samples == args.samples, (samples, args.samples)
    assert expected_batches == math.ceil(samples / BATCH_SIZE)
    for token in coverage_for(args.sources):
        source = "l3_ssh" if token.token == "ssh" else "l3_swot"
        assert nonempty[source] > 0, f"{source} never read a local data array."
        assert available[source] > 0, f"{source} had no available samples."
    print(
        f"{samples} samples in {elapsed:.2f}s: {samples / elapsed:.2f} samples/s "
        f"({expected_batches / elapsed:.2f} batches/s)"
    )
    print("Available samples: " + ", ".join(f"{key}={value}" for key, value in available.items()))


if __name__ == "__main__":
    main()
