"""Independently verify that published footprints cover the released ocean mask.

This deliberately reads only positions.parquet. QuerySet.read would materialise
the full coverage table, which is irrelevant to a geometric coverage check.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from scipy import ndimage

from ocean_taco.geobox import PatchSize
from ocean_taco.sampling.ocean_mask import load_released_ocean_mask

# 1.5x the post-fix measurements, leaving room for ordinary coastline slivers.
LIMITS = {
    "128-eval": (0.018, 0.018),
    "256-eval": (0.030, 0.030),
    "512-eval": (0.040, 0.040),
    "128-training": (0.005, 0.005),
    "256-training": (0.006, 0.006),
    "512-training": (0.012, 0.012),
}


def _largest_component_fraction(uncovered: np.ndarray, ocean_cells: int) -> float:
    """Return largest 4-connected component, joining the two longitude edges."""
    labels, count = ndimage.label(
        uncovered, structure=np.array(((0, 1, 0), (1, 1, 1), (0, 1, 0)))
    )
    if not count:
        return 0.0
    parent = np.arange(count + 1)

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = int(parent[value])
        return value

    for left, right in zip(labels[:, 0], labels[:, -1], strict=True):
        if left and right:
            a, b = find(int(left)), find(int(right))
            if a != b:
                parent[b] = a
    sizes = np.zeros(count + 1, dtype=np.int64)
    for label, cells in enumerate(np.bincount(labels.ravel()), start=0):
        if label:
            sizes[find(label)] += cells
    return float(sizes.max() / ocean_cells)


def _coverage(mask, positions: Path, patch_size: PatchSize) -> np.ndarray:
    table = pq.read_table(positions, columns=["centre_lon", "centre_lat"])
    covered = np.zeros(mask.ocean_mask.shape, dtype=bool)
    for lon, lat in zip(
        table.column("centre_lon").to_numpy(),
        table.column("centre_lat").to_numpy(),
        strict=True,
    ):
        box = patch_size.footprint(float(lon), float(lat))
        row_start = int(np.searchsorted(mask.lat, box.lat_min, side="left"))
        row_stop = int(np.searchsorted(mask.lat, box.lat_max, side="right"))
        for segment in box.segments():
            col_start = int(np.searchsorted(mask.lon, segment.lon_min, side="left"))
            col_stop = int(np.searchsorted(mask.lon, segment.lon_max, side="right"))
            covered[row_start:row_stop, col_start:col_stop] = True
    return covered


def check(root: Path) -> list[str]:
    """Return geometric coverage failures for every required released set."""
    mask = load_released_ocean_mask()
    ocean_cells = int(mask.ocean_mask.sum())
    failures = []
    for name, (fraction_limit, component_limit) in LIMITS.items():
        directory = root / name
        header = json.loads((directory / "header.json").read_text(encoding="utf-8"))
        patch_size = PatchSize(
            float(header["patch_size"]["value"]), str(header["patch_size"]["unit"])
        )
        covered = _coverage(mask, directory / "positions.parquet", patch_size)
        uncovered = mask.ocean_mask & ~covered
        fraction = float(uncovered.sum() / ocean_cells)
        largest = _largest_component_fraction(uncovered, ocean_cells)
        print(
            f"{name:14s} uncovered={fraction:.3%} largest-component={largest:.3%}",
            flush=True,
        )
        if fraction > fraction_limit:
            failures.append(
                f"{name}: uncovered-ocean fraction {fraction:.3%} exceeds {fraction_limit:.3%}"
            )
        if largest > component_limit:
            failures.append(
                f"{name}: largest uncovered component {largest:.3%} exceeds {component_limit:.3%}"
            )
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(os.environ.get("ROOT", "release/querysets/v1")),
    )
    args = parser.parse_args()
    failures = check(args.root)
    for failure in failures:
        print(f"FAIL: {failure}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
