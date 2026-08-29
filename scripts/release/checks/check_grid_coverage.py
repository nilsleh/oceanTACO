"""Independently verify that published footprints cover open ocean.

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


def open_ocean_mask(mask, patch_size: PatchSize) -> np.ndarray:
    """Return mask cells whose complete patch is inside the ocean-mask domain.

    Published positions are deliberately restricted to ocean centres, so their
    footprint union need not cover coastal slivers. It also cannot cover the
    mandatory latitude edge margin: no full patch is admitted there. This mask
    defines the independent, meaningful coverage population instead. A
    conservative mask-cell rectangle is used: it may exclude an extra edge
    cell, but can never classify a coastal cell as open ocean.
    """
    ocean = mask.ocean_mask
    rows, columns = ocean.shape
    lat_step = float(np.abs(np.diff(mask.lat)).max())
    lon_step = float(np.abs(np.diff(mask.lon)).max())
    half_rows = int(np.ceil(patch_size.to_degrees(0.0)[1] / (2.0 * lat_step)))

    # Prefix sums let each latitude row inspect all wrapped longitude windows
    # without consulting the grid builder or the published positions.
    land = np.tile(~ocean, (1, 3)).astype(np.int64)
    prefix = np.pad(land.cumsum(axis=0).cumsum(axis=1), ((1, 0), (1, 0)))

    def rectangle_sum(
        row_start: int, row_stop: int, col_start: np.ndarray, col_stop: np.ndarray
    ) -> np.ndarray:
        return (
            prefix[row_stop, col_stop]
            - prefix[row_start, col_stop]
            - prefix[row_stop, col_start]
            + prefix[row_start, col_start]
        )

    result = np.zeros_like(ocean, dtype=bool)
    centres = np.arange(columns) + columns
    for row, latitude in enumerate(mask.lat):
        row_start, row_stop = row - half_rows, row + half_rows + 1
        if row_start < 0 or row_stop > rows:
            continue
        half_columns = int(
            np.ceil(patch_size.to_degrees(float(latitude))[0] / (2.0 * lon_step))
        )
        land_cells = rectangle_sum(
            row_start,
            row_stop,
            centres - half_columns,
            centres + half_columns + 1,
        )
        result[row] = ocean[row] & (land_cells == 0)
    return result


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
    """Return open-ocean coverage failures for every required released set."""
    mask = load_released_ocean_mask()
    ocean_cells = int(mask.ocean_mask.sum())
    failures = []
    open_ocean_by_patch: dict[PatchSize, np.ndarray] = {}
    for name in (
        "128-eval",
        "256-eval",
        "512-eval",
        "128-training",
        "256-training",
        "512-training",
    ):
        directory = root / name
        header = json.loads((directory / "header.json").read_text(encoding="utf-8"))
        patch_size = PatchSize(
            float(header["patch_size"]["value"]), str(header["patch_size"]["unit"])
        )
        covered = _coverage(mask, directory / "positions.parquet", patch_size)
        if patch_size not in open_ocean_by_patch:
            open_ocean_by_patch[patch_size] = open_ocean_mask(mask, patch_size)
        open_ocean = open_ocean_by_patch[patch_size]
        uncovered = open_ocean & ~covered
        open_ocean_cells = int(open_ocean.sum())
        full_uncovered = mask.ocean_mask & ~covered
        fraction = float(uncovered.sum() / open_ocean_cells)
        largest = _largest_component_fraction(uncovered, open_ocean_cells)
        print(
            f"{name:14s} open-ocean-uncovered={fraction:.3%} "
            f"largest-component={largest:.3%} "
            f"full-ocean-uncovered={full_uncovered.sum() / ocean_cells:.3%}",
            flush=True,
        )
        if uncovered.any():
            failures.append(
                f"{name}: {uncovered.sum()}/{open_ocean_cells} open-ocean cells are "
                f"outside every published footprint; largest component={largest:.3%}"
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
