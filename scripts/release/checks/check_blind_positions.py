"""Blind geometric spot-check for published QuerySet positions.

Candidates are drawn from independently computed open ocean, never from
positions.parquet. A published set therefore cannot certify a missing open-ocean
region by sampling only itself.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from ocean_taco.geobox import PatchSize
from ocean_taco.sampling.ocean_mask import load_released_ocean_mask

from check_grid_coverage import open_ocean_mask

SETS = (
    "128-eval",
    "256-eval",
    "512-eval",
    "128-training",
    "256-training",
    "512-training",
)


def check(root: Path, draws: int = 60) -> list[str]:
    """Return unexplained open-ocean candidates for every published set."""
    mask = load_released_ocean_mask()
    rng = np.random.default_rng(20260827)
    failures = []
    for name in SETS:
        directory = root / name
        header = json.loads((directory / "header.json").read_text(encoding="utf-8"))
        patch = PatchSize(
            float(header["patch_size"]["value"]), str(header["patch_size"]["unit"])
        )
        candidate_rows, candidate_cols = np.nonzero(open_ocean_mask(mask, patch))
        table = pq.read_table(
            directory / "positions.parquet", columns=["centre_lon", "centre_lat"]
        )
        lons = table.column("centre_lon").to_numpy()
        lats = table.column("centre_lat").to_numpy()
        widths = np.array([patch.to_degrees(float(lat))[0] for lat in lats])
        height = patch.to_degrees(0.0)[1]
        chosen = rng.choice(
            len(candidate_rows), size=min(draws, len(candidate_rows)), replace=False
        )
        unexplained = []
        for index in chosen:
            lon = float(mask.lon[candidate_cols[index]])
            lat = float(mask.lat[candidate_rows[index]])
            lon_distance = np.abs((lons - lon + 180.0) % 360.0 - 180.0)
            if not np.any(
                (np.abs(lats - lat) <= height / 2) & (lon_distance <= widths / 2)
            ):
                unexplained.append((lon, lat))
        print(
            f"{name:14s} open-ocean-blind-draws={len(chosen)} "
            f"unexplained={len(unexplained)}",
            flush=True,
        )
        if unexplained:
            lon, lat = unexplained[0]
            failures.append(
                f"{name}: {len(unexplained)}/{len(chosen)} blind open-ocean "
                f"candidates are outside every published footprint; "
                f"first={lon:.3f},{lat:.3f}"
            )
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path(os.environ.get("ROOT", "release/querysets/v1"))
    )
    parser.add_argument("--draws", type=int, default=60)
    args = parser.parse_args()
    failures = check(args.root, args.draws)
    for failure in failures:
        print(f"FAIL: {failure}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
