"""Blind geometric spot-check for published QuerySet positions.

Candidate centres are drawn from the frozen ocean mask, not from positions.parquet:
a published set cannot certify a missing part of the ocean by sampling only
itself. A candidate must be explainable by containment in a published footprint.
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

SETS = ("128-eval", "256-eval", "512-eval", "128-training", "256-training", "512-training")


def check(root: Path, draws: int = 60) -> list[str]:
    """Return unexplained ocean-mask candidates for every published set."""
    mask = load_released_ocean_mask()
    ocean_rows, ocean_cols = np.nonzero(mask.ocean_mask)
    rng = np.random.default_rng(20260827)
    failures = []
    for name in SETS:
        directory = root / name
        header = json.loads((directory / "header.json").read_text(encoding="utf-8"))
        patch = PatchSize(
            float(header["patch_size"]["value"]), str(header["patch_size"]["unit"])
        )
        table = pq.read_table(
            directory / "positions.parquet", columns=["centre_lon", "centre_lat"]
        )
        lons = table.column("centre_lon").to_numpy()
        lats = table.column("centre_lat").to_numpy()
        widths = np.array([patch.to_degrees(float(lat))[0] for lat in lats])
        height = patch.to_degrees(0.0)[1]
        eligible = np.flatnonzero(
            (mask.lat[ocean_rows] - height / 2 >= mask.lat[0])
            & (mask.lat[ocean_rows] + height / 2 <= mask.lat[-1])
        )
        chosen = rng.choice(eligible, size=min(draws, len(eligible)), replace=False)
        unexplained = []
        for index in chosen:
            lon, lat = float(mask.lon[ocean_cols[index]]), float(mask.lat[ocean_rows[index]])
            lon_distance = np.abs((lons - lon + 180.0) % 360.0 - 180.0)
            if not np.any((np.abs(lats - lat) <= height / 2) & (lon_distance <= widths / 2)):
                unexplained.append((lon, lat))
        print(f"{name:14s} blind-draws={len(chosen)} unexplained={len(unexplained)}", flush=True)
        if unexplained:
            lon, lat = unexplained[0]
            failures.append(
                f"{name}: {len(unexplained)}/{len(chosen)} blind ocean candidates are outside every published footprint; first={lon:.3f},{lat:.3f}"
            )
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(os.environ.get("ROOT", "release/querysets/v1")))
    parser.add_argument("--draws", type=int, default=60)
    args = parser.parse_args()
    failures = check(args.root, args.draws)
    for failure in failures:
        print(f"FAIL: {failure}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
