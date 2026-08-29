"""Recover source observations and require a published footprint to contain them.

The check opens released SWOT/Argo assets directly. It does not trust coverage
rows to decide where measurements should exist; position geometry is recomputed
from positions.parquet for every sampled source date.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import xarray as xr

from ocean_taco.geobox import PatchSize
from ocean_taco.registry import get_modality
from ocean_taco.sampling.ocean_mask import load_released_ocean_mask

from check_grid_coverage import open_ocean_mask

SETS = ("128-eval", "256-eval", "512-eval", "128-training", "256-training", "512-training")


def _canonical_lon(values: np.ndarray) -> np.ndarray:
    return (np.asarray(values, dtype=float) + 180.0) % 360.0 - 180.0


def _mask_values(mask, values: np.ndarray, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Look up mask-aligned values at arbitrary source coordinates."""
    lat, lon = np.broadcast_arrays(np.asarray(lat, dtype=float), np.asarray(lon, dtype=float))
    flat_lat, flat_lon = lat.ravel(), lon.ravel()
    lat_index = np.searchsorted(mask.lat, flat_lat).clip(1, len(mask.lat) - 1)
    lon_index = np.searchsorted(mask.lon, flat_lon).clip(1, len(mask.lon) - 1)
    lat_index -= np.abs(mask.lat[lat_index - 1] - flat_lat) <= np.abs(mask.lat[lat_index] - flat_lat)
    lon_index -= np.abs(mask.lon[lon_index - 1] - flat_lon) <= np.abs(mask.lon[lon_index] - flat_lon)
    domain = (flat_lat >= mask.lat[0]) & (flat_lat <= mask.lat[-1])
    return (values[lat_index, lon_index] & domain).reshape(lat.shape)


def _cover_grid(lat: np.ndarray, lon: np.ndarray, lats: np.ndarray, lons: np.ndarray, patch: PatchSize) -> np.ndarray:
    covered = np.zeros((lat.size, lon.size), dtype=bool)
    for centre_lon, centre_lat in zip(lons, lats, strict=True):
        box = patch.footprint(float(centre_lon), float(centre_lat))
        rs = np.searchsorted(lat, box.lat_min, side="left")
        re = np.searchsorted(lat, box.lat_max, side="right")
        for segment in box.segments():
            cs = np.searchsorted(lon, segment.lon_min, side="left")
            ce = np.searchsorted(lon, segment.lon_max, side="right")
            covered[int(rs):int(re), int(cs):int(ce)] = True
    return covered


def _contains_points(lat: np.ndarray, lon: np.ndarray, lats: np.ndarray, lons: np.ndarray, patch: PatchSize) -> np.ndarray:
    height = patch.to_degrees(0.0)[1]
    widths = np.asarray([patch.to_degrees(float(value))[0] for value in lats])
    found = np.zeros(lat.size, dtype=bool)
    order = np.argsort(lats)
    sorted_lats = lats[order]
    for index, (point_lat, point_lon) in enumerate(zip(lat, lon, strict=True)):
        start = np.searchsorted(sorted_lats, point_lat - height / 2, side="left")
        stop = np.searchsorted(sorted_lats, point_lat + height / 2, side="right")
        rows = order[start:stop]
        if rows.size:
            distance = np.abs((lons[rows] - point_lon + 180.0) % 360.0 - 180.0)
            found[index] = bool(np.any(distance <= widths[rows] / 2))
    return found


def _sample_dates(total: int, count: int) -> set[int]:
    return set(np.linspace(0, total - 1, min(total, count), dtype=int).tolist())


def _swot_misses(rows, mask, open_ocean, lats, lons, patch) -> int:
    """Count uncovered finite open-ocean SWOT cells across a date's assets."""
    axes = []
    for row in rows:
        with xr.open_dataset(row["uri"], engine="h5netcdf") as dataset:
            axes.append((np.asarray(dataset["lat"].values, dtype=float), _canonical_lon(dataset["lon"].values)))
    global_lat = np.unique(np.concatenate([lat for lat, _ in axes]))
    global_lon = np.unique(np.concatenate([lon for _, lon in axes]))
    covered = _cover_grid(global_lat, global_lon, lats, lons, patch)
    spec = get_modality("l3_swot")
    missed = 0
    for row, (lat, lon) in zip(rows, axes, strict=True):
        with xr.open_dataset(row["uri"], engine="h5netcdf") as dataset:
            if spec.primary_variable not in dataset:
                raise ValueError(f"{row['uri']} lacks {spec.primary_variable}")
            order = np.argsort(lon)
            lon = lon[order]
            values = np.asarray(dataset[spec.primary_variable].squeeze().values)[:, order]
            rs = np.searchsorted(global_lat, lat)
            cs = np.searchsorted(global_lon, lon)
            eligible = _mask_values(mask, open_ocean, *np.meshgrid(lat, lon, indexing="ij"))
            missed += int((np.isfinite(values) & eligible & ~covered[np.ix_(rs, cs)]).sum())
    return missed


def check(root: Path, date_count: int, sets: tuple[str, ...] = SETS) -> list[str]:
    mask = load_released_ocean_mask()
    failures = []
    for name in sets:
        directory = root / name
        header = json.loads((directory / "header.json").read_text(encoding="utf-8"))
        patch = PatchSize(float(header["patch_size"]["value"]), str(header["patch_size"]["unit"]))
        open_ocean = open_ocean_mask(mask, patch)
        positions = pq.read_table(directory / "positions.parquet", columns=["centre_lon", "centre_lat"])
        lons, lats = positions.column("centre_lon").to_numpy(), positions.column("centre_lat").to_numpy()
        groups = defaultdict(list)
        for row in pq.read_table(directory / "assets.parquet", columns=["date_index", "token", "uri", "status"]).to_pylist():
            if row["status"] == "present" and row["token"] in {"l3_swot", "argo"}:
                groups[(row["date_index"], row["token"])].append(row)
        missed = {"l3_swot": 0, "argo": 0}
        for date_index in _sample_dates(len(header["dates"]), date_count):
            swot = groups.get((date_index, "l3_swot"), [])
            if swot:
                missed["l3_swot"] += _swot_misses(swot, mask, open_ocean, lats, lons, patch)
            for row in groups.get((date_index, "argo"), []):
                with xr.open_dataset(row["uri"], engine="h5netcdf") as dataset:
                    point_lat = np.asarray(dataset["lat"].values, dtype=float).reshape(-1)
                    point_lon = _canonical_lon(dataset["lon"].values).reshape(-1)
                    keep = _mask_values(mask, open_ocean, point_lat, point_lon)
                    if keep.any():
                        missed["argo"] += int((~_contains_points(point_lat[keep], point_lon[keep], lats, lons, patch)).sum())
        print(f"{name:14s} sampled-dates={date_count} missed-swot={missed['l3_swot']} missed-argo={missed['argo']}", flush=True)
        for token, count in missed.items():
            if count:
                failures.append(f"{name}: {count} open-ocean {token} observations are outside every published footprint")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(os.environ.get("ROOT", "release/querysets/v1")))
    parser.add_argument("--dates", type=int, default=12)
    parser.add_argument("--sets", default=",".join(SETS), help="comma-separated released set names")
    args = parser.parse_args()
    sets = tuple(name for name in args.sets.split(",") if name)
    unknown = set(sets) - set(SETS)
    if unknown:
        parser.error(f"unknown sets: {sorted(unknown)}")
    failures = check(args.root, args.dates, sets)
    for failure in failures:
        print(f"FAIL: {failure}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
