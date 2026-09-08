from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ocean_taco import GeoBox, PatchSize, QuerySet
from ocean_taco.sampling import load_released_ocean_mask
from ocean_taco.sampling.grids import (
    _eligible_random_centres,
    _stratum_ids,
    _stratum_quotas,
    patch_iou,
)

TARGET = 8800
MAXIMUM_IOU = 0.20
SEED = 20260907
PATCH = PatchSize(256, "km")
BOX = GeoBox(-60, -52, 28, 34)
ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "docs/images/training-queryset-current-vs-best-candidate-iou020.png"


def sample() -> tuple[list[dict[str, float]], float]:
    mask = load_released_ocean_mask()
    latitudes, longitudes = _eligible_random_centres(mask, PATCH)
    weights = np.cos(np.radians(latitudes))
    strata = _stratum_ids(latitudes, longitudes)
    quotas = _stratum_quotas(strata, weights, TARGET)
    rng = np.random.Generator(np.random.PCG64(SEED))
    keys = -np.log1p(-rng.random(len(latitudes))) / weights
    stratum_orders = [
        indices[np.argsort(keys[indices], kind="stable")]
        for indices in (np.flatnonzero(strata == value) for value in range(len(quotas)))
    ]
    cursors = np.zeros(len(quotas), dtype=np.int64)
    buffers = [[] for _ in quotas]
    selected: list[dict[str, float]] = []
    buckets: dict[tuple[int, int], list[int]] = {}
    max_seen = 0.0
    latitude_bucket = PATCH.to_degrees(0.0)[1]
    longitude_bucket = max(PATCH.to_degrees(float(value))[0] for value in (latitudes[0], latitudes[-1]))
    longitude_bucket_count = max(1, int(np.ceil(360.0 / longitude_bucket)))

    def nearby(candidate: dict[str, float]) -> list[int]:
        latitude_key = int(np.floor((candidate["centre_lat"] - float(latitudes[0])) / latitude_bucket))
        longitude_key = int(np.floor((candidate["centre_lon"] + 180.0) / longitude_bucket)) % longitude_bucket_count
        result: list[int] = []
        for lat_key in range(latitude_key - 1, latitude_key + 2):
            for offset in (-1, 0, 1):
                result.extend(buckets.get((lat_key, (longitude_key + offset) % longitude_bucket_count), ()))
        return result

    def add(candidate: dict[str, float]) -> None:
        latitude_key = int(np.floor((candidate["centre_lat"] - float(latitudes[0])) / latitude_bucket))
        longitude_key = int(np.floor((candidate["centre_lon"] + 180.0) / longitude_bucket)) % longitude_bucket_count
        buckets.setdefault((latitude_key, longitude_key), []).append(len(selected))
        selected.append(candidate)

    remaining = quotas.copy()
    while int(remaining.sum()):
        active = np.flatnonzero(remaining > 0)
        stratum = int(active[np.argmax(remaining[active] / quotas[active])])
        order = stratum_orders[stratum]
        while len(buffers[stratum]) < 16 and cursors[stratum] < len(order):
            stop = min(int(cursors[stratum]) + 16, len(order))
            buffers[stratum].extend(int(index) for index in order[int(cursors[stratum]):stop])
            cursors[stratum] = stop
        scored: list[tuple[float, int, dict[str, float]]] = []
        for index in buffers[stratum]:
            candidate = {"centre_lon": float(longitudes[index]), "centre_lat": float(latitudes[index])}
            score = max((patch_iou(candidate, selected[other], PATCH) for other in nearby(candidate)), default=0.0)
            if score <= MAXIMUM_IOU:
                scored.append((score, index, candidate))
        if not scored:
            buffers[stratum] = []
            if cursors[stratum] >= len(order):
                deficit = int(remaining[stratum])
                remaining[stratum] = 0
                destinations = np.flatnonzero((remaining > 0) & (cursors < np.array([len(value) for value in stratum_orders])))
                if not len(destinations):
                    raise RuntimeError(f"no eligible stratum remains for {deficit} positions")
                for index in range(deficit):
                    remaining[destinations[index % len(destinations)]] += 1
            continue
        score, chosen_index, candidate = min(scored, key=lambda value: (value[0], value[1]))
        buffers[stratum].remove(chosen_index)
        max_seen = max(max_seen, score)
        add(candidate)
        remaining[stratum] -= 1
    return selected, max_seen


def plot(current: QuerySet, alternative: list[dict[str, float]], max_iou: float) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)
    groups = (("Current v2: 11,001 positions, IoU cap 0.35", current.positions, "#2474a6"),
              (f"Best-candidate preview: {len(alternative):,} positions, IoU cap 0.20", alternative, "#d95f02"))
    for axis, (title, positions, colour) in zip(axes, groups, strict=True):
        rows = [row for row in positions if BOX.lon_min <= row["centre_lon"] <= BOX.lon_max and BOX.lat_min <= row["centre_lat"] <= BOX.lat_max]
        for row in rows:
            footprint = PATCH.footprint(row["centre_lon"], row["centre_lat"])
            axis.add_patch(plt.Rectangle((footprint.lon_min, footprint.lat_min), footprint.lon_max-footprint.lon_min, footprint.lat_max-footprint.lat_min, fill=False, edgecolor=colour, linewidth=1.1, alpha=.7))
        axis.scatter([row["centre_lon"] for row in rows], [row["centre_lat"] for row in rows], s=9, color=colour, zorder=3)
        axis.set(title=f"{title}\n{len(rows)} footprints in this zoom", xlim=(BOX.lon_min-.5, BOX.lon_max+.5), ylim=(BOX.lat_min-.5, BOX.lat_max+.5), xlabel="longitude [°]", ylabel="latitude [°]")
        axis.grid(alpha=.2)
    fig.suptitle(f"256 km training footprints; experimental maximum IoU = {max_iou:.3f}")
    fig.savefig(OUTPUT, dpi=180)
    print(OUTPUT)


def main() -> None:
    alternative, max_iou = sample()
    current = QuerySet.read(ROOT / "release/querysets/v2/256-training")
    plot(current, alternative, max_iou)
    print(f"positions={len(alternative)} max_iou={max_iou:.6f}")


if __name__ == "__main__":
    main()
