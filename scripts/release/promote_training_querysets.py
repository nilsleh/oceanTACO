#!/usr/bin/env python3
"""Promote regenerated best-candidate training QuerySets into release/querysets/v2."""

from __future__ import annotations

from pathlib import Path

from ocean_taco import QuerySet
from ocean_taco.sampling.grids import MAX_TRAINING_IOU, RANDOM_SAMPLER_METHOD

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "release" / "querysets" / "best-candidate-iou020" / "v2"
TARGET = ROOT / "release" / "querysets" / "v2"
BACKUP = ROOT / "release" / "querysets" / "_training-before-best-candidate"
SIZES = (128, 256, 512)
COUNTS = {128: 35000, 256: 8800, 512: 2170}


def validate(directory: Path, size: int) -> None:
    """Raise unless the directory holds the expected best-candidate QuerySet."""
    queryset = QuerySet.read(directory)
    sampling = queryset.header["position_sampling"]
    if queryset.header["kind"] != "training" or len(queryset.positions) != COUNTS[size]:
        raise ValueError(f"{directory} does not have the expected training positions.")
    if sampling["method"] != RANDOM_SAMPLER_METHOD:
        raise ValueError(f"{directory} has the wrong sampler.")
    if sampling["maximum_pair_iou"] != MAX_TRAINING_IOU:
        raise ValueError(f"{directory} has the wrong IoU ceiling.")
    centres = {(row["centre_lon"], row["centre_lat"]) for row in queryset.positions}
    if len(centres) != len(queryset.positions):
        raise ValueError(f"{directory} has duplicate centres.")


def main() -> None:
    """Back up the current training QuerySets and promote the new ones."""
    for size in SIZES:
        validate(SOURCE / f"{size}-training", size)
    if BACKUP.exists():
        raise SystemExit(f"Backup directory already exists: {BACKUP}")
    BACKUP.mkdir()
    for size in SIZES:
        name = f"{size}-training"
        (TARGET / name).replace(BACKUP / name)
        (SOURCE / name).replace(TARGET / name)
    print(f"Promoted 0.20-IoU training QuerySets into {TARGET}")


if __name__ == "__main__":
    main()
