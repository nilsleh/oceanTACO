#!/usr/bin/env python3
"""Upload the verified best-candidate v2 training QuerySets to the Hub."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi

from ocean_taco import QuerySet
from ocean_taco.sampling.grids import MAX_TRAINING_IOU, RANDOM_SAMPLER_METHOD

REPOSITORY = Path(__file__).resolve().parents[2]
QUERYSETS = REPOSITORY / "release" / "querysets" / "v2"
FILES = ("header.json", "positions.parquet", "coverage.parquet", "assets.parquet")
SIZES = (128, 256, 512)
COUNTS = {128: 35_000, 256: 8_800, 512: 2_170}


def parse_args() -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default="nilsleh/OceanTACO")
    parser.add_argument(
        "--revision",
        default="main",
        help="Hub branch or revision to update (default: %(default)s).",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def validate(directory: Path, size: int) -> QuerySet:
    """Reject anything other than the corrected 0.20-IoU training set."""
    missing = [name for name in FILES if not (directory / name).is_file()]
    if missing:
        raise FileNotFoundError(f"{directory}: missing {', '.join(missing)}")

    queryset = QuerySet.read(directory)
    sampling = queryset.header["position_sampling"]
    if queryset.header["kind"] != "training":
        raise ValueError(f"{directory}: expected a training QuerySet.")
    if len(queryset.positions) != COUNTS[size]:
        raise ValueError(
            f"{directory}: expected {COUNTS[size]} positions, "
            f"found {len(queryset.positions)}."
        )
    if sampling.get("position_count") != COUNTS[size]:
        raise ValueError(f"{directory}: header has the wrong position count.")
    if sampling.get("method") != RANDOM_SAMPLER_METHOD:
        raise ValueError(f"{directory}: header has the wrong sampler.")
    if sampling.get("maximum_pair_iou") != MAX_TRAINING_IOU:
        raise ValueError(f"{directory}: header has the wrong IoU ceiling.")

    centres = {(row["centre_lon"], row["centre_lat"]) for row in queryset.positions}
    if len(centres) != len(queryset.positions):
        raise ValueError(f"{directory}: duplicate centres are not uploadable.")
    return queryset


def operations() -> list[CommitOperationAdd]:
    """Validate every training QuerySet and build its Hub upload operations."""
    result: list[CommitOperationAdd] = []
    for size in SIZES:
        directory = QUERYSETS / f"{size}-training"
        queryset = validate(directory, size)
        print(
            f"Verified {size}-training: {len(queryset.positions)} positions; "
            f"queryset_id={queryset.queryset_id}"
        )
        for name in FILES:
            result.append(
                CommitOperationAdd(
                    path_in_repo=f"querysets/{size}-training/{name}",
                    path_or_fileobj=str(directory / name),
                )
            )
    return result


def main() -> None:
    """Upload the validated training QuerySets to the Hub."""
    args = parse_args()
    upload_operations = operations()
    print(f"Prepared {len(upload_operations)} files for {args.repo_id}@{args.revision}.")
    for operation in upload_operations:
        print(operation.path_in_repo)
    if args.dry_run:
        return

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN is required. Set it before running this script.")
    api = HfApi(token=token)
    account = api.whoami()
    print(f"Authenticated as {account['name']}.")
    commit = api.create_commit(
        repo_id=args.repo_id,
        repo_type="dataset",
        revision=args.revision,
        operations=upload_operations,
        commit_message="Publish v2 best-candidate training QuerySets (IoU <= 0.20)",
    )
    print(f"Uploaded: {commit.commit_url}")
    print(
        "Pin QuerySet.from_hub to this revision before releasing the notebook: "
        f"{commit.oid}"
    )


if __name__ == "__main__":
    main()
