"""Regression coverage for the loader-redesign Phase 0 fixes."""

import torch

from ocean_taco.geobox import TimeRange

from ocean_taco.torch.dataset import _pad_points, _stack_fixed_grid
from ocean_taco.torch.sampler import ShapeBucketSampler


def _points(count: int, **extra):
    values = torch.arange(count, dtype=torch.float32)
    return {
        "data": values,
        "lat": values,
        "lon": values,
        "pres": values,
        "time": [],
        "profile_id": [],
        "valid_mask": torch.ones(count, dtype=torch.bool),
        "source_valid": torch.ones(count, dtype=torch.bool),
        "support_mask": torch.ones(count, dtype=torch.bool),
        **extra,
    }


def test_point_direction_survives_an_empty_first_record():
    batch = _pad_points([_points(0), _points(2, direction=["A", "D"])])
    assert batch["direction"] == [None, ["A", "D"]]


def test_native_shape_survives_fixed_grid_collation():
    record = {
        "data": torch.ones((1, 2, 2)),
        "valid_mask": torch.ones((1, 2, 2), dtype=torch.bool),
        "source_valid": torch.ones((1, 2, 2), dtype=torch.bool),
        "support_mask": torch.ones((1, 2, 2), dtype=torch.bool),
        "support": torch.ones((1, 2, 2)),
        "lat": torch.arange(2.0),
        "lon": torch.arange(2.0),
        "times": ["2024-01-01T00:00:00Z"],
        "native_shape": (8, 9),
    }
    assert _stack_fixed_grid([record])["native_shapes"] == [(8, 9)]


def test_shape_bucket_sampler_uses_epoch():
    sampler = ShapeBucketSampler([(4, 4)] * 8, batch_size=2, seed=2)
    first = list(sampler)
    sampler.set_epoch(1)
    assert first != list(sampler)


def _published_queryset(tmp_path, position_count=3, date_count=4):
    """Write a small published QuerySet and return its directory."""
    from ocean_taco.manifest import (
        QuerySet,
        _schemas,
        canonical_json,
        content_sha256,
        position_id,
    )

    dates = [f"2024-01-0{index + 1}T00:00:00.000000Z" for index in range(date_count)]
    tokens = ["l3_ssh"]
    positions = [
        {
            "position_index": index,
            "position_id": position_id(
                grid_id="grid", centre_lon=float(index), centre_lat=float(index)
            ),
            "centre_lon": float(index),
            "centre_lat": float(index),
            "region_mask": 1,
            "swot_footprint_cells": 2,
            "swot_ocean_cells": 2,
            "ssh_footprint_cells": 2,
            "ssh_ocean_cells": 2,
        }
        for index in range(position_count)
    ]
    coverage = [
        {
            "position_index": position,
            "date_index": date,
            "swot_valid_cells": position + date,
            "swot_valid_ocean_cells": position,
            "swot_n_obs_sum": date,
            "ssh_valid_cells": 1,
            "ssh_valid_ocean_cells": 1,
            # A null must survive the round trip as null, never as zero.
            "argo_profile_count": None if date == 0 else date,
        }
        for position in range(position_count)
        for date in range(date_count)
    ]
    assets = [
        {
            "date_index": date,
            "region": "GLOBAL",
            "token": token,
            "asset_id": "",
            "uri": "",
            "identity_kind": "",
            "identity_value": "",
            "status": "missing",
        }
        for date in range(date_count)
        for token in tokens
    ]

    directory = tmp_path / "published"
    directory.mkdir()
    schemas = _schemas()
    checksums = {}
    for name, rows in (("positions", positions), ("coverage", coverage), ("assets", assets)):
        path = directory / f"{name}.parquet"
        # Publish with the library's own writer so the fixture's bytes are the
        # bytes a real published set has.
        QuerySet._write_parquet(path, tuple(rows), schemas[name])
        checksums[name] = __import__("hashlib").sha256(path.read_bytes()).hexdigest()

    header = {
        "schema_version": "queryset/v1",
        "patch_size": {"value": 20.0, "unit": "km"},
        "kind": "training",
        "grid_spacing_km": 20.0,
        "grid_id": "grid",
        "dataset_revision": "fixture-revision",
        "catalog_sha256": "catalog",
        "registry_sha256": "registry",
        "source_records_sha256": "records",
        "ocean_mask_id": "mask",
        "ocean_mask_sha256": "mask-sha",
        "dates": dates,
        "date_sha256": content_sha256(dates),
        "tokens": tokens,
        "parquet_profile": {"compression": "zstd"},
        "code_commit": "commit",
        "environment_lock_hash": "environment",
        "table_sha256": checksums,
    }
    identity = {key: value for key, value in header.items() if key != "table_sha256"}
    header["queryset_id"] = content_sha256(
        {"header": identity, "table_sha256": checksums}
    )
    (directory / "header.json").write_bytes(canonical_json(header) + b"\n")
    return directory, coverage, header["queryset_id"]


def test_published_coverage_reads_without_materialising_rows(tmp_path):
    """Coverage is served from Arrow columns, with identical values and identity."""
    from ocean_taco.manifest import QuerySet, _ArrowRows

    directory, expected, identifier = _published_queryset(tmp_path)
    queryset = QuerySet.read(directory)

    assert isinstance(queryset.coverage, _ArrowRows)
    assert queryset.queryset_id == identifier
    assert len(queryset.coverage) == len(expected)
    assert [dict(row) for row in queryset.coverage] == expected
    # Indexing, negative indexing, and the canonical accessor agree.
    assert dict(queryset.coverage[5]) == expected[5]
    assert dict(queryset.coverage[-1]) == expected[-1]
    assert dict(queryset.coverage_row(1, 2)) == expected[1 * 4 + 2]
    # A null coverage value stays null: it is unmeasured, not measured zero.
    assert queryset.coverage[0]["argo_profile_count"] is None


def test_published_queryset_round_trips_through_the_columnar_view(tmp_path):
    """Reading and rewriting a published set preserves its content identity."""
    from ocean_taco.manifest import QuerySet

    directory, _, identifier = _published_queryset(tmp_path)
    queryset = QuerySet.read(directory)
    queryset.write(tmp_path / "republished")
    assert QuerySet.read(tmp_path / "republished").queryset_id == identifier


def test_corrupt_published_coverage_is_rejected(tmp_path):
    """The columnar validator enforces the contract the row loop enforced."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    import pytest

    from ocean_taco.manifest import QuerySet, _schemas

    directory, coverage, _ = _published_queryset(tmp_path)
    # Drop one pair, so the table is no longer the complete cartesian product.
    truncated = [row for row in coverage if not (row["position_index"] == 1 and row["date_index"] == 2)]
    path = directory / "coverage.parquet"
    pq.write_table(
        pa.Table.from_pylist(truncated, schema=_schemas()["coverage"]), path
    )
    header_path = directory / "header.json"
    import json
    from hashlib import sha256

    header = json.loads(header_path.read_text())
    header["table_sha256"]["coverage"] = sha256(path.read_bytes()).hexdigest()
    header.pop("queryset_id")
    header_path.write_text(json.dumps(header))

    with pytest.raises(ValueError, match="every published"):
        QuerySet.read(directory)


def _daily_labelled(stamp_hour: int):
    """Build a one-day dense source stamped at a given hour of that day."""
    import numpy as np
    import xarray as xr

    return xr.Dataset(
        {"zos": (("time", "lat", "lon"), np.zeros((1, 2, 2), dtype="float32"))},
        coords={
            "time": [np.datetime64(f"2025-05-15T{stamp_hour:02d}:00:00", "ns")],
            "lat": [0.0, 1.0],
            "lon": [0.0, 1.0],
        },
    )


def test_midday_stamped_daily_sources_survive_a_single_day_request():
    """GLORYS and L4 SSS label a day at 12:00; a midnight request must keep them.

    The request interval for one day is zero-width at midnight, so comparing a
    12:00 label against it as an instant drops the source and reports it as
    unavailable -- indistinguishable from data that is genuinely absent.
    """
    from datetime import datetime, timezone

    from ocean_taco.geobox import TimeRange
    from ocean_taco.retrieve import _select_time_range

    interval = TimeRange(
        start=datetime(2025, 5, 15, tzinfo=timezone.utc),
        end=datetime(2025, 5, 15, tzinfo=timezone.utc),
    )
    for token in ("glorys_ssh", "glorys_uo", "glorys_vo", "l4_sss"):
        selected = _select_time_range(_daily_labelled(12), interval, token)
        assert selected.sizes["time"] == 1, f"{token} dropped its midday label"


def test_instant_sources_still_compare_against_the_exact_timestamp():
    """The daily-label widening must not loosen selection for instant sources."""
    from datetime import datetime, timezone

    from ocean_taco.geobox import TimeRange
    from ocean_taco.retrieve import _select_time_range

    interval = TimeRange(
        start=datetime(2025, 5, 15, tzinfo=timezone.utc),
        end=datetime(2025, 5, 15, tzinfo=timezone.utc),
    )
    assert _select_time_range(_daily_labelled(12), interval, "l4_sst").sizes["time"] == 0
    assert _select_time_range(_daily_labelled(0), interval, "l4_sst").sizes["time"] == 1


def test_context_window_keeps_a_midday_label_for_a_single_day_patch():
    """The dataset's own window applies the same rule as retrieval.

    Retrieval and rendering narrow the time axis separately, so fixing only
    `_select_time_range` still leaves `VectorPair` reporting GLORYS velocity as
    structurally absent.
    """
    from datetime import datetime, timezone

    from ocean_taco.registry import get_modality
    from ocean_taco.torch.dataset import OceanTACODataset

    class _Spec:
        context = TimeRange(
            start=datetime(2025, 5, 15, tzinfo=timezone.utc),
            end=datetime(2025, 5, 15, tzinfo=timezone.utc),
        )

    window = OceanTACODataset._context_window
    assert window(_daily_labelled(12), _Spec, get_modality("glorys_uo")).sizes["time"] == 1
    assert window(_daily_labelled(0), _Spec, get_modality("l4_sst")).sizes["time"] == 1
    assert window(_daily_labelled(12), _Spec, get_modality("l4_sst")).sizes["time"] == 0
