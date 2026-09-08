"""Offline matrix coverage for loader output and collation semantics."""

from __future__ import annotations

import pytest
import torch

from ocean_taco.filter import QueryFilter, select_queryset
from ocean_taco.geobox import GeoBox, PatchSize, PatchSpec, TimeRange
from ocean_taco.registry import get_modality
from ocean_taco.render import Native, Points, Resample, canonicalise_dense
from ocean_taco.torch.dataset import (
    OceanTACODataset,
    _pad_points,
    _SourceRequest,
    _stack_fixed_grid,
    _stack_vector_pair,
    collate_ocean_samples,
    native_pad_collate,
)
from ocean_taco.torch.sampler import ShapeBucketSampler, native_shapes


def _grid_record(*, native_shape: tuple[int, int] | None = None) -> dict:
    record = {
        "data": torch.ones((1, 2, 2)),
        "valid_mask": torch.ones((1, 2, 2), dtype=torch.bool),
        "source_valid": torch.ones((1, 2, 2), dtype=torch.bool),
        "support_mask": torch.ones((1, 2, 2), dtype=torch.bool),
        "support": torch.ones((1, 2, 2)),
        "lat": torch.arange(2.0),
        "lon": torch.arange(2.0),
        "times": ["2024-01-02T00:00:00Z"],
    }
    if native_shape is not None:
        record["native_shape"] = native_shape
    return record


def _native_grid_record() -> dict:
    record = _grid_record()
    record.pop("support")
    return record


def _point_record(count: int) -> dict:
    values = torch.arange(count, dtype=torch.float32)
    return {
        "data": values,
        "lat": values,
        "lon": values,
        "pres": values,
        "time": ["2024-01-02T00:00:00Z"] * count,
        "profile_id": ["profile"] * count,
        "valid_mask": torch.ones(count, dtype=torch.bool),
        "source_valid": torch.ones(count, dtype=torch.bool),
        "support_mask": torch.ones(count, dtype=torch.bool),
    }


def test_synthetic_dense_and_point_sources_render(
    ocean_mask, coastal_mask, synthetic_argo, synthetic_grid, synthetic_port
):
    assert (synthetic_port / "l4_sst.nc").is_file()
    assert not coastal_mask.ocean_mask.all()
    dense = canonicalise_dense(synthetic_grid, get_modality("l4_sst"))
    footprint = GeoBox(170.0, -170.0, 0.0, 1.0, wraps_antimeridian=True)
    assert Native().render(dense, footprint)["data"].shape == (1, 2, 4)
    assert Resample((3, 5), 0.0).render(dense, footprint, ocean_mask=ocean_mask)[
        "data"
    ].shape == (1, 3, 5)
    rendered = Points().render(
        synthetic_argo,
        GeoBox(-1.0, 1.0, 0.0, 1.0),
        time=TimeRange("2024-01-02T00:00:00Z", "2024-01-03T00:00:00Z"),
    )
    assert rendered["data"].shape == (2,)
    assert rendered["profile_id"].tolist() == ["a:1", "b:2"]


def test_synthetic_queryset_is_filterable(synthetic_queryset):
    queryset = synthetic_queryset
    assert len(queryset.positions) == 2
    assert len(tuple(select_queryset(queryset, QueryFilter()).iter_pairs())) == 4
    assert queryset.header["tokens"] == ["argo", "l3_ssh", "l3_swot"]


def test_collators_preserve_native_shapes_and_padding():
    fixed = _stack_fixed_grid([_grid_record(), _grid_record(native_shape=(8, 9))])
    assert fixed["native_shapes"] == [None, (8, 9)]
    vector = _grid_record(native_shape=(3, 4))
    vector.update(components=("uo", "vo"), pair_available=True)
    paired = _stack_vector_pair([vector])
    assert paired["native_shapes"] == [(3, 4)]
    assert paired["components"] == ("uo", "vo")
    points = _pad_points([_point_record(0), _point_record(2)])
    assert points["point_mask"].tolist() == [[False, False], [True, True]]
    native = native_pad_collate(
        [
            {
                "query": "one",
                "availability": {"sst": True},
                "sst": _native_grid_record(),
            },
            {
                "query": "two",
                "availability": {"sst": True},
                "sst": _native_grid_record(),
            },
        ]
    )
    assert native["sst"]["data"].shape == (2, 1, 2, 2)
    assert not native["sst"]["spatial_padding_mask"].any()


def test_native_shapes_uses_rendered_shape_when_metadata_is_absent():
    class Samples:
        records = (
            {"sst": _native_grid_record()},
            {"sst": _grid_record(native_shape=(5, 6))},
        )

        def __len__(self):
            return len(self.records)

        def __getitem__(self, index):
            return self.records[index]

    assert native_shapes(Samples(), "sst") == [(2, 2), (5, 6)]


def test_shape_bucket_sampler_keeps_shape_homogeneous_batches():
    sampler = ShapeBucketSampler(
        [(2, 2), (3, 4), (2, 2), (3, 4), (3, 4)], batch_size=2, shuffle=False
    )
    assert list(sampler) == [[0, 2], [1, 3], [4]]
    assert len(sampler) == 3


def _target_spec(**overrides) -> PatchSpec:
    fields = {
        "centre_lon": 0.5,
        "centre_lat": 0.5,
        "patch_size": PatchSize(1.0, "deg"),
        "anchor_time": "2024-01-05T00:00:00Z",
        "context_start_offset_days": -2,
        "context_end_offset_days": 0,
    }
    fields.update(overrides)
    return PatchSpec(**fields)


def test_forecast_relation_derives_a_target_window_disjoint_from_context():
    """A forecast lead must render at the lead day, not merely be recorded."""
    spec = _target_spec(relation="forecast", target_lead_days=3)

    assert spec.target_offsets == (3, 3)
    assert spec.target.start == spec.target_time
    assert spec.target.start > spec.context.end
    assert spec.target_spec.context.start == spec.target_time


def test_explicit_target_offsets_express_a_midpoint_on_a_same_time_filter():
    """The assimilation case: a target inside a symmetric context window.

    ``same_time`` forces ``target_lead_days`` to zero, so before explicit
    offsets existed this window could not be expressed at all -- the midpoint
    was only ever right by coincidence.
    """
    spec = _target_spec(
        context_start_offset_days=-2,
        context_end_offset_days=2,
        target_start_offset_days=0,
        target_end_offset_days=0,
    )

    assert spec.relation == "same_time"
    assert spec.target_lead_days == 0
    assert spec.target_offsets == (0, 0)
    assert spec.target.start == spec.anchor_time
    assert spec.context.start < spec.target.start < spec.context.end


def test_non_zero_target_window_survives_a_same_time_relation():
    """The direct regression test for the accidental-zero midpoint."""
    spec = _target_spec(target_start_offset_days=1, target_end_offset_days=2)

    assert spec.target_offsets == (1, 2)
    target = spec.target_spec
    assert target.context_start_offset_days == 1
    assert target.context_end_offset_days == 2
    assert target.target_offsets is None


def test_same_time_without_target_offsets_renders_no_target():
    """Back-compat: today's specs must keep producing context only."""
    spec = _target_spec()

    assert spec.target_offsets is None
    assert spec.target is None
    assert spec.target_spec is None


def test_query_filter_without_target_offsets_hashes_as_before():
    """Published experiment records must keep replaying byte for byte."""
    unset = QueryFilter(context_start_offset_days=-2, context_end_offset_days=2)

    assert "target_start_offset_days" not in unset.to_dict()
    assert (
        unset.sha256
        == QueryFilter(context_start_offset_days=-2, context_end_offset_days=2).sha256
    )
    declared = QueryFilter(
        context_start_offset_days=-2,
        context_end_offset_days=2,
        target_start_offset_days=0,
        target_end_offset_days=0,
    )
    assert declared.to_dict()["target_start_offset_days"] == 0
    assert declared.sha256 != unset.sha256


def test_target_offsets_must_be_paired_and_ordered():
    with pytest.raises(ValueError, match="pair or omitted"):
        QueryFilter(target_start_offset_days=0)
    with pytest.raises(ValueError, match="ordered contiguous"):
        QueryFilter(target_start_offset_days=2, target_end_offset_days=1)
    with pytest.raises(ValueError, match="pair or omitted"):
        _target_spec(target_end_offset_days=1)


def test_selection_rejects_anchors_whose_target_window_leaves_the_domain(
    synthetic_queryset,
):
    """An explicit target window is a domain requirement like the lead is."""
    inside = select_queryset(
        synthetic_queryset,
        QueryFilter(target_start_offset_days=0, target_end_offset_days=0),
    )
    outside = select_queryset(
        synthetic_queryset,
        QueryFilter(target_start_offset_days=5, target_end_offset_days=5),
    )

    assert inside.count > 0
    assert outside.count == 0


def test_collate_stacks_targets_without_mistaking_them_for_tokens():
    spec = _target_spec(
        context_start_offset_days=-2,
        context_end_offset_days=2,
        target_start_offset_days=0,
        target_end_offset_days=0,
    )
    sample = {
        "sst": _grid_record(),
        "query": spec,
        "target": {"sst": _grid_record()},
        "target_query": spec.target_spec,
        "availability": {"sst": True, "target": {"sst": True}},
    }

    batch = collate_ocean_samples([sample, sample])

    assert set(batch) == {"sst", "query", "target", "target_query", "availability"}
    assert batch["sst"]["data"].shape[0] == 2
    assert batch["target"]["sst"]["data"].shape[0] == 2
    assert len(batch["target_query"]) == 2
    assert batch["availability"]["target"] == {"sst": [True, True]}


def test_planning_requests_include_the_target_window():
    """Targets must be pre-resolved too, or they fall off the plan in workers.

    A missing target spec here fails only under ``num_workers > 0`` with a
    planned loader, which no in-process test would catch.
    """
    spec = _target_spec(
        context_start_offset_days=-2,
        context_end_offset_days=2,
        target_start_offset_days=0,
        target_end_offset_days=0,
    )
    dataset = OceanTACODataset.__new__(OceanTACODataset)
    dataset.rows = (spec,)
    dataset.source_requests = (_SourceRequest("l4_sst", Resample((2, 2), 0.0)),)

    windows = {
        (planned.context.start, planned.context.end)
        for _, planned in dataset._planning_requests()
    }

    assert (spec.context.start, spec.context.end) in windows
    assert (spec.target.start, spec.target.end) in windows
