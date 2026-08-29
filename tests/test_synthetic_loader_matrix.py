"""Offline matrix coverage for loader output and collation semantics."""

from __future__ import annotations

import torch

from ocean_taco.filter import QueryFilter, select_queryset
from ocean_taco.geobox import GeoBox, TimeRange
from ocean_taco.registry import get_modality
from ocean_taco.render import Native, Points, Resample, canonicalise_dense
from ocean_taco.torch.dataset import (
    _pad_points,
    _stack_fixed_grid,
    _stack_vector_pair,
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
