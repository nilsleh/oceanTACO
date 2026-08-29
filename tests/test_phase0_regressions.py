"""Regression coverage for the loader-redesign Phase 0 fixes."""

import torch

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
