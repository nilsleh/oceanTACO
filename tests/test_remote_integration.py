"""Opt-in HuggingFace integration coverage for the worker-safe loader path."""

from __future__ import annotations

import pytest

from ocean_taco import PatchSize, PatchSpec
from ocean_taco.torch import CoreSourceLoader

pytestmark = pytest.mark.remote


def _spec() -> PatchSpec:
    return PatchSpec(
        centre_lon=-55.0,
        centre_lat=25.0,
        patch_size=PatchSize(2.0, "deg"),
        anchor_time="2023-03-29T00:00:00Z",
        context_start_offset_days=0,
        context_end_offset_days=0,
    )


def test_remote_catalog_planning_returns_http_assets(remote_config):
    """Planning must retain remote URLs rather than local port paths."""
    planned = CoreSourceLoader(remote_config).plan((("l4_sst", _spec()),))
    assert planned.plan
    assert all(
        asset.location.startswith("https://")
        for assets in planned.plan.values()
        for asset in assets
    )
