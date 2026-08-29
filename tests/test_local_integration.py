"""Guarded integration coverage for the verified single-date Core port."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ocean_taco import CatalogConfig, GeoBox, PatchSize, PatchSpec
from ocean_taco.render import Resample
from ocean_taco.retrieve import load_bbox_nc, load_hf_dataset
from ocean_taco.torch import CoreSourceLoader, OceanTACODataset

ROOT = Path(__file__).resolve().parents[1]
LOCAL_PORT = (
    ROOT
    / "results/generation_audit_20260828/port_20230329_verified/taco/OceanTACO"
)


@pytest.fixture
def local_config(tmp_path) -> CatalogConfig:
    if not LOCAL_PORT.is_dir():
        pytest.skip("local verified OceanTACO port is not available")
    return CatalogConfig(taco_path=LOCAL_PORT, cache_dir=tmp_path / "cache")


@pytest.fixture
def local_catalog(local_config):
    return load_hf_dataset(local_config)


@pytest.mark.local
def test_local_dense_points_and_cross_region_retrieval(local_catalog, local_config):
    dense = load_bbox_nc(
        local_catalog,
        "2023-03-29",
        GeoBox(-60.0, -50.0, 20.0, 30.0),
        "l4_sst",
        config=local_config,
    )
    assert dense is not None
    assert dense.sizes["time"] == 1
    assert dense.sizes["lat"] > 0 and dense.sizes["lon"] > 0

    boundary = load_bbox_nc(
        local_catalog,
        "2023-03-29",
        GeoBox(-95.0, -85.0, 20.0, 30.0),
        "l4_sst",
        config=local_config,
    )
    assert boundary is not None
    longitude = np.asarray(boundary["lon"].values)
    assert (longitude < -90.0).any() and (longitude > -90.0).any()

    points = load_bbox_nc(
        local_catalog,
        "2023-03-29",
        GeoBox(-60.0, -50.0, 20.0, 30.0),
        "argo",
        config=local_config,
    )
    assert points is not None
    assert {"lat", "lon", "time"}.issubset(points.variables)


@pytest.mark.local
def test_local_core_loader_and_supervised_sample(local_config):
    spec = PatchSpec(
        centre_lon=-55.0,
        centre_lat=25.0,
        patch_size=PatchSize(2.0, "deg"),
        anchor_time="2023-03-29T00:00:00Z",
        context_start_offset_days=0,
        context_end_offset_days=1,
    )
    loader = CoreSourceLoader(local_config)
    assert loader.load("l4_sst", spec) is not None
    pair = loader.load_pair(("glorys_uo", "glorys_vo"), spec)
    assert pair is not None
    assert set(pair) == {"glorys_uo", "glorys_vo"}

    dataset = OceanTACODataset(
        queries=(spec,),
        sources={
            "l4_sst": Resample((8, 8), 0.0),
            "l3_swot": Resample((8, 8), 0.0),
        },
        catalog_config=local_config,
    )
    sample = dataset[0]
    assert sample["availability"]["l4_sst"]
    assert sample["l4_sst"]["data"].shape == (1, 8, 8)
    assert sample["l3_swot"]["data"].shape == (1, 8, 8)
