"""Regression tests for the parent-planned, worker-safe Core loader."""

from __future__ import annotations

import multiprocessing
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import xarray as xr
from torch.utils.data import DataLoader

from ocean_taco import CatalogConfig, PatchSize, PatchSpec
from ocean_taco.render import Resample
from ocean_taco.sampling import OceanMaskArtifact
from ocean_taco.torch import OceanTACODataset, seed_ocean_taco_worker
from ocean_taco.torch import loader as loader_module

ROOT = Path(__file__).resolve().parents[1]
LOCAL_PORT = (
    ROOT
    / "results/generation_audit_20260828/port_20230329_verified/taco/OceanTACO"
)


def _mask() -> OceanMaskArtifact:
    return OceanMaskArtifact(
        lat=np.array([-60.0, 60.0]),
        lon=np.array([-180.0, 179.0]),
        ocean_mask=np.ones((2, 2), dtype=bool),
        manifest={},
    )


def _spec() -> PatchSpec:
    return PatchSpec(
        centre_lon=0.5,
        centre_lat=0.5,
        patch_size=PatchSize(1.0, "deg"),
        anchor_time="2024-01-02T00:00:00Z",
        context_start_offset_days=0,
        context_end_offset_days=0,
    )


def _asset() -> xr.Dataset:
    return xr.Dataset(
        {
            "analysed_sst": (
                ("time", "lat", "lon"),
                np.arange(9, dtype=np.float32).reshape(1, 3, 3),
                {"units": "degC"},
            )
        },
        coords={
            "time": [np.datetime64("2024-01-02T00:00:00")],
            "lat": [0.0, 0.5, 1.0],
            "lon": [0.0, 0.5, 1.0],
        },
    )


class _DateFrame:
    def __init__(self, frame: pd.DataFrame) -> None:
        self.frame = frame

    def flatten(self) -> pd.DataFrame:
        return self.frame


class _Catalog:
    def __init__(self, frame: pd.DataFrame) -> None:
        self.frame = frame

    def filter_datetime(self, _interval: str) -> _DateFrame:
        return _DateFrame(self.frame)


def test_planned_loader_constructs_the_catalog_once_and_workers_match_parent(
    tmp_path, monkeypatch
):
    """Workers must receive paths, not an inherited/rebuilt Core catalog."""
    path = tmp_path / "l4_sst.nc"
    _asset().to_netcdf(path, engine="h5netcdf")
    catalog = _Catalog(
        pd.DataFrame(
            {
                "l1:id": ["NORTH_ATLANTIC"],
                "l2:id": ["l4_sst.nc"],
                "gdal_vsi": [str(path)],
            }
        )
    )
    constructions = multiprocessing.Value("i", 0)

    def load_catalog_once(_config):
        with constructions.get_lock():
            constructions.value += 1
        return catalog

    monkeypatch.setattr(loader_module, "load_hf_dataset", load_catalog_once)
    spec = _spec()
    dataset = OceanTACODataset(
        queries=(spec, spec),
        sources={"l4_sst": Resample((2, 2), 0.0, method="nearest")},
        catalog_config=CatalogConfig(taco_path=tmp_path, cache_dir=tmp_path / "cache"),
        ocean_mask=_mask(),
    )
    assert constructions.value == 1

    parent = [dataset[index] for index in range(len(dataset))]
    worker = list(
        DataLoader(
            dataset,
            batch_size=None,
            num_workers=2,
            worker_init_fn=seed_ocean_taco_worker,
        )
    )

    assert constructions.value == 1
    for expected, actual in zip(parent, worker, strict=True):
        torch.testing.assert_close(
            actual["l4_sst"]["data"], expected["l4_sst"]["data"], equal_nan=True
        )
        torch.testing.assert_close(
            actual["l4_sst"]["valid_mask"], expected["l4_sst"]["valid_mask"]
        )
        assert actual["availability"] == expected["availability"]


@pytest.mark.local
def test_warm_parent_worker_loading_does_not_crash(tmp_path):
    """Keep the crash-prone ordering in a subprocess so a SIGSEGV is reportable."""
    if not LOCAL_PORT.is_dir():
        pytest.skip("local verified OceanTACO port is not available")
    script = """
from pathlib import Path
from torch.utils.data import DataLoader
from ocean_taco import CatalogConfig, PatchSize, PatchSpec
from ocean_taco.render import Resample
from ocean_taco.torch import OceanTACODataset, seed_ocean_taco_worker

port = Path(r'''%s''')
cache = Path(r'''%s''')
spec = PatchSpec(
    centre_lon=-55.0, centre_lat=25.0, patch_size=PatchSize(2.0, 'deg'),
    anchor_time='2023-03-29T00:00:00Z', context_start_offset_days=0,
    context_end_offset_days=1,
)
dataset = OceanTACODataset(
    queries=(spec, spec), sources={'l4_sst': Resample((8, 8), 0.0)},
    catalog_config=CatalogConfig(taco_path=port, cache_dir=cache),
)
assert dataset[0]['availability']['l4_sst']
for sample in DataLoader(dataset, batch_size=None, num_workers=2,
                         worker_init_fn=seed_ocean_taco_worker):
    assert sample['availability']['l4_sst']
print('worker loading completed')
""" % (LOCAL_PORT, tmp_path / "cache")
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT)
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "worker loading completed" in result.stdout
