"""Shared fixtures and tier-selection rules for OceanTACO tests."""

from __future__ import annotations

import os
import socket
import sys
from pathlib import Path

import numpy as np
import pytest

# `tools/` holds repository-only packages that are never shipped in the wheel,
# so they are importable only from a checkout. Putting the path here makes it
# the single mechanism for the whole suite; CI sets no PYTHONPATH of its own.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from ocean_taco import CatalogConfig  # noqa: E402
from ocean_taco.sampling import OceanMaskArtifact  # noqa: E402


def _local_port_path() -> Path | None:
    """Resolve the optional verified local Core port from the environment.

    The port is machine-specific, so there is no meaningful default: point
    OCEANTACO_LOCAL_PORT at a verified port to run the `local` tier.
    """
    value = os.environ.get("OCEANTACO_LOCAL_PORT")
    return Path(value) if value else None


def _remote_available() -> bool:
    """Return whether the remote test host is reachable without HTTP I/O."""
    try:
        with socket.create_connection(("huggingface.co", 443), timeout=2):
            return True
    except OSError:
        return False


def pytest_collection_modifyitems(config, items) -> None:
    """Skip opt-in integration tiers cleanly when their prerequisite is absent."""
    _local_port = _local_port_path()
    local_missing = _local_port is None or not _local_port.is_dir()
    remote_available = (
        _remote_available()
        if any(item.get_closest_marker("remote") for item in items)
        else False
    )
    for item in items:
        if local_missing and item.get_closest_marker("local"):
            item.add_marker(
                pytest.mark.skip(reason="OCEANTACO_LOCAL_PORT is not available")
            )
        if item.get_closest_marker("remote") and not remote_available:
            item.add_marker(pytest.mark.skip(reason="HuggingFace is unreachable"))


@pytest.fixture(autouse=True)
def reset_upsampling_warnings(monkeypatch) -> None:
    """Keep the renderer warning memo from leaking across tests."""
    from ocean_taco.render import resample

    monkeypatch.setattr(resample, "_WARNED_UPSAMPLING_TOKENS", set())


@pytest.fixture(scope="session")
def ocean_mask() -> OceanMaskArtifact:
    """Small all-ocean mask covering synthetic fixture patches."""
    return OceanMaskArtifact(
        lat=np.array([-60.0, 60.0]),
        lon=np.array([-180.0, 179.0]),
        ocean_mask=np.ones((2, 2), dtype=bool),
        manifest={},
    )


@pytest.fixture(scope="session")
def local_port() -> Path:
    """Verified one-day local Core port, or a clear tier skip."""
    port = _local_port_path()
    if port is None or not port.is_dir():
        pytest.skip("OCEANTACO_LOCAL_PORT is not available")
    return port


@pytest.fixture(scope="session")
def local_cache(tmp_path_factory) -> Path:
    """One cache shared by the optional local integration tier."""
    return tmp_path_factory.mktemp("local-cache")


@pytest.fixture(scope="session")
def local_config(local_port: Path, local_cache: Path) -> CatalogConfig:
    """Catalog configuration for the verified local Core port."""
    return CatalogConfig(taco_path=local_port, cache_dir=local_cache)


@pytest.fixture(scope="session")
def remote_cache(tmp_path_factory) -> Path:
    """One cache shared by opt-in HuggingFace integration tests."""
    return tmp_path_factory.mktemp("remote-cache")


@pytest.fixture(scope="session")
def remote_config(remote_cache: Path) -> CatalogConfig:
    """Remote-only configuration exercising the HTTP catalog branch."""
    return CatalogConfig(cache_dir=remote_cache)


@pytest.fixture
def synthetic_grid():
    """One-day antimeridian-crossing SST field for offline loader tests."""
    import xarray as xr

    return xr.Dataset(
        {
            "analysed_sst": (
                ("time", "lat", "lon"),
                np.arange(8, dtype=np.float32).reshape(1, 2, 4),
                {"units": "degC"},
            )
        },
        coords={
            "time": [np.datetime64("2024-01-02")],
            "lat": [0.0, 1.0],
            "lon": [-179.0, -170.0, 170.0, 179.0],
        },
    )


@pytest.fixture
def synthetic_argo():
    """Two profile IDs with one duplicate level for point-renderer tests."""
    import xarray as xr

    return xr.Dataset(
        {
            "TEMP": (("obs",), np.array([12.0, 13.0, 14.0], dtype=np.float32)),
            "lat": (("obs",), np.array([0.0, 0.5, 1.0], dtype=np.float32)),
            "lon": (("obs",), np.array([0.0, 0.5, 1.0], dtype=np.float32)),
            "PRES": (("obs",), np.array([5.0, 10.0, 15.0], dtype=np.float32)),
            "PLATFORM_NUMBER": (("obs",), np.array(["a", "a", "b"], dtype=str)),
            "CYCLE_NUMBER": (("obs",), np.array([1, 1, 2], dtype=np.int32)),
        },
        coords={
            "time": (
                ("obs",),
                np.array(
                    ["2024-01-02", "2024-01-02", "2024-01-03"], dtype="datetime64[ns]"
                ),
            )
        },
    )


@pytest.fixture
def synthetic_port(tmp_path):
    """Minimal local NetCDF asset tree for loader tests without Core data."""
    asset = tmp_path / "l4_sst.nc"
    synthetic_grid_data = synthetic_grid.__wrapped__()
    synthetic_grid_data.to_netcdf(asset, engine="h5netcdf")
    return tmp_path


@pytest.fixture(scope="session")
def synthetic_queryset():
    """Small valid QuerySet for selection tests without release artifacts."""
    from ocean_taco.queryset import QuerySet, content_sha256, position_id

    dates = ["2024-01-02T00:00:00.000000Z", "2024-01-03T00:00:00.000000Z"]
    grid_id = "synthetic-grid"
    tokens = ["argo", "l3_ssh", "l3_swot"]
    positions = tuple(
        {
            "position_index": index,
            "position_id": position_id(
                grid_id=grid_id, centre_lon=longitude, centre_lat=0.5
            ),
            "centre_lon": longitude,
            "centre_lat": 0.5,
            "region_mask": 1,
            "swot_footprint_cells": 4,
            "swot_ocean_cells": 3,
            "ssh_footprint_cells": 4,
            "ssh_ocean_cells": 3,
        }
        for index, longitude in enumerate((0.0, 1.0))
    )
    coverage = tuple(
        {
            "position_index": position["position_index"],
            "date_index": date_index,
            "swot_valid_cells": 1,
            "swot_valid_ocean_cells": 1,
            "swot_n_obs_sum": 1,
            "ssh_valid_cells": 1,
            "ssh_valid_ocean_cells": 1,
            "argo_profile_count": 1,
        }
        for position in positions
        for date_index in range(len(dates))
    )
    return QuerySet(
        header={
            "patch_size": {"value": 1.0, "unit": "deg"},
            "kind": "training",
            "grid_spacing_km": 1.0,
            "grid_id": grid_id,
            "dataset_revision": "synthetic",
            "catalog_sha256": "catalog",
            "registry_sha256": "registry",
            "source_records_sha256": "records",
            "ocean_mask_id": "mask",
            "ocean_mask_sha256": "mask-hash",
            "dates": dates,
            "date_sha256": content_sha256(dates),
            "tokens": tokens,
            "parquet_profile": {"writer": "pyarrow"},
            "code_commit": "test",
            "environment_lock_hash": "test",
        },
        positions=positions,
        coverage=coverage,
        assets=tuple(
            {
                "date_index": date_index,
                "region": "FIXTURE",
                "token": token,
                "asset_id": f"{token}-{date_index}",
                "uri": f"fixture://{token}/{date_index}",
                "identity_kind": "sha256",
                "identity_value": f"identity-{token}-{date_index}",
                "status": "present",
            }
            for date_index in range(len(dates))
            for token in tokens
        ),
    )


@pytest.fixture(scope="session")
def coastal_mask() -> OceanMaskArtifact:
    """Small mixed land/ocean mask for synthetic rendering assertions."""
    return OceanMaskArtifact(
        lat=np.array([0.0, 1.0]),
        lon=np.array([0.0, 1.0]),
        ocean_mask=np.array([[True, False], [True, False]], dtype=bool),
        manifest={},
    )
