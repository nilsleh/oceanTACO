from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from ocean_taco.generate_dataset.format_processors import (
    apply_published_metadata_compatibility,
    process_and_split,
    process_glorys_data,
)


def test_gridded_processor_preserves_singleton_time_axis(tmp_path: Path) -> None:
    source = xr.Dataset(
        {"adt": (("time", "lat", "lon"), np.ones((1, 2, 2)))},
        coords={"time": ["2023-03-29"], "lat": [1.0, 2.0], "lon": [-2.0, -1.0]},
    )

    count, _ = process_and_split(source, "20230329", tmp_path, "l4_ssh")

    assert count == 1
    result = xr.open_dataset(tmp_path / "l4_ssh" / "l4_ssh_NORTH_ATLANTIC_20230329.nc")
    assert result.sizes["time"] == 1


def test_glorys_salinity_uses_published_packing_and_keeps_time(tmp_path: Path) -> None:
    source = xr.Dataset(
        {
            "zos": (("time", "lat", "lon"), np.ones((1, 2, 2))),
            "so": (("time", "depth", "lat", "lon"), np.full((1, 1, 2, 2), 34.055)),
        },
        coords={
            "time": ["2023-03-29"],
            "depth": [0.0],
            "lat": [1.0, 2.0],
            "lon": [-2.0, -1.0],
        },
    )

    count, _ = process_glorys_data(source, "20230329", tmp_path)

    assert count == 1
    result = xr.open_dataset(
        tmp_path / "glorys" / "glorys_NORTH_ATLANTIC_20230329.nc",
        decode_times=False,
        mask_and_scale=False,
    )
    assert result.sizes["time"] == 1
    assert result["so"].attrs["scale_factor"] == 0.001
    assert result["so"].attrs["add_offset"] == 25.0


def test_sst_metadata_compatibility_matches_published_inventory() -> None:
    l3_sst = {
        "data_source": "l3_sst", "region": "NORTH_ATLANTIC",
        "bbox": [-89.95, 0.05, -0.05, 79.95],
        "_istac_spatial_wkb": b"l3-sst", "resolution_deg_lat": 0.1,
        "resolution_deg_lon": 0.1, "resolution_km_lat": 11.0574,
        "resolution_km_lon": 11.0574, "_istac_time_start": 1,
        "_istac_time_end": 2,
    }
    l4_sst = {
        **l3_sst, "data_source": "l4_sst",
        "bbox": [-89.975, 0.025, -0.025, 89.975],
        "_istac_spatial_wkb": b"l4-sst", "resolution_deg_lat": 0.05,
    }

    apply_published_metadata_compatibility(
        [l3_sst, l4_sst], pd.Timestamp("2023-03-29")
    )

    assert l4_sst["bbox"] == l3_sst["bbox"]
    assert l4_sst["_istac_spatial_wkb"] == l3_sst["_istac_spatial_wkb"]
    assert l4_sst["resolution_deg_lat"] == l3_sst["resolution_deg_lat"]
    assert l3_sst["_istac_time_start"] == l3_sst["_istac_time_end"]
    assert l4_sst["_istac_time_start"] == l4_sst["_istac_time_end"]
