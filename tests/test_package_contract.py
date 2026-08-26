"""Offline contract tests for the released OceanTACO package surface."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from ocean_taco import CatalogConfig
from ocean_taco.access import LocalCacheBackend
from ocean_taco.filter import CoverageRequirement, QueryFilter
from ocean_taco.geobox import GeoBox, PatchSize, PatchSpec, TimeRange
from ocean_taco.manifest import QuerySet, content_sha256, position_id
from ocean_taco.render import (
    Native,
    Points,
    Resample,
    VectorPair,
    canonicalise_dense,
    crop_dense,
)
from ocean_taco.sampling import (
    DenseCoverage,
    OceanMaskArtifact,
    area_share_ratios,
    build_ocean_mask,
    build_position_grid,
    build_queryset,
    maximum_pair_iou,
)
from ocean_taco.sampling.draw import draw_queryset, replay_experiment
from ocean_taco.torch import CoreSourceLoader, OceanTACODataset, collate_ocean_samples


def _grid(values=None) -> xr.Dataset:
    values = np.asarray(
        values
        if values is not None
        else [[[0.0, 1.0, np.nan, 3.0], [4.0, 5.0, 6.0, 7.0]]],
        dtype=np.float32,
    )
    return xr.Dataset(
        {"analysed_sst": (("time", "lat", "lon"), values, {"units": "degC"})},
        coords={
            "time": [np.datetime64("2024-01-02")],
            "lat": [0.0, 1.0],
            "lon": [-179.0, -170.0, 170.0, 179.0],
        },
    )


def _spec() -> PatchSpec:
    return PatchSpec(
        centre_lon=179.0,
        centre_lat=0.5,
        patch_size=PatchSize(30.0, "deg"),
        anchor_time="2024-01-02T00:00:00Z",
        context_start_offset_days=0,
        context_end_offset_days=0,
    )


def _ocean_mask() -> OceanMaskArtifact:
    return OceanMaskArtifact(
        lat=np.array([-60.0, 60.0]),
        lon=np.array([-180.0, 179.0]),
        ocean_mask=np.ones((2, 2), dtype=bool),
        manifest={},
    )


def _coastal_mask() -> OceanMaskArtifact:
    return OceanMaskArtifact(
        lat=np.array([0.0, 1.0]),
        lon=np.array([0.0, 1.0]),
        ocean_mask=np.array([[True, False], [True, False]], dtype=bool),
        manifest={},
    )


def _queryset() -> QuerySet:
    dates = ["2024-01-02T00:00:00.000000Z", "2024-01-03T00:00:00.000000Z"]
    grid = "fixture-grid"
    positions = tuple(
        {
            "position_index": index,
            "position_id": position_id(
                grid_id=grid, centre_lon=longitude, centre_lat=0.5
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
            "swot_valid_cells": date_index,
            "swot_valid_ocean_cells": date_index,
            "swot_n_obs_sum": date_index,
            "ssh_valid_cells": 2,
            "ssh_valid_ocean_cells": 2,
            "argo_profile_count": date_index,
        }
        for position in positions
        for date_index in range(len(dates))
    )
    header = {
        "patch_size": {"value": 1.0, "unit": "deg"},
        "kind": "training",
        "grid_spacing_km": 1.0,
        "grid_id": grid,
        "dataset_revision": "fixture-revision",
        "catalog_sha256": "catalog",
        "registry_sha256": "registry",
        "source_records_sha256": "records",
        "ocean_mask_id": "mask",
        "ocean_mask_sha256": "mask-hash",
        "dates": dates,
        "date_sha256": content_sha256(dates),
        "tokens": ["argo", "l3_ssh", "l3_swot"],
        "parquet_profile": {"writer": "pyarrow"},
        "code_commit": "commit",
        "environment_lock_hash": "environment",
    }
    assets = tuple(
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
        for token in header["tokens"]
    )
    return QuerySet(
        header=header, positions=positions, coverage=coverage, assets=assets
    )


def test_geobox_requires_declared_wrap_and_preserves_full_extent():
    with pytest.raises(ValueError, match="wraps_antimeridian"):
        GeoBox(170.0, -170.0, -1.0, 1.0)
    box = GeoBox(170.0, -170.0, -1.0, 1.0, wraps_antimeridian=True)
    assert box.longitude_width_degrees == 20.0
    assert [segment.wraps_antimeridian for segment in box.segments()] == [False, False]
    assert PatchSize(20.0, "deg").footprint(179.0, 0.0).longitude_width_degrees == 20.0


def test_native_crop_is_exact_and_wrap_axis_is_query_ordered():
    from ocean_taco.registry import get_modality

    dense = canonicalise_dense(_grid(), get_modality("l4_sst"))
    box = GeoBox(170.0, -170.0, 0.0, 1.0, wraps_antimeridian=True)
    cropped = crop_dense(dense, box)
    np.testing.assert_array_equal(
        cropped["lon"].values, np.array([170.0, 179.0, 181.0, 190.0])
    )
    rendered = Native().render(dense, box)
    np.testing.assert_array_equal(rendered["data"], cropped.values)
    assert np.isnan(rendered["data"][0, 0, 0])


def test_mask_weighted_resample_keeps_zero_and_never_invents_nan_hole():
    from ocean_taco.registry import get_modality

    dense = canonicalise_dense(_grid(), get_modality("l4_sst"))
    output = Resample(shape=(4, 8), support_threshold=0.75).render(
        dense, GeoBox(-179.0, 179.0, 0.0, 1.0)
    )
    assert output["data"].shape == (1, 4, 8)
    assert output["support"].shape == output["data"].shape
    assert np.isnan(output["data"])[~output["valid_mask"]].all()
    assert np.nanmin(output["data"]) == pytest.approx(0.0)
    with pytest.raises(TypeError):
        Resample((4, 8))  # type: ignore[call-arg]


def test_resample_preserves_finite_source_values_over_static_land_and_empty_coordinates_are_nan():
    from ocean_taco.registry import get_modality

    dense = canonicalise_dense(
        _grid(np.full((1, 2, 4), 7.0, dtype=np.float32)), get_modality("l4_sst")
    )
    output = Resample((2, 4), 0.0).render(
        dense, GeoBox(-179.0, 179.0, 0.0, 1.0), ocean_mask=_coastal_mask()
    )
    assert np.isfinite(output["data"]).all()
    assert output["source_valid"].all()
    assert not output["ocean_mask"].all()
    assert not output["valid_mask"][:, ~output["ocean_mask"]].any()
    empty = Resample((4, 4), 0.5).empty()
    assert np.isnan(empty["lat"]).all()
    assert np.isnan(empty["lon"]).all()


def test_resample_coordinates_follow_the_bilinear_sampling_grid():
    values = np.arange(100, dtype=np.float32).reshape(1, 10, 10)
    data = xr.DataArray(
        values,
        dims=("time", "lat", "lon"),
        coords={
            "time": [np.datetime64("2024-01-02")],
            "lat": np.arange(10.0),
            "lon": np.arange(10.0),
        },
    )
    output = Resample((5, 5), 0.0, method="bilinear").render(
        data, GeoBox(-1.0, 10.0, -1.0, 10.0)
    )
    np.testing.assert_allclose(output["lat"], [0.5, 2.5, 4.5, 6.5, 8.5])
    np.testing.assert_allclose(output["lon"], [0.5, 2.5, 4.5, 6.5, 8.5])
    oracle = data.interp(lat=output["lat"], lon=output["lon"], method="linear")
    np.testing.assert_allclose(output["data"], oracle.values)


def test_ocean_mask_is_binary_and_round_trips(tmp_path):
    mask = xr.Dataset(
        {
            "mask": (
                ("lat", "lon"),
                np.array([[1.0, 6.0], [9.0, -32767.0]], dtype=np.float32),
                {
                    "flag_masks": [1, 2, 4, 8],
                    "flag_meanings": "sea land lake ice",
                    "_FillValue": -32767.0,
                },
            )
        },
        coords={"lat": [-0.05, 0.05], "lon": [-0.05, 0.05]},
    )
    artifact = build_ocean_mask(
        mask, source_asset_ids=["a"], source_revision="revision"
    )
    assert artifact.ocean_mask.dtype == bool
    np.testing.assert_array_equal(
        artifact.ocean_mask, np.array([[True, False], [True, False]])
    )
    artifact.write(tmp_path / "mask.npz")
    assert artifact.read(tmp_path / "mask.npz").sha256 == artifact.sha256
    assert artifact.ocean_mask.sum() == 2


def test_queryset_is_factored_checked_and_draw_replays(tmp_path):
    queries = _queryset()
    directory = queries.write(tmp_path / "published")
    published = QuerySet.read(directory)
    assert published.sha256 == queries.sha256
    first = published.patch_row(0, 1)
    assert (
        first["patch_id"]
        == published.patch_row(
            0, 1, context_start_offset_days=-1, context_end_offset_days=0
        )["patch_id"]
    )

    record_path = tmp_path / "experiment.json"
    draw = draw_queryset(
        published,
        requested_row_count=1,
        seed=7,
        record_path=record_path,
        query_filter=QueryFilter(
            coverage=(CoverageRequirement("ssh", "valid_cells", 2.0),)
        ),
    )
    assert draw.inclusion_probability == pytest.approx(1 / 4)
    assert replay_experiment(published, record_path).rows == draw.rows
    (directory / "coverage.parquet").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        QuerySet.read(directory)


def test_constant_physical_grid_is_unweighted_trimmed_and_overlap_bounded():
    mask = OceanMaskArtifact(
        lat=np.arange(-1.5, 1.6, 0.1),
        lon=np.arange(-1.5, 1.6, 0.1),
        ocean_mask=np.ones((31, 31), dtype=bool),
        manifest={},
    )
    patch = PatchSize(20.0, "km")
    positions = build_position_grid(
        mask,
        patch_size=patch,
        spacing_km=20.0 * 2.0 / 3.0,
        static_counts=lambda lon, lat: {
            "swot_footprint_cells": 2,
            "swot_ocean_cells": 2,
            "ssh_footprint_cells": 2,
            "ssh_ocean_cells": 2,
        },
    )
    assert positions
    assert all(
        mask.nearest(
            np.array([row["centre_lat"]]), np.array([row["centre_lon"]])
        ).item()
        for row in positions
    )
    assert maximum_pair_iou(positions, patch) <= 0.2 + 1e-12
    assert area_share_ratios({"low": 10, "high": 20}, {"low": 1.0, "high": 2.0}) == {
        "low": 1.0,
        "high": 1.0,
    }


def test_published_builder_keeps_every_position_date_without_qualification():
    mask = OceanMaskArtifact(
        lat=np.arange(-1.5, 1.6, 0.1),
        lon=np.arange(-1.5, 1.6, 0.1),
        ocean_mask=np.ones((31, 31), dtype=bool),
        manifest={},
    )
    dates = ("2024-01-01", "2024-01-02")
    tokens = ("argo", "l3_ssh", "l3_swot")
    assets = tuple(
        {
            "date_index": date_index,
            "region": "FIXTURE",
            "token": token,
            "asset_id": "",
            "uri": "",
            "identity_kind": "",
            "identity_value": "",
            "status": "missing",
        }
        for date_index in range(len(dates))
        for token in tokens
    )
    queryset = build_queryset(
        ocean_mask=mask,
        patch_size=PatchSize(20.0, "km"),
        kind="training",
        dates=dates,
        tokens=tokens,
        provenance={
            "dataset_revision": "fixture-revision",
            "catalog_sha256": "catalog",
            "registry_sha256": "registry",
            "source_records_sha256": "records",
            "code_commit": "commit",
            "environment_lock_hash": "environment",
        },
        assets=assets,
        static_counts=lambda lon, lat: {
            "swot_footprint_cells": 2,
            "swot_ocean_cells": 2,
            "ssh_footprint_cells": 2,
            "ssh_ocean_cells": 2,
        },
        measure_coverage=lambda position, date_index, date: {
            "swot": DenseCoverage(0, 0, 0),
            "ssh": DenseCoverage(0, 0),
            "argo": 0,
        },
    )
    assert len(queryset.coverage) == len(queryset.positions) * len(dates)
    assert {row["argo_profile_count"] for row in queryset.coverage} == {0}


def test_dataset_flat_schema_and_opt_in_native_padding():
    spec = _spec()
    source = _grid()
    dataset = OceanTACODataset(
        queries=(spec,),
        patch=spec.patch_size,
        sources={"l4_sst": Resample((2, 3), 0.1)},
        source_loader=lambda token, patch: source,
        ocean_mask=_ocean_mask(),
    )
    sample = dataset[0]
    assert set(sample) == {"l4_sst", "query", "availability"}
    assert sample["l4_sst"]["data"].shape == (1, 2, 3)
    assert sample["availability"] == {"l4_sst": True}
    batch = collate_ocean_samples([sample, sample])
    assert batch["l4_sst"]["data"].shape == (2, 1, 2, 3)

    native_dataset = OceanTACODataset(
        queries=(spec,),
        patch=spec.patch_size,
        sources={"l4_sst": Native()},
        source_loader=lambda token, patch: source,
        ocean_mask=_ocean_mask(),
    )
    native_batch = collate_ocean_samples(
        [native_dataset[0], native_dataset[0]], native="padded"
    )
    assert not native_batch["l4_sst"]["spatial_padding_mask"].any()


def test_argo_points_preserve_actual_surface_pressure_and_allow_empty():
    argo = xr.Dataset(
        {
            "TEMP": ("point", [3.0, 2.0, 5.0]),
            "PSAL": ("point", [35.0, 35.1, 34.0]),
            "lat": ("point", [0.0, 0.0, 30.0]),
            "lon": ("point", [0.0, 0.0, 30.0]),
            "time": (
                "point",
                np.array(
                    ["2024-01-01", "2024-01-01", "2024-01-01"], dtype="datetime64[ns]"
                ),
            ),
            "PRES": ("point", [10.0, 2.0, 1.0]),
            "PLATFORM_NUMBER": ("point", ["a", "a", "b"]),
            "CYCLE_NUMBER": ("point", [1, 1, 1]),
        }
    )
    output = Points().render(
        argo, GeoBox(-1, 1, -1, 1), time=TimeRange("2024-01-01", "2024-01-02")
    )
    np.testing.assert_array_equal(output["pres"], [2.0])
    assert output["profile_id"].tolist() == ["a:1"]
    assert (
        Points()
        .render(argo, GeoBox(-1, 1, -1, 1), time=TimeRange("2025-01-01", "2025-01-02"))[
            "data"
        ]
        .size
        == 0
    )


def test_points_report_static_geography_without_discarding_observations():
    argo = xr.Dataset(
        {
            "TEMP": ("point", [2.0, 3.0]),
            "lat": ("point", [0.0, 61.0]),
            "lon": ("point", [1.0, 0.0]),
            "time": (
                "point",
                np.array(["2024-01-01", "2024-01-01"], dtype="datetime64[ns]"),
            ),
            "PRES": ("point", [1.0, 1.0]),
            "PLATFORM_NUMBER": ("point", ["a", "b"]),
            "CYCLE_NUMBER": ("point", [1, 1]),
        }
    )
    mask = OceanMaskArtifact(
        lat=np.array([-60.0, 60.0]),
        lon=np.array([-1.0, 1.0]),
        ocean_mask=np.array([[True, False], [True, False]]),
        manifest={},
    )
    output = Points().render(
        argo,
        GeoBox(-1, 1, -1, 70),
        time=TimeRange("2024-01-01", "2024-01-02"),
        ocean_mask=mask,
    )
    assert output["data"].size == 2
    assert output["valid_mask"].all()
    assert output["ocean_mask"].tolist() == [False, False]
    assert output["in_mask_domain"].tolist() == [True, False]


def test_dataset_rejects_a_patch_that_leaves_the_mask_domain():
    spec = PatchSpec(
        centre_lon=0.0,
        centre_lat=60.0,
        patch_size=PatchSize(2.0, "deg"),
        anchor_time="2024-01-02",
        context_start_offset_days=0,
        context_end_offset_days=0,
    )
    dataset = OceanTACODataset(
        queries=(spec,),
        sources={"l4_sst": Native()},
        source_loader=lambda token, patch: _grid(),
        ocean_mask=_ocean_mask(),
    )
    with pytest.raises(ValueError, match="leaves the configured ocean-mask domain"):
        dataset[0]


def test_vector_pair_uses_joint_support_and_one_pair_availability():
    from ocean_taco.registry import get_modality

    raw = xr.Dataset(
        {
            "uo": (
                ("time", "lat", "lon"),
                [[[1.0, 2.0], [3.0, 4.0]]],
                {"units": "m s-1"},
            ),
            "vo": (
                ("time", "lat", "lon"),
                [[[1.0, np.nan], [3.0, 4.0]]],
                {"units": "m s-1"},
            ),
        },
        coords={
            "time": [np.datetime64("2024-01-02")],
            "lat": [0.0, 1.0],
            "lon": [0.0, 1.0],
        },
    )
    pair = VectorPair(Resample((2, 2), 0.0, method="nearest"))
    output = pair.render(
        canonicalise_dense(raw, get_modality("glorys_uo")),
        canonicalise_dense(raw, get_modality("glorys_vo")),
        GeoBox(0.0, 1.0, 0.0, 1.0),
    )
    assert output["data"].shape == (1, 2, 2, 2)
    assert not output["valid_mask"][0, 0, 1]
    assert output["pair_available"]


def test_vector_pair_batches_as_one_group_and_core_loader_is_shipped(tmp_path):
    raw = xr.Dataset(
        {
            "uo": (
                ("time", "lat", "lon"),
                [[[1.0, 2.0], [3.0, 4.0]]],
                {"units": "m s-1"},
            ),
            "vo": (
                ("time", "lat", "lon"),
                [[[1.0, 2.0], [3.0, 4.0]]],
                {"units": "m s-1"},
            ),
        },
        coords={
            "time": [np.datetime64("2024-01-02")],
            "lat": [0.0, 1.0],
            "lon": [0.0, 1.0],
        },
    )
    spec = PatchSpec(
        centre_lon=0.5,
        centre_lat=0.5,
        patch_size=PatchSize(1.0, "deg"),
        anchor_time="2024-01-02",
        context_start_offset_days=0,
        context_end_offset_days=0,
    )
    dataset = OceanTACODataset(
        queries=(spec,),
        sources={"velocity": VectorPair(Resample((2, 2), 0.0, method="nearest"))},
        source_loader=lambda token, patch: raw,
        ocean_mask=_ocean_mask(),
    )
    batch = collate_ocean_samples([dataset[0], dataset[0]])
    assert batch["velocity"]["data"].shape == (2, 1, 2, 2, 2)
    assert batch["velocity"]["pair_available"].all()
    assert isinstance(
        CoreSourceLoader(CatalogConfig(cache_dir=tmp_path / "cache")), CoreSourceLoader
    )


def test_grid_crop_preserves_categorical_dtype():
    from ocean_taco.retrieve import _crop

    data = xr.Dataset(
        {"mask": (("lat", "lon"), np.ones((3, 3), dtype=np.int8))},
        coords={"lat": [0.0, 1.0, 2.0], "lon": [0.0, 1.0, 2.0]},
    )
    assert _crop(data, GeoBox(0.0, 1.0, 0.0, 1.0))["mask"].dtype == np.dtype("int8")


def test_local_cache_is_revision_qualified_atomic_and_reuses_a_worker_handle(tmp_path):
    asset = _grid()
    source = tmp_path / "source.nc"
    asset.to_netcdf(source, engine="h5netcdf")
    payload = source.read_bytes()
    cache = LocalCacheBackend(
        tmp_path / "cache", revision="pinned-revision", max_open_files=1
    )

    first = cache.open_or_fetch(
        "2024-01-02", "NORTH_ATLANTIC", "l4_sst.nc", lambda: payload
    )
    second = cache.open_or_fetch(
        "2024-01-02",
        "NORTH_ATLANTIC",
        "l4_sst.nc",
        lambda: pytest.fail("a valid immutable cache entry must not be fetched twice"),
    )

    assert first is second
    assert cache.path_for("2024-01-02", "NORTH_ATLANTIC", "l4_sst.nc").is_file()
    assert cache.path_for("2024-01-02", "NORTH_ATLANTIC", "l4_sst.nc").parts[-4:] == (
        "pinned-revision",
        "2024-01-02",
        "NORTH_ATLANTIC",
        "l4_sst.nc",
    )
    with pytest.raises(ValueError, match="one concrete"):
        cache.path_for("../escape", "NORTH_ATLANTIC", "l4_sst.nc")
    cache.close()
