"""Equality tests for the release builder's fast measurement path.

``scripts/release/queryset_build`` deliberately reimplements three released
measurement routines so a full cartesian coverage table can be built without
re-opening source assets tens of millions of times.  These tests pin the
reimplementations to the released functions they replace, including the
antimeridian cases where the two could plausibly diverge.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

xr = pytest.importorskip("xarray")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "release"))

from queryset_build import (  # noqa: E402
    ArgoDay,
    build_argo_day,
    build_global_grid,
    build_position_plan,
    canonical_lon,
    check_units,
    dense_coverage_for,
    footprint_regions,
    measure_dense,
    scatter_tiles,
)

from ocean_taco.geobox import PatchSize  # noqa: E402
from ocean_taco.registry import get_modality  # noqa: E402
from ocean_taco.sampling.coverage import (  # noqa: E402
    measure_argo_profile_count,
    measure_dense_coverage,
    native_footprint_counts,
)
from ocean_taco.sampling.ocean_mask import OceanMaskArtifact  # noqa: E402

REGIONS = (
    "SOUTH_PACIFIC_WEST",
    "SOUTH_ATLANTIC",
    "SOUTH_INDIAN",
    "SOUTH_PACIFIC_EAST",
    "NORTH_PACIFIC_WEST",
    "NORTH_ATLANTIC",
    "NORTH_INDIAN",
    "NORTH_PACIFIC_EAST",
)
_LON_BOUNDS = {"WEST": (-180.0, -90.0), "ATLANTIC": (-90.0, 0.0), "INDIAN": (0.0, 90.0), "EAST": (90.0, 180.0)}
STEP = 0.5


def _tile_axes(region: str) -> tuple[np.ndarray, np.ndarray]:
    """Axes for one synthetic quadrant, mirroring the released tiling."""
    lat_lo = -60.0 if region.startswith("SOUTH") else 0.0
    key = next(name for name in _LON_BOUNDS if region.endswith(name))
    lon_lo, lon_hi = _LON_BOUNDS[key]
    lat = np.arange(lat_lo, lat_lo + 60.0, STEP)
    lon = np.arange(lon_lo, lon_hi, STEP)
    return lat, lon


@pytest.fixture(scope="module")
def ocean_mask() -> OceanMaskArtifact:
    lat = np.arange(-59.5, 60.0, 0.5)
    lon = np.arange(-180.0, 180.0, 0.5)
    grid_lon, grid_lat = np.meshgrid(lon, lat)
    # A deterministic land/ocean pattern with structure on both axes, so an
    # index or ordering error cannot coincidentally agree.
    mask = ((np.sin(np.radians(grid_lon * 3)) + np.cos(np.radians(grid_lat * 5))) > -0.4)
    return OceanMaskArtifact(lat, lon, mask, {"source_revision": "test", "source_date": "2024-01-01"})


@pytest.fixture(scope="module")
def dense_tiles() -> dict[str, xr.Dataset]:
    """Synthetic l3_ssh tiles with reproducible NaN gaps."""
    rng = np.random.default_rng(11)
    tiles = {}
    for region in REGIONS:
        lat, lon = _tile_axes(region)
        values = rng.normal(size=(1, lat.size, lon.size)).astype(np.float32)
        values[values < -0.3] = np.nan
        n_obs = rng.integers(0, 5, size=(1, lat.size, lon.size)).astype(np.float32)
        tiles[region] = xr.Dataset(
            {
                "sla_filtered": (("time", "lat", "lon"), values),
                "n_obs": (("time", "lat", "lon"), n_obs),
            },
            coords={"time": [np.datetime64("2024-06-02")], "lat": lat, "lon": lon},
        )
    return tiles


def _merged_reference(tiles: dict[str, xr.Dataset], grid) -> xr.Dataset:
    """The released path's input: one global dataset with declared units."""
    values = scatter_tiles(grid, {r: np.asarray(d["sla_filtered"].values[0], np.float32) for r, d in tiles.items()})
    counts = scatter_tiles(grid, {r: np.asarray(d["n_obs"].values[0], np.float32) for r, d in tiles.items()})
    return xr.Dataset(
        {
            "sla_filtered": (("time", "lat", "lon"), values[None], {"units": "m"}),
            "n_obs": (("time", "lat", "lon"), counts[None]),
        },
        coords={"time": [np.datetime64("2024-06-02")], "lat": grid.lat, "lon": grid.lon},
    )


# Centres deliberately include both antimeridian wrap directions, a
# four-region corner at (0, 0), and ordinary interior positions.
CENTRES = [(-40.0, 30.0), (179.6, -20.0), (-179.7, 10.0), (0.0, 0.0), (120.0, -45.0), (89.9, 25.0)]


def test_global_grid_tiles_exactly_once(dense_tiles):
    grid = build_global_grid("l3_ssh", dense_tiles)
    assert grid.shape == (240, 720)
    covered = np.zeros(grid.shape, dtype=int)
    for row, col in grid.slices.values():
        covered[row, col] += 1
    assert (covered == 1).all()


def test_global_grid_rejects_overlapping_tiles(dense_tiles):
    broken = dict(dense_tiles)
    lat, lon = _tile_axes("NORTH_ATLANTIC")
    broken["NORTH_INDIAN"] = broken["NORTH_INDIAN"].assign_coords(lon=lon)
    with pytest.raises(ValueError, match="tile the global grid exactly once"):
        build_global_grid("l3_ssh", broken)


@pytest.mark.parametrize("centre", CENTRES)
def test_dense_coverage_matches_released(dense_tiles, ocean_mask, centre):
    grid = build_global_grid("l3_ssh", dense_tiles)
    reference = _merged_reference(dense_tiles, grid)
    values = np.asarray(reference["sla_filtered"].values[0])
    patch_size = PatchSize(256, "km")
    plan = build_position_plan(grid, patch_size=patch_size, centre_lon=centre[0], centre_lat=centre[1], ocean_mask=ocean_mask)
    fast = measure_dense(plan, values, None)
    expected = measure_dense_coverage(
        reference, token="l3_ssh", patch_size=patch_size, centre_lon=centre[0], centre_lat=centre[1], ocean_mask=ocean_mask
    )
    assert (fast.valid_cells, fast.valid_ocean_cells) == (expected.valid_cells, expected.valid_ocean_cells)
    assert fast.valid_cells > 0


@pytest.mark.parametrize("centre", CENTRES)
def test_static_counts_match_released(dense_tiles, ocean_mask, centre):
    grid = build_global_grid("l3_ssh", dense_tiles)
    reference = _merged_reference(dense_tiles, grid)
    patch_size = PatchSize(256, "km")
    plan = build_position_plan(grid, patch_size=patch_size, centre_lon=centre[0], centre_lat=centre[1], ocean_mask=ocean_mask)
    expected = native_footprint_counts(
        reference, token="l3_ssh", patch_size=patch_size, centre_lon=centre[0], centre_lat=centre[1], ocean_mask=ocean_mask
    )
    assert (plan.footprint_cells, plan.ocean_cells) == expected


def test_n_obs_sum_matches_released(dense_tiles, ocean_mask):
    grid = build_global_grid("l3_ssh", dense_tiles)
    reference = _merged_reference(dense_tiles, grid)
    values = np.asarray(reference["sla_filtered"].values[0])
    counts = np.asarray(reference["n_obs"].values[0])
    patch_size = PatchSize(256, "km")
    for centre in CENTRES:
        plan = build_position_plan(grid, patch_size=patch_size, centre_lon=centre[0], centre_lat=centre[1], ocean_mask=ocean_mask)
        fast = measure_dense(plan, values, counts)
        expected = measure_dense_coverage(
            reference, token="l3_ssh", patch_size=patch_size, centre_lon=centre[0],
            centre_lat=centre[1], ocean_mask=ocean_mask, n_obs_variable="n_obs",
        )
        assert fast.n_obs_sum == expected.n_obs_sum


def _argo_tiles() -> dict[str, xr.Dataset]:
    rng = np.random.default_rng(5)
    tiles = {}
    for index, region in enumerate(REGIONS):
        lat_lo = -60.0 if region.startswith("SOUTH") else 0.0
        key = next(name for name in _LON_BOUNDS if region.endswith(name))
        lon_lo, lon_hi = _LON_BOUNDS[key]
        n = 400
        lat = rng.uniform(lat_lo, lat_lo + 59.0, n)
        lon = rng.uniform(lon_lo, lon_hi, n)
        # Repeated platform/cycle pairs exercise the uniqueness rule; a few
        # out-of-window times exercise the half-open date filter.
        platform = rng.integers(1000, 1010, n)
        cycle = rng.integers(0, 4, n)
        hours = rng.integers(-6, 30, n)
        times = np.datetime64("2024-06-02T00:00:00") + hours.astype("timedelta64[h]")
        tiles[region] = xr.Dataset(
            {"PLATFORM_NUMBER": ("N_POINTS", platform), "CYCLE_NUMBER": ("N_POINTS", cycle)},
            coords={"lat": ("N_POINTS", lat), "lon": ("N_POINTS", lon), "time": ("N_POINTS", times)},
        )
    return tiles


@pytest.mark.parametrize("centre", CENTRES)
def test_argo_count_matches_released(centre):
    from ocean_taco.retrieve import _merge_points

    tiles = _argo_tiles()
    day = build_argo_day(tiles, "2024-06-02")
    merged = _merge_points(list(tiles.values()))
    patch_size = PatchSize(512, "km")
    fast = day.count(patch_size, centre[0], centre[1])
    expected = measure_argo_profile_count(
        merged, patch_size=patch_size, centre_lon=centre[0], centre_lat=centre[1], date="2024-06-02"
    )
    assert fast == expected


def test_argo_day_excludes_out_of_window_times():
    tiles = _argo_tiles()
    day = build_argo_day(tiles, "2024-06-02")
    assert day.lat.size < sum(tile.sizes["N_POINTS"] for tile in tiles.values())


def test_incomplete_region_closure_is_null(dense_tiles, ocean_mask):
    """A footprint touching a region with no asset must be null, not partial."""
    grid = build_global_grid("l3_ssh", dense_tiles)
    reference = _merged_reference(dense_tiles, grid)
    values = np.asarray(reference["sla_filtered"].values[0])
    patch_size = PatchSize(256, "km")
    plan = build_position_plan(grid, patch_size=patch_size, centre_lon=0.0, centre_lat=0.0, ocean_mask=ocean_mask)
    assert len(plan.regions) == 4
    complete = dense_coverage_for(plan, frozenset(REGIONS), values, None)
    assert complete.valid_cells is not None
    partial = dense_coverage_for(plan, frozenset(REGIONS) - {"NORTH_INDIAN"}, values, None)
    assert (partial.valid_cells, partial.valid_ocean_cells, partial.n_obs_sum) == (None, None, None)
    # An unrelated missing region must not null an unaffected footprint.
    elsewhere = build_position_plan(grid, patch_size=patch_size, centre_lon=-40.0, centre_lat=30.0, ocean_mask=ocean_mask)
    assert dense_coverage_for(elsewhere, frozenset(REGIONS) - {"SOUTH_INDIAN"}, values, None).valid_cells is not None


def test_missing_asset_yields_null():
    from queryset_build import PositionPlan

    plan = PositionPlan((0, 1), ((0, 1),), np.ones((1, 1), bool), 1, 1, frozenset({"NORTH_ATLANTIC"}))
    coverage = dense_coverage_for(plan, frozenset(), None, None)
    assert coverage.valid_cells is None


def test_check_units_accepts_missing_and_rejects_wrong():
    spec = get_modality("l3_ssh")
    assert check_units(spec, None) == spec.canonical_unit
    assert check_units(spec, "") == spec.canonical_unit
    assert check_units(spec, "m") == "m"
    with pytest.raises(ValueError, match="unsupported decoded units"):
        check_units(spec, "degC")


def test_canonical_lon_matches_released_convention():
    values = np.array([-180.0, -0.5, 0.0, 179.5, 180.0])
    assert np.array_equal(canonical_lon(values), np.array([-180.0, -0.5, 0.0, 179.5, -180.0]))


def test_footprint_regions_detects_wrap():
    wrapped = PatchSize(512, "km").footprint(179.8, -20.0)
    assert wrapped.wraps_antimeridian
    assert footprint_regions(wrapped) == {"SOUTH_PACIFIC_EAST", "SOUTH_PACIFIC_WEST"}


def test_shard_resume_is_idempotent(tmp_path):
    """A verified shard is reused; a plan change or corruption invalidates it."""
    import build_querysets as builder

    measured = {
        "columns": {"512-training/swot_valid_cells": np.arange(4, dtype=np.int64)},
        "identities": {("NORTH_ATLANTIC", "l3_ssh"): {"uri": "/x/l3_ssh.nc", "identity_kind": "sha256", "identity_value": "ab"}},
        "present": {"l3_ssh": frozenset({"NORTH_ATLANTIC"})},
    }
    assert not builder.shard_is_valid(tmp_path, "2024-06-02", "plan-a")
    path = builder.write_shard(tmp_path, "2024-06-02", "plan-a", measured)
    assert builder.shard_is_valid(tmp_path, "2024-06-02", "plan-a")
    # A different plan must not reuse shards measured under the old one.
    assert not builder.shard_is_valid(tmp_path, "2024-06-02", "plan-b")
    path.write_bytes(path.read_bytes() + b"corrupt")
    assert not builder.shard_is_valid(tmp_path, "2024-06-02", "plan-a")


def test_region_mask_fits_uint8():
    import build_querysets as builder

    assert max(builder.REGION_BIT.values()) <= 128
    box = PatchSize(256, "km").footprint(0.0, 0.0)
    assert 0 < builder.region_mask_value(box) <= 255


def test_asset_without_primary_variable_is_unmeasurable(dense_tiles, ocean_mask):
    """A tile that exists but lacks the primary variable must yield nulls.

    Some released l3_swot assets carry only auxiliary fields.  Such a region
    is an unmeasurable closure exactly like an absent file: any footprint
    touching it is null, and unaffected footprints stay measured.
    """
    import build_querysets as builder

    grid = build_global_grid("l3_ssh", dense_tiles)
    stripped = dict(dense_tiles)
    stripped["NORTH_INDIAN"] = stripped["NORTH_INDIAN"].drop_vars("sla_filtered")
    values, _, measured = builder.dense_arrays(grid, stripped, "l3_ssh", want_n_obs=False)
    assert measured == frozenset(REGIONS) - {"NORTH_INDIAN"}
    patch_size = PatchSize(256, "km")
    touching = build_position_plan(grid, patch_size=patch_size, centre_lon=0.0, centre_lat=0.0, ocean_mask=ocean_mask)
    assert "NORTH_INDIAN" in touching.regions
    assert dense_coverage_for(touching, measured, values, None).valid_cells is None
    elsewhere = build_position_plan(grid, patch_size=patch_size, centre_lon=-40.0, centre_lat=30.0, ocean_mask=ocean_mask)
    assert dense_coverage_for(elsewhere, measured, values, None).valid_cells is not None


def test_all_tiles_without_primary_variable_returns_none(dense_tiles):
    import build_querysets as builder

    grid = build_global_grid("l3_ssh", dense_tiles)
    stripped = {r: d.drop_vars("sla_filtered") for r, d in dense_tiles.items()}
    values, n_obs, measured = builder.dense_arrays(grid, stripped, "l3_ssh", want_n_obs=False)
    assert (values, n_obs, measured) == (None, None, frozenset())
