"""Scientific equivalence and bounded reuse for the throughput optimizations."""

import multiprocessing
import pickle
from dataclasses import replace

import numpy as np
import pytest
import torch
import xarray as xr
from torch.utils.data import DataLoader

from ocean_taco import (
    CatalogConfig,
    CoverageRequirement,
    GeoBox,
    PatchSize,
    PatchSpec,
    QueryFilter,
)
from ocean_taco.access import LocalCacheBackend
from ocean_taco.filter import SelectedPairs, select_queryset
from ocean_taco.render import Native, Resample, VectorPair
from ocean_taco.retrieve import (
    ResolvedAsset,
    _crop,
    _merge_grid_tiles,
    load_planned_multisource_time_series_nc,
)
from ocean_taco.temporal import _cluster_indices
from ocean_taco.torch import (
    OceanTACODataset,
    collate_ocean_samples,
    seed_ocean_taco_worker,
)
from ocean_taco.torch.loader import PlannedSourceLoader


def grid(lon, lat=(0.0, 1.0), variable="analysed_sst", dtype=np.float32):
    return xr.Dataset(
        {
            variable: (
                ("time", "lat", "lon"),
                np.ones((1, len(lat), len(lon)), dtype=dtype),
                {"units": "degC"},
            )
        },
        coords={
            "time": [np.datetime64("2024-01-02")],
            "lat": lat if isinstance(lat, np.ndarray) else list(lat),
            "lon": list(lon),
        },
    )


@pytest.mark.parametrize(
    "box",
    [
        GeoBox(-0.1, 0.6, 0, 1),
        GeoBox(170, -170, 0, 1, wraps_antimeridian=True),
        GeoBox(80, 81, 20, 21),
    ],
)
@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int16])
def test_crop_first_matches_full_merge_with_jitter_and_wrap(box, dtype):
    tiles = [
        grid([-179, -1, 0], dtype=dtype),
        grid([0.00002, 1, 179], lat=(0.00002, 1.00002), dtype=dtype),
    ]
    expected = _crop(_merge_grid_tiles(tiles, coordinate_tolerance=1e-4), box)
    actual = _merge_grid_tiles(tiles, coordinate_tolerance=1e-4, box=box)
    xr.testing.assert_identical(actual, expected)
    assert actual.analysed_sst.dtype == expected.analysed_sst.dtype


def test_clustering_precedes_boundary_selection_and_conflicts_are_local():
    first, second = grid([-1, 0]), grid([0.00002, 1])
    second.analysed_sst.values[:, :, 0] = 2
    with pytest.raises(ValueError, match="disagree"):
        _merge_grid_tiles(
            [first, second], coordinate_tolerance=1e-4, box=GeoBox(-0.1, 0.1, 0, 1)
        )
    # The snapped seam is 0.00001, just outside this closed footprint.
    actual = _merge_grid_tiles(
        [first, second], coordinate_tolerance=1e-4, box=GeoBox(-1, 0, 0, 1)
    )
    np.testing.assert_array_equal(actual.lon, [-1])
    np.testing.assert_array_equal(actual.analysed_sst, 1)


def test_axis_matching_matches_quadratic_reference():
    axis = np.array([-2.0, 0.0, 2.0, 5.0])
    values = np.array([-2.1, 1.0, 2.1, 5.1])
    expected = np.abs(axis[:, None] - values[None, :]).argmin(axis=0)
    np.testing.assert_array_equal(_cluster_indices(axis, values, 1.1), expected)
    with pytest.raises(ValueError, match="collapses"):
        _cluster_indices(axis, np.array([0.0, 0.1]), 1.0)


def test_cache_eviction_serialization_and_coordinate_views(tmp_path, monkeypatch):
    paths = [tmp_path / f"{index}.nc" for index in range(3)]
    for path in paths:
        grid([0, 1]).to_netcdf(path, engine="h5netcdf")
    backend = LocalCacheBackend(max_open_files=2)
    first = backend.open_path(paths[0])
    normalized = backend.canonical_grid(first)
    assert backend.canonical_grid(first) is normalized
    backend.open_path(paths[1])
    assert backend.open_path(paths[0]) is first
    backend.open_path(paths[2])
    assert list(backend._handles) == [paths[0], paths[2]]
    assert backend.file_opens == 3 and backend.cache_hits == 1
    restored = pickle.loads(pickle.dumps(backend))
    assert restored._handles == {} and restored._views == {}
    assert restored.max_open_files == 2 and restored.file_opens == 0
    backend.open_path(paths[1])
    assert id(first) not in backend._views
    backend.close()
    assert not backend._handles and not backend._views


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_invalid_handle_limit(value):
    with pytest.raises(ValueError, match="max_open_files"):
        CatalogConfig(max_open_files=value)
    with pytest.raises(ValueError, match="max_open_files"):
        LocalCacheBackend(max_open_files=value)


@pytest.mark.parametrize(
    "coverage", [(), (CoverageRequirement("ssh", "valid_ocean_cells", 1),)]
)
def test_bulk_ranks_preserve_order_duplicates_and_use_one_scan(
    synthetic_queryset, monkeypatch, coverage
):
    selected = select_queryset(synthetic_queryset, QueryFilter(coverage=coverage))
    ranks = (selected.count - 1, 0, 1, 0)
    expected = tuple(selected.resolve_rank(rank) for rank in ranks)
    scans = []
    original = SelectedPairs.iter_pairs

    def scan(self):
        scans.append(1)
        yield from original(self)

    monkeypatch.setattr(SelectedPairs, "iter_pairs", scan)
    assert selected.resolve_ranks(ranks) == expected
    assert len(scans) == (0 if selected.is_cartesian else 1)
    assert selected.resolve_ranks(()) == ()
    with pytest.raises(IndexError):
        selected.resolve_ranks((selected.count,))


def assert_sample_equal(actual, expected):
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            assert_sample_equal(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected, strict=True):
            assert_sample_equal(left, right)
    elif isinstance(expected, torch.Tensor):
        assert actual.dtype == expected.dtype
        torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
    else:
        assert actual == expected


def planned_dataset(tmp_path, ocean_mask, renderer=None):
    spec = PatchSpec(
        0.5,
        0.5,
        PatchSize(1, "deg"),
        "2024-01-02",
        0,
        1,
        target_start_offset_days=1,
        target_end_offset_days=2,
    )
    plan = {}
    for index, day in enumerate(("2024-01-02", "2024-01-03", "2024-01-04")):
        path = tmp_path / f"{day}.nc"
        raw = grid([0, 0.5, 1], lat=(0, 0.5, 1)).assign_coords(
            time=[np.datetime64(day)]
        )
        raw.analysed_sst.values *= index + 1
        raw.analysed_sst.values[0, 0, 0] = np.nan
        raw["unused"] = xr.full_like(raw.analysed_sst, 99)
        raw.to_netcdf(path, engine="h5netcdf")
        plan[("l4_sst", day, spec.footprint)] = (
            ResolvedAsset(str(path), "NORTH_INDIAN"),
        )
    loader = PlannedSourceLoader(CatalogConfig(max_open_files=1), plan)
    return OceanTACODataset(
        queries=[spec, replace(spec, context_end_offset_days=0)],
        sources={"l4_sst": renderer or Resample((2, 2), 0.5)},
        source_loader=loader,
        ocean_mask=ocean_mask,
    )


@pytest.mark.parametrize(
    "renderer", [Native(), Resample((2, 2), 0.5), Resample((4, 4), 1.0)]
)
def test_batch_reuses_context_target_days_and_preserves_every_field(
    tmp_path, ocean_mask, renderer
):
    dataset = planned_dataset(tmp_path, ocean_mask, renderer)
    loader = dataset.source_loader
    indices = [1, 0, 1]
    expected = [dataset[index] for index in indices]
    loader.close()
    actual = dataset.__getitems__(indices)
    assert_sample_equal(actual, expected)
    assert (
        loader._backend.file_opens == 3
    )  # three days, including the shared target day
    assert loader._daily_cache is None and loader._batch_variables is None
    raw = loader.load("l4_sst", dataset.rows[0])
    assert list(raw.data_vars) == ["analysed_sst"]
    scientific = load_planned_multisource_time_series_nc(
        loader.plan,
        ["l4_sst"],
        dataset.rows[0].footprint,
        dataset.rows[0].context,
        config=loader.config,
        backend=loader._backend,
    )
    assert "unused" in scientific["l4_sst"]
    assert_sample_equal(collate_ocean_samples(actual), collate_ocean_samples(expected))
    loader.close()


@pytest.mark.parametrize("renderer", [Native(), Resample((4, 4), 0.5)])
def test_shared_glorys_variables_are_read_together(tmp_path, ocean_mask, renderer):
    spec = PatchSpec(0.5, 0.5, PatchSize(1, "deg"), "2024-01-02", 0, 0)
    raw = grid([0, 1], variable="uo")
    raw.uo.attrs["units"] = "m s-1"
    raw["vo"] = raw.uo * 2
    raw.vo.attrs["units"] = "m s-1"
    raw["thetao"] = raw.uo.copy()
    raw.thetao.attrs["units"] = "degC"
    raw.vo.values[0, 0, 0] = np.nan
    path = tmp_path / "glorys.nc"
    raw.to_netcdf(path, engine="h5netcdf")
    plan = {
        (token, "2024-01-02", spec.footprint): (
            ResolvedAsset(str(path), "NORTH_INDIAN"),
        )
        for token in ("glorys_uo", "glorys_vo", "glorys_sst")
    }
    loader = PlannedSourceLoader(CatalogConfig(), plan)
    dataset = OceanTACODataset(
        queries=[spec],
        sources={"currents": VectorPair(renderer), "glorys_sst": Native()},
        source_loader=loader,
        ocean_mask=ocean_mask,
    )
    expected = dataset[0]
    loader.close()
    actual = dataset.__getitems__([0])[0]
    assert_sample_equal(actual, expected)
    assert loader._backend.file_opens == 1 and loader._backend.cache_hits == 0
    loader.close()


def test_custom_loader_keeps_order_and_repeated_calls(ocean_mask):
    calls = []
    spec = PatchSpec(0.5, 0.5, PatchSize(1, "deg"), "2024-01-02", 0, 0)

    def custom(token, patch):
        calls.append((token, patch))
        return None

    dataset = OceanTACODataset(
        queries=[spec],
        sources={"l4_sst": Native()},
        source_loader=custom,
        ocean_mask=ocean_mask,
    )
    assert dataset.__getitems__([]) == []
    assert len(dataset.__getitems__([0, 0])) == 2
    assert calls == [("l4_sst", spec)] * 2


@pytest.mark.workers
@pytest.mark.parametrize(
    "start_method",
    [
        method
        for method in ("fork", "spawn")
        if method in multiprocessing.get_all_start_methods()
    ],
)
def test_warm_parent_batch_matches_persistent_workers(
    tmp_path, ocean_mask, start_method
):
    dataset = planned_dataset(tmp_path, ocean_mask)
    expected = collate_ocean_samples(dataset.__getitems__([0, 1]))
    loader = DataLoader(
        dataset,
        batch_size=2,
        num_workers=2,
        multiprocessing_context=start_method,
        persistent_workers=True,
        worker_init_fn=seed_ocean_taco_worker,
        collate_fn=collate_ocean_samples,
    )
    try:
        for _ in range(2):
            assert_sample_equal(next(iter(loader)), expected)
    finally:
        if loader._iterator is not None:
            loader._iterator._shutdown_workers()
        dataset.source_loader.close()


def test_file_reads_are_cropped_and_unused_variables_are_not_loaded(
    tmp_path, monkeypatch
):
    from xarray.backends.h5netcdf_ import H5NetCDFArrayWrapper

    spec = PatchSpec(-0.5, 0.5, PatchSize(1, "deg"), "2024-01-02", 0, 0)
    assets = []
    for index, lon in enumerate((np.linspace(-20, 0, 201), np.linspace(0, 20, 201))):
        path = tmp_path / f"tile-{index}.nc"
        raw = grid(lon, lat=np.linspace(-10, 10, 201))
        raw["unused"] = raw.analysed_sst + 10
        raw.to_netcdf(path, engine="h5netcdf")
        assets.append(ResolvedAsset(str(path), str(index)))
    reads = []
    original = H5NetCDFArrayWrapper._getitem

    def read(self, key):
        value = original(self, key)
        if self.variable_name in {"analysed_sst", "unused"}:
            reads.append((self.variable_name, value.shape))
        return value

    monkeypatch.setattr(H5NetCDFArrayWrapper, "_getitem", read)
    loader = PlannedSourceLoader(
        CatalogConfig(max_open_files=1),
        {("l4_sst", "2024-01-02", spec.footprint): tuple(assets)},
    )
    result = loader.load("l4_sst", spec)
    assert result.sizes["lat"] <= 11 and result.sizes["lon"] <= 11
    assert reads and all(
        name == "analysed_sst" and np.prod(shape) <= 121 for name, shape in reads
    )
    loader.close()


def test_empty_on_disk_crop_and_failed_batch_release_state(tmp_path, ocean_mask):
    dataset = planned_dataset(tmp_path, ocean_mask)
    loader = dataset.source_loader
    spec = replace(dataset.rows[0], centre_lat=20.0)
    for token, day, box in tuple(loader.plan):
        loader.plan[(token, day, spec.footprint)] = loader.plan[(token, day, box)]
    empty = loader.load("l4_sst", spec)
    assert empty.sizes["lat"] == 0
    with pytest.raises(RuntimeError, match="render failed"):
        with loader.batch([("l4_sst", dataset.rows[0])]):
            raise RuntimeError("render failed")
    assert loader._daily_cache is None and loader._batch_variables is None
    assert loader.load("l4_sst", dataset.rows[0]) is not None
    loader.close()



@pytest.mark.parametrize("failure", ["unknown_source", "missing_asset"])
def test_failed_batch_setup_allows_retry(tmp_path, ocean_mask, failure):
    dataset = planned_dataset(tmp_path, ocean_mask)
    loader = dataset.source_loader
    original_plan = loader.plan.copy()
    token = "unknown_source" if failure == "unknown_source" else "l4_sst"
    if failure == "missing_asset":
        loader.plan = {
            key: (ResolvedAsset(str(tmp_path / "missing.nc"), "tile"),)
            for key in loader.plan
        }
    try:
        error = ValueError if failure == "unknown_source" else FileNotFoundError
        with pytest.raises(error):
            with loader.batch([(token, dataset.rows[0])]):
                pytest.fail("Batch setup should fail before yielding")
        assert loader._daily_cache is None and loader._batch_variables is None
        loader.plan = original_plan
        actual = dataset.__getitems__([0, 1])
        assert_sample_equal(actual, [dataset[0], dataset[1]])
    finally:
        loader.close()


def test_single_day_preserves_static_scientific_variables(tmp_path):
    spec = PatchSpec(0.5, 0.5, PatchSize(1, "deg"), "2024-01-02", 0, 0)
    raw = grid([0, 1])
    raw["static"] = (("lat", "lon"), np.ones((2, 2), dtype=np.int16))
    path = tmp_path / "static.nc"
    raw.to_netcdf(path, engine="h5netcdf")
    backend = LocalCacheBackend()
    plan = {
        ("l4_sst", "2024-01-02", spec.footprint): (ResolvedAsset(str(path), "tile"),)
    }
    actual = load_planned_multisource_time_series_nc(
        plan,
        ["l4_sst"],
        spec.footprint,
        spec.context,
        config=CatalogConfig(),
        backend=backend,
    )["l4_sst"]
    assert actual.static.dims == ("time", "lat", "lon")
    backend.close()


def test_bulk_draw_and_replay_preserve_record_digests(
    synthetic_queryset, tmp_path, monkeypatch
):
    from ocean_taco.queryset import content_sha256
    from ocean_taco.sampling import draw_queryset, replay_experiment
    from ocean_taco.sampling.draw import _floyd_ordinals

    query_filter = QueryFilter(
        coverage=(CoverageRequirement("ssh", "valid_ocean_cells", 1),)
    )
    selection = select_queryset(synthetic_queryset, query_filter)
    ranks = _floyd_ordinals(selection.count, 3, 42)
    expected_ids = [
        synthetic_queryset.patch_row(*selection.resolve_rank(rank))["patch_id"]
        for rank in ranks
    ]

    def no_individual_scan(*_args):
        pytest.fail("draw/replay must use bulk rank resolution")

    monkeypatch.setattr(SelectedPairs, "resolve_rank", no_individual_scan)
    draw = draw_queryset(
        synthetic_queryset,
        requested_row_count=3,
        seed=42,
        record_path=tmp_path / "draw.json",
        query_filter=query_filter,
    )
    assert draw.record["selected_ranks_sha256"] == content_sha256(ranks)
    assert draw.record["emitted_patch_id_digest"] == content_sha256(expected_ids)
    assert replay_experiment(synthetic_queryset, draw.record).rows == draw.rows


def test_argo_batch_retains_fields_for_nondefault_variable(
    tmp_path, synthetic_argo, ocean_mask
):
    from ocean_taco.render import Points

    raw = synthetic_argo.copy()
    raw["PSAL"] = raw.TEMP + 20
    raw["DIRECTION"] = (("obs",), ["A", "A", "D"])
    path = tmp_path / "argo.nc"
    raw.to_netcdf(path, engine="h5netcdf")
    spec = PatchSpec(0.5, 0.5, PatchSize(1, "deg"), "2024-01-02", 0, 1)
    plan = {("argo", "2024-01-02", spec.footprint): (ResolvedAsset(str(path), "tile"),)}
    loader = PlannedSourceLoader(CatalogConfig(), plan)
    dataset = OceanTACODataset(
        queries=[spec],
        sources={"argo": Points(variable="PSAL", pres_range=(0, 100))},
        source_loader=loader,
        ocean_mask=ocean_mask,
    )
    expected = dataset[0]
    loader.close()
    assert_sample_equal(dataset.__getitems__([0, 0]), [expected, expected])
    assert loader._backend.file_opens == 1
    loader.close()


@pytest.mark.parametrize("day", ["2023-03-29", "2023-07-15", "2023-08-05"])
def test_swot_phase_dates_keep_sparse_and_missing_availability(
    tmp_path, ocean_mask, day
):
    spec = PatchSpec(0.5, 0.5, PatchSize(1, "deg"), day, 0, 0)
    plan = {}
    if day != "2023-07-15":
        raw = grid([0, 1], variable="ssha_filtered").assign_coords(
            time=[np.datetime64(day)]
        )
        raw.ssha_filtered.attrs["units"] = "m"
        raw.ssha_filtered.values[:] = np.nan
        raw.ssha_filtered.values[0, 0, 0] = 1
        raw["unused_point_field"] = (("obs",), [1.0, 2.0])
        path = tmp_path / "swot.nc"
        raw.to_netcdf(path, engine="h5netcdf")
        plan[("l3_swot", day, spec.footprint)] = (ResolvedAsset(str(path), "tile"),)
    loader = PlannedSourceLoader(CatalogConfig(), plan)
    dataset = OceanTACODataset(
        queries=[spec],
        sources={"l3_swot": Resample((1, 1), 0.25)},
        source_loader=loader,
        ocean_mask=ocean_mask,
    )
    expected = dataset[0]
    assert_sample_equal(dataset.__getitems__([0])[0], expected)
    assert expected["availability"]["l3_swot"] == (day != "2023-07-15")
    loader.close()


@pytest.mark.parametrize("shape", [(1, 1), (2, 2), (4, 4)])
def test_vector_source_valid_uses_joint_output_support(shape, ocean_mask):
    spec = PatchSpec(0.5, 0.5, PatchSize(1, "deg"), "2024-01-02", 0, 0)
    first = grid([0, 1], variable="uo").uo
    second = first.copy(deep=True)
    first.values[0, 0, 0] = np.nan
    second.values[0, 1, 1] = np.nan
    renderer = Resample(shape, 0.5)
    output = renderer.render_pair(first, second, spec.footprint, ocean_mask=ocean_mask)
    assert output["source_valid"].shape == (1, *shape)
    np.testing.assert_array_equal(output["source_valid"], output["support"] > 0)
    np.testing.assert_array_equal(
        output["valid_mask"], output["source_valid"] & output["support_mask"]
    )
    # Mixed available/missing records must pad every mask on the same grid.
    from ocean_taco.torch.dataset import _stack_vector_pair, _to_tensors

    pair = VectorPair(renderer)
    record = pair.render(first, second, spec.footprint, ocean_mask=ocean_mask)
    batch = _stack_vector_pair([_to_tensors(record), _to_tensors(pair.empty())])
    assert batch["source_valid"].shape == batch["valid_mask"].shape == (2, 1, *shape)
    assert not batch["source_valid"][1].any()
