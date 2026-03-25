"""Pytest tests for OceanTACODataset — unit and remote-integration tests.

Unit tests run offline and cover the new patch-size interpolation logic.
Remote/integration tests require network access to HuggingFace and are
gated behind ``@pytest.mark.slow`` (excluded by default via pytest.ini).

Run all fast tests:
    pytest ocean_taco/test_dataset.py

Run remote integration tests:
    pytest -m slow ocean_taco/test_dataset.py
"""

from __future__ import annotations

import numpy as np
import pytest

from ocean_taco.dataset.dataset import (
    OceanTACODataset,
    _interpolate_to_patch,
    collate_ocean_samples,
)
from ocean_taco.dataset.queries import PatchSize, Query, QueryGenerator

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HF_TACO_PATH = "https://huggingface.co/datasets/nilsleh/OceanTACO/resolve/main/"

# A small Gulf Stream patch that typically has l4_ssh, l4_sst, l4_sss data
_GULF_STREAM_BBOX = (-80, -40, 10, 50)
_TEST_DATE = "2024-06-15"
_PATCH_SIZE = PatchSize(32, "deg")
_DEFAULT_IMAGE_SIZE = (128, 128)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_queries(n: int = 1, seed: int = 42) -> list[Query]:
    qg = QueryGenerator()
    raw = qg.generate_training_queries(
        n_queries=n,
        patch_size=_PATCH_SIZE,
        date_range=(_TEST_DATE, _TEST_DATE),
        bbox_constraint=_GULF_STREAM_BBOX,
        seed=seed,
        verbose=False,
    )
    return [
        Query(
            lon_min=q.lon_min,
            lon_max=q.lon_max,
            lat_min=q.lat_min,
            lat_max=q.lat_max,
            time_start=q.time_start,
            time_end=q.time_end,
        )
        for q in raw
    ]


# ===========================================================================
# 1. _interpolate_to_patch  (pure unit tests, no I/O)
# ===========================================================================

class TestInterpolateToPatch:
    def test_2d_output_shape(self):
        data = np.random.randn(64, 80).astype(np.float32)
        out = _interpolate_to_patch(data, (128, 128))
        assert out.shape == (128, 128)

    def test_3d_time_output_shape(self):
        data = np.random.randn(3, 64, 80).astype(np.float32)
        out = _interpolate_to_patch(data, (128, 128))
        assert out.shape == (3, 128, 128)

    def test_output_dtype_float32(self):
        data = np.ones((32, 32), dtype=np.float64)
        out = _interpolate_to_patch(data, (64, 64))
        assert out.dtype == np.float32

    def test_nan_preserved_in_full_nan_region(self):
        """A 2×2 block that is entirely NaN should remain NaN after interp."""
        data = np.ones((64, 64), dtype=np.float32)
        data[:32, :32] = np.nan
        out = _interpolate_to_patch(data, (64, 64))
        # Top-left quadrant should still be NaN-dominated
        assert np.isnan(out[:32, :32]).all()

    def test_valid_region_stays_finite(self):
        data = np.full((32, 32), 5.0, dtype=np.float32)
        out = _interpolate_to_patch(data, (128, 128))
        assert np.isfinite(out).all()

    def test_already_correct_size_returns_same_values(self):
        """When shape already matches target, values should be ~identical."""
        data = np.random.randn(128, 128).astype(np.float32)
        out = _interpolate_to_patch(data, (128, 128))
        assert out.shape == (128, 128)
        # bilinear of already-correct size may have minor floating-point diffs
        np.testing.assert_allclose(out, data, atol=1e-5)

    def test_asymmetric_target_size(self):
        data = np.random.randn(40, 60).astype(np.float32)
        out = _interpolate_to_patch(data, (64, 128))
        assert out.shape == (64, 128)


# ===========================================================================
# 2. OceanTACODataset init / parameter storage (offline)
# ===========================================================================

class TestDatasetInit:
    """Tests that don't touch the network — just check parameter handling."""

    def _minimal_dataset(self, **kwargs) -> OceanTACODataset:
        """Build a dataset that does nothing at init (empty query list)."""
        return OceanTACODataset(
            taco_path=HF_TACO_PATH,
            queries=[],
            input_variables=["l4_ssh"],
            target_variables=["l4_ssh"],
            **kwargs,
        )

    def test_default_patch_size_stored(self):
        ds = self._minimal_dataset()
        assert ds.default_patch_size == (128, 128)

    def test_custom_default_patch_size(self):
        ds = self._minimal_dataset(default_patch_size=(64, 64))
        assert ds.default_patch_size == (64, 64)

    def test_patch_sizes_stored(self):
        overrides = {"l4_ssh": (64, 64)}
        ds = self._minimal_dataset(patch_sizes=overrides)
        assert ds.patch_sizes == overrides

    def test_patch_sizes_default_empty(self):
        ds = self._minimal_dataset()
        assert ds.patch_sizes == {}

    def test_len_empty_queries(self):
        ds = self._minimal_dataset()
        assert len(ds) == 0

    def test_invalid_variable_raises(self):
        with pytest.raises(ValueError, match="Unknown variables"):
            OceanTACODataset(
                taco_path=HF_TACO_PATH,
                queries=[],
                input_variables=["not_a_var"],
                target_variables=[],
            )


# ===========================================================================
# 3. Remote integration tests (require network + HuggingFace)
# ===========================================================================

@pytest.mark.slow
class TestRemoteDatasetRetrieval:
    """End-to-end tests against the live HuggingFace dataset."""

    # ------------------------------------------------------------------
    # Basic retrieval
    # ------------------------------------------------------------------

    def test_dataset_len(self):
        queries = _make_queries(3)
        ds = OceanTACODataset(
            taco_path=HF_TACO_PATH,
            queries=queries,
            input_variables=["l4_ssh"],
            target_variables=["l4_ssh"],
        )
        assert len(ds) == 3

    def test_sample_keys_present(self):
        queries = _make_queries(1)
        ds = OceanTACODataset(
            taco_path=HF_TACO_PATH,
            queries=queries,
            input_variables=["l4_ssh"],
            target_variables=["l4_ssh"],
        )
        sample = ds[0]
        assert set(sample.keys()) >= {"inputs", "targets", "coords", "metadata"}

    # ------------------------------------------------------------------
    # Patch-size interpolation: default (128, 128)
    # ------------------------------------------------------------------

    def test_default_patch_size_shape(self):
        queries = _make_queries(1)
        ds = OceanTACODataset(
            taco_path=HF_TACO_PATH,
            queries=queries,
            input_variables=["l4_ssh"],
            target_variables=["l4_ssh"],
            default_patch_size=(128, 128),
        )
        sample = ds[0]
        inp = sample["inputs"]["l4_ssh"]
        assert inp is not None, "l4_ssh input is None — data missing for this date/bbox"
        assert inp["data"].shape == (128, 128), f"Expected (128, 128), got {inp['data'].shape}"

    def test_default_patch_size_coords_length(self):
        queries = _make_queries(1)
        ds = OceanTACODataset(
            taco_path=HF_TACO_PATH,
            queries=queries,
            input_variables=["l4_ssh"],
            target_variables=["l4_ssh"],
            default_patch_size=(128, 128),
        )
        sample = ds[0]
        inp = sample["inputs"]["l4_ssh"]
        if inp is None:
            pytest.skip("l4_ssh data not available for this query")
        assert len(inp["lats"]) == 128
        assert len(inp["lons"]) == 128

    def test_coords_match_bbox(self):
        """lats/lons returned in sample should span the query bbox."""
        queries = _make_queries(1)
        ds = OceanTACODataset(
            taco_path=HF_TACO_PATH,
            queries=queries,
            input_variables=["l4_ssh"],
            target_variables=["l4_ssh"],
            default_patch_size=(128, 128),
        )
        query = queries[0]
        sample = ds[0]
        inp = sample["inputs"]["l4_ssh"]
        if inp is None:
            pytest.skip("l4_ssh data not available for this query")
        lon_min, lon_max, lat_min, lat_max = query.bbox
        assert float(inp["lats"][0]) == pytest.approx(lat_min, abs=1e-3)
        assert float(inp["lats"][-1]) == pytest.approx(lat_max, abs=1e-3)
        assert float(inp["lons"][0]) == pytest.approx(lon_min, abs=1e-3)
        assert float(inp["lons"][-1]) == pytest.approx(lon_max, abs=1e-3)

    # ------------------------------------------------------------------
    # Per-variable patch_sizes override
    # ------------------------------------------------------------------

    def test_per_variable_patch_size_override(self):
        queries = _make_queries(1)
        ds = OceanTACODataset(
            taco_path=HF_TACO_PATH,
            queries=queries,
            input_variables=["l4_ssh", "l4_sst"],
            target_variables=[],
            default_patch_size=(128, 128),
            patch_sizes={"l4_ssh": (64, 64)},
        )
        sample = ds[0]
        ssh = sample["inputs"]["l4_ssh"]
        sst = sample["inputs"]["l4_sst"]
        if ssh is None:
            pytest.skip("l4_ssh data not available")
        assert ssh["data"].shape == (64, 64), f"Expected (64,64), got {ssh['data'].shape}"
        if sst is not None:
            assert sst["data"].shape == (128, 128), f"Expected (128,128), got {sst['data'].shape}"

    def test_custom_small_patch_size(self):
        queries = _make_queries(1)
        ds = OceanTACODataset(
            taco_path=HF_TACO_PATH,
            queries=queries,
            input_variables=["l4_ssh"],
            target_variables=["l4_ssh"],
            default_patch_size=(32, 32),
        )
        sample = ds[0]
        inp = sample["inputs"]["l4_ssh"]
        if inp is None:
            pytest.skip("l4_ssh data not available")
        assert inp["data"].shape == (32, 32)
        assert len(inp["lats"]) == 32
        assert len(inp["lons"]) == 32

    # ------------------------------------------------------------------
    # Argo passthrough — point sources must NOT be resized
    # ------------------------------------------------------------------

    def test_argo_not_resized(self):
        queries = _make_queries(1)
        ds = OceanTACODataset(
            taco_path=HF_TACO_PATH,
            queries=queries,
            input_variables=["l4_ssh", "argo"],
            target_variables=[],
            default_patch_size=(128, 128),
        )
        sample = ds[0]
        argo = sample["inputs"]["argo"]
        if argo is None:
            pytest.skip("No Argo profiles in this bbox/date")
        # Argo is a 1-D point source — must NOT be 128×128
        data = argo["data"]
        assert data.ndim == 1, f"Argo data should be 1-D, got shape {data.shape}"

    # ------------------------------------------------------------------
    # DataLoader / collate compatibility
    # ------------------------------------------------------------------

    def test_dataloader_batch_shape(self):
        from torch.utils.data import DataLoader

        queries = _make_queries(4)
        ds = OceanTACODataset(
            taco_path=HF_TACO_PATH,
            queries=queries,
            input_variables=["l4_ssh"],
            target_variables=["l4_ssh"],
            default_patch_size=(128, 128),
        )
        loader = DataLoader(ds, batch_size=2, shuffle=False,
                            num_workers=0, collate_fn=collate_ocean_samples)
        batch = next(iter(loader))
        ssh_batch = batch["inputs"]["l4_ssh"]
        if ssh_batch is None:
            pytest.skip("l4_ssh data not available for any query in the batch")
        # (batch, H, W) — all items now have the same shape, no padding needed
        assert ssh_batch.shape == (2, 128, 128), f"Got {ssh_batch.shape}"

    def test_multiple_vars_consistent_shape(self):
        queries = _make_queries(1)
        ds = OceanTACODataset(
            taco_path=HF_TACO_PATH,
            queries=queries,
            input_variables=["l4_ssh", "l4_sst", "l4_sss"],
            target_variables=[],
            default_patch_size=(128, 128),
        )
        sample = ds[0]
        for var in ["l4_ssh", "l4_sst", "l4_sss"]:
            v = sample["inputs"][var]
            if v is not None:
                assert v["data"].shape == (128, 128), \
                    f"{var}: expected (128, 128), got {v['data'].shape}"


# ---------------------------------------------------------------------------
# All-variable guard-rail tests
# ---------------------------------------------------------------------------

ALL_VARS = [
    "l4_ssh", "l4_sst", "l4_sss", "l4_wind",
    "l3_sst", "l3_sss_smos_asc", "l3_sss_smos_desc", "l3_ssh", "l3_swot",
    "argo", "glorys_ssh", "glorys_sst", "glorys_sss", "glorys_uo", "glorys_vo",
]

_VAR_RANGES: dict[str, tuple[float, float]] = {
    "l4_ssh":           (-2, 2),
    "l4_sst":           (-2, 35),
    "l4_sss":           (20, 45),
    "l4_wind":          (-40, 40),
    "l3_sst":           (-2, 35),
    "l3_sss_smos_asc":  (20, 45),
    "l3_sss_smos_desc": (20, 45),
    "l3_ssh":           (-2, 2),
    "l3_swot":          (-2, 2),
    "argo":             (-2, 35),
    "glorys_ssh":       (-2, 2),
    "glorys_sst":       (-2, 35),
    "glorys_sss":       (20, 45),
    "glorys_uo":        (-3, 3),
    "glorys_vo":        (-3, 3),
}


@pytest.fixture(scope="session")
def _all_vars_sample():
    queries = _make_queries(1)
    ds = OceanTACODataset(
        taco_path=HF_TACO_PATH,
        queries=queries,
        input_variables=ALL_VARS,
        target_variables=[],
        default_patch_size=_DEFAULT_IMAGE_SIZE,
    )
    return ds[0]


@pytest.mark.slow
class TestAllVariablesRetrieval:
    @pytest.mark.parametrize("var", ALL_VARS)
    def test_dtype_float32(self, var, _all_vars_sample):
        entry = _all_vars_sample["inputs"][var]
        if entry is None:
            pytest.skip(f"{var} not available for this query")
        assert entry["data"].dtype == np.float32, \
            f"{var}: expected float32, got {entry['data'].dtype}"

    @pytest.mark.parametrize("var", [v for v in ALL_VARS if v != "argo"])
    def test_gridded_shape(self, var, _all_vars_sample):
        entry = _all_vars_sample["inputs"][var]
        if entry is None:
            pytest.skip(f"{var} not available for this query")
        assert entry["data"].shape[-2:] == _DEFAULT_IMAGE_SIZE, \
            f"{var}: expected last two dims {_DEFAULT_IMAGE_SIZE}, got {entry['data'].shape}"

    @pytest.mark.parametrize("var", ALL_VARS)
    def test_value_range(self, var, _all_vars_sample):
        entry = _all_vars_sample["inputs"][var]
        if entry is None:
            pytest.skip(f"{var} not available for this query")
        data = entry["data"]
        finite = data[np.isfinite(data)]
        if finite.size == 0:
            pytest.skip(f"{var} has no finite values in this query")
        lo, hi = _VAR_RANGES[var]
        assert finite.min() >= lo, f"{var}: min {finite.min():.4f} < {lo}"
        assert finite.max() <= hi, f"{var}: max {finite.max():.4f} > {hi}"
