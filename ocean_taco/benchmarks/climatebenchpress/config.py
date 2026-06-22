"""Static benchmark configuration for OceanTACO comparison runs."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BenchmarkModalityConfig:
    """Benchmark packaging and evaluation settings for one modality.

    ``scale_factor`` / ``add_offset`` / ``fill_value`` mirror the int16 packing in
    ``generate_dataset/new_format_ssh_data.py::get_variable_encoding`` for the
    primary variable. ``strict_abs_error`` is the OceanTACO physical archival bound
    (= ``scale_factor / 2``) used as the 1x anchor error bound. These are asserted to
    stay in sync with ``get_variable_encoding`` by the benchmark tests.
    """

    modality: str
    dataset_name: str
    primary_var: str
    error_family: str
    strict_abs_error: float
    display_name: str
    scale_factor: float
    add_offset: float
    fill_value: int = -32767


BENCHMARK_MODALITY_CONFIGS: dict[str, BenchmarkModalityConfig] = {
    "l4_ssh": BenchmarkModalityConfig(
        modality="l4_ssh",
        dataset_name="oceantaco_l4_ssh_subset",
        primary_var="sla",
        error_family="ssh",
        strict_abs_error=0.00025,
        display_name="OceanTACO L4 SSH",
        scale_factor=0.0005,
        add_offset=0.0,
    ),
    "l4_sst": BenchmarkModalityConfig(
        modality="l4_sst",
        dataset_name="oceantaco_l4_sst_subset",
        primary_var="analysed_sst",
        error_family="sst",
        strict_abs_error=0.0005,
        display_name="OceanTACO L4 SST",
        scale_factor=0.001,
        add_offset=20.0,
    ),
    "l4_sss": BenchmarkModalityConfig(
        modality="l4_sss",
        dataset_name="oceantaco_l4_sss_subset",
        primary_var="sos",
        error_family="sss",
        strict_abs_error=0.001,
        display_name="OceanTACO L4 SSS",
        scale_factor=0.002,
        add_offset=30.0,
    ),
    "l4_wind": BenchmarkModalityConfig(
        modality="l4_wind",
        dataset_name="oceantaco_l4_wind_subset",
        primary_var="eastward_wind",
        error_family="wind",
        strict_abs_error=0.005,
        display_name="OceanTACO L4 Wind",
        scale_factor=0.01,
        add_offset=0.0,
    ),
    "glorys": BenchmarkModalityConfig(
        modality="glorys",
        dataset_name="oceantaco_glorys_ssh_subset",
        primary_var="zos",
        error_family="ssh",
        strict_abs_error=0.00025,
        display_name="OceanTACO GLORYS SSH",
        scale_factor=0.0005,
        add_offset=0.0,
    ),
}

VARIABLE_PACKING: dict[str, BenchmarkModalityConfig] = {
    config.primary_var: config for config in BENCHMARK_MODALITY_CONFIGS.values()
}

DATASET_NAME_TO_MODALITY = {
    config.dataset_name: config.modality
    for config in BENCHMARK_MODALITY_CONFIGS.values()
}

DEFAULT_BENCHMARK_MODALITIES = tuple(BENCHMARK_MODALITY_CONFIGS)
# Anchor-only: only the 1x bound (OceanTACO's physical archival precision) feeds the
# headline comparison table. The 2x/4x tiers fed only the dropped rate-distortion figure,
# so the sweep is removed. Kept as a one-tuple (not a scalar) so restoring the
# (1.0, 2.0, 4.0) sweep is a one-line change here plus a re-package.
DEFAULT_ERROR_MULTIPLIERS = (1.0,)
