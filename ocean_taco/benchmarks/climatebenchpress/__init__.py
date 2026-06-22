"""Prepare OceanTACO subsets and run the real ClimateBenchPress benchmark.

This package prepares OceanTACO regular-grid subsets (export pre-encoding float32 tiles,
package ``standardized.zarr`` + physical ``error_bounds.json``) and drives the genuine
ClimateBenchPress framework in-process, with OceanTACO's shipped scaled-int16 + zlib
encoding registered as the ``oceantaco`` ``Compressor`` so it is evaluated head-to-head
with the upstream WASM codecs (SZ3/ZFP/SPERR/bitround/...).

The ClimateBenchPress run requires conda env ``testpy312`` (Python >= 3.12).
"""

from ocean_taco.benchmarks.climatebenchpress.config import (
    BENCHMARK_MODALITY_CONFIGS,
    BenchmarkModalityConfig,
)
from ocean_taco.benchmarks.climatebenchpress.export_subset import (
    BenchmarkTileExporter,
    collect_export_records,
)
from ocean_taco.benchmarks.climatebenchpress.package_standardized import (
    package_all_datasets,
    package_dataset,
)
from ocean_taco.benchmarks.climatebenchpress.run_cbp_benchmark import run_cbp_benchmark
from ocean_taco.benchmarks.climatebenchpress.run_pipeline import run_benchmark_pipeline

__all__ = [
    "BENCHMARK_MODALITY_CONFIGS",
    "BenchmarkModalityConfig",
    "BenchmarkTileExporter",
    "collect_export_records",
    "package_all_datasets",
    "package_dataset",
    "run_benchmark_pipeline",
    "run_cbp_benchmark",
]
