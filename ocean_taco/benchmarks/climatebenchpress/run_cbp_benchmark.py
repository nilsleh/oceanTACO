"""In-process driver for the real ClimateBenchPress benchmark.

This runs the genuine ClimateBenchPress pipeline (``compress`` -> ``compute_metrics``
-> ``concatenate_metrics`` -> optional ``plot_metrics``) as a library, in the same
process as the imported OceanTACO codec so it lands in ``Compressor.registry`` and is
evaluated head-to-head with the upstream WASM codecs (SZ3/ZFP/SPERR/bitround/...).

Must run in conda env ``testpy312`` (ClimateBenchPress requires Python >= 3.12). Shelling
out to ``python -m climatebenchpress...`` cannot work because the OceanTACO codec must be
registered in the *same* process as the compress loop.

The benchmark root must already contain the packaged subsets
(``datasets/<name>/standardized.zarr`` + ``datasets-error-bounds/<name>/error_bounds.json``);
build those with :mod:`ocean_taco.benchmarks.climatebenchpress.package_standardized` (or the
full :mod:`ocean_taco.benchmarks.climatebenchpress.run_pipeline`).
"""

from __future__ import annotations

import argparse
import contextlib
import json
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import xarray as xr

# The four metrics the paper table (analyze_results.py) actually consumes. "Spectral
# Error" and "DSSIM" are dropped from CBP's EVALUATION_METRICS before compute_metrics:
# they are unused by the table and error on land-NaN. (The remaining table columns --
# Satisfies-Bound and Compression-Ratio -- come from CBP's tests / compression, not from
# EVALUATION_METRICS, so they are unaffected.)
DEFAULT_METRIC_NAMES = ["MAE", "Max Absolute Error", "Max Relative Error", "PSNR"]

# Codecs that raise on ocean land-NaN and yield no metrics row, so they are excluded by
# default (overridable on the CLI). ebcc/ebcc-abs/jpeg2000 raise on NaN; sperr/tthresh
# were already excluded (sperr raises on NaN, tthresh ignores the abs bound and is
# memory-heavy). The retained ~15 codecs are the ones that actually produce data.
DEFAULT_EXCLUDED_CODECS = ["ebcc", "ebcc-abs", "jpeg2000", "sperr", "tthresh"]

# NOTE: ClimateBenchPress (and the oceantaco_codec that subclasses its Compressor ABC)
# are imported lazily inside the functions below, not at module top, so this module
# (and the package __init__ / run_pipeline that import it) remain importable in the
# OceanTACO generation env (testpy311) where ClimateBenchPress is absent. The actual
# benchmark run requires conda env testpy312.


def _register_codecs():
    """Import ClimateBenchPress + OceanTACO codec for their registration side effects.

    Returns the ``Compressor`` registry class. Activating cf-xarray's realization
    (E-axis) criteria and registering the OceanTACO codec are import side effects, so
    these imports must happen before the upstream pipeline is invoked.
    """
    import climatebenchpress.compressor.compressors  # noqa: F401  (registers WASM codecs)
    import climatebenchpress.data_loader.cf  # noqa: F401  (registers E-axis cf criteria)
    from climatebenchpress.compressor.compressors.abc import Compressor

    import ocean_taco.benchmarks.climatebenchpress.oceantaco_codec  # noqa: F401

    return Compressor


def _discover_datasets(benchmark_root: Path) -> list[str]:
    """Return packaged dataset names that have a standardized.zarr on disk."""
    datasets_dir = benchmark_root / "datasets"
    if not datasets_dir.exists():
        raise FileNotFoundError(f"No datasets directory at {datasets_dir}.")
    names = [
        path.name
        for path in sorted(datasets_dir.iterdir())
        if (path / "standardized.zarr").exists()
    ]
    if not names:
        raise FileNotFoundError(
            f"No packaged datasets (datasets/*/standardized.zarr) under {datasets_dir}."
        )
    return names


def validate_standardized_dataset(store_path: Path) -> dict:
    """Check one standardized.zarr is ClimateBenchPress-compatible.

    Confirms the primary variable carries the canonical (E, T, Z, Y, X) dimension order
    and that cf-xarray resolves the Y and X axes unambiguously (which ``compress.py``
    relies on). Returns a small summary dict. Raises on incompatibility.
    """
    ds = xr.open_dataset(store_path, engine="zarr", chunks={})
    data_vars = list(ds.data_vars)
    if len(data_vars) != 1:
        raise ValueError(f"{store_path} must hold exactly one data variable; got {data_vars}.")
    var = data_vars[0]
    dims = tuple(ds[var].dims)
    if dims != ("E", "T", "Z", "Y", "X"):
        raise ValueError(
            f"{store_path}: variable '{var}' has dims {dims}, expected ('E','T','Z','Y','X')."
        )
    try:
        y_name = ds[var].cf["Y"].name
        x_name = ds[var].cf["X"].name
    except Exception as exc:  # pragma: no cover - defensive
        raise ValueError(
            f"{store_path}: cf-xarray could not resolve Y/X axes for compress.py: {exc}"
        ) from exc
    return {"variable": var, "dims": dims, "cf_Y": y_name, "cf_X": x_name}


@contextlib.contextmanager
def _pruned_metrics(metric_names: list[str]):
    """Temporarily restrict CBP's EVALUATION_METRICS to ``metric_names``.

    Runtime-patches the module-level dict that ``compute_metrics`` reads (no CBP source
    edit) and restores it on exit, so the on-disk CBP checkout stays bit-identical. Each
    requested name must exist in the upstream dict.
    """
    from climatebenchpress.compressor.scripts import compute_metrics as cm

    original = cm.EVALUATION_METRICS
    missing = [name for name in metric_names if name not in original]
    if missing:
        raise KeyError(
            f"Unknown metric(s) {missing}; available: {sorted(original)}."
        )
    cm.EVALUATION_METRICS = {name: original[name] for name in metric_names}
    try:
        yield
    finally:
        cm.EVALUATION_METRICS = original


def _git_revision(path: Path) -> str | None:
    """Best-effort git HEAD revision for a checkout (None if unavailable)."""
    try:
        return (
            subprocess.check_output(
                ["git", "-C", str(path), "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except Exception:
        return None


def _record_provenance(
    benchmark_root: Path,
    climatebenchpress_root: Path,
    dataset_names: list[str],
    include_compressors: list[str] | None,
    exclude_compressors: list[str],
    metric_names: list[str],
    validations: dict[str, dict],
    registry_names: list[str],
) -> Path:
    """Write a provenance JSON capturing pinned revisions, env, and invocation."""
    provenance = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "registry": registry_names,
        "metric_names": metric_names,
        "climatebenchpress_root": str(climatebenchpress_root),
        "climatebenchpress_revisions": {
            "compressor": _git_revision(climatebenchpress_root / "compressor"),
            "data-loader": _git_revision(climatebenchpress_root / "data-loader"),
        },
        "invocation": {
            "benchmark_root": str(benchmark_root),
            "dataset_names": dataset_names,
            "include_compressors": include_compressors,
            "exclude_compressors": exclude_compressors,
        },
        "dataset_validations": validations,
    }
    for module in ("climatebenchpress.compressor", "numcodecs", "xarray", "zarr"):
        try:
            provenance.setdefault("package_versions", {})[module] = __import__(
                module
            ).__version__
        except Exception:
            pass
    out_path = benchmark_root / "cbp_provenance.json"
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(provenance, handle, indent=2, sort_keys=True)
    return out_path


def run_cbp_benchmark(
    benchmark_root: str | Path,
    climatebenchpress_root: str | Path,
    dataset_names: list[str] | None = None,
    include_compressors: list[str] | None = None,
    exclude_compressors: list[str] | None = None,
    metric_names: list[str] | None = None,
    overwrite: bool = False,
    make_plots: bool = False,
) -> Path:
    """Run the real ClimateBenchPress pipeline in-process over OceanTACO subsets.

    Parameters
    ----------
    benchmark_root
        Root holding ``datasets/`` and ``datasets-error-bounds/``; CBP writes
        ``compressed-datasets/`` and ``metrics/`` here.
    climatebenchpress_root
        Directory containing the pinned ``compressor`` and ``data-loader`` checkouts
        (used only to record provenance revisions).
    dataset_names
        Packaged datasets to evaluate; defaults to all discovered on disk.
    include_compressors / exclude_compressors
        Forwarded to CBP's ``compress`` / ``compute_metrics`` to filter codecs.
        ``oceantaco`` is always registered and (unless excluded) evaluated.
        ``exclude_compressors`` defaults to :data:`DEFAULT_EXCLUDED_CODECS` (the
        NaN-incompatible codecs) when not given.
    metric_names
        Evaluation metrics to keep; defaults to :data:`DEFAULT_METRIC_NAMES` (the four the
        paper table consumes). CBP's ``EVALUATION_METRICS`` is runtime-patched to this set
        for the ``compute_metrics`` call and restored afterward.
    overwrite
        Recompute metrics even if metric CSVs already exist. (CBP skips already-compressed
        datasets regardless; delete ``compressed-datasets`` to force re-compression.)
    make_plots
        Best-effort call to upstream ``plot_metrics`` (fragile for single custom codecs;
        OceanTACO's own ``analyze_results`` produces the paper artifacts).

    Returns:
    -------
    Path
        The concatenated ``metrics/all_results.csv``.
    """
    Compressor = _register_codecs()
    from climatebenchpress.compressor.scripts.compress import compress
    from climatebenchpress.compressor.scripts.compute_metrics import compute_metrics
    from climatebenchpress.compressor.scripts.concatenate_metrics import (
        concatenate_metrics,
    )

    benchmark_root = Path(benchmark_root)
    climatebenchpress_root = Path(climatebenchpress_root)
    exclude_compressors = (
        list(exclude_compressors)
        if exclude_compressors is not None
        else list(DEFAULT_EXCLUDED_CODECS)
    )
    metric_names = list(metric_names) if metric_names is not None else list(DEFAULT_METRIC_NAMES)

    names = dataset_names or _discover_datasets(benchmark_root)
    error_bounds_dir = benchmark_root / "datasets-error-bounds"
    validations: dict[str, dict] = {}
    for name in names:
        store = benchmark_root / "datasets" / name / "standardized.zarr"
        if not store.exists():
            raise FileNotFoundError(f"Missing packaged dataset: {store}")
        if not (error_bounds_dir / name / "error_bounds.json").exists():
            raise FileNotFoundError(
                f"Missing error bounds: {error_bounds_dir / name / 'error_bounds.json'}"
            )
        validations[name] = validate_standardized_dataset(store)

    if "oceantaco" not in Compressor.registry:
        raise RuntimeError("OceanTACO codec failed to register in Compressor.registry.")

    print(f"Running ClimateBenchPress on {len(names)} dataset(s): {names}")
    compress(
        basepath=benchmark_root,
        data_loader_basepath=benchmark_root,
        include_dataset=names,
        include_compressor=include_compressors,
        exclude_compressor=exclude_compressors,
        progress=False,
    )
    with _pruned_metrics(metric_names):
        compute_metrics(
            basepath=benchmark_root,
            data_loader_basepath=benchmark_root,
            include_dataset=names,
            include_compressor=include_compressors,
            exclude_compressor=exclude_compressors,
            overwrite=overwrite,
        )
    concatenate_metrics(basepath=benchmark_root)

    if make_plots:
        try:
            from climatebenchpress.compressor.plotting.plot_metrics import plot_metrics

            plot_metrics(
                basepath=benchmark_root,
                data_loader_basepath=benchmark_root,
                use_latex=False,
            )
        except Exception as exc:  # pragma: no cover - upstream plotting is fragile
            print(f"plot_metrics (best-effort) failed, continuing: {exc}")

    provenance_path = _record_provenance(
        benchmark_root,
        climatebenchpress_root,
        names,
        include_compressors,
        exclude_compressors,
        metric_names,
        validations,
        sorted(Compressor.registry.keys()),
    )
    print(f"Wrote provenance: {provenance_path}")

    all_results = benchmark_root / "metrics" / "all_results.csv"
    print(f"Concatenated metrics: {all_results}")
    return all_results


def main() -> None:
    """CLI entrypoint (run under conda env testpy312)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", required=True)
    parser.add_argument(
        "--climatebenchpress-root",
        required=True,
        help="Directory containing the pinned compressor/ and data-loader/ checkouts.",
    )
    parser.add_argument("--dataset-names", nargs="+")
    parser.add_argument("--include-compressors", nargs="+")
    parser.add_argument(
        "--exclude-compressors",
        nargs="+",
        default=None,
        help=f"Codecs to exclude (default: {DEFAULT_EXCLUDED_CODECS}).",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=None,
        help=f"Evaluation metrics to keep (default: {DEFAULT_METRIC_NAMES}).",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--make-plots", action="store_true")
    args = parser.parse_args()

    run_cbp_benchmark(
        benchmark_root=args.benchmark_root,
        climatebenchpress_root=args.climatebenchpress_root,
        dataset_names=args.dataset_names,
        include_compressors=args.include_compressors,
        exclude_compressors=args.exclude_compressors,
        metric_names=args.metrics,
        overwrite=args.overwrite,
        make_plots=args.make_plots,
    )


if __name__ == "__main__":
    main()
