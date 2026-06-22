"""Parallel big-box orchestrator for the real ClimateBenchPress benchmark.

Replaces the serial ``run_full.sh`` loop. The numcodecs-wasm codecs leak memory per
invocation (grow-only WASM linear memory), so running all codecs in one process balloons
RSS to OOM. This driver keeps the per-``(dataset, codec)`` process isolation -- one fresh
subprocess per pair, each handling a single codec, then exiting and releasing the leaked
memory -- but runs ``N`` such subprocesses concurrently. CBP partitions outputs by
``dataset / error_bound / compressor``, so concurrent jobs write disjoint paths and the
grid is fully resumable (a job whose metrics already exist is skipped).

Each job shells out to ``run_cbp_benchmark`` under ``conda run -n testpy312`` so the driver
itself never imports ClimateBenchPress (it can be imported and unit-tested anywhere). After
all jobs join, a single cheap concat + provenance pass and then ``analyze_results`` run as
their own subprocesses.

Sizing ``N`` (``-j``): pick it from the per-codec peak RSS so that ``N x peak_RSS <
box_RAM``. The per-job summary written at the end (elapsed + peak RSS KB) feeds this for
the next run. Start conservative and raise.
"""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from ocean_taco.benchmarks.climatebenchpress.run_cbp_benchmark import (
    DEFAULT_EXCLUDED_CODECS,
    DEFAULT_METRIC_NAMES,
)

# The full codec panel CBP can register (mirrors run_full.sh's CODECS). The retained set
# drops the NaN-incompatible codecs that yield no metrics row.
ALL_CODECS = [
    "bitround",
    "bitround-pco",
    "ebcc",
    "ebcc-abs",
    "jpeg2000",
    "oceantaco",
    "safeguarded-bitround-pco",
    "safeguarded-ebcc",
    "safeguarded-sperr",
    "safeguarded-sz3",
    "safeguarded-zero",
    "safeguarded-zero-dssim",
    "safeguarded-zfp-round",
    "sperr",
    "stochround",
    "stochround-pco",
    "sz3",
    "sz3-abs",
    "tthresh",
    "zfp",
    "zfp-round",
]
RETAINED_CODECS = [c for c in ALL_CODECS if c not in DEFAULT_EXCLUDED_CODECS]

DEFAULT_CONDA_ENV = "testpy312"
# Conservative default; size up from the per-job peak RSS in the run summary.
DEFAULT_MAX_WORKERS = 4


@dataclass(frozen=True)
class Job:
    """One isolated ``(dataset, codec)`` unit of work."""

    dataset: str
    codec: str


@dataclass
class JobResult:
    """Outcome of a single job (or a skipped one)."""

    job: Job
    returncode: int
    elapsed_s: float
    peak_rss_kb: int | None
    log_path: Path
    skipped: bool = False

    @property
    def ok(self) -> bool:
        """True if the job's subprocess exited cleanly."""
        return self.returncode == 0


def discover_datasets(benchmark_root: Path) -> list[str]:
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


def metrics_exist(benchmark_root: Path, dataset: str, codec: str) -> bool:
    """True if CBP metrics for this ``(dataset, codec)`` already exist (resume skip).

    CBP writes ``metrics/<dataset>/<var>-abs_error=<bound>/<codec>/metrics.csv`` and may
    append a ``-conservative-abs`` / ``-conservative-rel`` error-bound-conversion suffix to
    the codec directory. Under anchor-only there is a single bound tier, so any matching
    ``metrics.csv`` means the job is done.
    """
    dataset_dir = benchmark_root / "metrics" / dataset
    if not dataset_dir.exists():
        return False
    for bound_dir in dataset_dir.iterdir():
        if not bound_dir.is_dir():
            continue
        for codec_dir in bound_dir.iterdir():
            name = codec_dir.name
            if name == codec or name.startswith(f"{codec}-conservative"):
                if (codec_dir / "metrics.csv").exists():
                    return True
    return False


def enumerate_jobs(
    benchmark_root: Path,
    dataset_names: list[str],
    codecs: list[str],
    *,
    overwrite: bool = False,
) -> tuple[list[Job], list[Job]]:
    """Split the ``dataset x codec`` grid into (jobs to run, jobs skipped on resume)."""
    pending: list[Job] = []
    skipped: list[Job] = []
    for dataset in dataset_names:
        for codec in codecs:
            job = Job(dataset=dataset, codec=codec)
            if not overwrite and metrics_exist(benchmark_root, dataset, codec):
                skipped.append(job)
            else:
                pending.append(job)
    return pending, skipped


def build_job_command(
    job: Job,
    benchmark_root: Path,
    climatebenchpress_root: Path,
    metric_names: list[str],
    *,
    conda_env: str,
    rss_path: Path | None,
    overwrite: bool,
) -> list[str]:
    """Build the ``conda run ... /usr/bin/time ... run_cbp_benchmark`` argv for one job."""
    inner = [
        "python",
        "-u",
        "-m",
        "ocean_taco.benchmarks.climatebenchpress.run_cbp_benchmark",
        "--benchmark-root",
        str(benchmark_root),
        "--climatebenchpress-root",
        str(climatebenchpress_root),
        "--dataset-names",
        job.dataset,
        "--include-compressors",
        job.codec,
        "--metrics",
        *metric_names,
    ]
    if overwrite:
        inner.append("--overwrite")
    # /usr/bin/time writes peak RSS (KB) to rss_path; absent on some systems -> skip it.
    time_bin = shutil.which("time") or "/usr/bin/time"
    if rss_path is not None and Path(time_bin).exists():
        inner = [time_bin, "-f", "%M", "-o", str(rss_path), *inner]
    cmd = ["conda", "run", "-n", conda_env, "--no-capture-output", *inner]
    return cmd


def run_job(
    job: Job,
    benchmark_root: Path,
    climatebenchpress_root: Path,
    metric_names: list[str],
    *,
    conda_env: str,
    logs_dir: Path,
    overwrite: bool,
) -> JobResult:
    """Run one isolated subprocess, capturing its log, elapsed time, and peak RSS."""
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / f"{job.dataset}__{job.codec}.log"
    rss_path = logs_dir / f"{job.dataset}__{job.codec}.rss"
    rss_path.unlink(missing_ok=True)
    cmd = build_job_command(
        job,
        benchmark_root,
        climatebenchpress_root,
        metric_names,
        conda_env=conda_env,
        rss_path=rss_path,
        overwrite=overwrite,
    )
    start = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"$ {' '.join(cmd)}\n\n")
        log.flush()
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
    elapsed = time.monotonic() - start

    peak_rss_kb: int | None = None
    if rss_path.exists():
        try:
            peak_rss_kb = int(rss_path.read_text().strip().splitlines()[-1])
        except (ValueError, IndexError):
            peak_rss_kb = None
        rss_path.unlink(missing_ok=True)
    return JobResult(
        job=job,
        returncode=proc.returncode,
        elapsed_s=elapsed,
        peak_rss_kb=peak_rss_kb,
        log_path=log_path,
    )


def _run_subprocess(cmd: list[str], log_path: Path) -> int:
    """Run a final-pass subprocess, streaming to both a log file and stdout."""
    print(f"$ {' '.join(cmd)}")
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        print(log_path.read_text(encoding="utf-8"))
    return proc.returncode


def write_run_summary(results: list[JobResult], summary_path: Path) -> Path:
    """Write per-job elapsed + peak RSS so the next run can size ``-j``."""
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["dataset", "codec", "returncode", "skipped", "elapsed_s", "peak_rss_kb"]
        )
        for r in sorted(results, key=lambda r: (r.job.dataset, r.job.codec)):
            writer.writerow(
                [
                    r.job.dataset,
                    r.job.codec,
                    r.returncode,
                    int(r.skipped),
                    f"{r.elapsed_s:.1f}",
                    "" if r.peak_rss_kb is None else r.peak_rss_kb,
                ]
            )
    return summary_path


def run_parallel(
    benchmark_root: str | Path,
    climatebenchpress_root: str | Path,
    dataset_names: list[str] | None = None,
    codecs: list[str] | None = None,
    metric_names: list[str] | None = None,
    *,
    max_workers: int = DEFAULT_MAX_WORKERS,
    conda_env: str = DEFAULT_CONDA_ENV,
    overwrite: bool = False,
    run_analysis: bool = True,
    job_runner=run_job,
) -> Path:
    """Run the ``dataset x codec`` grid concurrently, then concat + analyze.

    Parameters
    ----------
    benchmark_root
        Root holding ``datasets/`` and ``datasets-error-bounds/``; CBP writes
        ``compressed-datasets/`` and ``metrics/`` here. Logs go to ``logs/``.
    climatebenchpress_root
        Directory containing the pinned ``compressor`` / ``data-loader`` checkouts.
    dataset_names
        Packaged datasets to evaluate; defaults to all discovered on disk.
    codecs
        Codecs to evaluate; defaults to :data:`RETAINED_CODECS`.
    metric_names
        Evaluation metrics to keep; defaults to ``DEFAULT_METRIC_NAMES``.
    max_workers
        Number of concurrent isolated subprocesses (``-j``). Size from the per-job peak
        RSS so ``max_workers x peak_RSS < box_RAM``.
    overwrite
        Re-run jobs (and recompute metrics) even if outputs already exist.
    run_analysis
        After concat + provenance, run ``analyze_results`` to build the paper artifacts.
    job_runner
        The per-job callable (injectable for testing); defaults to :func:`run_job`.

    Returns:
    -------
    Path
        The concatenated ``metrics/all_results.csv``.
    """
    benchmark_root = Path(benchmark_root)
    climatebenchpress_root = Path(climatebenchpress_root)
    metric_names = list(metric_names) if metric_names is not None else list(DEFAULT_METRIC_NAMES)
    codecs = list(codecs) if codecs is not None else list(RETAINED_CODECS)
    names = dataset_names or discover_datasets(benchmark_root)
    logs_dir = benchmark_root / "logs"

    pending, skipped = enumerate_jobs(
        benchmark_root, names, codecs, overwrite=overwrite
    )
    total = len(pending) + len(skipped)
    print(
        f"{total} job(s): {len(pending)} to run, {len(skipped)} already done "
        f"({len(names)} dataset(s) x {len(codecs)} codec(s), -j {max_workers})."
    )

    results: list[JobResult] = [
        JobResult(job, returncode=0, elapsed_s=0.0, peak_rss_kb=None,
                  log_path=logs_dir / f"{job.dataset}__{job.codec}.log", skipped=True)
        for job in skipped
    ]
    done = 0
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(
                job_runner,
                job,
                benchmark_root,
                climatebenchpress_root,
                metric_names,
                conda_env=conda_env,
                logs_dir=logs_dir,
                overwrite=overwrite,
            ): job
            for job in pending
        }
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            done += 1
            status = "ok" if result.ok else f"FAILED rc={result.returncode}"
            rss = "" if result.peak_rss_kb is None else f"  PEAK {result.peak_rss_kb} KB"
            print(
                f"[{done}/{len(pending)}] {result.job.dataset} :: {result.job.codec} "
                f"-> {status}  {result.elapsed_s:.1f} s{rss}"
            )

    summary_path = write_run_summary(results, logs_dir / "run_summary.csv")
    print(f"Wrote run summary: {summary_path}")

    failures = [r for r in results if not r.ok]
    if failures:
        print(f"\n{len(failures)} job(s) failed (continuing to concat):")
        for r in failures:
            print(f"  {r.job.dataset} :: {r.job.codec}  (see {r.log_path})")

    # Single cheap concat + provenance pass (compress / compute_metrics skip existing
    # outputs; concatenate_metrics rebuilds all_results.csv).
    print("\nFinal concat + provenance pass...")
    concat_cmd = [
        "conda", "run", "-n", conda_env, "--no-capture-output",
        "python", "-u", "-m",
        "ocean_taco.benchmarks.climatebenchpress.run_cbp_benchmark",
        "--benchmark-root", str(benchmark_root),
        "--climatebenchpress-root", str(climatebenchpress_root),
        "--metrics", *metric_names,
    ]
    rc = _run_subprocess(concat_cmd, logs_dir / "_concat.log")
    if rc != 0:
        raise RuntimeError(f"Final concat pass failed (rc={rc}).")

    if run_analysis:
        print("Building analysis artifacts...")
        analyze_cmd = [
            "conda", "run", "-n", conda_env, "--no-capture-output",
            "python", "-u", "-m",
            "ocean_taco.benchmarks.climatebenchpress.analyze_results",
            "--benchmark-root", str(benchmark_root),
        ]
        rc = _run_subprocess(analyze_cmd, logs_dir / "_analyze.log")
        if rc != 0:
            raise RuntimeError(f"analyze_results failed (rc={rc}).")

    all_results = benchmark_root / "metrics" / "all_results.csv"
    print(f"\nDone. Concatenated metrics: {all_results}")
    return all_results


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", required=True)
    parser.add_argument("--climatebenchpress-root", required=True)
    parser.add_argument("--dataset-names", nargs="+")
    parser.add_argument(
        "--codecs", nargs="+", help=f"Codecs to run (default: {RETAINED_CODECS})."
    )
    parser.add_argument(
        "--metrics", nargs="+", help=f"Metrics to keep (default: {DEFAULT_METRIC_NAMES})."
    )
    parser.add_argument(
        "-j", "--max-workers", type=int, default=DEFAULT_MAX_WORKERS,
        help="Concurrent isolated subprocesses.",
    )
    parser.add_argument("--conda-env", default=DEFAULT_CONDA_ENV)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--no-analysis", action="store_true", help="Skip the analyze_results pass."
    )
    args = parser.parse_args()

    run_parallel(
        benchmark_root=args.benchmark_root,
        climatebenchpress_root=args.climatebenchpress_root,
        dataset_names=args.dataset_names,
        codecs=args.codecs,
        metric_names=args.metrics,
        max_workers=args.max_workers,
        conda_env=args.conda_env,
        overwrite=args.overwrite,
        run_analysis=not args.no_analysis,
    )


if __name__ == "__main__":
    main()
