"""Build paper artifacts from real ClimateBenchPress benchmark outputs.

Consumes the concatenated ``metrics/all_results.csv`` written by ClimateBenchPress'
``concatenate_metrics`` (plus the packaged ``standardized.zarr`` datasets) and produces:

1. A modality x compressor comparison table **at the 1x anchor bound** (compression
   ratio, bound compliance, MAE, max-abs / max-rel error, PSNR), where every codec is
   held to OceanTACO's exact shipped fidelity.
2. A separate, clearly-labelled table of OceanTACO's **actual NetCDF/HDF5 container**
   sizes (includes container + chunking overhead, not directly comparable to raw codec
   bitstreams). At the anchor the in-framework ``oceantaco`` ratio should approximately
   equal this table -- a built-in cross-check.

Run under conda env ``testpy312``.
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from ocean_taco.benchmarks.climatebenchpress.config import (
    BENCHMARK_MODALITY_CONFIGS,
    DATASET_NAME_TO_MODALITY,
)

# Anchor (1x) corresponds to the smallest abs_error tier, which CBP labels "low".
ANCHOR_BOUND_NAME = "low"
CR_COLUMN = "Compression Ratio [raw B / enc B]"
DISTORTION_COLUMN = "Max Absolute Error"
METRIC_COLUMNS = [
    CR_COLUMN,
    "MAE",
    "Max Absolute Error",
    "Max Relative Error",
    "PSNR",
    "Satisfies Bound (Passed)",
    "Satisfies Bound (Value)",
]


def _strip_variant_suffix(compressor: str) -> str:
    """Drop CBP's -conservative-abs/-rel error-bound conversion suffix for display."""
    for suffix in ("-conservative-abs", "-conservative-rel"):
        if compressor.endswith(suffix):
            return compressor[: -len(suffix)]
    return compressor


def _dataset_display_name(dataset_name: str) -> str:
    modality = DATASET_NAME_TO_MODALITY.get(dataset_name)
    if modality is None:
        return dataset_name
    return BENCHMARK_MODALITY_CONFIGS[modality].display_name


def load_all_results(benchmark_root: str | Path) -> pd.DataFrame:
    """Load CBP's concatenated all_results.csv into a tidy DataFrame."""
    path = Path(benchmark_root) / "metrics" / "all_results.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"No concatenated metrics at {path}. Run run_cbp_benchmark first."
        )
    df = pd.read_csv(path)
    df["Codec"] = df["Compressor"].map(_strip_variant_suffix)
    df["Dataset Display"] = df["Dataset"].map(_dataset_display_name)
    return df


def build_anchor_table(df: pd.DataFrame) -> pd.DataFrame:
    """Comparison table at the 1x anchor bound: modality x codec x metrics."""
    anchor = df[df["Error Bound Name"] == ANCHOR_BOUND_NAME].copy()
    if anchor.empty:
        raise ValueError(
            f"No rows at anchor bound '{ANCHOR_BOUND_NAME}'. Found: "
            f"{sorted(df['Error Bound Name'].unique())}."
        )
    columns = ["Dataset Display", "Codec", "Variable"] + [
        c for c in METRIC_COLUMNS if c in anchor.columns
    ]
    table = (
        anchor[columns]
        .sort_values(["Dataset Display", CR_COLUMN], ascending=[True, False])
        .reset_index(drop=True)
    )
    return table


def write_anchor_table(table: pd.DataFrame, output_dir: Path) -> tuple[Path, Path]:
    """Write the anchor comparison table as CSV and a LaTeX table."""
    csv_path = output_dir / "anchor_comparison.csv"
    tex_path = output_dir / "anchor_comparison.tex"
    table.to_csv(csv_path, index=False)
    fmt = table.copy()
    if CR_COLUMN in fmt:
        fmt[CR_COLUMN] = fmt[CR_COLUMN].map(lambda x: f"{x:.2f}")
    for col in ("MAE", "Max Absolute Error", "Max Relative Error"):
        if col in fmt:
            fmt[col] = fmt[col].map(lambda x: f"{x:.3e}")
    if "PSNR" in fmt:
        fmt["PSNR"] = fmt["PSNR"].map(lambda x: f"{x:.1f}")
    if "Satisfies Bound (Value)" in fmt:
        fmt["Satisfies Bound (Value)"] = fmt["Satisfies Bound (Value)"].map(
            lambda x: f"{100 * x:.2f}%"
        )
    tex_path.write_text(
        fmt.to_latex(index=False, escape=True, longtable=False), encoding="utf-8"
    )
    return csv_path, tex_path


def plot_rate_distortion(df: pd.DataFrame, output_dir: Path) -> Path:
    """One rate-distortion panel per dataset: CR vs max-abs error across the sweep."""
    datasets = sorted(df["Dataset"].unique())
    ncols = min(2, len(datasets))
    nrows = int(np.ceil(len(datasets) / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(6.5 * ncols, 5 * nrows), squeeze=False
    )
    for idx, dataset in enumerate(datasets):
        ax = axes[idx // ncols][idx % ncols]
        sub = df[df["Dataset"] == dataset]
        for codec in sorted(sub["Codec"].unique()):
            cd = sub[sub["Codec"] == codec].sort_values(CR_COLUMN)
            is_oceantaco = codec == "oceantaco"
            ax.plot(
                cd[CR_COLUMN],
                cd[DISTORTION_COLUMN],
                marker="*" if is_oceantaco else "o",
                markersize=14 if is_oceantaco else 7,
                linewidth=3 if is_oceantaco else 1.5,
                label=codec,
                zorder=5 if is_oceantaco else 2,
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(CR_COLUMN)
        ax.set_ylabel(DISTORTION_COLUMN)
        ax.set_title(_dataset_display_name(dataset))
        ax.grid(True, which="major", alpha=0.3)
        ax.legend(fontsize=8)
    for idx in range(len(datasets), nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")
    fig.tight_layout()
    out_path = output_dir / "rate_distortion.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def compute_netcdf_container_sizes(benchmark_root: str | Path) -> pd.DataFrame:
    """Measure OceanTACO's real NetCDF container sizes vs float32.

    For each packaged dataset, write the primary variable to NetCDF using OceanTACO's
    shipped int16 + zlib(level=5) encoding and compare the on-disk size to the raw
    float32 size. This includes NetCDF/HDF5 container + chunking overhead, so it is not
    directly comparable to raw codec bitstreams -- but at the anchor it should be close
    to the in-framework ``oceantaco`` compression ratio.
    """
    benchmark_root = Path(benchmark_root)
    rows = []
    for modality, config in BENCHMARK_MODALITY_CONFIGS.items():
        store = (
            benchmark_root / "datasets" / config.dataset_name / "standardized.zarr"
        )
        if not store.exists():
            continue
        ds = xr.open_dataset(store, engine="zarr", chunks={})
        var = config.primary_var
        data = ds[var].astype(np.float32)
        float32_bytes = int(data.size * 4)
        encoding = {
            var: {
                "zlib": True,
                "complevel": 5,
                "_FillValue": config.fill_value,
                "dtype": "int16",
                "scale_factor": config.scale_factor,
                "add_offset": config.add_offset,
            }
        }
        with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as handle:
            tmp_path = Path(handle.name)
        try:
            data.to_dataset(name=var).to_netcdf(
                tmp_path, engine="h5netcdf", encoding=encoding
            )
            netcdf_bytes = tmp_path.stat().st_size
        finally:
            tmp_path.unlink(missing_ok=True)
        rows.append(
            {
                "Dataset Display": config.display_name,
                "Variable": var,
                "Float32 bytes": float32_bytes,
                "NetCDF bytes": netcdf_bytes,
                "Container CR [float32 / netcdf]": float32_bytes / netcdf_bytes,
            }
        )
    return pd.DataFrame(rows)


def generate_analysis(benchmark_root: str | Path) -> dict[str, Path]:
    """Produce all paper artifacts from CBP outputs under ``benchmark_root/analysis``."""
    benchmark_root = Path(benchmark_root)
    output_dir = benchmark_root / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_all_results(benchmark_root)
    anchor_table = build_anchor_table(df)
    csv_path, tex_path = write_anchor_table(anchor_table, output_dir)

    container = compute_netcdf_container_sizes(benchmark_root)
    container_path = output_dir / "netcdf_container_sizes.csv"
    container.to_csv(container_path, index=False)

    artifacts = {
        "anchor_csv": csv_path,
        "anchor_tex": tex_path,
        "container_csv": container_path,
    }
    print(f"Anchor comparison table: {csv_path}")
    print(f"Anchor comparison LaTeX: {tex_path}")
    print(f"NetCDF container table:   {container_path}")

    # The rate-distortion figure needs the 2x/4x sweep; under anchor-only there is a single
    # error bound and the figure is meaningless, so it is skipped. Restoring the sweep
    # (DEFAULT_ERROR_MULTIPLIERS) automatically re-enables it.
    if df["Error Bound Name"].nunique() > 1:
        rd_path = plot_rate_distortion(df, output_dir)
        artifacts["rate_distortion"] = rd_path
        print(f"Rate-distortion figure:  {rd_path}")

    return artifacts


def main() -> None:
    """CLI entrypoint (run under conda env testpy312)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", required=True)
    args = parser.parse_args()
    generate_analysis(args.benchmark_root)


if __name__ == "__main__":
    main()
