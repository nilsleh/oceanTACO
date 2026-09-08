#!/usr/bin/env python3
"""Render the L3 SWOT mission-phase figures used by the README and docs.

SWOT flew a 1-day repeat orbit during its calibration ("fast-sampling") phase
and a 21-day repeat orbit during the science phase. Over a fixed bounding box
the two look nothing alike, and both are easy to mistake for broken data:

- Calibration: the swath sits in the same place every day, so a daily mosaic
  looks "identical" even though the values underneath change.
- Science: the swath moves every day, so a small box is empty on most days and
  the empty days are pixel-identical to each other.

These figures make both regimes legible. They read the published per-date TACO
folder directly rather than downloading from Hugging Face, so they can be
regenerated from a local dataset copy.
"""

from __future__ import annotations

import argparse
import datetime as dt
import itertools
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

DEFAULT_DATA_ROOT = Path(
    "/p/project1/hai_uqmethodbox/data/new_ssh_dataset_taco_folder/OceanTACO/DATA"
)

VARIABLE = "ssha_filtered"
REGION = "NORTH_ATLANTIC"

# The box the reported "data looks all the same" screenshots used.
REPORT_BOX = (40.0, 50.0, -50.0, -40.0)

# Phase boundaries measured from the published Core release. The calibration
# phase ends on the last date flown in the 1-day repeat orbit; no SWOT asset is
# published during the orbit-change window that follows.
CALIBRATION_END = dt.date(2023, 7, 10)
SCIENCE_START = dt.date(2023, 7, 26)

CALIBRATION_WINDOW = (dt.date(2023, 4, 5), 12)
SCIENCE_WINDOW = (dt.date(2024, 6, 1), 24)

SSHA_LIMIT = 0.3
DIFF_LIMIT = 0.15
EMPTY_FACECOLOR = "#e8e8e8"


@dataclass(frozen=True)
class Field:
    """One day's gridded SWOT crop, or a record that the day has no asset."""

    date: dt.date
    lat: np.ndarray | None
    lon: np.ndarray | None
    values: np.ndarray | None

    @property
    def missing(self) -> bool:
        """Whether the date has no ``l3_swot.nc`` at all."""
        return self.values is None

    @property
    def coverage(self) -> float:
        """Fraction of the box carrying a finite observation."""
        if self.values is None:
            return math.nan
        return float(np.isfinite(self.values).mean())

    @property
    def extent(self) -> list[float]:
        """Matplotlib ``imshow`` extent for this crop."""
        assert self.lat is not None and self.lon is not None
        return [self.lon.min(), self.lon.max(), self.lat.min(), self.lat.max()]

    @property
    def label(self) -> str:
        """Date rendered the way the panel titles want it."""
        return self.date.isoformat()


def label_for(date: dt.date) -> str:
    """Return the ``YYYY_MM_DD`` folder name for a date."""
    return date.strftime("%Y_%m_%d")


def consecutive(start: dt.date, count: int) -> list[dt.date]:
    """Return ``count`` consecutive calendar dates beginning at ``start``."""
    return [start + dt.timedelta(days=offset) for offset in range(count)]


def describe_box(box: tuple[float, float, float, float]) -> str:
    """Render a bounding box the way the figure titles want it."""
    lat_min, lat_max, lon_min, lon_max = box
    lat = f"lat {abs(lat_min):g}–{abs(lat_max):g}°{'N' if lat_min >= 0 else 'S'}"
    lon = f"lon {abs(lon_max):g}–{abs(lon_min):g}°{'W' if lon_min < 0 else 'E'}"
    return f"{lat}, {lon}"


def phase_of(date: dt.date) -> str:
    """Classify a date into the SWOT mission phase it was flown in."""
    if date <= CALIBRATION_END:
        return "calibration"
    if date < SCIENCE_START:
        return "orbit change"
    return "science"


def read_field(
    data_root: Path, date: dt.date, box: tuple[float, float, float, float]
) -> Field:
    """Read one day's SWOT crop, tolerating dates that publish no SWOT asset."""
    path = data_root / label_for(date) / REGION / "l3_swot.nc"
    if not path.exists():
        return Field(date=date, lat=None, lon=None, values=None)
    lat_min, lat_max, lon_min, lon_max = box
    with xr.open_dataset(path) as dataset:
        crop = (
            dataset[VARIABLE]
            .isel(time=0)
            .sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))
        )
        return Field(
            date=date,
            lat=np.asarray(crop["lat"].values, dtype=np.float64),
            lon=np.asarray(crop["lon"].values, dtype=np.float64),
            values=np.asarray(crop.values, dtype=np.float32),
        )


def draw_panel(ax: plt.Axes, field: Field) -> plt.cm.ScalarMappable | None:
    """Draw one day of SSHA, annotating empty and missing days explicitly."""
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor(EMPTY_FACECOLOR)
    if field.missing:
        ax.set_title(f"{field.label}\nno SWOT asset", fontsize=7)
        ax.text(
            0.5,
            0.5,
            "no data",
            ha="center",
            va="center",
            fontsize=8,
            color="0.35",
            transform=ax.transAxes,
        )
        return None
    image = ax.imshow(
        np.ma.masked_invalid(field.values),
        origin="lower",
        extent=field.extent,
        aspect="auto",
        interpolation="nearest",
        cmap="RdBu_r",
        vmin=-SSHA_LIMIT,
        vmax=SSHA_LIMIT,
    )
    coverage = field.coverage
    note = "  (empty)" if coverage == 0.0 else ""
    ax.set_title(f"{field.label}\ncoverage={coverage:.3f}{note}", fontsize=7)
    return image


def mosaic(
    fields: list[Field], title: str, out_path: Path, cols: int, dpi: int
) -> None:
    """Save a daily mosaic of SSHA panels."""
    rows = math.ceil(len(fields) / cols)
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(2.6 * cols, 2.55 * rows),
        constrained_layout=True,
        squeeze=False,
    )
    image = None
    for ax, field in zip(axes.ravel(), fields):
        drawn = draw_panel(ax, field)
        image = image or drawn
    for ax in axes.ravel()[len(fields) :]:
        ax.axis("off")
    if image is not None:
        fig.colorbar(
            image, ax=axes.ravel().tolist(), fraction=0.02, label=f"{VARIABLE} [m]"
        )
    fig.suptitle(title, fontsize=11)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    print(f"wrote {out_path}")


def differences(
    fields: list[Field], title: str, out_path: Path, cols: int, dpi: int
) -> None:
    """Save consecutive-day difference maps proving the values are not repeated."""
    pairs = [
        (before, after)
        for before, after in itertools.pairwise(fields)
        if not before.missing and not after.missing
    ]
    rows = math.ceil(len(pairs) / cols)
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(3.1 * cols, 2.9 * rows),
        constrained_layout=True,
        squeeze=False,
    )
    image = None
    for ax, (before, after) in zip(axes.ravel(), pairs):
        delta = after.values - before.values
        image = ax.imshow(
            np.ma.masked_invalid(delta),
            origin="lower",
            extent=after.extent,
            aspect="auto",
            interpolation="nearest",
            cmap="PuOr",
            vmin=-DIFF_LIMIT,
            vmax=DIFF_LIMIT,
        )
        overlap = np.isfinite(delta)
        mean_abs = (
            float(np.nanmean(np.abs(delta[overlap]))) if overlap.any() else math.nan
        )
        ax.set_title(
            f"{after.date.strftime('%m-%d')} − {before.date.strftime('%m-%d')}\n"
            f"mean|Δ|={mean_abs:.4f} m",
            fontsize=8,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_facecolor(EMPTY_FACECOLOR)
    for ax in axes.ravel()[len(pairs) :]:
        ax.axis("off")
    if image is not None:
        fig.colorbar(image, ax=axes.ravel().tolist(), fraction=0.02, label="Δ SSHA [m]")
    fig.suptitle(title, fontsize=11)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    print(f"wrote {out_path}")


def coverage_timeline(
    data_root: Path, box: tuple[float, float, float, float], out_path: Path, dpi: int
) -> None:
    """Save a coverage-fraction timeline spanning both mission phases."""
    start, end = dt.date(2023, 3, 29), dt.date(2024, 3, 31)
    dates, values = [], []
    for date in consecutive(start, (end - start).days + 1):
        field = read_field(data_root, date, box)
        dates.append(date)
        values.append(field.coverage if not field.missing else math.nan)
    series = np.asarray(values, dtype=float)

    fig, ax = plt.subplots(figsize=(13, 4.2), constrained_layout=True)
    ax.axvspan(
        dates[0], CALIBRATION_END, color="#cfe3f2", label="calibration (1-day repeat)"
    )
    ax.axvspan(
        CALIBRATION_END, SCIENCE_START, color="#f0d9d9", label="orbit change (no SWOT)"
    )
    ax.axvspan(
        SCIENCE_START, dates[-1], color="#dcefdc", label="science (21-day repeat)"
    )
    ax.plot(dates, series, color="#1f3b57", linewidth=0.9)
    missing = np.isnan(series)
    if missing.any():
        ax.plot(
            [d for d, flag in zip(dates, missing) if flag],
            np.zeros(int(missing.sum())),
            linestyle="none",
            marker="x",
            markersize=4,
            color="#a03030",
            label="no SWOT asset",
        )
    ax.set_ylabel(f"coverage fraction of box\n({VARIABLE} finite)")
    ax.set_xlabel("date")
    ax.set_ylim(bottom=0)
    ax.margins(x=0.01)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.95)
    ax.set_title(
        f"L3 SWOT coverage over a fixed box ({REGION}, {describe_box(box)})\n"
        "Flat plateau = calibration's fixed swath; spiky signal = the science-phase 21-day cycle",
        fontsize=11,
    )
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    print(f"wrote {out_path}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data_root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="OceanTACO DATA folder.",
    )
    parser.add_argument(
        "--out", type=Path, default=Path("docs/images"), help="Directory for the PNGs."
    )
    parser.add_argument("--dpi", type=int, default=130)
    parser.add_argument(
        "--skip_timeline",
        action="store_true",
        help="Skip the coverage timeline, which reads a full year of files.",
    )
    return parser.parse_args()


def main() -> int:
    """Render every SWOT mission-phase figure."""
    args = parse_args()
    if not args.data_root.exists():
        raise FileNotFoundError(f"OceanTACO DATA folder not found: {args.data_root}")
    args.out.mkdir(parents=True, exist_ok=True)

    box_label = describe_box(REPORT_BOX)

    cal_start, cal_days = CALIBRATION_WINDOW
    calibration = [
        read_field(args.data_root, d, REPORT_BOX)
        for d in consecutive(cal_start, cal_days)
    ]
    mosaic(
        calibration,
        f"SWOT calibration phase — 1-day repeat orbit ({REGION}, {box_label})\n"
        "The swath sits in the same place every day, so the mosaic looks identical — "
        "but the values inside it change (see the difference figure).",
        args.out / "swot_phase_calibration.png",
        cols=6,
        dpi=args.dpi,
    )
    differences(
        calibration,
        f"SWOT calibration phase — consecutive-day differences ({REGION}, {box_label})\n"
        "Non-zero, spatially coherent structure everywhere: the days are genuinely different, not duplicated.",
        args.out / "swot_phase_differences.png",
        cols=5,
        dpi=args.dpi,
    )

    sci_start, sci_days = SCIENCE_WINDOW
    science = [
        read_field(args.data_root, d, REPORT_BOX)
        for d in consecutive(sci_start, sci_days)
    ]
    mosaic(
        science,
        f"SWOT science phase — 21-day repeat orbit ({REGION}, {box_label})\n"
        "The swath moves every day, so a fixed box is empty on most days; "
        "empty days are pixel-identical to each other.",
        args.out / "swot_phase_science.png",
        cols=6,
        dpi=args.dpi,
    )

    if not args.skip_timeline:
        coverage_timeline(
            args.data_root, REPORT_BOX, args.out / "swot_revisit_coverage.png", args.dpi
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
