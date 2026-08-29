"""Static, manifest-local maps for comparing published QuerySets.

The command-line entry point intentionally reads only a QuerySet's published
header and factored Parquet tables.  It does not stage assets or render any
ocean data; the maps are a diagnostic for the population and its stored
coverage evidence.

Run it from a repository checkout with the optional plotting dependencies::

    python -m ocean_taco.viz.queryset_maps \
        --train querysets/256-training \
        --eval querysets/256-eval \
        --output queryset-map.pdf
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..filter import CoverageRequirement, QueryFilter, select_queryset
from ..geobox import GeoBox, PatchSize, _utc_datetime, utc_isoformat
from ..manifest import QuerySet
from ..retrieve import _REGION_BOUNDS as CORE_REGION_BOUNDS


@dataclass(frozen=True, slots=True)
class SpatialScenario:
    """One labelled spatial selection displayed as four report rows."""

    label: str
    box: GeoBox | None = None


@dataclass(frozen=True, slots=True)
class FootprintRectangle:
    """One non-wrapping geographic piece of a patch footprint."""

    lon_min: float
    lat_min: float
    width: float
    height: float


@dataclass(frozen=True, slots=True)
class ReportPanel:
    """Manifest-derived content for one date/set/coverage map panel."""

    scenario: SpatialScenario
    set_label: str
    coverage_label: str
    date: str
    queryset: QuerySet
    positions: tuple[Mapping[str, Any], ...]

    @property
    def count(self) -> int:
        """Number of selected positions on this panel's date."""
        return len(self.positions)


@dataclass(frozen=True, slots=True)
class ReportRow:
    """The aligned panels for one QuerySet/availability report row."""

    scenario: SpatialScenario
    set_label: str
    coverage_label: str
    panels: tuple[ReportPanel, ...]


def select_common_dates(
    training: QuerySet,
    evaluation: QuerySet,
    requested_dates: Sequence[str] | None = None,
) -> tuple[str, ...]:
    """Return selected shared dates, validating the comparison domain.

    Three shared dates are required even when explicit dates are supplied: the
    command is a three-date diagnostic by default, and this catches a train /
    evaluation population mismatch before producing a partial report.
    """
    common = tuple(sorted(set(training.dates).intersection(evaluation.dates)))
    if len(common) < 3:
        raise ValueError(
            "Training and evaluation QuerySets must share at least three dates; "
            f"found {len(common)}."
        )
    if requested_dates is None:
        return (common[0], common[len(common) // 2], common[-1])

    selected = tuple(utc_isoformat(value) for value in requested_dates)
    if not selected:
        raise ValueError("--dates requires at least one shared ISO-8601 date.")
    if len(set(selected)) != len(selected):
        raise ValueError("--dates must not contain the same date more than once.")
    missing = sorted(set(selected) - set(common))
    if missing:
        rendered = ", ".join(missing)
        raise ValueError(f"--dates includes dates not shared by both QuerySets: {rendered}.")
    return selected


def build_spatial_scenarios(
    *,
    regions: Iterable[str] = (),
    boxes: Iterable[SpatialScenario] = (),
) -> tuple[SpatialScenario, ...]:
    """Build the global report scenario plus requested Core tiles and boxes."""
    scenarios = [SpatialScenario("Global")]
    for region in regions:
        try:
            box = CORE_REGION_BOUNDS[region]
        except KeyError as error:
            choices = ", ".join(sorted(CORE_REGION_BOUNDS))
            raise ValueError(
                f"Unknown Core region {region!r}; choose one of: {choices}."
            ) from error
        scenarios.append(SpatialScenario(region, box))
    scenarios.extend(boxes)

    labels = [scenario.label for scenario in scenarios]
    duplicates = sorted({label for label in labels if labels.count(label) > 1})
    if duplicates:
        raise ValueError(
            "Spatial scenario labels must be unique; duplicates: "
            + ", ".join(duplicates)
            + "."
        )
    return tuple(scenarios)


def covered_filter(*, date: str, box: GeoBox | None = None) -> QueryFilter:
    """Return the stored-evidence filter for usable SWOT and SSH coverage."""
    return QueryFilter(
        date_start=date,
        date_end=date,
        box=box,
        coverage=(
            CoverageRequirement("swot", "valid_ocean_cells", 1.0),
            CoverageRequirement("ssh", "valid_ocean_cells", 1.0),
        ),
    )


def panel_positions(
    queryset: QuerySet,
    *,
    date: str,
    scenario: SpatialScenario,
    require_coverage: bool,
) -> tuple[Mapping[str, Any], ...]:
    """Select positions for one panel through the standard QueryFilter path."""
    query_filter = (
        covered_filter(date=date, box=scenario.box)
        if require_coverage
        else QueryFilter(date_start=date, date_end=date, box=scenario.box)
    )
    selected = select_queryset(queryset, query_filter)
    return tuple(queryset.position(position_index) for position_index, _ in selected.iter_pairs())


def build_report_rows(
    training: QuerySet,
    evaluation: QuerySet,
    *,
    dates: Sequence[str],
    scenarios: Sequence[SpatialScenario] | None = None,
) -> tuple[ReportRow, ...]:
    """Build all aligned report rows without importing plotting dependencies."""
    if training.header["kind"] != "training":
        raise ValueError("--train must name a QuerySet with kind='training'.")
    if evaluation.header["kind"] != "eval":
        raise ValueError("--eval must name a QuerySet with kind='eval'.")
    selected_scenarios = (
        build_spatial_scenarios() if scenarios is None else tuple(scenarios)
    )
    if not selected_scenarios:
        raise ValueError("At least one spatial scenario is required.")
    selected_dates = tuple(utc_isoformat(date) for date in dates)
    if not selected_dates:
        raise ValueError("At least one report date is required.")
    if len(set(selected_dates)) != len(selected_dates):
        raise ValueError("Report dates must not contain duplicates.")
    missing_dates = sorted(
        set(selected_dates) - set(training.dates).intersection(evaluation.dates)
    )
    if missing_dates:
        raise ValueError(
            "Report dates must be shared by both QuerySets: "
            + ", ".join(missing_dates)
            + "."
        )

    rows: list[ReportRow] = []
    for scenario in selected_scenarios:
        for set_label, queryset in (("Training", training), ("Evaluation", evaluation)):
            for coverage_label, require_coverage in (("All positions", False), ("SWOT + SSH covered", True)):
                panels = tuple(
                    ReportPanel(
                        scenario=scenario,
                        set_label=set_label,
                        coverage_label=coverage_label,
                        date=date,
                        queryset=queryset,
                        positions=panel_positions(
                            queryset,
                            date=date,
                            scenario=scenario,
                            require_coverage=require_coverage,
                        ),
                    )
                    for date in selected_dates
                )
                rows.append(
                    ReportRow(
                        scenario=scenario,
                        set_label=set_label,
                        coverage_label=coverage_label,
                        panels=panels,
                    )
                )
    return tuple(rows)


def footprint_rectangles(
    patch_size: PatchSize,
    position: Mapping[str, Any],
) -> tuple[FootprintRectangle, ...]:
    """Return latitude-aware, antimeridian-split rectangles for one position."""
    footprint = patch_size.footprint(
        float(position["centre_lon"]), float(position["centre_lat"])
    )
    return tuple(
        FootprintRectangle(
            lon_min=segment.lon_min,
            lat_min=segment.lat_min,
            width=segment.longitude_width_degrees,
            height=segment.latitude_height_degrees,
        )
        for segment in footprint.segments()
    )


def _require_plotting_dependencies():
    """Import optional plotting dependencies with an actionable installation hint."""
    try:
        import cartopy
        import cartopy.crs as ccrs
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        from matplotlib.patches import Rectangle
    except ImportError as error:  # pragma: no cover - depends on environment
        raise ImportError(
            "QuerySet maps require the optional visualization dependencies. "
            "Install them with 'pip install -e .[viz]'."
        ) from error
    return cartopy, ccrs, plt, Line2D, Rectangle


def _coastline_data_is_available(cartopy: Any) -> bool:
    """Check for Cartopy's local coastline data without invoking its downloader."""
    try:
        from cartopy.io import Downloader

        downloader = Downloader.from_config(
            ("shapefiles", "natural_earth", "110m", "physical", "coastline")
        )
        if downloader is None:
            return False
        format_dict = {
            "config": cartopy.config,
            "resolution": "110m",
            "category": "physical",
            "name": "coastline",
        }
        return any(
            path.is_file()
            for path in (
                downloader.pre_downloaded_path(format_dict),
                downloader.target_path(format_dict),
            )
        )
    except (AttributeError, KeyError, TypeError):
        return False


def _add_coastlines(ax: Any, cartopy: Any) -> None:
    """Draw local Cartopy coastlines, never triggering a Natural Earth fetch."""
    if _coastline_data_is_available(cartopy):
        ax.coastlines(resolution="110m", linewidth=0.35, color="#4b5563", zorder=2)


def _add_box_outline(
    ax: Any,
    box: GeoBox,
    *,
    rectangle: Any,
    transform: Any,
    color: str,
    linewidth: float,
    linestyle: str = "-",
    zorder: int = 3,
) -> None:
    """Draw every non-wrapping component of a geographic selection box."""
    for segment in box.segments():
        ax.add_patch(
            rectangle(
                (segment.lon_min, segment.lat_min),
                segment.longitude_width_degrees,
                segment.latitude_height_degrees,
                fill=False,
                edgecolor=color,
                linewidth=linewidth,
                linestyle=linestyle,
                transform=transform,
                zorder=zorder,
            )
        )


def _set_metadata(queryset: QuerySet) -> str:
    """Format the immutable spatial metadata shown in every map panel."""
    patch_size = queryset.patch_size
    return (
        f"patch {patch_size.value:g} {patch_size.unit}; "
        f"grid {float(queryset.header['grid_spacing_km']):g} km"
    )


def _date_label(value: str) -> str:
    """Format a canonical manifest date for a compact panel title."""
    return _utc_datetime(value).date().isoformat()


def _draw_panel(
    ax: Any,
    panel: ReportPanel,
    *,
    cartopy: Any,
    transform: Any,
    rectangle: Any,
    color: str,
) -> None:
    """Draw one map panel with only report-table derived patch geometry."""
    ax.set_global()
    _add_coastlines(ax, cartopy)
    ax.gridlines(
        draw_labels=False,
        linewidth=0.25,
        color="#9ca3af",
        alpha=0.55,
        linestyle=":",
        zorder=1,
    )
    for region in CORE_REGION_BOUNDS.values():
        _add_box_outline(
            ax,
            region,
            rectangle=rectangle,
            transform=transform,
            color="#9ca3af",
            linewidth=0.35,
            linestyle="--",
        )
    if panel.scenario.box is not None:
        _add_box_outline(
            ax,
            panel.scenario.box,
            rectangle=rectangle,
            transform=transform,
            color="#111827",
            linewidth=0.8,
            zorder=4,
        )
    for position in panel.positions:
        for footprint in footprint_rectangles(panel.queryset.patch_size, position):
            ax.add_patch(
                rectangle(
                    (footprint.lon_min, footprint.lat_min),
                    footprint.width,
                    footprint.height,
                    fill=False,
                    edgecolor=color,
                    linewidth=0.4,
                    alpha=0.75,
                    transform=transform,
                    zorder=5,
                )
            )
    ax.set_title(
        "\n".join(
            (
                _date_label(panel.date),
                f"{panel.set_label} — {panel.coverage_label} (n={panel.count})",
                _set_metadata(panel.queryset),
            )
        ),
        fontsize=7.5,
        loc="left",
        pad=4,
    )


def render_report(
    rows: Sequence[ReportRow],
    output: Path | str,
) -> Path:
    """Render report rows to a PDF or PNG, returning the written path.

    Cartopy coastlines are drawn only when their Natural Earth shapefile is
    already local.  This protects the diagnostic's manifest-local contract
    from Cartopy's otherwise implicit download behavior.
    """
    if not rows:
        raise ValueError("Cannot render a report with no rows.")
    date_count = len(rows[0].panels)
    if date_count == 0 or any(len(row.panels) != date_count for row in rows):
        raise ValueError("Every report row must contain the same non-empty date columns.")

    output_path = Path(output)
    if not output_path.suffix:
        output_path = output_path.with_suffix(".pdf")
    if output_path.suffix.lower() not in {".pdf", ".png"}:
        raise ValueError("--output must end in .pdf or .png.")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cartopy, ccrs, plt, line2d, rectangle = _require_plotting_dependencies()
    transform = ccrs.PlateCarree()
    figure, axes = plt.subplots(
        nrows=len(rows),
        ncols=date_count,
        figsize=(4.65 * date_count, 2.5 * len(rows) + 0.6),
        subplot_kw={"projection": transform},
        squeeze=False,
        layout="constrained",
    )
    colors = {"Training": "#1479b8", "Evaluation": "#d95f02"}
    for row_index, row in enumerate(rows):
        for column_index, panel in enumerate(row.panels):
            _draw_panel(
                axes[row_index, column_index],
                panel,
                cartopy=cartopy,
                transform=transform,
                rectangle=rectangle,
                color=colors[panel.set_label],
            )

    scenario_labels = [row.scenario.label for row in rows]
    figure.suptitle(
        "QuerySet spatial diagnostics — " + ", ".join(dict.fromkeys(scenario_labels)),
        fontsize=11,
    )
    figure.legend(
        handles=(
            line2d([], [], color=colors["Training"], linewidth=1.2, label="Training footprint"),
            line2d([], [], color=colors["Evaluation"], linewidth=1.2, label="Evaluation footprint"),
            line2d([], [], color="#9ca3af", linestyle="--", linewidth=0.8, label="Core tile boundary"),
            line2d([], [], color="#111827", linewidth=1.0, label="Selected scenario"),
        ),
        loc="lower center",
        ncols=4,
        fontsize=8,
        frameon=False,
    )
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return output_path


def create_report(
    training: QuerySet,
    evaluation: QuerySet,
    output: Path | str,
    *,
    requested_dates: Sequence[str] | None = None,
    scenarios: Sequence[SpatialScenario] | None = None,
) -> Path:
    """Select and render the complete report from two published QuerySets."""
    dates = select_common_dates(training, evaluation, requested_dates)
    rows = build_report_rows(
        training,
        evaluation,
        dates=dates,
        scenarios=scenarios,
    )
    return render_report(rows, output)


def _parse_bbox(values: Sequence[str], *, wraps_antimeridian: bool) -> SpatialScenario:
    """Parse one CLI bbox declaration into an explicitly wrapped GeoBox."""
    label, lon_min, lon_max, lat_min, lat_max = values
    try:
        box = GeoBox(
            float(lon_min),
            float(lon_max),
            float(lat_min),
            float(lat_max),
            wraps_antimeridian=wraps_antimeridian,
        )
    except ValueError as error:
        raise ValueError(f"Invalid --bbox {label!r}: {error}") from error
    return SpatialScenario(label, box)


class _BboxAction(argparse.Action):
    """Remember each bbox with its own following antimeridian flag."""

    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: Sequence[str],
        option_string: str | None = None,
    ) -> None:
        declarations = list(getattr(namespace, self.dest) or ())
        declarations.append((tuple(values), False))
        setattr(namespace, self.dest, declarations)


class _WrapsAntimeridianAction(argparse.Action):
    """Apply ``--wraps-antimeridian`` to the immediately preceding bbox."""

    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: Any,
        option_string: str | None = None,
    ) -> None:
        declarations = list(getattr(namespace, "bbox") or ())
        if not declarations:
            raise argparse.ArgumentError(
                self, "must follow the --bbox declaration it modifies"
            )
        bbox, already_wrapped = declarations[-1]
        if already_wrapped:
            raise argparse.ArgumentError(
                self, "was already supplied for the preceding --bbox declaration"
            )
        declarations[-1] = (bbox, True)
        setattr(namespace, "bbox", declarations)


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a manifest-local world-map report for training and evaluation QuerySets."
    )
    parser.add_argument("--train", required=True, type=Path, help="Published training QuerySet directory.")
    parser.add_argument("--eval", required=True, dest="evaluation", type=Path, help="Published evaluation QuerySet directory.")
    parser.add_argument("--output", required=True, type=Path, help="Destination .pdf (default) or .png file.")
    parser.add_argument(
        "--dates",
        nargs="+",
        metavar="DATE",
        help="Explicit shared ISO-8601 dates; default is first, midpoint, and last.",
    )
    parser.add_argument(
        "--region",
        action="append",
        default=[],
        type=str.upper,
        choices=sorted(CORE_REGION_BOUNDS),
        help="Add one of the eight named Core tile scenarios (repeatable).",
    )
    parser.add_argument(
        "--bbox",
        action=_BboxAction,
        default=[],
        nargs=5,
        metavar=("LABEL", "LON_MIN", "LON_MAX", "LAT_MIN", "LAT_MAX"),
        help="Add a labelled geographic scenario (repeatable).",
    )
    parser.add_argument(
        "--wraps-antimeridian",
        action=_WrapsAntimeridianAction,
        nargs=0,
        help="Mark the preceding --bbox longitude interval as antimeridian-wrapping.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the repository CLI and return a conventional process status."""
    parser = _argument_parser()
    args = parser.parse_args(argv)
    try:
        boxes = tuple(
            _parse_bbox(values, wraps_antimeridian=wraps_antimeridian)
            for values, wraps_antimeridian in args.bbox
        )
        scenarios = build_spatial_scenarios(regions=args.region, boxes=boxes)
        training = QuerySet.read(args.train)
        evaluation = QuerySet.read(args.evaluation)
        output = create_report(
            training,
            evaluation,
            args.output,
            requested_dates=args.dates,
            scenarios=scenarios,
        )
    except (ImportError, ValueError) as error:
        parser.error(str(error))
    print(f"Wrote QuerySet map report to {output}")
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through the module CLI
    raise SystemExit(main())


__all__ = [
    "CORE_REGION_BOUNDS",
    "FootprintRectangle",
    "ReportPanel",
    "ReportRow",
    "SpatialScenario",
    "build_report_rows",
    "build_spatial_scenarios",
    "covered_filter",
    "create_report",
    "footprint_rectangles",
    "main",
    "panel_positions",
    "render_report",
    "select_common_dates",
]
