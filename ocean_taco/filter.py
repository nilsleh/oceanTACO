"""Experiment-time filtering over published QuerySet columns only."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import asdict, dataclass, field
from datetime import timedelta
from typing import Any, Literal

from .geobox import GeoBox, _utc_datetime, utc_isoformat
from .manifest import QuerySet, content_sha256

Aggregate = Literal["sum", "mean", "min"]


@dataclass(frozen=True, slots=True)
class CoverageRequirement:
    """One declarative predicate over stored native coverage evidence."""

    token: Literal["swot", "ssh", "argo"]
    metric: str
    minimum: float
    aggregate: Aggregate = "sum"

    def __post_init__(self) -> None:
        allowed = {
            "swot": {
                "valid_cells",
                "valid_ocean_cells",
                "n_obs_sum",
                "valid_fraction_footprint",
                "valid_fraction_ocean",
            },
            "ssh": {
                "valid_cells",
                "valid_ocean_cells",
                "valid_fraction_footprint",
                "valid_fraction_ocean",
            },
            "argo": {"profile_count"},
        }
        if self.metric not in allowed[self.token]:
            raise ValueError(
                f"Unsupported {self.token} coverage metric {self.metric!r}."
            )
        if self.aggregate not in {"sum", "mean", "min"}:
            raise ValueError("Coverage aggregate must be 'sum', 'mean', or 'min'.")

    def to_dict(self) -> dict[str, Any]:
        """Return a canonical, recordable coverage predicate."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class QueryFilter:
    """A serialisable selection over a published population.

    The filter contains no source callback and performs neither source I/O nor
    re-measurement.  Context offsets and lead only restrict which anchors are
    within the canonical date domain; coverage aggregation uses context dates.
    """

    date_start: str | None = None
    date_end: str | None = None
    box: GeoBox | None = None
    region_mask_any: int | None = None
    coverage: tuple[CoverageRequirement, ...] = ()
    context_start_offset_days: int = 0
    context_end_offset_days: int = 0
    relation: Literal["same_time", "forecast"] = "same_time"
    target_lead_days: int = 0

    def __post_init__(self) -> None:
        if (
            self.date_start is not None
            and self.date_end is not None
            and _utc_datetime(self.date_end) < _utc_datetime(self.date_start)
        ):
            raise ValueError("date_end must not precede date_start.")
        if self.region_mask_any is not None and not 0 <= self.region_mask_any <= 255:
            raise ValueError("region_mask_any must fit the uint8 position mask.")
        if self.context_end_offset_days < self.context_start_offset_days:
            raise ValueError("Context offsets must form an ordered contiguous range.")
        if self.relation == "same_time" and self.target_lead_days != 0:
            raise ValueError("same_time filters require target_lead_days=0.")
        if self.relation == "forecast" and self.target_lead_days <= 0:
            raise ValueError("forecast filters require a positive target_lead_days.")

    def to_dict(self) -> dict[str, Any]:
        """Return a canonical, recordable selection description."""
        return {
            "date_start": None
            if self.date_start is None
            else utc_isoformat(self.date_start),
            "date_end": None if self.date_end is None else utc_isoformat(self.date_end),
            "box": None if self.box is None else self.box.to_dict(),
            "region_mask_any": self.region_mask_any,
            "coverage": [requirement.to_dict() for requirement in self.coverage],
            "context_start_offset_days": self.context_start_offset_days,
            "context_end_offset_days": self.context_end_offset_days,
            "relation": self.relation,
            "target_lead_days": self.target_lead_days,
        }

    @property
    def sha256(self) -> str:
        """Stable recordable filter identity."""
        return content_sha256(self.to_dict())


def _inside_box(lon: float, lat: float, box: GeoBox) -> bool:
    if not box.lat_min <= lat <= box.lat_max:
        return False
    return (
        (lon >= box.lon_min or lon <= box.lon_max)
        if box.wraps_antimeridian
        else box.lon_min <= lon <= box.lon_max
    )


def _coverage_value(
    queryset: QuerySet,
    position: Mapping[str, Any],
    date_index: int,
    requirement: CoverageRequirement,
) -> float | None:
    row = queryset.coverage_row(int(position["position_index"]), date_index)
    if requirement.token == "argo":
        return (
            None
            if row["argo_profile_count"] is None
            else float(row["argo_profile_count"])
        )
    prefix = requirement.token
    if requirement.metric == "valid_fraction_footprint":
        numerator, denominator = (
            row[f"{prefix}_valid_cells"],
            position[f"{prefix}_footprint_cells"],
        )
        return (
            None
            if numerator is None or denominator == 0
            else float(numerator) / float(denominator)
        )
    if requirement.metric == "valid_fraction_ocean":
        numerator, denominator = (
            row[f"{prefix}_valid_ocean_cells"],
            position[f"{prefix}_ocean_cells"],
        )
        return (
            None
            if numerator is None or denominator == 0
            else float(numerator) / float(denominator)
        )
    value = row[f"{prefix}_{requirement.metric}"]
    return None if value is None else float(value)


@dataclass(slots=True)
class SelectedPairs:
    """Lazily resolvable canonical pair selection.

    The unfiltered date/geography case remains a Cartesian product.  Coverage
    predicates scan the local factored coverage table deterministically; no
    product table is materialised and no proposal/rejection loop is involved.
    """

    queryset: QuerySet
    query_filter: QueryFilter = field(default_factory=QueryFilter)
    _positions: tuple[int, ...] = field(init=False, repr=False)
    _dates: tuple[int, ...] = field(init=False, repr=False)
    _date_lookup: Mapping[str, int] = field(init=False, repr=False)
    _count: int | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        lookup = {value: index for index, value in enumerate(self.queryset.dates)}
        self._date_lookup = lookup
        start = (
            _utc_datetime(self.query_filter.date_start)
            if self.query_filter.date_start
            else None
        )
        end = (
            _utc_datetime(self.query_filter.date_end)
            if self.query_filter.date_end
            else None
        )
        dates: list[int] = []
        for index, value in enumerate(self.queryset.dates):
            anchor = _utc_datetime(value)
            if start is not None and anchor < start or end is not None and anchor > end:
                continue
            if not self._anchor_domain_valid(anchor):
                continue
            dates.append(index)
        positions: list[int] = []
        for position in self.queryset.positions:
            if self.query_filter.box is not None and not _inside_box(
                float(position["centre_lon"]),
                float(position["centre_lat"]),
                self.query_filter.box,
            ):
                continue
            if self.query_filter.region_mask_any is not None and not (
                int(position["region_mask"]) & self.query_filter.region_mask_any
            ):
                continue
            positions.append(int(position["position_index"]))
        self._dates, self._positions = tuple(dates), tuple(positions)

    def _anchor_domain_valid(self, anchor) -> bool:
        for offset in range(
            self.query_filter.context_start_offset_days,
            self.query_filter.context_end_offset_days + 1,
        ):
            if utc_isoformat(anchor + timedelta(days=offset)) not in self._date_lookup:
                return False
        if (
            self.query_filter.relation == "forecast"
            and utc_isoformat(
                anchor + timedelta(days=self.query_filter.target_lead_days)
            )
            not in self._date_lookup
        ):
            return False
        return True

    @property
    def is_cartesian(self) -> bool:
        """Whether no coverage predicate turns the logical product sparse."""
        return not self.query_filter.coverage

    def _coverage_matches(self, position_index: int, date_index: int) -> bool:
        if not self.query_filter.coverage:
            return True
        position = self.queryset.position(position_index)
        anchor = _utc_datetime(self.queryset.dates[date_index])
        context_indices = [
            self._date_lookup[utc_isoformat(anchor + timedelta(days=offset))]
            for offset in range(
                self.query_filter.context_start_offset_days,
                self.query_filter.context_end_offset_days + 1,
            )
        ]
        for requirement in self.query_filter.coverage:
            values = [
                _coverage_value(self.queryset, position, context_index, requirement)
                for context_index in context_indices
            ]
            # A null native fraction/count is unavailable evidence, never a
            # numeric zero that can accidentally pass a relaxed threshold.
            if any(value is None for value in values):
                return False
            numbers = [float(value) for value in values]
            aggregate = (
                sum(numbers)
                if requirement.aggregate == "sum"
                else min(numbers)
                if requirement.aggregate == "min"
                else sum(numbers) / len(numbers)
            )
            if aggregate < requirement.minimum:
                return False
        return True

    def iter_pairs(self) -> Iterator[tuple[int, int]]:
        """Yield selected pairs in canonical ``(position, date)`` order."""
        for position_index in self._positions:
            for date_index in self._dates:
                if self._coverage_matches(position_index, date_index):
                    yield position_index, date_index

    @property
    def count(self) -> int:
        """Number of selected pairs, calculated without remote/source I/O."""
        if self._count is None:
            self._count = (
                len(self._positions) * len(self._dates)
                if self.is_cartesian
                else sum(1 for _ in self.iter_pairs())
            )
        return self._count

    def resolve_rank(self, rank: int) -> tuple[int, int]:
        """Resolve one selected canonical ordinal without materialising a product."""
        if not 0 <= rank < self.count:
            raise IndexError("Selected pair rank is outside the population.")
        if self.is_cartesian:
            width = len(self._dates)
            return self._positions[rank // width], self._dates[rank % width]
        for current, pair in enumerate(self.iter_pairs()):
            if current == rank:
                return pair
        raise AssertionError("Selected pair count changed during deterministic scan.")

    def to_dict(self) -> dict[str, Any]:
        """Return the exact selection description stored in experiment records."""
        return self.query_filter.to_dict()


def select_queryset(
    queryset: QuerySet, query_filter: QueryFilter | None = None
) -> SelectedPairs:
    """Create a local-only selection over one published population."""
    return SelectedPairs(queryset, query_filter or QueryFilter())


__all__ = ["CoverageRequirement", "QueryFilter", "SelectedPairs", "select_queryset"]
