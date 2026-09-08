"""Fast, date-major measurement engine for building released QuerySets.

The released measurement functions in :mod:`ocean_taco.sampling.coverage` are
per-(position, date) and re-open every source asset on each call.  A published
QuerySet stores the full cartesian product of positions and dates, so the
released call pattern would re-open source files tens of millions of times.

This module keeps the released *semantics* and replaces only the access
pattern: each date's eight regional tiles are read once, scattered onto a
global array, and then every position of every published set is measured
against the hot in-memory arrays.

Two deliberate deviations from released code, both validated by
``tests/test_build_querysets.py`` against the released functions:

* Tiles are merged by direct quadrant scatter instead of
  :func:`ocean_taco.retrieve._merge_grid_tiles`, whose coordinate clustering
  materialises a dense ``len(axis) x len(values)`` matrix per axis.
* Argo profile counting is vectorised over a whole date instead of re-parsing
  every timestamp per footprint.

Source assets in the released local dataset carry no ``units`` attribute on
their primary variables.  Coverage counting is unit-independent (it counts
finite cells), so a missing attribute is treated as the registry's canonical
unit and recorded as an explicit provenance assumption; a *present* but
unrecognised unit is still a hard failure.
"""

from __future__ import annotations

import numpy as np

from ocean_taco.geobox import GeoBox, PatchSize
from ocean_taco.registry import ModalitySpec, get_modality
from ocean_taco.retrieve import _REGION_BOUNDS, _intersects
from ocean_taco.sampling.coverage import DenseCoverage, unavailable_dense_coverage
from ocean_taco.sampling.ocean_mask import OceanMaskArtifact

REGIONS: tuple[str, ...] = tuple(sorted(_REGION_BOUNDS))

#: Coverage tokens, in the fixed order used by the coverage table columns.
DENSE_TOKENS: tuple[str, ...] = ("l3_swot", "l3_ssh")
POINT_TOKEN = "argo"

#: Coverage column names keyed by the ``build_coverage_table`` measure key.
COVERAGE_KEY = {"l3_swot": "swot", "l3_ssh": "ssh"}


def canonical_lon(values: np.ndarray) -> np.ndarray:
    """Map longitudes onto ``[-180, 180)`` exactly as the released code does."""
    values = np.asarray(values, dtype=np.float64)
    return np.where(values == 180.0, -180.0, ((values + 180.0) % 360.0) - 180.0)


def _normalise_unit(value: str) -> str:
    """Fold documented spelling differences before comparing a source unit.

    Ingest owns this: the loader deliberately does not inspect source units,
    so the rule lives with the check that still enforces it.
    """
    return value.strip().replace("\u00b0", "deg").replace("_", " ").lower()


def check_units(spec: ModalitySpec, units: object) -> str:
    """Validate a source unit, allowing a missing attribute only.

    Returns the effective unit.  ``canonicalise_dense`` rejects an absent
    ``units`` attribute outright; the released local assets have none, so this
    substitutes the registry's canonical unit and lets the caller record the
    substitution in provenance.  A present-but-unknown unit still raises, so a
    genuine product change cannot pass silently.
    """
    if units is None or str(units).strip() == "":
        return spec.canonical_unit
    supplied = _normalise_unit(str(units))
    accepted = {_normalise_unit(unit) for unit in spec.accepted_units}
    if supplied in accepted or supplied == _normalise_unit(spec.canonical_unit):
        return str(units)
    raise ValueError(
        f"{spec.token} declares unsupported decoded units {units!r}; "
        f"expected {spec.canonical_unit!r}."
    )


class GlobalGrid:
    """Global axes for one dense token plus each region's exact slice.

    The merged axis is *not* uniformly spaced: the regional tiles abut with a
    narrower step at the equator and the +/-90 degree seams.  Every index
    computed against it therefore uses :func:`numpy.searchsorted`, never
    ``(x - x0) / step``.
    """

    __slots__ = ("token", "lat", "lon", "slices")

    def __init__(
        self,
        token: str,
        lat: np.ndarray,
        lon: np.ndarray,
        slices: dict[str, tuple[slice, slice]],
    ) -> None:
        self.token = token
        self.lat = lat
        self.lon = lon
        self.slices = slices

    @property
    def shape(self) -> tuple[int, int]:
        """Global ``(lat, lon)`` array shape."""
        return self.lat.size, self.lon.size


def _tile_axes(dataset) -> tuple[np.ndarray, np.ndarray]:
    lat = np.asarray(dataset["lat"].values, dtype=np.float64)
    lon = canonical_lon(np.asarray(dataset["lon"].values, dtype=np.float64))
    if lat.ndim != 1 or lon.ndim != 1:
        raise ValueError("Dense source tiles require one-dimensional lat/lon.")
    order = np.argsort(lon)
    if not np.all(np.diff(lat) > 0):
        raise ValueError("Dense source tile latitudes are not strictly ascending.")
    return lat, lon[order]


def build_global_grid(token: str, tile_datasets: dict[str, object]) -> GlobalGrid:
    """Derive global axes and per-region slices, asserting an exact tiling.

    The eight Core regions are disjoint aligned quadrants, so the union of
    their axes must reproduce each tile as one contiguous slice and the eight
    rectangles must cover the global array exactly once.  Anything else means
    the product layout changed and a silent misalignment would corrupt every
    measured coverage value, so this fails loudly instead.
    """
    if set(tile_datasets) != set(REGIONS):
        raise ValueError(
            f"{token} global grid needs all {len(REGIONS)} regions; "
            f"got {sorted(tile_datasets)}."
        )
    axes = {
        region: _tile_axes(dataset) for region, dataset in sorted(tile_datasets.items())
    }
    lat = np.unique(np.concatenate([value[0] for value in axes.values()]))
    lon = np.unique(np.concatenate([value[1] for value in axes.values()]))
    slices: dict[str, tuple[slice, slice]] = {}
    cover = np.zeros((lat.size, lon.size), dtype=np.int8)
    for region, (tile_lat, tile_lon) in axes.items():
        row = slice(
            int(np.searchsorted(lat, tile_lat[0])),
            int(np.searchsorted(lat, tile_lat[-1])) + 1,
        )
        col = slice(
            int(np.searchsorted(lon, tile_lon[0])),
            int(np.searchsorted(lon, tile_lon[-1])) + 1,
        )
        if not np.array_equal(lat[row], tile_lat) or not np.array_equal(
            lon[col], tile_lon
        ):
            raise ValueError(
                f"{token} region {region} does not map onto a contiguous global slice."
            )
        cover[row, col] += 1
        slices[region] = (row, col)
    if not np.all(cover == 1):
        raise ValueError(
            f"{token} regions do not tile the global grid exactly once "
            f"(found multiplicities {sorted(np.unique(cover).tolist())})."
        )
    return GlobalGrid(token, lat, lon, slices)


def grid_signature(grid: GlobalGrid) -> dict[str, object]:
    """Return a compact, comparable description of a global grid."""
    from hashlib import sha256

    digest = sha256()
    for axis in (grid.lat, grid.lon):
        digest.update(np.ascontiguousarray(axis, dtype=np.float64).tobytes())
    return {
        "token": grid.token,
        "lat_size": int(grid.lat.size),
        "lon_size": int(grid.lon.size),
        "axes_sha256": digest.hexdigest(),
    }


def scatter_tiles(
    grid: GlobalGrid,
    tile_values: dict[str, np.ndarray],
    *,
    fill: float = np.nan,
    dtype: type = np.float32,
) -> np.ndarray:
    """Place per-region 2-D tiles into one global array.

    Regions absent from ``tile_values`` stay at ``fill``.  Callers never read
    those cells: a position whose footprint touches a missing region yields a
    null coverage tuple rather than a partially measured one.
    """
    out = np.full(grid.shape, fill, dtype=dtype)
    for region, values in tile_values.items():
        row, col = grid.slices[region]
        expected = (row.stop - row.start, col.stop - col.start)
        if values.shape != expected:
            raise ValueError(
                f"{grid.token} tile {region} has shape {values.shape}, expected {expected}."
            )
        out[row, col] = values
    return out


def _segment_columns(lon_axis: np.ndarray, box: GeoBox) -> tuple[tuple[int, int], ...]:
    """Return inclusive-endpoint column spans reproducing ``crop_dense``.

    ``crop_dense`` selects with ``.sel(lat=slice(a, b), lon=slice(c, d))``,
    whose endpoints are inclusive.  ``searchsorted`` with ``left``/``right``
    reproduces that on a sorted ascending axis.  A wrapped footprint yields two
    spans, matching :meth:`GeoBox.segments`.
    """
    spans = []
    for segment in box.segments():
        start = int(np.searchsorted(lon_axis, segment.lon_min, side="left"))
        stop = int(np.searchsorted(lon_axis, segment.lon_max, side="right"))
        if stop > start:
            spans.append((start, stop))
    return tuple(spans)


class PositionPlan:
    """Date-invariant crop geometry and static denominators for one position.

    ``ocean`` is cached here because
    :meth:`OceanMaskArtifact.nearest_with_domain` costs milliseconds per call
    and depends only on the footprint, not the date.
    """

    __slots__ = (
        "row",
        "cols",
        "ocean",
        "footprint_cells",
        "ocean_cells",
        "regions",
    )

    def __init__(
        self,
        row: tuple[int, int],
        cols: tuple[tuple[int, int], ...],
        ocean: np.ndarray,
        footprint_cells: int,
        ocean_cells: int,
        regions: frozenset[str],
    ) -> None:
        self.row = row
        self.cols = cols
        self.ocean = ocean
        self.footprint_cells = footprint_cells
        self.ocean_cells = ocean_cells
        self.regions = regions

    def crop(self, values: np.ndarray) -> np.ndarray:
        """Return the footprint crop of a global array."""
        row_start, row_stop = self.row
        if len(self.cols) == 1:
            start, stop = self.cols[0]
            return values[row_start:row_stop, start:stop]
        return np.concatenate(
            [values[row_start:row_stop, start:stop] for start, stop in self.cols],
            axis=1,
        )


def footprint_regions(box: GeoBox) -> frozenset[str]:
    """Return the Core regions a footprint intersects, per released bounds."""
    return frozenset(
        region
        for region, bounds in _REGION_BOUNDS.items()
        if _intersects(box, bounds)
    )


def build_position_plan(
    grid: GlobalGrid,
    *,
    patch_size: PatchSize,
    centre_lon: float,
    centre_lat: float,
    ocean_mask: OceanMaskArtifact,
) -> PositionPlan:
    """Build the crop plan and static denominators for one position.

    The two static counts are exactly what
    :func:`ocean_taco.sampling.coverage.native_footprint_counts` returns, so
    that released function is never called during a bulk build.
    """
    box = patch_size.footprint(centre_lon, centre_lat)
    row = (
        int(np.searchsorted(grid.lat, box.lat_min, side="left")),
        int(np.searchsorted(grid.lat, box.lat_max, side="right")),
    )
    cols = _segment_columns(grid.lon, box)
    lat_values = grid.lat[row[0] : row[1]]
    if cols:
        lon_values = np.concatenate([grid.lon[a:b] for a, b in cols])
    else:
        lon_values = np.empty((0,), dtype=np.float64)
    ocean, in_domain = ocean_mask.nearest_with_domain(lat_values, lon_values)
    if not in_domain.all():
        raise ValueError(
            "Coverage denominator footprint leaves the frozen ocean-mask domain."
        )
    return PositionPlan(
        row=row,
        cols=cols,
        ocean=ocean,
        footprint_cells=int(lat_values.size * lon_values.size),
        ocean_cells=int(ocean.sum()),
        regions=footprint_regions(box),
    )


def measure_dense(
    plan: PositionPlan,
    values: np.ndarray,
    n_obs: np.ndarray | None,
) -> DenseCoverage:
    """Count finite decoded cells exactly as ``measure_dense_coverage`` does."""
    crop = plan.crop(values)
    valid = np.isfinite(crop)
    valid_cells = int(valid.sum())
    valid_ocean_cells = int((valid & plan.ocean).sum())
    if n_obs is None:
        return DenseCoverage(valid_cells, valid_ocean_cells, None)
    total = np.nansum(plan.crop(n_obs))
    return DenseCoverage(valid_cells, valid_ocean_cells, int(total))


def dense_coverage_for(
    plan: PositionPlan,
    present_regions: frozenset[str],
    values: np.ndarray | None,
    n_obs: np.ndarray | None,
) -> DenseCoverage:
    """Return measured coverage, or the null tuple for an incomplete closure.

    A footprint spanning several regions is measurable only when *every*
    region it touches supplied an asset.  Measuring the available part would
    silently report a smaller swath as a real observation.
    """
    if values is None or not plan.regions <= present_regions:
        return unavailable_dense_coverage()
    return measure_dense(plan, values, n_obs)


class ArgoDay:
    """One date's Argo profiles, pre-filtered and vectorised for counting.

    The released :func:`measure_argo_profile_count` parses every timestamp
    through a Python list comprehension on each call.  Parsing once per date
    and reusing the arrays is the only change; the selection predicate and the
    ``platform:cycle`` uniqueness rule are identical.
    """

    __slots__ = ("lat", "lon", "keys", "regions")

    def __init__(
        self,
        lat: np.ndarray,
        lon: np.ndarray,
        keys: np.ndarray,
        regions: frozenset[str],
    ) -> None:
        self.lat = lat
        self.lon = lon
        self.keys = keys
        self.regions = regions

    def count(self, patch_size: PatchSize, centre_lon: float, centre_lat: float) -> int:
        """Count unique profiles inside one footprint."""
        box = patch_size.footprint(centre_lon, centre_lat)
        selected = (self.lat >= box.lat_min) & (self.lat <= box.lat_max)
        if box.wraps_antimeridian:
            selected &= (self.lon >= box.lon_min) | (self.lon <= box.lon_max)
        else:
            selected &= (self.lon >= box.lon_min) & (self.lon <= box.lon_max)
        if not selected.any():
            return 0
        return int(np.unique(self.keys[selected]).size)


def build_argo_day(
    tile_datasets: dict[str, object], date: str
) -> ArgoDay:
    """Concatenate one date's Argo tiles and pre-compute the selection arrays.

    Mirrors :func:`ocean_taco.sampling.coverage.measure_argo_profile_count`:
    the same required fields, the same half-open ``[date, date + 1 day)``
    window, the same canonical longitude mapping, and the same
    ``PLATFORM_NUMBER:CYCLE_NUMBER`` uniqueness rule.
    """
    from datetime import timedelta

    from ocean_taco.geobox import _utc_datetime

    required = ("lat", "lon", "time", "PLATFORM_NUMBER", "CYCLE_NUMBER")
    lats, lons, keys = [], [], []
    for region, dataset in sorted(tile_datasets.items()):
        for field in required:
            if field not in dataset:
                raise ValueError(
                    f"Argo coverage asset lacks required field {field!r}."
                )
        lats.append(np.asarray(dataset["lat"].values, dtype=np.float64))
        lons.append(canonical_lon(np.asarray(dataset["lon"].values, dtype=np.float64)))
        platform = np.asarray(dataset["PLATFORM_NUMBER"].values).astype(str)
        cycle = np.asarray(dataset["CYCLE_NUMBER"].values).astype(str)
        keys.append(np.char.add(np.char.add(platform, ":"), cycle))
    lat = np.concatenate(lats) if lats else np.empty((0,), dtype=np.float64)
    lon = np.concatenate(lons) if lons else np.empty((0,), dtype=np.float64)
    key = (
        np.concatenate(keys) if keys else np.empty((0,), dtype="<U1")
    )
    time_values = np.concatenate(
        [
            np.asarray(dataset["time"].values, dtype="datetime64[ns]")
            for _, dataset in sorted(tile_datasets.items())
        ]
    ) if tile_datasets else np.empty((0,), dtype="datetime64[ns]")
    target = _utc_datetime(date)
    start = np.datetime64(target.replace(tzinfo=None), "ns")
    end = np.datetime64((target + timedelta(days=1)).replace(tzinfo=None), "ns")
    keep = (
        np.isfinite(lat)
        & np.isfinite(lon)
        & (time_values >= start)
        & (time_values < end)
    )
    return ArgoDay(
        lat=lat[keep],
        lon=lon[keep],
        keys=key[keep],
        regions=frozenset(tile_datasets),
    )


__all__ = [
    "ArgoDay",
    "COVERAGE_KEY",
    "DENSE_TOKENS",
    "GlobalGrid",
    "POINT_TOKEN",
    "PositionPlan",
    "REGIONS",
    "build_argo_day",
    "build_global_grid",
    "build_position_plan",
    "canonical_lon",
    "check_units",
    "dense_coverage_for",
    "footprint_regions",
    "grid_signature",
    "measure_dense",
    "scatter_tiles",
]
