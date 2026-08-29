"""Named spatial and temporal contracts for OceanTACO.

The public API intentionally uses named objects rather than coordinate tuples.
That removes the two incompatible bbox orderings that existed in the legacy
loader and makes antimeridian behaviour explicit at construction time.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from math import cos, isfinite, radians
from typing import Literal

KM_PER_DEGREE_LATITUDE = 111.32


def _utc_datetime(value: datetime | str) -> datetime:
    """Return ``value`` as a timezone-aware UTC datetime.

    A trailing ``Z`` is accepted.  Naive datetimes are interpreted as UTC so
    serialised query manifests do not acquire a machine-local timezone.
    """
    if isinstance(value, str):
        value = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def utc_isoformat(value: datetime | str) -> str:
    """Canonical UTC ISO-8601 representation used in manifests."""
    return _utc_datetime(value).isoformat(timespec="microseconds").replace("+00:00", "Z")


@dataclass(frozen=True, slots=True)
class GeoBox:
    """A geographic rectangle in the canonical ``[-180, 180]`` convention.

    ``wraps_antimeridian=True`` is required for an interval whose longitude
    bounds are inverted.  The flag is deliberately not inferred: accidental
    inverted arguments must fail instead of returning a plausible empty crop.
    """

    lon_min: float
    lon_max: float
    lat_min: float
    lat_max: float
    wraps_antimeridian: bool = False

    def __post_init__(self) -> None:
        values = (self.lon_min, self.lon_max, self.lat_min, self.lat_max)
        if not all(isfinite(value) for value in values):
            raise ValueError("GeoBox coordinates must be finite.")
        if not -180.0 <= self.lon_min <= 180.0 or not -180.0 <= self.lon_max <= 180.0:
            raise ValueError("GeoBox longitudes must lie in [-180, 180].")
        if not -90.0 <= self.lat_min <= self.lat_max <= 90.0:
            raise ValueError("GeoBox latitudes must be ordered and lie in [-90, 90].")
        if self.wraps_antimeridian and self.lon_min <= self.lon_max:
            raise ValueError(
                "A wrapped GeoBox must have lon_min > lon_max; do not mark an ordinary box as wrapped."
            )
        if not self.wraps_antimeridian and self.lon_min > self.lon_max:
            raise ValueError(
                "Inverted longitude bounds require wraps_antimeridian=True."
            )

    @property
    def longitude_width_degrees(self) -> float:
        """Requested longitudinal width, preserving a wrapped extent."""
        if self.wraps_antimeridian:
            return (180.0 - self.lon_min) + (self.lon_max + 180.0)
        return self.lon_max - self.lon_min

    @property
    def latitude_height_degrees(self) -> float:
        """Requested latitudinal extent."""
        return self.lat_max - self.lat_min

    @property
    def centre_lat(self) -> float:
        """Latitude of the box centre."""
        return (self.lat_min + self.lat_max) / 2.0

    @property
    def centre_lon(self) -> float:
        """Longitude of the box centre in the canonical convention."""
        lon = self.lon_min + self.longitude_width_degrees / 2.0
        return lon - 360.0 if lon > 180.0 else lon

    def segments(self) -> tuple[GeoBox, ...]:
        """Return non-wrapped selection boxes in query order.

        This method is intentionally the only public decomposition route used
        by retrieval code.  A caller never has to rediscover wrap semantics.
        """
        if not self.wraps_antimeridian:
            return (self,)
        return (
            GeoBox(self.lon_min, 180.0, self.lat_min, self.lat_max),
            GeoBox(-180.0, self.lon_max, self.lat_min, self.lat_max),
        )

    def unwrap_longitudes(self, longitudes):
        """Map a query-ordered wrapped longitude vector onto a continuous axis."""
        if not self.wraps_antimeridian:
            return longitudes
        import numpy as np

        values = np.asarray(longitudes).copy()
        values[values < self.lon_min] += 360.0
        return values

    def to_dict(self) -> dict[str, float | bool]:
        """Return a manifest-safe representation."""
        return {
            "lon_min": self.lon_min,
            "lon_max": self.lon_max,
            "lat_min": self.lat_min,
            "lat_max": self.lat_max,
            "wraps_antimeridian": self.wraps_antimeridian,
        }


@dataclass(frozen=True, slots=True)
class TimeRange:
    """Closed UTC interval used for explicit temporal selection."""

    start: datetime | str
    end: datetime | str

    def __post_init__(self) -> None:
        start = _utc_datetime(self.start)
        end = _utc_datetime(self.end)
        if end < start:
            raise ValueError("TimeRange end must not precede start.")
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "end", end)

    def to_dict(self) -> dict[str, str]:
        """Return canonical UTC boundaries."""
        return {"start": utc_isoformat(self.start), "end": utc_isoformat(self.end)}


@dataclass(frozen=True, slots=True)
class Query:
    """Science-retrieval query: one :class:`GeoBox` and one closed time range."""

    box: GeoBox
    time: TimeRange

    def to_dict(self) -> dict[str, object]:
        """Return a manifest-safe representation."""
        return {"box": self.box.to_dict(), "time": self.time.to_dict()}


@dataclass(frozen=True, slots=True)
class PatchSize:
    """A square patch size with an explicit unit and no implicit default."""

    value: float
    unit: Literal["km", "deg"]

    def __post_init__(self) -> None:
        if not isfinite(self.value) or self.value <= 0:
            raise ValueError("PatchSize value must be a finite positive number.")
        if self.unit not in {"km", "deg"}:
            raise ValueError("PatchSize unit must be 'km' or 'deg'.")

    def to_degrees(self, centre_lat: float) -> tuple[float, float]:
        """Return ``(longitude_degrees, latitude_degrees)`` at ``centre_lat``."""
        if not -90.0 <= centre_lat <= 90.0:
            raise ValueError("centre_lat must lie in [-90, 90].")
        if self.unit == "deg":
            return self.value, self.value
        longitude_scale = KM_PER_DEGREE_LATITUDE * cos(radians(centre_lat))
        if longitude_scale <= 0.0:
            raise ValueError("A kilometre PatchSize cannot be represented at a pole.")
        return self.value / longitude_scale, self.value / KM_PER_DEGREE_LATITUDE

    def footprint(self, centre_lon: float, centre_lat: float) -> GeoBox:
        """Construct the full-size geographic footprint around a centre.

        The caller is responsible for restricting centres so the latitude
        bounds remain inside the supported domain.  Longitude wraps rather
        than clamps, which preserves the stated patch size at the dateline.
        """
        lon_width, lat_height = self.to_degrees(centre_lat)
        if lon_width > 360.0:
            raise ValueError("Patch footprint cannot be wider than one global longitude circuit.")
        lat_min, lat_max = centre_lat - lat_height / 2.0, centre_lat + lat_height / 2.0
        if lat_min < -90.0 or lat_max > 90.0:
            raise ValueError("Patch footprint extends beyond a pole.")
        lon_min, lon_max = centre_lon - lon_width / 2.0, centre_lon + lon_width / 2.0
        if lon_min < -180.0:
            return GeoBox(lon_min + 360.0, lon_max, lat_min, lat_max, True)
        if lon_max > 180.0:
            return GeoBox(lon_min, lon_max - 360.0, lat_min, lat_max, True)
        return GeoBox(lon_min, lon_max, lat_min, lat_max)

    def to_dict(self) -> dict[str, float | str]:
        """Return a manifest-safe representation."""
        return {"value": self.value, "unit": self.unit}


@dataclass(frozen=True, slots=True)
class PatchSpec:
    """Logical ML sample specification, independent of rendering choices."""

    centre_lon: float
    centre_lat: float
    patch_size: PatchSize
    anchor_time: datetime | str
    context_start_offset_days: int
    context_end_offset_days: int
    relation: Literal["same_time", "forecast"] = "same_time"
    target_lead_days: int = 0

    def __post_init__(self) -> None:
        if not isfinite(self.centre_lon) or not -180.0 <= self.centre_lon <= 180.0:
            raise ValueError("PatchSpec centre_lon must lie in [-180, 180].")
        if not isfinite(self.centre_lat) or not -90.0 <= self.centre_lat <= 90.0:
            raise ValueError("PatchSpec centre_lat must lie in [-90, 90].")
        if self.context_end_offset_days < self.context_start_offset_days:
            raise ValueError("Context offsets must form an ordered contiguous range.")
        if self.relation not in {"same_time", "forecast"}:
            raise ValueError("relation must be 'same_time' or 'forecast'.")
        if self.relation == "same_time" and self.target_lead_days != 0:
            raise ValueError("same_time PatchSpecs must have target_lead_days=0.")
        if self.relation == "forecast" and self.target_lead_days <= 0:
            raise ValueError("forecast PatchSpecs require a positive target_lead_days.")
        object.__setattr__(self, "anchor_time", _utc_datetime(self.anchor_time))

    @property
    def footprint(self) -> GeoBox:
        """Spatial footprint implied by the patch size and centre."""
        return self.patch_size.footprint(self.centre_lon, self.centre_lat)

    @property
    def context(self) -> TimeRange:
        """Closed context interval relative to the anchor time."""
        return TimeRange(
            self.anchor_time + timedelta(days=self.context_start_offset_days),
            self.anchor_time + timedelta(days=self.context_end_offset_days),
        )

    @property
    def target_time(self) -> datetime:
        """Prediction time for this task relation."""
        return self.anchor_time + timedelta(days=self.target_lead_days)

    def to_dict(self) -> dict[str, object]:
        """Return the canonical logical sample payload."""
        return {
            "centre_lon": self.centre_lon,
            "centre_lat": self.centre_lat,
            "patch_size": self.patch_size.to_dict(),
            "footprint": self.footprint.to_dict(),
            "anchor_time": utc_isoformat(self.anchor_time),
            "context_start_offset_days": self.context_start_offset_days,
            "context_end_offset_days": self.context_end_offset_days,
            "relation": self.relation,
            "target_lead_days": self.target_lead_days,
        }
