"""Train and Eval Query Generation for OceanTACODataset."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

import cartopy.io.shapereader as shpreader
import numpy as np
import pandas as pd
from shapely.geometry import Point
from shapely.ops import unary_union
from shapely.prepared import prep

# =============================================================================
# Query Representation
# =============================================================================

# https://github.com/torchgeo/torchgeo/blob/8b9bf3e10555b7e0087368c05ac9792389d00890/torchgeo/datasets/utils.py#L42
GeoSlice = (  # noqa: UP040
    slice | tuple[slice] | tuple[slice, slice] | tuple[slice, slice, slice]
)


@dataclass
class Query:
    """Spatial-temporal query specification."""

    lon_min: float
    lon_max: float
    lat_min: float
    lat_max: float
    time_start: str | pd.Timestamp
    time_end: str | pd.Timestamp

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        return (self.lon_min, self.lon_max, self.lat_min, self.lat_max)

    def to_geoslice(self):
        """Convert to GeoSlice format for dataset indexing."""
        t_start = (
            pd.Timestamp(self.time_start)
            if isinstance(self.time_start, str)
            else self.time_start
        )
        t_end = (
            pd.Timestamp(self.time_end)
            if isinstance(self.time_end, str)
            else self.time_end
        )

        return (
            slice(self.lon_min, self.lon_max),
            slice(self.lat_min, self.lat_max),
            slice(t_start, t_end),
        )

    def to_dict(self) -> dict:
        return {
            "lon_min": self.lon_min,
            "lon_max": self.lon_max,
            "lat_min": self.lat_min,
            "lat_max": self.lat_max,
            "time_start": str(self.time_start),
            "time_end": str(self.time_end),
        }

    @classmethod
    def from_dict(cls, d: dict) -> Query:
        return cls(**d)


# =============================================================================
# Patch Size Specification
# =============================================================================


@dataclass
class PatchSize:
    """Patch size with unit conversion support."""

    value: float
    unit: Literal["deg", "km"] = "deg"

    def to_degrees(self, center_lat: float = 0.0) -> tuple[float, float]:
        """Convert to (lon_degrees, lat_degrees) accounting for latitude."""
        if self.unit == "deg":
            return (self.value, self.value)

        # km to degrees: ~111 km per degree latitude
        lat_deg = self.value / 111.0
        lon_deg = self.value / (111.0 * max(np.cos(np.radians(center_lat)), 0.1))
        return (lon_deg, lat_deg)

    def to_km(self, center_lat: float = 0.0) -> float:
        """Convert to approximate km."""
        if self.unit == "km":
            return self.value
        return self.value * 111.0

    def __str__(self) -> str:
        return f"{self.value}{self.unit}"


# =============================================================================
# Land Mask for Ocean Validation
# =============================================================================


class LandMask:
    """Pre-computed land/ocean mask for fast land fraction queries.

    Resolution: 0.25° (1440 x 720 grid)
    """

    RESOLUTION = 0.25
    N_LON = 1440
    N_LAT = 720

    def __init__(self, path: str | Path | None = None):
        """Load or generate land mask."""
        self.lons = np.linspace(
            -180 + self.RESOLUTION / 2, 180 - self.RESOLUTION / 2, self.N_LON
        )
        self.lats = np.linspace(
            -90 + self.RESOLUTION / 2, 90 - self.RESOLUTION / 2, self.N_LAT
        )

        if path and Path(path).exists():
            self.mask = np.load(path)
        else:
            self.mask = self._generate_mask()
            if path:
                Path(path).parent.mkdir(parents=True, exist_ok=True)
                np.save(path, self.mask)

    def _generate_mask(self) -> np.ndarray:
        """Generate land mask using cartopy Natural Earth data."""
        print("Generating land mask from Natural Earth...")
        land_shp = shpreader.natural_earth(
            resolution="110m", category="physical", name="land"
        )
        land_geoms = list(shpreader.Reader(land_shp).geometries())
        land_union = unary_union(land_geoms)
        land_prepared = prep(land_union)

        mask = np.zeros((self.N_LAT, self.N_LON), dtype=bool)
        for i, lat in enumerate(self.lats):
            for j, lon in enumerate(self.lons):
                mask[i, j] = land_prepared.contains(Point(lon, lat))
            if i % 100 == 0:
                print(f"  Processing latitude {i}/{self.N_LAT}")

        return mask

    def get_land_fraction(self, bbox: tuple[float, float, float, float]) -> float:
        """Get land fraction for bbox (lon_min, lon_max, lat_min, lat_max)."""
        lon_min, lon_max, lat_min, lat_max = bbox

        j_min = max(0, int((lon_min + 180) / self.RESOLUTION))
        j_max = min(self.N_LON, int((lon_max + 180) / self.RESOLUTION) + 1)
        i_min = max(0, int((lat_min + 90) / self.RESOLUTION))
        i_max = min(self.N_LAT, int((lat_max + 90) / self.RESOLUTION) + 1)

        if j_max <= j_min or i_max <= i_min:
            return 0.0

        subset = self.mask[i_min:i_max, j_min:j_max]
        return float(subset.mean()) if subset.size > 0 else 0.0

    def is_ocean(self, bbox: tuple, max_land_fraction: float = 0.3) -> bool:
        """Check if bbox is predominantly ocean."""
        return self.get_land_fraction(bbox) <= max_land_fraction


# =============================================================================
# Query Generator
# =============================================================================


class QueryGenerator:
    """Generate queries for training (random) and evaluation (grid)."""

    def __init__(self, land_mask_path: str | Path | None = None):
        """Initialize with optional land mask."""
        self.land_mask = LandMask(land_mask_path) if land_mask_path else None

    def generate_training_queries(
        self,
        n_queries: int,
        patch_size: PatchSize | float,
        date_range: tuple[str, str],
        bbox_constraint: tuple[float, float, float, float] = (-180, 180, -60, 60),
        time_window_days: int = 1,
        max_land_fraction: float = 0.3,
        seed: int = 42,
        oversample_factor: float = 2.0,
        verbose: bool = True,
        max_spatial_overlap: float = 1.0,
    ) -> list[Query]:
        """Generate random training queries over ocean regions.

        Args:
            n_queries: Number of queries to generate.
            patch_size: Spatial extent (PatchSize or degrees).
            date_range: (start_date, end_date) strings.
            bbox_constraint: Region to sample from (lon_min, lon_max, lat_min, lat_max).
            time_window_days: Temporal extent of each query.
            max_land_fraction: Maximum allowed land fraction (0-1).
            seed: Random seed for reproducibility.
            oversample_factor: Generate extra candidates to account for rejections.
            verbose: Print progress.
            max_spatial_overlap: Maximum allowed IoU (0-1) with existing queries.

        Returns:
            List of Query objects.
        """
        if isinstance(patch_size, (int, float)):
            patch_size = PatchSize(patch_size, "deg")

        rng = np.random.default_rng(seed)
        lon_min, lon_max, lat_min, lat_max = bbox_constraint

        # Generate date range
        dates = pd.date_range(date_range[0], date_range[1], freq="D")
        if time_window_days > 1:
            dates = dates[: -time_window_days + 1]

        if len(dates) == 0:
            raise ValueError("Date range too short for time window")

        valid_queries = []
        # Cache for fast overlap checking: start_date -> list of (lon_min, lon_max, lat_min, lat_max, start_ts, end_ts)
        valid_cache = defaultdict(list)

        n_candidates = int(n_queries * oversample_factor)
        n_checked = 0
        max_attempts = n_candidates * 10  # Increased for strict overlap constraints

        if verbose:
            print(f"Generating {n_queries} training queries...")

        while len(valid_queries) < n_queries and n_checked < max_attempts:
            # Batch generation for efficiency
            batch_size = min(1000, (n_queries - len(valid_queries)) * 2)

            center_lats = rng.uniform(lat_min, lat_max, batch_size)
            center_lons = rng.uniform(lon_min, lon_max, batch_size)
            date_indices = rng.integers(0, len(dates), batch_size)

            for i in range(batch_size):
                if len(valid_queries) >= n_queries:
                    break

                n_checked += 1
                center_lat = center_lats[i]
                center_lon = center_lons[i]

                # Compute bbox
                lon_size, lat_size = patch_size.to_degrees(center_lat)
                bbox = (
                    float(max(-180, center_lon - lon_size / 2)),
                    float(min(180, center_lon + lon_size / 2)),
                    float(max(-90, center_lat - lat_size / 2)),
                    float(min(90, center_lat + lat_size / 2)),
                )

                # Land check
                if self.land_mask and not self.land_mask.is_ocean(
                    bbox, max_land_fraction
                ):
                    continue

                # Time range
                start_date = dates[date_indices[i]]
                end_date = start_date + pd.Timedelta(days=time_window_days - 1)

                # Overlap check
                if max_spatial_overlap < 1.0:
                    if self._check_overlap_fast(
                        bbox, start_date, end_date, valid_cache, max_spatial_overlap
                    ):
                        continue

                valid_queries.append(
                    Query(
                        lon_min=bbox[0],
                        lon_max=bbox[1],
                        lat_min=bbox[2],
                        lat_max=bbox[3],
                        time_start=start_date.strftime("%Y-%m-%d"),
                        time_end=end_date.strftime("%Y-%m-%d"),
                    )
                )
                valid_cache[start_date].append((*bbox, start_date, end_date))

        if verbose:
            print(f"Generated {len(valid_queries)} queries from {n_checked} candidates")

        return valid_queries

    def _check_overlap_fast(
        self,
        bbox: tuple[float, float, float, float],
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        valid_cache: dict[pd.Timestamp, list[tuple]],
        max_overlap: float,
    ) -> bool:
        """Check if candidate overlaps with any existing query > max_overlap."""
        c_lon_min, c_lon_max, c_lat_min, c_lat_max = bbox
        c_area = (c_lon_max - c_lon_min) * (c_lat_max - c_lat_min)

        # Calculate time window to check relevant start dates
        # A query Q (start, end) overlaps with C (start, end) if Q.start <= C.end and Q.end >= C.start
        # Assuming Q and C have similar duration W:
        # Q.start must be in [C.start - W + 1, C.start + W - 1]

        days_window = (end_date - start_date).days + 1
        relevant_start_dates = [
            start_date + pd.Timedelta(days=d)
            for d in range(-days_window + 1, days_window)
        ]

        for check_date in relevant_start_dates:
            if check_date in valid_cache:
                for (
                    q_lon_min,
                    q_lon_max,
                    q_lat_min,
                    q_lat_max,
                    q_start,
                    q_end,
                ) in valid_cache[check_date]:
                    # Time check (double check to be safe)
                    if start_date <= q_end and q_start <= end_date:
                        # Spatial check
                        xi1 = max(c_lon_min, q_lon_min)
                        yi1 = max(c_lat_min, q_lat_min)
                        xi2 = min(c_lon_max, q_lon_max)
                        yi2 = min(c_lat_max, q_lat_max)

                        if xi2 > xi1 and yi2 > yi1:
                            inter_area = (xi2 - xi1) * (yi2 - yi1)
                            q_area = (q_lon_max - q_lon_min) * (q_lat_max - q_lat_min)
                            union_area = c_area + q_area - inter_area

                            if (
                                union_area > 0
                                and (inter_area / union_area) > max_overlap
                            ):
                                return True
        return False

    def generate_eval_queries(
        self,
        bbox: tuple[float, float, float, float],
        patch_size: PatchSize | float,
        date_range: tuple[str, str],
        spatial_overlap: float = 0.0,
        temporal_stride_days: int = 1,
        time_window_days: int = 1,
        max_land_fraction: float = 0.5,
        verbose: bool = True,
    ) -> list[Query]:
        """Generate systematic grid of evaluation queries.

        Args:
            bbox: Region to cover (lon_min, lon_max, lat_min, lat_max).
            patch_size: Spatial extent of each query.
            date_range: (start_date, end_date) strings.
            spatial_overlap: Overlap fraction (0 = no overlap, 0.5 = 50% overlap).
            temporal_stride_days: Days between query start times.
            time_window_days: Temporal extent of each query.
            max_land_fraction: Skip patches with more land than this.
            verbose: Print progress.

        Returns:
            List of Query objects covering the region.
        """
        if isinstance(patch_size, (int, float)):
            patch_size = PatchSize(patch_size, "deg")

        lon_min, lon_max, lat_min, lat_max = bbox

        # Compute patch size at center latitude
        center_lat = (lat_min + lat_max) / 2
        lon_size, lat_size = patch_size.to_degrees(center_lat)

        # Compute strides
        stride_lon = lon_size * (1 - spatial_overlap)
        stride_lat = lat_size * (1 - spatial_overlap)

        # Generate spatial grid
        lon_starts = np.arange(lon_min, lon_max - lon_size + 0.001, stride_lon)
        # Ensure full coverage: if last patch doesn't reach the edge, add one aligned to the edge
        if len(lon_starts) > 0 and (lon_starts[-1] + lon_size) < (lon_max - 0.001):
            lon_starts = np.append(lon_starts, lon_max - lon_size)

        lat_starts = np.arange(lat_min, lat_max - lat_size + 0.001, stride_lat)
        if len(lat_starts) > 0 and (lat_starts[-1] + lat_size) < (lat_max - 0.001):
            lat_starts = np.append(lat_starts, lat_max - lat_size)

        # Generate temporal grid
        dates = pd.date_range(date_range[0], date_range[1], freq="D")
        if time_window_days > 1:
            dates = dates[: -time_window_days + 1]
        date_starts = dates[::temporal_stride_days]

        if verbose:
            n_total = len(lon_starts) * len(lat_starts) * len(date_starts)
            print(
                f"Generating eval grid: {len(lon_starts)}x{len(lat_starts)}x{len(date_starts)} = {n_total} candidates"
            )

        valid_queries = []

        for lon_start in lon_starts:
            for lat_start in lat_starts:
                # Recompute size for local latitude
                local_lon_size, local_lat_size = patch_size.to_degrees(
                    lat_start + lat_size / 2
                )

                query_bbox = (
                    float(lon_start),
                    float(lon_start + local_lon_size),
                    float(lat_start),
                    float(lat_start + local_lat_size),
                )

                # Land check
                if self.land_mask and not self.land_mask.is_ocean(
                    query_bbox, max_land_fraction
                ):
                    continue

                for date_start in date_starts:
                    end_date = date_start + pd.Timedelta(days=time_window_days - 1)

                    valid_queries.append(
                        Query(
                            lon_min=query_bbox[0],
                            lon_max=query_bbox[1],
                            lat_min=query_bbox[2],
                            lat_max=query_bbox[3],
                            time_start=date_start.strftime("%Y-%m-%d"),
                            time_end=end_date.strftime("%Y-%m-%d"),
                        )
                    )

        # Sort by date to ensure chronological evaluation
        valid_queries.sort(key=lambda x: x.time_start)

        if verbose:
            print(f"Generated {len(valid_queries)} valid eval queries")

        return valid_queries

    # -------------------------------------------------------------------------
    # Save / Load
    # -------------------------------------------------------------------------

    @staticmethod
    def save_queries(
        queries: list[Query], path: str | Path, metadata: dict | None = None
    ):
        """Save queries to parquet with JSON metadata."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        df = pd.DataFrame([q.to_dict() for q in queries])
        df.to_parquet(path.with_suffix(".parquet"), index=False)

        # Save metadata sidecar
        meta = {
            "n_queries": len(queries),
            "generated_at": datetime.now().isoformat(),
            **(metadata or {}),
        }

        with open(path.with_suffix(".json"), "w") as f:
            json.dump(meta, f, indent=2, default=str)

        print(f"Saved {len(queries)} queries to {path}")

    @staticmethod
    def load_queries(path: str | Path) -> tuple[list[Query], dict]:
        """Load queries from parquet file."""
        path = Path(path)
        df = pd.read_parquet(path.with_suffix(".parquet"))

        queries = [Query.from_dict(row.to_dict()) for _, row in df.iterrows()]

        metadata = {}
        json_path = path.with_suffix(".json")
        if json_path.exists():
            with open(json_path) as f:
                metadata = json.load(f)

        return queries, metadata
