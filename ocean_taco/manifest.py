"""Published QuerySet and PatchSet artifacts.

The query-set boundary is deliberately factored.  A set stores one position
table, one per-position/per-date coverage table, and one asset-identity table;
it never serialises a rendered sample or a pre-qualified population.
"""

from __future__ import annotations

import json
import os
import shutil
import uuid
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime
from hashlib import sha256
from math import isfinite
from pathlib import Path
from typing import Any

from .geobox import PatchSize, utc_isoformat

SCHEMA_VERSION = "queryset/v1"
TABLE_FILENAMES = {
    "positions": "positions.parquet",
    "coverage": "coverage.parquet",
    "assets": "assets.parquet",
}
_NON_SEMANTIC_HEADER_KEYS = {"queryset_id", "created_at", "notes", "report_paths"}
_HEADER_KEYS = {
    "schema_version",
    "patch_size",
    "kind",
    "grid_spacing_km",
    "grid_id",
    "dataset_revision",
    "catalog_sha256",
    "registry_sha256",
    "source_records_sha256",
    "ocean_mask_id",
    "ocean_mask_sha256",
    "dates",
    "date_sha256",
    "tokens",
    "parquet_profile",
    "table_sha256",
    "queryset_id",
    "code_commit",
    "environment_lock_hash",
    "coverage_rules",
    "grid_validation",
    "created_at",
    "notes",
    "report_paths",
}
_REQUIRED_HEADER_KEYS = (
    _HEADER_KEYS
    - _NON_SEMANTIC_HEADER_KEYS
    - {"table_sha256", "queryset_id", "coverage_rules", "grid_validation"}
)
_TIME_KEYS = {"anchor_time", "time_start", "time_end", "target_time", "start", "end"}


def _normalise(value: Any) -> Any:
    """Convert a value to canonical JSON-compatible content."""
    if hasattr(value, "to_dict"):
        return _normalise(value.to_dict())
    if is_dataclass(value):
        return _normalise(asdict(value))
    if isinstance(value, datetime):
        return utc_isoformat(value)
    if isinstance(value, Mapping):
        normalised: dict[str, Any] = {}
        for key, item in sorted(value.items(), key=lambda item: str(item[0])):
            name = str(key)
            if name in _TIME_KEYS and isinstance(item, str):
                try:
                    normalised[name] = utc_isoformat(item)
                    continue
                except ValueError:
                    pass
            normalised[name] = _normalise(item)
        return normalised
    if isinstance(value, (list, tuple)):
        return [_normalise(item) for item in value]
    if hasattr(value, "item"):
        return _normalise(value.item())
    if isinstance(value, float):
        if not isfinite(value):
            raise ValueError("Canonical manifests forbid NaN and infinity.")
        return 0.0 if value == 0.0 else float(f"{value:.12f}")
    if value is None or isinstance(value, (str, int, bool)):
        return value
    raise TypeError(f"Cannot canonicalise {type(value).__name__} in a manifest.")


def canonical_json(value: Any) -> bytes:
    """Return deterministic UTF-8 JSON with sorted keys and no whitespace."""
    return json.dumps(
        _normalise(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def content_sha256(value: Any) -> str:
    """Return the SHA-256 digest of canonical JSON content."""
    return sha256(canonical_json(value)).hexdigest()


def patch_id(
    *,
    patch_size: PatchSize | Mapping[str, Any],
    centre_lon: float,
    centre_lat: float,
    anchor_time: datetime | str,
) -> str:
    """Return the stable identity of one position/date patch.

    Context offsets, relation, lead, source selection, and renderer settings
    are intentionally absent: all of those are experiment-time choices.
    """
    size = (
        patch_size.to_dict() if isinstance(patch_size, PatchSize) else dict(patch_size)
    )
    return content_sha256(
        {
            "patchspec_schema": "patchspec/v1",
            "patch_size": size,
            "centre_lon": float(centre_lon),
            "centre_lat": float(centre_lat),
            "anchor_time": utc_isoformat(anchor_time),
        }
    )


def position_id(*, grid_id: str, centre_lon: float, centre_lat: float) -> str:
    """Return the stable identity of a grid position."""
    return content_sha256(
        {
            "grid_id": grid_id,
            "centre_lon": float(centre_lon),
            "centre_lat": float(centre_lat),
        }
    )


def _arrow():
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as error:  # pragma: no cover - clean-install diagnostic
        raise ImportError(
            "Published QuerySets require pyarrow. Install ocean_taco with its base dependencies."
        ) from error
    return pa, pq


def _schemas():
    pa, _ = _arrow()
    return {
        "positions": pa.schema(
            [
                ("position_index", pa.int32()),
                ("position_id", pa.string()),
                ("centre_lon", pa.float64()),
                ("centre_lat", pa.float64()),
                ("region_mask", pa.uint8()),
                ("swot_footprint_cells", pa.int32()),
                ("swot_ocean_cells", pa.int32()),
                ("ssh_footprint_cells", pa.int32()),
                ("ssh_ocean_cells", pa.int32()),
            ]
        ),
        "coverage": pa.schema(
            [
                ("position_index", pa.int32()),
                ("date_index", pa.int16()),
                ("swot_valid_cells", pa.int32()),
                ("swot_valid_ocean_cells", pa.int32()),
                ("swot_n_obs_sum", pa.int32()),
                ("ssh_valid_cells", pa.int32()),
                ("ssh_valid_ocean_cells", pa.int32()),
                ("argo_profile_count", pa.int32()),
            ]
        ),
        "assets": pa.schema(
            [
                ("date_index", pa.int16()),
                ("region", pa.string()),
                ("token", pa.string()),
                ("asset_id", pa.string()),
                ("uri", pa.string()),
                ("identity_kind", pa.string()),
                ("identity_value", pa.string()),
                ("status", pa.string()),
            ]
        ),
    }


def _table_digest(rows: tuple[Mapping[str, Any], ...]) -> str:
    """Use canonical content for an unpublished in-memory builder identity."""
    return content_sha256(rows)


def _require_exact_keys(row: Mapping[str, Any], keys: set[str], table: str) -> None:
    actual = set(row)
    if actual != keys:
        missing, extra = sorted(keys - actual), sorted(actual - keys)
        raise ValueError(
            f"{table} row has wrong columns; missing={missing}, extra={extra}."
        )


def _validate_non_negative(
    row: Mapping[str, Any], keys: Iterable[str], table: str
) -> None:
    for key in keys:
        value = row[key]
        if value is not None and (
            not isinstance(value, int) or isinstance(value, bool) or value < 0
        ):
            raise ValueError(f"{table}.{key} must be a nullable non-negative integer.")


@dataclass(frozen=True, slots=True)
class QuerySet:
    """One immutable factored published population.

    ``positions`` and ``coverage`` are always complete: each published
    position has exactly one coverage row for every canonical date.  Coverage
    values may be null, which means the relevant asset closure could not be
    measured; zero means it was measured and no observation was present.
    """

    header: Mapping[str, Any]
    positions: tuple[Mapping[str, Any], ...]
    coverage: tuple[Mapping[str, Any], ...]
    assets: tuple[Mapping[str, Any], ...]

    def __post_init__(self) -> None:
        header = _normalise(dict(self.header))
        header.setdefault("schema_version", SCHEMA_VERSION)
        unknown = set(header) - _HEADER_KEYS
        if unknown:
            raise ValueError(f"QuerySet header has unknown keys: {sorted(unknown)}.")
        missing = _REQUIRED_HEADER_KEYS - set(header)
        if missing:
            raise ValueError(
                f"QuerySet header is missing required keys: {sorted(missing)}."
            )
        if header["schema_version"] != SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported QuerySet schema {header['schema_version']!r}."
            )
        if header["kind"] not in {"training", "eval"}:
            raise ValueError("QuerySet kind must be 'training' or 'eval'.")
        size = header["patch_size"]
        if not isinstance(size, Mapping):
            raise ValueError("QuerySet patch_size must be a PatchSize mapping.")
        PatchSize(float(size.get("value")), str(size.get("unit")))
        if (
            not isinstance(header["grid_spacing_km"], (float, int))
            or header["grid_spacing_km"] <= 0
        ):
            raise ValueError("QuerySet grid_spacing_km must be positive.")
        if not isinstance(header["dates"], list) or not header["dates"]:
            raise ValueError("QuerySet dates must be a non-empty canonical list.")
        dates = tuple(utc_isoformat(item) for item in header["dates"])
        if list(dates) != sorted(dates) or len(set(dates)) != len(dates):
            raise ValueError("QuerySet dates must be unique and UTC-sorted.")
        header["dates"] = list(dates)
        if header["date_sha256"] != content_sha256(header["dates"]):
            raise ValueError("QuerySet date_sha256 does not match dates.")
        if (
            not isinstance(header["tokens"], list)
            or not header["tokens"]
            or header["tokens"] != sorted(set(header["tokens"]))
        ):
            raise ValueError("QuerySet tokens must be a non-empty sorted unique list.")
        if (
            not isinstance(header["parquet_profile"], Mapping)
            or not header["parquet_profile"]
        ):
            raise ValueError("QuerySet parquet_profile must be a non-empty mapping.")

        normal_positions = tuple(
            sorted(
                (_normalise(dict(row)) for row in self.positions),
                key=lambda row: row["position_index"],
            )
        )
        normal_coverage = tuple(
            sorted(
                (_normalise(dict(row)) for row in self.coverage),
                key=lambda row: (row["position_index"], row["date_index"]),
            )
        )
        normal_assets = tuple(
            sorted(
                (_normalise(dict(row)) for row in self.assets),
                key=lambda row: (row["date_index"], row["region"], row["token"]),
            )
        )
        if not normal_positions:
            raise ValueError("QuerySet positions cannot be empty.")
        position_keys = set(_schemas()["positions"].names)
        coverage_keys = set(_schemas()["coverage"].names)
        asset_keys = set(_schemas()["assets"].names)
        indices: set[int] = set()
        for expected, row in enumerate(normal_positions):
            _require_exact_keys(row, position_keys, "positions")
            if row["position_index"] != expected:
                raise ValueError(
                    "positions.position_index must be contiguous canonical int32 order."
                )
            if not isinstance(row["position_id"], str) or not row["position_id"]:
                raise ValueError("positions.position_id must be non-empty.")
            if (
                not -180.0 <= float(row["centre_lon"]) < 180.0
                or not -60.0 <= float(row["centre_lat"]) <= 60.0
            ):
                raise ValueError(
                    "QuerySet positions must remain in the frozen mask domain."
                )
            _validate_non_negative(
                row,
                (
                    "region_mask",
                    "swot_footprint_cells",
                    "swot_ocean_cells",
                    "ssh_footprint_cells",
                    "ssh_ocean_cells",
                ),
                "positions",
            )
            if row["region_mask"] > 255:
                raise ValueError("positions.region_mask must fit uint8.")
            if (
                row["swot_ocean_cells"] > row["swot_footprint_cells"]
                or row["ssh_ocean_cells"] > row["ssh_footprint_cells"]
            ):
                raise ValueError(
                    "Ocean cell counts cannot exceed footprint cell counts."
                )
            expected_id = position_id(
                grid_id=str(header["grid_id"]),
                centre_lon=float(row["centre_lon"]),
                centre_lat=float(row["centre_lat"]),
            )
            if row["position_id"] != expected_id:
                raise ValueError(
                    "positions.position_id does not match the declared grid identity."
                )
            indices.add(row["position_index"])

        seen_pairs: set[tuple[int, int]] = set()
        for row in normal_coverage:
            _require_exact_keys(row, coverage_keys, "coverage")
            pair = (row["position_index"], row["date_index"])
            if pair in seen_pairs:
                raise ValueError(
                    "coverage contains a duplicate (position_index, date_index) pair."
                )
            seen_pairs.add(pair)
            if pair[0] not in indices or not 0 <= pair[1] < len(dates):
                raise ValueError("coverage references an unknown position or date.")
            _validate_non_negative(
                row,
                tuple(
                    key
                    for key in coverage_keys
                    if key not in {"position_index", "date_index"}
                ),
                "coverage",
            )
        expected_pairs = {
            (position, date) for position in indices for date in range(len(dates))
        }
        if seen_pairs != expected_pairs:
            raise ValueError(
                "coverage must contain every published (position, date) pair exactly once."
            )

        seen_assets: set[tuple[int, str, str]] = set()
        for row in normal_assets:
            _require_exact_keys(row, asset_keys, "assets")
            key = (row["date_index"], row["region"], row["token"])
            if key in seen_assets:
                raise ValueError(
                    "assets contains a duplicate (date_index, region, token) row."
                )
            seen_assets.add(key)
            if (
                not 0 <= row["date_index"] < len(dates)
                or row["token"] not in header["tokens"]
            ):
                raise ValueError(
                    "assets references a date or token outside the published set."
                )
            if row["status"] not in {"present", "missing"}:
                raise ValueError("assets.status must be 'present' or 'missing'.")
            if row["status"] == "present" and any(
                not row[field]
                for field in ("asset_id", "uri", "identity_kind", "identity_value")
            ):
                raise ValueError("Present assets require immutable identity details.")
        if not normal_assets:
            raise ValueError(
                "assets must record the immutable closure for every published date and token."
            )
        for date_index in range(len(dates)):
            regions = {
                region
                for current_date, region, _ in seen_assets
                if current_date == date_index
            }
            if not regions:
                raise ValueError(
                    "assets must record at least one region closure for every published date."
                )
            for region in regions:
                expected_assets = {
                    (date_index, region, token) for token in header["tokens"]
                }
                if not expected_assets <= seen_assets:
                    raise ValueError(
                        "assets must contain one identity/status row per date, region, and registered token."
                    )

        supplied_checksums = header.get("table_sha256")
        content_checksums = {
            "positions": _table_digest(normal_positions),
            "coverage": _table_digest(normal_coverage),
            "assets": _table_digest(normal_assets),
        }
        if supplied_checksums is not None and set(supplied_checksums) != set(
            TABLE_FILENAMES
        ):
            raise ValueError("table_sha256 must name positions, coverage, and assets.")
        checksums = dict(supplied_checksums or content_checksums)
        identity_header = {
            key: value
            for key, value in header.items()
            if key not in _NON_SEMANTIC_HEADER_KEYS | {"table_sha256"}
        }
        identifier = content_sha256(
            {"header": identity_header, "table_sha256": checksums}
        )
        supplied_identifier = header.get("queryset_id")
        if supplied_identifier is not None and supplied_identifier != identifier:
            raise ValueError(
                "queryset_id does not match the semantic header and table checksums."
            )
        header["table_sha256"] = checksums
        header["queryset_id"] = identifier
        object.__setattr__(self, "header", header)
        object.__setattr__(self, "positions", normal_positions)
        object.__setattr__(self, "coverage", normal_coverage)
        object.__setattr__(self, "assets", normal_assets)

    @property
    def queryset_id(self) -> str:
        """Content identity including the three table checksums."""
        return str(self.header["queryset_id"])

    @property
    def sha256(self) -> str:
        """Compatibility alias for the published QuerySet identity."""
        return self.queryset_id

    @property
    def patch_size(self) -> PatchSize:
        """The patch size fixed by this published set."""
        value = self.header["patch_size"]
        return PatchSize(float(value["value"]), str(value["unit"]))

    @property
    def dates(self) -> tuple[str, ...]:
        """Canonical UTC anchor-time list."""
        return tuple(self.header["dates"])

    def position(self, index: int) -> Mapping[str, Any]:
        """Return one canonical position row."""
        return self.positions[index]

    def coverage_row(self, position_index: int, date_index: int) -> Mapping[str, Any]:
        """Return one canonical coverage row in O(1) canonical-table order."""
        return self.coverage[position_index * len(self.dates) + date_index]

    def patch_row(
        self,
        position_index: int,
        date_index: int,
        *,
        context_start_offset_days: int = 0,
        context_end_offset_days: int = 0,
        relation: str = "same_time",
        target_lead_days: int = 0,
    ) -> dict[str, Any]:
        """Materialise one logical patch only after experiment-time selection."""
        position = self.position(position_index)
        anchor = self.dates[date_index]
        identifier = patch_id(
            patch_size=self.patch_size,
            centre_lon=float(position["centre_lon"]),
            centre_lat=float(position["centre_lat"]),
            anchor_time=anchor,
        )
        return {
            "position_index": position_index,
            "date_index": date_index,
            "position_id": position["position_id"],
            "patch_id": identifier,
            "centre_lon": position["centre_lon"],
            "centre_lat": position["centre_lat"],
            "patch_size": self.patch_size.to_dict(),
            "anchor_time": anchor,
            "context_start_offset_days": context_start_offset_days,
            "context_end_offset_days": context_end_offset_days,
            "relation": relation,
            "target_lead_days": target_lead_days,
        }

    def write(self, directory: Path | str) -> Path:
        """Atomically publish ``header.json`` plus the three Parquet tables."""
        directory = Path(directory)
        if directory.exists():
            raise FileExistsError(
                f"Refusing to overwrite published QuerySet directory {directory}."
            )
        directory.parent.mkdir(parents=True, exist_ok=True)
        staging = directory.parent / f".{directory.name}.staging-{uuid.uuid4().hex}"
        staging.mkdir()
        try:
            schemas = _schemas()
            checksums: dict[str, str] = {}
            for table_name, rows in (
                ("positions", self.positions),
                ("coverage", self.coverage),
                ("assets", self.assets),
            ):
                table_path = staging / TABLE_FILENAMES[table_name]
                self._write_parquet(table_path, rows, schemas[table_name])
                checksums[table_name] = sha256(table_path.read_bytes()).hexdigest()
            header = dict(self.header)
            header["table_sha256"] = checksums
            identity_header = {
                key: value
                for key, value in header.items()
                if key
                not in _NON_SEMANTIC_HEADER_KEYS | {"table_sha256", "queryset_id"}
            }
            header["queryset_id"] = content_sha256(
                {"header": identity_header, "table_sha256": checksums}
            )
            (staging / "header.json").write_bytes(canonical_json(header) + b"\n")
            os.replace(staging, directory)
            # A builder becomes the just-published immutable identity once the
            # physical table bytes are known.  This is the sole controlled
            # mutation of the frozen in-memory value.
            object.__setattr__(self, "header", header)
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        return directory

    @staticmethod
    def _write_parquet(path: Path, rows: tuple[Mapping[str, Any], ...], schema) -> None:
        pa, pq = _arrow()
        table = pa.Table.from_pylist([dict(row) for row in rows], schema=schema)
        pq.write_table(
            table,
            path,
            compression="zstd",
            compression_level=3,
            version="2.6",
            data_page_version="1.0",
            row_group_size=65_536,
            use_dictionary=True,
            write_statistics=True,
        )

    @classmethod
    def read(cls, directory: Path | str) -> QuerySet:
        """Read a published set, verify all table bytes, and recompute its ID."""
        directory = Path(directory)
        header = json.loads((directory / "header.json").read_text(encoding="utf-8"))
        checksums = header.get("table_sha256")
        if not isinstance(checksums, Mapping) or set(checksums) != set(TABLE_FILENAMES):
            raise ValueError(
                "Published QuerySet header has no complete table_sha256 mapping."
            )
        tables: dict[str, tuple[Mapping[str, Any], ...]] = {}
        _, pq = _arrow()
        for table_name, filename in TABLE_FILENAMES.items():
            path = directory / filename
            actual = sha256(path.read_bytes()).hexdigest()
            if checksums[table_name] != actual:
                raise ValueError(f"QuerySet checksum mismatch for {filename}.")
            tables[table_name] = tuple(pq.read_table(path).to_pylist())
        return cls(
            header=header,
            positions=tables["positions"],
            coverage=tables["coverage"],
            assets=tables["assets"],
        )


@dataclass(frozen=True, slots=True)
class PatchSet:
    """Optional materialised rendering tied to one published QuerySet."""

    queryset_id: str
    render_config: Mapping[str, Any]
    artifact_sha256: str
    metadata: Mapping[str, Any]

    @property
    def patchset_id(self) -> str:
        """Content identity tied to set, renderer configuration, and outputs."""
        return content_sha256(
            {
                "queryset_id": self.queryset_id,
                "render_config": self.render_config,
                "artifact_sha256": self.artifact_sha256,
                "metadata": self.metadata,
            }
        )
