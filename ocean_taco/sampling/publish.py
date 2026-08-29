"""Offline construction of factored published QuerySets."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from datetime import datetime
from typing import Any, Literal

from ..geobox import PatchSize, _utc_datetime, utc_isoformat
from ..manifest import QuerySet, content_sha256
from .coverage import DenseCoverage, build_coverage_table
from .grids import build_position_grid, grid_id
from .ocean_mask import OceanMaskArtifact

QuerySetKind = Literal["training", "eval"]
REFERENCE_PATCH_SIZES_KM = (128, 256, 512)
GRID_SPACING_RATIO = {"training": 2.0 / 3.0, "eval": 0.9}
PARQUET_PROFILE = {
    "writer": "pyarrow",
    "format_version": "2.6",
    "compression": "zstd",
    "compression_level": 3,
    "row_group_size": 65_536,
    "data_page_version": "1.0",
}


def _canonical_dates(dates: Sequence[datetime | str]) -> list[str]:
    result = [utc_isoformat(_utc_datetime(value)) for value in dates]
    if not result or result != sorted(result) or len(set(result)) != len(result):
        raise ValueError(
            "Published QuerySet dates must be a non-empty unique UTC-sorted sequence."
        )
    return result


def build_queryset(
    *,
    ocean_mask: OceanMaskArtifact,
    patch_size: PatchSize,
    kind: QuerySetKind,
    dates: Sequence[datetime | str],
    tokens: Sequence[str],
    provenance: Mapping[str, Any],
    assets: Sequence[Mapping[str, Any]],
    measure_coverage: Callable[
        [Mapping[str, Any], int, str], Mapping[str, DenseCoverage | int | None]
    ],
    static_counts: Callable[[float, float], Mapping[str, int]],
    region_mask: Callable[[Any], int] | None = None,
) -> QuerySet:
    """Build an unsplit published population with no policy-based rejection.

    ``measure_coverage`` and ``static_counts`` are offline builder callbacks.
    Their outputs are facts stored in the Parquet tables, not admission
    criteria.  Every grid position appears for every canonical date.
    """
    if kind not in GRID_SPACING_RATIO:
        raise ValueError("kind must be 'training' or 'eval'.")
    if patch_size.unit != "km":
        raise ValueError("Released QuerySets use kilometre PatchSize values.")
    canonical_dates = _canonical_dates(dates)
    sorted_tokens = sorted(set(tokens))
    if not sorted_tokens or list(tokens) != sorted_tokens:
        raise ValueError("tokens must be a non-empty sorted unique sequence.")
    required_provenance = {
        "dataset_revision",
        "catalog_sha256",
        "registry_sha256",
        "source_records_sha256",
        "code_commit",
        "environment_lock_hash",
    }
    missing = required_provenance - set(provenance)
    if missing:
        raise ValueError(
            f"Published QuerySet provenance is missing: {sorted(missing)}."
        )
    if any(provenance[key] in (None, "", "unknown") for key in required_provenance):
        raise ValueError(
            "Published QuerySet provenance must use concrete identities, never 'unknown'."
        )
    spacing = patch_size.value * GRID_SPACING_RATIO[kind]
    positions = build_position_grid(
        ocean_mask,
        patch_size=patch_size,
        spacing_km=spacing,
        region_mask=region_mask,
        static_counts=static_counts,
    )
    if not positions:
        raise ValueError(
            "Position grid is empty after ocean-centre and footprint-domain filtering."
        )
    coverage = build_coverage_table(
        positions, canonical_dates, measure=measure_coverage
    )
    header = {
        "schema_version": "queryset/v1",
        "patch_size": patch_size.to_dict(),
        "kind": kind,
        "grid_spacing_km": spacing,
        "grid_id": grid_id(spacing_km=spacing, ocean_mask_id=ocean_mask.artifact_id),
        "dataset_revision": provenance["dataset_revision"],
        "catalog_sha256": provenance["catalog_sha256"],
        "registry_sha256": provenance["registry_sha256"],
        "source_records_sha256": provenance["source_records_sha256"],
        "ocean_mask_id": ocean_mask.artifact_id,
        "ocean_mask_sha256": ocean_mask.sha256,
        "dates": canonical_dates,
        "date_sha256": content_sha256(canonical_dates),
        "tokens": sorted_tokens,
        "parquet_profile": PARQUET_PROFILE,
        "code_commit": provenance["code_commit"],
        "environment_lock_hash": provenance["environment_lock_hash"],
        "coverage_rules": {
            "sparse_grid_tokens": ["l3_swot", "l3_ssh"],
            "point_token": "argo",
            "null_semantics": "unavailable_or_unmeasurable_asset_closure",
            "renderer_independent": True,
        },
        "grid_validation": provenance.get("grid_validation", {}),
    }
    return QuerySet(
        header=header, positions=positions, coverage=coverage, assets=tuple(assets)
    )


__all__ = [
    "GRID_SPACING_RATIO",
    "PARQUET_PROFILE",
    "REFERENCE_PATCH_SIZES_KM",
    "build_queryset",
]
