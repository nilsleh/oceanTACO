"""Offline construction of factored published QuerySets."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from datetime import datetime
from typing import Any, Literal

from ..geobox import PatchSize, _utc_datetime, utc_isoformat
from ..queryset import QuerySet, content_sha256
from .coverage import DenseCoverage, build_coverage_table
from .grids import (
    build_position_grid,
    build_random_position_sample,
    grid_id,
    random_grid_id,
    random_position_sampling_metadata,
)
from .ocean_mask import OceanMaskArtifact

QuerySetKind = Literal["training", "eval"]
REFERENCE_PATCH_SIZES_KM = (128, 256, 512)
GRID_SPACING_RATIO = {"training": 2.0 / 3.0, "eval": 0.9}
TRAINING_SAMPLER_SEED = 20260907
TRAINING_DENSITY_RATIO = 0.8
TRAINING_POSITION_LIMITS = {128: 35_000}
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


def training_position_count(mask: OceanMaskArtifact, patch_size: PatchSize) -> int:
    """Return the released best-candidate training density."""
    legacy_count = len(
        build_position_grid(
            mask,
            patch_size=patch_size,
            spacing_km=patch_size.value * GRID_SPACING_RATIO["training"],
        )
    )
    target = int(legacy_count * TRAINING_DENSITY_RATIO)
    return min(target, TRAINING_POSITION_LIMITS.get(int(patch_size.value), target))


def build_positions(
    mask: OceanMaskArtifact,
    *,
    patch_size: PatchSize,
    kind: QuerySetKind,
    training_seed: int = TRAINING_SAMPLER_SEED,
    region_mask: Callable[[Any], int] | None = None,
    static_counts: Callable[[float, float], Mapping[str, int]] | None = None,
) -> tuple[dict[str, Any], ...]:
    """Build systematic eval positions or stochastic training positions."""
    if kind == "eval":
        return build_position_grid(
            mask,
            patch_size=patch_size,
            spacing_km=patch_size.value * GRID_SPACING_RATIO["eval"],
            region_mask=region_mask,
            static_counts=static_counts,
        )
    if kind == "training":
        return build_random_position_sample(
            mask,
            patch_size=patch_size,
            seed=training_seed,
            position_count=training_position_count(mask, patch_size),
            region_mask=region_mask,
            static_counts=static_counts,
        )
    raise ValueError("kind must be 'training' or 'eval'.")


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
    training_seed: int = TRAINING_SAMPLER_SEED,
    region_mask: Callable[[Any], int] | None = None,
) -> QuerySet:
    """Build an unsplit published QuerySet with no policy-based rejection.

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
    positions = build_positions(
        ocean_mask,
        patch_size=patch_size,
        kind=kind,
        training_seed=training_seed,
        region_mask=region_mask,
        static_counts=static_counts,
    )
    if not positions:
        raise ValueError(
            "Position grid is empty after ocean-centre and footprint-domain filtering."
        )
    if kind == "eval":
        grid_spacing = patch_size.value * GRID_SPACING_RATIO["eval"]
        grid_identifier = grid_id(
            spacing_km=grid_spacing, ocean_mask_id=ocean_mask.artifact_id
        )
        position_sampling = {
            "method": "equidistant_ocean/v1",
            "spacing_km": grid_spacing,
        }
    else:
        grid_spacing = None
        grid_identifier = random_grid_id(
            patch_size=patch_size,
            seed=training_seed,
            position_count=len(positions),
            ocean_mask_id=ocean_mask.artifact_id,
        )
        position_sampling = random_position_sampling_metadata(
            ocean_mask,
            patch_size=patch_size,
            seed=training_seed,
            position_count=len(positions),
        )
    coverage = build_coverage_table(
        positions, canonical_dates, measure=measure_coverage
    )
    header = {
        "schema_version": "queryset/v2",
        "patch_size": patch_size.to_dict(),
        "kind": kind,
        "grid_spacing_km": grid_spacing,
        "grid_id": grid_identifier,
        "position_sampling": position_sampling,
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
    "TRAINING_SAMPLER_SEED",
    "build_positions",
    "build_queryset",
    "training_position_count",
]
