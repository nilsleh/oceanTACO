"""Exact uniform draws and experiment-record replay for QuerySets."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..filter import CoverageRequirement, QueryFilter, SelectedPairs, select_queryset
from ..geobox import GeoBox
from ..manifest import QuerySet, canonical_json, content_sha256


def _floyd_ordinals(population_size: int, count: int, seed: int) -> tuple[int, ...]:
    """Draw exact uniform ordinals without replacement using Floyd's method."""
    if not 0 <= count <= population_size:
        raise ValueError("requested row count must be in [0, selected pair count].")
    generator = np.random.Generator(np.random.PCG64(seed))
    selected: set[int] = set()
    for value in range(population_size - count, population_size):
        candidate = int(generator.integers(0, value + 1))
        selected.add(value if candidate in selected else candidate)
    return tuple(sorted(selected))


def _filter_from_dict(payload: Mapping[str, Any]) -> QueryFilter:
    box_payload = payload.get("box")
    box = GeoBox(**box_payload) if box_payload is not None else None
    coverage = tuple(
        CoverageRequirement(**item) for item in payload.get("coverage", ())
    )
    return QueryFilter(
        date_start=payload.get("date_start"),
        date_end=payload.get("date_end"),
        box=box,
        region_mask_any=payload.get("region_mask_any"),
        coverage=coverage,
        context_start_offset_days=int(payload.get("context_start_offset_days", 0)),
        context_end_offset_days=int(payload.get("context_end_offset_days", 0)),
        relation=str(payload.get("relation", "same_time")),
        target_lead_days=int(payload.get("target_lead_days", 0)),
    )


def _record(
    selected: SelectedPairs,
    *,
    seed: int,
    requested_row_count: int,
    ranks: tuple[int, ...],
    rows: tuple[Mapping[str, Any], ...],
) -> dict[str, Any]:
    queryset = selected.queryset
    query_filter = selected.query_filter
    return {
        "schema_version": "experiment-record/v1",
        "queryset_id": queryset.queryset_id,
        "table_sha256": dict(queryset.header["table_sha256"]),
        "rng_algorithm": "numpy.PCG64/floyd-v1",
        "seed": seed,
        "selection": selected.to_dict(),
        "selection_sha256": query_filter.sha256,
        "selected_pair_count": selected.count,
        "requested_row_count": requested_row_count,
        "inclusion_probability": 0.0
        if selected.count == 0
        else requested_row_count / selected.count,
        "selected_ranks_sha256": content_sha256(ranks),
        "emitted_patch_id_digest": content_sha256([row["patch_id"] for row in rows]),
        "context_start_offset_days": query_filter.context_start_offset_days,
        "context_end_offset_days": query_filter.context_end_offset_days,
        "relation": query_filter.relation,
        "target_lead_days": query_filter.target_lead_days,
        "code_commit": queryset.header["code_commit"],
        "environment_lock_hash": queryset.header["environment_lock_hash"],
    }


def _write_record(path: Path | str, record: Mapping[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(canonical_json(record) + b"\n")
    os.replace(temporary, path)
    return path


@dataclass(frozen=True, slots=True)
class QueryDraw:
    """One reproducible uniform draw from a selected QuerySet population."""

    queryset: QuerySet
    rows: tuple[Mapping[str, Any], ...]
    record: Mapping[str, Any]

    @property
    def inclusion_probability(self) -> float:
        """Exact common inclusion probability for every emitted patch."""
        return float(self.record["inclusion_probability"])


def draw_queryset(
    queryset: QuerySet,
    *,
    requested_row_count: int,
    seed: int,
    record_path: Path | str,
    query_filter: QueryFilter | None = None,
) -> QueryDraw:
    """Uniformly draw without replacement and atomically write its record."""
    if requested_row_count < 0:
        raise ValueError("requested_row_count must be non-negative.")
    selected = select_queryset(queryset, query_filter)
    if requested_row_count > selected.count:
        raise ValueError(
            f"requested_row_count={requested_row_count} exceeds {selected.count} selected pairs."
        )
    ranks = _floyd_ordinals(selected.count, requested_row_count, seed)
    rows = tuple(
        queryset.patch_row(
            *selected.resolve_rank(rank),
            context_start_offset_days=selected.query_filter.context_start_offset_days,
            context_end_offset_days=selected.query_filter.context_end_offset_days,
            relation=selected.query_filter.relation,
            target_lead_days=selected.query_filter.target_lead_days,
        )
        for rank in ranks
    )
    record = _record(
        selected,
        seed=seed,
        requested_row_count=requested_row_count,
        ranks=ranks,
        rows=rows,
    )
    _write_record(record_path, record)
    return QueryDraw(queryset=queryset, rows=rows, record=record)


def replay_experiment(
    queryset: QuerySet, record: Mapping[str, Any] | Path | str
) -> QueryDraw:
    """Reconstruct and verify a draw from its experiment record."""
    if isinstance(record, (Path, str)):
        record = json.loads(Path(record).read_text(encoding="utf-8"))
    if record.get("schema_version") != "experiment-record/v1":
        raise ValueError("Unsupported experiment record schema.")
    if (
        record.get("queryset_id") != queryset.queryset_id
        or record.get("table_sha256") != queryset.header["table_sha256"]
    ):
        raise ValueError(
            "Experiment record does not name this exact published QuerySet."
        )
    query_filter = _filter_from_dict(record["selection"])
    if record.get("selection_sha256") != query_filter.sha256:
        raise ValueError("Experiment record filter hash does not match its selection.")
    selected = select_queryset(queryset, query_filter)
    requested = int(record["requested_row_count"])
    if int(record.get("selected_pair_count", -1)) != selected.count:
        raise ValueError(
            "Experiment record selected pair count no longer matches the published set."
        )
    ranks = _floyd_ordinals(selected.count, requested, int(record["seed"]))
    if record.get("selected_ranks_sha256") != content_sha256(ranks):
        raise ValueError(
            "Experiment record rank digest does not match its seed and selection."
        )
    rows = tuple(
        queryset.patch_row(
            *selected.resolve_rank(rank),
            context_start_offset_days=query_filter.context_start_offset_days,
            context_end_offset_days=query_filter.context_end_offset_days,
            relation=query_filter.relation,
            target_lead_days=query_filter.target_lead_days,
        )
        for rank in ranks
    )
    if record.get("emitted_patch_id_digest") != content_sha256(
        [row["patch_id"] for row in rows]
    ):
        raise ValueError(
            "Experiment record patch-ID digest does not match the reconstructed draw."
        )
    return QueryDraw(queryset=queryset, rows=rows, record=dict(record))


__all__ = ["QueryDraw", "draw_queryset", "replay_experiment"]
