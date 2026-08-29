"""Flat-dict PyTorch dataset and explicit ragged/dense collate strategies."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from torch.utils.data import Dataset

from ..catalog import CatalogConfig
from ..geobox import PatchSize, PatchSpec
from ..manifest import QuerySet
from ..registry import get_modality
from ..render import Native, Points, Resample, VectorPair, canonicalise_dense
from ..sampling import QueryDraw, load_released_ocean_mask, replay_experiment
from .loader import CoreSourceLoader

Renderer = Native | Resample | Points | VectorPair


def _patch_from_row(row: Mapping[str, Any]) -> PatchSpec:
    if "patch_spec" in row:
        row = row["patch_spec"]
    patch = row.get("patch_size")
    if not isinstance(patch, Mapping):
        raise ValueError("QuerySet rows must carry a patch_size mapping.")
    return PatchSpec(
        centre_lon=float(row["centre_lon"]),
        centre_lat=float(row["centre_lat"]),
        patch_size=PatchSize(float(patch["value"]), str(patch["unit"])),
        anchor_time=str(row["anchor_time"]),
        context_start_offset_days=int(row["context_start_offset_days"]),
        context_end_offset_days=int(row["context_end_offset_days"]),
        relation=str(row.get("relation", "same_time")),
        target_lead_days=int(row.get("target_lead_days", 0)),
    )


def _to_tensors(value: Any):
    import torch

    if isinstance(value, np.ndarray):
        if value.dtype.kind in {"U", "S", "O"}:
            return value.tolist()
        return torch.from_numpy(value.copy() if not value.flags.writeable else value)
    if isinstance(value, Mapping):
        return {key: _to_tensors(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_tensors(item) for item in value]
    return value


@dataclass(frozen=True, slots=True)
class _SourceRequest:
    token: str
    renderer: Renderer


class OceanTACODataset(Dataset):
    """Map-style ML dataset with a flat, per-source sample schema.

    ``source_loader`` is the sole access boundary.  It receives the source
    token and logical :class:`PatchSpec`, returns a decoded ``xarray.Dataset``
    (or ``None``), and is constructed outside workers.  This keeps sampler,
    renderer, and access concerns independent and makes fixture testing fully
    offline.
    """

    def __init__(
        self,
        *,
        queries: QuerySet | QueryDraw | Sequence[Mapping[str, Any] | PatchSpec],
        sources: Mapping[str, Renderer],
        source_loader: Callable[[str, PatchSpec], Any] | Any | None = None,
        catalog_config: CatalogConfig | None = None,
        patch: PatchSize | None = None,
        ocean_mask=None,
        draw: QueryDraw | None = None,
        experiment_record: Mapping[str, Any] | Path | str | None = None,
        plan_sources: bool = True,
    ) -> None:
        if not sources:
            raise ValueError("sources cannot be empty.")
        if isinstance(queries, QuerySet):
            if draw is not None and experiment_record is not None:
                raise ValueError(
                    "Pass either draw or experiment_record for a published QuerySet, not both."
                )
            if draw is not None:
                if draw.queryset.queryset_id != queries.queryset_id:
                    raise ValueError(
                        "The supplied draw belongs to a different QuerySet."
                    )
                resolved_draw = draw
            elif experiment_record is not None:
                resolved_draw = replay_experiment(queries, experiment_record)
            else:
                raise ValueError(
                    "A published QuerySet is a population, not a sample list; pass a QueryDraw or experiment_record."
                )
            self.queries = queries
            self.rows = tuple(resolved_draw.rows)
        elif isinstance(queries, QueryDraw):
            if draw is not None or experiment_record is not None:
                raise ValueError(
                    "QueryDraw already contains its explicit experiment selection."
                )
            self.queries = queries.queryset
            self.rows = tuple(queries.rows)
        else:
            if draw is not None or experiment_record is not None:
                raise ValueError(
                    "draw and experiment_record only apply to a published QuerySet."
                )
            self.queries = queries
            self.rows = tuple(queries)
        self.source_requests = tuple(
            _SourceRequest(token, renderer) for token, renderer in sources.items()
        )
        for request in self.source_requests:
            if isinstance(request.renderer, VectorPair):
                for component in request.renderer.components:
                    get_modality(component)
            else:
                source = get_modality(request.token)
                if source.render_class == "vector_pair":
                    raise ValueError(
                        f"{request.token!r} is a vector component; request its registered pair through VectorPair()."
                    )
        if source_loader is not None and catalog_config is not None:
            raise ValueError("Pass either source_loader or catalog_config, not both.")
        if source_loader is None:
            if catalog_config is None:
                raise ValueError(
                    "Provide source_loader or CatalogConfig(cache_dir=...) for the shipped CoreSourceLoader."
                )
            source_loader = CoreSourceLoader(catalog_config)
        self.source_loader = source_loader
        self.patch = patch
        self.ocean_mask = ocean_mask or load_released_ocean_mask()
        if plan_sources and isinstance(self.source_loader, CoreSourceLoader):
            self.source_loader = self.source_loader.plan(self._planning_requests())

    def _planning_requests(self) -> tuple[tuple[str, PatchSpec], ...]:
        """Return the unique logical source requests resolved before workers start."""
        tokens = {
            component
            for request in self.source_requests
            for component in (
                request.renderer.components
                if isinstance(request.renderer, VectorPair)
                else (request.token,)
            )
        }
        specs = {
            row if isinstance(row, PatchSpec) else _patch_from_row(row)
            for row in self.rows
        }
        return tuple((token, spec) for token in sorted(tokens) for spec in specs)

    def __len__(self) -> int:
        """Number of immutable logical sample rows."""
        return len(self.rows)

    def _load(self, token: str, patch: PatchSpec):
        loader = self.source_loader
        if callable(loader):
            return loader(token, patch)
        return loader.load(token, patch)

    def _load_pair(self, components: tuple[str, str], patch: PatchSpec):
        loader = self.source_loader
        paired = getattr(loader, "load_pair", None)
        if callable(paired):
            return paired(components, patch)
        first, second = (self._load(token, patch) for token in components)
        if first is None or second is None:
            return None
        return {components[0]: first, components[1]: second}

    def _spec_for(self, row: Mapping[str, Any] | PatchSpec) -> PatchSpec:
        spec = row if isinstance(row, PatchSpec) else _patch_from_row(row)
        if self.patch is not None and spec.patch_size != self.patch:
            raise ValueError(
                "Dataset patch does not match the QuerySet row's declared PatchSize."
            )
        footprint = spec.footprint
        if (
            footprint.lat_min < self.ocean_mask.lat[0]
            or footprint.lat_max > self.ocean_mask.lat[-1]
        ):
            raise ValueError(
                "PatchSpec footprint leaves the configured ocean-mask domain; "
                "OceanTACO ML rendering is supported only where its mask has a classification."
            )
        return spec

    @staticmethod
    def _context_window(dense, spec: PatchSpec):
        """Limit a canonical dense source to the patch's UTC context window."""
        start = spec.context.start.replace(tzinfo=None)
        end = spec.context.end.replace(tzinfo=None)
        return dense.sel(time=slice(start, end))

    @staticmethod
    def _unavailable(
        output: dict[str, Any], availability: dict[str, bool], request: _SourceRequest
    ) -> None:
        """Record the one canonical representation for an unavailable source."""
        availability[request.token] = False
        output[request.token] = request.renderer.empty()

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Render one logical patch into a flat dict keyed by source token."""
        row = self.rows[index]
        spec = self._spec_for(row)
        output: dict[str, Any] = {}
        availability: dict[str, bool] = {}
        for request in self.source_requests:
            if isinstance(request.renderer, VectorPair):
                raw_pair = self._load_pair(request.renderer.components, spec)
                if raw_pair is None:
                    self._unavailable(output, availability, request)
                    continue
                components = request.renderer.components
                raw_values = tuple(raw_pair[token] for token in components)
                sources = tuple(get_modality(token) for token in components)
            else:
                raw = self._load(request.token, spec)
                if raw is None:
                    self._unavailable(output, availability, request)
                    continue
                raw_values = (raw,)
                sources = (get_modality(request.token),)
            if isinstance(request.renderer, Points):
                source = sources[0]
                if request.renderer.variable not in source.available_variables:
                    raise ValueError(
                        f"{request.renderer.variable!r} is not an exposed variable for {request.token!r}."
                    )
                rendered = request.renderer.render(
                    raw_values[0],
                    spec.footprint,
                    time=spec.context,
                    ocean_mask=self.ocean_mask,
                )
            else:
                dense_values = tuple(
                    self._context_window(
                        canonicalise_dense(raw_value, source, fallback_time=spec.anchor_time),
                        spec,
                    )
                    for raw_value, source in zip(raw_values, sources, strict=True)
                )
                if any(dense.sizes["time"] == 0 for dense in dense_values):
                    self._unavailable(output, availability, request)
                    continue
                if isinstance(request.renderer, VectorPair):
                    rendered = request.renderer.render(
                        *dense_values,
                        spec.footprint,
                        ocean_mask=self.ocean_mask,
                        token=request.token,
                    )
                else:
                    rendered = request.renderer.render(
                        dense_values[0],
                        spec.footprint,
                        ocean_mask=self.ocean_mask,
                        token=request.token,
                        source=sources[0],
                    )
            availability[request.token] = (
                bool(rendered["pair_available"])
                if isinstance(request.renderer, VectorPair)
                else bool(np.asarray(rendered["valid_mask"]).any())
            )
            output[request.token] = rendered
        output["query"] = spec
        output["availability"] = availability
        return _to_tensors(output)


def _stack_fixed_grid(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Stack fixed-shape resampled sources, padding only a variable time axis."""
    import torch

    max_time = max(record["data"].shape[0] for record in records)
    data, valid, source_valid, support_mask, support, time_mask = [], [], [], [], [], []
    for record in records:
        value = record["data"]
        count = value.shape[0]
        if count < max_time:
            pad_shape = (max_time - count, *value.shape[1:])
            value = torch.cat(
                (value, torch.full(pad_shape, float("nan"), dtype=value.dtype)), dim=0
            )
            mask = torch.cat(
                (record["valid_mask"], torch.zeros(pad_shape, dtype=torch.bool)), dim=0
            )
            source_valid_value = torch.cat(
                (
                    record.get("source_valid", record["valid_mask"]),
                    torch.zeros(pad_shape, dtype=torch.bool),
                ),
                dim=0,
            )
            support_mask_value = torch.cat(
                (
                    record.get("support_mask", record["valid_mask"]),
                    torch.zeros(pad_shape, dtype=torch.bool),
                ),
                dim=0,
            )
            if "support" in record:
                support_value = torch.cat(
                    (
                        record["support"],
                        torch.zeros(pad_shape, dtype=record["support"].dtype),
                    ),
                    dim=0,
                )
            else:
                support_value = None
        else:
            mask, support_value = record["valid_mask"], record.get("support")
            source_valid_value = record.get("source_valid", record["valid_mask"])
            support_mask_value = record.get("support_mask", record["valid_mask"])
        data.append(value)
        valid.append(mask)
        source_valid.append(source_valid_value)
        support_mask.append(support_mask_value)
        if support_value is not None:
            support.append(support_value)
        time_mask.append(torch.arange(max_time) < count)
    result: dict[str, Any] = {
        "data": torch.stack(data),
        "valid_mask": torch.stack(valid),
        "source_valid": torch.stack(source_valid),
        "support_mask": torch.stack(support_mask),
        "lat": torch.stack([record["lat"] for record in records]),
        "lon": torch.stack([record["lon"] for record in records]),
        "time_mask": torch.stack(time_mask),
        "times": [record["times"] for record in records],
    }
    if support:
        result["support"] = torch.stack(support)
    if any("ocean_mask" in record for record in records):
        result["ocean_mask"] = torch.stack(
            [
                record.get(
                    "ocean_mask",
                    torch.zeros(record["data"].shape[-2:], dtype=torch.bool),
                )
                for record in records
            ]
        )
    if any("in_mask_domain" in record for record in records):
        result["in_mask_domain"] = torch.stack(
            [
                record.get(
                    "in_mask_domain",
                    torch.zeros(record["data"].shape[-2:], dtype=torch.bool),
                )
                for record in records
            ]
        )
    return result


def _stack_vector_pair(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Stack a fixed-grid vector pair without splitting component support."""
    import torch

    max_time = max(record["data"].shape[0] for record in records)
    data, valid, source_valid, support_mask, support, time_mask = [], [], [], [], [], []
    for record in records:
        count = record["data"].shape[0]
        if count < max_time:
            data_pad = (max_time - count, *record["data"].shape[1:])
            mask_pad = (max_time - count, *record["valid_mask"].shape[1:])
            value = torch.cat(
                (
                    record["data"],
                    torch.full(data_pad, float("nan"), dtype=record["data"].dtype),
                ),
                dim=0,
            )
            mask = torch.cat(
                (record["valid_mask"], torch.zeros(mask_pad, dtype=torch.bool)), dim=0
            )
            source_valid_value = torch.cat(
                (record["source_valid"], torch.zeros(mask_pad, dtype=torch.bool)), dim=0
            )
            support_mask_value = torch.cat(
                (record["support_mask"], torch.zeros(mask_pad, dtype=torch.bool)), dim=0
            )
            support_value = (
                torch.cat(
                    (
                        record["support"],
                        torch.zeros(mask_pad, dtype=record["support"].dtype),
                    ),
                    dim=0,
                )
                if "support" in record
                else None
            )
        else:
            value = record["data"]
            mask = record["valid_mask"]
            source_valid_value = record["source_valid"]
            support_mask_value = record["support_mask"]
            support_value = record.get("support")
        data.append(value)
        valid.append(mask)
        source_valid.append(source_valid_value)
        support_mask.append(support_mask_value)
        if support_value is not None:
            support.append(support_value)
        time_mask.append(torch.arange(max_time) < count)
    result: dict[str, Any] = {
        "data": torch.stack(data),
        "valid_mask": torch.stack(valid),
        "source_valid": torch.stack(source_valid),
        "support_mask": torch.stack(support_mask),
        "lat": torch.stack([record["lat"] for record in records]),
        "lon": torch.stack([record["lon"] for record in records]),
        "time_mask": torch.stack(time_mask),
        "times": [record["times"] for record in records],
        "components": records[0]["components"],
        "pair_available": torch.tensor(
            [record["pair_available"] for record in records], dtype=torch.bool
        ),
    }
    if support:
        result["support"] = torch.stack(support)
    for key in ("ocean_mask", "in_mask_domain"):
        if any(key in record for record in records):
            result[key] = torch.stack(
                [
                    record.get(
                        key, torch.zeros(record["data"].shape[-2:], dtype=torch.bool)
                    )
                    for record in records
                ]
            )
    return result


def _pad_native_grid(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Dense opt-in Native collation with NaN padding and explicit semantics."""
    import torch

    max_t = max(record["data"].shape[0] for record in records)
    max_h = max(record["data"].shape[-2] for record in records)
    max_w = max(record["data"].shape[-1] for record in records)
    (
        data,
        valid,
        source_valid,
        support_mask,
        latitudes,
        longitudes,
        padding,
        time_mask,
        true_shapes,
    ) = [], [], [], [], [], [], [], [], []
    for record in records:
        value, mask = record["data"], record["valid_mask"]
        t, h, w = value.shape
        padded_data = torch.full((max_t, max_h, max_w), float("nan"), dtype=value.dtype)
        padded_valid = torch.zeros((max_t, max_h, max_w), dtype=torch.bool)
        padded_source_valid = torch.zeros((max_t, max_h, max_w), dtype=torch.bool)
        padded_support_mask = torch.zeros((max_t, max_h, max_w), dtype=torch.bool)
        padded_data[:t, :h, :w], padded_valid[:t, :h, :w] = value, mask
        padded_source_valid[:t, :h, :w] = record.get("source_valid", mask)
        padded_support_mask[:t, :h, :w] = record.get("support_mask", mask)
        padded_lat = torch.full((max_h,), float("nan"), dtype=record["lat"].dtype)
        padded_lon = torch.full((max_w,), float("nan"), dtype=record["lon"].dtype)
        padded_lat[:h], padded_lon[:w] = record["lat"], record["lon"]
        spatial_padding = torch.ones((max_h, max_w), dtype=torch.bool)
        spatial_padding[:h, :w] = False
        data.append(padded_data)
        valid.append(padded_valid)
        source_valid.append(padded_source_valid)
        support_mask.append(padded_support_mask)
        latitudes.append(padded_lat)
        longitudes.append(padded_lon)
        padding.append(spatial_padding)
        time_mask.append(torch.arange(max_t) < t)
        true_shapes.append((t, h, w))
    result = {
        "data": torch.stack(data),
        "valid_mask": torch.stack(valid),
        "source_valid": torch.stack(source_valid),
        "support_mask": torch.stack(support_mask),
        "lat": torch.stack(latitudes),
        "lon": torch.stack(longitudes),
        "spatial_padding_mask": torch.stack(padding),
        "time_mask": torch.stack(time_mask),
        "true_shapes": true_shapes,
        "times": [record["times"] for record in records],
    }
    if any("ocean_mask" in record for record in records):
        ocean_masks = []
        for record in records:
            mask = torch.zeros((max_h, max_w), dtype=torch.bool)
            if "ocean_mask" in record:
                h, w = record["ocean_mask"].shape
                mask[:h, :w] = record["ocean_mask"]
            ocean_masks.append(mask)
        result["ocean_mask"] = torch.stack(ocean_masks)
    if any("in_mask_domain" in record for record in records):
        domains = []
        for record in records:
            domain = torch.zeros((max_h, max_w), dtype=torch.bool)
            if "in_mask_domain" in record:
                h, w = record["in_mask_domain"].shape
                domain[:h, :w] = record["in_mask_domain"]
            domains.append(domain)
        result["in_mask_domain"] = torch.stack(domains)
    return result


def _pad_points(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Pad ragged points with NaN and a boolean point mask; retain N=0."""
    import torch

    maximum = max(record["data"].shape[0] for record in records)
    result: dict[str, Any] = {
        "time": [record["time"] for record in records],
        "profile_id": [record["profile_id"] for record in records],
    }
    for key in ("data", "lat", "lon", "pres"):
        values = []
        for record in records:
            padded = torch.full((maximum,), float("nan"), dtype=record[key].dtype)
            padded[: record[key].shape[0]] = record[key]
            values.append(padded)
        result[key] = torch.stack(values)
    masks = []
    for record in records:
        mask = torch.zeros((maximum,), dtype=torch.bool)
        mask[: record["data"].shape[0]] = True
        masks.append(mask)
    result["point_mask"] = torch.stack(masks)
    valid = []
    source_valid = []
    support_mask = []
    for record in records:
        for target, key in (
            (valid, "valid_mask"),
            (source_valid, "source_valid"),
            (support_mask, "support_mask"),
        ):
            padded = torch.zeros((maximum,), dtype=torch.bool)
            padded[: record[key].shape[0]] = record[key]
            target.append(padded)
    result["valid_mask"] = torch.stack(valid)
    result["source_valid"] = torch.stack(source_valid)
    result["support_mask"] = torch.stack(support_mask)
    if any("ocean_mask" in record for record in records):
        ocean_masks = []
        for record in records:
            padded = torch.zeros((maximum,), dtype=torch.bool)
            if "ocean_mask" in record:
                padded[: record["ocean_mask"].shape[0]] = record["ocean_mask"]
            ocean_masks.append(padded)
        result["ocean_mask"] = torch.stack(ocean_masks)
    if any("in_mask_domain" in record for record in records):
        domains = []
        for record in records:
            domain = torch.zeros((maximum,), dtype=torch.bool)
            if "in_mask_domain" in record:
                domain[: record["in_mask_domain"].shape[0]] = record["in_mask_domain"]
            domains.append(domain)
        result["in_mask_domain"] = torch.stack(domains)
    if "direction" in records[0]:
        result["direction"] = [record["direction"] for record in records]
    return result


def collate_ocean_samples(
    batch: Sequence[Mapping[str, Any]],
    *,
    native: Literal["ragged", "padded"] = "ragged",
) -> dict[str, Any]:
    """Collate flat samples without silently zero-padding native grids.

    The default keeps native crops as exact ragged records.  ``native='padded'``
    is opt-in and labels every padded cell through ``spatial_padding_mask``.
    """
    if not batch:
        return {}
    if native not in {"ragged", "padded"}:
        raise ValueError("native must be 'ragged' or 'padded'.")
    tokens = [key for key in batch[0] if key not in {"query", "availability"}]
    result: dict[str, Any] = {
        "query": [sample["query"] for sample in batch],
        "availability": {
            token: [sample["availability"][token] for sample in batch]
            for token in tokens
        },
    }
    for token in tokens:
        records = [sample[token] for sample in batch]
        if "pres" in records[0]:
            result[token] = _pad_points(records)
        elif "pair_available" in records[0]:
            result[token] = _stack_vector_pair(records)
        elif "support" in records[0]:
            result[token] = _stack_fixed_grid(records)
        elif native == "padded":
            result[token] = _pad_native_grid(records)
        else:
            result[token] = {"items": records}
    return result


def native_pad_collate(batch: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """DataLoader-ready opt-in dense Native collation with NaN padding."""
    return collate_ocean_samples(batch, native="padded")
