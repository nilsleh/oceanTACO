# OceanTACO ML loader

OceanTACO training samples start with a published `QuerySet`, not with ad-hoc
query generation. A QuerySet fixes the candidate position/date population and
its coverage evidence. An experiment then filters and draws rows reproducibly,
and `OceanTACODataset` renders those rows into PyTorch samples.

The published `pilot10/` directory contains six QuerySets named
`<patch-size-km>-<kind>/`, such as `512-eval/` and `512-training/`. Read one
concrete directory, never the parent `pilot10/` directory. `kind` is a
release-level label; choose and record date/geographic guards appropriate to
your scientific split (see {doc}`train-eval-splits`).

## From a published QuerySet to a batch

```python
from pathlib import Path

from torch.utils.data import DataLoader

from ocean_taco import CatalogConfig, QueryFilter, QuerySet, draw_queryset
from ocean_taco.render import Resample
from ocean_taco.torch import OceanTACODataset, collate_ocean_samples, seed_ocean_taco_worker

queryset = QuerySet.read("release/querysets/pilot10/512-eval")
draw = draw_queryset(
    queryset,
    requested_row_count=256,
    seed=42,
    record_path=Path("runs/experiment-42.json"),
    query_filter=QueryFilter(context_start_offset_days=0, context_end_offset_days=1),
)

dataset = OceanTACODataset(
    queries=draw,
    sources={
        "l4_sst": Resample((128, 128), support_threshold=0.5),
        "l3_swot": Resample((128, 128), support_threshold=0.5),
    },
    catalog_config=CatalogConfig(cache_dir=".oceantaco-cache"),
)
loader = DataLoader(
    dataset,
    batch_size=16,
    shuffle=True,
    num_workers=4,
    persistent_workers=True,
    worker_init_fn=seed_ocean_taco_worker,
    collate_fn=collate_ocean_samples,
)
batch = next(iter(loader))
```

The `QueryDraw` and its JSON experiment record are the reproducibility
boundary. To replay exactly the same rows, pass the record to the dataset
instead of drawing again:

```python
replayed = OceanTACODataset(
    queries=queryset,
    experiment_record="runs/experiment-42.json",
    sources={"l4_sst": Resample((128, 128), support_threshold=0.5)},
    catalog_config=CatalogConfig(cache_dir=".oceantaco-cache"),
)
```

## Worker behaviour

For a `CoreSourceLoader`, dataset construction resolves the catalog into plain
asset locations in the parent process. Workers only open those resolved NetCDF
paths or URLs; they do not construct or use a TACO catalog. This makes the
normal PyTorch `fork` worker path safe even when the parent has already opened
the catalog.

Use `num_workers=0` for short interactive work. For repeated training epochs,
benchmark your own patch sizes and source mix; start with a small worker count
(such as 2–8) and enable `persistent_workers=True` when the loader is reused.
Measure steady-state epochs, not only the first epoch. Remote URLs follow the
same planning split, although their throughput also depends on network and
cache behaviour.

## Sample schema and collation

Each sample is a flat mapping keyed by the requested source name, plus `query`
and `availability`. Dense renderers expose `data`, `valid_mask`, coordinates,
and support information. Missing measurements remain `NaN`; zero is never
used as a missing-data sentinel. `collate_ocean_samples` stacks fixed grids and
keeps ragged point samples as explicit ragged batches. Use
`native="padded"` only when a native-coordinate model explicitly needs padded
native grids.

## Vector and point sources

```python
from ocean_taco.render import Points, Resample, VectorPair

sources = {
    "velocity": VectorPair(Resample((128, 128), support_threshold=0.5)),
    "argo": Points(variable="TEMP"),
}
```

`VectorPair` keeps GLORYS eastward/northward components together with one joint
validity mask. `Points` preserves Argo observations as ragged records. Neither
is silently coerced into a dense scalar grid.

## Dataset contract and options

`OceanTACODataset` is a map-style adapter from immutable logical patch rows to
rendered source records. It has two access modes:

- Pass a `QueryDraw` directly, or a `QuerySet` together with exactly one of
  `draw=` and `experiment_record=`. A bare published QuerySet is intentionally
  rejected because it is a population, not an experiment selection.
- Pass a sequence of `PatchSpec`/row mappings for a fully caller-owned list.
  `draw` and `experiment_record` are then rejected.

`sources` is a non-empty mapping from a public source token to a renderer:
`Resample`, `Native`, `Points`, or `VectorPair`. `VectorPair` owns its two
registered components and may not be requested as an ordinary component token.
For access, provide either a callable/test `source_loader` or a
`CatalogConfig(cache_dir=...)`, not both. `catalog_config` constructs the
shipped `CoreSourceLoader`; it plans all unique assets in the parent process by
default (`plan_sources=True`). `patch=` is an optional assertion that every row
declares that exact patch size. `ocean_mask=` overrides the shipped mask for
controlled tests or experiments.

The most common `ValueError`s are useful contract failures: an empty source
mapping, missing/redundant draw selection, a draw for another QuerySet, an
unknown token, requesting a vector component outside `VectorPair`, a renderer
variable unavailable for that source, a patch-size mismatch, or a footprint
outside the mask domain. An unavailable *measurement* is not an error: it is
represented with the empty schema below.

## Complete sample schema

Every unbatched item is a flat mapping:

```text
{
  "<token>": <source record>, ...,
  "query": PatchSpec,
  "availability": {"<token>": bool, ...},
}
```

Dense scalar records (`Resample` or `Native`) have `data` shaped `(T,H,W)`,
boolean `source_valid`, `support_mask`, and `valid_mask`, `lat`/`lon`, and a
list of decoded `times`. Resampled records also carry floating-point `support`
and `native_shape`; mask-aware rendering may add `ocean_mask` and
`in_mask_domain`. `valid_mask` is the model-facing conjunction of source
validity, support threshold, and (when present) the ocean mask.

A `VectorPair` record has `data` shaped `(T,2,H,W)`, those same shared masks
and coordinates, `components=("glorys_uo", "glorys_vo")`, and sample-level
`pair_available`. The joint mask means both components have support. A `Points`
record has one-dimensional `data`, `lat`, `lon`, `pres`, `source_valid`,
`support_mask`, and `valid_mask`, plus ragged `time` and `profile_id` lists;
it can additionally include `direction`, `ocean_mask`, and `in_mask_domain`.

Empty records keep the same type-specific fields. A missing dense record has
zero time steps, NaN data, false masks, fixed output coordinates for
`Resample`, and `availability[token] == False`. A missing point record has
zero-length arrays/lists. Consequently, zero is never a missing-data sentinel;
check `availability`, masks, and `time_mask`/`point_mask` after batching.

## Collation behavior

`collate_ocean_samples` returns `query` as a list and `availability` as one
list of booleans per source. It then chooses by record type:

- Fixed `Resample` scalar grids stack to `(B,T,H,W)`. Variable time is padded
  with NaN values and false masks; `time_mask` identifies real timesteps and
  `times` remains a per-item list.
- Fixed vector pairs stack to `(B,T,2,H,W)` and preserve `components` and a
  `(B,)` `pair_available` tensor. Their time padding has the same NaN/false
  semantics.
- Ragged `Points` stack to `(B,N)` using NaN values and `point_mask`; strings
  (`time`, `profile_id`, and optional `direction`) remain per-item lists. A
  batch with zero points is valid.
- `Native` records stay exact ragged `{"items": [...]}` by default. No hidden
  zero padding occurs.

Use `collate_ocean_samples(batch, native="padded")` or
`native_pad_collate` only for a model that explicitly accepts a padded native
grid. It returns NaN-padded `data`, false masks, `spatial_padding_mask`,
`time_mask`, and per-item `true_shapes`. Padded native grids require
floating-point sources because padding uses NaN.

## Native shapes and bucketing

Native-coordinate crops can have different spatial shapes. To batch without
padding, render the source once to determine its exact shapes and give those
to `ShapeBucketSampler`:

```python
from torch.utils.data import DataLoader
from ocean_taco.render import Native
from ocean_taco.torch import ShapeBucketSampler, native_shapes, collate_ocean_samples

native_dataset = OceanTACODataset(
    queries=draw,
    sources={"l4_sst": Native()},
    catalog_config=CatalogConfig(cache_dir=".oceantaco-cache"),
)
sampler = ShapeBucketSampler(native_shapes(native_dataset, "l4_sst"), batch_size=4)
loader = DataLoader(native_dataset, batch_sampler=sampler, collate_fn=collate_ocean_samples)
```

Shape discovery is deliberately an O(N) rendering pass; cache the list in your
experiment metadata if it is expensive. `set_epoch(epoch)` changes deterministic
batch order without dropping a sample.

## No normalisation hook

The loader applies no transform or normalisation hook. It preserves canonical
units and NaNs so the experiment, not hidden loader state, owns preprocessing.
Fit statistics only on the training draw and apply them with masks, for example:

```python
import torch

def normalise_valid(data: torch.Tensor, valid_mask: torch.Tensor, mean, std):
    safe_std = torch.as_tensor(std, dtype=data.dtype, device=data.device).clamp_min(1e-6)
    result = torch.full_like(data, float("nan"))
    result[valid_mask] = (data[valid_mask] - mean) / safe_std
    return result
```

This wrapper deliberately leaves unavailable, unsupported, and padding cells
as NaN. Record the training-only statistics alongside the QueryDraw record.

## Planning, workers, and troubleshooting

The default Core path resolves an in-memory `AssetPlan` in the parent and
serialises it into `PlannedSourceLoader`; workers open only planned URLs or
paths and do not construct a TACO catalog. The plan is intentionally not
persisted: it is a runtime optimisation, not a reproducibility artifact. The
QuerySet, experiment record, `CatalogConfig`, and source/renderer settings are
the persisted provenance.

- **Catalog or downloads fail:** check the pinned `CatalogConfig.revision`,
  network access, token spelling, and a writable `cache_dir`. Start with one
  row and `num_workers=0`.
- **A batch is empty/NaN:** inspect `availability`, `source_valid`,
  `support_mask`, `valid_mask`, and the QuerySet coverage facts. Do not replace
  NaNs with zero without carrying an explicit mask.
- **Native default collation looks unusual:** it is intentionally ragged. Use
  `ShapeBucketSampler` or opt into padded collation and consume both padding
  masks.
- **Workers hang or are slow:** retain the default `plan_sources=True`, use
  `seed_ocean_taco_worker`, benchmark a small worker count, and enable
  `persistent_workers=True` only when the same DataLoader serves many epochs.
- **Replay differs:** use the exact QuerySet directory and the unchanged JSON
  experiment record. Replay verifies the QuerySet/table identity before it
  returns rows.
