# OceanTACO ML loader

OceanTACO training samples start with a published `QuerySet`, not with ad-hoc
query generation. A QuerySet fixes the candidate position/date population and
its coverage evidence. An experiment then filters and draws rows reproducibly,
and `OceanTACODataset` renders those rows into PyTorch samples.

## From a published QuerySet to a batch

```python
from pathlib import Path

from torch.utils.data import DataLoader

from ocean_taco import CatalogConfig, QueryFilter, QuerySet, draw_queryset
from ocean_taco.render import Resample
from ocean_taco.torch import OceanTACODataset, collate_ocean_samples, seed_ocean_taco_worker

queryset = QuerySet.read("release/querysets/pilot10")
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
