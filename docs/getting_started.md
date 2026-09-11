# Getting Started

OceanTACO has two complementary workflows:

- direct native-coordinate retrieval for inspection and analysis;
- reproducible `QuerySet` draws rendered as PyTorch batches for ML.

Python 3.12 or newer is required.

## Installation

```sh
pip install oceantaco
```

For a development checkout, install the test extras with `pip install -e
".[tests]"`.

## Retrieve one native-coordinate subset

```python
from ocean_taco import CatalogConfig, GeoBox
from ocean_taco.retrieve import load_bbox_nc, load_hf_dataset

config = CatalogConfig()
catalog = load_hf_dataset(config)
sst = load_bbox_nc(
    catalog,
    "2024-06-01",
    GeoBox(-80.0, -30.0, 25.0, 50.0),
    "l4_sst",
    config=config,
)
```

`CatalogConfig()` needs no arguments: remote assets are fetched and cached by
`huggingface_hub` under `HF_HOME`. Pass
`CatalogConfig(taco_path="/path/to/OceanTACO")` to read a local catalog instead;
the location must be the `OceanTACO` directory itself.

## Build reproducible ML samples

```python
from torch.utils.data import DataLoader

from ocean_taco import CatalogConfig, QuerySet, draw_queryset
from ocean_taco.render import Resample
from ocean_taco.torch import OceanTACODataset, collate_ocean_samples, seed_ocean_taco_worker

queryset = QuerySet.from_hub(256, "eval")
draw = draw_queryset(queryset, requested_row_count=64, seed=7, record_path="run.json")
dataset = OceanTACODataset(
    queries=draw,
    sources={"l4_sst": Resample((64, 64), support_threshold=0.5)},
    catalog_config=CatalogConfig(),
)
loader = DataLoader(
    dataset,
    batch_size=8,
    num_workers=2,
    persistent_workers=True,
    worker_init_fn=seed_ocean_taco_worker,
    collate_fn=collate_ocean_samples,
)
```

The dataset resolves the catalog before worker processes start; workers read
only resolved assets. Start with `num_workers=0` in notebooks, then benchmark
steady-state training with a small persistent worker pool.

Both stages are worked through with executed output in the tutorials:
[QuerySet selection and native-coordinate
retrieval](tutorials/data_retrieval_workflows.ipynb) for filters, coverage, and
the retrieval functions, and [From a published QuerySet to a rendered
sample](tutorials/ml_dataset.ipynb) for the ML path. The
{doc}`api/index` documents every class named above.

## When something looks wrong

Each symptom below has a first thing to check.

**Catalog access or a download fails.** Check the pinned `CatalogConfig.revision`,
network access, the spelling of the source token, and that `cache_dir` is
writable. Reduce to one row and `num_workers=0` before changing anything else.

**A batch is entirely NaN.** Inspect `availability`, `source_valid`,
`support_mask`, `valid_mask`, and the QuerySet's recorded coverage for those
rows. An absent measurement is represented by NaN and a false mask, not by
zero, so do not fill NaNs without carrying the mask alongside.

**Native-grid batches collate into a ragged structure.** That is the default and
it is deliberate, since native crops of one patch differ in shape between rows.
Group matching shapes with `ShapeBucketSampler`, or opt into
`collate_ocean_samples(batch, native="padded")` and consume both padding masks.

**Workers hang or the loader is slow.** Keep the default `plan_sources=True` so
the catalog is resolved in the parent, pass
`worker_init_fn=seed_ocean_taco_worker`, and start from a small worker count.
Set `persistent_workers=True` only when one DataLoader serves many epochs.

**A replay returns different rows.** Replay needs the same QuerySet and an
unmodified JSON record. `replay_experiment` verifies the QuerySet and table
identity before returning rows, so a mismatch raises rather than returning
silently different data.
