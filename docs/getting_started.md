# Getting Started

OceanTACO has two complementary workflows:

- direct native-coordinate retrieval for inspection and analysis;
- reproducible `QuerySet` draws rendered as PyTorch batches for ML.

Python 3.12 or newer is required.

## Installation

```sh
pip install ocean-taco
```

For a development checkout, install the test extras with `pip install -e
".[tests]"`.

## Retrieve one native-coordinate subset

```python
from ocean_taco import CatalogConfig, GeoBox
from ocean_taco.retrieve import load_bbox_nc, load_hf_dataset

config = CatalogConfig(cache_dir=".oceantaco-cache")
catalog = load_hf_dataset(config)
sst = load_bbox_nc(
    catalog,
    "2024-06-01",
    GeoBox(-80.0, -30.0, 25.0, 50.0),
    "l4_sst",
    config=config,
)
```

Pass `CatalogConfig(taco_path="/path/to/OceanTACO", cache_dir=".oceantaco-cache")`
for a local catalog. The location must be the `OceanTACO` directory itself.

## Build reproducible ML samples

```python
from torch.utils.data import DataLoader

from ocean_taco import CatalogConfig, QuerySet, draw_queryset
from ocean_taco.render import Resample
from ocean_taco.torch import OceanTACODataset, collate_ocean_samples, seed_ocean_taco_worker

queryset = QuerySet.read("release/querysets/pilot10")
draw = draw_queryset(queryset, requested_row_count=64, seed=7, record_path="run.json")
dataset = OceanTACODataset(
    queries=draw,
    sources={"l4_sst": Resample((64, 64), support_threshold=0.5)},
    catalog_config=CatalogConfig(cache_dir=".oceantaco-cache"),
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

See {doc}`dataset-workflows` for retrieval details and
{doc}`dataset-ml-loader` for QuerySet selection, replay, and multimodal
batches.
