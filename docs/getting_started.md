# Getting Started

OceanTACO supports two complementary access patterns:

- direct retrieval with `xarray` data loading for inspection and analysis
- query-based sampling with `OceanTACODataset` for ML training/evaluation

`OceanTACODataset` works with both local data and remote HuggingFace URLs, so the key decision is workflow style (analysis vs ML), not local vs remote. Note that currently, due to storage limits streaming only works for the Core dataset.

## Prerequisites

- Python 3.11+
- Conda (recommended) or pip

## Installation

From the repository root, choose the dependency profile that matches your use case:

```sh
# Dataset loading + queries + visualization (default)
pip install -e .

# Add HuggingFace client helpers for streaming/downloading
pip install -e ".[hf]"

# Add data-generation pipeline dependencies
pip install -e ".[generate]"

# Full development profile
pip install -e ".[generate,hf,tests]"
```

If using conda:

```sh
conda activate testpy311
pip install -e ".[hf]"
```

## Choose the Right Workflow

Use this quick rule:

- use direct retrieval when you want to inspect one date/region/source as an `xarray.Dataset`
- use `OceanTACODataset` when you need repeated sampling over many queries and batches

| Goal | Recommended API | Typical output |
|---|---|---|
| Inspect and visualize a few subsets | `ocean_taco.dataset.retrieve` helpers | `xr.Dataset` |
| Build training/eval samples with query generation | `OceanTACODataset` + `QueryGenerator` | PyTorch-ready sample dicts |
| Use local files only | either API | same as above |
| Stream from HuggingFace | either API | same as above |

## Quick Start: Retrieval (Direct `xarray` Access)

Use this for cloud-native subsetting, and plotting workflows.

```python
from ocean_taco.dataset.retrieve import HF_DEFAULT_URL, load_hf_dataset, load_bbox_nc

dataset_hf = load_hf_dataset(HF_DEFAULT_URL)

# Retrieve tiles overlapping a bbox for one date and one data source.
ds = load_bbox_nc(
    dataset_hf,
    date="2024-06-01",
    bbox=(-80, -30, 25, 50),   # (lon_min, lon_max, lat_min, lat_max)
    data_source="l4_sst",
    cache_dir="./cache",       # optional local cache
)
```

For more retrieval patterns (single tile, full snapshot, stream records), see {doc}`dataset-workflows`.

## Quick Start: ML Loader (`OceanTACODataset`)

Use this when you need consistent query sampling for model training/evaluation.

`taco_path` can be either:

- a local OceanTACO dataset path
- a remote dataset URL such as `HF_DEFAULT_URL`

```python
from torch.utils.data import DataLoader
from ocean_taco.dataset import OceanTACODataset, collate_ocean_samples
from ocean_taco.dataset.queries import QueryGenerator, PatchSize
from ocean_taco.dataset.retrieve import HF_DEFAULT_URL

dataset = OceanTACODataset(
    taco_path=HF_DEFAULT_URL,  # or "/path/to/OceanTACO"
    input_variables=["l4_ssh", "l4_sst", "glorys_sss"],
    target_variables=["l3_swot"],
    temporal_agg="mean",
)

generator = QueryGenerator(land_mask_path=".ocean_mask_cache/land_mask.npy")
queries = generator.generate_training_queries(
    n_queries=1000,
    patch_size=PatchSize(2.0, "deg"),
    date_range=("2024-01-01", "2024-03-31"),
)

loader = DataLoader(
    dataset,
    sampler=queries,
    batch_size=16,
    collate_fn=collate_ocean_samples,
    num_workers=4,
)
```

For detailed ML guidance (query design, batching, patch sizing, training vs eval), see {doc}`dataset-ml-loader`.

## Next Steps

- See the {doc}`dataset_description` for a full list of variables and ocean regions.
- See {doc}`dataset-workflows` for retrieval and streaming workflows.
- See {doc}`dataset-ml-loader` for end-to-end ML loader workflows.
- See the {doc}`dataset_generation` page for the full raw-data -> formatted -> TACO build pipeline.
- Walk through the {doc}`tutorials/index` for end-to-end examples.
- Consult the {doc}`api/index` for the full public API reference.
