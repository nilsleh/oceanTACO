# OceanTACO ML Loader Workflows

This guide explains how to use `OceanTACODataset` for machine learning workflows.

Use this Dataset when you need:

- repeated sampling over many spatial-temporal windows
- train/eval query generation
- PyTorch `DataLoader` integration with variable-size ocean patches

For direct `xarray` retrieval and inspection, see {doc}`dataset-workflows`.

## 1) Why `OceanTACODataset` for ML

`OceanTACODataset` is designed for query-driven sampling rather than one-off file reads.

Core behavior:

- pre-indexes matching files once at initialization
- loads and merges tiles at `__getitem__` time for each query
- returns structured dictionaries (`inputs`, `targets`, `coords`, `metadata`) ready for model pipelines

This is typically the right interface for training loops, evaluation grids, and reproducible experiments.

## 2) Local and Remote Usage

`OceanTACODataset` accepts `taco_path` values supported by `tacoreader.load`, including:

- local OceanTACO dataset paths
- remote HuggingFace URL (for streaming access)

So the main choice is not local vs remote, but:

- direct retrieval (`xarray`) for analysis
- query-based loader (`OceanTACODataset`) for ML

## 3) End-to-End Minimal ML Example

```python
from torch.utils.data import DataLoader

from ocean_taco.dataset import OceanTACODataset, collate_ocean_samples
from ocean_taco.dataset.queries import QueryGenerator, PatchSize
from ocean_taco.dataset.retrieve import HF_DEFAULT_URL

# 1) Generate training queries
generator = QueryGenerator(land_mask_path=".ocean_mask_cache/land_mask.npy")
queries = generator.generate_training_queries(
    n_queries=256,
    patch_size=PatchSize(2.0, "deg"),
    date_range=("2024-01-01", "2024-03-31"),
    seed=42,
)

# 2) Build dataset (remote streaming path shown; local path also works)
dataset = OceanTACODataset(
    taco_path=HF_DEFAULT_URL,  # or "/path/to/OceanTACO"
    queries=queries,
    input_variables=["l4_ssh", "l4_sst", "glorys_sss"],
    target_variables=["l3_swot"],
    temporal_agg="mean",
)

# 3) Build dataloader
loader = DataLoader(
    dataset,
    batch_size=16,
    shuffle=True,
    num_workers=4,
    collate_fn=collate_ocean_samples,
)

batch = next(iter(loader))
print(batch.keys())
```

## 4) Query Design: Training vs Evaluation

Use random ocean-biased queries for training, and deterministic sliding windows for evaluation.

```python
from ocean_taco.dataset.queries import PatchSize

train_queries = generator.generate_training_queries(
    n_queries=1000,
    patch_size=PatchSize(2.0, "deg"),
    date_range=("2024-01-01", "2024-03-31"),
)

eval_queries = generator.generate_eval_queries(
    bbox=(-80, -20, 20, 50),
    patch_size=PatchSize(2.0, "deg"),
    date_range=("2024-01-01", "2024-01-31"),
    spatial_overlap=0.25,
    temporal_stride_days=3,
)
```

## 5) Patch Size Guidance

For target grid resolution `resolution_deg` and desired patch width `pixels`:

- `patch_deg ~= pixels * resolution_deg`

Examples:

- `0.25 deg` data and `64` pixels -> `~16.0 deg`
- `0.10 deg` data and `128` pixels -> `~12.8 deg`

For km-based patch definitions:

```python
from ocean_taco.dataset.queries import PatchSize

patch = PatchSize(100, "km")
lon_deg, lat_deg = patch.to_degrees(center_lat=45.0)
print(lon_deg, lat_deg)
```

Notes:

- latitude degrees are about `111 km/deg`
- longitude degrees shrink with latitude (`cos(latitude)`)

## 6) Query Set Persistence

Persist generated query sets for reproducibility.

```python
from ocean_taco.dataset.queries import QueryGenerator

QueryGenerator.save_queries(train_queries, "queries/train")
loaded_queries, metadata = QueryGenerator.load_queries("queries/train")
print(len(loaded_queries), metadata)
```

This writes:

- `queries/train.parquet`
- `queries/train.json`

## 7) Practical Checklist

Before long runs:

- confirm variable tokens against {doc}`dataset_description`
- verify one sample with `dataset[0]` and inspect `metadata`
- validate DataLoader collation with a small `batch_size`
- save query sets so experiments can be repeated exactly

## 8) Related Pages

- {doc}`getting_started`: quick decision guide and first examples
- {doc}`dataset-workflows`: retrieval and streaming workflows
- {doc}`tutorials/index`: end-to-end runnable notebooks