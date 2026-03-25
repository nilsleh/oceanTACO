# OceanTACO Dataset Workflows

This guide focuses on data retrieval and access patterns before ML batching.

- Use this page when you want `xarray.Dataset` outputs for inspection, or plotting.
- For ML query sampling with `OceanTACODataset`, see {doc}`dataset-ml-loader`.

## 1) Install Profiles

```sh
# Main usage (dataset + queries + visualization)
pip install -e .

# Add data generation pipeline deps
pip install -e ".[generate]"

# Add Hugging Face client libs for direct download/stream examples
pip install -e ".[hf]"
```

## 2) Main Retrieval APIs

From `ocean_taco/dataset/remote.py`:

- `load_hf_dataset`: open OceanTACO catalog from HuggingFace.
- `load_bbox_nc`: load and merge all tiles overlapping a bbox for one source.
- `load_tile_nc`: load exactly one named region tile.
- `load_region_product_nc`: alias for loading one region product file.
- `load_bbox_swot_nc`: helper for SWOT bbox loading.
- `load_multisource_time_series_nc`: load multiple sources over a date range
    and stack each source over `time`.

From `datasets` / `huggingface_hub`:

- `load_dataset(..., streaming=True)`: iterate rows without full download.
- `snapshot_download(...)`: download full dataset snapshot locally.

## 3) End-to-End Retrieval Example

```python
from ocean_taco.dataset.retrieve import HF_DEFAULT_URL, load_hf_dataset, load_bbox_nc

# 1) Open remote catalog
catalog = load_hf_dataset(HF_DEFAULT_URL)

# 2) Load one source over one bbox/date
sst = load_bbox_nc(
    dataset=catalog,
    date="2024-06-01",
    bbox=(-80, -30, 25, 50),
    data_source="l4_sst",
    cache_dir="./cache",  # optional
)

print(sst)
```

## 4) Retrieval Pattern: One Exact Tile

Use this when you already know the region name (for reproducible region-level analysis).

```python
from ocean_taco.dataset.retrieve import load_hf_dataset, load_tile_nc

catalog = load_hf_dataset()
ds = load_tile_nc(
    dataset=catalog,
    date="2024-06-01",
    tile="NORTH_ATLANTIC",
    data_source="l4_ssh",
    cache_dir="./cache",
)
```

## 5) Retrieval Pattern: Stream Records

Use this for record-level streaming with HuggingFace Datasets.

```python
from datasets import load_dataset

stream = load_dataset("nilsleh/OceanTACO", split="train", streaming=True)
row = next(iter(stream))
print(row)
```

## 6) Retrieval Pattern: Local Full Snapshot

Use this when offline access or repeated full scans are needed.

```python
from huggingface_hub import snapshot_download

local_dir = snapshot_download(repo_id="nilsleh/OceanTACO", repo_type="dataset")
print(local_dir)
```

## 7) Retrieval Pattern: Multiple Sources, Same Region, Date Range

Use this when you want a small time series bundle over one fixed region, with multiple products loaded for each day.

```python
from ocean_taco.dataset.retrieve import (
    HF_DEFAULT_URL,
    load_hf_dataset,
    load_multisource_time_series_nc,
)

catalog = load_hf_dataset(HF_DEFAULT_URL)

sources = ["l4_ssh", "l4_sst", "l4_sss", "glorys"]

# Option A: fixed named region tile
stacked_by_source = load_multisource_time_series_nc(
    dataset=catalog,
    data_sources=sources,
    date_start="2024-06-01",
    date_end="2024-06-07",
    tile="NORTH_ATLANTIC",
    cache_dir="./cache",  # optional
)

# Access by source name; each dataset is stacked on time
print(stacked_by_source["l4_sst.nc"])
```

You can use the same utility with a custom bbox:

```python
stacked_bbox = load_multisource_time_series_nc(
    dataset=catalog,
    data_sources=["l4_ssh", "l4_sst"],
    date_start="2024-06-01",
    date_end="2024-06-07",
    bbox=(-80, 25, -30, 50),
    cache_dir="./cache",
)
```

Notes:

- return type is `dict[source_filename, xr.Dataset | None]`
- keys are normalized filenames, e.g. `l4_sst.nc`
- values can be `None` when no days were found for that source in the range
- pass exactly one of `tile` or `bbox`

## 8) Choosing Retrieval Mode

| Need | Recommended method |
|---|---|
| One date + bbox + source | `load_bbox_nc` |
| One known region tile | `load_tile_nc` |
| Multi-source time series over one region | `load_multisource_time_series_nc(..., tile=...)` |
| Multi-source time series over a custom area | `load_multisource_time_series_nc(..., bbox=...)` |
| Record-by-record streaming | `datasets.load_dataset(..., streaming=True)` |
| Full local copy | `snapshot_download` |

## 9) Next Step: ML Loader Workflows

After retrieval validation, move to query-based sampling and DataLoader integration in {doc}`dataset-ml-loader`.
