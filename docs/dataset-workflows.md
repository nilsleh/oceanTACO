# Native-coordinate retrieval workflows

The shipped retrieval surface is `ocean_taco.retrieve`. It uses named
`GeoBox`, `TimeRange`, and source-token objects so longitude order and temporal
selection are explicit.

## Open a catalog and crop one day

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

`load_bbox_nc` returns `None` when no matching asset exists. It merges every
intersecting Core region using decoded coordinate labels, then crops the named
box. Use `GeoBox(..., wraps_antimeridian=True)` only for an intentionally
wrapped longitude interval.

## Retrieve a named tile or a time series

```python
from ocean_taco import GeoBox, TimeRange
from ocean_taco.retrieve import load_multisource_time_series_nc, load_tile_nc

tile = load_tile_nc(catalog, "2024-06-01", "NORTH_ATLANTIC", "l4_ssh", config=config)
series = load_multisource_time_series_nc(
    catalog,
    tokens=("l4_sst", "l3_swot"),
    box=GeoBox(-80.0, -30.0, 25.0, 50.0),
    time=TimeRange("2024-06-01", "2024-06-07"),
    config=config,
)
```

The time-series keys are source tokens. Dense products are selected using their
decoded source timestamps; point products retain their ragged observations.

For ML batches and a parent-planned worker-safe loader, continue with
{doc}`dataset-ml-loader`.
