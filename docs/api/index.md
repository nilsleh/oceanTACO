# API Reference

The public package exposes reproducible sampling, native-coordinate retrieval,
and the PyTorch loader.

```python
from ocean_taco import CatalogConfig, GeoBox, PatchSize, PatchSpec, QuerySet, draw_queryset
from ocean_taco.retrieve import load_bbox_nc, load_hf_dataset, load_tile_nc
from ocean_taco.torch import CoreSourceLoader, OceanTACODataset, collate_ocean_samples
```

```{toctree}
:maxdepth: 1

dataset
queries
remote
```
