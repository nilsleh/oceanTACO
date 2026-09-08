# Tutorials

These notebooks run against the pinned Hugging Face Core revision and need no
configuration: install the package and execute them.

```sh
pip install "oceantaco[tutorials]"
```

Each notebook builds its own `CatalogConfig()` and fetches its QuerySet with
`QuerySet.from_hub(256, "eval")`, so no paths, cache directories, or
environment variables are involved. Draws are capped at a handful of rows, and
only the granules those rows need are downloaded.

## Where the data lands

Remote assets are fetched with `huggingface_hub`, which caches them under
`HF_HOME`. The default is `~/.cache/huggingface`, and because the full Core
catalog is 291 GB, that default sits on a small or quota-limited volume often
enough to be worth redirecting:

```sh
export HF_HOME=/path/with/space/hf-home
```

A tutorial run touches only a few granules, so the cost is modest. The reason
to set this is the size of the volume rather than the size of any one draw.

`hf_hub_download` preserves the repository layout, which means a fetched
granule lands at
`$HF_HOME/hub/datasets--nilsleh--OceanTACO/snapshots/<revision>/DATA/<date>/<region>/`.
Loading a remote catalog also fetches `COLLECTION.json` and `METADATA/`, about
0.7 MB, so the snapshot has the same shape as a full local copy and **a
populated snapshot directory is itself a valid `taco_path`** for the granules
it contains. A remote snapshot and a full local copy share one layout at
different levels of completeness, which is why OceanTACO works out per catalog
row whether an asset is already local instead of reading that from
configuration.

## Loader configuration

The tutorials continue to use PyTorch `DataLoader`. OceanTACO's source adapter
plans catalog requests in the parent and reuses cropped reads within each batch.
`CatalogConfig(max_open_files=16)` limits each process's source-loader file cache,
including local files opened without `cache_dir`. For repeated epochs, reuse one
DataLoader with persistent workers; the single-batch illustrations do not need
them. See [worker usage](ml_dataset.ipynb) and the
[throughput validation report](../throughput-validation.md).

## What each notebook covers

Read them in the order below. The first is the entry point and links out to the
other three.

| Notebook | Role |
|---|---|
| `ml_dataset` | Overview and entry point: the four stages from a published QuerySet to a rendered sample. |
| `data_retrieval_workflows` | QuerySet and filter deep-dive: selection, coverage evidence, and the lower-level retrieval API. |
| `spatio_temporal_query_generation` | ML use cases (forecasting, midpoint retrieval, super-resolution, multi-source) and the working training loader. |
| `ml_configuration_cookbook` | Renderer reference, organised by renderer. |
| `plot_hurricane_milton*` | Paper-figure reproductions. |

## The Hurricane Milton notebooks

The last two notebooks are visualization reproductions and additionally require
the repository's `ocean_taco.viz` helpers, which `[tutorials]` installs.

```{toctree}
:maxdepth: 1
:caption: Tutorials

ml_dataset
data_retrieval_workflows
spatio_temporal_query_generation
ml_configuration_cookbook
plot_hurricane_milton
plot_hurricane_milton_cross_product
```

## Editing these notebooks

The `.ipynb` files are generated. `scripts/dev/restore_tutorial_notebooks.py`
is the source of truth for their narrative and code, and edits made directly to
a notebook are reverted the next time it runs. The generator currently differs
from the checked-in notebooks in some workflow and prose sections, so review
generated diffs before replacing those notebooks. Loader guidance is maintained
in both locations. After regeneration, execute with
`scripts/dev/execute_tutorial_notebooks.py`, which records partial
output and a traceback into notebook metadata when a cell fails.
