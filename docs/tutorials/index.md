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

A tutorial run touches only a few granules, so the cost is modest; the reason
to set this is the volume, not the size of any one draw.

`hf_hub_download` preserves the repository layout, which means a fetched
granule lands at
`$HF_HOME/hub/datasets--nilsleh--OceanTACO/snapshots/<revision>/DATA/<date>/<region>/`.
That directory has the same `COLLECTION.json` / `DATA/` / `METADATA/` shape as
a full local copy of the catalog, so **a populated snapshot directory is itself
a valid `taco_path`**. Remote access and a local catalog are not two
mechanisms; they are the same layout at different levels of completeness, which
is why OceanTACO decides per catalog row whether an asset is local rather than
reading it from configuration.

To work against a full local copy instead, pass
`CatalogConfig(taco_path="/path/to/OceanTACO")` and nothing is downloaded.

## The Hurricane Milton notebooks

The last two notebooks are visualization reproductions and additionally require
the repository's `ocean_taco.viz` helpers, which `[tutorials]` installs.

```{toctree}
:maxdepth: 1
:caption: Tutorials

ml_dataset
ml_configuration_cookbook
spatio_temporal_query_generation
data_retrieval_workflows
plot_hurricane_milton
plot_hurricane_milton_cross_product
```

## Editing these notebooks

The `.ipynb` files are generated. `scripts/dev/restore_tutorial_notebooks.py`
is the source of truth for their narrative and code; edits made directly to a
notebook are reverted the next time it runs. Regenerate with that script, then
execute with `scripts/dev/execute_tutorial_notebooks.py`, which records partial
output and a traceback into notebook metadata when a cell fails.
