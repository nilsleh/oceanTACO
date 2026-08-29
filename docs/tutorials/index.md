# Tutorials

The first four notebooks use the pinned Hugging Face Core revision and a shared
cache directory outside the repository. They download at most eight QuerySet
rows per draw. Install `.[tutorials,viz,hf]`, then execute and validate their
stored outputs with:

```sh
bash scripts/dev/execute_notebooks.sh
```

The Hurricane Milton notebooks are retained legacy visualization reproductions
and require the repository's `ocean_taco.viz` helpers.

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
