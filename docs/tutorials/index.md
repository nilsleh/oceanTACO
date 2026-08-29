# Tutorials

The first three notebooks below use the shipped QuerySet, retrieval, and PyTorch
APIs. They expect a published QuerySet and Core catalog access; set `taco_path`
in `CatalogConfig` to run against a local port. Documentation does not execute notebooks automatically; run them manually after installing `.[tutorials,viz]`. The Hurricane Milton notebooks
are retained legacy visualization reproductions and require the repository's
`ocean_taco.viz` helpers.

```{toctree}
:maxdepth: 1
:caption: Tutorials

ml_dataset
spatio_temporal_query_generation
data_retrieval_workflows
plot_hurricane_milton
plot_hurricane_milton_cross_product
```
