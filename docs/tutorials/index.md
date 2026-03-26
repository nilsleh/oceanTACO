# Tutorials

The tutorials below demonstrate end-to-end workflows with OceanTACO.
They are rendered from Jupyter notebooks — all outputs are pre-computed so you can read them without running anything.

When you open any tutorial notebook page, use the top-right action buttons to run or download it quickly:

- `Open in Colab`: launches the notebook directly in Google Colab.
- `Launch on Binder`: starts a temporary cloud Jupyter environment (can take 1-2 minutes to start).
- `Download`: download the `.ipynb` notebook source file.
- `Edit this page`: jump to GitHub to suggest edits.

To run the notebooks yourself:

```sh
conda activate testpy311
pip install "ocean_taco[hf] @ git+https://github.com/nilsleh/oceanTACO.git@main"
jupyter lab notebooks/
```

---

```{toctree}
:maxdepth: 1
:caption: Tutorials

data_retrieval_workflows
ml_dataset
spatio_temporal_query_generation
plot_hurricane_milton
plot_hurricane_milton_cross_product
```
