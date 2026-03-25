# OceanTACO

OceanTACO is a multi-source sea surface variable dataset with dataloaders for machine learning workflows.

## Installation

From the repository root, we provide different dependency groups depending on what workflows you are interested in

```sh
# Dataset loading + queries + visualization (default profile)
pip install -e .

# Add dataset-generation dependencies (download/format/build pipeline)
pip install -e ".[generate]"

# Add Hugging Face client helpers for direct download/stream examples
pip install -e ".[hf]"

# Full development profile
pip install -e ".[generate,hf,tests]"
```

## Repository Structure

- `ocean_taco/dataset/`: main user API for loading data and generating queries.
- `ocean_taco/generate_dataset/`: data acquisition and dataset build pipeline.
- `ocean_taco/viz/`: visualization and analysis scripts.
- `notebooks/`: tutorial and task-focused notebooks.

## Dataset + Queries

Most users will interact with `ocean_taco/dataset/dataset.py` and `ocean_taco/dataset/queries.py`.

```python
from ocean_taco.dataset import OceanTACODataset, QueryGenerator, PatchSize

ds = OceanTACODataset(
	taco_path="/path/to/OceanTACO",
	input_variables=["l4_ssh", "l4_sst", "glorys_sss"],
	target_variables=["l3_swot"],
	temporal_agg="mean",
)

generator = QueryGenerator(land_mask_path=".ocean_mask_cache/land_mask.npy")
queries = generator.generate_training_queries(
	n_queries=32,
	patch_size=PatchSize(1.0, "deg"),
	date_range=("2024-01-01", "2024-01-31"),
	max_land_fraction=0.3,
)

sample = ds[queries[0].to_geoslice()]
```

## Patch Size from Resolution

Use this rule of thumb when choosing patch extents:

- `patch_deg ≈ pixels * resolution_deg`
- Example: for ~`0.1°` data and `64 x 64` patches, choose about `6.4°` patch size.

`PatchSize(unit="km")` is also supported; conversion to degrees is latitude-dependent for longitude.

## Hugging Face Dataset Access

OceanTACO is available at:

- https://huggingface.co/datasets/nilsleh/OceanTACO

### Stream OceanTACO

TACO is cloud-native. Therefore, the dataset can be accessed remotely without downloading it to local disk, both for data analysis and machine learning workflows.

Instead of specifying a local path to the TACO, point to the Huggingface repository


### Download snapshot locally (huggingface_hub)

```python
from huggingface_hub import snapshot_download

local_dir = snapshot_download(repo_id="nilsleh/OceanTACO", repo_type="dataset")
print(local_dir)
```

## More Detailed Guide

For full examples (query save/load, train/eval dataloaders, patch-size recipes, and troubleshooting), see:

- `docs/dataset-workflows.md`

## Documentation

OceanTACO uses Sphinx for project documentation.

Install docs dependencies:

```sh
pip install -e ".[docs]"
```

Build the docs locally:

```sh
make docs-build
```

The generated HTML site is written to:

- `docs/_build/html/`

Serve the built docs locally:

```sh
make docs-serve
```

Then open:

- `http://localhost:8080`

Clean docs build artifacts:

```sh
make docs-clean
```

## PyPI Publishing

This repository includes automated publishing workflows using Trusted Publishing (OIDC):

- `.github/workflows/publish-testpypi.yml` for manual TestPyPI releases
- `.github/workflows/publish-pypi.yml` for PyPI releases on GitHub Release publish

One-time setup:

1. Create the project on TestPyPI and PyPI (or reserve the name).
2. In TestPyPI and PyPI, configure a Trusted Publisher for:
	- owner: `nilsleh`
	- repository: `oceanTACO`
	- workflow files: `publish-testpypi.yml` and `publish-pypi.yml`
3. Ensure GitHub environments `testpypi` and `pypi` exist (recommended for approval gates).

Release flow:

1. Update `version` in `pyproject.toml`.
2. Run TestPyPI publish manually from GitHub Actions (`Publish to TestPyPI`).
3. Create a GitHub Release to trigger production PyPI publish (`Publish to PyPI`).

## CODE LICENSE

The Code is licensed under Apache - 2.0.

## DATASET LICENSE

OceanTACO is released under Creative Commons Attribution 4.0 International (CC BY 4.0). However, please see the [OceanTACO Dataset Card](https://huggingface.co/datasets/nilsleh/OceanTACO) for full license information, and required attribution, acknowledgements and citations.