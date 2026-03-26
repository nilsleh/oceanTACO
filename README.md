# OceanTACO

[![docs](https://app.readthedocs.org/projects/oceantaco/badge/?version=latest)](https://oceantaco.readthedocs.io/en/latest/)
[![pypi](https://badge.fury.io/py/oceantaco.svg)](https://pypi.org/project/ocean-taco/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Language: Python](https://img.shields.io/badge/language-Python%203.11%2B-green?logo=python&logoColor=green)](https://www.python.org)


OceanTACO is a multi-source sea surface variable dataset for Earth-system analysis workflows and also provides dataloaders for machine learning workflows.

Documentation page: https://ocean-taco.readthedocs.io/en/latest/

Current dataset coverage includes:

- Sea surface height (SSH): L4, L3, SWOT L3
- Sea surface temperature (SST): L4, L3, GLORYS
- Sea surface salinity (SSS): L4, L3 SMOS
- Argo float profile observations (point-source)
- Additional co-located sources including wind and GLORYS currents

The Core dataset spans 2023-03-29 until 2025-08-01 and includes the SWOT data. It is available on [Hugging Face](https://huggingface.co/datasets/nilsleh/OceanTACO). The extended dataset spans 2015-01-01 until 2023-03-29 but preceeds the SWOT era and is available on [Hugging Face](https://huggingface.co/datasets/nilsleh/OceanTACO_extended).

<img src="docs/images/oceantaco.svg" alt="OceanTACO Figure" width="760" />

Generated directly from OceanTACO sources (GLORYS SST, SSH L4, SSH SWOT, SST L4, SSS L4, Argo).

To regenerate the README figure:

```sh
python ocean_taco/viz/readme_figure.py --date 2025-03-04
```

## Documentation and Notebooks

If you are new to OceanTACO, start with the hosted documentation and tutorials:

- Documentation home: https://oceantaco.readthedocs.io/en/latest/
- Getting started guide: https://oceantaco.readthedocs.io/en/latest/getting_started.html
- Dataset workflows: https://oceantaco.readthedocs.io/en/latest/dataset-workflows.html
- ML dataset loader guide: https://oceantaco.readthedocs.io/en/latest/dataset-ml-loader.html
- Tutorial notebooks index: https://oceantaco.readthedocs.io/en/latest/tutorials/index.html

OceanTACO includes several tutorial notebooks in the docs, with rendered outputs and downloadable `.ipynb` files, so you can get started quickly before writing your own workflows.

## Installation

Most users should install directly from PyPI:

```sh
# Core package
pip install ocean-taco

# With Hugging Face helpers
pip install "ocean-taco[hf]"
```

If you want the latest development version from GitHub:

```sh
pip install "ocean_taco[hf] @ git+https://github.com/nilsleh/oceanTACO.git@main"
```

If you have cloned this repository and want a local editable install, run the following from the repository root:

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
- `docs/tutorials/index.md`

## CODE LICENSE

The Code is licensed under Apache - 2.0.

## DATASET LICENSE

OceanTACO is released under Creative Commons Attribution 4.0 International (CC BY 4.0). However, please see the [OceanTACO Dataset Card](https://huggingface.co/datasets/nilsleh/OceanTACO) for full license information, and required attribution, acknowledgements and citations.

## Citation
In Progress.