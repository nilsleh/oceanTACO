#!/usr/bin/env bash
# Execute and validate the pinned-Hugging-Face tutorial notebooks in place.
set -euo pipefail

REPOSITORY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ACTIVATE_SCRIPT="/p/project1/hai_uqmethodbox/nils/oceanTACO/sc-venv-template-uv/activate.sh"
NOTEBOOK_CACHE_DIR="${OCEANTACO_NOTEBOOK_CACHE_DIR:-/tmp/oceantaco-tutorial-cache}"

source "${ACTIVATE_SCRIPT}"
unset PYTHONPATH
export HF_HOME="${NOTEBOOK_CACHE_DIR}/hf-home"
mkdir -p "${NOTEBOOK_CACHE_DIR}"
cd "${REPOSITORY_ROOT}"

notebooks=(
  docs/tutorials/ml_dataset.ipynb
  docs/tutorials/ml_configuration_cookbook.ipynb
  docs/tutorials/spatio_temporal_query_generation.ipynb
  docs/tutorials/data_retrieval_workflows.ipynb
  docs/tutorials/plot_hurricane_milton.ipynb
  docs/tutorials/plot_hurricane_milton_cross_product.ipynb
)

for notebook in "${notebooks[@]}"; do
  jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=1800 \
    --ExecutePreprocessor.kernel_name=python3 \
    "${notebook}"
done

python - "${notebooks[@]}" <<'PY'
from pathlib import Path
import sys

import nbformat

for value in sys.argv[1:]:
    path = Path(value)
    notebook = nbformat.read(path, as_version=4)
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type != "code":
            continue
        if not cell.get("outputs"):
            raise SystemExit(f"{path}: code cell {index} has no committed output")
        if any(output.output_type == "error" for output in cell.outputs):
            raise SystemExit(f"{path}: code cell {index} has an error output")
    print(f"verified {path}")
PY
