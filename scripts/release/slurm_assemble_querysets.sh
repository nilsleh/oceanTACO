#!/bin/bash
# Assemble published QuerySets once every measurement shard exists.
#
# Usage:
#   ARRAY=$(sbatch --parsable --array=0-7 scripts/release/slurm_build_querysets.sh)
#   sbatch --dependency=afterok:$ARRAY scripts/release/slurm_assemble_querysets.sh
#
#SBATCH --account=fm4eo2
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --job-name=oceantaco-assemble
#SBATCH --output=logs/assemble-%j.out
#SBATCH --error=logs/assemble-%j.err

set -euo pipefail

REPO="${REPO:-/p/project1/hai_uqmethodbox/nils/oceanTACO}"
TACO_PATH="${TACO_PATH:-/p/project1/hai_uqmethodbox/data/new_ssh_dataset_taco_folder/OceanTACO}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO}/release/querysets/v1}"

cd "${REPO}"
mkdir -p logs
source "${REPO}/venv_oceantaco/activate.sh"

# The working tree carries unrelated in-progress edits to dataset generation
# (download_sources.py, pyproject.toml), so --allow-dirty is required.  The
# builder still records the exact commit in the code_commit provenance field.
python scripts/release/build_querysets.py \
    --taco-path "${TACO_PATH}" \
    --output-root "${OUTPUT_ROOT}" \
    --patch-size 128 --patch-size 256 --patch-size 512 \
    --kind training --kind eval \
    --stage assemble \
    --allow-dirty
