#!/bin/bash
# Build the released OceanTACO QuerySets on the CPU batch partition.
#
# Usage:
#   sbatch --array=0-7 scripts/release/slurm_build_querysets.sh
#
# Each array task measures the dates congruent to its task id modulo the array
# width and writes one shard per date.  Shards are content-addressed by
# plan_id, so an overlapping or re-run task recomputes identical bytes rather
# than corrupting a partial result.  Assembly is a separate, single dependent
# job because it must see every shard.
#
#SBATCH --account=fm4eo2
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=90G
#SBATCH --time=02:00:00
#SBATCH --job-name=oceantaco-querysets
#SBATCH --output=logs/querysets-%A_%a.out
#SBATCH --error=logs/querysets-%A_%a.err

set -euo pipefail

REPO="${REPO:-/p/project1/hai_uqmethodbox/nils/oceanTACO}"
TACO_PATH="${TACO_PATH:-/p/project1/hai_uqmethodbox/data/new_ssh_dataset_taco_folder/OceanTACO}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO}/release/querysets/v1}"
# Worker memory, not core count, is the binding constraint: each worker holds
# the global SWOT primary and n_obs grids plus the SSH grid.
JOBS="${JOBS:-24}"

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
    --stage measure \
    --jobs "${JOBS}" \
    --shard-index "${SLURM_ARRAY_TASK_ID:-0}" \
    --shard-count "${SLURM_ARRAY_TASK_COUNT:-1}" \
    --allow-dirty
