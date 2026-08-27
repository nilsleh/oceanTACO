#!/bin/bash
# Verify the published QuerySets: round-trip + invariants, independent
# spot-check against released measurement functions, and map reports.
#SBATCH --account=fm4eo2
#SBATCH --partition=batch
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64G --time=03:00:00
#SBATCH --job-name=oceantaco-checks
#SBATCH --output=logs/checks-%j.out
#SBATCH --error=logs/checks-%j.err
set -euo pipefail
REPO="${REPO:-/p/project1/hai_uqmethodbox/nils/oceanTACO}"
ROOT="${ROOT:-release/querysets/v1}"
cd "${REPO}"
source venv_oceantaco/activate.sh

echo "=========== ROUND-TRIP + INVARIANTS ==========="
python scripts/release/checks/verify_published.py --positions-only --root "${ROOT}"

echo "=========== OCEAN-MASK COVERAGE ==========="
python scripts/release/checks/check_grid_coverage.py --root "${ROOT}"

echo "=========== BLIND POSITION CHECK ==========="
python scripts/release/checks/check_blind_positions.py --root "${ROOT}"

echo "=========== SOURCE RECOVERY ==========="
python scripts/release/checks/check_source_recovery.py --root "${ROOT}"

echo "=========== ROUND-TRIP + INVARIANTS ==========="
python scripts/release/checks/verify_published.py --root "${ROOT}"

echo "=========== INDEPENDENT SPOT-CHECK ==========="
python scripts/release/checks/spot_check_published.py --root "${ROOT}"

echo "=========== MAP REPORTS ==========="
for s in 512 256 128; do
  python -m ocean_taco.viz.queryset_maps \
    --train "${ROOT}/$s-training" --eval "${ROOT}/$s-eval" \
    --output "${ROOT}/queryset-map-$s.pdf"
done
echo "=========== ALL CHECKS DONE ==========="
