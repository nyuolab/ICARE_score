#!/bin/bash
#SBATCH --job-name=green_radpref
#SBATCH --partition=a100_long
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=rd3571@nyu.edu
#SBATCH --output=/gpfs/data/oermannlab/users/rd3571/ICARE_score/outputs/radpref/baselines/green/logs/%x-%j.out
#SBATCH --error=/gpfs/data/oermannlab/users/rd3571/ICARE_score/outputs/radpref/baselines/green/logs/%x-%j.err

# =============================================================================
# Run GREEN on the RadPref preference dataset (100 cases x 2 candidates).
#
# Usage:
#   cd /gpfs/data/oermannlab/users/rd3571/ICARE_score
#   sbatch scripts/radpref_data/green_eval_baselines.sh
# =============================================================================

set -eo pipefail

SCRIPT_ROOT="/gpfs/data/oermannlab/users/rd3571/ICARE_score"
RESULTS_DIR="${SCRIPT_ROOT}/outputs/radpref/baselines"
GREEN_DIR="${RESULTS_DIR}/green"
LOGS_DIR="${GREEN_DIR}/logs"
RADPREF_CSV="/gpfs/data/oermannlab/users/rd3571/cxr_report_datasets/radpref/radpref_icare.csv"
PREPARED_CSV="${RESULTS_DIR}/radpref_prepared.csv"

mkdir -p "${LOGS_DIR}"

echo "--- Activating Conda Environment ---"
ORIG_DIR=$(pwd)
source ~/.bashrc
cd "$ORIG_DIR"
conda activate regularization
export PYTHONNOUSERSITE=1
if ! python -c "import torch, transformers, datasets"; then
    echo "Error: regularization env missing GREEN dependencies. Active python: $(which python)"
    exit 1
fi
echo "Using python: $(which python)"

cd "${SCRIPT_ROOT}" || { echo "Failed to change to ${SCRIPT_ROOT}"; exit 1; }

echo "Job ID     : ${SLURM_JOB_ID}"
echo "Node       : ${SLURM_JOB_NODELIST}"
echo "Start      : $(date)"

echo ""
echo ">>> Step 1: Preparing RadPref CSV..."
python - <<'PYEOF'
import pandas as pd

src = "/gpfs/data/oermannlab/users/rd3571/cxr_report_datasets/radpref/radpref_icare.csv"
dst = "/gpfs/data/oermannlab/users/rd3571/ICARE_score/outputs/radpref/baselines/radpref_prepared.csv"

df = pd.read_csv(src)
df.insert(0, "id", range(len(df)))
df.to_csv(dst, index=False)
print(f"Prepared CSV written to {dst}  ({len(df)} rows)")
PYEOF
echo ">>> Step 1 complete."

echo ""
echo ">>> Step 2: Running GREEN..."
python src/run_green_eval.py \
    --input-csv "${PREPARED_CSV}" \
    --output-dir "${GREEN_DIR}"
echo ">>> Step 2 complete."

echo ""
echo "============================================="
echo "  RadPref GREEN complete."
echo "  Results: ${GREEN_DIR}"
echo "  Note: rows 0-99 = C1 candidates,"
echo "        rows 100-199 = C2 candidates."
echo "============================================="
echo "End: $(date)"
