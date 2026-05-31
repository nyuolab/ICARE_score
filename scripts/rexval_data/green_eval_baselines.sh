#!/bin/bash
#SBATCH --job-name=green_rexval
#SBATCH --partition=a100_long
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=rd3571@nyu.edu
#SBATCH --output=/gpfs/data/oermannlab/users/rd3571/ICARE_score/outputs/rexval/rexval_test_200/baselines/green/logs/%x-%j.out
#SBATCH --error=/gpfs/data/oermannlab/users/rd3571/ICARE_score/outputs/rexval/rexval_test_200/baselines/green/logs/%x-%j.err

# =============================================================================
# Run GREEN on the RexVal 200-pair test set.
#
# Usage:
#   cd /gpfs/data/oermannlab/users/rd3571/ICARE_score
#   sbatch scripts/rexval_data/green_eval_baselines.sh
# =============================================================================

set -eo pipefail

SCRIPT_ROOT="/gpfs/data/oermannlab/users/rd3571/ICARE_score"
RESULTS_DIR="${SCRIPT_ROOT}/outputs/rexval/rexval_test_200/baselines"
GREEN_DIR="${RESULTS_DIR}/green"
LOGS_DIR="${GREEN_DIR}/logs"
REXVAL_CSV="/gpfs/data/oermannlab/users/rd3571/cxr_report_datasets/rexval/RexVal_test_icare_200.csv"
PREPARED_CSV="${RESULTS_DIR}/rexval_prepared.csv"

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
echo ">>> Step 1: Preparing RexVal CSV..."
python - <<'PYEOF'
import pandas as pd

src = "/gpfs/data/oermannlab/users/rd3571/cxr_report_datasets/rexval/RexVal_test_icare_200.csv"
dst = "/gpfs/data/oermannlab/users/rd3571/ICARE_score/outputs/rexval/rexval_test_200/baselines/rexval_prepared.csv"

df = pd.read_csv(src)
df = df.rename(columns={"Unnamed: 0": "id"})
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
echo "  RexVal GREEN complete."
echo "  Results: ${GREEN_DIR}"
echo "============================================="
echo "End: $(date)"
