#!/bin/bash
#SBATCH --job-name=rrg_baselines_rexval
#SBATCH --partition=gpu8_long
#SBATCH --gpus=4
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=10:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=rd3571@nyu.edu
#SBATCH --output=/gpfs/data/oermannlab/users/rd3571/ICARE_score/outputs/rexval/rexval_test_200/baselines/logs/%x-%j.out
#SBATCH --error=/gpfs/data/oermannlab/users/rd3571/ICARE_score/outputs/rexval/rexval_test_200/baselines/logs/%x-%j.err

# =============================================================================
# Run CXR-Report-Metric baselines (BLEU, BERTScore, s-emb, RadGraph, RadCliQ)
# on the RexVal 200-pair test set for fair comparison with ICARE.
#
# Usage (from login node):
#   cd /gpfs/data/oermannlab/users/rd3571/ICARE_score
#   sbatch scripts/rexval_data/rrg_eval_baselines.sh
# =============================================================================

set -eo pipefail

PROJECT_ROOT="/gpfs/data/oermannlab/users/rd3571/RRG_evaluation/CXR-Report-Metric"
RESULTS_DIR="/gpfs/data/oermannlab/users/rd3571/ICARE_score/outputs/rexval/rexval_test_200/baselines"
LOGS_DIR="${RESULTS_DIR}/logs"
REXVAL_CSV="/gpfs/data/oermannlab/users/rd3571/cxr_report_datasets/rexval/RexVal_test_icare_200.csv"
PREPARED_CSV="${RESULTS_DIR}/rexval_prepared.csv"

mkdir -p "$LOGS_DIR"

echo "--- Activating Conda Environment ---"
ORIG_DIR=$(pwd)
source ~/.bashrc
cd "$ORIG_DIR"
conda activate rrg-eval-clean
export PYTHONNOUSERSITE=1

cd "$PROJECT_ROOT" || { echo "Failed to change to $PROJECT_ROOT"; exit 1; }

echo "Job ID   : $SLURM_JOB_ID"
echo "Node     : $SLURM_JOB_NODELIST"
echo "Start    : $(date)"

# -----------------------------------------------------------------------------
# Step 1: Prepare RexVal CSV — add 'id' column required by run_bulk_eval.py
# (Unnamed: 0 is already 0-199 and unique per row, maps to Report_ID in ICARE)
# -----------------------------------------------------------------------------
echo ""
echo ">>> Step 1: Preparing RexVal CSV..."
python - <<'PYEOF'
import pandas as pd

src = "/gpfs/data/oermannlab/users/rd3571/cxr_report_datasets/rexval/RexVal_test_icare_200.csv"
dst = "/gpfs/data/oermannlab/users/rd3571/ICARE_score/outputs/rexval/rexval_test_200/baselines/rexval_prepared.csv"

df = pd.read_csv(src)
df = df.rename(columns={"Unnamed: 0": "id"})   # row index 0-199 → id (joins to Report_ID in ICARE)
df.to_csv(dst, index=False)
print(f"Prepared CSV written to {dst}  ({len(df)} rows)")
PYEOF
echo ">>> Step 1 complete."

# -----------------------------------------------------------------------------
# Step 2: Run CXR-Report-Metric
# -----------------------------------------------------------------------------
echo ""
echo ">>> Step 2: Running CXR-Report-Metric on RexVal..."
python run_bulk_eval.py \
    --input-files "${PREPARED_CSV}" \
    --output-dir  "${RESULTS_DIR}"
echo ">>> Step 2 complete."

# -----------------------------------------------------------------------------
# Step 3: Summarize average scores
# -----------------------------------------------------------------------------
echo ""
echo ">>> Step 3: Summarizing results..."
python /gpfs/data/oermannlab/users/rd3571/RRG_evaluation/CXR-Report-Metric/summarize_results.py \
    --results-dir "${RESULTS_DIR}"
echo ">>> Step 3 complete."

echo ""
echo "============================================="
echo "  RexVal baseline metrics complete."
echo "  Results: ${RESULTS_DIR}"
echo "============================================="
echo "End: $(date)"
