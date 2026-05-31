#!/bin/bash
#SBATCH --job-name=rrg_baselines_radpref
#SBATCH --partition=gpu8_long
#SBATCH --gpus=4
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=10:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=rd3571@nyu.edu
#SBATCH --output=/gpfs/data/oermannlab/users/rd3571/ICARE_score/outputs/radpref/baselines/logs/%x-%j.out
#SBATCH --error=/gpfs/data/oermannlab/users/rd3571/ICARE_score/outputs/radpref/baselines/logs/%x-%j.err

# =============================================================================
# Run CXR-Report-Metric baselines (BLEU, BERTScore, s-emb, RadGraph, RadCliQ)
# on the RadPref preference dataset (100 cases × 2 candidates) for fair
# comparison with ICARE.
#
# Usage (from login node):
#   cd /gpfs/data/oermannlab/users/rd3571/ICARE_score
#   sbatch scripts/radpref_data/rrg_eval_baselines.sh
# =============================================================================

set -eo pipefail

PROJECT_ROOT="/gpfs/data/oermannlab/users/rd3571/RRG_evaluation/CXR-Report-Metric"
RESULTS_DIR="/gpfs/data/oermannlab/users/rd3571/ICARE_score/outputs/radpref/baselines"
LOGS_DIR="${RESULTS_DIR}/logs"
RADPREF_CSV="/gpfs/data/oermannlab/users/rd3571/cxr_report_datasets/radpref/radpref_icare.csv"
PREPARED_CSV="${RESULTS_DIR}/radpref_prepared.csv"

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
# Step 1: Prepare RadPref CSV — add 'id' column required by run_bulk_eval.py.
# The radpref CSV has no Unnamed: 0; we use a sequential row index (0-based)
# so that id aligns with Report_ID in the ICARE output for direct comparison.
# -----------------------------------------------------------------------------
echo ""
echo ">>> Step 1: Preparing RadPref CSV..."
python - <<'PYEOF'
import pandas as pd

src = "/gpfs/data/oermannlab/users/rd3571/cxr_report_datasets/radpref/radpref_icare.csv"
dst = "/gpfs/data/oermannlab/users/rd3571/ICARE_score/outputs/radpref/baselines/radpref_prepared.csv"

df = pd.read_csv(src)
df.insert(0, "id", range(len(df)))   # sequential 0-based id → aligns to Report_ID in ICARE
df.to_csv(dst, index=False)
print(f"Prepared CSV written to {dst}  ({len(df)} rows)")
PYEOF
echo ">>> Step 1 complete."

# -----------------------------------------------------------------------------
# Step 2: Run CXR-Report-Metric
# -----------------------------------------------------------------------------
echo ""
echo ">>> Step 2: Running CXR-Report-Metric on RadPref..."
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
echo "  RadPref baseline metrics complete."
echo "  Results: ${RESULTS_DIR}"
echo "  Note: rows 0-99 = C1 candidates,"
echo "        rows 100-199 = C2 candidates."
echo "        Split on 'candidate' column to compare."
echo "============================================="
echo "End: $(date)"
