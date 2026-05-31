#!/bin/bash
#SBATCH --job-name=crimson_rexval
#SBATCH --partition=gpu8_long
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=40:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=rd3571@nyu.edu
#SBATCH --output=/gpfs/data/oermannlab/users/rd3571/ICARE_score/outputs/rexval/rexval_test_200/baselines/logs/%x-%j.out
#SBATCH --error=/gpfs/data/oermannlab/users/rd3571/ICARE_score/outputs/rexval/rexval_test_200/baselines/logs/%x-%j.err

# =============================================================================
# Run CRIMSON metric on RexVal 200-pair test set as a baseline score.
#
# CRIMSON uses medgemma-4b-it-crimson (HuggingFace) by default.
# No patient context (age/indication) is available in RexVal.
#
# Usage (from any directory):
#   sbatch /gpfs/data/oermannlab/users/rd3571/ICARE_score/scripts/rexval_data/crimson_baselines_rexval.sh
# =============================================================================

set -eo pipefail

CRIMSON_DIR="/gpfs/data/oermannlab/users/rd3571/CRIMSON"
REXVAL_CSV="/gpfs/data/oermannlab/users/rd3571/cxr_report_datasets/rexval/RexVal_test_icare_200.csv"
RESULTS_DIR="/gpfs/data/oermannlab/users/rd3571/ICARE_score/outputs/rexval/rexval_test_200/baselines"
OUTPUT_JSON="${RESULTS_DIR}/crimson_results.json"
LOGS_DIR="${RESULTS_DIR}/logs"
BATCH_SIZE="${BATCH_SIZE:-8}"   # HuggingFace forward-pass batch size

mkdir -p "$LOGS_DIR"

echo "--- Activating Conda Environment ---"
ORIG_DIR=$(pwd)
source ~/.bashrc
cd "$ORIG_DIR"
conda activate crimson
# Clear PYTHONPATH to prevent base conda's Python 3.8 packages from
# contaminating the crimson (Python 3.12) environment.
unset PYTHONPATH

cd "$CRIMSON_DIR" || { echo "Failed to change to $CRIMSON_DIR"; exit 1; }

echo "Job ID     : $SLURM_JOB_ID"
echo "Node       : $SLURM_JOB_NODELIST"
echo "Start      : $(date)"
echo "Input      : ${REXVAL_CSV}"
echo "Output     : ${OUTPUT_JSON}"
echo "Batch size : ${BATCH_SIZE}"
echo ""

python evaluate_reports.py \
    --input        "${REXVAL_CSV}" \
    --gt-column    ground_truth_report \
    --pred-column  generated_report \
    --output       "${OUTPUT_JSON}" \
    --details \
    --batch-size   "${BATCH_SIZE}"

echo ""
echo "============================================="
echo "  CRIMSON baseline complete."
echo "  Results: ${OUTPUT_JSON}"
echo "============================================="
echo "End: $(date)"
