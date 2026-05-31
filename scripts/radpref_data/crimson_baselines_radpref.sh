#!/bin/bash
#SBATCH --job-name=crimson_radpref
#SBATCH --partition=a100_long
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=30:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=rd3571@nyu.edu
#SBATCH --output=/gpfs/data/oermannlab/users/rd3571/ICARE_score/outputs/radpref/baselines/logs/%x-%j.out
#SBATCH --error=/gpfs/data/oermannlab/users/rd3571/ICARE_score/outputs/radpref/baselines/logs/%x-%j.err

# =============================================================================
# Run CRIMSON metric on RadPref preference dataset (100 cases × 2 candidates)
# as a baseline score.
#
# CRIMSON uses medgemma-4b-it-crimson (HuggingFace) by default.
# The input CSV contains both C1 and C2 candidate rows; CRIMSON scores each
# generated_report against the ground_truth_report independently.
#
# Usage (from any directory):
#   sbatch /gpfs/data/oermannlab/users/rd3571/ICARE_score/scripts/radpref_data/crimson_baselines_radpref.sh
# =============================================================================

set -eo pipefail

CRIMSON_DIR="/gpfs/data/oermannlab/users/rd3571/CRIMSON"
RADPREF_CSV="/gpfs/data/oermannlab/users/rd3571/cxr_report_datasets/radpref/radpref_icare.csv"
RESULTS_DIR="/gpfs/data/oermannlab/users/rd3571/ICARE_score/outputs/radpref/baselines"
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
echo "Input      : ${RADPREF_CSV}"
echo "Output     : ${OUTPUT_JSON}"
echo "Batch size : ${BATCH_SIZE}"
echo ""

python evaluate_reports.py \
    --input        "${RADPREF_CSV}" \
    --gt-column    ground_truth_report \
    --pred-column  generated_report \
    --output       "${OUTPUT_JSON}" \
    --details \
    --batch-size   "${BATCH_SIZE}"

echo ""
echo "============================================="
echo "  CRIMSON baseline complete."
echo "  Results: ${OUTPUT_JSON}"
echo "  Note: rows 0-99 = C1 candidates,"
echo "        rows 100-199 = C2 candidates."
echo "        Split on 'candidate' column to compare."
echo "============================================="
echo "End: $(date)"
