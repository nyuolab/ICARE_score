#!/bin/bash
#SBATCH --job-name=icare_radpref_predefined
#SBATCH --partition=gpu8_long
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=20:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=rd3571@nyu.edu

# =============================================================================
# Run ICARE on RadPref using a predefined (fixed) question list.
# Skips MCQ generation and filtering; runs evaluation once.
#
# Usage:
#   cd ICARE_score
#   sbatch scripts/radpref_data/icare_radpref_predefined.sh
#   # or local run:
#   bash scripts/radpref_data/icare_radpref_predefined.sh
# =============================================================================

set -eo pipefail

if [ ! -f "src/mcqa_evaluation.py" ]; then
    echo "Error: run this script from ICARE_score repository root."
    exit 1
fi

if [ -f ".env" ]; then
    echo "Loading environment variables from .env file..."
    export $(cat .env | grep -v '^#' | xargs)
else
    echo "Warning: .env file not found. Falling back to explicit defaults."
fi

ORIG_DIR=$(pwd)
source ~/.bashrc
cd "$ORIG_DIR"
conda activate rrg-eval-clean

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
EVAL_SEED="${EVAL_SEED:-123}"
BASE_DATA_PATH="${RRGEVAL_BASE_DATA_PATH:-/gpfs/data/oermannlab/users/rd3571}"

PREDEFINED_JSON="${PREDEFINED_JSON:-${ORIG_DIR}/outputs/predefined_ques_list/mcqa_data.json}"
PREDEFINED_CSV="${PREDEFINED_CSV:-${ORIG_DIR}/outputs/predefined_ques_list/predefined_questions.csv}"

RAW_INPUT_JSON="${RAW_INPUT_JSON:-${BASE_DATA_PATH}/CRIMSON/RadPref/preference_data.json}"
NORMALIZED_INPUT_CSV="${NORMALIZED_INPUT_CSV:-${BASE_DATA_PATH}/cxr_report_datasets/radpref/radpref_icare.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-${ORIG_DIR}/outputs/radpref/predefined/eval_seed_${EVAL_SEED}}"

export PYTHONHASHSEED="${EVAL_SEED}"

echo "============================================="
echo "  ICARE Score - RadPref (predefined questions)"
echo "============================================="
echo "Predefined JSON : ${PREDEFINED_JSON}"
echo "Predefined CSV  : ${PREDEFINED_CSV}"
echo "Raw input JSON  : ${RAW_INPUT_JSON}"
echo "Prepared CSV    : ${NORMALIZED_INPUT_CSV}"
echo "Output dir      : ${OUTPUT_DIR}"
echo "Eval seed       : ${EVAL_SEED}"
echo "============================================="
echo ""

mkdir -p "${OUTPUT_DIR}"
mkdir -p "$(dirname "${NORMALIZED_INPUT_CSV}")"

# -----------------------------------------------------------------------------
# Step 0: Convert RadPref JSON → ICARE-ready CSV
# (skipped if CSV already exists)
# -----------------------------------------------------------------------------
if [ ! -f "${NORMALIZED_INPUT_CSV}" ]; then
    echo ">>> Step 0: Preparing RadPref input..."
    python scripts/radpref_data/prepare_radpref_for_icare.py \
        --input_json "${RAW_INPUT_JSON}" \
        --output_csv "${NORMALIZED_INPUT_CSV}"
    echo ">>> Step 0 complete."
else
    echo ">>> Step 0: Prepared CSV already exists, skipping."
fi
echo ""

# -----------------------------------------------------------------------------
# Step 1: Convert predefined mcqa_data.json → predefined_questions.csv
# (skipped if CSV already exists)
# -----------------------------------------------------------------------------
if [ ! -f "${PREDEFINED_CSV}" ]; then
    echo ">>> Step 1: Converting predefined questions JSON to CSV..."
    python src/predefined_mcqa_to_csv.py \
        --input_json "${PREDEFINED_JSON}" \
        --output_csv "${PREDEFINED_CSV}"
    echo ">>> Step 1 complete."
else
    echo ">>> Step 1: Predefined CSV already exists, skipping conversion."
fi
echo ""

# -----------------------------------------------------------------------------
# Step 2: MCQA evaluation (single run, predefined questions)
# -----------------------------------------------------------------------------
echo ">>> Step 2: Running MCQA evaluation..."
python src/mcqa_evaluation.py \
    --base_dir "${OUTPUT_DIR}" \
    --seed "${EVAL_SEED}" \
    --gen_report_csv_file "${NORMALIZED_INPUT_CSV}" \
    --gt_report_csv_file "${NORMALIZED_INPUT_CSV}" \
    --predefined_ques_csv "${PREDEFINED_CSV}"
echo ">>> Step 2 complete."

echo ""
echo "============================================="
echo "  Predefined RadPref ICARE pipeline completed."
echo "============================================="
echo "Key output files:"
echo "  ${OUTPUT_DIR}/mcqa_eval/mcqa_eval_answer_predictions.csv"
echo "  ${OUTPUT_DIR}/mcqa_eval/mcq_eval_dataset_level_agreement_stats.csv"
echo "  ${OUTPUT_DIR}/mcqa_eval/mcq_eval_report_level_stats.csv"
echo "  ${OUTPUT_DIR}/mcqa_eval/mcq_eval_report_level_agreement_hist.png"
echo ""
