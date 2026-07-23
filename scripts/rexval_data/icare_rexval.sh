#!/bin/bash
#SBATCH --job-name=icare_rexval
#SBATCH --partition=oermannlab
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=40:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=rd3571@nyu.edu

# =============================================================================
# Run ICARE on RexVal long-format test set (200 report pairs).
#
# Usage:
#   cd ICARE_score
#   sbatch scripts/rexval_data/icare_rexval.sh
#   # or local run:
#   bash scripts/rexval_data/icare_rexval.sh
# =============================================================================

set -eo pipefail

if [ ! -f "src/mcq_generation.py" ]; then
    echo "Error: run this script from ICARE_score repository root."
    exit 1
fi

# Load environment variables from env file (default: .env)
# Load shared base settings from .env
if [ -f ".env" ]; then
    echo "Loading base environment from .env ..."
    set -a
    # shellcheck disable=SC1091
    source ".env"
    set +a
else
    echo "Warning: .env file not found. Falling back to explicit defaults."
fi

# Load model-specific overrides (optional)
ENV_FILE="${ENV_FILE:-}"
if [ -n "$ENV_FILE" ] && [ -f "$ENV_FILE" ]; then
    echo "Loading model overrides from $ENV_FILE ..."
    set -a
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    set +a
elif [ -n "$ENV_FILE" ]; then
    echo "Warning: ENV_FILE set to $ENV_FILE but file not found."
fi

# Load conda (save/restore cwd in case ~/.bashrc changes it)
ORIG_DIR=$(pwd)
BASHRCSOURCED=0
source ~/.bashrc
cd "$ORIG_DIR"
conda activate rrg-eval-clean

# -----------------------------------------------------------------------------
# Configuration (override with exported env vars if needed)
# -----------------------------------------------------------------------------
EVAL_SEED="${EVAL_SEED:-123}"
NUM_QUESTIONS="${NUM_QUESTIONS:-60}"
BASE_DATA_PATH="${RRGEVAL_BASE_DATA_PATH:-/gpfs/data/oermannlab/users/rd3571}"

# Prompt pack: folder with generation.txt + qa_with_document.txt
# Override via .env or: sbatch --export=ALL,ICARE_PROMPT_DIR=prompts/generic,...
ICARE_PROMPT_DIR="${ICARE_PROMPT_DIR:-prompts/radiology_specific}"
export ICARE_PROMPT_DIR

RAW_INPUT_CSV="${RAW_INPUT_CSV:-${BASE_DATA_PATH}/cxr_report_datasets/rexval/RexVal_test.csv}"
NORMALIZED_INPUT_CSV="${NORMALIZED_INPUT_CSV:-${BASE_DATA_PATH}/cxr_report_datasets/rexval/RexVal_test_icare_200.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-${ORIG_DIR}/outputs/rexval/rexval_test_200/eval_seed_${EVAL_SEED}}"
KEY_COUNTS_CSV="${KEY_COUNTS_CSV:-${OUTPUT_DIR}/input_key_counts.csv}"
LABELS_CSV="${LABELS_CSV:-${BASE_DATA_PATH}/cxr_report_datasets/rexval_physionet_labels/physionet.org/files/rexval-dataset/1.0.0/6_valid_raters_per_rater_error_categories.csv}"
VALIDATION_OUTPUT_DIR="${VALIDATION_OUTPUT_DIR:-${ORIG_DIR}/outputs/rexval/rexval_test_200/key_validation}"

export PYTHONHASHSEED="${EVAL_SEED}"

echo "============================================="
echo "  ICARE Score - RexVal (200 pairs)"
echo "============================================="
echo "Raw input:      ${RAW_INPUT_CSV}"
echo "Prepared input: ${NORMALIZED_INPUT_CSV}"
echo "Output dir:     ${OUTPUT_DIR}"
echo "Eval seed:      ${EVAL_SEED}"
echo "Num questions:  ${NUM_QUESTIONS}"
echo "Prompt dir:     ${ICARE_PROMPT_DIR}"
echo "Labels CSV:     ${LABELS_CSV}"
echo "============================================="
echo ""

mkdir -p "${OUTPUT_DIR}"

# -----------------------------------------------------------------------------
# Step 0: Normalize RexVal schema for ICARE + emit join-key counts
# -----------------------------------------------------------------------------
echo ">>> Step 0: Preparing RexVal input..."
python scripts/rexval_data/prepare_rexval_for_icare.py \
    --input_csv "${RAW_INPUT_CSV}" \
    --output_csv "${NORMALIZED_INPUT_CSV}" \
    --key_counts_csv "${KEY_COUNTS_CSV}" \
    --labels_csv "${LABELS_CSV}" \
    --validation_output_dir "${VALIDATION_OUTPUT_DIR}"
echo ">>> Step 0 complete."

# -----------------------------------------------------------------------------
# Step 1: Generate MCQs
# -----------------------------------------------------------------------------
echo ""
echo ">>> Step 1: Generating MCQs..."
for ref in "gt" "gen"; do
    echo "  Generating MCQs for ${ref} reports..."
    python src/mcq_generation.py \
        --input_csv "${NORMALIZED_INPUT_CSV}" \
        --output_dir "${OUTPUT_DIR}" \
        --reference "$ref" \
        --num_questions "${NUM_QUESTIONS}" \
        --seed "${EVAL_SEED}"
done
echo ">>> Step 1 complete."

# -----------------------------------------------------------------------------
# Step 2: Filter and shuffle MCQs
# -----------------------------------------------------------------------------
echo ""
echo ">>> Step 2: Filtering and shuffling MCQs..."
# for data_type in "orig_data" "shuffled_ans_choices_data"; do
for data_type in "shuffled_ans_choices_data"; do
    for ref in "gt" "gen"; do
        INPUT_DIR_MCQ="${OUTPUT_DIR}/${data_type}/${ref}_reports_as_ref"
        echo "  Filtering ${data_type}/${ref}_reports_as_ref..."
        python src/mcq_filtering.py \
            --input-json "${INPUT_DIR_MCQ}/mcqa_data.json" \
            --output-dir "${INPUT_DIR_MCQ}/mcqa_filtering" \
            --seed "${EVAL_SEED}"
    done
done
echo ">>> Step 2 complete."

# -----------------------------------------------------------------------------
# Step 3: MCQA evaluation
# -----------------------------------------------------------------------------
echo ""
echo ">>> Step 3: Running MCQA evaluation..."
# for data_type in "orig_data" "shuffled_ans_choices_data"; do
for data_type in "shuffled_ans_choices_data"; do
    echo "  Evaluating ${data_type}..."
    python src/mcqa_evaluation.py \
        --base_dir "${OUTPUT_DIR}" \
        --data_type "${data_type}" \
        --seed "${EVAL_SEED}" \
        --gen_report_csv_file "${NORMALIZED_INPUT_CSV}" \
        --gt_report_csv_file "${NORMALIZED_INPUT_CSV}"
done
echo ">>> Step 3 complete."

echo ""
echo "============================================="
echo "  RexVal ICARE pipeline completed."
echo "============================================="
echo "Key files:"
echo "  - ${KEY_COUNTS_CSV}"
echo "  - ${VALIDATION_OUTPUT_DIR}/key_validation_summary.csv"
echo "  - ${OUTPUT_DIR}/shuffled_ans_choices_data/gt_reports_as_ref/mcqa_eval/mcq_eval_dataset_level_agreement_stats.csv"
echo "  - ${OUTPUT_DIR}/shuffled_ans_choices_data/gen_reports_as_ref/mcqa_eval/mcq_eval_dataset_level_agreement_stats.csv"
echo ""

if command -v tree >/dev/null 2>&1; then
    tree "${OUTPUT_DIR}" -L 4
else
    echo "tree not found; showing first output files with rg..."
    rg --files "${OUTPUT_DIR}" | sed -n '1,40p'
fi
