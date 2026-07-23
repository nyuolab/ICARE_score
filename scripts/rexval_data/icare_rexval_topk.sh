#!/bin/bash
#SBATCH --job-name=icare_rexval_topk
#SBATCH --partition=gpu4_medium
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=40:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=rd3571@nyu.edu

# =============================================================================
# RexVal ICARE with post-filter top-up to MIN_FILTERED_K keepers per report.
#
# One-shot generation + filter (same as icare_rexval.sh), then for reports
# with fewer than MIN_FILTERED_K filtered questions, generate more with an
# anti-repeat prompt, filter the new batch, and append keepers.
#
# Usage (from ICARE_score repo root):
#   # Reuse an existing filtered Llama run (recommended for testing):
#   sbatch --export=ALL,\
#     SRC_OUTPUT_DIR=/gpfs/data/oermannlab/users/rd3571/ICARE_score/outputs/rexval/rexval_test_200/eval_seed_123,\
#     MIN_FILTERED_K=8,\
#     OUTPUT_DIR=/gpfs/data/oermannlab/users/rd3571/ICARE_score/outputs/rexval/rexval_test_200_topk8/eval_seed_123 \
#     scripts/rexval_data/icare_rexval_topk.sh
#
#   # Or full pipeline from scratch (no SRC_OUTPUT_DIR):
#   sbatch --export=ALL,MIN_FILTERED_K=8,\
#     OUTPUT_DIR=.../rexval_test_200_topk8/eval_seed_123 \
#     scripts/rexval_data/icare_rexval_topk.sh
# =============================================================================

set -eo pipefail

if [ ! -f "src/mcq_generation.py" ]; then
    echo "Error: run this script from ICARE_score repository root."
    exit 1
fi

if [ -f ".env" ]; then
    echo "Loading base environment from .env ..."
    set -a
    # shellcheck disable=SC1091
    source ".env"
    set +a
else
    echo "Warning: .env file not found. Falling back to explicit defaults."
fi

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

ORIG_DIR=$(pwd)
BASHRCSOURCED=0
source ~/.bashrc
cd "$ORIG_DIR"
conda activate rrg-eval-clean

EVAL_SEED="${EVAL_SEED:-123}"
NUM_QUESTIONS="${NUM_QUESTIONS:-60}"
MIN_FILTERED_K="${MIN_FILTERED_K:-8}"
TOPUP_MAX_ROUNDS="${TOPUP_MAX_ROUNDS:-3}"
TOPUP_BATCH_SIZE="${TOPUP_BATCH_SIZE:-60}"
BASE_DATA_PATH="${RRGEVAL_BASE_DATA_PATH:-/gpfs/data/oermannlab/users/rd3571}"

RAW_INPUT_CSV="${RAW_INPUT_CSV:-${BASE_DATA_PATH}/cxr_report_datasets/rexval/RexVal_test.csv}"
NORMALIZED_INPUT_CSV="${NORMALIZED_INPUT_CSV:-${BASE_DATA_PATH}/cxr_report_datasets/rexval/RexVal_test_icare_200.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-${ORIG_DIR}/outputs/rexval/rexval_test_200_topk${MIN_FILTERED_K}/eval_seed_${EVAL_SEED}}"
SRC_OUTPUT_DIR="${SRC_OUTPUT_DIR:-}"
KEY_COUNTS_CSV="${KEY_COUNTS_CSV:-${OUTPUT_DIR}/input_key_counts.csv}"
LABELS_CSV="${LABELS_CSV:-${BASE_DATA_PATH}/cxr_report_datasets/rexval_physionet_labels/physionet.org/files/rexval-dataset/1.0.0/6_valid_raters_per_rater_error_categories.csv}"
VALIDATION_OUTPUT_DIR="${VALIDATION_OUTPUT_DIR:-${ORIG_DIR}/outputs/rexval/rexval_test_200/key_validation}"

export PYTHONHASHSEED="${EVAL_SEED}"

echo "============================================="
echo "  ICARE Score - RexVal (top-up to min_k)"
echo "============================================="
echo "Output dir:       ${OUTPUT_DIR}"
echo "SRC_OUTPUT_DIR:   ${SRC_OUTPUT_DIR:-<none — full pipeline>}"
echo "Eval seed:        ${EVAL_SEED}"
echo "Num questions:    ${NUM_QUESTIONS}"
echo "MIN_FILTERED_K:   ${MIN_FILTERED_K}"
echo "TOPUP_MAX_ROUNDS: ${TOPUP_MAX_ROUNDS}"
echo "TOPUP_BATCH_SIZE: ${TOPUP_BATCH_SIZE}"
echo "============================================="
echo ""

mkdir -p "${OUTPUT_DIR}"

if [ -n "${SRC_OUTPUT_DIR}" ]; then
    echo ">>> Reusing generation+filter from SRC_OUTPUT_DIR..."
    if [ ! -d "${SRC_OUTPUT_DIR}/shuffled_ans_choices_data" ]; then
        echo "Error: SRC_OUTPUT_DIR missing shuffled_ans_choices_data: ${SRC_OUTPUT_DIR}"
        exit 1
    fi
    # Copy shuffled tree only (RexVal eval path); do not overwrite if already present.
    if [ ! -d "${OUTPUT_DIR}/shuffled_ans_choices_data/gt_reports_as_ref/mcqa_filtering" ]; then
        mkdir -p "${OUTPUT_DIR}"
        cp -a "${SRC_OUTPUT_DIR}/shuffled_ans_choices_data" "${OUTPUT_DIR}/"
    else
        echo "  OUTPUT_DIR already has shuffled_ans_choices_data; skipping copy."
    fi
    # Restore pre-topup CSVs if a previous top-up left backups and we re-run.
    for ref in gt gen; do
        FDIR="${OUTPUT_DIR}/shuffled_ans_choices_data/${ref}_reports_as_ref/mcqa_filtering"
        if [ -f "${FDIR}/filtered_questions.csv.pre_topup" ]; then
            cp -f "${FDIR}/filtered_questions.csv.pre_topup" "${FDIR}/filtered_questions.csv"
            cp -f "${FDIR}/filtered_questions_shuffled.csv.pre_topup" "${FDIR}/filtered_questions_shuffled.csv"
        fi
    done
    echo ">>> Reuse complete."
else
    echo ">>> Step 0: Preparing RexVal input..."
    python scripts/rexval_data/prepare_rexval_for_icare.py \
        --input_csv "${RAW_INPUT_CSV}" \
        --output_csv "${NORMALIZED_INPUT_CSV}" \
        --key_counts_csv "${KEY_COUNTS_CSV}" \
        --labels_csv "${LABELS_CSV}" \
        --validation_output_dir "${VALIDATION_OUTPUT_DIR}"
    echo ">>> Step 0 complete."

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

    echo ""
    echo ">>> Step 2: Filtering and shuffling MCQs..."
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
fi

echo ""
echo ">>> Step 2b: Top-up filtered questions to MIN_FILTERED_K=${MIN_FILTERED_K}..."
for ref in "gt" "gen"; do
    INPUT_DIR_MCQ="${OUTPUT_DIR}/shuffled_ans_choices_data/${ref}_reports_as_ref"
    echo "  Top-up ${ref}_reports_as_ref..."
    python src/mcq_topup_to_min_k.py \
        --mcqa-json "${INPUT_DIR_MCQ}/mcqa_data.json" \
        --filter-dir "${INPUT_DIR_MCQ}/mcqa_filtering" \
        --min_k "${MIN_FILTERED_K}" \
        --max_rounds "${TOPUP_MAX_ROUNDS}" \
        --batch_size "${TOPUP_BATCH_SIZE}" \
        --seed "${EVAL_SEED}"
done
echo ">>> Step 2b complete."

echo ""
echo ">>> Step 3: Running MCQA evaluation..."
python src/mcqa_evaluation.py \
    --base_dir "${OUTPUT_DIR}" \
    --data_type "shuffled_ans_choices_data" \
    --seed "${EVAL_SEED}" \
    --gen_report_csv_file "${NORMALIZED_INPUT_CSV}" \
    --gt_report_csv_file "${NORMALIZED_INPUT_CSV}"
echo ">>> Step 3 complete."

echo ""
echo "============================================="
echo "  RexVal ICARE top-up pipeline completed."
echo "============================================="
echo "  OUTPUT_DIR=${OUTPUT_DIR}"
echo "  Summaries: .../mcqa_filtering/topup_to_min_k_summary.json"
echo ""
