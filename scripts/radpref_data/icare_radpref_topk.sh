#!/bin/bash
#SBATCH --job-name=icare_radpref_topk
#SBATCH --partition=a100_long
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=40:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=rd3571@nyu.edu

# =============================================================================
# RadPref ICARE with post-filter top-up to MIN_FILTERED_K keepers per report.
#
# Usage (from ICARE_score repo root):
#   # Reuse existing filtered Llama run (recommended):
#   sbatch --export=ALL,\
#     SRC_OUTPUT_DIR=/gpfs/data/oermannlab/users/rd3571/ICARE_score/outputs/radpref/eval_seed_123,\
#     MIN_FILTERED_K=8,\
#     OUTPUT_DIR=/gpfs/data/oermannlab/users/rd3571/ICARE_score/outputs/radpref/eval_seed_123_topk8 \
#     scripts/radpref_data/icare_radpref_topk.sh
#
#   # Full pipeline:
#   sbatch --export=ALL,MIN_FILTERED_K=8,\
#     OUTPUT_DIR=.../radpref/eval_seed_123_topk8 \
#     scripts/radpref_data/icare_radpref_topk.sh
# =============================================================================

set -eo pipefail

if [ ! -f "src/mcq_generation.py" ]; then
    echo "Error: run this script from ICARE_score repository root."
    exit 1
fi

if [ -f ".env" ]; then
    echo "Loading environment variables from .env file..."
    # shellcheck disable=SC2046
    export $(grep -v '^#' .env | xargs)
else
    echo "Warning: .env file not found. Falling back to explicit defaults."
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
TOPUP_BATCH_SIZE="${TOPUP_BATCH_SIZE:-10}"
BASE_DATA_PATH="${RRGEVAL_BASE_DATA_PATH:-/gpfs/data/oermannlab/users/rd3571}"

RAW_INPUT_JSON="${RAW_INPUT_JSON:-${BASE_DATA_PATH}/CRIMSON/RadPref/preference_data.json}"
NORMALIZED_INPUT_CSV="${NORMALIZED_INPUT_CSV:-${BASE_DATA_PATH}/cxr_report_datasets/radpref/radpref_icare.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-${ORIG_DIR}/outputs/radpref/eval_seed_${EVAL_SEED}_topk${MIN_FILTERED_K}}"
SRC_OUTPUT_DIR="${SRC_OUTPUT_DIR:-}"

export PYTHONHASHSEED="${EVAL_SEED}"

echo "============================================="
echo "  ICARE Score - RadPref (top-up to min_k)"
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
mkdir -p "$(dirname "${NORMALIZED_INPUT_CSV}")"

if [ -n "${SRC_OUTPUT_DIR}" ]; then
    echo ">>> Reusing generation+filter from SRC_OUTPUT_DIR..."
    if [ ! -d "${SRC_OUTPUT_DIR}/shuffled_ans_choices_data" ]; then
        echo "Error: SRC_OUTPUT_DIR missing shuffled_ans_choices_data: ${SRC_OUTPUT_DIR}"
        exit 1
    fi
    if [ ! -d "${OUTPUT_DIR}/shuffled_ans_choices_data/gt_reports_as_ref/mcqa_filtering" ]; then
        cp -a "${SRC_OUTPUT_DIR}/shuffled_ans_choices_data" "${OUTPUT_DIR}/"
        # RadPref also evaluates orig_data when present; copy if available.
        if [ -d "${SRC_OUTPUT_DIR}/orig_data" ] && [ ! -d "${OUTPUT_DIR}/orig_data" ]; then
            cp -a "${SRC_OUTPUT_DIR}/orig_data" "${OUTPUT_DIR}/"
        fi
    else
        echo "  OUTPUT_DIR already has shuffled_ans_choices_data; skipping copy."
    fi
    for data_type in shuffled_ans_choices_data orig_data; do
        for ref in gt gen; do
            FDIR="${OUTPUT_DIR}/${data_type}/${ref}_reports_as_ref/mcqa_filtering"
            if [ -f "${FDIR}/filtered_questions.csv.pre_topup" ]; then
                cp -f "${FDIR}/filtered_questions.csv.pre_topup" "${FDIR}/filtered_questions.csv"
                cp -f "${FDIR}/filtered_questions_shuffled.csv.pre_topup" "${FDIR}/filtered_questions_shuffled.csv"
            fi
        done
    done
    echo ">>> Reuse complete."
else
    echo ">>> Step 0: Preparing RadPref input..."
    python scripts/radpref_data/prepare_radpref_for_icare.py \
        --input_json "${RAW_INPUT_JSON}" \
        --output_csv "${NORMALIZED_INPUT_CSV}"
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
    for data_type in "orig_data" "shuffled_ans_choices_data"; do
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
# Top-up the shuffled path used for correlation plots; also orig_data if present.
for data_type in shuffled_ans_choices_data orig_data; do
    if [ ! -d "${OUTPUT_DIR}/${data_type}" ]; then
        continue
    fi
    for ref in "gt" "gen"; do
        INPUT_DIR_MCQ="${OUTPUT_DIR}/${data_type}/${ref}_reports_as_ref"
        if [ ! -f "${INPUT_DIR_MCQ}/mcqa_data.json" ]; then
            continue
        fi
        echo "  Top-up ${data_type}/${ref}_reports_as_ref..."
        python src/mcq_topup_to_min_k.py \
            --mcqa-json "${INPUT_DIR_MCQ}/mcqa_data.json" \
            --filter-dir "${INPUT_DIR_MCQ}/mcqa_filtering" \
            --min_k "${MIN_FILTERED_K}" \
            --max_rounds "${TOPUP_MAX_ROUNDS}" \
            --batch_size "${TOPUP_BATCH_SIZE}" \
            --seed "${EVAL_SEED}"
    done
done
echo ">>> Step 2b complete."

echo ""
echo ">>> Step 3: Running MCQA evaluation..."
for data_type in shuffled_ans_choices_data orig_data; do
    if [ ! -d "${OUTPUT_DIR}/${data_type}" ]; then
        continue
    fi
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
echo "  RadPref ICARE top-up pipeline completed."
echo "============================================="
echo "  OUTPUT_DIR=${OUTPUT_DIR}"
echo "  Summaries: .../mcqa_filtering/topup_to_min_k_summary.json"
echo ""
