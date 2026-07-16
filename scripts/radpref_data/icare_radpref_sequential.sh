#!/bin/bash
#SBATCH --job-name=icare_radpref_seq
#SBATCH --partition=a100_long
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=40:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=rd3571@nyu.edu

# =============================================================================
# Run ICARE on the RadPref preference dataset (100 cases x 2 candidates),
# using sequential (context-aware batched) MCQ generation instead of the
# single-shot approach in icare_radpref.sh. Writes to a separate output
# directory so the existing single-shot baseline results are untouched.
#
# Usage:
#   cd ICARE_score
#   sbatch scripts/radpref_data/icare_radpref_sequential.sh
#   # or local run:
#   bash scripts/radpref_data/icare_radpref_sequential.sh
# =============================================================================

set -eo pipefail

if [ ! -f "src/mcq_generation.py" ]; then
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
BASHRCSOURCED=0
source ~/.bashrc
cd "$ORIG_DIR"
conda activate rrg-eval-clean

EVAL_SEED="${EVAL_SEED:-123}"
NUM_QUESTIONS="${NUM_QUESTIONS:-60}"
SEQUENTIAL_BATCH_SIZE="${SEQUENTIAL_BATCH_SIZE:-10}"
SEQUENTIAL_STOP_THRESHOLD="${SEQUENTIAL_STOP_THRESHOLD:-2}"
BASE_DATA_PATH="${RRGEVAL_BASE_DATA_PATH:-/gpfs/data/oermannlab/users/rd3571}"

RAW_INPUT_JSON="${RAW_INPUT_JSON:-${BASE_DATA_PATH}/CRIMSON/RadPref/preference_data.json}"
NORMALIZED_INPUT_CSV="${NORMALIZED_INPUT_CSV:-${BASE_DATA_PATH}/cxr_report_datasets/radpref/radpref_icare.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-${ORIG_DIR}/outputs/radpref/eval_seed_${EVAL_SEED}_sequential_b${SEQUENTIAL_BATCH_SIZE}}"

export PYTHONHASHSEED="${EVAL_SEED}"

echo "============================================="
echo "  ICARE Score - RadPref (100 cases x 2 candidates) - SEQUENTIAL"
echo "============================================="
echo "Raw input JSON:      ${RAW_INPUT_JSON}"
echo "Prepared CSV:        ${NORMALIZED_INPUT_CSV}"
echo "Output dir:          ${OUTPUT_DIR}"
echo "Eval seed:           ${EVAL_SEED}"
echo "Num questions:       ${NUM_QUESTIONS}"
echo "Sequential batch:    ${SEQUENTIAL_BATCH_SIZE}"
echo "Sequential stop thr: ${SEQUENTIAL_STOP_THRESHOLD}"
echo "============================================="
echo ""

mkdir -p "${OUTPUT_DIR}"
mkdir -p "$(dirname "${NORMALIZED_INPUT_CSV}")"

echo ">>> Step 0: Preparing RadPref input..."
python scripts/radpref_data/prepare_radpref_for_icare.py \
    --input_json "${RAW_INPUT_JSON}" \
    --output_csv "${NORMALIZED_INPUT_CSV}"
echo ">>> Step 0 complete."

echo ""
echo ">>> Step 1: Generating MCQs (sequential mode)..."
for ref in "gt" "gen"; do
    echo "  Generating MCQs for ${ref} reports..."
    python src/mcq_generation.py \
        --input_csv "${NORMALIZED_INPUT_CSV}" \
        --output_dir "${OUTPUT_DIR}" \
        --reference "$ref" \
        --num_questions "${NUM_QUESTIONS}" \
        --seed "${EVAL_SEED}" \
        --use_sequential_generation \
        --sequential_batch_size "${SEQUENTIAL_BATCH_SIZE}" \
        --sequential_stop_threshold "${SEQUENTIAL_STOP_THRESHOLD}"
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

echo ""
echo ">>> Step 3: Running MCQA evaluation..."
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
echo "  RadPref ICARE (sequential) pipeline completed."
echo "============================================="
echo "Key files:"
echo "  - ${NORMALIZED_INPUT_CSV}"
echo "  - ${OUTPUT_DIR}/shuffled_ans_choices_data/gt_reports_as_ref/mcqa_eval/mcq_eval_dataset_level_agreement_stats.csv"
echo "  - ${OUTPUT_DIR}/shuffled_ans_choices_data/gen_reports_as_ref/mcqa_eval/mcq_eval_dataset_level_agreement_stats.csv"
echo ""
echo "Note: report-level scores cover rows 0-99 (C1) and rows 100-199 (C2)."
echo "      Split mcq_eval_report_level_stats.csv on Report_ID to compare candidates."
echo ""

if command -v tree >/dev/null 2>&1; then
    tree "${OUTPUT_DIR}" -L 4
else
    echo "tree not found; listing output files..."
    find "${OUTPUT_DIR}" -type f | sort | sed -n '1,40p'
fi
