#!/bin/bash
#SBATCH --job-name=icare_radpref_topk_resume
#SBATCH --partition=gpu4_medium
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=rd3571@nyu.edu

# Resume RadPref topk20 after node failure (job 25848062).
#
# Do NOT use icare_radpref_topk.sh with SRC_OUTPUT_DIR here — it restores
# .pre_topup backups and wipes completed shuffled top-ups.
#
# Usage (from ICARE_score repo root):
#   sbatch --export=ALL,\
#     OUTPUT_DIR=/gpfs/data/oermannlab/users/rd3571/ICARE_score/outputs/radpref/eval_seed_123_topk20,\
#     MIN_FILTERED_K=20 \
#     scripts/radpref_data/icare_radpref_topk_resume.sh
#
# Optional: eval shuffled only (fast path to main ICARE scores):
#   sbatch --export=ALL,SKIP_ORIG_TOPUP=1,SKIP_ORIG_EVAL=1,... scripts/...

set -eo pipefail

REPO=/gpfs/data/oermannlab/users/rd3571/ICARE_score
cd "$REPO"

source ~/.bashrc
conda activate rrg-eval-clean
cd "$REPO"

if [ -f ".env" ]; then
    echo "Loading environment variables from .env file..."
    # shellcheck disable=SC2046
    export $(grep -v '^#' .env | xargs)
fi

EVAL_SEED="${EVAL_SEED:-123}"
MIN_FILTERED_K="${MIN_FILTERED_K:-20}"
TOPUP_MAX_ROUNDS="${TOPUP_MAX_ROUNDS:-3}"
TOPUP_BATCH_SIZE="${TOPUP_BATCH_SIZE:-10}"
BASE_DATA_PATH="${RRGEVAL_BASE_DATA_PATH:-/gpfs/data/oermannlab/users/rd3571}"
NORMALIZED_INPUT_CSV="${NORMALIZED_INPUT_CSV:-${BASE_DATA_PATH}/cxr_report_datasets/radpref/radpref_icare.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO}/outputs/radpref/eval_seed_${EVAL_SEED}_topk${MIN_FILTERED_K}}"
SKIP_ORIG_TOPUP="${SKIP_ORIG_TOPUP:-1}"
SKIP_ORIG_EVAL="${SKIP_ORIG_EVAL:-1}"

export PYTHONHASHSEED="${EVAL_SEED}"

echo "============================================="
echo "  RadPref topk RESUME (after node failure)"
echo "============================================="
echo "Output dir:       ${OUTPUT_DIR}"
echo "MIN_FILTERED_K:   ${MIN_FILTERED_K}"
echo "SKIP_ORIG_TOPUP:  ${SKIP_ORIG_TOPUP}"
echo "SKIP_ORIG_EVAL:   ${SKIP_ORIG_EVAL}"
echo "============================================="
echo ""

if [ ! -d "${OUTPUT_DIR}/shuffled_ans_choices_data/gt_reports_as_ref/mcqa_filtering" ]; then
    echo "Error: expected topped shuffled outputs missing in ${OUTPUT_DIR}"
    exit 1
fi

echo ">>> Step 3a: Eval shuffled_ans_choices_data (top-up already done)..."
python "${REPO}/src/mcqa_evaluation.py" \
    --base_dir "${OUTPUT_DIR}" \
    --data_type shuffled_ans_choices_data \
    --seed "${EVAL_SEED}" \
    --gen_report_csv_file "${NORMALIZED_INPUT_CSV}" \
    --gt_report_csv_file "${NORMALIZED_INPUT_CSV}"
echo ">>> Step 3a complete."

if [ "${SKIP_ORIG_TOPUP}" != "1" ]; then
    echo ""
    echo ">>> Step 2b-resume: Top-up orig_data only..."
    for ref in gt gen; do
        INPUT_DIR_MCQ="${OUTPUT_DIR}/orig_data/${ref}_reports_as_ref"
        if [ ! -f "${INPUT_DIR_MCQ}/mcqa_data.json" ]; then
            echo "  Skip missing ${INPUT_DIR_MCQ}"
            continue
        fi
        echo "  Top-up orig_data/${ref}_reports_as_ref..."
        python "${REPO}/src/mcq_topup_to_min_k.py" \
            --mcqa-json "${INPUT_DIR_MCQ}/mcqa_data.json" \
            --filter-dir "${INPUT_DIR_MCQ}/mcqa_filtering" \
            --min_k "${MIN_FILTERED_K}" \
            --max_rounds "${TOPUP_MAX_ROUNDS}" \
            --batch_size "${TOPUP_BATCH_SIZE}" \
            --seed "${EVAL_SEED}"
    done
    echo ">>> Step 2b-resume complete."
fi

if [ "${SKIP_ORIG_EVAL}" != "1" ]; then
    echo ""
    echo ">>> Step 3b: Eval orig_data..."
    python "${REPO}/src/mcqa_evaluation.py" \
        --base_dir "${OUTPUT_DIR}" \
        --data_type orig_data \
        --seed "${EVAL_SEED}" \
        --gen_report_csv_file "${NORMALIZED_INPUT_CSV}" \
        --gt_report_csv_file "${NORMALIZED_INPUT_CSV}"
    echo ">>> Step 3b complete."
fi

echo ""
echo "============================================="
echo "  RadPref topk resume finished."
echo "  Check eval CSV timestamps under ${OUTPUT_DIR}"
echo "============================================="
