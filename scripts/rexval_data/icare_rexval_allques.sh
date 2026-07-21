#!/bin/bash
#SBATCH --job-name=icare_rexval_allques
#SBATCH --partition=gpu4_medium
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=40:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=rd3571@nyu.edu

# =============================================================================
# RexVal ablation: evaluate ALL generated questions (skip report-dependent
# filtering). Reuses Step-1 mcqa_data.json from an existing filtered run.
#
# Does NOT modify icare_rexval.sh or overwrite filtered results.
#
# Usage (from ICARE_score repo root):
#   sbatch --export=ALL,\
#     SRC_OUTPUT_DIR=/path/to/rexval_test_200_opus46/eval_seed_123,\
#     OUTPUT_DIR=/path/to/rexval_test_200_opus46_allques/eval_seed_123,\
#     ENV_FILE=config/envs/rexval_opus_46.env \
#     scripts/rexval_data/icare_rexval_allques.sh
#
# Or use: bash scripts/rexval_data/submit_rexval_allques_model.sh opus46
# =============================================================================

set -eo pipefail

if [ ! -f "src/mcqa_evaluation.py" ]; then
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
BASE_DATA_PATH="${RRGEVAL_BASE_DATA_PATH:-/gpfs/data/oermannlab/users/rd3571}"
DATA_TYPES="${DATA_TYPES:-shuffled_ans_choices_data}"
SRC_OUTPUT_DIR="${SRC_OUTPUT_DIR:-}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
NORMALIZED_INPUT_CSV="${NORMALIZED_INPUT_CSV:-${BASE_DATA_PATH}/cxr_report_datasets/rexval/RexVal_test_icare_200.csv}"

if [ -z "$SRC_OUTPUT_DIR" ] || [ -z "$OUTPUT_DIR" ]; then
    echo "Error: set SRC_OUTPUT_DIR (existing filtered run) and OUTPUT_DIR (new all-ques run)."
    exit 1
fi

if [ ! -d "$SRC_OUTPUT_DIR" ]; then
    echo "Error: SRC_OUTPUT_DIR not found: $SRC_OUTPUT_DIR"
    exit 1
fi

export PYTHONHASHSEED="${EVAL_SEED}"
mkdir -p "${OUTPUT_DIR}"

MANIFEST="${OUTPUT_DIR}/run_manifest.txt"
{
    echo "ablation=all_questions_no_report_filter"
    echo "date=$(date -Iseconds 2>/dev/null || date)"
    echo "src_output_dir=${SRC_OUTPUT_DIR}"
    echo "output_dir=${OUTPUT_DIR}"
    echo "eval_seed=${EVAL_SEED}"
    echo "data_types=${DATA_TYPES}"
    echo "env_file=${ENV_FILE:-none}"
    echo "normalized_input_csv=${NORMALIZED_INPUT_CSV}"
} > "${MANIFEST}"

echo "============================================="
echo "  ICARE RexVal — ALL QUESTIONS ablation"
echo "============================================="
echo "Source (Step 1 JSON): ${SRC_OUTPUT_DIR}"
echo "Output dir:           ${OUTPUT_DIR}"
echo "Eval seed:            ${EVAL_SEED}"
echo "Data types:           ${DATA_TYPES}"
echo "Manifest:             ${MANIFEST}"
echo "============================================="
echo ""

# -----------------------------------------------------------------------------
# Step A: Copy shuffled MCQ JSON from source (Step 1 only; skip filtering)
# -----------------------------------------------------------------------------
echo ">>> Step A: Copying mcqa_data.json from source run..."
IFS=',' read -ra DT_ARR <<< "${DATA_TYPES}"
for data_type in "${DT_ARR[@]}"; do
    data_type="$(echo "$data_type" | xargs)"
    for ref in "gt" "gen"; do
        src="${SRC_OUTPUT_DIR}/${data_type}/${ref}_reports_as_ref/mcqa_data.json"
        dst_dir="${OUTPUT_DIR}/${data_type}/${ref}_reports_as_ref"
        if [ ! -f "$src" ]; then
            echo "Error: missing source JSON: $src"
            exit 1
        fi
        mkdir -p "$dst_dir"
        cp "$src" "${dst_dir}/mcqa_data.json"
        echo "  copied ${data_type}/${ref}_reports_as_ref/mcqa_data.json"
    done
done
echo ">>> Step A complete."

# -----------------------------------------------------------------------------
# Step B: JSON -> all_questions.csv (no mcq_filtering.py)
# -----------------------------------------------------------------------------
echo ""
echo ">>> Step B: Converting mcqa_data.json -> all_questions.csv..."
for data_type in "${DT_ARR[@]}"; do
    data_type="$(echo "$data_type" | xargs)"
    for ref in "gt" "gen"; do
        input_json="${OUTPUT_DIR}/${data_type}/${ref}_reports_as_ref/mcqa_data.json"
        out_csv="${OUTPUT_DIR}/${data_type}/${ref}_reports_as_ref/mcqa_eval_input/all_questions.csv"
        echo "  ${data_type}/${ref}_reports_as_ref ..."
        python scripts/rexval_data/mcqa_json_to_csv.py \
            --input-json "${input_json}" \
            --output-csv "${out_csv}"
    done
done
echo ">>> Step B complete."

# -----------------------------------------------------------------------------
# Step C: MCQA evaluation with --question_set all
# -----------------------------------------------------------------------------
echo ""
echo ">>> Step C: Running MCQA evaluation (all questions)..."
for data_type in "${DT_ARR[@]}"; do
    data_type="$(echo "$data_type" | xargs)"
    echo "  Evaluating ${data_type}..."
    python src/mcqa_evaluation.py \
        --base_dir "${OUTPUT_DIR}" \
        --data_type "${data_type}" \
        --seed "${EVAL_SEED}" \
        --question_set all \
        --gen_report_csv_file "${NORMALIZED_INPUT_CSV}" \
        --gt_report_csv_file "${NORMALIZED_INPUT_CSV}"
done
echo ">>> Step C complete."

echo ""
echo "============================================="
echo "  RexVal all-questions ablation completed."
echo "============================================="
echo "Key files:"
echo "  - ${MANIFEST}"
echo "  - ${OUTPUT_DIR}/shuffled_ans_choices_data/gt_reports_as_ref/mcqa_eval/mcq_eval_dataset_level_agreement_stats.csv"
echo "  - ${OUTPUT_DIR}/shuffled_ans_choices_data/gen_reports_as_ref/mcqa_eval/mcq_eval_dataset_level_agreement_stats.csv"
echo ""

if command -v tree >/dev/null 2>&1; then
    tree "${OUTPUT_DIR}" -L 4
else
    rg --files "${OUTPUT_DIR}" | sed -n '1,40p'
fi
