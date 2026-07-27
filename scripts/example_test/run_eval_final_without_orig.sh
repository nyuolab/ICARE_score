#!/bin/bash
#SBATCH --job-name=icare_test
#SBATCH --partition=gpu4_medium
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=4:00:00
#SBATCH --output=logs/icare_test_%j.log

# =============================================================================
# End-to-end ICARE evaluation pipeline on sample test data.
#
# Same as run_eval.sh except:
#   - Steps 2-3 run on shuffled_ans_choices_data only (see run_eval.sh for orig_data ablation)
#   - Step 4 compiles icare_results.json + icare_results_summary.csv (+ step timing)
#
# Optional knobs (env vars):
#   DOCUMENT_TYPE=radiology|generic   (default: radiology)
#   ENV_FILE=path/to/model_override.sh  (optional; sourced after .env)
#   MIN_FILTERED_K=N                  (optional; top up after filtering; unset = no top-up)
#   SKIP_FILTERING=1                  (optional; evaluate all generated questions)
#
# Usage:
#   cd ICARE_score
#   sbatch scripts/example_test/run_eval_final_without_orig.sh
#   # Or: bash scripts/example_test/run_eval_final_without_orig.sh
# =============================================================================

set -e  # Exit on any error

# Create logs directory if it doesn't exist
mkdir -p logs

# Load base environment variables from .env
if [ -f ".env" ]; then
    echo "Loading base environment from .env ..."
    set -a
    # shellcheck disable=SC1091
    source ".env"
    set +a
else
    echo "Error: .env file not found. Copy .env.example to .env and configure it."
    echo "  cp .env.example .env"
    exit 1
fi

# Optional model-specific overrides (URL / API key / auth / model name)
ENV_FILE="${ENV_FILE:-}"
if [ -n "$ENV_FILE" ] && [ -f "$ENV_FILE" ]; then
    echo "Loading model overrides from $ENV_FILE ..."
    set -a
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    set +a
elif [ -n "$ENV_FILE" ]; then
    echo "Error: ENV_FILE set to $ENV_FILE but file not found."
    exit 1
fi

# Load conda (save/restore cwd since ~/.bashrc may change it)
ORIG_DIR=$(pwd)
source ~/.bashrc
cd "$ORIG_DIR"
conda activate rrg-eval-clean

# =============================================================================
# Configuration
# =============================================================================
EVAL_SEED="${EVAL_SEED:-123}"
NUM_QUESTIONS="${NUM_QUESTIONS:-40}" # e.g. NUM_QUESTIONS=5 for a smaller test run
INPUT_CSV="${INPUT_CSV:-test_data/sample_iuxray_reports.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-test_data/output}"
NUM_SAMPLES=$(( $(wc -l < "${INPUT_CSV}") - 1 ))

# radiology (default) -> prompts/radiology_specific ; generic -> prompts/generic
DOCUMENT_TYPE="${DOCUMENT_TYPE:-radiology}"
export DOCUMENT_TYPE

SKIP_FILTERING="${SKIP_FILTERING:-0}"
MIN_FILTERED_K="${MIN_FILTERED_K:-}"
TOPUP_MAX_ROUNDS="${TOPUP_MAX_ROUNDS:-3}"
TOPUP_BATCH_SIZE="${TOPUP_BATCH_SIZE:-10}"

if [ "${DOCUMENT_TYPE}" != "radiology" ] && [ "${DOCUMENT_TYPE}" != "generic" ]; then
    echo "Error: DOCUMENT_TYPE must be 'radiology' or 'generic' (got: ${DOCUMENT_TYPE})"
    exit 1
fi

if [ "${SKIP_FILTERING}" = "1" ] && [ -n "${MIN_FILTERED_K}" ]; then
    echo "Warning: SKIP_FILTERING=1 ignores MIN_FILTERED_K (top-up only applies after filtering)."
fi

export PYTHONHASHSEED=$EVAL_SEED

echo "============================================="
echo "  ICARE Score - End-to-End Test Run"
echo "============================================="
echo "Input CSV:        ${INPUT_CSV}"
echo "Output Dir:       ${OUTPUT_DIR}"
echo "Num Samples:      ${NUM_SAMPLES}"
echo "Eval Seed:        ${EVAL_SEED}"
echo "Num Questions:    ${NUM_QUESTIONS}"
echo "Document type:    ${DOCUMENT_TYPE}"
echo "ENV_FILE:         ${ENV_FILE:-<none>}"
echo "Skip filtering:   ${SKIP_FILTERING}"
echo "Min filtered k:   ${MIN_FILTERED_K:-<none — no top-up>}"
echo "============================================="
echo ""

# Clean previous test output if it exists
if [ -d "${OUTPUT_DIR}" ]; then
    echo "Cleaning previous test output..."
    rm -rf "${OUTPUT_DIR}"
fi
mkdir -p "${OUTPUT_DIR}"

PIPELINE_START=$(date +%s)

# =============================================================================
# Step 1: Generate MCQs for both GT and Gen reports
# =============================================================================
echo ""
echo ">>> Step 1: Generating MCQs..."
STEP1_START=$(date +%s)
for ref in "gt" "gen"; do
    echo "  Generating MCQs for ${ref} reports..."
    python src/mcq_generation.py \
        --input_csv "${INPUT_CSV}" \
        --output_dir "${OUTPUT_DIR}" \
        --reference "$ref" \
        --num_questions ${NUM_QUESTIONS} \
        --seed ${EVAL_SEED}
done
STEP1_SEC=$(( $(date +%s) - STEP1_START ))
echo ">>> Step 1 complete (${STEP1_SEC}s)."

QUESTION_SET="filtered"

# =============================================================================
# Step 2: Filter (default) or convert all questions (SKIP_FILTERING=1)
# =============================================================================
echo ""
STEP2_START=$(date +%s)
if [ "${SKIP_FILTERING}" = "1" ]; then
    QUESTION_SET="all"
    echo ">>> Step 2: Skipping filter — converting mcqa_data.json -> all_questions.csv..."
    for data_type in "shuffled_ans_choices_data"; do
        for ref in "gt" "gen"; do
            INPUT_JSON="${OUTPUT_DIR}/${data_type}/${ref}_reports_as_ref/mcqa_data.json"
            OUT_CSV="${OUTPUT_DIR}/${data_type}/${ref}_reports_as_ref/mcqa_eval_input/all_questions.csv"
            echo "  ${data_type}/${ref}_reports_as_ref..."
            python scripts/example_test/mcqa_json_to_csv.py \
                --input-json "${INPUT_JSON}" \
                --output-csv "${OUT_CSV}"
        done
    done
else
    echo ">>> Step 2: Filtering and shuffling MCQs..."
    for data_type in "shuffled_ans_choices_data"; do
        for ref in "gt" "gen"; do
            INPUT_DIR_MCQ="${OUTPUT_DIR}/${data_type}/${ref}_reports_as_ref"
            echo "  Filtering ${data_type}/${ref}_reports_as_ref..."
            python src/mcq_filtering.py \
                --input-json "${INPUT_DIR_MCQ}/mcqa_data.json" \
                --output-dir "${INPUT_DIR_MCQ}/mcqa_filtering" \
                --seed ${EVAL_SEED}
        done
    done

    if [ -n "${MIN_FILTERED_K}" ]; then
        echo ">>> Step 2b: Top-up filtered questions to min_k=${MIN_FILTERED_K}..."
        for data_type in "shuffled_ans_choices_data"; do
            for ref in "gt" "gen"; do
                REF_DIR="${OUTPUT_DIR}/${data_type}/${ref}_reports_as_ref"
                echo "  Top-up ${data_type}/${ref}_reports_as_ref..."
                python src/mcq_topup_to_min_k.py \
                    --mcqa-json "${REF_DIR}/mcqa_data.json" \
                    --filter-dir "${REF_DIR}/mcqa_filtering" \
                    --min_k "${MIN_FILTERED_K}" \
                    --batch_size "${TOPUP_BATCH_SIZE}" \
                    --max_rounds "${TOPUP_MAX_ROUNDS}" \
                    --seed "${EVAL_SEED}"
            done
        done
    fi
fi
STEP2_SEC=$(( $(date +%s) - STEP2_START ))
echo ">>> Step 2 complete (${STEP2_SEC}s)."

# =============================================================================
# Step 3: MCQA Evaluation
# =============================================================================
echo ""
echo ">>> Step 3: Running MCQA evaluation (question_set=${QUESTION_SET})..."
STEP3_START=$(date +%s)
for data_type in "shuffled_ans_choices_data"; do
    echo "  Evaluating ${data_type}..."
    python src/mcqa_evaluation.py \
        --base_dir "${OUTPUT_DIR}" \
        --data_type "${data_type}" \
        --seed ${EVAL_SEED} \
        --question_set "${QUESTION_SET}" \
        --gen_report_csv_file "${INPUT_CSV}" \
        --gt_report_csv_file "${INPUT_CSV}"
done
STEP3_SEC=$(( $(date +%s) - STEP3_START ))
echo ">>> Step 3 complete (${STEP3_SEC}s)."

TOTAL_SEC=$(( $(date +%s) - PIPELINE_START ))
TIMING_FILE="${OUTPUT_DIR}/pipeline_timing.json"
cat > "${TIMING_FILE}" <<EOF
{
  "eval_seed": ${EVAL_SEED},
  "num_samples": ${NUM_SAMPLES},
  "num_questions_per_report": ${NUM_QUESTIONS},
  "document_type": "${DOCUMENT_TYPE}",
  "skip_filtering": "${SKIP_FILTERING}",
  "min_filtered_k": "${MIN_FILTERED_K:-null}",
  "step1_sec": ${STEP1_SEC},
  "step2_sec": ${STEP2_SEC},
  "step3_sec": ${STEP3_SEC},
  "total_sec": ${TOTAL_SEC}
}
EOF

# =============================================================================
# Step 4: Compile per-sample results
# =============================================================================
echo ""
echo ">>> Step 4: Compiling per-sample results..."
python src/compile_results.py \
    --base_dir    "${OUTPUT_DIR}" \
    --input_csv   "${INPUT_CSV}" \
    --output      "${OUTPUT_DIR}/icare_results.json" \
    --summary_csv "${OUTPUT_DIR}/icare_results_summary.csv" \
    --timing_file "${TIMING_FILE}" \
    --question_set "${QUESTION_SET}"
echo ">>> Step 4 complete."

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "============================================="
echo "  Test Pipeline Completed Successfully!"
echo "============================================="
echo ""
echo "Results structure:"
if command -v tree &> /dev/null; then
    tree "${OUTPUT_DIR}" -L 4
else
    find "${OUTPUT_DIR}" -type f | head -30
fi
echo ""
echo "Key result files:"
echo "  - ${OUTPUT_DIR}/icare_results.json"
echo "  - ${OUTPUT_DIR}/icare_results_summary.csv"
echo "  - ${OUTPUT_DIR}/pipeline_timing.json"
echo "  - ${OUTPUT_DIR}/shuffled_ans_choices_data/gt_reports_as_ref/mcqa_eval/mcq_eval_dataset_level_agreement_stats.csv"
echo "  - ${OUTPUT_DIR}/shuffled_ans_choices_data/gen_reports_as_ref/mcqa_eval/mcq_eval_dataset_level_agreement_stats.csv"
echo "  - ${OUTPUT_DIR}/shuffled_ans_choices_data/gt_reports_as_ref/mcqa_eval/mcq_eval_report_level_stats.csv"
echo "  - ${OUTPUT_DIR}/shuffled_ans_choices_data/gen_reports_as_ref/mcqa_eval/mcq_eval_report_level_stats.csv"
