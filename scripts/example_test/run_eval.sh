#!/bin/bash
#SBATCH --job-name=icare_test
#SBATCH --partition=gpu4_medium
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=4:00:00
#SBATCH --output=logs/icare_test_%j.log

# =============================================================================
# End-to-end test of the ICARE evaluation pipeline using sample test data.
# Runs Steps 1–3 on both orig_data and shuffled_ans_choices_data (ablation path).
#
# Usage:
#   cd ICARE_score
#   sbatch scripts/example_test/run_eval.sh
#   # Or: bash scripts/example_test/run_eval.sh
# =============================================================================

set -e  # Exit on any error

# Create logs directory if it doesn't exist
mkdir -p logs

# Load environment variables from .env file
if [ -f ".env" ]; then
    echo "Loading environment variables from .env file..."
    export $(cat .env | grep -v '^#' | xargs)
else
    echo "Error: .env file not found. Copy .env.example to .env and configure it."
    echo "  cp .env.example .env"
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
EVAL_SEED=123
NUM_QUESTIONS=40 # reduce to 5 for a smaller test run
INPUT_CSV="test_data/sample_iuxray_reports.csv"
OUTPUT_DIR="test_data/output"

export PYTHONHASHSEED=$EVAL_SEED

echo "============================================="
echo "  ICARE Score - End-to-End Test Run"
echo "============================================="
echo "Input CSV:      ${INPUT_CSV}"
echo "Output Dir:     ${OUTPUT_DIR}"
echo "Eval Seed:      ${EVAL_SEED}"
echo "Num Questions:  ${NUM_QUESTIONS}"
echo "============================================="
echo ""

# Clean previous test output if it exists
if [ -d "${OUTPUT_DIR}" ]; then
    echo "Cleaning previous test output..."
    rm -rf "${OUTPUT_DIR}"
fi
mkdir -p "${OUTPUT_DIR}"

# =============================================================================
# Step 1: Generate MCQs for both GT and Gen reports
# =============================================================================
echo ""
echo ">>> Step 1: Generating MCQs..."
for ref in "gt" "gen"; do
    echo "  Generating MCQs for ${ref} reports..."
    python src/mcq_generation.py \
        --input_csv "${INPUT_CSV}" \
        --output_dir "${OUTPUT_DIR}" \
        --reference "$ref" \
        --num_questions ${NUM_QUESTIONS} \
        --seed ${EVAL_SEED}
done
echo ">>> Step 1 complete."

# =============================================================================
# Step 2: Filter and Shuffle MCQs
# =============================================================================
echo ""
echo ">>> Step 2: Filtering and shuffling MCQs..."
for data_type in "orig_data" "shuffled_ans_choices_data"; do
    for ref in "gt" "gen"; do
        INPUT_DIR_MCQ="${OUTPUT_DIR}/${data_type}/${ref}_reports_as_ref"
        echo "  Filtering ${data_type}/${ref}_reports_as_ref..."
        python src/mcq_filtering.py \
            --input-json "${INPUT_DIR_MCQ}/mcqa_data.json" \
            --output-dir "${INPUT_DIR_MCQ}/mcqa_filtering" \
            --seed ${EVAL_SEED}
    done
done
echo ">>> Step 2 complete."

# =============================================================================
# Step 3: MCQA Evaluation
# =============================================================================
echo ""
echo ">>> Step 3: Running MCQA evaluation..."
for data_type in "orig_data" "shuffled_ans_choices_data"; do
    echo "  Evaluating ${data_type}..."
    python src/mcqa_evaluation.py \
        --base_dir "${OUTPUT_DIR}" \
        --data_type "${data_type}" \
        --seed ${EVAL_SEED} \
        --gen_report_csv_file "${INPUT_CSV}" \
        --gt_report_csv_file "${INPUT_CSV}"
done
echo ">>> Step 3 complete."

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
echo "  - ${OUTPUT_DIR}/shuffled_ans_choices_data/gt_reports_as_ref/mcqa_eval/mcq_eval_dataset_level_agreement_stats.csv"
echo "  - ${OUTPUT_DIR}/shuffled_ans_choices_data/gen_reports_as_ref/mcqa_eval/mcq_eval_dataset_level_agreement_stats.csv"
echo "  - ${OUTPUT_DIR}/shuffled_ans_choices_data/gt_reports_as_ref/mcqa_eval/mcq_eval_report_level_stats.csv"
echo "  - ${OUTPUT_DIR}/shuffled_ans_choices_data/gen_reports_as_ref/mcqa_eval/mcq_eval_report_level_stats.csv"
