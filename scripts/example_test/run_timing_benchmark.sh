#!/bin/bash
#SBATCH --job-name=icare_timing
#SBATCH --partition=gpu4_medium
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=4:00:00
#SBATCH --output=logs/icare_timing_%j.log

# =============================================================================
# Timing benchmark for the ICARE pipeline with 40 questions.
# Reports wall-clock time per step, total time, and time-per-sample.
#
# Usage:
#   cd ICARE_score
#   bash scripts/example_test/run_timing_benchmark.sh
#   # Or: sbatch scripts/example_test/run_timing_benchmark.sh
# =============================================================================

set -e
mkdir -p logs

# Load environment variables
if [ -f ".env" ]; then
    export $(cat .env | grep -v '^#' | xargs)
else
    echo "Error: .env file not found."
    exit 1
fi

ORIG_DIR=$(pwd)
source ~/.bashrc
cd "$ORIG_DIR"
conda activate rrg-eval-clean

# =============================================================================
# Configuration
# =============================================================================
EVAL_SEED=123
NUM_QUESTIONS=40
INPUT_CSV="test_data/sample_iuxray_reports.csv"
OUTPUT_DIR="test_data/timing_output"
NUM_SAMPLES=$(( $(wc -l < "${INPUT_CSV}") - 1 ))  # subtract header row

export PYTHONHASHSEED=$EVAL_SEED

echo "============================================="
echo "  ICARE Timing Benchmark"
echo "============================================="
echo "Input CSV:      ${INPUT_CSV}"
echo "Num Samples:    ${NUM_SAMPLES}"
echo "Eval Seed:      ${EVAL_SEED}"
echo "Num Questions:  ${NUM_QUESTIONS}"
echo "============================================="
echo ""

# =============================================================================
# Helper: elapsed seconds between two epoch timestamps
# =============================================================================
elapsed() { echo $(( $2 - $1 )); }

# Clean previous output
rm -rf "${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}"

PIPELINE_START=$(date +%s)

# =============================================================================
# Step 1: Generate MCQs
# =============================================================================
echo ""
echo ">>> Step 1: Generating MCQs (gt)..."
STEP1_GT_START=$(date +%s)
python src/mcq_generation.py \
    --input_csv "${INPUT_CSV}" \
    --output_dir "${OUTPUT_DIR}" \
    --reference "gt" \
    --num_questions ${NUM_QUESTIONS} \
    --seed ${EVAL_SEED}
STEP1_GT_END=$(date +%s)
STEP1_GT_SEC=$(elapsed $STEP1_GT_START $STEP1_GT_END)

echo ">>> Step 1: Generating MCQs (gen)..."
STEP1_GEN_START=$(date +%s)
python src/mcq_generation.py \
    --input_csv "${INPUT_CSV}" \
    --output_dir "${OUTPUT_DIR}" \
    --reference "gen" \
    --num_questions ${NUM_QUESTIONS} \
    --seed ${EVAL_SEED}
STEP1_GEN_END=$(date +%s)
STEP1_GEN_SEC=$(elapsed $STEP1_GEN_START $STEP1_GEN_END)

STEP1_TOTAL=$(( STEP1_GT_SEC + STEP1_GEN_SEC ))
echo ">>> Step 1 done: gt=${STEP1_GT_SEC}s  gen=${STEP1_GEN_SEC}s  total=${STEP1_TOTAL}s"

# =============================================================================
# Step 2: Filter and Shuffle MCQs
# =============================================================================
echo ""
echo ">>> Step 2: Filtering MCQs (gt)..."
STEP2_GT_START=$(date +%s)
python src/mcq_filtering.py \
    --input-json "${OUTPUT_DIR}/shuffled_ans_choices_data/gt_reports_as_ref/mcqa_data.json" \
    --output-dir "${OUTPUT_DIR}/shuffled_ans_choices_data/gt_reports_as_ref/mcqa_filtering" \
    --seed ${EVAL_SEED}
STEP2_GT_END=$(date +%s)
STEP2_GT_SEC=$(elapsed $STEP2_GT_START $STEP2_GT_END)

echo ">>> Step 2: Filtering MCQs (gen)..."
STEP2_GEN_START=$(date +%s)
python src/mcq_filtering.py \
    --input-json "${OUTPUT_DIR}/shuffled_ans_choices_data/gen_reports_as_ref/mcqa_data.json" \
    --output-dir "${OUTPUT_DIR}/shuffled_ans_choices_data/gen_reports_as_ref/mcqa_filtering" \
    --seed ${EVAL_SEED}
STEP2_GEN_END=$(date +%s)
STEP2_GEN_SEC=$(elapsed $STEP2_GEN_START $STEP2_GEN_END)

STEP2_TOTAL=$(( STEP2_GT_SEC + STEP2_GEN_SEC ))
echo ">>> Step 2 done: gt=${STEP2_GT_SEC}s  gen=${STEP2_GEN_SEC}s  total=${STEP2_TOTAL}s"

# =============================================================================
# Step 3: MCQA Evaluation
# =============================================================================
echo ""
echo ">>> Step 3: Running MCQA evaluation..."
STEP3_START=$(date +%s)
python src/mcqa_evaluation.py \
    --base_dir "${OUTPUT_DIR}" \
    --data_type "shuffled_ans_choices_data" \
    --seed ${EVAL_SEED} \
    --gen_report_csv_file "${INPUT_CSV}" \
    --gt_report_csv_file "${INPUT_CSV}"
STEP3_END=$(date +%s)
STEP3_SEC=$(elapsed $STEP3_START $STEP3_END)
echo ">>> Step 3 done: ${STEP3_SEC}s"

# =============================================================================
# Summary
# =============================================================================
PIPELINE_END=$(date +%s)
TOTAL_SEC=$(elapsed $PIPELINE_START $PIPELINE_END)

PER_SAMPLE_STEP1=$(echo "scale=2; ${STEP1_TOTAL} / ${NUM_SAMPLES}" | bc)
PER_SAMPLE_STEP2=$(echo "scale=2; ${STEP2_TOTAL} / ${NUM_SAMPLES}" | bc)
PER_SAMPLE_STEP3=$(echo "scale=2; ${STEP3_SEC} / ${NUM_SAMPLES}" | bc)
PER_SAMPLE_TOTAL=$(echo "scale=2; ${TOTAL_SEC} / ${NUM_SAMPLES}" | bc)

echo ""
echo "============================================="
echo "  Timing Summary (NUM_QUESTIONS=${NUM_QUESTIONS}, N=${NUM_SAMPLES} samples)"
echo "============================================="
printf "  %-35s %6d s   (%6.2f s/sample)\n" "Step 1 gt (MCQ generation):"  ${STEP1_GT_SEC}  $(echo "scale=2; ${STEP1_GT_SEC}  / ${NUM_SAMPLES}" | bc)
printf "  %-35s %6d s   (%6.2f s/sample)\n" "Step 1 gen (MCQ generation):" ${STEP1_GEN_SEC} $(echo "scale=2; ${STEP1_GEN_SEC} / ${NUM_SAMPLES}" | bc)
printf "  %-35s %6d s   (%6.2f s/sample)\n" "Step 1 total:"                ${STEP1_TOTAL}   ${PER_SAMPLE_STEP1}
echo "  ---"
printf "  %-35s %6d s   (%6.2f s/sample)\n" "Step 2 gt (MCQ filtering):"   ${STEP2_GT_SEC}  $(echo "scale=2; ${STEP2_GT_SEC}  / ${NUM_SAMPLES}" | bc)
printf "  %-35s %6d s   (%6.2f s/sample)\n" "Step 2 gen (MCQ filtering):"  ${STEP2_GEN_SEC} $(echo "scale=2; ${STEP2_GEN_SEC} / ${NUM_SAMPLES}" | bc)
printf "  %-35s %6d s   (%6.2f s/sample)\n" "Step 2 total:"                ${STEP2_TOTAL}   ${PER_SAMPLE_STEP2}
echo "  ---"
printf "  %-35s %6d s   (%6.2f s/sample)\n" "Step 3 (MCQA evaluation):"    ${STEP3_SEC}     ${PER_SAMPLE_STEP3}
echo "  ---"
printf "  %-35s %6d s   (%6.2f s/sample)\n" "Total pipeline:"              ${TOTAL_SEC}     ${PER_SAMPLE_TOTAL}
echo "============================================="
