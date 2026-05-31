#!/bin/bash
#SBATCH --job-name=icare_iuxray_predefined_%a
#SBATCH --partition=a100_short
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=10:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=rd3571@nyu.edu
#SBATCH --array=0-2

# =============================================================================
# Run ICARE on IU-Xray (all 3 models) using a predefined (fixed) question list.
# Array tasks 0-2 run all 3 models in parallel.
# Skips MCQ generation and filtering; runs evaluation once per model.
#
# Usage:
#   cd ICARE_score
#   sbatch scripts/iuxray_data/icare_predefined.sh
#   # or locally (runs task 0 = maira-2 only):
#   bash scripts/iuxray_data/icare_predefined.sh
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
MODEL_SEED="${MODEL_SEED:-1}"
EVAL_SEED="${EVAL_SEED:-202}"
BASE_DATA_PATH="${RRGEVAL_BASE_DATA_PATH:-/gpfs/data/oermannlab/users/rd3571}"

PREDEFINED_JSON="${PREDEFINED_JSON:-${ORIG_DIR}/outputs/predefined_ques_list/mcqa_data.json}"
PREDEFINED_CSV="${PREDEFINED_CSV:-${ORIG_DIR}/outputs/predefined_ques_list/predefined_questions.csv}"

export PYTHONHASHSEED="${EVAL_SEED}"

# -----------------------------------------------------------------------------
# Pick model for this array task
# -----------------------------------------------------------------------------
MODEL_NAMES=("maira-2" "mimic-cxr-findings-baseline" "chexpert-mimic-cxr-findings-baseline")
MODEL_CSV_PATHS=(
    "${BASE_DATA_PATH}/RRG_models/maira-2/results/iuxray_report_gen_findings_frontal+lateral_seed${MODEL_SEED}_20250107_003058.csv"
    "${BASE_DATA_PATH}/RRG_models/mimic-cxr-findings-baseline/results/iuxray_report_gen_findings_frontal_seed${MODEL_SEED}_20250106_213559.csv"
    "${BASE_DATA_PATH}/RRG_models/chexpert-mimic-cxr-findings-baseline/results/iuxray_report_gen_findings_frontal+lateral_seed${MODEL_SEED}_20250106_211756.csv"
)

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
MODEL_NAME="${MODEL_NAMES[$TASK_ID]}"
INPUT_CSV="${MODEL_CSV_PATHS[$TASK_ID]}"
OUTPUT_DIR="${ORIG_DIR}/outputs/IU_xray/${MODEL_NAME}/model_seed_${MODEL_SEED}/eval_seed_${EVAL_SEED}/predefined"

echo "============================================="
echo "  ICARE Score - IU-Xray (predefined questions)"
echo "============================================="
echo "Model       : ${MODEL_NAME}"
echo "Model seed  : ${MODEL_SEED}"
echo "Eval seed   : ${EVAL_SEED}"
echo "Input CSV   : ${INPUT_CSV}"
echo "Output dir  : ${OUTPUT_DIR}"
echo "Predefined  : ${PREDEFINED_CSV}"
echo "============================================="

mkdir -p "${OUTPUT_DIR}"

# -----------------------------------------------------------------------------
# Step 1: Convert predefined JSON → CSV (each task checks; first one writes)
# -----------------------------------------------------------------------------
if [ ! -f "${PREDEFINED_CSV}" ]; then
    echo ""
    echo ">>> Step 1: Converting predefined questions JSON to CSV..."
    python src/predefined_mcqa_to_csv.py \
        --input_json "${PREDEFINED_JSON}" \
        --output_csv "${PREDEFINED_CSV}"
    echo ">>> Step 1 complete."
else
    echo ">>> Step 1: Predefined CSV already exists, skipping."
fi

# -----------------------------------------------------------------------------
# Step 2: MCQA evaluation
# -----------------------------------------------------------------------------
echo ""
echo ">>> Step 2: Running MCQA evaluation for ${MODEL_NAME}..."
python src/mcqa_evaluation.py \
    --base_dir "${OUTPUT_DIR}" \
    --seed "${EVAL_SEED}" \
    --gen_report_csv_file "${INPUT_CSV}" \
    --gt_report_csv_file "${INPUT_CSV}" \
    --predefined_ques_csv "${PREDEFINED_CSV}"
echo ">>> Step 2 complete."

echo ""
echo "============================================="
echo "  Done: ${MODEL_NAME}"
echo "  ${OUTPUT_DIR}/mcqa_eval/"
echo "============================================="
