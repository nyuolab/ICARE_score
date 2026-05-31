#!/bin/bash
#SBATCH --job-name=crimson_iuxray_%a
#SBATCH --partition=a100_long
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=40:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=rd3571@nyu.edu
#SBATCH --array=0-2
#SBATCH --output=/gpfs/data/oermannlab/users/rd3571/ICARE_score/outputs/IU_xray/baselines/logs/crimson-%A_%a.out
#SBATCH --error=/gpfs/data/oermannlab/users/rd3571/ICARE_score/outputs/IU_xray/baselines/logs/crimson-%A_%a.err

# =============================================================================
# Run CRIMSON metric on IU-Xray (all 3 models) at model seed 1 as a baseline.
# Array tasks 0-2 run all 3 models in parallel.
#
# Usage:
#   cd ICARE_score
#   sbatch scripts/iuxray_data/crimson_baselines_iuxray.sh
#   # or locally (runs task 0 = maira-2 only):
#   bash scripts/iuxray_data/crimson_baselines_iuxray.sh
# =============================================================================

set -eo pipefail

CRIMSON_DIR="/gpfs/data/oermannlab/users/rd3571/CRIMSON"
SCRIPT_ROOT="/gpfs/data/oermannlab/users/rd3571/ICARE_score"

if [ -f "${SCRIPT_ROOT}/.env" ]; then
    echo "Loading environment variables from .env file..."
    set -a
    # shellcheck disable=SC1090
    source "${SCRIPT_ROOT}/.env"
    set +a
else
    echo "Warning: .env file not found. Falling back to default base path."
fi

BASE_DATA_PATH="${RRGEVAL_BASE_DATA_PATH:-/gpfs/data/oermannlab/users/rd3571}"
MODEL_SEED=1
BATCH_SIZE="${BATCH_SIZE:-8}"   # HuggingFace forward-pass batch size

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

RESULTS_DIR="${SCRIPT_ROOT}/outputs/IU_xray/baselines/${MODEL_NAME}/model_seed_${MODEL_SEED}/crimson"
OUTPUT_JSON="${RESULTS_DIR}/crimson_results.json"
LOGS_DIR="${SCRIPT_ROOT}/outputs/IU_xray/baselines/logs"

mkdir -p "${RESULTS_DIR}" "${LOGS_DIR}"

if [ ! -f "${INPUT_CSV}" ]; then
    echo "Error: input CSV not found: ${INPUT_CSV}"
    exit 1
fi

echo "--- Activating Conda Environment ---"
ORIG_DIR=$(pwd)
source ~/.bashrc
cd "$ORIG_DIR"
conda activate crimson
# Clear PYTHONPATH to prevent base conda's Python 3.8 packages from
# contaminating the crimson (Python 3.12) environment.
unset PYTHONPATH

cd "$CRIMSON_DIR" || { echo "Failed to change to $CRIMSON_DIR"; exit 1; }

echo "Job ID     : ${SLURM_JOB_ID}"
echo "Array task : ${TASK_ID}"
echo "Node       : ${SLURM_JOB_NODELIST}"
echo "Model      : ${MODEL_NAME}"
echo "Model seed : ${MODEL_SEED}"
echo "Start      : $(date)"
echo "Input      : ${INPUT_CSV}"
echo "Output     : ${OUTPUT_JSON}"
echo "Batch size : ${BATCH_SIZE}"
echo ""

python evaluate_reports.py \
    --input        "${INPUT_CSV}" \
    --gt-column    ground_truth_report \
    --pred-column  generated_report \
    --output       "${OUTPUT_JSON}" \
    --details \
    --batch-size   "${BATCH_SIZE}"

echo ""
echo "============================================="
echo "  IU-Xray ${MODEL_NAME} CRIMSON complete."
echo "  Results: ${OUTPUT_JSON}"
echo "============================================="
echo "End: $(date)"
