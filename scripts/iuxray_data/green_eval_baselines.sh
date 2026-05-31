#!/bin/bash
#SBATCH --job-name=green_iuxray_%a
#SBATCH --partition=a100_long
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=rd3571@nyu.edu
#SBATCH --array=0-2

# =============================================================================
# Run GREEN for IU-Xray (all 3 models) at model seed 1.
# Array tasks 0-2 run all 3 models in parallel.
#
# Usage:
#   cd /gpfs/data/oermannlab/users/rd3571/ICARE_score
#   sbatch scripts/iuxray_data/green_eval_baselines.sh
# =============================================================================

set -eo pipefail

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

# -----------------------------------------------------------------------------
# Pick model for this array task
# -----------------------------------------------------------------------------
MODEL_NAMES=(
    "maira-2"
    "mimic-cxr-findings-baseline"
    "chexpert-mimic-cxr-findings-baseline"
)
MODEL_CSV_PATHS=(
    "${BASE_DATA_PATH}/RRG_models/maira-2/results/iuxray_report_gen_findings_frontal+lateral_seed${MODEL_SEED}_20250107_003058.csv"
    "${BASE_DATA_PATH}/RRG_models/mimic-cxr-findings-baseline/results/iuxray_report_gen_findings_frontal_seed${MODEL_SEED}_20250106_213559.csv"
    "${BASE_DATA_PATH}/RRG_models/chexpert-mimic-cxr-findings-baseline/results/iuxray_report_gen_findings_frontal+lateral_seed${MODEL_SEED}_20250106_211756.csv"
)

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
MODEL_NAME="${MODEL_NAMES[$TASK_ID]}"
INPUT_CSV="${MODEL_CSV_PATHS[$TASK_ID]}"
RESULTS_DIR="${SCRIPT_ROOT}/outputs/IU_xray/baselines/${MODEL_NAME}/model_seed_${MODEL_SEED}/green"
LOGS_DIR="${RESULTS_DIR}/logs"

mkdir -p "${LOGS_DIR}"

if [ ! -f "${INPUT_CSV}" ]; then
    echo "Error: input CSV not found: ${INPUT_CSV}"
    exit 1
fi

echo "--- Activating Conda Environment ---"
ORIG_DIR=$(pwd)
source ~/.bashrc
cd "$ORIG_DIR"
conda activate regularization
export PYTHONNOUSERSITE=1
if ! python -c "import torch, transformers, datasets"; then
    echo "Error: regularization env missing GREEN dependencies. Active python: $(which python)"
    exit 1
fi
echo "Using python: $(which python)"

cd "${SCRIPT_ROOT}" || { echo "Failed to change to ${SCRIPT_ROOT}"; exit 1; }

echo "Job ID     : ${SLURM_JOB_ID}"
echo "Node       : ${SLURM_JOB_NODELIST}"
echo "Model      : ${MODEL_NAME}"
echo "Model seed : ${MODEL_SEED}"
echo "Input CSV  : ${INPUT_CSV}"
echo "Results dir: ${RESULTS_DIR}"
echo "Start      : $(date)"

echo ""
echo ">>> Running GREEN..."
python src/run_green_eval.py \
    --input-csv "${INPUT_CSV}" \
    --output-dir "${RESULTS_DIR}"
echo ">>> GREEN complete."

echo ""
echo "============================================="
echo "  IU-Xray ${MODEL_NAME} GREEN complete."
echo "  Results: ${RESULTS_DIR}"
echo "============================================="
echo "End: $(date)"
