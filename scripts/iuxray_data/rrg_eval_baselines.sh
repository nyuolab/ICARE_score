#!/bin/bash
#SBATCH --job-name=rrg_baselines_iuxray
#SBATCH --partition=gpu8_long
#SBATCH --gpus=4
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=rd3571@nyu.edu
#SBATCH --output=/gpfs/data/oermannlab/users/rd3571/ICARE_score/outputs/IU_xray/baselines/logs/%x-%j.out
#SBATCH --error=/gpfs/data/oermannlab/users/rd3571/ICARE_score/outputs/IU_xray/baselines/logs/%x-%j.err

# =============================================================================
# Run CXR-Report-Metric baselines (BLEU, BERTScore, s-emb, RadGraph, RadCliQ)
# for IU-Xray (all 3 models) at model seed 1, sequentially in one job.
# Sequential execution avoids ./temp_dygie_output.json race conditions that
# occur when multiple tasks share a node.
#
# Usage:
#   cd /gpfs/data/oermannlab/users/rd3571/ICARE_score
#   mkdir -p outputs/IU_xray/baselines/logs
#   sbatch scripts/iuxray_data/rrg_eval_baselines.sh
# =============================================================================

set -eo pipefail

PROJECT_ROOT="/gpfs/data/oermannlab/users/rd3571/RRG_evaluation/CXR-Report-Metric"
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

if [ ! -f "${PROJECT_ROOT}/run_bulk_eval.py" ]; then
    echo "Error: CXR-Report-Metric not found at ${PROJECT_ROOT}"
    exit 1
fi

echo "--- Activating Conda Environment ---"
ORIG_DIR=$(pwd)
source ~/.bashrc
cd "$ORIG_DIR"
conda activate rrg-eval-clean
export PYTHONNOUSERSITE=1

cd "${PROJECT_ROOT}" || { echo "Failed to change to ${PROJECT_ROOT}"; exit 1; }

echo "Job ID     : ${SLURM_JOB_ID}"
echo "Node       : ${SLURM_JOB_NODELIST}"
echo "Model seed : ${MODEL_SEED}"
echo "Start      : $(date)"

# -----------------------------------------------------------------------------
# Run all 3 models sequentially to avoid temp_dygie_output.json conflicts
# -----------------------------------------------------------------------------
for i in 0 1 2; do
    MODEL_NAME="${MODEL_NAMES[$i]}"
    INPUT_CSV="${MODEL_CSV_PATHS[$i]}"
    RESULTS_DIR="${SCRIPT_ROOT}/outputs/IU_xray/baselines/${MODEL_NAME}/model_seed_${MODEL_SEED}"

    echo ""
    echo "============================================="
    echo "  Model: ${MODEL_NAME}"
    echo "  Input: ${INPUT_CSV}"
    echo "  Output: ${RESULTS_DIR}"
    echo "============================================="

    if [ ! -f "${INPUT_CSV}" ]; then
        echo "Error: input CSV not found: ${INPUT_CSV} — skipping."
        continue
    fi

    mkdir -p "${RESULTS_DIR}"

    echo ">>> Step 1: Running CXR-Report-Metric..."
    python run_bulk_eval.py \
        --input-files "${INPUT_CSV}" \
        --output-dir "${RESULTS_DIR}"
    echo ">>> Step 1 complete."

    echo ">>> Step 2: Summarizing average scores..."
    python summarize_results.py \
        --results-dir "${RESULTS_DIR}"
    echo ">>> Step 2 complete."

    echo "  Done: ${MODEL_NAME}  $(date)"
done

echo ""
echo "============================================="
echo "  All IU-Xray baseline metrics complete."
echo "============================================="
echo "End: $(date)"
