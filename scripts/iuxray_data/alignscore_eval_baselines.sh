#!/bin/bash
#SBATCH --job-name=alignscore_iuxray_%a
#SBATCH --partition=a100_short
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=rd3571@nyu.edu
#SBATCH --array=0-2

# =============================================================================
# Run AlignScore for IU-Xray (all 3 models) at model seed 1.
# Array tasks 0-2 run all 3 models in parallel.
#
# Usage:
#   cd /gpfs/data/oermannlab/users/rd3571/ICARE_score
#   mkdir -p outputs/IU_xray/baselines/logs
#   sbatch scripts/iuxray_data/alignscore_eval_baselines.sh
# =============================================================================

set -eo pipefail

SCRIPT_ROOT="/gpfs/data/oermannlab/users/rd3571/ICARE_score"
CKPT_PATH="/gpfs/data/oermannlab/users/rd3571/AlignScore-base.ckpt"

if [ -f "${SCRIPT_ROOT}/.env" ]; then
    set -a
    # shellcheck disable=SC1090
    source "${SCRIPT_ROOT}/.env"
    set +a
else
    echo "Warning: .env not found. Falling back to default base path."
fi

BASE_DATA_PATH="${RRGEVAL_BASE_DATA_PATH:-/gpfs/data/oermannlab/users/rd3571}"
MODEL_SEED=1

# -----------------------------------------------------------------------------
# Model list — must match green_eval_baselines.sh / rrg_eval_baselines.sh
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
RESULTS_DIR="${SCRIPT_ROOT}/outputs/IU_xray/baselines/${MODEL_NAME}/model_seed_${MODEL_SEED}/alignscore"
DONE_FLAG="${RESULTS_DIR}/summary_of_averages.csv"

# -----------------------------------------------------------------------------
# Activate environment — source ~/.bashrc first, then module load (critical order)
# -----------------------------------------------------------------------------
ORIG_DIR=$(pwd)
source ~/.bashrc
cd "$ORIG_DIR"
module load cuda/11.8
conda activate alignscore
export PYTHONNOUSERSITE=1

if ! python -c "from alignscore import AlignScore" 2>/dev/null; then
    echo "Error: alignscore env missing AlignScore. Active python: $(which python)"
    exit 1
fi

mkdir -p "${RESULTS_DIR}"

echo "Job ID     : ${SLURM_JOB_ID}"
echo "Node       : ${SLURM_JOB_NODELIST}"
echo "Array task : ${TASK_ID}"
echo "Model      : ${MODEL_NAME}"
echo "Model seed : ${MODEL_SEED}"
echo "Input CSV  : ${INPUT_CSV}"
echo "Results dir: ${RESULTS_DIR}"
echo "Checkpoint : ${CKPT_PATH}"
echo "Start      : $(date)"

if [ -f "${DONE_FLAG}" ]; then
    echo "Already done — skipping."
    exit 0
fi

if [ ! -f "${INPUT_CSV}" ]; then
    echo "Error: input CSV not found: ${INPUT_CSV}"
    exit 1
fi

echo ""
echo ">>> Running AlignScore..."
python "${SCRIPT_ROOT}/src/run_alignscore_eval.py" \
    --input-csv  "${INPUT_CSV}" \
    --output-dir "${RESULTS_DIR}" \
    --ckpt-path  "${CKPT_PATH}" \
    --model      roberta-base \
    --batch-size 32 \
    --device     cuda:0 \
    --eval-mode  nli_sp
echo ">>> AlignScore complete."

echo ""
echo "============================================="
echo "  IU-Xray ${MODEL_NAME} AlignScore complete."
echo "  Results: ${RESULTS_DIR}"
echo "============================================="
echo "End: $(date)"
