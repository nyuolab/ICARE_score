#!/bin/bash
#SBATCH --partition=a100_short
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=0-02:00:00
#SBATCH --job-name=ctrl_perturb_alignscore_%a
#SBATCH --output=logs/ctrl_perturb_alignscore_%A_%a.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=rd3571@nyu.edu
#SBATCH --array=0-14

# =============================================================================
# AlignScore evaluation for the controlled matched deletion experiment (maira-2).
# Runs in parallel: each array task handles one condition × rate combination.
#
# Array layout:
#   0-4  : clinical_ctrl  @ 0 10 20 30 40%
#   5-9  : nonclinical    @ 0 10 20 30 40%
#   10-14: random_ctrl    @ 0 10 20 30 40%
#
# Reuses hybrid CSVs already created by controlled_perturbation_baselines.sh.
# Submit AFTER controlled_perturbation_baselines.sh completes.
# =============================================================================

set -eo pipefail

mkdir -p logs

SCRIPT_ROOT="/gpfs/data/oermannlab/users/rd3571/ICARE_score"
CKPT_PATH="/gpfs/data/oermannlab/users/rd3571/AlignScore-base.ckpt"

if [ -f "${SCRIPT_ROOT}/.env" ]; then
    set -a; source "${SCRIPT_ROOT}/.env"; set +a
else
    echo "Warning: .env not found."
fi

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

cd "${SCRIPT_ROOT}"

MODEL_SEED=1
MODEL="maira-2"

# Map array task ID → (condition, degree)
CONDITIONS=("clinical_ctrl" "clinical_ctrl" "clinical_ctrl" "clinical_ctrl" "clinical_ctrl" \
            "nonclinical"   "nonclinical"   "nonclinical"   "nonclinical"   "nonclinical" \
            "random_ctrl"   "random_ctrl"   "random_ctrl"   "random_ctrl"   "random_ctrl")
DEGREES=(0 10 20 30 40  0 10 20 30 40  0 10 20 30 40)

COND="${CONDITIONS[$SLURM_ARRAY_TASK_ID]}"
DEG="${DEGREES[$SLURM_ARRAY_TASK_ID]}"

RESULTS_DIR="${SCRIPT_ROOT}/outputs/IU_xray/baselines_controlled_perturbation/${MODEL}/model_seed_${MODEL_SEED}/${COND}/perturbation_degree${DEG}"
HYBRID_CSV="${RESULTS_DIR}/hybrid_perturbed_gen_orig_gt.csv"
ALIGNSCORE_DIR="${RESULTS_DIR}/alignscore"
DONE_FLAG="${ALIGNSCORE_DIR}/summary_of_averages.csv"

echo "Array task : ${SLURM_ARRAY_TASK_ID}"
echo "Condition  : ${COND}"
echo "Degree     : ${DEG}%"
echo "Job ID     : ${SLURM_JOB_ID}"
echo "Node       : ${SLURM_JOB_NODELIST}"
echo "Start      : $(date)"

if [ -f "${DONE_FLAG}" ]; then
    echo "Already done — skipping."
    exit 0
fi

if [ ! -f "${HYBRID_CSV}" ]; then
    echo "ERROR: hybrid CSV not found at ${HYBRID_CSV}"
    exit 1
fi

mkdir -p "${ALIGNSCORE_DIR}"

python src/run_alignscore_eval.py \
    --input-csv  "${HYBRID_CSV}" \
    --output-dir "${ALIGNSCORE_DIR}" \
    --ckpt-path  "${CKPT_PATH}" \
    --model      roberta-base \
    --batch-size 32 \
    --device     cuda:0 \
    --eval-mode  nli_sp

echo "Done: ${MODEL} / ${COND} / ${DEG}%  $(date)"
