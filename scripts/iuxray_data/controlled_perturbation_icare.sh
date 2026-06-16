#!/bin/bash
#SBATCH --partition gpu4_medium
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu 20G
#SBATCH --time 0-24:00:00
#SBATCH --job-name ctrl_perturb_icare
#SBATCH --output logs/ctrl_perturb_icare_%A_%a.log
#SBATCH --array=0-12   # 13 tasks: 1 baseline + 3 conditions × 4 rates (maira-2 only)

# =============================================================================
# ICARE evaluation for the controlled matched deletion experiment.
#
# Array layout (per model block of 13):
#   0  : 0% baseline  (clinical_ctrl, degree 0)
#   1-4: clinical_ctrl  degrees 10 20 30 40
#   5-8: nonclinical    degrees 10 20 30 40
#   9-12: random_ctrl   degrees 10 20 30 40
#
# Submit AFTER controlled_perturbation_generate.sh completes.
# =============================================================================

mkdir -p logs

if [ -f ".env" ]; then
    set -a; source .env; set +a
else
    echo "Warning: .env not found."
fi

ORIG_DIR=$(pwd)
source ~/.bashrc
cd "$ORIG_DIR"
conda activate rrg-eval-clean

MODEL_SEED=1
EVAL_SEED=123

MODEL="maira-2"
INPUT_CSV="${RRGEVAL_BASE_DATA_PATH}/RRG_models/maira-2/results/iuxray_report_gen_findings_frontal+lateral_seed${MODEL_SEED}_20250107_003058.csv"
BASE_DIR="${ORIG_DIR}/outputs/IU_xray/${MODEL}/model_seed_${MODEL_SEED}/eval_seed_${EVAL_SEED}"

# Map task index (0-12) to condition and degree
CONDITIONS=("clinical_ctrl" "clinical_ctrl" "clinical_ctrl" "clinical_ctrl" "clinical_ctrl" \
            "nonclinical"   "nonclinical"   "nonclinical"   "nonclinical" \
            "random_ctrl"   "random_ctrl"   "random_ctrl"   "random_ctrl")
DEGREES=(0 10 20 30 40 10 20 30 40 10 20 30 40)

CONDITION="${CONDITIONS[$SLURM_ARRAY_TASK_ID]}"
DEGREE="${DEGREES[$SLURM_ARRAY_TASK_ID]}"

echo "Array task : ${SLURM_ARRAY_TASK_ID}"
echo "Model      : ${MODEL}"
echo "Condition  : ${CONDITION}"
echo "Degree     : ${DEGREE}%"
echo "Base dir   : ${BASE_DIR}"

# Guard: skip if result already exists
RESULT="${BASE_DIR}/shuffled_ans_choices_data/gt_reports_as_ref/mcqa_eval_perturbed_gen_reports_${CONDITION}_level/perturbation_degree${DEGREE}/mcq_eval_report_level_stats_aggregated.csv"
if [ -f "${RESULT}" ]; then
    echo "Result already exists — skipping."
    conda deactivate
    exit 0
fi

python src/mcqa_evaluation.py \
    --base_dir            "${BASE_DIR}" \
    --data_type           "shuffled_ans_choices_data" \
    --seed                ${EVAL_SEED} \
    --gen_report_csv_file "${INPUT_CSV}" \
    --gt_report_csv_file  "${INPUT_CSV}" \
    --perturbation        "perturbed" \
    --perturbation_degree ${DEGREE} \
    --perturbation_type   "${CONDITION}"

echo "Done: ${MODEL} / ${CONDITION} / ${DEGREE}%  $(date)"
conda deactivate
