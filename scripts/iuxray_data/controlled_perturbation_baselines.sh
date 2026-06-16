#!/bin/bash
#SBATCH --partition gpu8_long
#SBATCH --gpus 1
#SBATCH --cpus-per-task 4
#SBATCH --mem 64G
#SBATCH --time 0-12:00:00
#SBATCH --job-name ctrl_perturb_baselines
#SBATCH --output logs/ctrl_perturb_baselines_%j.log
#SBATCH --mail-type END,FAIL
#SBATCH --mail-user rd3571@nyu.edu

# =============================================================================
# CXR-Report-Metric baselines (BLEU, BERTScore, RadGraph, RadCliQ) for the
# controlled matched deletion experiment.
#
# Evaluates metric(perturbed_GT, original_GT) for all 3 conditions × 5 rates
# across all 3 RRG models.  Runs sequentially to avoid temp_dygie race condition.
#
# Submit AFTER controlled_perturbation_generate.sh completes.
# =============================================================================

set -eo pipefail

mkdir -p logs

PROJECT_ROOT="/gpfs/data/oermannlab/users/rd3571/RRG_evaluation/CXR-Report-Metric"
SCRIPT_ROOT="/gpfs/data/oermannlab/users/rd3571/ICARE_score"

if [ -f "${SCRIPT_ROOT}/.env" ]; then
    set -a; source "${SCRIPT_ROOT}/.env"; set +a
else
    echo "Warning: .env not found."
fi

ORIG_DIR=$(pwd)
source ~/.bashrc
cd "$ORIG_DIR"
conda activate rrg-eval-clean
export PYTHONNOUSERSITE=1

cd "${PROJECT_ROOT}"

MODEL_SEED=1
EVAL_SEED=123
MODELS=("maira-2")
CONDITIONS=("clinical_ctrl" "nonclinical" "random_ctrl")
DEGREES=(0 10 20 30 40)

echo "Job ID : ${SLURM_JOB_ID}"
echo "Start  : $(date)"

for MODEL in "${MODELS[@]}"; do
    BASE_DIR="${SCRIPT_ROOT}/outputs/IU_xray/${MODEL}/model_seed_${MODEL_SEED}/eval_seed_${EVAL_SEED}"
    ORIG_CSV="${BASE_DIR}/perturbed_reports_clinical_ctrl_level/perturbed_0percent.csv"

    if [ ! -f "${ORIG_CSV}" ]; then
        echo "ERROR: baseline CSV not found at ${ORIG_CSV} — skipping ${MODEL}"
        continue
    fi

    echo ""
    echo "=========================================="
    echo "  Model: ${MODEL}"
    echo "=========================================="

    for COND in "${CONDITIONS[@]}"; do
        PERTURBED_DIR="${BASE_DIR}/perturbed_reports_${COND}_level"

        echo ""
        echo "  Condition: ${COND}"

        for DEG in "${DEGREES[@]}"; do
            PERTURBED_CSV="${PERTURBED_DIR}/perturbed_${DEG}percent.csv"
            RESULTS_DIR="${SCRIPT_ROOT}/outputs/IU_xray/baselines_controlled_perturbation/${MODEL}/model_seed_${MODEL_SEED}/${COND}/perturbation_degree${DEG}"
            DONE_FLAG="${RESULTS_DIR}/summary_of_averages.csv"

            if [ -f "${DONE_FLAG}" ]; then
                echo "    ${DEG}%: already done — skipping."
                continue
            fi

            if [ ! -f "${PERTURBED_CSV}" ]; then
                echo "    WARNING: ${PERTURBED_CSV} not found — skipping."
                continue
            fi

            mkdir -p "${RESULTS_DIR}"
            HYBRID_CSV="${RESULTS_DIR}/hybrid_perturbed_gen_orig_gt.csv"

            # Build hybrid CSV: perturbed generated_report + original ground_truth_report
            # (The perturbed CSV already has original GT in ground_truth_report,
            #  but we explicitly pull from the 0% CSV for clarity and safety.)
            python - <<PYEOF
import pandas as pd
orig     = pd.read_csv("${ORIG_CSV}")
perturbed = pd.read_csv("${PERTURBED_CSV}")
hybrid   = perturbed.copy()
hybrid["ground_truth_report"] = orig["ground_truth_report"]
hybrid.to_csv("${HYBRID_CSV}", index=False)
print(f"    Hybrid CSV: {len(hybrid)} reports  [${COND} @ ${DEG}%]")
PYEOF

            echo "    Running CXR-Report-Metric  [${COND} @ ${DEG}%]..."
            python run_bulk_eval.py \
                --input-files "${HYBRID_CSV}" \
                --output-dir  "${RESULTS_DIR}"

            python summarize_results.py \
                --results-dir "${RESULTS_DIR}"

            echo "    Done: ${MODEL} / ${COND} / ${DEG}%  $(date)"
        done
    done
done

echo ""
echo "All baseline evaluations complete."
echo "End: $(date)"
