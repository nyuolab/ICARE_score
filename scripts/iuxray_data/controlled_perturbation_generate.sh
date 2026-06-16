#!/bin/bash
#SBATCH --partition cpu_short
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 4
#SBATCH --mem 16G
#SBATCH --time 0-01:00:00
#SBATCH --job-name ctrl_perturb_generate
#SBATCH --output logs/ctrl_perturb_generate_%j.log

# =============================================================================
# Generate controlled perturbed GT CSVs for matched deletion experiment.
# Three conditions: clinical_ctrl / nonclinical / random_ctrl
# Five rates: 0%, 10%, 20%, 30%, 40% of clinical tokens.
#
# Run this FIRST, then submit controlled_perturbation_icare.sh and
# controlled_perturbation_baselines.sh as dependents:
#
#   cd /gpfs/data/oermannlab/users/rd3571/ICARE_score
#   mkdir -p logs
#   JID=$(sbatch --parsable scripts/iuxray_data/controlled_perturbation_generate.sh)
#   sbatch --dependency=afterok:${JID} scripts/iuxray_data/controlled_perturbation_icare.sh
#   sbatch --dependency=afterok:${JID} scripts/iuxray_data/controlled_perturbation_baselines.sh
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

declare -A INPUT_CSVS
INPUT_CSVS["maira-2"]="${RRGEVAL_BASE_DATA_PATH}/RRG_models/maira-2/results/iuxray_report_gen_findings_frontal+lateral_seed${MODEL_SEED}_20250107_003058.csv"
INPUT_CSVS["mimic-cxr-findings-baseline"]="${RRGEVAL_BASE_DATA_PATH}/RRG_models/mimic-cxr-findings-baseline/results/iuxray_report_gen_findings_frontal_seed${MODEL_SEED}_20250106_213559.csv"
INPUT_CSVS["chexpert-mimic-cxr-findings-baseline"]="${RRGEVAL_BASE_DATA_PATH}/RRG_models/chexpert-mimic-cxr-findings-baseline/results/iuxray_report_gen_findings_frontal+lateral_seed${MODEL_SEED}_20250106_211756.csv"

for MODEL in "maira-2"; do
    INPUT_CSV="${INPUT_CSVS[$MODEL]}"
    OUTPUT_DIR="${ORIG_DIR}/outputs/IU_xray/${MODEL}/model_seed_${MODEL_SEED}/eval_seed_${EVAL_SEED}"
    SENTINEL="${OUTPUT_DIR}/perturbed_reports_random_ctrl_level/perturbed_40percent.csv"

    echo ""
    echo "=== ${MODEL} ==="

    if [ -f "${SENTINEL}" ]; then
        echo "CSVs already exist — skipping generation for ${MODEL}."
        continue
    fi

    echo "Generating controlled perturbed CSVs..."
    python src/generate_controlled_perturbed_reports.py \
        --input_csv  "${INPUT_CSV}" \
        --output_dir "${OUTPUT_DIR}" \
        --seed       "${EVAL_SEED}" \
        --rates      0.0 0.1 0.2 0.3 0.4

    echo "Done: ${MODEL}"
done

echo ""
echo "All CSVs generated."
conda deactivate
