#!/bin/bash
#SBATCH --job-name=mimic_%a
#SBATCH --partition=gpu4_medium
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=40:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=rd3571@nyu.edu
#SBATCH --array=0-4

# Load environment variables from .env file
if [ -f ".env" ]; then
    echo "Loading environment variables from .env file..."
    export $(cat .env | grep -v '^#' | xargs)
else
    echo "Warning: .env file not found. Make sure environment variables are set."
fi

# Load conda (save/restore cwd since ~/.bashrc may change it)
ORIG_DIR=$(pwd)
source ~/.bashrc
cd "$ORIG_DIR"
conda activate rrg-eval-clean

# MODEL_SEED: single value per submission. Change it (1-5) and resubmit to run
# other model seeds. EVAL_SEED is swept across the SLURM array tasks below.
MODEL_SEED=1

# Each array task evaluates one eval seed; all 5 run simultaneously.
EVAL_SEEDS=(123 456 789 202 101)
EVAL_SEED=${EVAL_SEEDS[$SLURM_ARRAY_TASK_ID]}
export PYTHONHASHSEED=$EVAL_SEED

# Define paths (RRGEVAL_BASE_DATA_PATH is loaded from .env)
INPUT_CSV="${RRGEVAL_BASE_DATA_PATH}/RRG_models/mimic-cxr-findings-baseline/results/iuxray_report_gen_findings_frontal_seed${MODEL_SEED}_20250106_213559.csv"
OUTPUT_DIR="${ORIG_DIR}/outputs/IU_xray/mimic-cxr-findings-baseline/model_seed_${MODEL_SEED}/eval_seed_${EVAL_SEED}"

# Create necessary directories
mkdir -p "${OUTPUT_DIR}"

# Step 2: Generate MCQs for both GT and Gen reports
echo "Processing EVAL_SEED: ${EVAL_SEED} (model report seed fixed at ${MODEL_SEED})"
echo "Generating MCQs..."
for ref in "gt" "gen" ; do
    python src/mcq_generation.py \
        --input_csv "${INPUT_CSV}" \
        --output_dir "${OUTPUT_DIR}" \
        --reference "$ref" \
        --num_questions 40 \
        --seed ${EVAL_SEED}
done

# Step 3: Filter and Shuffle MCQs
echo "Filtering and shuffling MCQs..."
for data_type in "orig_data" "shuffled_ans_choices_data"; do
    for ref in "gt" "gen"; do
        INPUT_DIR="${OUTPUT_DIR}/${data_type}/${ref}_reports_as_ref"
        FILTERED_DIR="${INPUT_DIR}"
        
        python src/mcq_filtering.py \
            --input-json "${INPUT_DIR}/mcqa_data.json" \
            --output-dir "${FILTERED_DIR}/mcqa_filtering" \
            --seed ${EVAL_SEED}
    done
done

# Step 4: MCQA evaluation
echo "MCQA evaluation..."
for data_type in "orig_data" "shuffled_ans_choices_data"; do
    # Step 4: Evaluate MCQs for each setting
    echo "Evaluating ${data_type} setting..."
    python src/mcqa_evaluation.py \
        --base_dir "${OUTPUT_DIR}" \
        --data_type "${data_type}" \
        --seed ${EVAL_SEED} \
        --gen_report_csv_file "$INPUT_CSV" \
        --gt_report_csv_file "$INPUT_CSV"
done

echo "Pipeline completed successfully for EVAL_SEED: ${EVAL_SEED}!"

# Print summary of generated files
echo -e "\nGenerated files structure for EVAL_SEED ${EVAL_SEED}:"
tree "${OUTPUT_DIR}" -L 4