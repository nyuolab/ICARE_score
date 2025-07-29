#!/bin/bash
#SBATCH --partition gpu8_medium
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu 20G
#SBATCH --time 0-24:00:00
#SBATCH --job-name maira2_perturb_reports
#SBATCH --output logs/maira2_perturb_reports_%A_%a.log
#SBATCH --array=0-19%10 # 20 total jobs (for perturbations 5-100), run 10 at a time

# Create logs directory if it doesn't exist
mkdir -p logs

# Load environment variables from .env file
if [ -f ".env" ]; then
    echo "Loading environment variables from .env file..."
    export $(cat .env | grep -v '^#' | xargs)
else
    echo "Warning: .env file not found. Make sure environment variables are set."
fi

# Load modules
source ~/.bashrc
conda activate green

# Set variables
EVAL_SEED="123"
MODEL_SEED="1"

# Define paths
INPUT_CSV="/gpfs/data/oermannlab/users/rd3571/RRG_models/maira-2/results/iuxray_report_gen_findings_frontal+lateral_seed${MODEL_SEED}_20250107_003058.csv"
OUTPUT_DIR="/gpfs/data/oermannlab/users/rd3571/RRG_evaluation/MCQ_generation/MCQ_gen_data_our_eval_seed${EVAL_SEED}/IU_xray/maira-2/seed_${MODEL_SEED}"

# Run the script with arguments
python src/generate_perturbed_reports.py \
    --input_csv "$INPUT_CSV" \
    --output_dir "$OUTPUT_DIR" \
    --seed "$EVAL_SEED"

# Calculate perturbation degree based on array task ID
PERTURBATION_DEGREE=$((($SLURM_ARRAY_TASK_ID * 5) + 5))

# Step 4: MCQA evaluation
echo "MCQA evaluation..."
for data_type in "shuffled_ans_choices_data"; do
    echo "Evaluating ${data_type} setting with perturbation degree ${PERTURBATION_DEGREE}..."
    python src/mcqa_evaluation.py \
        --base_dir "${OUTPUT_DIR}" \
        --data_type "${data_type}" \
        --seed ${EVAL_SEED} \
        --gen_report_csv_file "$INPUT_CSV" \
        --gt_report_csv_file "$INPUT_CSV" \
        --perturbation "perturbed" \
        --perturbation_degree ${PERTURBATION_DEGREE} \
        --perturbation_type "char"
done

# Clean up
conda deactivate


