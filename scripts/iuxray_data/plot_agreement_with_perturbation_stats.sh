#!/bin/bash
#SBATCH --partition gpu8_medium
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu 20G
#SBATCH --time 0-2:00:00
#SBATCH --job-name plot_agreement_with_perturbation_stats
#SBATCH --output logs/plot_agreement_with_perturbation_stats_%A_%a.log

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
INPUT_DIR="/gpfs/data/oermannlab/users/rd3571/RRG_evaluation/MCQ_generation/MCQ_gen_data_our_eval_seed${EVAL_SEED}/IU_xray"

# Run the script with arguments
python src/plot_agreement_with_perturbation_stats.py \
    --eval_seed "$EVAL_SEED" \
    --model_seed "$MODEL_SEED" \
    --base_dir "$INPUT_DIR" \
    --dataset "iuxray"


# Clean up
conda deactivate