#!/bin/bash
#SBATCH --job-name=mcq_analysis
#SBATCH --output=mcq_analysis_%j.out
#SBATCH --error=mcq_analysis_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpu4_medium 
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=rd3571@nyu.edu

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

# Define configuration parameters
MODEL_SEEDS='[1, 2, 3, 4, 5]'
EVAL_SEEDS='[123, 456, 789, 202, 101]'
DATASETS='["IU_xray"]'
MODELS='["chexpert-mimic-cxr-findings-baseline", "mimic-cxr-findings-baseline", "maira-2"]'
METRICS='["gt_reports_as_ref", "gen_reports_as_ref"]'

# Define paths (RRGEVAL_BASE_DATA_PATH is loaded from .env)
BASE_DIR="${ORIG_DIR}/outputs"
OUTPUT_DIR="${ORIG_DIR}/outputs/IU_xray/_summary/question_categorization_and_analysis"

# Run the Python script with command-line arguments
python src/question_categorization_and_analysis/create_combined_questions.py \
    --model_seeds "$MODEL_SEEDS" \
    --eval_seeds "$EVAL_SEEDS" \
    --datasets "$DATASETS" \
    --models "$MODELS" \
    --metrics "$METRICS" \
    --base_dir "$BASE_DIR" \
    --output_dir "$OUTPUT_DIR"
