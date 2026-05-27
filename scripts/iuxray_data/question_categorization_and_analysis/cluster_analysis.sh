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

# Define paths (RRGEVAL_BASE_DATA_PATH is loaded from .env)
OUTPUT_DIR="${ORIG_DIR}/outputs/IU_xray/_summary/question_categorization_and_analysis"
CLUSTERED_DATA_PATH="$OUTPUT_DIR/clustered_questions.csv"
CLUSTER_NAMES_PATH="$OUTPUT_DIR/cluster_names.json"
ANALYSIS_FOLDER="$OUTPUT_DIR/analysis"

python src/question_categorization_and_analysis/cluster_analysis.py \
    --clustered_data_path "$CLUSTERED_DATA_PATH" \
    --cluster_names_path "$CLUSTER_NAMES_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --analysis_folder "$ANALYSIS_FOLDER"
