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

# Load conda
module load anaconda3/gpu/2023.09
conda activate green

# Set the working directory
cd /gpfs/data/oermannlab/users/rd3571/RRG_evaluation/MCQ_generation/RRGEval/src

# Define paths
OUTPUT_DIR="/gpfs/data/oermannlab/users/rd3571/RRG_evaluation/MCQ_generation/MCQ_gen_data_our_eval_summarized_results/IU_xray/question_categorization_and_analysis"
CLUSTERED_DATA_PATH="$OUTPUT_DIR/clustered_questions.csv"
CLUSTER_NAMES_PATH="$OUTPUT_DIR/cluster_names.json"
ANALYSIS_FOLDER="$OUTPUT_DIR/analysis"

python question_categorization_and_analysis/cluster_analysis.py \
    --clustered_data_path "$CLUSTERED_DATA_PATH" \
    --cluster_names_path "$CLUSTER_NAMES_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --analysis_folder "$ANALYSIS_FOLDER"
