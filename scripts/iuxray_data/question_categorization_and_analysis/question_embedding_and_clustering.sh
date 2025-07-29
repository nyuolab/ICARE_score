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
source ~/.bashrc
# module load miniconda3/gpu/4.9.2
conda activate green


# Define paths
OUTPUT_DIR="/gpfs/data/oermannlab/users/rd3571/RRG_evaluation/MCQ_generation/MCQ_gen_data_our_eval_summarized_results/IU_xray/question_categorization_and_analysis"
COMBINED_DATA_PATH="$OUTPUT_DIR/combined_mcqa_data.csv"

# Set the working directory
cd /gpfs/data/oermannlab/users/rd3571/RRG_evaluation/MCQ_generation/RRGEval/src
python question_categorization_and_analysis/question_embedding_and_clustering.py \
    --output_dir "$OUTPUT_DIR" \
    --combined_data_path "$COMBINED_DATA_PATH"