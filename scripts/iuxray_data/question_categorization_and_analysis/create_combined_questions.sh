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
# module load miniconda3/gpu/4.9.2
source ~/.bashrc
conda activate green

# Set the working directory
cd /gpfs/data/oermannlab/users/rd3571/RRG_evaluation/MCQ_generation/RRGEval/src

# Define configuration parameters
MODEL_SEEDS='[1, 2, 3, 4, 5]'
EVAL_SEEDS='[123, 456, 789, 202, 101]'
DATASETS='["IU_xray"]'
MODELS='["chexpert-mimic-cxr-findings-baseline", "mimic-cxr-findings-baseline", "maira-2"]'
METRICS='["gt_reports_as_ref", "gen_reports_as_ref"]'
BASE_DIR="/gpfs/data/oermannlab/users/rd3571/RRG_evaluation/MCQ_generation/MCQ_gen_data_our_eval_seed"
OUTPUT_DIR="/gpfs/data/oermannlab/users/rd3571/RRG_evaluation/MCQ_generation/MCQ_gen_data_our_eval_summarized_results/IU_xray/question_categorization_and_analysis"

# Run the Python script with command-line arguments
python question_categorization_and_analysis/create_combined_questions.py \
    --model_seeds "$MODEL_SEEDS" \
    --eval_seeds "$EVAL_SEEDS" \
    --datasets "$DATASETS" \
    --models "$MODELS" \
    --metrics "$METRICS" \
    --base_dir "$BASE_DIR" \
    --output_dir "$OUTPUT_DIR"
