#!/bin/bash
#SBATCH --job-name=icare_qcat
#SBATCH --partition=gpu4_medium
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=logs/icare_question_categorization_%j.log

# =============================================================================
# Run question categorization and analysis on example test output.
# Run scripts/example_test/run_eval.sh first to generate test_data/output/.
#
# Usage:
#   cd ICARE_score
#   sbatch scripts/example_test/run_question_categorization.sh
#
# Or interactively:
#   bash scripts/example_test/run_question_categorization.sh
# =============================================================================

set -e

mkdir -p logs

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

# Paths relative to repo root (run from ICARE_score)
INPUT_BASE="test_data/output"
OUTPUT_DIR="test_data/output/question_categorization"
METRICS='["gt_reports_as_ref", "gen_reports_as_ref"]'

if [ ! -d "${INPUT_BASE}/shuffled_ans_choices_data" ]; then
    echo "Error: ${INPUT_BASE}/shuffled_ans_choices_data not found."
    echo "Run scripts/example_test/run_eval.sh first to generate the evaluation output."
    exit 1
fi

echo "============================================="
echo "  ICARE - Question Categorization (Example Test)"
echo "============================================="
echo "Input base:   ${INPUT_BASE}"
echo "Output dir:   ${OUTPUT_DIR}"
echo "============================================="

mkdir -p "${OUTPUT_DIR}"

# Step 1: Create combined questions
echo ""
echo ">>> Step 1: Creating combined MCQ data..."
python src/question_categorization_and_analysis/create_combined_questions.py \
    --metrics "$METRICS" \
    --flat_base_dir "${INPUT_BASE}" \
    --metrics "$METRICS" \
    --output_dir "${OUTPUT_DIR}"

COMBINED_CSV="${OUTPUT_DIR}/combined_mcqa_data.csv"
if [ ! -f "$COMBINED_CSV" ]; then
    echo "Error: combined_mcqa_data.csv was not created."
    exit 1
fi

# Step 2: Embedding and clustering
# Set MEDCPT_MODEL_PATH in .env if compute nodes lack Hugging Face access (pre-download: git clone https://huggingface.co/ncbi/MedCPT-Query-Encoder)
echo ""
echo ">>> Step 2: Question embedding and clustering..."
EMBED_ARGS=(--output_dir "${OUTPUT_DIR}" --combined_data_path "${COMBINED_CSV}")
[ -n "${MEDCPT_MODEL_PATH}" ] && EMBED_ARGS+=(--model_path "${MEDCPT_MODEL_PATH}")
python src/question_categorization_and_analysis/question_embedding_and_clustering.py "${EMBED_ARGS[@]}"

CLUSTERED_CSV="${OUTPUT_DIR}/clustered_questions.csv"
CLUSTER_NAMES="${OUTPUT_DIR}/cluster_names.json"
if [ ! -f "$CLUSTERED_CSV" ] || [ ! -f "$CLUSTER_NAMES" ]; then
    echo "Error: clustering outputs not found."
    exit 1
fi

# Step 3: Cluster analysis
echo ""
echo ">>> Step 3: Cluster analysis..."
ANALYSIS_FOLDER="${OUTPUT_DIR}/analysis"
python src/question_categorization_and_analysis/cluster_analysis.py \
    --clustered_data_path "${CLUSTERED_CSV}" \
    --cluster_names_path "${CLUSTER_NAMES}" \
    --output_dir "${OUTPUT_DIR}" \
    --analysis_folder "${ANALYSIS_FOLDER}"

echo ""
echo "============================================="
echo "  Question Categorization Complete!"
echo "============================================="
echo "Results in: ${OUTPUT_DIR}"
echo "  - combined_mcqa_data.csv"
echo "  - clustered_questions_with_names.csv"
echo "  - cluster_names.json"
echo "  - analysis/all_models_gt_vs_gen_agreement.png"
