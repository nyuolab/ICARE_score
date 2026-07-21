#!/bin/bash
set -eo pipefail

if [ $# -ne 1 ]; then
    echo "Usage: bash scripts/rexval_data/submit_rexval_allques_model.sh <model_key>"
    echo "Examples: llama | opus46 | sonnet46 | gpt54"
    exit 1
fi

MODEL_KEY="$1"
REPO_ROOT="/gpfs/data/oermannlab/users/rd3571/ICARE_score"
BASE_DATA="/gpfs/data/oermannlab/users/rd3571"
EVAL_SEED=123

case "$MODEL_KEY" in
  llama)
    ENV_FILE=""
    SRC_OUTPUT_DIR="${REPO_ROOT}/outputs/rexval/rexval_test_200/eval_seed_${EVAL_SEED}"
    OUTPUT_DIR="${REPO_ROOT}/outputs/rexval/rexval_test_200_allques/eval_seed_${EVAL_SEED}"
    JOB_NAME="rexval_llama_allques"
    ;;
  opus46)
    ENV_FILE="config/envs/rexval_opus_46.env"
    SRC_OUTPUT_DIR="${REPO_ROOT}/outputs/rexval/rexval_test_200_opus46/eval_seed_${EVAL_SEED}"
    OUTPUT_DIR="${REPO_ROOT}/outputs/rexval/rexval_test_200_opus46_allques/eval_seed_${EVAL_SEED}"
    JOB_NAME="rexval_opus46_allques"
    ;;
  sonnet46)
    ENV_FILE="config/envs/rexval_sonnet_46.env"
    SRC_OUTPUT_DIR="${REPO_ROOT}/outputs/rexval/rexval_test_200_sonnet46/eval_seed_${EVAL_SEED}"
    OUTPUT_DIR="${REPO_ROOT}/outputs/rexval/rexval_test_200_sonnet46_allques/eval_seed_${EVAL_SEED}"
    JOB_NAME="rexval_sonnet46_allques"
    ;;
  gpt54)
    ENV_FILE="config/envs/rexval_gpt54.env"
    SRC_OUTPUT_DIR="${REPO_ROOT}/outputs/rexval/rexval_test_200_gpt54/eval_seed_${EVAL_SEED}"
    OUTPUT_DIR="${REPO_ROOT}/outputs/rexval/rexval_test_200_gpt54_allques/eval_seed_${EVAL_SEED}"
    JOB_NAME="rexval_gpt54_allques"
    ;;
  *)
    echo "Unknown model key: $MODEL_KEY"
    exit 1
    ;;
esac

cd "$REPO_ROOT"

EXPORT_VARS="ALL,SRC_OUTPUT_DIR=${SRC_OUTPUT_DIR},OUTPUT_DIR=${OUTPUT_DIR},EVAL_SEED=${EVAL_SEED},DATA_TYPES=shuffled_ans_choices_data,RRGEVAL_BASE_DATA_PATH=${BASE_DATA}"
if [ -n "$ENV_FILE" ]; then
    EXPORT_VARS="${EXPORT_VARS},ENV_FILE=${ENV_FILE}"
fi

sbatch --job-name="${JOB_NAME}" \
  --export="${EXPORT_VARS}" \
  scripts/rexval_data/icare_rexval_allques.sh
