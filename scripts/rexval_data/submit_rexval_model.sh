#!/bin/bash
set -eo pipefail

if [ $# -ne 1 ]; then
    echo "Usage: bash scripts/rexval_data/submit_rexval_model.sh <model_key>"
    echo "Examples: opus46 | sonnet46 | gpt54"
    exit 1
fi

MODEL_KEY="$1"
REPO_ROOT="/gpfs/data/oermannlab/users/rd3571/ICARE_score"
BASE_DATA="/gpfs/data/oermannlab/users/rd3571"
EVAL_SEED=123
NUM_QUESTIONS=60

case "$MODEL_KEY" in
  opus46)
    ENV_FILE="config/envs/rexval_opus_46.env"
    OUTPUT_DIR="${REPO_ROOT}/outputs/rexval/rexval_test_200_opus46/eval_seed_${EVAL_SEED}"
    JOB_NAME="rexval_opus46"
    ;;
  sonnet46)
    ENV_FILE="config/envs/rexval_sonnet_46.env"
    OUTPUT_DIR="${REPO_ROOT}/outputs/rexval/rexval_test_200_sonnet46/eval_seed_${EVAL_SEED}"
    JOB_NAME="rexval_sonnet46"
    ;;
  gpt54)
    ENV_FILE="config/envs/rexval_gpt54.env"
    OUTPUT_DIR="${REPO_ROOT}/outputs/rexval/rexval_test_200_gpt54/eval_seed_${EVAL_SEED}"
    JOB_NAME="rexval_gpt54"
    ;;
  *)
    echo "Unknown model key: $MODEL_KEY"
    exit 1
    ;;
esac

cd "$REPO_ROOT"

sbatch --job-name="${JOB_NAME}" \
  --export=ALL,ENV_FILE="${ENV_FILE}",OUTPUT_DIR="${OUTPUT_DIR}",EVAL_SEED="${EVAL_SEED}",NUM_QUESTIONS="${NUM_QUESTIONS}",RRGEVAL_BASE_DATA_PATH="${BASE_DATA}" \
  scripts/rexval_data/icare_rexval.sh