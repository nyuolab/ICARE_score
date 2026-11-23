#!/bin/bash
#SBATCH --job-name=exp5_seed_consistency
#SBATCH --partition=cpu_short
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH -o results/revision_r2/exp5/logs/%x_%j.out
#SBATCH -e results/revision_r2/exp5/logs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=rd3571@nyu.edu
#
# Works with sbatch or: bash this_script.sh [--quick]
# Prefer -p oermannlab if cpu_short is busy. Never request GPUs; never login/sp-0010.

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ORIG_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "$ORIG_DIR"
mkdir -p results/revision_r2/exp5/logs

if [ -f ".env" ]; then
    echo "Loading environment variables from .env file..."
    # Safe load: KEY=VALUE lines only; ignore blanks/comments; do not use set -u here
    while IFS= read -r line || [ -n "$line" ]; do
        case "$line" in
            ''|\#*) continue ;;
        esac
        if [[ "$line" == *=* ]]; then
            export "$line" 2>/dev/null || true
        fi
    done < .env
fi

PY_BIN=/gpfs/data/oermannlab/users/rd3571/.conda/envs/rrg-eval-clean/bin/python
export OPENBLAS_NUM_THREADS=8
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=

echo "Host=$(hostname) pwd=$(pwd) python=$PY_BIN"
"$PY_BIN" -c "import sklearn; print('sklearn', sklearn.__version__)"
"$PY_BIN" src/question_categorization_and_analysis/seed_consistency.py "$@"
