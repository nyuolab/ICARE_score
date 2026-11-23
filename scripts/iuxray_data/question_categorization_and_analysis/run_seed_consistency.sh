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
# Runs BOTH model-seed modes (all, then 1). Extra args (e.g. --quick) are forwarded.
# Works with sbatch (from repo root) or bash inside srun. Never GPUs / login / sp-0010.

set -eo pipefail

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
  ORIG_DIR="$SLURM_SUBMIT_DIR"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  ORIG_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
fi
cd "$ORIG_DIR"
mkdir -p results/revision_r2/exp5/logs

if [ -f ".env" ]; then
    echo "Loading environment variables from .env file..."
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

echo "Host=$(hostname) pwd=$(pwd) python=$PY_BIN submit=${SLURM_SUBMIT_DIR:-none}"
"$PY_BIN" -c "import sklearn; print('sklearn', sklearn.__version__)"

echo "=== Exp 5 mode=all (primary) ==="
"$PY_BIN" src/question_categorization_and_analysis/seed_consistency.py --model-seeds all "$@"
echo "=== Exp 5 mode=1 ==="
"$PY_BIN" src/question_categorization_and_analysis/seed_consistency.py --model-seeds 1 "$@"
echo "Exp 5 both modes done."
