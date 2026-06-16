#!/bin/bash
#SBATCH --job-name=alignscore_radpref
#SBATCH --partition=a100_short
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=rd3571@nyu.edu
#SBATCH --output=/gpfs/data/oermannlab/users/rd3571/ICARE_score/outputs/radpref/baselines/alignscore/logs/%x-%j.out
#SBATCH --error=/gpfs/data/oermannlab/users/rd3571/ICARE_score/outputs/radpref/baselines/alignscore/logs/%x-%j.err

# =============================================================================
# Run AlignScore on the RadPref preference dataset (100 cases x 2 candidates).
#
# Usage:
#   cd /gpfs/data/oermannlab/users/rd3571/ICARE_score
#   sbatch scripts/radpref_data/alignscore_eval_baselines.sh
# =============================================================================

set -eo pipefail

SCRIPT_ROOT="/gpfs/data/oermannlab/users/rd3571/ICARE_score"
CKPT_PATH="/gpfs/data/oermannlab/users/rd3571/AlignScore-base.ckpt"
RESULTS_DIR="${SCRIPT_ROOT}/outputs/radpref/baselines"
ALIGNSCORE_DIR="${RESULTS_DIR}/alignscore"
LOGS_DIR="${ALIGNSCORE_DIR}/logs"
RADPREF_CSV="/gpfs/data/oermannlab/users/rd3571/cxr_report_datasets/radpref/radpref_icare.csv"
PREPARED_CSV="${RESULTS_DIR}/radpref_prepared.csv"

mkdir -p "${LOGS_DIR}"

echo "--- Activating Conda Environment ---"
ORIG_DIR=$(pwd)
source ~/.bashrc
cd "$ORIG_DIR"
module load cuda/11.8
conda activate alignscore
export PYTHONNOUSERSITE=1
if ! python -c "from alignscore import AlignScore" 2>/dev/null; then
    echo "Error: alignscore env missing AlignScore. Active python: $(which python)"
    exit 1
fi
echo "Using python: $(which python)"

cd "${SCRIPT_ROOT}" || { echo "Failed to change to ${SCRIPT_ROOT}"; exit 1; }

echo "Job ID     : ${SLURM_JOB_ID}"
echo "Node       : ${SLURM_JOB_NODELIST}"
echo "Start      : $(date)"

echo ""
echo ">>> Step 1: Preparing RadPref CSV..."
python - <<'PYEOF'
import pandas as pd

src = "/gpfs/data/oermannlab/users/rd3571/cxr_report_datasets/radpref/radpref_icare.csv"
dst = "/gpfs/data/oermannlab/users/rd3571/ICARE_score/outputs/radpref/baselines/radpref_prepared.csv"

df = pd.read_csv(src)
df.insert(0, "id", range(len(df)))
df.to_csv(dst, index=False)
print(f"Prepared CSV written to {dst}  ({len(df)} rows)")
PYEOF
echo ">>> Step 1 complete."

echo ""
echo ">>> Step 2: Running AlignScore..."
python src/run_alignscore_eval.py \
    --input-csv  "${PREPARED_CSV}" \
    --output-dir "${ALIGNSCORE_DIR}" \
    --ckpt-path  "${CKPT_PATH}" \
    --model      roberta-base \
    --batch-size 32 \
    --device     cuda:0 \
    --eval-mode  nli_sp
echo ">>> Step 2 complete."

echo ""
echo "============================================="
echo "  RadPref AlignScore complete."
echo "  Results: ${ALIGNSCORE_DIR}"
echo "  Note: rows 0-99 = C1 candidates,"
echo "        rows 100-199 = C2 candidates."
echo "============================================="
echo "End: $(date)"
