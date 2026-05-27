#!/bin/bash
# One-time migration: copy existing per-run results from the old shared layout
#   <OLD_BASE>/MCQ_gen_data_our_eval_seed<ES>/IU_xray/<model>/seed_<MS>/
# into the new local layout
#   ./outputs/IU_xray/<model>/model_seed_<MS>/eval_seed_<ES>/
#
# Full grid: all model seeds 1-5 x eval seeds x 3 models.
# The old shared data is left untouched as a backup. Safe to re-run (rsync idempotent).
set -euo pipefail

# Load RRGEVAL_BASE_DATA_PATH from .env if present
if [ -f ".env" ]; then
    export $(grep -v '^#' .env | xargs)
fi

OLD_BASE="${RRGEVAL_BASE_DATA_PATH}/RRG_evaluation/MCQ_generation"
NEW_BASE="$(pwd)/outputs/IU_xray"

MODEL_SEEDS=(1 2 3 4 5)
EVAL_SEEDS=(123 456 789 202 101)
MODELS=(maira-2 mimic-cxr-findings-baseline chexpert-mimic-cxr-findings-baseline)

echo "OLD_BASE: ${OLD_BASE}"
echo "NEW_BASE: ${NEW_BASE}"
echo

copied=0
missing=0
for model in "${MODELS[@]}"; do
    for ms in "${MODEL_SEEDS[@]}"; do
        for es in "${EVAL_SEEDS[@]}"; do
            src="${OLD_BASE}/MCQ_gen_data_our_eval_seed${es}/IU_xray/${model}/seed_${ms}"
            dst="${NEW_BASE}/${model}/model_seed_${ms}/eval_seed_${es}"
            if [ -d "$src" ]; then
                mkdir -p "$dst"
                rsync -a "$src/" "$dst/"
                copied=$((copied+1))
            else
                echo "  [MISSING] ${src}"
                missing=$((missing+1))
            fi
        done
    done
    echo "  [OK] ${model}: model_seed_{1..5} x eval_seed_{${EVAL_SEEDS[*]}}"
done

# Remove redundant bare eval_seed_<ES> dirs left by the earlier (model_seed-less)
# migration; their data is now under model_seed_1/eval_seed_<ES>.
echo
echo "Cleaning up old model_seed-less dirs:"
for model in "${MODELS[@]}"; do
    for es in "${EVAL_SEEDS[@]}"; do
        old="${NEW_BASE}/${model}/eval_seed_${es}"
        if [ -d "$old" ]; then
            rm -rf "$old"
            echo "  removed ${model}/eval_seed_${es}"
        fi
    done
done

echo
echo "Migration done: ${copied} copied, ${missing} missing."
