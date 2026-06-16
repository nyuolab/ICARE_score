#!/bin/bash
# =============================================================================
# Launch a local vLLM OpenAI-compatible server for ICARE evaluation.
#
# This lets you run ICARE without any private API keys. The server exposes
# an OpenAI-compatible endpoint that the ICARE pipeline talks to.
#
# Requirements
# ------------
# - NVIDIA GPU(s) with enough VRAM:
#     Llama-3.3-70B (full precision): 2× A100-80GB or 2× H100-80GB
#     Llama-3.3-70B (4-bit AWQ):      1× A100-40GB (see QUANTIZATION below)
# - HuggingFace access to the model:
#     https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct
#     Log in once: huggingface-cli login
# - vLLM installed:
#     pip install "vllm>=0.4.0"
#
# Quick start (two-terminal workflow)
# ------------------------------------
# Terminal 1 — start the server:
#   bash scripts/local_llm/launch_vllm_server.sh
#
# Terminal 2 — run ICARE once the server is ready:
#   cp .env.local_example .env
#   bash scripts/example_test/run_eval.sh
#   # or with plain python: see README.md "Local LLM Setup" section
# =============================================================================

# --------------------------------------------------------------------------
# Configuration — override with environment variables if needed
# --------------------------------------------------------------------------
MODEL=${ICARE_LOCAL_MODEL:-"meta-llama/Llama-3.3-70B-Instruct"}
PORT=${ICARE_VLLM_PORT:-8000}
HOST=${ICARE_VLLM_HOST:-"0.0.0.0"}
TENSOR_PARALLEL=${ICARE_TENSOR_PARALLEL:-2}   # number of GPUs for tensor parallelism
API_KEY="icare-local"                          # matches .env.local_example

# For a single GPU with limited VRAM, enable 4-bit AWQ quantization:
#   ICARE_LOCAL_MODEL="hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4" \
#   ICARE_TENSOR_PARALLEL=1 \
#   bash scripts/local_llm/launch_vllm_server.sh
# (AWQ models run ~70B in ~40GB VRAM)
QUANTIZATION=${ICARE_VLLM_QUANTIZATION:-""}

# --------------------------------------------------------------------------
echo "============================================="
echo "  ICARE — Local vLLM Server"
echo "============================================="
echo "  Model:            ${MODEL}"
echo "  Host:             ${HOST}:${PORT}"
echo "  Tensor parallel:  ${TENSOR_PARALLEL}"
[ -n "${QUANTIZATION}" ] && echo "  Quantization:     ${QUANTIZATION}"
echo "  API key:          ${API_KEY}  (set in .env.local_example)"
echo "============================================="
echo ""
echo "Once 'Application startup complete' appears, open a second terminal and run:"
echo "  cp .env.local_example .env"
echo "  bash scripts/example_test/run_eval.sh"
echo ""

EXTRA_ARGS=()
[ -n "${QUANTIZATION}" ] && EXTRA_ARGS+=(--quantization "${QUANTIZATION}")

python -m vllm.entrypoints.openai.api_server \
    --model "${MODEL}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --api-key "${API_KEY}" \
    --tensor-parallel-size "${TENSOR_PARALLEL}" \
    --trust-remote-code \
    "${EXTRA_ARGS[@]}"
