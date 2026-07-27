# Example model override (same KEY=VALUE shape as BigPurple config/envs/*.env).
# Copy and fill in real values, then pass:
#   ENV_FILE=config/envs/example_model.sh bash scripts/example_test/run_eval_final_without_orig.sh
#
# Loaded after .env so these override the base model settings.

RRGEVAL_API_URL=https://your-endpoint/chat/completions
RRGEVAL_API_KEY=your_api_key_here
RRGEVAL_API_AUTH_HEADER_TYPE=api-key
RRGEVAL_MODEL_NAME=your-model-name
