#!/bin/bash
# =============================================================================
# One-time setup: create the 'crimson' conda environment and install CRIMSON.
#
# Run ONCE from the login node before submitting crimson_baselines_rexval.sh:
#   bash scripts/rexval_data/setup_crimson_env.sh
# =============================================================================

set -eo pipefail

CRIMSON_DIR="/gpfs/data/oermannlab/users/rd3571/CRIMSON"
ENV_NAME="crimson"

source ~/.bashrc

echo "============================================="
echo "  Setting up '${ENV_NAME}' conda environment"
echo "============================================="

# Create env with Python 3.12 (required by CRIMSON)
if conda env list | grep -qE "^${ENV_NAME}\s"; then
    echo "Environment '${ENV_NAME}' already exists — skipping creation."
else
    echo "Creating environment '${ENV_NAME}' with Python 3.12..."
    conda create -n "${ENV_NAME}" python=3.12 -y
fi

conda activate "${ENV_NAME}"
# Clear PYTHONPATH to prevent base conda's Python 3.8 site-packages
# from contaminating the new Python 3.12 environment.
unset PYTHONPATH
/gpfs/data/oermannlab/users/rd3571/.conda/envs/${ENV_NAME}/bin/python -m pip install --upgrade pip

echo ""
echo ">>> Installing CRIMSON from source..."
cd "${CRIMSON_DIR}"
# Non-editable install: copies the package into site-packages so Python
# finds it correctly even when the repo root is on sys.path.
/gpfs/data/oermannlab/users/rd3571/.conda/envs/${ENV_NAME}/bin/python -m pip install .

echo ""
echo "============================================="
echo "  Setup complete. Environment: ${ENV_NAME}"
echo "  Test it with:"
echo "    conda activate ${ENV_NAME}"
echo "    python -c \"from CRIMSON import CRIMSONScore; print('OK')\""
echo "============================================="
