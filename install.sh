#!/bin/bash
set -euo pipefail

ENV_NAME="SGG"
REQUIREMENTS_FILE="requirements.txt"
FLASH_ATTN_VERSION="2.8.1"

if ! command -v conda >/dev/null 2>&1; then
    echo "conda was not found. Please install Miniconda or Anaconda first."
    exit 1
fi

if [ ! -f "${REQUIREMENTS_FILE}" ]; then
    echo "Missing ${REQUIREMENTS_FILE}. Run this script from the repository root."
    exit 1
fi

if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    echo "Conda environment ${ENV_NAME} already exists."
else
    conda create -n "${ENV_NAME}" python=3.10 -y
fi

conda run -n "${ENV_NAME}" python -m pip install --upgrade pip setuptools wheel
conda run -n "${ENV_NAME}" python -m pip install -r "${REQUIREMENTS_FILE}"
conda run -n "${ENV_NAME}" python -m pip install "flash-attn==${FLASH_ATTN_VERSION}" --no-build-isolation

echo "Environment is ready."
echo "Activate it with: conda activate ${ENV_NAME}"
