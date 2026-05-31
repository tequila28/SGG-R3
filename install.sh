#!/bin/bash
set -euo pipefail

ENV_FILE="environment_SGG.yml"
ENV_NAME="SGG"
REQUIREMENTS_FILE="requirements.txt"
FLASH_ATTN_VERSION="2.8.1"

if ! command -v conda >/dev/null 2>&1; then
    echo "conda was not found. Please install Miniconda or Anaconda first."
    exit 1
fi

if [ ! -f "${ENV_FILE}" ]; then
    echo "Missing ${ENV_FILE}. Run this script from the repository root."
    exit 1
fi

if [ ! -f "${REQUIREMENTS_FILE}" ]; then
    echo "Missing ${REQUIREMENTS_FILE}. Run this script from the repository root."
    exit 1
fi

if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    conda env update -n "${ENV_NAME}" -f "${ENV_FILE}" --prune
else
    conda env create -f "${ENV_FILE}"
fi

conda run -n "${ENV_NAME}" python -m pip install --upgrade pip setuptools wheel
conda run -n "${ENV_NAME}" python -m pip install -r "${REQUIREMENTS_FILE}"
conda run -n "${ENV_NAME}" python -m pip install "flash-attn==${FLASH_ATTN_VERSION}" --no-build-isolation

echo "Environment is ready."
echo "Activate it with: conda activate ${ENV_NAME}"
