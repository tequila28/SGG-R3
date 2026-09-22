#!/bin/bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

ENV_NAME="${ENV_NAME:-SGG}"
REQUIREMENTS_FILE="requirements.txt"
INSTALL_FLASH_ATTN="${INSTALL_FLASH_ATTN:-false}"
FLASH_ATTN_VERSION="${FLASH_ATTN_VERSION:-2.8.3}"

if [[ "$(uname -s)" != "Linux" ]]; then
    echo "This environment targets Linux with NVIDIA CUDA GPUs."
    exit 1
fi

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

conda run -n "${ENV_NAME}" python -m pip install --upgrade pip 'setuptools>=77.0.3,<81' wheel
conda run -n "${ENV_NAME}" python -m pip install -r "${REQUIREMENTS_FILE}"
if [[ "${INSTALL_FLASH_ATTN}" == "true" ]]; then
    conda run -n "${ENV_NAME}" python -m pip install "flash-attn==${FLASH_ATTN_VERSION}" --no-build-isolation
fi
conda run -n "${ENV_NAME}" python -m pip check

echo "Environment is ready."
echo "Activate it with: conda activate ${ENV_NAME}"
