#!/bin/bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

DATASET="${DATASET:-JosephZ/vg150_val_sgg_prompt}"
DATASET_TYPE="${DATASET_TYPE:-auto}"
DATASET_SPLIT="${DATASET_SPLIT:-train}"
MODEL_NAME="${MODEL_NAME:?Set MODEL_NAME to the trained checkpoint path or model ID}"
OUTPUT_DIR="${OUTPUT_DIR:-./output_results}"

BATCH_SIZE=16
TENSOR_PARALLEL_SIZE=1
MAX_MODEL_LEN=4096
MAX_NEW_TOKENS=2048
TEMPERATURE=0.6
TOP_K=50
TOP_P=0.95
REPETITION_PENALTY=1.05
GPU_MEMORY_UTILIZATION=0.85
DTYPE="${DTYPE:-bfloat16}"
SEED="${SEED:-42}"
MIN_PIXELS="${MIN_PIXELS:-3136}"
MAX_PIXELS="${MAX_PIXELS:-802816}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-128}"
NUM_WORKERS="${NUM_WORKERS:-0}"

echo "Starting scene graph inference..."
echo "Model: ${MODEL_NAME}"
echo "Dataset: ${DATASET}"
echo "Output directory: ${OUTPUT_DIR}"
echo "Tensor parallel size: ${TENSOR_PARALLEL_SIZE}"
echo "Batch size: ${BATCH_SIZE}"
echo "Sampling: max_new_tokens=${MAX_NEW_TOKENS}, temperature=${TEMPERATURE}, top_k=${TOP_K}, top_p=${TOP_P}"

python src/inference/sgg_inference.py \
    --dataset "${DATASET}" \
    --dataset_type "${DATASET_TYPE}" \
    --dataset_split "${DATASET_SPLIT}" \
    --model "${MODEL_NAME}" \
    --output_dir "${OUTPUT_DIR}" \
    --max_model_len "${MAX_MODEL_LEN}" \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    --temperature "${TEMPERATURE}" \
    --top_k "${TOP_K}" \
    --top_p "${TOP_P}" \
    --repetition_penalty "${REPETITION_PENALTY}" \
    --batch_size "${BATCH_SIZE}" \
    --tensor_parallel_size "${TENSOR_PARALLEL_SIZE}" \
    --gpu_memory_utilization "${GPU_MEMORY_UTILIZATION}" \
    --dtype "${DTYPE}" \
    --seed "${SEED}" \
    --min_pixels "${MIN_PIXELS}" \
    --max_pixels "${MAX_PIXELS}" \
    --max_num_seqs "${MAX_NUM_SEQS}" \
    --num_workers "${NUM_WORKERS}" \
    "$@"
