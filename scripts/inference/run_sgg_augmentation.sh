#!/bin/bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

DATASET_TYPE="${DATASET_TYPE:-auto}"
DATASET_PATH="${DATASET_PATH:-JosephZ/vg150_train_sgg_prompt}"
DATASET_SPLIT="${DATASET_SPLIT:-train}"
MODEL_NAME="${MODEL_NAME:-models/qwen2-vl-2b-sft-vg/checkpoint-200}"
OUTPUT_FILE="${OUTPUT_FILE:-./augmentation_results/relation_augmentation.json}"
TEMP_OUTPUT_FILE="./augmentation_results/temp_relationship_augmentation.json"
STATS_FILE="./augmentation_results/processing_statistics.json"

BATCH_SIZE=16
TENSOR_PARALLEL_SIZE=4
MAX_MODEL_LEN=8192
MAX_NEW_TOKENS=1024
TEMPERATURE=0.2
TOP_K=50
TOP_P=0.9
REPETITION_PENALTY=1.0
GPU_MEMORY_UTILIZATION=0.85
SAVE_INTERVAL=2
DTYPE="${DTYPE:-bfloat16}"
SEED="${SEED:-42}"
MIN_PIXELS="${MIN_PIXELS:-3136}"
MAX_PIXELS="${MAX_PIXELS:-802816}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-128}"
NUM_WORKERS="${NUM_WORKERS:-0}"

echo "Starting relationship augmentation..."
echo "Model: ${MODEL_NAME}"
echo "Dataset: ${DATASET_PATH}"
echo "Dataset type: ${DATASET_TYPE}"
echo "Output file: ${OUTPUT_FILE}"
echo "Tensor parallel size: ${TENSOR_PARALLEL_SIZE}"
echo "Batch size: ${BATCH_SIZE}"
echo "Sampling: max_new_tokens=${MAX_NEW_TOKENS}, temperature=${TEMPERATURE}, top_k=${TOP_K}, top_p=${TOP_P}"

python src/data_augmentation/relationship_augmentation.py \
    --dataset_type "${DATASET_TYPE}" \
    --dataset_path "${DATASET_PATH}" \
    --dataset_split "${DATASET_SPLIT}" \
    --output_file "${OUTPUT_FILE}" \
    --temp_output_file "${TEMP_OUTPUT_FILE}" \
    --stats_file "${STATS_FILE}" \
    --model_name "${MODEL_NAME}" \
    --max_model_len "${MAX_MODEL_LEN}" \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    --temperature "${TEMPERATURE}" \
    --top_k "${TOP_K}" \
    --top_p "${TOP_P}" \
    --repetition_penalty "${REPETITION_PENALTY}" \
    --batch_size "${BATCH_SIZE}" \
    --tensor_parallel_size "${TENSOR_PARALLEL_SIZE}" \
    --gpu_memory_utilization "${GPU_MEMORY_UTILIZATION}" \
    --save_interval "${SAVE_INTERVAL}" \
    --dtype "${DTYPE}" \
    --seed "${SEED}" \
    --min_pixels "${MIN_PIXELS}" \
    --max_pixels "${MAX_PIXELS}" \
    --max_num_seqs "${MAX_NUM_SEQS}" \
    --num_workers "${NUM_WORKERS}" \
    "$@"
