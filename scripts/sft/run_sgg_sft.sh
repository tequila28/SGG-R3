#!/bin/bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export TORCH_DISTRIBUTED_DEBUG="INFO"
export NCCL_DEBUG="INFO"

GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
NUM_NODES=1
NODE_RANK=0

MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-Qwen/Qwen2.5-VL-3B-Instruct}"
DATASET_NAME="${DATASET_NAME:-JosephZ/psg_train_sg}"
DATASET_TYPE="${DATASET_TYPE:-auto}"
DATASET_SPLIT="${DATASET_SPLIT:-train}"
USE_AUGMENTED_DATA="${USE_AUGMENTED_DATA:-true}"
PSG_AUGMENTED_DATA_PATH="tequila3009/psg_augmentation_data"
VG_AUGMENTED_DATA_PATH="tequila3009/vg_augmentation_data"
RUN_NAME="Qwen2.5-VL-3B_sft"
OUTPUT_DIR="${OUTPUT_DIR:-models/qwen2.5-vl-3b-sft-psg}"
FSDP_CONFIG="local_scripts/fsdp_config.json"

LEARNING_RATE="1e-5"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-16}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
WARMUP_RATIO=0.05
MAX_GRAD_NORM=0.3
LOGGING_STEPS=1
NUM_TRAIN_EPOCHS=1
SAVE_STEPS=1000
SEED=42
MAX_LENGTH="${MAX_LENGTH:-0}" # 0 preserves the previous no-truncation behavior.
MIN_PIXELS="${MIN_PIXELS:-3136}"
MAX_PIXELS="${MAX_PIXELS:-802816}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"
SAVE_ONLY_MODEL="${SAVE_ONLY_MODEL:-true}"
RDZV_PORT="${RDZV_PORT:-29500}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"
EXTRA_ARGS=()
if [[ -n "${RESUME_FROM_CHECKPOINT}" ]]; then
    EXTRA_ARGS+=(--resume_from_checkpoint "${RESUME_FROM_CHECKPOINT}")
fi

if [[ "${USE_AUGMENTED_DATA}" == "true" ]]; then
    if [[ "${DATASET_TYPE}" == "psg" || ( "${DATASET_TYPE}" == "auto" && "${DATASET_NAME,,}" == *"psg"* ) ]]; then
        AUGMENTED_DATA_PATH="${PSG_AUGMENTED_DATA_PATH}"
    else
        AUGMENTED_DATA_PATH="${VG_AUGMENTED_DATA_PATH}"
    fi
else
    AUGMENTED_DATA_PATH=""
fi

torchrun --nnodes "${NUM_NODES}" \
    --nproc_per_node "${GPUS_PER_NODE}" \
    --node_rank "${NODE_RANK}" \
    --master_port "${RDZV_PORT}" \
    src/sft/sft_train.py \
    --model_name_or_path "${MODEL_NAME_OR_PATH}" \
    --dataset_name "${DATASET_NAME}" \
    --dataset_type "${DATASET_TYPE}" \
    --dataset_train_split "${DATASET_SPLIT}" \
    --learning_rate "${LEARNING_RATE}" \
    --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}" \
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
    --warmup_ratio "${WARMUP_RATIO}" \
    --max_grad_norm "${MAX_GRAD_NORM}" \
    --logging_steps "${LOGGING_STEPS}" \
    --bf16 true \
    --tf32 true \
    --report_to none \
    --attn_implementation "${ATTN_IMPLEMENTATION}" \
    --gradient_checkpointing false \
    --num_train_epochs "${NUM_TRAIN_EPOCHS}" \
    --run_name "${RUN_NAME}" \
    --save_steps "${SAVE_STEPS}" \
    --save_only_model "${SAVE_ONLY_MODEL}" \
    --dtype bfloat16 \
    --fsdp "full_shard auto_wrap" \
    --fsdp_config "${FSDP_CONFIG}" \
    --output_dir "${OUTPUT_DIR}" \
    --seed "${SEED}" \
    --max_length "${MAX_LENGTH}" \
    --min_pixels "${MIN_PIXELS}" \
    --max_pixels "${MAX_PIXELS}" \
    --loss_type nll \
    --packing false \
    --assistant_only_loss false \
    --completion_only_loss false \
    --lr_scheduler_type linear \
    --weight_decay 0.0 \
    --optim adamw_torch_fused \
    --dataloader_num_workers 0 \
    --augmented_data_path "${AUGMENTED_DATA_PATH}" \
    "${EXTRA_ARGS[@]}" "$@"
