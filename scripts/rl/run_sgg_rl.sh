#!/bin/bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

VLLM_CUDA_DEVICES="${VLLM_CUDA_DEVICES:-6,7}"
TRAIN_CUDA_DEVICES="${TRAIN_CUDA_DEVICES:-0,1,2,3,4,5}"

export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

GROUP_SIZE=8
MODEL_PATH="${MODEL_PATH:-models/qwen2.5-vl-3b-sft-psg}"
DATA_PATH="${DATA_PATH:-JosephZ/psg_train_sg}"
DATASET_TYPE="${DATASET_TYPE:-psg}"
DATASET_SPLIT="${DATASET_SPLIT:-train}"
RUN_NAME="${RUN_NAME:-qwen2.5vl-3b-gspo-g${GROUP_SIZE}-${DATASET_TYPE}}"
OUTPUT_DIR="${OUTPUT_DIR:-models/${RUN_NAME}}"

VLLM_DTYPE="${VLLM_DTYPE:-bfloat16}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-5120}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.85}"
VLLM_TENSOR_PARALLEL_SIZE="${VLLM_TENSOR_PARALLEL_SIZE:-2}"
PORT_BASE="${PORT_BASE:-8020}"
VLLM_GROUP_PORT="${VLLM_GROUP_PORT:-51216}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-128}"
VLLM_MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-8192}"
VLLM_ENFORCE_EAGER="${VLLM_ENFORCE_EAGER:-false}"

MIN_PIXELS="${MIN_PIXELS:-3136}"
MAX_PIXELS="${MAX_PIXELS:-401408}"
NUM_NODES=1
TRAIN_PROCESSES="${TRAIN_PROCESSES:-6}"
HEAD_NODE_IP="127.0.0.1"
RDZV_PORT="${RDZV_PORT:-29500}"

PROMPT_LENGTH_LIMIT="${PROMPT_LENGTH_LIMIT:-4096}"
MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-1024}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-8}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-2}"
LEARNING_RATE="3e-7"
LOGGING_STEPS=1
VLLM_SERVER_TIMEOUT=600
TEMPERATURE=1
TOP_P=0.9
TOP_K=50
NUM_TRAIN_EPOCHS=1
SAVE_STEPS=4656
NUM_GENERATIONS="${NUM_GENERATIONS:-${GROUP_SIZE}}"
NUM_ITERATIONS=1
BETA=0.0
SEED=42
LOSS_TYPE="dapo"
IMPORTANCE_SAMPLING_LEVEL="sequence"
DEEPSPEED_CONFIG="./local_scripts/zero3.json"
REWARD_CONFIG_PATH="${REWARD_CONFIG_PATH:-src/rl/configs/reward_config.yaml}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"
SAVE_ONLY_MODEL="${SAVE_ONLY_MODEL:-true}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"
EXTRA_ARGS=()
if [[ -n "${RESUME_FROM_CHECKPOINT}" ]]; then
    EXTRA_ARGS+=(--resume_from_checkpoint "${RESUME_FROM_CHECKPOINT}")
fi

if (( VLLM_MAX_MODEL_LEN < PROMPT_LENGTH_LIMIT + MAX_COMPLETION_LENGTH )); then
    echo "VLLM_MAX_MODEL_LEN must cover PROMPT_LENGTH_LIMIT + MAX_COMPLETION_LENGTH." >&2
    exit 1
fi
if (( NUM_GENERATIONS < 2 || (TRAIN_PROCESSES * PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS) % NUM_GENERATIONS != 0 )); then
    echo "Generation batch size must be divisible by NUM_GENERATIONS (>=2)." >&2
    exit 1
fi
IFS=',' read -ra TRAIN_GPUS <<< "${TRAIN_CUDA_DEVICES}"
IFS=',' read -ra VLLM_GPUS <<< "${VLLM_CUDA_DEVICES}"
if (( ${#TRAIN_GPUS[@]} != TRAIN_PROCESSES || ${#VLLM_GPUS[@]} != VLLM_TENSOR_PARALLEL_SIZE )); then
    echo "GPU lists must match TRAIN_PROCESSES and VLLM_TENSOR_PARALLEL_SIZE." >&2
    exit 1
fi
for gpu in "${TRAIN_GPUS[@]}"; do
    if [[ ",${VLLM_CUDA_DEVICES}," == *",${gpu},"* ]]; then
        echo "Training and vLLM GPUs must be disjoint." >&2
        exit 1
    fi
done
command -v curl >/dev/null || { echo "curl is required for the server health check." >&2; exit 1; }
command -v setsid >/dev/null || { echo "setsid (util-linux) is required." >&2; exit 1; }

mkdir -p "${OUTPUT_DIR}"

SERVER_PID=""
cleanup() {
    if [[ -n "${SERVER_PID}" ]]; then
        kill -TERM -- "-${SERVER_PID}" 2>/dev/null || true
        wait "${SERVER_PID}" 2>/dev/null || true
    fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
if curl --noproxy '*' -fsS --max-time 2 "http://${HEAD_NODE_IP}:${PORT_BASE}/health/" >/dev/null 2>&1; then
    echo "A server is already listening on PORT_BASE=${PORT_BASE}; choose another port." >&2
    exit 1
fi
echo "Starting vLLM server on CUDA ${VLLM_CUDA_DEVICES}..."
CUDA_VISIBLE_DEVICES="${VLLM_CUDA_DEVICES}" setsid python -m src.rl.vllm_server \
    --model "${MODEL_PATH}" \
    --dtype "${VLLM_DTYPE}" \
    --max_model_len "${VLLM_MAX_MODEL_LEN}" \
    --gpu_memory_utilization "${VLLM_GPU_MEMORY_UTILIZATION}" \
    --enable_prefix_caching true \
    --tensor_parallel_size "${VLLM_TENSOR_PARALLEL_SIZE}" \
    --min_pixels "${MIN_PIXELS}" \
    --max_pixels "${MAX_PIXELS}" \
    --seed "${SEED}" \
    --max_num_seqs "${VLLM_MAX_NUM_SEQS}" \
    --max_num_batched_tokens "${VLLM_MAX_NUM_BATCHED_TOKENS}" \
    --enforce_eager "${VLLM_ENFORCE_EAGER}" \
    --host "${HEAD_NODE_IP}" \
    --port "${PORT_BASE}" > "${OUTPUT_DIR}/vllm_server.log" 2>&1 &

SERVER_PID=$!
DEADLINE=$((SECONDS + VLLM_SERVER_TIMEOUT))
until curl --noproxy '*' -fsS --max-time 2 "http://${HEAD_NODE_IP}:${PORT_BASE}/health/" >/dev/null 2>&1; do
    if ! kill -0 "${SERVER_PID}" 2>/dev/null || (( SECONDS >= DEADLINE )); then
        echo "vLLM failed to become ready; see ${OUTPUT_DIR}/vllm_server.log" >&2
        tail -n 50 "${OUTPUT_DIR}/vllm_server.log" >&2
        exit 1
    fi
    sleep 2
done

echo "Starting RL training on CUDA ${TRAIN_CUDA_DEVICES}..."
CUDA_VISIBLE_DEVICES="${TRAIN_CUDA_DEVICES}" torchrun \
    --nnodes "${NUM_NODES}" \
    --nproc_per_node "${TRAIN_PROCESSES}" \
    --node_rank 0 \
    --rdzv_id grpo_run \
    --rdzv_backend c10d \
    --rdzv_endpoint "${HEAD_NODE_IP}:${RDZV_PORT}" \
    src/rl/rl_train.py \
    --output_dir "${OUTPUT_DIR}" \
    --model_name_or_path "${MODEL_PATH}" \
    --dataset_name "${DATA_PATH}" \
    --dataset_type "${DATASET_TYPE}" \
    --dataset_train_split "${DATASET_SPLIT}" \
    --reward_config_path "${REWARD_CONFIG_PATH}" \
    --deepspeed "${DEEPSPEED_CONFIG}" \
    --prompt_length_limit "${PROMPT_LENGTH_LIMIT}" \
    --max_completion_length "${MAX_COMPLETION_LENGTH}" \
    --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}" \
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
    --learning_rate "${LEARNING_RATE}" \
    --logging_steps "${LOGGING_STEPS}" \
    --use_vllm true \
    --vllm_mode server \
    --vllm_server_host "${HEAD_NODE_IP}" \
    --vllm_server_port "${PORT_BASE}" \
    --vllm_group_port "${VLLM_GROUP_PORT}" \
    --vllm_server_timeout "${VLLM_SERVER_TIMEOUT}" \
    --bf16 true \
    --dtype "${VLLM_DTYPE}" \
    --attn_implementation "${ATTN_IMPLEMENTATION}" \
    --report_to none \
    --gradient_checkpointing true \
    --max_pixels "${MAX_PIXELS}" \
    --min_pixels "${MIN_PIXELS}" \
    --temperature "${TEMPERATURE}" \
    --top_p "${TOP_P}" \
    --top_k "${TOP_K}" \
    --num_train_epochs "${NUM_TRAIN_EPOCHS}" \
    --run_name "${RUN_NAME}" \
    --save_steps "${SAVE_STEPS}" \
    --num_generations "${NUM_GENERATIONS}" \
    --num_iterations "${NUM_ITERATIONS}" \
    --beta "${BETA}" \
    --save_only_model "${SAVE_ONLY_MODEL}" \
    --seed "${SEED}" \
    --loss_type "${LOSS_TYPE}" \
    --importance_sampling_level "${IMPORTANCE_SAMPLING_LEVEL}" \
    --scale_rewards group \
    --epsilon 0.2 \
    --epsilon_high 0.2 \
    --mask_truncated_completions false \
    --vllm_importance_sampling_correction false \
    --repetition_penalty 1.0 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.0 \
    --weight_decay 0.0 \
    --max_grad_norm 1.0 \
    --optim adamw_torch_fused \
    --dataloader_num_workers 0 \
    "${EXTRA_ARGS[@]}" "$@"
