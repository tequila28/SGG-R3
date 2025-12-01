#!/bin/bash

# ---------- GPU分配配置 ----------
# 物理GPU 1-4 对应 CUDA索引 0,1,2,3（根据实际服务器配置确认）
VLLM_CUDA_DEVICES="6,7"  # vLLM Server占 GPU 
TRAIN_CUDA_DEVICES="0,1,2,3,4,5"  # 训练使用 GPU 

# ---------- Environment Setup ----------
export NCCL_ASYNC_ERROR_HANDLING=1
export DEBUG_MODE=True
export WANDB_PROJECT=RL4SGG
export HF_ENDPOINT=https://hf-mirror.com

# ---------- Model & Data Config ----------
GROUP_SIZE=8
MODEL_PATH="sft model"    #对应数据集sft后的模型路径
DATA_PATH="JosephZ/vg150_train_sgg_prompt"
RUN_NAME="qwen2vl-7b-gspo-g${GROUP_SIZE}-vg"
OUTPUT_DIR="models/${RUN_NAME}"
mkdir -p "$OUTPUT_DIR"

# ---------- Training Config ----------
TP_SIZE=1 
PORT_BASE=8020
MAX_PIXELS=$((512 * 28 * 28))
NUM_NODES=1
MIXED_NODES=1

# ---------- IP Setup ----------
HEAD_NODE_IP="127.0.0.1"
RDZV_PORT=29500

# ---------- vLLM Server 启动 ----------
echo "Starting vLLM server via TRL command on CUDA ${VLLM_CUDA_DEVICES}..."
CUDA_VISIBLE_DEVICES=${VLLM_CUDA_DEVICES} trl vllm-serve \
    --model ${MODEL_PATH} \
    --dtype "bfloat16" \
    --max_model_len 4096 \
    --gpu_memory_utilization 0.85 \
    --enable-prefix-caching true \
    --tensor-parallel-size 2\
    --host "0.0.0.0" \
    --port ${PORT_BASE} > "${OUTPUT_DIR}/vllm_server.log" 2>&1 &

sleep 20  # 等待vLLM加载完成

# ---------- Training Command ----------
TRAIN_CMD="rl/gspo_train.py \
    --output_dir ${OUTPUT_DIR} \
    --model_name_or_path ${MODEL_PATH} \
    --dataset_name ${DATA_PATH} \
    --deepspeed ./local_scripts/zero3.json \
    --max_prompt_length 4096 \
    --max_completion_length 1024 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 2 \
    --learning_rate 2e-7 \
    --logging_steps 1 \
    --use_vllm true \
    --vllm_mode server \
    --vllm_server_host ${HEAD_NODE_IP} \
    --vllm_server_port ${PORT_BASE} \
    --vllm_server_timeout 600 \
    --bf16 true \
    --report_to none \
    --gradient_checkpointing true \
    --max_pixels ${MAX_PIXELS} \
    --temperature 1 \
    --top_p 0.9 \
    --top_k 50 \
    --num_train_epochs 1 \
    --run_name ${RUN_NAME} \
    --save_steps 4656 \
    --num_generations 8 \
    --num_iterations 1 \
    --beta 0.0 \
    --save_only_model true \
    --seed 42 \
    --use_predefined_cats true \
    --loss_type dapo \
    --importance_sampling_level sequence \
    --top_entropy_quantile 1.0"

# ---------- Training 启动 ----------
echo "Starting training on CUDA ${TRAIN_CUDA_DEVICES}..."
CUDA_VISIBLE_DEVICES=${TRAIN_CUDA_DEVICES} torchrun \
    --nnodes ${NUM_NODES} \
    --nproc_per_node 6 \
    --node_rank 0 \
    --rdzv_id grpo_run \
    --rdzv_backend c10d \
    --rdzv_endpoint ${HEAD_NODE_IP}:${RDZV_PORT} \
    ${TRAIN_CMD}

pkill -9 -f "trl vllm-serve"  # 训练结束后杀死vLLM Server进程（替换原pkill内容）