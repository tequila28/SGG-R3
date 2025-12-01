#!/bin/bash




export TORCH_DISTRIBUTED_DEBUG=INFO
export NCCL_DEBUG=INFO

export HF_ENDPOINT=https://hf-mirror.com


export GPUS_PER_NODE=8
export WANDB_PROJECT=RL4SGG


# batch size=4 * 2 * 16 = 128
torchrun --nnodes 1 \
    --nproc_per_node $GPUS_PER_NODE \
    --node_rank 0 \
    src/sft_sgg_cot.py \
    --model_name_or_path Qwen/Qwen2-VL-7B-Instruct \
    --dataset_name JosephZ/psg_train_sg \
    --learning_rate 1e-5 \
    --per_device_train_batch_size 8\
    --gradient_accumulation_steps 1\
    --warmup_ratio 0.05 \
    --max_grad_norm 0.3 \
    --logging_steps 1 \
    --bf16 true\
    --tf32 true\
    --report_to none \
    --attn_implementation flash_attention_2 \
    --num_train_epochs 1 \
    --run_name Qwen2-VL-2B_sft \
    --save_steps 100000 \
    --save_only_model true \
    --torch_dtype bfloat16 \
    --fsdp "full_shard auto_wrap" \
    --fsdp_config local_scripts/fsdp_config.json \
    --output_dir models/qwen2-vl-7b-sft-psg \
    --seed 42 \
    --use_predefined_cats true \
    --use_augmented_data true \
    --augmented_data_path  #增强后数据集路径
