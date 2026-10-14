# SGG-R3: From Next-Token Prediction to End-to-End Unbiased Scene Graph Generation

[中文文档](README_zh-CN.md) | English

[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.10.0-ee4c2c)](https://pytorch.org/)
[![TRL](https://img.shields.io/badge/TRL-1.7.1-ffd21e)](https://github.com/huggingface/trl)
[![vLLM](https://img.shields.io/badge/vLLM-0.17.1-4b8bbe)](https://github.com/vllm-project/vllm)
[![Paper](https://img.shields.io/badge/Paper-arXiv%3A2603.07961-b31b1b)](https://arxiv.org/abs/2603.07961)

Official implementation of **SGG-R3**, a structured-reasoning framework for end-to-end unbiased Scene Graph Generation (SGG). It decomposes generation into category detection, object grounding, and relation extraction, then combines relation augmentation, supervised fine-tuning, and reward-driven reinforcement learning for VG150 and PSG.

## Overview

```text
scene graph data -> relation augmentation (optional) -> SFT -> RL -> inference -> evaluation
```

- **Structured reasoning prompts** guide category, object, and relation prediction.
- **Relation augmentation** expands relation supervision from annotated objects.
- **SFT + RL** combine supervised learning with staged, verifiable rewards.
- **VG150 and PSG** are supported through dataset-specific prompts, relation groups, post-processing, and evaluation utilities.

## Repository Layout

```text
SGG-R3/
├── docs/                         # Version migration and runtime notes
├── local_scripts/                # DeepSpeed and FSDP configurations
├── scripts/                      # Launchers for augmentation, SFT, RL, and inference
├── src/
│   ├── common/                   # Shared dataset, model, and inference helpers
│   ├── data_augmentation/        # Relation augmentation pipeline
│   ├── evaluation/               # Prediction gathering and SGG metrics
│   ├── inference/                # vLLM inference entry point
│   ├── prompt/                   # Prompts and category definitions
│   ├── rl/                       # GRPO training, rewards, and configuration
│   └── sft/                      # Supervised fine-tuning and data construction
├── tests/                        # CPU-only reward configuration checks
├── install.sh                    # Environment setup helper
└── requirements.txt              # Pinned runtime dependencies
```

## Setup

The runtime targets Linux with NVIDIA GPUs, Python 3.10, **TRL 1.7.1**, **vLLM 0.17.1**, PyTorch 2.10.0, and Transformers 4.57.6. See [the version migration guide](docs/version_migration.md) for compatibility notes.

```bash
bash install.sh
conda activate SGG
```

Manual setup:

```bash
conda create -n SGG python=3.10 -y
conda activate SGG
python -m pip install --upgrade pip 'setuptools>=77.0.3,<81' wheel
python -m pip install -r requirements.txt
python -m pip check
```

Flash Attention is optional: install it with `INSTALL_FLASH_ATTN=true bash install.sh`, then set `ATTN_IMPLEMENTATION=flash_attention_2` in a launcher. Scripts default to `https://hf-mirror.com`; override `HF_ENDPOINT` as needed.

## Data

Each dataset must include `image_id`, `image`, `objects`, and `relationships`. The latter two may be JSON strings or lists. Common dataset IDs are:

```text
# Training
JosephZ/vg150_train_sgg_prompt
JosephZ/psg_train_sg

# Evaluation
JosephZ/vg150_val_sgg_prompt
JosephZ/vg150_test_sg
JosephZ/psg_test_sg

# Optional augmented relations for SFT
tequila3009/vg_augmentation_data
tequila3009/psg_augmentation_data
```

Local datasets can be supplied as normal `load_dataset` sources or directories created with `Dataset.save_to_disk` / `DatasetDict.save_to_disk`. Do not use an internal Hugging Face cache directory as a dataset path.

## Training

Set the dataset, model, GPU, and output variables near the top of each launcher before running an experiment.

### 1. Relation Augmentation (optional)

```bash
bash scripts/inference/run_sgg_augmentation.sh
```

The launcher controls `DATASET_TYPE`, `DATASET_PATH`, `MODEL_NAME`, `OUTPUT_FILE`, sampling settings, context length, batch size, and tensor parallelism.

### 2. Supervised Fine-Tuning

```bash
bash scripts/sft/run_sgg_sft.sh
```

Set `USE_AUGMENTED_DATA=true` to select an augmented relation dataset from `DATASET_NAME`; set it to `false` to train directly on the source `relationships` field. `MAX_LENGTH=0` keeps the full multimodal sequence.

### 3. Reinforcement Learning

```bash
bash scripts/rl/run_sgg_rl.sh
```

The default single-node layout reserves GPUs `0-5` for training and `6-7` for vLLM generation. When changing hardware allocation, update `TRAIN_CUDA_DEVICES`, `VLLM_CUDA_DEVICES`, `TRAIN_PROCESSES`, and `VLLM_TENSOR_PARALLEL_SIZE` together. Training and generation GPUs must not overlap.

## Reward Configuration

RL reward parameters are centralized in [`src/rl/configs/reward_config.yaml`](src/rl/configs/reward_config.yaml). TRL aggregates four callbacks in a fixed order:

```text
R = weights.format * R_format
  + weights.category * R_category
  + weights.node * R_node
  + weights.edge * R_edge
```

| Stage | Purpose | YAML section |
| --- | --- | --- |
| Format | Validates CATEGORY, OBJECT, and RELATION JSON structure | `format_reward` |
| Category | Computes category-level F1 | `stage1_category` |
| Node | Scores object matching, localization, and soft recall | `matching`, `stage2_node` |
| Edge | Scores weighted relation similarity and coarse coverage | `stage3_edge` |

All reward weights, thresholds, penalties, fallback scores, semantic-model selection, and matching-cache settings used by the RL path are explicit YAML fields. Relation frequencies are stored in [`relation_frequencies.yaml`](src/rl/configs/relation_frequencies.yaml); its selected `vg` or `psg` table must match `DATASET_TYPE`.

An experiment config can override only the fields it changes:

```yaml
# reward_override.yaml
weights:
  edge: 2.0
stage2_node:
  recall:
    iou_threshold: 0.6
stage3_edge:
  coarse:
    match_threshold: 0.8
```

```bash
REWARD_CONFIG_PATH=/path/to/reward_override.yaml \
  bash scripts/rl/run_sgg_rl.sh
```

Unknown fields and invalid values are rejected at startup. See the [complete reward reference](src/rl/README.md) for formulas, defaults, and migration notes. Run the CPU-only configuration test with:

```bash
python -B -m unittest discover -s tests -p test_reward_config.py
```

## Inference and Evaluation

```bash
# Inference
MODEL_NAME="models/qwen2.5-vl-3b-sft-psg" \
DATASET="JosephZ/psg_test_sg" \
bash scripts/inference/run_sgg_inference.sh

# VG150 evaluation
python src/evaluation/sgg_gather_preds_cot.py vg ./output_results ./output_results/vg150_predictions.json
python src/evaluation/vg150_eval.py JosephZ/vg150_val_sgg_prompt ./output_results/vg150_predictions.json

# PSG evaluation
python src/evaluation/sgg_gather_preds_cot.py psg ./output_results ./output_results/psg_predictions.json
python src/evaluation/vg150_eval.py JosephZ/psg_test_sg ./output_results/psg_predictions.json
```

Inference writes one JSON file per image under `OUTPUT_DIR`. `sgg_gather_preds_cot.py` uses `JosephZ/psg_test_sg` by default for PSG; set `PSG_TEST_DATASET` to select another Hugging Face dataset ID or a local dataset path.

## Reproducibility Notes

- Qwen2-VL uses normalized `[0, 1000]` boxes; Qwen2.5-VL uses image-size scaling during inference post-processing.
- RL requires `VLLM_MAX_MODEL_LEN >= PROMPT_LENGTH_LIMIT + MAX_COMPLETION_LENGTH`.
- Use `SAVE_ONLY_MODEL=false` when an optimizer and scheduler checkpoint is required for resuming.
- Static and configuration checks are included; GPU training, inference, and distributed vLLM integration must be validated on the target hardware.

## Acknowledgements

This project builds on [R1-SGG](https://github.com/gpt4vision/R1-SGG) and Hugging Face-format scene graph datasets.

## Citation

```bibtex
@misc{feng2026sggr3,
  title         = {{SGG-R}$^{\rm 3}$: From Next-Token Prediction to End-to-End Unbiased Scene Graph Generation},
  author        = {Feng, Jiaye and Yin, Qixiang and Liu, Yuankun and Mo, Tong and Li, Weiping},
  year          = {2026},
  eprint        = {2603.07961},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  url           = {https://arxiv.org/abs/2603.07961}
}
```
