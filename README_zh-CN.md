# SGG-R3：从 Next-Token Prediction 到端到端无偏场景图生成

中文 | [English](README.md)

[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.10.0-ee4c2c)](https://pytorch.org/)
[![TRL](https://img.shields.io/badge/TRL-1.7.1-ffd21e)](https://github.com/huggingface/trl)
[![vLLM](https://img.shields.io/badge/vLLM-0.17.1-4b8bbe)](https://github.com/vllm-project/vllm)
[![论文](https://img.shields.io/badge/论文-arXiv%3A2603.07961-b31b1b)](https://arxiv.org/abs/2603.07961)

**SGG-R3** 是一个面向端到端无偏场景图生成（Scene Graph Generation，SGG）的结构化推理框架。它将生成拆分为类别识别、目标定位和关系抽取，并结合关系增强、监督微调与基于可验证奖励的强化学习，支持 VG150 和 PSG。

## 项目概览

```text
场景图数据 -> 关系增强（可选）-> SFT -> RL -> 推理 -> 评估
```

- **结构化推理提示词**：引导类别、目标和关系的分阶段预测。
- **关系增强**：基于已标注目标扩充关系监督，缓解长尾关系稀疏问题。
- **SFT + RL**：先进行监督微调，再使用分阶段、可验证的奖励进行强化学习。
- **VG150 / PSG 支持**：内置数据集专用提示词、关系分组、后处理和评估工具。

## 仓库结构

```text
SGG-R3/
├── docs/                         # 版本迁移与运行说明
├── local_scripts/                # DeepSpeed / FSDP 配置
├── scripts/                      # 关系增强、SFT、RL 和推理启动脚本
├── src/
│   ├── common/                   # 通用数据、模型与推理工具
│   ├── data_augmentation/        # 关系增强流程
│   ├── evaluation/               # 预测汇总与 SGG 评估
│   ├── inference/                # vLLM 推理入口
│   ├── prompt/                   # 提示词和类别定义
│   ├── rl/                       # GRPO 训练、奖励函数与配置
│   └── sft/                      # 监督微调与数据构造
├── tests/                        # 仅 CPU 的 reward 配置测试
├── install.sh                    # 环境安装脚本
└── requirements.txt              # 固定版本依赖
```

## 环境安装

本项目面向 Linux + NVIDIA GPU，使用 Python 3.10、**TRL 1.7.1**、**vLLM 0.17.1**、PyTorch 2.10.0 与 Transformers 4.57.6。兼容性说明见[版本迁移文档](docs/version_migration.md)。

```bash
bash install.sh
conda activate SGG
```

手动安装：

```bash
conda create -n SGG python=3.10 -y
conda activate SGG
python -m pip install --upgrade pip 'setuptools>=77.0.3,<81' wheel
python -m pip install -r requirements.txt
python -m pip check
```

Flash Attention 是可选项：使用 `INSTALL_FLASH_ATTN=true bash install.sh` 安装后，在启动脚本设置 `ATTN_IMPLEMENTATION=flash_attention_2`。脚本默认使用 `https://hf-mirror.com`，可按需覆盖 `HF_ENDPOINT`。

## 数据格式

数据集至少需要 `image_id`、`image`、`objects` 和 `relationships` 四个字段，后两项可以是 JSON 字符串或列表。常用数据集 ID：

```text
# 训练
JosephZ/vg150_train_sgg_prompt
JosephZ/psg_train_sg

# 评估
JosephZ/vg150_val_sgg_prompt
JosephZ/vg150_test_sg
JosephZ/psg_test_sg

# SFT 可选关系增强数据
tequila3009/vg_augmentation_data
tequila3009/psg_augmentation_data
```

本地数据可使用 `load_dataset` 支持的来源，或 `Dataset.save_to_disk` / `DatasetDict.save_to_disk` 导出的目录。不要使用 Hugging Face 内部缓存目录作为数据集路径。

## 训练

请先在各启动脚本顶部配置数据集、模型、显卡和输出目录。

### 1. 关系增强（可选）

```bash
bash scripts/inference/run_sgg_augmentation.sh
```

该脚本可配置 `DATASET_TYPE`、`DATASET_PATH`、`MODEL_NAME`、`OUTPUT_FILE`、采样参数、上下文长度、batch size 和张量并行度。

### 2. 监督微调（SFT）

```bash
bash scripts/sft/run_sgg_sft.sh
```

设置 `USE_AUGMENTED_DATA=true` 时，脚本会根据 `DATASET_NAME` 选择增强关系数据集；设为 `false` 则直接使用源数据集的 `relationships`。`MAX_LENGTH=0` 表示保留完整多模态序列。

### 3. 强化学习（RL）

```bash
bash scripts/rl/run_sgg_rl.sh
```

默认单机配置将 GPU `0-5` 用于训练、`6-7` 用于 vLLM 生成。调整硬件分配时，请同步修改 `TRAIN_CUDA_DEVICES`、`VLLM_CUDA_DEVICES`、`TRAIN_PROCESSES` 和 `VLLM_TENSOR_PARALLEL_SIZE`；训练与生成显卡不能重叠。

## Reward 配置

RL 的所有奖励参数集中在 [`src/rl/configs/reward_config.yaml`](src/rl/configs/reward_config.yaml)。TRL 按固定顺序聚合四个回调：

```text
R = weights.format * R_format
  + weights.category * R_category
  + weights.node * R_node
  + weights.edge * R_edge
```

| 阶段 | 作用 | YAML 配置 |
| --- | --- | --- |
| Format | 校验 CATEGORY、OBJECT、RELATION 的 JSON 结构 | `format_reward` |
| Category | 计算类别级 F1 | `stage1_category` |
| Node | 计算目标匹配、定位质量与软召回 | `matching`、`stage2_node` |
| Edge | 计算带长尾权重的关系相似度与粗粒度覆盖 | `stage3_edge` |

RL 路径中会改变奖励数值的权重、阈值、惩罚、回退分数、语义模型选择与匹配缓存容量，均显式写在 YAML 中。VG/PSG 谓词频率表位于 [`relation_frequencies.yaml`](src/rl/configs/relation_frequencies.yaml)，选择的 `vg` 或 `psg` 表必须与 `DATASET_TYPE` 一致。

实验配置只需覆盖要调整的字段：

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

未知字段和非法值会在启动时被拒绝。完整公式、默认值和迁移说明请见 [reward 配置文档](src/rl/README.md)。可通过下列命令运行仅 CPU 的配置测试：

```bash
python -B -m unittest discover -s tests -p test_reward_config.py
```

## 推理与评估

```bash
# 推理
MODEL_NAME="models/qwen2.5-vl-3b-sft-psg" \
DATASET="JosephZ/psg_test_sg" \
bash scripts/inference/run_sgg_inference.sh

# VG150 评估
python src/evaluation/sgg_gather_preds_cot.py vg ./output_results ./output_results/vg150_predictions.json
python src/evaluation/vg150_eval.py JosephZ/vg150_val_sgg_prompt ./output_results/vg150_predictions.json

# PSG 评估
python src/evaluation/sgg_gather_preds_cot.py psg ./output_results ./output_results/psg_predictions.json
python src/evaluation/vg150_eval.py JosephZ/psg_test_sg ./output_results/psg_predictions.json
```

推理会在 `OUTPUT_DIR` 下为每张图片写入一个 JSON 文件。PSG 汇总脚本默认使用 `JosephZ/psg_test_sg`；可通过 `PSG_TEST_DATASET` 指定其他 Hugging Face 数据集 ID 或本地数据路径。

## 复现实验说明

- Qwen2-VL 使用归一化坐标 `[0, 1000]`；Qwen2.5-VL 在推理后处理阶段使用图片尺度坐标。
- RL 需要满足 `VLLM_MAX_MODEL_LEN >= PROMPT_LENGTH_LIMIT + MAX_COMPLETION_LENGTH`。
- 若需恢复优化器和学习率调度器状态，应使用 `SAVE_ONLY_MODEL=false` 保存完整 checkpoint。
- 仓库提供静态与配置检查；GPU 训练、推理和分布式 vLLM 集成仍应在目标硬件上验证。

## 致谢

本项目基于 [R1-SGG](https://github.com/gpt4vision/R1-SGG) 和 Hugging Face 格式的场景图数据集构建。

## 引用

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
