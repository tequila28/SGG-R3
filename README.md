# SGG-R3: From Next-Token Prediction to End-to-End Unbiased Scene Graph Generation

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE) [![Python](https://img.shields.io/badge/Python-3.10-blue.svg)]() [![PyTorch](https://img.shields.io/badge/PyTorch-2.7.1-orange.svg)]() [![Paper](https://img.shields.io/badge/Paper-ArXiv-red.svg)](https://arxiv.org/abs/2603.07961) [![trl](https://img.shields.io/badge/trl-0.22.0-green.svg)](https://github.com/huggingface/trl) [![vLLM](https://img.shields.io/badge/vLLM-0.10.1-8A2BE2.svg)](https://github.com/vllm-project/vllm)

Official implementation of **SGG-R3**, a structured reasoning framework for end-to-end unbiased Scene Graph Generation (SGG). The project combines task-specific chain-of-thought prompting, relation augmentation, supervised fine-tuning, and reinforcement learning for VG150 and PSG scene graph generation.

## Highlights

- **Structured scene graph reasoning**: decomposes SGG into category detection, object grounding and relation extraction with task-specific prompts.
- **Relation augmentation**: generates additional relation supervision from pre-annotated objects to reduce sparse long-tail relation bias.
- **SFT + RL training**: supports supervised fine-tuning and reinforcement learning with verifiable rewards.
- **VG150 and PSG support**: includes dataset-specific prompts, relation categories, post-processing, and evaluation utilities.

## Repository Structure

```text
SGG-R3/
├── src/rl/configs/reward_config.yaml # Reward-function hyperparameters
├── requirements.txt                  # Main Python training/inference dependencies
├── install.sh                        # Environment installation helper
├── configs/                          # DeepSpeed/FSDP configs
├── scripts/
│   ├── inference/
│   │   ├── run_sgg_inference.sh      # Scene graph inference
│   │   └── run_sgg_augmentation.sh   # Relationship augmentation inference
│   ├── rl/run_sgg_rl.sh              # RL training entry
│   └── sft/run_sgg_sft.sh            # SFT training entry
└── src/
    ├── data_augmentation/            # Relation augmentation pipeline
    ├── evaluation/                   # Prediction gathering and SGG evaluation
    ├── inference/                    # vLLM inference pipeline
    ├── prompt/                       # Dataset prompts and category definitions
    ├── rl/                           # RL training and rewards
    └── sft/                          # SFT training and data construction
```

## Environment Setup

The recommended setup is a Linux machine with CUDA-capable GPUs. Create a conda environment named `SGG` with Python 3.10, install the main training/inference dependencies from `requirements.txt`, then install `flash-attn` separately with build isolation disabled.

### Install With Script

```bash
bash install.sh
conda activate SGG
```

### Install Manually

```bash
conda create -n SGG python=3.10 -y
conda activate SGG
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python -m pip install "flash-attn==2.8.1" --no-build-isolation
```

If the environment already exists, update dependencies with:

```bash
conda activate SGG
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python -m pip install "flash-attn==2.8.1" --no-build-isolation
```

`flash-attn` is intentionally not included in `requirements.txt` because it needs the already installed PyTorch/CUDA environment during build.

For Hugging Face access, log in when needed:

```bash
huggingface-cli login
```

The shell scripts use the Hugging Face mirror by default:

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

Remove or change this variable if you do not need the mirror.

## Data Preparation

The code expects Hugging Face-format scene graph datasets with at least these fields:

- `image_id`
- `image`
- `objects`: JSON string of object annotations
- `relationships`: JSON string of relation annotations

Training uses the train split datasets, such as `JosephZ/psg_train_sg` or `JosephZ/vg150_train_sgg_prompt`. Inference and evaluation should use the corresponding held-out/test datasets, such as `JosephZ/vg150_val_sgg_prompt` for VG150 and the PSG test dataset used in the evaluation script.

Augmented relation data is optional for SFT. The uploaded augmentation datasets are converted from the local JSON augmentation results and stored in Hugging Face dataset format:

- PSG augmentation: [tequila3009/psg_augmentation_data](https://huggingface.co/datasets/tequila3009/psg_augmentation_data)
- VG augmentation: [tequila3009/vg_augmentation_data](https://huggingface.co/datasets/tequila3009/vg_augmentation_data)

Common dataset names used by the scripts include:

Training datasets:

```text
JosephZ/vg150_train_sgg_prompt
JosephZ/psg_train_sg
```

Test / evaluation datasets:

```text
JosephZ/vg150_val_sgg_prompt
JosephZ/vg150_test_sg
JosephZ/psg_test_sg
```

Augmentation datasets:

```text
tequila3009/psg_augmentation_data
tequila3009/vg_augmentation_data
```

You can use Hugging Face dataset IDs directly or cached local dataset paths.

Before running experiments, edit the dataset and model variables at the top of each shell script.

## Training

The training pipeline contains three stages:

1. Optional relationship augmentation
2. Supervised fine-tuning (SFT)
3. Reinforcement learning (RL)

### 1. Relationship Augmentation

Generate augmented relation labels from a fine-tuned or instruction-tuned vision-language model:

```bash
bash scripts/inference/run_sgg_augmentation.sh
```

Important variables in `scripts/inference/run_sgg_augmentation.sh`:

```bash
DATASET_TYPE="auto"
DATASET_PATH="JosephZ/vg150_test_sg"
MODEL_NAME="Qwen/Qwen2.5-VL-32B-Instruct"
OUTPUT_FILE="./augmentation_results/vg_augmentation_data.json"
BATCH_SIZE=16
TENSOR_PARALLEL_SIZE=4
MAX_MODEL_LEN=4096
MAX_NEW_TOKENS=2048
TEMPERATURE=0.2
TOP_K=50
TOP_P=0.9
```

### 2. Supervised Fine-Tuning

Run SFT with FSDP:

```bash
bash scripts/sft/run_sgg_sft.sh
```

Key variables in `scripts/sft/run_sgg_sft.sh`:

```bash
MODEL_NAME_OR_PATH="Qwen/Qwen2.5-VL-3B-Instruct"
DATASET_NAME="JosephZ/psg_train_sg"
USE_AUGMENTED_DATA=true
PSG_AUGMENTED_DATA_PATH="tequila3009/psg_augmentation_data"
VG_AUGMENTED_DATA_PATH="tequila3009/vg_augmentation_data"
OUTPUT_DIR="./models/qwen2.5-vl-3b-sft-psg"
GPUS_PER_NODE=8
PER_DEVICE_TRAIN_BATCH_SIZE=16
MAX_LENGTH=8192
```

`USE_AUGMENTED_DATA` controls whether SFT uses augmented relations. When it is `true`, `AUGMENTED_DATA_PATH` is selected automatically from `DATASET_NAME` in the SFT script. When it is `false`, SFT uses the original `relationships` field from `DATASET_NAME`. PSG uses [tequila3009/psg_augmentation_data](https://huggingface.co/datasets/tequila3009/psg_augmentation_data), and VG uses [tequila3009/vg_augmentation_data](https://huggingface.co/datasets/tequila3009/vg_augmentation_data). The value can also be a local JSON file, a Hugging Face dataset ID, or a Hugging Face dataset URL.

### 3. Reinforcement Learning

Run RL training with a vLLM server and TRL:

```bash
bash scripts/rl/run_sgg_rl.sh
```

Key variables in `scripts/rl/run_sgg_rl.sh`:

```bash
VLLM_CUDA_DEVICES="6,7"
TRAIN_CUDA_DEVICES="0,1,2,3,4,5"
MODEL_PATH="./models/qwen2.5-vl-3b-sft-psg"
DATA_PATH="JosephZ/vg150_train_sgg_prompt"
OUTPUT_DIR="./models/qwen2.5vl-3b-gspo-psg"
REWARD_CONFIG_PATH="src/rl/configs/reward_config.yaml"
TOP_P=0.9
TOP_K=50
TEMPERATURE=1
NUM_GENERATIONS=8
```

Reward-function parameters are defined in `src/rl/configs/reward_config.yaml`, including debug behavior, matching weights, coarse-reward clustering parameters, and relation-frequency weighting.

## Inference and Testing

Run scene graph inference after SFT or RL training:

```bash
bash scripts/inference/run_sgg_inference.sh
```

Key variables in `scripts/inference/run_sgg_inference.sh`:

```bash
DATASET="JosephZ/vg150_test_sg"
MODEL_NAME="./models/qwen2.5-vl-3b-sft-psg"
OUTPUT_DIR="./output_results"
BATCH_SIZE=16
TENSOR_PARALLEL_SIZE=4
MAX_MODEL_LEN=4096
MAX_NEW_TOKENS=2048
TEMPERATURE=0.2
TOP_K=50
TOP_P=0.9
REPETITION_PENALTY=1.05
```

The inference script saves one JSON file per image under `OUTPUT_DIR`. Each file contains `image_id`, raw model response, ground-truth objects and relationships, and the `box_scale` used by post-processing.

## Evaluation

The evaluation pipeline has two steps:

1. Gather raw per-image JSON predictions into the evaluator format.
2. Run VG150/PSG scene graph evaluation.

Example for VG150:

```bash
python src/evaluation/sgg_gather_preds_cot.py \
    vg \
    ./output_results \
    ./output_results/vg150_predictions.json

python src/evaluation/vg150_eval.py \
    JosephZ/vg150_val_sgg_prompt \
    ./output_results/vg150_predictions.json
```

Example for PSG:

```bash
python src/evaluation/sgg_gather_preds_cot.py \
    psg \
    ./output_results \
    ./output_results/psg_predictions.json

python src/evaluation/vg150_eval.py \
    JosephZ/psg_test_sg \
    ./output_results/psg_predictions.json
```

The evaluator reports standard scene graph recall metrics and frequency-group statistics for long-tail relation analysis.

## Notes

- Qwen2-VL uses normalized box scale `[0, 1000]`; Qwen2.5-VL uses image-size box scaling in the inference post-processing path.
- Shell scripts define important hyperparameters near the top before passing them into Python entries.
- vLLM tensor parallel size should match the number of visible GPUs assigned to the vLLM process.
- Large-scale SFT/RL runs require substantial GPU memory; tune `BATCH_SIZE`, `TENSOR_PARALLEL_SIZE`, `MAX_MODEL_LEN`, and DeepSpeed/FSDP configs for your hardware.

## Acknowledgement

We thank [gpt4vision/R1-SGG](https://github.com/gpt4vision/R1-SGG) for releasing the open-source code and Hugging Face-format scene graph datasets, which formed the foundation of our implementation.

## Citation

If you find this project useful, please cite the paper:

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










