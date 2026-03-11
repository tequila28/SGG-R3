# SGG-R³: From Next-Token Prediction to End-to-End Unbiased Scene Graph Generation

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)
[![Paper](https://img.shields.io/badge/Paper-ArXiv-red)](https://arxiv.org/abs/2603.07961)

> **Official implementation of SGG-R3**, a structured reasoning framework for end-to-end unbiased scene graph generation. This work addresses the challenges of sparse, long-tailed relation distributions in Scene Graph Generation (SGG) by integrating task-specific chain-of-thought reasoning with reinforcement learning.

## 🔥 Highlights

- **Structured Three-Stage Reasoning**: Decomposes scene graph generation into sequential category detection, instance grounding, and multi-type relation extraction stages
- **Relation Augmentation**: Mitigates relation sparsity by generating high-quality augmented data using MLLM
- **Dual-granularity Reward**: Combines fine-grained and coarse-grained relation rewards to address long-tail distribution
- **Leading Performance**: Achieves superior results on VG150 and PSG benchmarks compared to existing methods


## 📊 Key Results

| Method | Dataset | Params | Recall | mRecall | zsRecall |
|--------|---------|--------|---------|---------|----------|
| SGG-R3 (SFT+RL) | VG150 | 3B | **36.0** | **14.8** | **6.1** |
| SGG-R3 (SFT+RL) | PSG | 3B | **52.5** | **44.3** | **7.7** |



## 🏗️ Framework Overview

SGG-R3 integrates supervised fine-tuning (SFT) and reinforcement learning (RL) with Group Sequence Policy Optimization (GSPO) in a three-stage structured reasoning framework:

```bash
Input Image
↓
Stage 1: Object Category Detection
↓
Stage 2: Object Instance Grounding
↓
Stage 3: Multi-type Relation Extraction
↓
Structured Scene Graph Output
```
