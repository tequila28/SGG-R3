# SGG-R³: From Next-Token Prediction to End-to-End Unbiased Scene Graph Generation

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)
[![Paper](https://img.shields.io/badge/Paper-ArXiv-red)](https://arxiv.org/abs/2603.07961)

> **Official implementation of SGG-R3**, a structured reasoning framework for end-to-end unbiased scene graph generation. This work addresses the challenges of sparse, long-tailed relation distributions in Scene Graph Generation (SGG) by integrating task-specific chain-of-thought reasoning with reinforcement learning.

## 🔥 Highlights

- **Structured Three-Stage Reasoning**: Decomposes scene graph generation into sequential category detection, instance grounding, and multi-type relation extraction stages
- **Relation Augmentation**: Mitigates relation sparsity by generating high-quality augmented data using MLLM
- **Dual-granularity Reward**: Combines fine-grained and coarse-grained relation rewards to address long-tail distribution
- **Leading Performance**: Achieves superior results on VG150 and PSG benchmarks compared to existing methods


