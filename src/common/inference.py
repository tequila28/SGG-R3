"""Shared vLLM 0.17.1 engine and sampling parameters."""

import argparse
from dataclasses import dataclass

from transformers import AutoProcessor
from vllm import LLM, SamplingParams


@dataclass(kw_only=True)
class InferenceOptions:
    dataset_split: str = "train"
    dataset_config: str | None = None
    model_revision: str = "main"
    dtype: str = "bfloat16"
    seed: int = 42
    min_pixels: int = 4 * 28 * 28
    max_pixels: int = 1024 * 28 * 28
    max_model_len: int = 4096
    max_new_tokens: int = 2048
    temperature: float = 0.1
    top_k: int = 50
    top_p: float = 0.95
    min_p: float = 0.0
    repetition_penalty: float = 1.05
    batch_size: int = 1
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.85
    max_num_seqs: int = 128
    max_num_batched_tokens: int | None = None
    enforce_eager: bool = False
    enable_prefix_caching: bool = True
    trust_remote_code: bool = False
    num_workers: int = 0


def add_inference_arguments(parser, **defaults):
    group = parser.add_argument_group("vLLM engine and sampling")
    options = InferenceOptions(**defaults)
    for name, value in vars(options).items():
        if isinstance(value, bool):
            group.add_argument(f"--{name}", action=argparse.BooleanOptionalAction, default=value)
        else:
            value_type = int if name == "max_num_batched_tokens" else str if value is None else type(value)
            group.add_argument(f"--{name}", type=value_type, default=value)


def validate_inference_options(options):
    for name in ("batch_size", "tensor_parallel_size", "max_model_len", "max_new_tokens", "max_num_seqs"):
        if getattr(options, name) <= 0:
            raise ValueError(f"{name} must be positive.")
    if options.min_pixels <= 0 or options.max_pixels < options.min_pixels:
        raise ValueError("Require 0 < min_pixels <= max_pixels.")
    if options.max_new_tokens >= options.max_model_len:
        raise ValueError("max_model_len must leave space for prompt tokens as well as max_new_tokens.")
    if not 0 < options.gpu_memory_utilization <= 1:
        raise ValueError("gpu_memory_utilization must be in (0, 1].")
    if options.max_num_batched_tokens is not None and options.max_num_batched_tokens < options.max_num_seqs:
        raise ValueError("max_num_batched_tokens must be at least max_num_seqs.")
    if options.num_workers < 0:
        raise ValueError("num_workers must be non-negative.")
    if options.temperature < 0 or not 0 < options.top_p <= 1 or not 0 <= options.min_p <= 1:
        raise ValueError("Require temperature >= 0, 0 < top_p <= 1 and 0 <= min_p <= 1.")
    if options.top_k < -1 or options.repetition_penalty <= 0:
        raise ValueError("Require top_k >= -1 and repetition_penalty > 0.")


def load_inference_model(model_name, options):
    validate_inference_options(options)
    processor = AutoProcessor.from_pretrained(
        model_name, revision=options.model_revision, trust_remote_code=options.trust_remote_code,
        min_pixels=options.min_pixels, max_pixels=options.max_pixels,
    )
    engine_kwargs = {
        "model": model_name,
        "revision": options.model_revision,
        "dtype": options.dtype,
        "seed": options.seed,
        "trust_remote_code": options.trust_remote_code,
        "max_model_len": options.max_model_len,
        "tensor_parallel_size": options.tensor_parallel_size,
        "gpu_memory_utilization": options.gpu_memory_utilization,
        "max_num_seqs": options.max_num_seqs,
        "enforce_eager": options.enforce_eager,
        "enable_prefix_caching": options.enable_prefix_caching,
        "limit_mm_per_prompt": {"image": 1, "video": 0},
        "mm_processor_kwargs": {"min_pixels": options.min_pixels, "max_pixels": options.max_pixels},
        "generation_config": "vllm",
    }
    if options.max_num_batched_tokens is not None:
        engine_kwargs["max_num_batched_tokens"] = options.max_num_batched_tokens
    return LLM(**engine_kwargs), processor


def build_sampling_params(options):
    validate_inference_options(options)
    return SamplingParams(
        n=1, temperature=options.temperature, top_k=options.top_k if options.top_k > 0 else -1,
        top_p=options.top_p, min_p=options.min_p, repetition_penalty=options.repetition_penalty,
        max_tokens=options.max_new_tokens, seed=options.seed,
    )
