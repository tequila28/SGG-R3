"""Model and processor options shared by SFT and GRPO."""

from dataclasses import dataclass, field
from typing import Optional

from transformers import AutoConfig, AutoProcessor
from trl import ScriptArguments


@dataclass
class VisionScriptArguments(ScriptArguments):
    dataset_type: str = field(default="auto", metadata={"choices": ["auto", "vg", "psg"]})
    processor_name_or_path: Optional[str] = field(
        default=None, metadata={"help": "Processor override for checkpoints without processor files"}
    )
    local_files_only: bool = False
    min_pixels: int = 4 * 28 * 28
    max_pixels: int = 1024 * 28 * 28


def validate_pixels(min_pixels, max_pixels):
    if min_pixels <= 0 or max_pixels < min_pixels:
        raise ValueError("Require 0 < min_pixels <= max_pixels.")


def resolve_model_type(model_name_or_path, **kwargs):
    model_type = AutoConfig.from_pretrained(model_name_or_path, **kwargs).model_type
    families = {"qwen2_vl": "qwen2vl", "qwen2_5_vl": "qwen2.5vl"}
    if model_type not in families:
        raise ValueError(f"Unsupported model_type: {model_type}; expected Qwen2-VL or Qwen2.5-VL.")
    return families[model_type]


def prepare_model_options(script_args, training_args, model_args):
    validate_pixels(script_args.min_pixels, script_args.max_pixels)
    if not model_args.model_name_or_path:
        raise ValueError("--model_name_or_path is required.")
    if model_args.load_in_4bit or model_args.load_in_8bit:
        raise ValueError("Quantized training is not configured for this FSDP/ZeRO pipeline.")
    options = dict(training_args.model_init_kwargs or {})
    options.update(
        revision=model_args.model_revision,
        trust_remote_code=training_args.trust_remote_code,
        local_files_only=script_args.local_files_only,
        dtype=model_args.dtype,
        use_cache=False,
    )
    if model_args.attn_implementation:
        options["attn_implementation"] = model_args.attn_implementation
    if options.get("device_map") is not None:
        raise ValueError("Do not use device_map for distributed training; let the trainer place the model.")
    options["device_map"] = None
    return options


def load_training_processor(script_args, training_args, model_args):
    processor = AutoProcessor.from_pretrained(
        script_args.processor_name_or_path or model_args.model_name_or_path,
        revision=model_args.model_revision,
        min_pixels=script_args.min_pixels,
        max_pixels=script_args.max_pixels,
        local_files_only=script_args.local_files_only,
        trust_remote_code=training_args.trust_remote_code,
    )
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    return processor
