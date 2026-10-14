import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import torch
from trl import (
    ModelConfig,
    SFTConfig,
    SFTTrainer,
    TrlParser,
    get_peft_config,
)


current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(src_dir)
sys.path.insert(0, project_root)
sys.path.insert(0, src_dir)

from src.sft.cot_data_builder import format_data, get_sample_by_image_id_cached
from src.common.data import load_scene_graph_dataset, resolve_dataset_type
from src.common.training import (
    VisionScriptArguments, load_training_processor, prepare_model_options, resolve_model_type,
)


@dataclass
class CustomScriptArguments(VisionScriptArguments):
    augmented_data_path: Optional[str] = field(
        default=None,
        metadata={"help": "Local JSON path, Hugging Face dataset ID, or Hugging Face dataset URL for augmented data"},
    )


@dataclass
class SGGSFTConfig(SFTConfig):
    # Keep full-sequence SFT and avoid truncating multimodal placeholders by default.
    max_length: Optional[int] = None
    loss_type: str = "nll"
    completion_only_loss: Optional[bool] = False
    assistant_only_loss: bool = False


class VLTrainingDataset(torch.utils.data.Dataset):
    """Vision-language training dataset wrapper."""

    def __init__(self, dataset, dataset_name: str, model_type: str, augmented_data_path: Optional[str] = None):
        self.dataset = dataset
        self.dataset_name = dataset_name
        self.model_type = model_type
        self.augmented_data_path = augmented_data_path

        if self.augmented_data_path:
            first_image_id = str(dataset[0]["image_id"]) if len(dataset) > 0 else "0"
            _ = get_sample_by_image_id_cached(first_image_id, self.augmented_data_path)
            print("Using augmented data.")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        """Return one formatted training sample."""
        example = self.dataset[idx]
        formatted_data = format_data(
            dataset_name=self.dataset_name,
            sample=example,
            model_type=self.model_type,
            augmented_data_path=self.augmented_data_path,
        )

        if not isinstance(formatted_data, dict):
            raise ValueError(f"format_data should return dict, got {type(formatted_data)}")

        return formatted_data


def main():
    fsdp_logger = logging.getLogger("torch.distributed.fsdp")
    fsdp_logger.setLevel(logging.ERROR)

    parser = TrlParser((CustomScriptArguments, SGGSFTConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    if training_args.max_length == 0:
        training_args.max_length = None
    elif training_args.max_length is not None and training_args.max_length < 0:
        raise ValueError("max_length must be positive, or 0 to disable truncation.")
    training_args.model_init_kwargs = prepare_model_options(script_args, training_args, model_args)
    training_args.remove_unused_columns = False
    training_args.dataset_kwargs = {**(training_args.dataset_kwargs or {}), "skip_prepare_dataset": True}
    activation_checkpointing = bool((training_args.fsdp_config or {}).get("activation_checkpointing", False))
    if training_args.fsdp and activation_checkpointing and training_args.gradient_checkpointing:
        raise ValueError("Use FSDP activation_checkpointing or gradient_checkpointing, not both.")
    if training_args.gradient_checkpointing:
        training_args.gradient_checkpointing_kwargs = training_args.gradient_checkpointing_kwargs or {}
        training_args.gradient_checkpointing_kwargs.setdefault("use_reentrant", False)
    dataset_type = resolve_dataset_type(script_args.dataset_name, script_args.dataset_type)

    print("Loading dataset...")
    train_dataset = load_scene_graph_dataset(
        script_args.dataset_name, script_args.dataset_train_split, script_args.dataset_config
    )
    print(f"Original training set size: {len(train_dataset)}")

    model_type = resolve_model_type(
        model_args.model_name_or_path, revision=model_args.model_revision,
        local_files_only=script_args.local_files_only, trust_remote_code=training_args.trust_remote_code,
    )
    print(f"Model type: {model_type}")

    print("Creating custom dataset...")
    custom_dataset = VLTrainingDataset(
        dataset=train_dataset,
        dataset_name=dataset_type,
        model_type=model_type,
        augmented_data_path=script_args.augmented_data_path,
    )

    print(f"Custom training set size: {len(custom_dataset)}")

    if len(custom_dataset) > 0:
        print(f"Dataset size: {len(custom_dataset)}")
        sample = custom_dataset[0]
        print("Sample data keys:", list(sample.keys()))
        print("Sample structure:", sample)

    try:
        world_size = torch.distributed.get_world_size()
        global_batch_size = (
            training_args.per_device_train_batch_size
            * training_args.gradient_accumulation_steps
            * world_size
        )
        total_steps = len(custom_dataset) // global_batch_size * training_args.num_train_epochs
        print("*" * 100)
        print(f"Global batch size: {global_batch_size}, Total steps: {total_steps}")
        print("*" * 100)
    except Exception:
        pass

    print("Creating trainer...")
    processor = load_training_processor(script_args, training_args, model_args)
    processor.tokenizer.padding_side = "right"

    trainer = SFTTrainer(
        model=model_args.model_name_or_path,
        args=training_args,
        train_dataset=custom_dataset,
        eval_dataset=None,
        processing_class=processor,
        peft_config=get_peft_config(model_args),
    )

    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)

    print("Saving model...")
    trainer.save_model(training_args.output_dir)
    trainer.save_state()
    if trainer.is_world_process_zero():
        processor.save_pretrained(training_args.output_dir)
    print("Training completed.")


if __name__ == "__main__":
    main()


