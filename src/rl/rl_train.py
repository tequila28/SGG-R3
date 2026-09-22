import json
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import torch
from trl import GRPOConfig, GRPOTrainer, ModelConfig, TrlParser, get_peft_config


current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(src_dir)
sys.path.insert(0, project_root)
sys.path.insert(0, src_dir)

from src.prompt.prompt_cot import PROMPT_PSG, PROMPT_VG150
from src.common.data import load_scene_graph_dataset, resolve_dataset_type
from src.common.training import (
    VisionScriptArguments, load_training_processor, prepare_model_options, resolve_model_type,
)


def scale_box(box, scale):
    sw, sh = scale
    assert len(box) == 4, "len(box) != 4"
    return [box[0] * sw, box[1] * sh, box[2] * sw, box[3] * sh]


@dataclass
class GRPOScriptArguments(VisionScriptArguments):
    """Script arguments for SGG GRPO training."""

    prompt_length_limit: int = field(
        default=4096,
        metadata={"help": "Validate full multimodal prompt length; never truncate image tokens"},
    )
    reward_config_path: Optional[str] = field(
        default="src/rl/configs/reward_config.yaml",
        metadata={"help": "Path to the reward configuration YAML file"},
    )


@dataclass
class SGGGRPOConfig(GRPOConfig):
    # Preserve the original server-based objective across TRL default changes.
    vllm_mode: str = "server"
    vllm_importance_sampling_correction: bool = False
    loss_type: str = "dapo"
    importance_sampling_level: str = "sequence"


class GRPOCustomDataset(torch.utils.data.Dataset):
    """Dataset wrapper that formats scene graph samples for GRPO training."""

    def __init__(self, dataset_name, original_dataset, processor, model_type, prompt_length_limit=4096):
        self.original_dataset = original_dataset
        self.processor = processor
        self.model_type = model_type
        self.dataset_name = dataset_name
        self.prompt_length_limit = prompt_length_limit

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, idx):
        example = self.original_dataset[idx]
        image = example["image"].convert("RGB") if hasattr(example["image"], "convert") else example["image"]
        org_iw, org_ih = image.size

        user_prompt = PROMPT_PSG if "psg" in self.dataset_name else PROMPT_VG150
        system_prompt = (
            "You are a scene graph parsing expert. Analyze images step by step "
            "to detect objects, their positions, and relationships."
        )
        prompt = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": user_prompt}]},
        ]

        text = self.processor.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
        image_inputs = self.processor(text=[text], images=[image], return_tensors="pt")
        prompt_length = image_inputs["input_ids"].shape[-1]
        if prompt_length > self.prompt_length_limit:
            raise ValueError(
                f"Image {example['image_id']}: prompt has {prompt_length} tokens, exceeding "
                f"prompt_length_limit={self.prompt_length_limit}. Increase the limit and server "
                "context together, or reduce max_pixels."
            )

        if self.model_type == "qwen2vl":
            input_width, input_height = 1000.0, 1000.0
        elif self.model_type == "qwen2.5vl":
            input_height = image_inputs["image_grid_thw"][0][1].item() * 14
            input_width = image_inputs["image_grid_thw"][0][2].item() * 14
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

        gt_objs = example["objects"]
        gt_rels = example["relationships"]
        if not isinstance(gt_objs, (list, tuple)):
            gt_objs = json.loads(gt_objs)
        if not isinstance(gt_rels, (list, tuple)):
            gt_rels = json.loads(gt_rels)

        normalized_objs = []
        for obj in gt_objs:
            obj_copy = obj.copy()
            obj_copy["bbox"] = scale_box(obj["bbox"], (1.0 / org_iw, 1.0 / org_ih))
            normalized_objs.append(obj_copy)

        gt_categories = list({obj["id"].split(".")[0].lower() for obj in normalized_objs})
        scene_graph = {
            "categories": gt_categories,
            "objects": normalized_objs,
            "relations": gt_rels,
        }

        return {
            "prompt": prompt,
            "image": image,
            "solution": scene_graph,
            "image_id": example["image_id"],
            "box_scale": [input_width, input_height],
            "original_iw": org_iw,
            "original_ih": org_ih,
        }


def main(script_args, training_args, model_args):
    if script_args.prompt_length_limit <= 0:
        raise ValueError("prompt_length_limit must be positive.")
    training_args.model_init_kwargs = prepare_model_options(script_args, training_args, model_args)
    training_args.remove_unused_columns = False
    if training_args.gradient_checkpointing:
        training_args.gradient_checkpointing_kwargs = training_args.gradient_checkpointing_kwargs or {}
        training_args.gradient_checkpointing_kwargs.setdefault("use_reentrant", False)

    dataset_type = resolve_dataset_type(script_args.dataset_name, script_args.dataset_type)
    # The reward module initializes Sentence-BERT at import time; select this rank's GPU first.
    if torch.cuda.is_available():
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    from src.rl.rewards import load_reward_config, reward_funcs_registry

    reward_dataset_type = load_reward_config(script_args.reward_config_path)["dataset"]["type"].lower()
    if reward_dataset_type != dataset_type:
        raise ValueError(f"Reward dataset type {reward_dataset_type!r} does not match {dataset_type!r}.")
    reward_funcs = reward_funcs_registry(script_args.reward_config_path)

    dataset = load_scene_graph_dataset(
        script_args.dataset_name, script_args.dataset_train_split, script_args.dataset_config
    )
    print("len(dataset):", len(dataset))

    try:
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
    except Exception:
        rank = 0
        world_size = 1

    model_type = resolve_model_type(
        model_args.model_name_or_path, revision=model_args.model_revision,
        local_files_only=script_args.local_files_only, trust_remote_code=training_args.trust_remote_code,
    )
    processor = load_training_processor(script_args, training_args, model_args)
    processor.tokenizer.padding_side = "left"

    print("training config:", training_args)
    print("script config:", script_args)

    train_dataset = GRPOCustomDataset(
        dataset_name=dataset_type,
        original_dataset=dataset,
        processor=processor,
        model_type=model_type,
        prompt_length_limit=script_args.prompt_length_limit,
    )

    print("*" * 100)
    print(f"rank={rank}, world size={world_size}, len(dataset)={len(train_dataset)}, dataset[0]:", [train_dataset[0]])
    print("use_vllm:", training_args.use_vllm)
    print("*" * 100)

    trainer = GRPOTrainer(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=train_dataset,
        peft_config=get_peft_config(model_args),
        processing_class=processor,
    )

    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)

    trainer.save_model(training_args.output_dir)
    trainer.save_state()
    if trainer.is_world_process_zero():
        processor.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, SGGGRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)


