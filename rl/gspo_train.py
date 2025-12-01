# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re
import json
import glob
import copy
import random
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional
from tqdm import tqdm
from collections import Counter, OrderedDict
from functools import partial

import torch
#import torch._dynamo
#torch._dynamo.config.suppress_errors = True

import numpy as np
from datasets import load_dataset, load_from_disk
from transformers import AutoProcessor

from trl import GRPOTrainer, GRPOConfig
from trl import ModelConfig, ScriptArguments, TrlParser, get_peft_config

import tracemalloc
from contextlib import contextmanager
from PIL import Image
from typing import List, Dict

# Import reward functions
from rewards_vg import reward_funcs_registry_vg

from rewards_psg import reward_funcs_registry_psg



#---------------------- prompt templates ----------------------------
from prompt_gallery_cot import (
    PROMPT_CLOSE_PSG, 
    PROMPT_CLOSE_VG150, 
)
#---------------------------------------------------------------------------


def scale_box(box, scale):
    sw, sh = scale
    assert len(box) == 4, " len(box) != 4 "
    return [box[0]*sw, box[1]*sh, box[2]*sw, box[3]*sh]



@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy', 'format'.
    """
    reward_funcs: Optional[list[str]] = field(
        default=None,
        metadata={"help": "List of reward functions. Possible values: 'accuracy', 'format'."},
    )
    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image"},
    )
    use_predefined_cats: bool = field(
        default=False, 
        metadata={"help": "Whether to use predefined object categories"}
    )
    task_type: list[str] = field(
        default_factory=lambda: ["sgg"],
        metadata={"help": "List of tasks. Possible values: 'sgg', 'det', or 'cls'."}
    )
    use_think_prompt_inplace: bool = field(
        default=False,
        metadata={"help": "Whether to place <think>...</think> in the user's prompt."}
    )
    disable_think_tags: bool=field(
        default=False,
        metadata={"help": "Whether to disable <think> tags."}
    )
    use_ovdr_split: bool = field(
        default=False,
        metadata={"help": "Whether to use ovdr split for the dataset."}
    )
    use_fp8: bool=field(
        default=False, 
        metadata={"help": "Whether to use FP8 for training."}
    )




def main(script_args, training_args, model_args):
    if script_args.reward_funcs is None:
        script_args.reward_funcs = [
            'format_reward',  
            "stage1_category_reward",
            'stage2_node_box_reward', 
            'stage2_node_recall_reward', 
            "stage3_edge_fine_reward",
            "stage3_edge_coarse_reward"
        ]
        

    if 'psg' in script_args.dataset_name:
                
        reward_funcs = [reward_funcs_registry_psg[func] for func in script_args.reward_funcs]


        #print("=========================The dataset is psg==============================")
                        
    else:

        reward_funcs = [reward_funcs_registry_vg[func] for func in script_args.reward_funcs]


        #print("=========================The dataset is vg ===============================")
    



    #reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]

    dataset = load_dataset(script_args.dataset_name)['train']

    def assign_task_type(example, task_pool, rng):
        example["task_type"] = rng.choice(task_pool)
        return example
    
    rng = random.Random(training_args.seed)  
    dataset = dataset.map(partial(assign_task_type, task_pool=script_args.task_type, rng=rng))


    print("len(dataset):", len(dataset), "with task_type:", script_args.task_type)


    try:
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
    except:
        rank = 0
        world_size = 1

    model_type=None
    base_name = None
      
    model_name = model_args.model_name_or_path.lower()
    
    if any(key in model_name for key in ['qwen2vl', 'qwen2-vl', 'qwen-2-vl']):
        model_type = "qwen2vl"
        if '7b' in model_name:
            base_name = "Qwen/Qwen2-VL-7B-Instruct"
        elif '2b' in model_name:
            base_name = "Qwen/Qwen2-VL-2B-Instruct"
        else:
            raise Exception(f"Unknown model size in: {model_name}")
    
    elif any(key in model_name for key in ['qwen2.5vl', 'qwen2.5-vl', 'qwen2-5-vl', 'qwen-2.5-vl']):
        model_type = "qwen2.5vl"
        if '7b' in model_name:
            base_name = "Qwen/Qwen2.5-VL-7B-Instruct"
        elif '3b' in model_name:
            base_name = "Qwen/Qwen2.5-VL-3B-Instruct"
        else:
            raise Exception(f"Unknown model size in: {model_name}")
    
    else:
        raise Exception(f"Unknown model type: {model_args.model_name_or_path}")

    processor = AutoProcessor.from_pretrained(base_name, 
                    min_pixels=script_args.min_pixels,
                    max_pixels=script_args.max_pixels)

    pad_token_id = processor.tokenizer.pad_token_id
    processor.pad_token_id = pad_token_id
    processor.eos_token_id = processor.tokenizer.eos_token_id
    
    if not hasattr(training_args, "model_init_kwargs") or training_args.model_init_kwargs is None:
        training_args.model_init_kwargs = {
            'torch_dtype': torch.bfloat16,
            'attn_implementation': 'flash_attention_2'
        }

    if not hasattr(training_args, "temperature"):
        training_args.temperature = getattr(script_args, "temperature", 0.9)
    if not hasattr(training_args, "top_p"):
        training_args.top_p = getattr(script_args, "top_p", 1.0)
    if not hasattr(training_args, "top_k"):
        training_args.top_k = getattr(script_args, "top_k", 10)
    if not hasattr(training_args, "repetition_penalty"):
        training_args.repetition_penalty = getattr(script_args, "repetition_penalty", 1.0)

    print("training config:", training_args)
    print("script config:", script_args)



    class GRPOCustomDataset(torch.utils.data.Dataset):
        def __init__(self, dataset_name, original_dataset, processor, model_type,
                    use_predefined_cats, use_think_prompt_inplace=False, 
                    disable_think_tags=False):
            """
            GRPO 自定义数据集处理类
            
            Args:
                original_dataset: 原始数据集
                dataset_name: 数据集名称
                processor: 处理器
                model_type: 模型类型 ('qwen2vl', 'qwen2.5vl' 等)
                use_predefined_cats: 是否使用预定义类别
                use_think_prompt_inplace: 是否使用思考提示
                disable_think_tags: 是否禁用思考标签
            """
            self.original_dataset = original_dataset
            self.processor = processor
            self.model_type = model_type
            self.use_predefined_cats = use_predefined_cats
            self.use_think_prompt_inplace = use_think_prompt_inplace
            self.disable_think_tags = disable_think_tags
            self.dataset_name = dataset_name
            
        def __len__(self):
            return len(self.original_dataset)
        
        def __getitem__(self, idx):
            # 获取原始数据
            example = self.original_dataset[idx]
            
            # 处理图像
            image = example["image"].convert('RGB') if hasattr(example["image"], 'convert') else example["image"]
            org_iw, org_ih = image.size
            
            # 构建提示词
            if self.use_predefined_cats: 

                if 'psg' in self.dataset_name:
                
                    user_prompt = PROMPT_CLOSE_PSG

                    #print("=========================The dataset is psg==============================")
                        
                else:

                    user_prompt = PROMPT_CLOSE_VG150

                    #print("=========================The dataset is vg ===============================")

            else:


                user_prompt = PROMPT_CLOSE_VG150

            
            # 构建系统提示
            system_prompt = "You are a scene graph parsing expert. Analyze images step by step to detect objects, their positions, and relationships."  
            # 构建对话格式的提示
            prompt = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            # 计算输入尺寸（单样本版本）
            if self.model_type == 'qwen2vl':
                input_width, input_height = 1000.0, 1000.0
            elif self.model_type == 'qwen2.5vl':
                # 单样本处理
                text = self.processor.apply_chat_template(
                    prompt, tokenize=False, add_generation_prompt=True
                )
                image_inputs = self.processor(
                    text=[text], images=[image], padding=True, return_tensors="pt"
                )
                input_height = image_inputs['image_grid_thw'][0][1].item() * 14
                input_width = image_inputs['image_grid_thw'][0][2].item() * 14
            else:
                raise ValueError(f"Unsupported model type: {self.model_type}")
            
            # 处理边界框标注
            box_scale = [input_width, input_height]
            
            # 加载GT标注
            gt_objs = example["objects"]
            gt_rels = example["relationships"]
            
            if not isinstance(gt_objs, (list, tuple)):
                gt_objs = json.loads(gt_objs)
            if not isinstance(gt_rels, (list, tuple)):
                gt_rels = json.loads(gt_rels)
            
            # 归一化边界框到 [0,1]
            new_objs = []
            for obj in gt_objs:
                obj_copy = obj.copy()  # 避免修改原始数据
                obj_copy['bbox'] = scale_box(obj['bbox'], (1.0/org_iw, 1.0/org_ih))
                new_objs.append(obj_copy)
            gt_objs = new_objs
            
            # 构建场景图
            scene_graph = {"objects": gt_objs, "relations": gt_rels}
            solution = scene_graph
            
            # 返回处理后的样本
            return {
                "prompt": prompt,
                "image": image,
                "solution": solution,
                "image_id": example['image_id'],
                "task_type_list": "sgg",
                "box_scale": box_scale,
                "original_iw": org_iw,
                "original_ih": org_ih
            }


    train_dataset = GRPOCustomDataset(
        dataset_name = script_args.dataset_name,
        original_dataset=dataset,
        processor=processor,
        model_type=model_type,
        use_predefined_cats=True,
        use_think_prompt_inplace=False,
        disable_think_tags=False,
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
        eval_dataset=None,
        peft_config=get_peft_config(model_args),
        processing_class=processor,
    )

    # Check for existing checkpoint
    def find_valid_checkpoint(output_dir):
        ckpt_re = re.compile(r"checkpoint-(\d+)$")      # ↳ ends right after the digits
        
        checkpoints = sorted(
            [
                p for p in glob.glob(os.path.join(output_dir, "checkpoint-*"))
                if ckpt_re.search(os.path.basename(p))   # keep only pure-numeric checkpoints
            ],
            key=lambda p: int(ckpt_re.search(os.path.basename(p)).group(1))
        )
        for ckpt in reversed(checkpoints):  # Check latest first
            if glob.glob(os.path.join(ckpt, "global_step*")):
                return ckpt
        return None
    
    ckpt_to_resume = find_valid_checkpoint(training_args.output_dir)
    if ckpt_to_resume:
        print(f"[INFO] Resuming from checkpoint: {ckpt_to_resume}")
        trainer.train(resume_from_checkpoint=ckpt_to_resume)
    else:
        print("[INFO] Starting training from scratch")
        trainer.train()

    trainer.save_model(training_args.output_dir)
    

if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)