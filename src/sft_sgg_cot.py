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

"""
Supervised fine-tuning script for decoder language models.

"""
import os
import json
import random
from tqdm import tqdm
import torch
import math
from dataclasses import dataclass, field
import re
import glob
from typing import Optional

from accelerate import Accelerator
from datasets import load_dataset, load_from_disk

from transformers import (
    AutoProcessor, 
    Qwen2VLProcessor,
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    Qwen2_5_VLProcessor
)

from trl import (
    ModelConfig,
    ScriptArguments,
    SFTConfig,
    SFTTrainer,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)

from qwen_vl_utils import process_vision_info

import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
#---------------------- prompt templates ----------------------------
from rl.prompt_gallery_cot import get_psg_categories, get_vg150_categories, PROMPT_CLOSE_VG150, PROMPT_CLOSE_PSG

from src.mega_1m_category import megasg_object_categories, megasg_relation_categories
#---------------------------------------------------------------------------

# 全局缓存变量
RELATIONSHIP_DATA_CACHE = None
IMAGE_INDEX_CACHE = None

# 全局关系分类变量
spatial_relations = []
possession_relations = []
interaction_relations = []
static_action_relations = []
dynamic_action_relations = []

def init_relation_categories(dataset_name):
    """根据数据集名称初始化关系分类"""
    global spatial_relations, possession_relations, interaction_relations, static_action_relations, dynamic_action_relations
    
    if 'psg' in dataset_name.lower():
        #print("=========================The dataset is PSG==============================")
        categories = get_psg_categories()
        spatial_relations = categories["spatial_relations"]
        static_action_relations = categories["static_action_relations"]
        dynamic_action_relations = categories["dynamic_action_relations"]
        # 为PSG数据集，将possession_relations和interaction_relations设为空，避免混淆
        possession_relations = []
        interaction_relations = []
    else:
        #print("=========================The dataset is VG150==============================")
        categories = get_vg150_categories()
        spatial_relations = categories["spatial_relations"]
        possession_relations = categories["possession_relations"]
        interaction_relations = categories["interaction_relations"]
        # 为VG150数据集，将static_action_relations和dynamic_action_relations设为空
        static_action_relations = []
        dynamic_action_relations = []

def classify_relationship(predicate, dataset_name):
    """根据谓词和数据集类型将关系分类到对应的语义类别"""
    if 'psg' in dataset_name.lower():
        # PSG数据集分类逻辑
        if predicate in spatial_relations:
            return "spatial"
        elif predicate in static_action_relations:
            return "static_action"
        elif predicate in dynamic_action_relations:
            return "dynamic_action"
        else:
            return "other"
    else:
        # VG150数据集分类逻辑
        if predicate in spatial_relations:
            return "spatial"
        elif predicate in possession_relations:
            return "possession"
        elif predicate in interaction_relations:
            return "interaction"
        else:
            return "other"

def should_normalize_bbox(model_type):
    """判断是否需要对bbox进行归一化处理"""
    # Qwen2.5模型不需要归一化，Qwen2模型需要归一化到[0, 1000]
    return model_type != "qwen2.5vl"

def normalize_bbox(box, iw, ih, model_type):
    """根据模型类型对bbox进行归一化处理"""
    if should_normalize_bbox(model_type):
        # Qwen2: 归一化到[0, 1000]
        return [
            int(box[0] / iw * 1000),
            int(box[1] / ih * 1000),
            int(box[2] / iw * 1000),
            int(box[3] / ih * 1000)
        ]
    else:
        # Qwen2.5: 保持原始坐标
        #print()
        return box

def process_objects_bboxes(objects, iw, ih, model_type):
    """处理对象列表中的bbox坐标"""
    processed_objects = []
    for obj in objects:
        # 复制对象以避免修改原始数据
        processed_obj = obj.copy()
        box = obj['bbox']
        
        # 根据模型类型决定是否归一化
        normalized_box = normalize_bbox(box, iw, ih, model_type)
        processed_obj['bbox'] = normalized_box
        processed_objects.append(processed_obj)
    
    return processed_objects

def format_answer_with_json(image_id: str, file_path: str, iw: int, ih: int, dataset_name: str, model_type: str) -> str:
    """
    数据增强版本的format_answer：从JSON文件获取数据并格式化输出
    """
    # 从缓存获取样本数据
    sample = get_sample_by_image_id_cached(image_id, file_path)
    if not sample:
        return f"Error: No data found for image_id {image_id}"
    
    # 获取objects和relationships
    objects = sample.get('objects', [])
    relationships = sample.get('relations', {})
    
    # 如果objects和relationships是字符串，则解析为JSON
    if isinstance(objects, str):
        objects = json.loads(objects)
    if isinstance(relationships, str):
        relationships = json.loads(relationships)
    
    # 处理bbox坐标
    normalized_objects = process_objects_bboxes(objects, iw, ih, model_type)
    
    # 按照stage2中物体的出现顺序对关系进行排序
    sorted_relationships = sort_relationships_by_subject_order(relationships, normalized_objects, dataset_name)
    
    # 提取去重后的类别列表
    categories = []
    seen = set()
    for obj in normalized_objects:
        category_name = obj["id"].split(".")[0]  # 提取类别名称
        if category_name not in seen:
            seen.add(category_name)
            categories.append({"id": category_name})
    
    # 生成JSON输出参数
    json_args = {"indent": None, "separators": (",", ":"), "ensure_ascii": False}

    # 构建三阶段输出
    stage1 = f"<CATEGORY>{json.dumps({'categories': categories}, **json_args)}</CATEGORY>"
    stage2 = f"<OBJECT>{json.dumps({'objects': normalized_objects}, **json_args)}</OBJECT>"
    stage3 = f"<RELATION>{json.dumps({'relations': sorted_relationships}, **json_args)}</RELATION>"
    
    return f"{stage1}\n{stage2}\n{stage3}"

def format_answer_original(image_id: str, objects: str, relationships: str, dataset_name: str, shuffle=False, model_type: str = None) -> str:
    """
    原始数据版本的format_answer：从objects和relationships字符串格式化输出
    """
    # 数据解析
    if isinstance(objects, str):
        objects = json.loads(objects)
    if isinstance(relationships, str):
        relationships = json.loads(relationships)
    
    # 创建原始ID到新ID的映射表
    obj_map = {}
    
    # 提取去重后的类别列表（Stage2要求）
    categories = []
    seen = set()
    for obj in objects:
        category_name = obj["id"].split(".")[0]
        if category_name not in seen:
            seen.add(category_name)
            categories.append({"id": category_name})
    
    # 创建类别到对象的映射
    category_to_objects = {}
    for obj in objects:
        category = obj["id"].split(".")[0]
        if category not in category_to_objects:
            category_to_objects[category] = []
        category_to_objects[category].append(obj)
    
    # 按照Stage2的类别顺序重新组织对象
    ordered_objects = []
    # 为每个类别创建实例计数器
    category_counters = {cat["id"]: 1 for cat in categories}
    
    for category in categories:
        cat_name = category["id"]
        if cat_name in category_to_objects:
            # 同一类别的对象保持原始顺序
            for obj in category_to_objects[cat_name]:
                # 记录原始ID到新ID的映射
                original_id = obj["id"]
                new_id = f"{cat_name}.{category_counters[cat_name]}"
                category_counters[cat_name] += 1
                obj_map[original_id] = new_id
                
                # 更新对象ID
                obj["id"] = new_id
                ordered_objects.append(obj)
    
    # 更新关系中的对象ID
    for r in relationships:
        r["subject"] = obj_map.get(r["subject"], r["subject"])
        r["object"] = obj_map.get(r["object"], r["object"])
    
    # 如果启用shuffle，需要进一步处理
    if shuffle:
        # 重新提取类别（用于shuffle）
        categories = []
        seen = set()
        for obj in ordered_objects:
            category_name = obj["id"].split(".")[0]
            if category_name not in seen:
                seen.add(category_name)
                categories.append({"id": category_name})
        
        # 创建类别到对象的映射（用于shuffle）
        category_to_objects = {}
        for obj in ordered_objects:
            category = obj["id"].split(".")[0]
            if category not in category_to_objects:
                category_to_objects[category] = []
            category_to_objects[category].append(obj)
        
        # 随机打乱类别顺序
        random.shuffle(categories)
        
        # 按照新的类别顺序重新组织对象
        shuffled_objects = []
        # 重置实例计数器
        category_counters = {cat["id"]: 1 for cat in categories}
        # 创建新的映射表
        new_obj_map = {}
        
        for category in categories:
            cat_name = category["id"]
            if cat_name in category_to_objects:
                # 同一类别内的对象也随机打乱
                random.shuffle(category_to_objects[cat_name])
                
                for obj in category_to_objects[cat_name]:
                    # 记录原始ID到新ID的映射
                    original_id = obj["id"]
                    new_id = f"{cat_name}.{category_counters[cat_name]}"
                    category_counters[cat_name] += 1
                    new_obj_map[original_id] = new_id
                    
                    # 更新对象ID
                    obj["id"] = new_id
                    shuffled_objects.append(obj)
        
        # 更新关系中的对象ID
        for r in relationships:
            # 先映射到第一次重命名的ID，再映射到第二次重命名的ID
            mapped_subject = obj_map.get(r["subject"], r["subject"])
            r["subject"] = new_obj_map.get(mapped_subject, mapped_subject)
            
            mapped_object = obj_map.get(r["object"], r["object"])
            r["object"] = new_obj_map.get(mapped_object, mapped_object)
        
        ordered_objects = shuffled_objects

    # 根据数据集类型对关系进行分类
    if 'psg' in dataset_name.lower():
        # PSG数据集分类
        spatial_rels = [r for r in relationships if classify_relationship(r["predicate"], dataset_name) == "spatial"]
        static_action_rels = [r for r in relationships if classify_relationship(r["predicate"], dataset_name) == "static_action"]
        dynamic_action_rels = [r for r in relationships if classify_relationship(r["predicate"], dataset_name) == "dynamic_action"]
        other_rels = [r for r in relationships if classify_relationship(r["predicate"], dataset_name) == "other"]
        
        classified_relationships = {
            "spatial_relations": spatial_rels,
            "static_action_relations": static_action_rels,
            "dynamic_action_relations": dynamic_action_rels
        }
    else:
        # VG150数据集分类
        spatial_rels = [r for r in relationships if classify_relationship(r["predicate"], dataset_name) == "spatial"]
        possession_rels = [r for r in relationships if classify_relationship(r["predicate"], dataset_name) == "possession"]
        interaction_rels = [r for r in relationships if classify_relationship(r["predicate"], dataset_name) == "interaction"]
        other_rels = [r for r in relationships if classify_relationship(r["predicate"], dataset_name) == "other"]
        
        classified_relationships = {
            "spatial_relations": spatial_rels,
            "possession_relations": possession_rels,
            "interaction_relations": interaction_rels
        }
    
    # 如果有其他类型的关系，可以单独处理或合并到某个类别中
    if other_rels:
        classified_relationships["other_relations"] = other_rels
    
    # 生成JSON输出
    json_args = {"indent": None, "separators": (",", ":"), "ensure_ascii": False}

    stage1 = f"<CATEGORY>{json.dumps({'categories': categories}, **json_args)}</CATEGORY>"
    stage2 = f"<OBJECT>{json.dumps({'objects': [{'id': o['id'], 'bbox': o['bbox']} for o in ordered_objects]}, **json_args)}</OBJECT>"
    stage3 = f"<RELATION>{json.dumps({'relations': classified_relationships}, **json_args)}</RELATION>"
    
    return f"{stage1}\n{stage2}\n{stage3}"

def sort_relationships_by_subject_order(relationships, normalized_objects, dataset_name):
    """
    按照stage2中物体的出现顺序对关系中的主语进行排序
    """
    # 创建物体ID到索引位置的映射
    object_order = {}
    for idx, obj in enumerate(normalized_objects):
        object_order[obj['id']] = idx
    
    # 对每个类别的关系进行排序
    sorted_relationships = {}
    
    if 'psg' in dataset_name.lower():
        # PSG数据集类别
        categories = ["spatial_relations", "static_action_relations", "dynamic_action_relations"]
    else:
        # VG150数据集类别
        categories = ["spatial_relations", "possession_relations", "interaction_relations"]
    
    for category in categories:
        if category in relationships and isinstance(relationships[category], list):
            # 获取该类别的关系列表
            rels = relationships[category]
            
            # 按照主语在normalized_objects中的出现顺序排序
            sorted_rels = sorted(rels, key=lambda rel: object_order.get(rel['subject'], float('inf')))
            sorted_relationships[category] = sorted_rels
        else:
            sorted_relationships[category] = []
    
    return sorted_relationships

def load_relationship_data(file_path):
    """从JSON文件加载关系数据"""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Error loading relationship data: {e}")
        return []

def get_sample_by_image_id_cached(image_id, file_path):
    """带缓存的样本查找函数"""
    global RELATIONSHIP_DATA_CACHE, IMAGE_INDEX_CACHE
    
    # 第一次调用时加载数据和建立索引
    if RELATIONSHIP_DATA_CACHE is None:
        print("Loading relationship data into cache...")
        try:
            with open(file_path, 'r') as f:
                RELATIONSHIP_DATA_CACHE = json.load(f)
            # 建立image_id到样本的索引
            IMAGE_INDEX_CACHE = {item['image_id']: item for item in RELATIONSHIP_DATA_CACHE}
            print(f"Loaded {len(RELATIONSHIP_DATA_CACHE)} samples into cache")
        except Exception as e:
            print(f"Error loading relationship data: {e}")
            return None
    
    return IMAGE_INDEX_CACHE.get(image_id)

def format_data(dataset_name, sample, use_predefined_cats=False, remove_image_size_in_prompt=True, shuffle=False, use_augmented_data=False, model_type=None, augmented_data_path=None):
    """Prepare dataset example for training."""

    # 初始化关系分类
    init_relation_categories(dataset_name)
    
    image = sample["image"].convert('RGB')
    iw, ih = image.size
    
    if use_predefined_cats:
        if 'psg' in dataset_name.lower():
            prompt = PROMPT_CLOSE_PSG
            #print("=========================The dataset is PSG==============================")
        else:
            prompt = PROMPT_CLOSE_VG150
            #print("=========================The dataset is VG150==============================")
    else:
        prompt = PROMPT_CLOSE_VG150

    use_think = 'think' in sample

    if remove_image_size_in_prompt:
        prompt = prompt.replace(f"of size ({iw} x {ih}) ", "")

    # 根据是否使用增强数据选择不同的format_answer函数
    if use_augmented_data:
        if not augmented_data_path:
            raise ValueError("augmented_data_path must be provided when use_augmented_data is True")
        # 使用数据增强版本（从JSON文件读取）
        answer = format_answer_with_json(str(sample['image_id']), augmented_data_path, iw, ih, dataset_name, model_type)
    else:
        # 使用原始数据版本
        # 处理对象bbox坐标
        objs = process_objects_bboxes(json.loads(sample['objects']), iw, ih, model_type)
        answer = format_answer_original(str(sample['image_id']), objs, sample["relationships"], dataset_name, shuffle=shuffle, model_type=model_type)

    system_prompt = "You are a scene graph parsing expert. Analyze images step by step to detect objects, their positions, and relationships." 

    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": answer}],
        },
    ]
    return {"messages": messages}

@dataclass
class CustomScriptArguments(ScriptArguments):
    use_predefined_cats: bool = field(
        default=True, 
        metadata={"help": "Whether to use predefined object categories"}
    )
    use_augmented_data: bool = field(
        default=False,                                                               
        metadata={"help": "Whether to use augmented dataset with JSON format"}
    )
    augmented_data_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the augmented data JSON file"}
    )
    max_pixels: Optional[int] = field(
        default=1024 * 28 * 28,
        metadata={"help": "Maximum number of pixels for the image"},
    )
    min_pixels: Optional[int] = field(
        default=4 * 28 * 28,
        metadata={"help": "Minimum number of pixels for the image"},
    )

def main():
    import logging
    # 获取FSDP相关的日志器
    fsdp_logger = logging.getLogger("torch.distributed.fsdp")
    # 将日志级别设置为ERROR（只显示错误，不显示警告）
    fsdp_logger.setLevel(logging.ERROR)

    accelerator = Accelerator()
    # args
    parser = TrlParser((CustomScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()

    # 验证参数
    if script_args.use_augmented_data and not script_args.augmented_data_path:
        raise ValueError("augmented_data_path must be provided when use_augmented_data is True")

    # load dataset 
    try:
        train_dataset = load_dataset(script_args.dataset_name)['train']
    except:
        train_dataset = load_from_disk(script_args.dataset_name)

    print(f"Training set size: {len(train_dataset)}")
    print("Train set[0]:", format_data(script_args.dataset_name, train_dataset[0], 
                                     use_predefined_cats=script_args.use_predefined_cats,
                                     use_augmented_data=script_args.use_augmented_data,
                                     augmented_data_path=script_args.augmented_data_path))

    # 预加载关系数据到缓存（如果使用增强数据）
    print()
    if script_args.use_augmented_data and script_args.augmented_data_path and not RELATIONSHIP_DATA_CACHE:
        _ = get_sample_by_image_id_cached("0", script_args.augmented_data_path)  # 触发缓存加载
        print("使用数据增强")

    # model config.
    quantization_config = get_quantization_config(model_args)
    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=model_args.torch_dtype,
        use_cache=False, #if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
  
    # training_args.model_init_kwargs = model_kwargs

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
        raise Exception(f"Unknown model type: {model_name}")
    

    print(f"model type is {model_type}")

    processor = AutoProcessor.from_pretrained(base_name,
                    min_pixels=script_args.min_pixels,
                    max_pixels=script_args.max_pixels,
                    local_files_only=True,
                    trust_remote_code=True
                    )
    model_cls = None
    if model_type == "qwen2vl":
        model_cls = Qwen2VLForConditionalGeneration
    elif model_type == "qwen2.5vl":
        model_cls = Qwen2_5_VLForConditionalGeneration

    assert model_cls is not None, " Unsupported model:{}".format(model_args.model_name_or_path)

    model = model_cls.from_pretrained(
        model_args.model_name_or_path, **model_kwargs
    )

    class Collator(object):
        def __init__(self, dataset_name, processor, use_predefined_cats, use_augmented_data, augmented_data_path, model_type):
            self.dataset_name = dataset_name
            self.processor = processor
            self.use_predefined_cats = use_predefined_cats
            self.use_augmented_data = use_augmented_data
            self.augmented_data_path = augmented_data_path
            self.model_type = model_type
            self._db = {}

        def __call__(self, examples):
            # Get the texts and images, and apply the chat template
            texts, image_inputs = [], []
            for example in examples:
                if str(example) not in self._db:
                    self._db[str(example)] = 0

                shuffle = (self._db[str(example)] > 0) & (random.random() > 0.5)
                format_example = format_data(self.dataset_name, example, 
                                        use_predefined_cats=self.use_predefined_cats, 
                                        use_augmented_data=self.use_augmented_data,
                                        augmented_data_path=self.augmented_data_path,
                                        shuffle=shuffle,
                                        model_type=self.model_type)['messages']
                self._db[str(example)] += 1

                text = self.processor.apply_chat_template(format_example, tokenize=False)
                image_input = process_vision_info(format_example)[0]
                texts.append(text)
                image_inputs.append(image_input)

            # Tokenize the texts and process the images
            batch = self.processor(text=texts, images=image_inputs, return_tensors="pt", padding=True)

            # The labels are the input_ids, and we mask the padding tokens in the loss computation
            labels = batch["input_ids"].clone()
            labels[labels == self.processor.tokenizer.pad_token_id] = -100  #
            # Ignore the image token index in the loss computation (model specific)
            if isinstance(self.processor, Qwen2VLProcessor) or isinstance(self.processor, Qwen2_5_VLProcessor):
                image_tokens = [151652,151653,151655]
            else:
                image_tokens = [self.processor.tokenizer.convert_tokens_to_ids(self.processor.image_token)]
            for image_token_id in image_tokens:
                labels[labels == image_token_id] = -100
            batch["labels"] = labels

            return batch
        

    ################
    # Training
    ################
    try:
        rank = torch.distributed.get_rank()  # GPU ID or node rank
        world_size = torch.distributed.get_world_size()  # Total number of GPUs/nodes

        global_batch_size = (
            training_args.per_device_train_batch_size
            * training_args.gradient_accumulation_steps
            * world_size
        )
        total_steps = len(train_dataset) // global_batch_size * training_args.num_train_epochs
        print("*"*100, "\nglobal_batch_size:", global_batch_size, " total steps:", total_steps, "\n", "*"*100)
    except:
        pass

    # 修复FSDP配置：完全禁用Trainer的梯度检查点，使用FSDP的激活检查点
    training_args.gradient_checkpointing = False  # 必须设置为False
    training_args.gradient_checkpointing_kwargs = {"use_reentrant": False}
    training_args.remove_unused_columns = False
    training_args.dataset_kwargs = {"skip_prepare_dataset": True}
    training_args.dataset_text_field = ""

    # 配置FSDP的激活检查点
    if hasattr(training_args, 'fsdp_config') and training_args.fsdp_config is not None:
        # 启用FSDP专用的激活检查点
        training_args.fsdp_config["fsdp_activation_checkpointing"] = True

    trainer = SFTTrainer(
        model=model,  # 使用已经实例化的模型
        args=training_args,
        train_dataset=train_dataset, 
        eval_dataset=None, #val_dataset,
        processing_class=processor.tokenizer,
        data_collator=Collator(script_args.dataset_name, processor, script_args.use_predefined_cats, script_args.use_augmented_data, script_args.augmented_data_path, model_type),
        peft_config=get_peft_config(model_args),
        # 不传递model_init_kwargs参数，因为model已经实例化了
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

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)

if __name__ == "__main__":
    main()