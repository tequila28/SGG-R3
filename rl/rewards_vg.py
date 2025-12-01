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
from datetime import datetime
from typing import List, Dict
from functools import lru_cache
import numpy as np
from scipy.optimize import linear_sum_assignment
import spacy
import numpy as np
from sentence_transformers import SentenceTransformer
import json
from datetime import datetime
from scipy.stats import entropy
import torch
import torch.distributed as dist
import time 
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN
import math
from prompt_gallery_cot import (
    VG150_PREDICATES, 
    VG150_OBJ_CATEGORIES
)


# ============================ 参数定义区域 ============================

# 全局参数
DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() 
LOG_PATH = os.getenv("LOG_PATH", "debug.log")
STRICT_FORMAT = os.getenv("STRICT_FORMAT", "true").lower() == "true"

# 奖励权重参数
FORMAT_REWARD_WEIGHT = float(os.getenv("FORMAT_REWARD_WEIGHT", '1.0'))
NODE_REWARD_WEIGHT = float(os.getenv("NODE_REWARD_WEIGHT", '3.0'))
EDGE_REWARD_WEIGHT = float(os.getenv("EDGE_REWARD_WEIGHT", '6.0'))
CATEGORY_REWARD_WEIGH = float(os.getenv("CATEGORY_REWARD_WEIGH", '2.0'))
DIVERSITY_REWARD_WEIGHT = 2.0

# 匹配权重参数
SEM_WEIGHT = 1.0
IOU_WEIGHT = 3.0
BOX_L1_WEIGHT = 3.0

# 多样性奖励参数
CLUSTER_THRESHOLD = 0.7
MIN_CLUSTER_SIZE = 1



# ============================ Sentence-BERT 相关部分 ============================

# 获取当前进程使用的CUDA设备
def get_current_cuda_device():
    """获取当前进程使用的CUDA设备号"""
    if torch.cuda.is_available():
        current_device = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(current_device)
        return f'cuda:{current_device}', current_device, device_name
    else:
        return 'cpu', None, 'CPU'

# 检查是否为第一个GPU进程
def is_first_gpu_process():
    """检查当前进程是否为第一个GPU进程"""
    if torch.cuda.is_available():
        return torch.cuda.current_device() == 0
    return True  # 如果没有GPU，也认为是第一个进程

# 初始化Sentence-BERT模型
def initialize_sentence_bert_model():
    """初始化Sentence-BERT模型并计算初始embedding"""
    # 获取设备信息
    device_str, device_id, device_name = get_current_cuda_device()
    is_first_process = is_first_gpu_process()
    
    if is_first_process:
        print(f"当前进程使用的设备: {device_str} ({device_name})")
        print("正在加载小型Sentence-BERT模型...")
    
    start_time = time.time()

    # 将模型放到当前CUDA设备上，并设置为eval模式
    word_model = SentenceTransformer('all-MiniLM-L6-v2').to(device_str)
    word_model.eval()  # 设置为评估模式

    if is_first_process:
        load_time = time.time() - start_time
        print(f"模型加载完成，耗时: {load_time:.2f}秒")
        print(f"模型设备: {next(word_model.parameters()).device}")
        print(f"模型模式: {'eval' if not word_model.training else 'train'}")
    
    return word_model, is_first_process

def initialize_embeddings(word_model, is_first_process):
    """预计算物体和谓词的初始embedding"""
    @lru_cache(maxsize=4096)
    def get_doc_cached(word: str):
        """获取单词的embedding,带缓存优化"""
        #if is_first_process:
            #print(f"计算{word}embedding")
        embedding = word_model.encode([word])
        return embedding

    # 预计算所有类别和关系的embedding
    if is_first_process:
        print("初始化物体和谓词embedding")
    
    for predicate in VG150_PREDICATES:
        get_doc_cached(predicate)
    for obj in VG150_OBJ_CATEGORIES:
        get_doc_cached(obj)
    
    if is_first_process:
        print("初始化embedding完毕")
        print()
    
    return get_doc_cached

# 全局模型和函数初始化
word_model, is_first_process = initialize_sentence_bert_model()
get_doc = initialize_embeddings(word_model, is_first_process)



def extract_answer_content(text: str) -> str:
    """
    Extracts the content between <OBJECT> and <RELATION> tags, combines their content.
    Supports nested relationship categories (spatial_relations, possession_relations, etc.)

    Returns:
        str: The extracted content.
    """
    text = text.replace("```", " ").replace("json", " ").strip()
    
    # 检查是否有 <OBJECT> 和 <RELATIONSHIP> 标签（新格式）
    object_match = re.search(r"<OBJECT>(.*?)</OBJECT>", text, re.DOTALL)
    relationship_match = re.search(r"<RELATION>(.*?)</RELATION>", text, re.DOTALL)
    
    if object_match and relationship_match:
        # 新格式：将 OBJECT 和 RELATION 的内容拼在一起
        object_content = object_match.group(1).strip()
        relationship_content = relationship_match.group(1).strip()
        
        try:
            # 解析对象内容，处理可能的嵌套格式
            obj_data = json.loads(object_content)
            if isinstance(obj_data, dict) and 'objects' in obj_data:
                # 格式: {"objects": [...]}
                object_list = obj_data['objects']
            elif isinstance(obj_data, list):
                # 格式: [...]
                object_list = obj_data
            else:
                object_list = []
            
            # 解析关系内容，提取所有关系类型并合并
            rel_data = json.loads(relationship_content)
            all_relationships = []
            
            if isinstance(rel_data, dict):
                # 处理嵌套的关系分类格式
                # 格式: {"relations": {"spatial_relations": [...], ...}}
                if 'relations' in rel_data and isinstance(rel_data['relations'], dict):
                    # 深度嵌套格式
                    for rel_type, rel_list in rel_data['relations'].items():
                        if isinstance(rel_list, list):
                            all_relationships.extend(rel_list)
                else:
                    # 格式: {"spatial_relations": [...], "possession_relations": [...]}
                    for rel_type, rel_list in rel_data.items():
                        if isinstance(rel_list, list):
                            all_relationships.extend(rel_list)
            elif isinstance(rel_data, list):
                # 格式: [...]
                all_relationships = rel_data
            
            # 重新构建为标准的JSON结构
            object_json = json.dumps(object_list)
            relationship_json = json.dumps(all_relationships)
            
            combined_content = f"""{{
    "objects": {object_json},
    "relations": {relationship_json}
}}"""
            return combined_content
        except json.JSONDecodeError as e:
            # 如果解析失败，回退到原始内容
            print(f"JSON解析错误: {e}")
            combined_content = f"""{{
    "objects": {object_content},
    "relations": {relationship_content}
}}"""
            return combined_content
        
    elif object_match:
        # 只有 OBJECT 标签
        object_content = object_match.group(1).strip()
        try:
            obj_data = json.loads(object_content)
            if isinstance(obj_data, dict) and 'objects' in obj_data:
                object_list = obj_data['objects']
            elif isinstance(obj_data, list):
                object_list = obj_data
            else:
                object_list = []
            object_json = json.dumps(object_list)
            return f'{{"objects": {object_json}, "relations": []}}'
        except json.JSONDecodeError:
            return f'{{"objects": {object_content}, "relations": []}}'
    
    elif relationship_match:
        # 只有 RELATIONSHIP 标签
        relationship_content = relationship_match.group(1).strip()
        try:
            rel_data = json.loads(relationship_content)
            all_relationships = []
            
            if isinstance(rel_data, dict):
                if 'relations' in rel_data and isinstance(rel_data['relations'], dict):
                    for rel_type, rel_list in rel_data['relations'].items():
                        if isinstance(rel_list, list):
                            all_relationships.extend(rel_list)
                else:
                    for rel_type, rel_list in rel_data.items():
                        if isinstance(rel_list, list):
                            all_relationships.extend(rel_list)
            elif isinstance(rel_data, list):
                all_relationships = rel_data
            
            relationship_json = json.dumps(all_relationships)
            return f'{{"objects": [], "relations": {relationship_json}}}'
        except json.JSONDecodeError:
            return f'{{"objects": [], "relations": {relationship_content}}}'
    
    else:
        # 如果没有 OBJECT 和 RELATIONSHIP 标签，回退到原来的 <answer> 标签处理
        # Try to find full <answer>...</answer>
        match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
        if match:
            return match.group(1).strip()
        # Fallback: everything after the first <answer>
        match = re.search(r"<answer>(.*)", text, re.DOTALL)
        return match.group(1).strip() if match else ""



def refine_node_edge(obj):
    return obj.strip().lower()



def category_semantic_similarity(pred_id: str, gt_id: str) -> float:
    """
    计算两个类别ID的语义相似度
    
    Args:
        pred_id: 预测类别ID (如 "cat.123")
        gt_id: 真实类别ID (如 "animal.456")
    
    Returns:
        float: 语义相似度分数 (0-1之间)
    """
    # Extract category names from ids (substring before the dot)
    cat_pred = pred_id.split('.')[0].lower().strip()
    cat_gt = gt_id.split('.')[0].lower().strip()
    
    # 如果是相同的类别，直接返回1.0，避免模型计算
    if cat_pred == cat_gt:
        return 1.0
    
    # 处理空字符串
    if not cat_pred or not cat_gt:
        return 0.0
    
    # 获取embedding并计算相似度
    emb_pred = get_doc(cat_pred)
    emb_gt = get_doc(cat_gt)
    
    # 计算余弦相似度
    similarity = word_model.similarity(emb_pred, emb_gt)
    
    return max(0.0, min(1.0, similarity))  # 确保在[0,1]范围内



print()
print(category_semantic_similarity('person', 'woman'))
print()


def compute_iou(boxA, boxB):
    # box format: [x1, y1, x2, y2]
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interWidth = max(0, xB - xA)
    interHeight = max(0, yB - yA)
    interArea = interWidth * interHeight
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    unionArea = boxAArea + boxBArea - interArea
    return 0.0 if unionArea == 0 else interArea / unionArea

def compute_giou(boxA, boxB):
    """
    Calculate the Generalized Intersection over Union (GIoU) of two bounding boxes.

    Parameters:
      boxA: list or tuple of [x1, y1, x2, y2]
      boxB: list or tuple of [x1, y1, x2, y2]

    Returns:
      giou: float, the Generalized IoU between boxA and boxB.
    """
    # Calculate the (x, y)-coordinates of the intersection rectangle.
    inter_x1 = max(boxA[0], boxB[0])
    inter_y1 = max(boxA[1], boxB[1])
    inter_x2 = min(boxA[2], boxB[2])
    inter_y2 = min(boxA[3], boxB[3])

    # Compute the width and height of the intersection rectangle.
    inter_width = max(0, inter_x2 - inter_x1)
    inter_height = max(0, inter_y2 - inter_y1)

    # Compute the area of intersection rectangle.
    intersection = inter_width * inter_height

    # Compute the area of both bounding boxes.
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    # Compute the union area.
    union = areaA + areaB - intersection
    # Ensure no division by zero.
    if union == 0:
        return 0.0

    # Compute the standard Intersection over Union (IoU).
    iou = intersection / union

    # Find the smallest (axis-aligned) box that encloses both boxes.
    c_x1 = min(boxA[0], boxB[0])
    c_y1 = min(boxA[1], boxB[1])
    c_x2 = max(boxA[2], boxB[2])
    c_y2 = max(boxA[3], boxB[3])
    areaC = (c_x2 - c_x1) * (c_y2 - c_y1)

    # Calculate the Generalized IoU.
    giou_value = iou - (areaC - union) / areaC

    return giou_value

def box_L1(boxA, boxB):
    # Calculate the sum of absolute differences between the coordinates
    l1_distance = sum(abs(a - b) for a, b in zip(boxA, boxB))
    return l1_distance

def cost_function(pred, gt, sem_weight=SEM_WEIGHT, iou_weight=IOU_WEIGHT, box_l1_weight=BOX_L1_WEIGHT):
    assert len(pred['bbox']) == 4, f"Invalid bbox length: {len(pred['bbox'])}"
    iou = compute_giou(pred['bbox'], gt['bbox']) # use giou
    sem_sim = category_semantic_similarity(pred['id'], gt['id'])
    return sem_weight * (1.0 - sem_sim) + iou_weight * (1.0 - iou) + box_l1_weight * box_L1(pred['bbox'], gt['bbox'])

def _freeze_objs(objs):
    """
    Turn a list of scene‑graph objects into a hashable key:
    ('id', (x1, y1, x2, y2))
    """
    return tuple(
        (o["id"], tuple(o["bbox"])) for o in objs
    )

def _bi_match_impl(groundtruths, predictions,
                   sem_weight=SEM_WEIGHT,
                   iou_weight=IOU_WEIGHT,
                   box_l1_weight=BOX_L1_WEIGHT):
    num_gt = len(groundtruths)
    num_pred = len(predictions)
    pad = max(0, num_gt - num_pred)
    cost_matrix = np.zeros((num_pred + pad, num_gt))

    for i, pred in enumerate(predictions):
        for j, gt in enumerate(groundtruths):
            cost_matrix[i, j] = cost_function(
                pred, gt, sem_weight, iou_weight, box_l1_weight
            )

    if pad:
        cost_matrix[num_pred:, :] = 1e5       # high cost for padded rows

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    return [
        {
            "groundtruth": groundtruths[c],
            "prediction":  predictions[r],
            "cost":        cost_matrix[r, c],
        }
        for r, c in zip(row_ind, col_ind)
        if r < num_pred
    ]

@lru_cache(maxsize=4096)
def _bi_match_cached(gt_key, pred_key,
                     sem_weight, iou_weight, box_l1_weight):
    """
    Hashable wrapper so we can cache across identical calls.
    Converts the frozen keys back to lists of dicts for the impl fn.
    """
    groundtruths = [
        {"id": obj_id, "bbox": list(bbox)}
        for obj_id, bbox in gt_key
    ]
    predictions = [
        {"id": obj_id, "bbox": list(bbox)}
        for obj_id, bbox in pred_key
    ]
    return _bi_match_impl(
        groundtruths, predictions,
        sem_weight, iou_weight, box_l1_weight
    )

def bi_match(groundtruths, predictions,
             sem_weight=SEM_WEIGHT,
             iou_weight=IOU_WEIGHT,
             box_l1_weight=BOX_L1_WEIGHT):
    """
    Thin, cached front‑end that keeps the original signature.
    """
    return _bi_match_cached(
        _freeze_objs(groundtruths),
        _freeze_objs(predictions),
        sem_weight, iou_weight, box_l1_weight,
    )

def bi_match_triplets(gt_rels, pred_rels):
    num_gt = len(gt_rels)
    num_pred = len(pred_rels)
    pad = max(0, num_gt - num_pred)
    cost_matrix = np.zeros((num_pred + pad, num_gt))
    for i, pred in enumerate(pred_rels):
        for j, gt in enumerate(gt_rels):
            cost_matrix[i, j] = 1.0 - category_semantic_similarity(refine_node_edge(pred["subject"]), refine_node_edge(gt["subject"]) ) *\
                                category_semantic_similarity(refine_node_edge(pred["object"]),  refine_node_edge(gt["object"]) ) *\
                                category_semantic_similarity(refine_node_edge(pred["predicate"]), refine_node_edge(gt["predicate"]))
        
    if pad:
        cost_matrix[num_pred:, :] = 1e5

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    return [
        {
            "groundtruth": gt_rels[c],
            "prediction":  pred_rels[r],
            "cost":        cost_matrix[r, c],
        }
        for r, c in zip(row_ind, col_ind)
        if r < num_pred
    ]

def scale_box(box, scale):
    sw, sh = scale
    assert len(box) == 4, " len(box) != 4 "
    return [box[0]*sw, box[1]*sh, box[2]*sw, box[3]*sh]


def stage1_category_reward(completions, solution, image_id, task_type_list, box_scale, **kwargs):
    """Compute class-level F1 score reward based on category matching"""
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")

    for content, sol, im_id, task_type, box_wh in zip(contents, solution, image_id, task_type_list, box_scale):
        reward = 0.0

        if task_type not in ['sgg', 'det']:
            rewards.append(0)
            continue

        try:
            gt_objs = sol['objects']
            
            # Parse predicted results
            object_match = re.search(r"<CATEGORY>(.*?)</CATEGORY>", content, re.DOTALL)
            if not object_match:
                rewards.append(0.0)
                continue

            try:
                stage1_preds = json.loads(object_match.group(1).strip())
                pred_objs = stage1_preds["categories"]
                
                # Get unique categories from ground truth and predictions
                gt_categories = set(obj["id"].split('.')[0].lower() for obj in gt_objs)
                
                # Check for duplicates in predictions
                pred_categories_list = [obj["id"].split('.')[0].lower() for obj in pred_objs]
                pred_categories = set(pred_categories_list)
                
                # If there are duplicates (list length > set length), set reward to 0
                if len(pred_categories_list) > len(pred_categories):
                    rewards.append(0.0)
                    if DEBUG_MODE:
                        with open(LOG_PATH, "a") as f:
                            f.write(f"------------- {current_time} Stage2 Class F1 Reward -------------\n")
                            f.write(f"Image: {im_id}\n")
                            f.write("Duplicate categories detected in predictions\n")
                            f.write(f"Final Reward: 0.0\n")
                    continue
                
                # Calculate precision and recall
                true_positives = len(gt_categories & pred_categories)
                false_positives = len(pred_categories - gt_categories)
                false_negatives = len(gt_categories - pred_categories)
                
                precision = true_positives / max(1, (true_positives + false_positives))
                recall = true_positives / max(1, (true_positives + false_negatives))
                
                # Calculate F1 score
                f1_score = 2 * (precision * recall) / max(1e-9, (precision + recall))
                
                # The reward is based on F1 score
                reward = f1_score * CATEGORY_REWARD_WEIGH

                if DEBUG_MODE:
                    with open(LOG_PATH, "a") as f:
                        f.write(f"------------- {current_time} Stage2 Class F1 Reward -------------\n")
                        f.write(f"Image: {im_id}\n")
                        f.write(f"GT Categories: {gt_categories}\n")
                        f.write(f"Pred Categories: {pred_categories}\n")
                        f.write(f"Precision: {precision:.3f}, Recall: {recall:.3f}\n")
                        f.write(f"F1 Score: {f1_score:.3f}\n")
                        f.write(f"Final Reward: {reward:.3f}\n")

            except json.JSONDecodeError:
                if DEBUG_MODE:
                    with open(LOG_PATH, "a") as f:
                        f.write(f"JSON parsing failed: {object_match.group(1)}\n")
            except Exception as e:
                if DEBUG_MODE:
                    with open(LOG_PATH, "a") as f:
                        f.write(f"Other error: {str(e)}\n")

        except Exception as e:
            if DEBUG_MODE:
                with open(LOG_PATH, "a") as f:
                    f.write(f"Global error: {str(e)}\n")

        rewards.append(reward)

    return rewards



def stage2_node_box_reward(completions, solution, image_id, task_type_list, box_scale, **kwargs):
    """Compute node-level rewards with stage 1 bounding box rewards only"""
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")


    for content, sol, im_id, task_type, box_wh in zip(contents, solution, image_id, task_type_list, box_scale):
        reward = 0.0
        match_objects = []
        
        if task_type not in ['sgg', 'det']:
            rewards.append(0)
            continue
        
        try:
            gt_objs = sol['objects']
            
            # ========== 只处理第一阶段<OBJECT_BOX>标签 ==========
            object_box_match = re.search(r"<OBJECT>(.*?)</OBJECT>", content, re.DOTALL)
            if object_box_match:
                try:
                    # 解析预测结果
                    stage1_preds = json.loads(object_box_match.group(1).strip())
                    pred_objs = []
                    for obj in stage1_preds["objects"]:
                        obj['bbox'] = scale_box(obj.get('bbox', [0,0,0,0]), (1.0 / box_wh[0], 1.0 / box_wh[1]))
                        obj['id'] = refine_node_edge(obj['id'])
                        pred_objs.append(obj)
                    
                    # 检查类别重复(防止刷分)
                    pred_categories = {}
                    for obj in pred_objs:
                        category = obj['id'].split('.')[0]
                        pred_categories[category] = pred_categories.get(category, 0) + 1
                    
                    if any(count > 20 for count in pred_categories.values()):
                        reward = 0.0
                    else:
                        # 计算边界框匹配奖励
                        assignments = bi_match(gt_objs, pred_objs)
                        for assign in assignments:
                            gt = assign['groundtruth']
                            pred = assign['prediction']
                            if pred and pred.get('id'):
                                # 计算IoU和L1距离
                                iou = compute_iou(gt['bbox'], pred['bbox'])
                                l1 = box_L1(gt['bbox'], pred['bbox'])
                                
                                # 综合边界框质量得分
                                box_score = (iou * IOU_WEIGHT + np.exp(-l1) * BOX_L1_WEIGHT) / (IOU_WEIGHT + BOX_L1_WEIGHT)
                                reward += box_score * NODE_REWARD_WEIGHT
                                match_objects.append(
                                    f"Matched: GT {gt['id']} -> Pred {pred['id']} (IoU:{iou:.2f}, L1:{l1:.1f})"
                                )
                        
                        reward /= max(1, len(gt_objs))
                        
                except Exception as e:
                    if DEBUG_MODE:
                        with open(LOG_PATH, "a") as f:
                            f.write(f"Error parsing stage1: {str(e)}\n")
                    reward = 0.0
            else:
                reward = 0.0
            
            rewards.append(reward)

        except Exception:
            rewards.append(0.0)

        if DEBUG_MODE:
            with open(LOG_PATH, "a") as f:
                f.write(f"------------- {current_time} task_type:{task_type} Stage1 Box Reward -------------\n")
                f.write(f"Content: {content}\n")
                f.write(f"Final Reward: {reward:.3f}\n")
                if match_objects:
                    f.write("Matches:\n" + "\n".join(match_objects) + "\n")

    return rewards



def stage2_node_recall_reward(completions, solution, image_id, task_type_list, box_scale, **kwargs):
    """严格版召回率奖励（IoU>0.5且类别正确），使用bi_match优化匹配策略"""
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")

    for content, sol, im_id, task_type, box_wh in zip(contents, solution, image_id, task_type_list, box_scale):
        recall = 0.0
        correct_count = 0
        matched_pairs = []
        
        if task_type not in ['sgg', 'det']:
            rewards.append(0)
            continue
        
        try:
            gt_objs = sol['objects']
            object_match = re.search(r"<OBJECT>(.*?)</OBJECT>", content, re.DOTALL)
            preds = json.loads(object_match.group(1).strip())
            pred_objs = preds['objects']
            
            # 归一化预测框坐标
            normalized_preds = []
            for obj in pred_objs:
                obj['bbox'] = scale_box(obj['bbox'], (1.0/box_wh[0], 1.0/box_wh[1]))
                obj['id'] = refine_node_edge(obj['id'])
                normalized_preds.append(obj)
            pred_objs = normalized_preds

            # 使用bi_match进行最优匹配
            assignments = bi_match(gt_objs, pred_objs)
            
            # 统计满足条件的匹配（类别匹配且IoU>0.5）
            
            for assign in assignments:
                gt_obj = assign['groundtruth']
                pred_obj = assign['prediction']
                
                gt_id = gt_obj['id']
                gt_box = gt_obj['bbox']
                gt_category = gt_id.split('.')[0]
                
                pred_id = pred_obj['id']
                pred_box = pred_obj['bbox']
                pred_category = pred_id.split('.')[0]
                
                if (gt_category == pred_category) and (compute_iou(gt_box, pred_box) > 0.5):
                    correct_count += 1
                    matched_pairs.append(
                        f"{gt_id}(IoU:{compute_iou(gt_box, pred_box):.2f})->{pred_id}"
                    )
                elif (gt_category == pred_category) or (compute_iou(gt_box, pred_box) > 0.5):
                    correct_count += 0.5

            # 计算召回率
            recall = correct_count / max(1, len(gt_objs)) * NODE_REWARD_WEIGHT
            
        except Exception as e:
            if DEBUG_MODE:
                with open(LOG_PATH, "a") as f:
                    f.write(f"Recall计算错误: {str(e)}\n")
            recall = 0.0

        rewards.append(recall)
        
        if DEBUG_MODE:
            with open(LOG_PATH, "a") as f:
                f.write(f"----- {current_time} Strict Recall Reward {recall:.3f} -----\n")
                f.write(f"Image: {im_id} | GT总物体数: {len(gt_objs)} | 正确匹配: {correct_count}\n")
                if matched_pairs:
                    f.write("匹配详情:\n" + "\n".join(matched_pairs) + "\n")

    return rewards






class RelationWeightManager:
    """关系权重管理器"""
    
    def __init__(self):
        self.weights = self._initialize_weights()
        self.default_weight = 2.0
    
    def _calculate_weight(self, frequency_percent):
        """根据频率计算权重"""
        import math
        freq = max(frequency_percent, 0.0001)
        inverse_freq = 1.0 / freq
        log_inverse = math.log(inverse_freq)
        min_log = math.log(1.0/29.49)
        max_log = math.log(1.0/0.0015)
        normalized = (log_inverse - min_log) / (max_log - min_log)
        weight = 2.0 + normalized * 8.0
        return round(weight, 2)
    
    def _initialize_weights(self):
        """初始化谓词权重字典"""
        return {
            'on': self._calculate_weight(29.49), 'has': self._calculate_weight(17.98),
            'wearing': self._calculate_weight(11.96), 'of': self._calculate_weight(9.15),
            'in': self._calculate_weight(5.66), 'near': self._calculate_weight(4.56),
            'behind': self._calculate_weight(3.17), 'with': self._calculate_weight(3.16),
            'holding': self._calculate_weight(2.48), 'above': self._calculate_weight(1.63),
            'wears': self._calculate_weight(1.25), 'sitting on': self._calculate_weight(1.19),
            'under': self._calculate_weight(1.10), 'in front of': self._calculate_weight(0.98),
            'riding': self._calculate_weight(0.85), 'standing on': self._calculate_weight(0.58),
            'at': self._calculate_weight(0.45), 'attached to': self._calculate_weight(0.39),
            'carrying': self._calculate_weight(0.37), 'walking on': self._calculate_weight(0.34),
            'over': self._calculate_weight(0.2470), 'for': self._calculate_weight(0.2452),
            'belonging to': self._calculate_weight(0.2028), 'hanging from': self._calculate_weight(0.1939),
            'looking at': self._calculate_weight(0.1805), 'parked on': self._calculate_weight(0.1735),
            'laying on': self._calculate_weight(0.1687), 'and': self._calculate_weight(0.1560),
            'covering': self._calculate_weight(0.1408), 'eating': self._calculate_weight(0.1308),
            'between': self._calculate_weight(0.1296), 'part of': self._calculate_weight(0.1259),
            'along': self._calculate_weight(0.1256), 'covered in': self._calculate_weight(0.1215),
            'using': self._calculate_weight(0.1129), 'watching': self._calculate_weight(0.1103),
            'on back of': self._calculate_weight(0.0973), 'to': self._calculate_weight(0.0906),
            'lying on': self._calculate_weight(0.0776), 'walking in': self._calculate_weight(0.0717),
            'mounted on': self._calculate_weight(0.0661), 'against': self._calculate_weight(0.0591),
            'across': self._calculate_weight(0.0520), 'from': self._calculate_weight(0.0505),
            'growing on': self._calculate_weight(0.0501), 'painted on': self._calculate_weight(0.0457),
            'made of': self._calculate_weight(0.0301), 'playing': self._calculate_weight(0.0193),
            'says': self._calculate_weight(0.0093), 'flying in': self._calculate_weight(0.0015),
        }
    
    def get_weight(self, predicate):
        """获取谓词权重"""
        return self.weights.get(predicate, self.default_weight)
    
    def get_weights_for_triplets(self, triplets):
        """获取三元组列表中所有谓词的权重"""
        return [self.get_weight(triplet['predicate']) for triplet in triplets]







class DiversityAwareRelationReward:
    """关系奖励处理器"""
    
    def __init__(self):
        self.model = word_model

    def calculate_diversity_reward(self, candidate_triplets, gt_triplet):
        """
        计算细粒度关系奖励
        """
        n_candidates = len(candidate_triplets)
        
        if n_candidates == 0:
            return 0.0, 0.0, {"error": "无候选三元组"}
        
        # 准备所有三元组的文本表示（包括GT）
        all_triplets = [gt_triplet] + candidate_triplets
        all_texts = [
            f"{triplet['subject'].split('.')[0].lower()} {triplet['predicate']} {triplet['object'].split('.')[0].lower()}" 
            for triplet in all_triplets
        ]

        all_predicates = [triplet['predicate'] for triplet in all_triplets]
        
        # 计算所有embedding
        embeddings_trplets = self.model.encode(all_texts)
        embeddings_predicates = self.model.encode(all_predicates)
        
        # 使用model.similarity计算相似度矩阵
        similarity_matrix_tirplets = self.model.similarity(embeddings_trplets, embeddings_trplets)
        similarity_matrix_predicates = self.model.similarity(embeddings_predicates, embeddings_predicates)

        # 提取GT与候选三元组的相似度（第一行，去掉与自身的相似度）
        gt_similarities = similarity_matrix_tirplets[0][1:] * similarity_matrix_predicates[0][1:]  # 相似度列表，对应每个候选三元组

        gt_similarities = np.array(gt_similarities)
        
        if n_candidates == 1:
            # 单一候选情况
            similarity_reward = gt_similarities[0]
            
            stats = {
                "n_candidates": 1,
                "best_similarity": float(similarity_reward),
                "all_similarities": gt_similarities.tolist(),
                "case": "single_relation"
            }

            return 0.0, float(similarity_reward), stats
            
        else:
            # 多个候选情况
            # 找到最佳相似度
            best_similarity = np.max(gt_similarities) if len(gt_similarities) > 0 else 0.0
            
            stats = {
                "n_candidates": n_candidates,
                "best_similarity": float(best_similarity),
                "all_similarities": gt_similarities.tolist(),
                "case": "multiple_relations"
            }

            return float(best_similarity), stats



def stage3_edge_reward(completions, solution, image_id, task_type_list, box_scale, **kwargs):
    """简化版边奖励：移除多样性奖励机制"""
    
    # 初始化奖励处理器
    reward_processor = DiversityAwareRelationReward()
    weight_manager = RelationWeightManager()
    
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    

    for content, sol, im_id, task_type, box_wh in zip(contents, solution, image_id, task_type_list, box_scale):
        reward = 0.0
        total_possible_weight = 0.0
        total_achieved_reward = 0.0
        
        if task_type not in ['sgg', 'cls']:
            rewards.append(0)
            continue
            
        try:
            preds = json.loads(extract_answer_content(content))
            
            if task_type == 'sgg':
                gt_objs = sol['objects']
                gt_rels = sol['relations']
                pred_objs = preds['objects']
                pred_rels = preds['relations']
                
                # 预处理预测边界框
                _objs = []
                for obj in pred_objs:
                    obj['bbox'] = scale_box(obj['bbox'], (1.0 / box_wh[0], 1.0 / box_wh[1]))
                    obj['id'] = refine_node_edge(obj['id'])
                    _objs.append(obj)
                pred_objs = _objs

                # 步骤1: 物体匹配建立映射
                assignments = bi_match(gt_objs, pred_objs)
                map_obj = {}
                for assign in assignments:
                    gt_id = assign['groundtruth']['id']
                    pred_entry = assign['prediction']
                    if pred_entry is None or pred_entry.get('id') is None:
                        continue
                    pred_id = pred_entry['id']
                    map_obj[gt_id] = pred_id

                # 步骤2: 构建预测关系的快速查找结构
                pred_relations_dict = {}
                for rel in pred_rels:
                    subj_id = refine_node_edge(rel['subject'])
                    obj_id = refine_node_edge(rel['object'])
                    predicate = refine_node_edge(rel['predicate'])
                    
                    key = (subj_id, obj_id)
                    if key not in pred_relations_dict:
                        pred_relations_dict[key] = []
                    pred_relations_dict[key].append({
                        'subject': subj_id,
                        'predicate': predicate,
                        'object': obj_id
                    })

                # 步骤3: 遍历GT关系，应用简化奖励逻辑
                for gt_rel in gt_rels:
                    sub, obj = gt_rel['subject'], gt_rel['object']
                    gt_predicate = refine_node_edge(gt_rel['predicate'])
                    
                    # 构建GT三元组
                    gt_triplet = {
                        'subject': sub,
                        'predicate': gt_predicate,
                        'object': obj
                    }
                    
                    # 检查主语和宾语是否都有映射
                    if (sub not in map_obj) or (obj not in map_obj):
                        predicate_weight = weight_manager.get_weight(gt_predicate)
                        total_possible_weight += predicate_weight
                        continue
                    
                    sub_mapped = map_obj[sub]
                    obj_mapped = map_obj[obj]
                    pred_key = (sub_mapped, obj_mapped)
                    
                    # 检查预测关系中是否存在这个主语-宾语对
                    if pred_key not in pred_relations_dict:
                        predicate_weight = weight_manager.get_weight(gt_predicate)
                        total_possible_weight += predicate_weight
                        continue
                    
                    # 获取候选三元组
                    candidate_triplets = pred_relations_dict[pred_key]
                    
                    # 计算相似度奖励
                    best_similarity, _ = reward_processor.calculate_diversity_reward(
                        candidate_triplets, gt_triplet)
                    
                    predicate_weight = weight_manager.get_weight(gt_predicate)
        
                    # 只使用相似度奖励
                    accuracy_reward = best_similarity * predicate_weight
                    
                    total_achieved_reward += accuracy_reward
                    total_possible_weight += predicate_weight

                # 计算最终奖励
                if total_possible_weight > 0:
                    base_reward = total_achieved_reward / total_possible_weight
                    reward = base_reward * EDGE_REWARD_WEIGHT
                else:
                    reward = 0.0
                    
        except Exception as e:
            print(e)
            reward = 0.0

        rewards.append(reward)

    return rewards





def stage3_edge_diversity_reward(completions, solution, image_id, task_type_list, box_scale, **kwargs):
    """基于GT簇匹配的多样性奖励：评估Pred覆盖GT簇的能力"""
    
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    
    for content, sol, im_id, task_type, box_wh in zip(contents, solution, image_id, task_type_list, box_scale):
        reward = 0.0
        
        if task_type not in ['sgg', 'cls']:
            rewards.append(0.0)
            continue
            
        try:
            preds = json.loads(extract_answer_content(content))
            
            if task_type == 'sgg':
                gt_rels = sol['relations']
                pred_rels = preds['relations']
                
                # 按照指定格式构建GT和Pred三元组文本
                gt_triplets = [
                    f"{triplet['subject'].split('.')[0].lower()} {triplet['predicate']} {triplet['object'].split('.')[0].lower()}" 
                    for triplet in gt_rels
                ]
                
                pred_triplets = [
                    f"{triplet['subject'].split('.')[0].lower()} {triplet['predicate']} {triplet['object'].split('.')[0].lower()}" 
                    for triplet in pred_rels
                ]
                
                # 计算多样性奖励
                if gt_triplets and pred_triplets:
                    diversity_reward, match_stats = calculate_diversity_by_gt_cluster_matching(
                        gt_triplets, pred_triplets, CLUSTER_THRESHOLD, MIN_CLUSTER_SIZE
                    )
                    reward = diversity_reward * DIVERSITY_REWARD_WEIGHT 
                    
                    if kwargs.get('verbose', False):
                        print(f"图像 {im_id} 多样性分析:")
                        print(f"  GT三元组数: {len(gt_triplets)}, Pred三元组数: {len(pred_triplets)}")
                        print(f"  GT簇数量: {match_stats['gt_cluster_count']}")
                        print(f"  Pred覆盖簇数量: {match_stats['covered_cluster_count']}")
                        print(f"  覆盖率: {match_stats['coverage_rate']:.3f}")
                        print(f"  平均每簇匹配数: {match_stats['avg_matches_per_cluster']:.2f}")
                        print(f"  总匹配数: {match_stats['total_matches']}")
                        print(f"  多样性奖励: {diversity_reward:.3f}")
                else:
                    reward = 0.0
                    
        except Exception as e:
            print(f"多样性奖励计算错误: {e}")
            reward = 0.0

        rewards.append(reward)

    return rewards


def calculate_diversity_by_gt_cluster_matching(gt_triplets, pred_triplets, threshold=0.7, min_cluster_size=1):
    """
    基于GT簇匹配的多样性计算
    """
    if not gt_triplets or not pred_triplets:
        return 0.0, {'gt_cluster_count': 0, 'covered_cluster_count': 0, 'coverage_rate': 0}
    
    # 步骤1: 对GT进行DBSCAN聚类（固定GT语义结构）
    gt_clusters = cluster_gt_triplets(gt_triplets, threshold, min_cluster_size)
    
    if not gt_clusters:
        return 0.0, {'gt_cluster_count': 0, 'covered_cluster_count': 0, 'coverage_rate': 0}
    
    # 步骤2: Pred与GT簇进行匹配
    cluster_matches = match_predicates_to_gt_clusters(gt_clusters, pred_triplets, threshold)
    
    # 步骤3: 计算多样性奖励
    diversity_reward = calculate_coverage_based_reward(gt_clusters, cluster_matches, len(pred_triplets))
    
    # 统计信息
    covered_clusters = [cluster for cluster in gt_clusters if cluster['matched_count'] > 0]
    avg_matches = np.mean([cluster['matched_count'] for cluster in covered_clusters]) if covered_clusters else 0
    
    match_stats = {
        'gt_triplet_count': len(gt_triplets),
        'pred_triplet_count': len(pred_triplets),
        'gt_cluster_count': len(gt_clusters),
        'covered_cluster_count': len(covered_clusters),
        'coverage_rate': len(covered_clusters) / len(gt_clusters) if gt_clusters else 0,
        'avg_matches_per_cluster': avg_matches,
        'total_matches': sum(cluster['matched_count'] for cluster in gt_clusters)
    }
    
    return diversity_reward, match_stats


def cluster_gt_triplets(gt_triplets, threshold=0.7, min_cluster_size=1):
    """对GT三元组进行聚类，返回簇结构"""
    if len(gt_triplets) <= 1:
        # 单个或没有三元组，每个作为一个簇
        clusters = []
        for i, triplet in enumerate(gt_triplets):
            clusters.append({
                'cluster_id': i,
                'triplets': [triplet],
                'size': 1,
                'center': None,
                'is_valid': True,
                'is_noise': False
            })
        return clusters
    
    # 计算GT三元组的embedding
    gt_embeddings = word_model.encode(gt_triplets)
    gt_embeddings_norm = gt_embeddings / np.linalg.norm(gt_embeddings, axis=1, keepdims=True)
    
    # 计算合适的eps（基于相似度阈值）
    eps = math.sqrt(2 * (1 - threshold))
    
    try:
        # DBSCAN聚类
        dbscan = DBSCAN(eps=eps, min_samples=min_cluster_size, metric='cosine')
        gt_labels = dbscan.fit_predict(gt_embeddings_norm)
        
        clusters = []
        unique_labels = set(gt_labels)
        
        for label in unique_labels:
            if label == -1:
                # 噪声点：每个噪声点单独作为一个簇
                noise_indices = np.where(gt_labels == -1)[0]
                for idx in noise_indices:
                    cluster_emb = gt_embeddings_norm[idx]
                    clusters.append({
                        'cluster_id': f"noise_{idx}",
                        'triplets': [gt_triplets[idx]],
                        'size': 1,
                        'center': cluster_emb,
                        'is_valid': True,
                        'is_noise': True
                    })
            else:
                # 正常簇
                cluster_indices = np.where(gt_labels == label)[0]
                if len(cluster_indices) >= min_cluster_size:
                    cluster_embs = gt_embeddings_norm[cluster_indices]
                    cluster_center = np.mean(cluster_embs, axis=0)
                    cluster_center = cluster_center / np.linalg.norm(cluster_center)
                    
                    clusters.append({
                        'cluster_id': f"cluster_{label}",
                        'triplets': [gt_triplets[i] for i in cluster_indices],
                        'size': len(cluster_indices),
                        'center': cluster_center,
                        'is_valid': True,
                        'is_noise': False
                    })
        
        return clusters
        
    except Exception as e:
        print(f"GT聚类失败，使用简单聚类: {e}")
        # 回退到简单聚类：每个三元组作为一个簇
        clusters = []
        for i, triplet in enumerate(gt_triplets):
            cluster_emb = gt_embeddings_norm[i]
            clusters.append({
                'cluster_id': i,
                'triplets': [triplet],
                'size': 1,
                'center': cluster_emb,
                'is_valid': True,
                'is_noise': False
            })
        return clusters


def match_predicates_to_gt_clusters(gt_clusters, pred_triplets, threshold=0.7):
    """将Pred三元组匹配到GT簇"""
    
    if not pred_triplets:
        for cluster in gt_clusters:
            cluster['matched_preds'] = []
            cluster['matched_count'] = 0
            cluster['avg_similarity'] = 0
        return gt_clusters
    
    # 计算Pred三元组的embedding
    pred_embeddings = word_model.encode(pred_triplets)
    pred_embeddings_norm = pred_embeddings / np.linalg.norm(pred_embeddings, axis=1, keepdims=True)
    
    # 准备有效的GT簇中心
    valid_clusters = [cluster for cluster in gt_clusters if cluster['center'] is not None and cluster['is_valid']]
    
    if not valid_clusters:
        for cluster in gt_clusters:
            cluster['matched_preds'] = []
            cluster['matched_count'] = 0
            cluster['avg_similarity'] = 0
        return gt_clusters
    
    cluster_centers = [cluster['center'] for cluster in valid_clusters]
    cluster_centers_array = np.array(cluster_centers)
    
    # 计算相似度矩阵
    similarity_matrix = cosine_similarity(pred_embeddings_norm, cluster_centers_array)
    
    # 初始化匹配信息
    for cluster in gt_clusters:
        cluster['matched_preds'] = []
        cluster['matched_similarities'] = []
    
    # 为每个Pred分配最佳匹配的簇（一对多匹配）
    for pred_idx in range(len(pred_triplets)):
        best_similarity = np.max(similarity_matrix[pred_idx])
        best_cluster_idx = np.argmax(similarity_matrix[pred_idx])
        
        if best_similarity >= threshold:
            matched_cluster = valid_clusters[best_cluster_idx]
            # 找到在原始gt_clusters中的索引
            original_idx = None
            for i, cluster in enumerate(gt_clusters):
                if cluster['cluster_id'] == matched_cluster['cluster_id']:
                    original_idx = i
                    break
            
            if original_idx is not None:
                gt_clusters[original_idx]['matched_preds'].append({
                    'pred_idx': pred_idx,
                    'similarity': best_similarity,
                    'triplet': pred_triplets[pred_idx]
                })
                gt_clusters[original_idx]['matched_similarities'].append(best_similarity)
    
    # 更新匹配统计
    for cluster in gt_clusters:
        cluster['matched_count'] = len(cluster['matched_preds'])
        cluster['avg_similarity'] = np.mean(cluster['matched_similarities']) if cluster['matched_similarities'] else 0
    
    return gt_clusters


def calculate_coverage_based_reward(gt_clusters, cluster_matches, total_pred_count):
    """
    基于覆盖率的多样性奖励计算（修改为匹配的GT数量比值）
    """
    if not gt_clusters or total_pred_count == 0:
        return 0.0
    
    # 只考虑有效的GT簇（排除无效簇）
    valid_gt_clusters = [c for c in gt_clusters if c['is_valid']]
    
    if not valid_gt_clusters:
        return 0.0
    
    # 1. 覆盖率奖励：覆盖的GT簇比例
    covered_clusters = [c for c in valid_gt_clusters if c['matched_count'] > 0]
    coverage_reward = len(covered_clusters) / len(valid_gt_clusters) if valid_gt_clusters else 0
    
    # 2. 密度奖励：基于匹配到的GT数量比值
    if covered_clusters:
        # 计算被覆盖簇中的GT三元组总数
        total_gt_in_covered = sum(len(cluster['triplets']) for cluster in covered_clusters)
        
        # 计算总匹配的预测数量
        total_pred_matches = sum(cluster['matched_count'] for cluster in covered_clusters)
        
        # 密度奖励 = 总匹配预测数 / 被覆盖簇的GT总数
        if total_gt_in_covered > 0:
            density_ratio = total_pred_matches / total_gt_in_covered
            density_reward = min(1.0, density_ratio)
        else:
            density_reward = 0.0
    else:
        density_reward = 0.0
    
    # 3. 综合奖励公式（使用乘积，鼓励均衡发展）
    diversity_reward = coverage_reward * density_reward
    
    # 4. 对过度集中匹配的惩罚（保留分布均匀性检查）
    if covered_clusters and len(covered_clusters) > 1:
        match_counts = [c['matched_count'] for c in covered_clusters]
        gt_counts = [len(c['triplets']) for c in covered_clusters]
        
        # 计算相对匹配密度（避免因GT簇大小差异导致的偏差）
        relative_densities = []
        for match_count, gt_count in zip(match_counts, gt_counts):
            if gt_count > 0:
                relative_densities.append(match_count / gt_count)
        
        if relative_densities:
            # 计算相对密度的变异系数
            mean_relative_density = np.mean(relative_densities)
            if mean_relative_density > 0:
                cv_relative = np.std(relative_densities) / mean_relative_density
                if cv_relative > 0.5:  # 相对密度分布不均匀
                    imbalance_penalty = min(0.5, cv_relative * 0.1)
                    diversity_reward = max(0, diversity_reward - imbalance_penalty)
    
    return min(1.0, max(0.0, diversity_reward))












def is_valid_id_format(s):
    return bool(re.fullmatch(r"[a-zA-Z_]+\.\d+", s))

def is_valid_box(item):
    if not isinstance(item, dict):
        return False

    bbox = item.get("bbox") 
    if "id" not in item or not isinstance(bbox, list) or len(bbox) != 4:
        return False

    # id format: [str].[number]
    if not is_valid_id_format(item['id']):
        pass
        #return False

    return all(isinstance(e, (int, float)) for e in bbox)

def is_valid_predicate(item):
    if not isinstance(item, dict):
        return False

    keys = ("subject", "object", "predicate")
    if not all(k in item for k in keys):
        return False

    return all(isinstance(item[k], str) for k in keys)




def is_valid_relationship_structure(relationships):
    """
    检查关系结构是否有效
    - 有'relations'这个key得0.5分
    - 三个大类的key都存在得0.75分  
    - 检查每个key对应的值,必须为列表结构,列表可以为空
    - 若不为空,则里面应该是完整的三元组字典结构,得1分
    """
    if not isinstance(relationships, dict):
        return 0.0
    
    # 检查是否有'relations'这个key
    if 'relations' not in relationships:
        return 0.0
    
    reward = 0.5
    
    # 检查三个大类的key是否存在
    expected_keys = ['spatial_relations', 'possession_relations', 'interaction_relations']
    relationships_data = relationships['relations']
    
    if not isinstance(relationships_data, dict):
        return reward
    
    # 检查是否包含所有三个大类
    has_all_categories = all(key in relationships_data for key in expected_keys)
    if has_all_categories:
        reward = 0.75
        
        # 检查每个大类的结构
        all_valid = True
        for category_key in expected_keys:
            category_data = relationships_data[category_key]
            
            # 必须是列表结构
            if not isinstance(category_data, list):
                all_valid = False
                break
                
            # 检查列表中的每个元素（如果不为空）
            if category_data:  # 列表不为空时检查内容
                for relation in category_data:
                    if not is_valid_predicate(relation):
                        all_valid = False
                        break
            
            if not all_valid:
                break
        
        if all_valid:
            reward = 1.0
    
    return reward



def format_reward(completions, image_id, task_type_list, **kwargs):
    """
    Reward function that checks if the completion has the correct format:
    """
    stage_tags = {
        "category": r"<CATEGORY>(.*?)</CATEGORY>",
        "object": r"<OBJECT>(.*?)</OBJECT>",
        "relation": r"<RELATION>(.*?)</RELATION>"
    }

    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")

    for completion, im_id, task_type in zip(completions, image_id, task_type_list):
        content = completion[0]["content"].strip()
        reward = 0.0
    
        # 检查每个阶段的标签是否存在
        stage_rewards = {
            "category": 0.0,
            "object": 0.0,
            "relation": 0.0
        }

        # 检查CATEGORY阶段标签
        category_match = re.search(stage_tags["category"], content, re.DOTALL)
        if category_match:
            stage_rewards["category"] = 0.5
            try:
                category_json = json.loads(category_match.group(1).strip())  
                if isinstance(category_json, dict) and ("categories" in category_json):  
                    # 检查category数组结构
                    if isinstance(category_json["categories"], list):
                        category_valid = True
                        for cat_item in category_json["categories"]:
                            if not isinstance(cat_item, dict) or "id" not in cat_item:
                                category_valid = False
                                break
                        if category_valid:
                            stage_rewards["category"] = 1.0
            except:
                pass

        # 检查OBJECT阶段标签
        object_match = re.search(stage_tags["object"], content, re.DOTALL)
        if object_match:
            stage_rewards["object"] = 0.5
            try:
                object_json = json.loads(object_match.group(1).strip())  
                if isinstance(object_json, dict) and ("objects" in object_json):
                    stage_rewards["object"] = 0.75
                    graph_valid = True
                    # 检查objects数组结构
                    if isinstance(object_json["objects"], list):
                        for obj in object_json["objects"]:
                            if not is_valid_box(obj):
                                graph_valid = False
                                break
                    else:
                        graph_valid = False
                    
                    if graph_valid:
                        stage_rewards["object"] = 1.0  
            except:
                pass

        # 检查RELATIONSHIP阶段标签
        relation_match = re.search(stage_tags["relation"], content, re.DOTALL)
        if relation_match:
            try:
                relation_json = json.loads(relation_match.group(1).strip())
                if isinstance(relation_json, dict):
                    # 使用新的关系结构检查函数
                    relation_score = is_valid_relationship_structure(relation_json)
                    stage_rewards["relation"] = relation_score
            except:
                pass
        
        stage_tag_reward = sum(stage_rewards.values()) 
        
        reward = stage_tag_reward * FORMAT_REWARD_WEIGHT
        rewards.append(reward)

    return rewards


# Reward functions registry with only stage rewards and format reward
reward_funcs_registry_vg = {
    # Format checking reward (must keep)
    "format_reward": format_reward,
    
    # Stage 1 rewards 
    "stage1_category_reward": stage1_category_reward,
    
    # Stage 2 rewards
    "stage2_node_box_reward": stage2_node_box_reward,
    "stage2_node_recall_reward": stage2_node_recall_reward,
    
    # Stage 3 rewards (Relations)
    "stage3_edge_fine_reward": stage3_edge_reward,

    "stage3_edge_coarse_reward": stage3_edge_diversity_reward
}