import json
import math
import os
import re
from typing import Dict, List

import numpy as np
import yaml
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity

from src.rl.utils.reward_tools import BipartiteMatcher, box_L1, compute_iou, refine_node_edge, scale_box
from src.rl.utils.sentence_bert import SemanticSimilarityManager


FORMAT_MISSING_TAG_SCORE = 0.0
FORMAT_INVALID_JSON_SCORE = 0.5
FORMAT_INVALID_SCHEMA_SCORE = 0.5
FORMAT_INVALID_ITEM_SCORE = 0.75
FORMAT_VALID_STAGE_SCORE = 1.0

DEFAULT_REWARD_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "configs", "reward_config.yaml")

DATASET_TYPE = "vg"
DEBUG_MODE = None
LOG_PATH = None
STRICT_FORMAT = None
SEM_WEIGHT = None
IOU_WEIGHT = None
BOX_L1_WEIGHT = None
CLUSTER_THRESHOLD = None
MIN_CLUSTER_SIZE = None
BASE_WEIGHT = None
INCREMENTAL_WEIGHT = None

sentence_bert = SemanticSimilarityManager(dataset_type=DATASET_TYPE)
category_semantic_similarity = sentence_bert.category_semantic_similarity
word_model = sentence_bert.word_model
matcher = BipartiteMatcher(
    category_semantic_similarity,
    sem_weight=1,
    iou_weight=1,
    box_l1_weight=1,
)
bi_match = matcher.bi_match


def load_reward_config(config_path=None):
    resolved_path = config_path or DEFAULT_REWARD_CONFIG_PATH
    if not os.path.isabs(resolved_path):
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        candidate = os.path.join(project_root, resolved_path)
        if os.path.exists(candidate):
            resolved_path = candidate

    with open(resolved_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def apply_reward_config(config):
    global DEBUG_MODE, LOG_PATH, STRICT_FORMAT
    global SEM_WEIGHT, IOU_WEIGHT, BOX_L1_WEIGHT
    global CLUSTER_THRESHOLD, MIN_CLUSTER_SIZE
    global BASE_WEIGHT, INCREMENTAL_WEIGHT
    global matcher, bi_match

    debug_config = config.get("debug", {})
    matching_config = config.get("matching", {})
    coarse_config = config.get("coarse_reward", {})
    relation_config = config.get("relation_weight", {})

    DEBUG_MODE = bool(debug_config["enabled"])
    LOG_PATH = debug_config["log_path"]
    STRICT_FORMAT = bool(debug_config["strict_format"])
    SEM_WEIGHT = float(matching_config["semantic_weight"])
    IOU_WEIGHT = float(matching_config["iou_weight"])
    BOX_L1_WEIGHT = float(matching_config["box_l1_weight"])
    CLUSTER_THRESHOLD = float(coarse_config["cluster_threshold"])
    MIN_CLUSTER_SIZE = int(coarse_config["min_cluster_size"])
    BASE_WEIGHT = float(relation_config["base_weight"])
    INCREMENTAL_WEIGHT = float(relation_config["incremental_weight"])

    matcher = BipartiteMatcher(
        category_semantic_similarity,
        sem_weight=SEM_WEIGHT,
        iou_weight=IOU_WEIGHT,
        box_l1_weight=BOX_L1_WEIGHT,
    )
    bi_match = matcher.bi_match


def _log_debug(message):
    if DEBUG_MODE:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(message + "\n")


def extract_answer_content(text: str) -> str:
    """Extract category, object, and relation items from model output."""
    text = text.replace("```", " ").replace("json", " ").strip()
    result = {"categories": [], "objects": [], "relations": []}

    category_matches = re.findall(r'\{\s*"id"\s*:\s*"[^"]*"\s*\}', text, re.DOTALL)
    for cat_str in category_matches:
        try:
            cat = json.loads(cat_str)
            if isinstance(cat, dict) and "id" in cat:
                result["categories"].append(cat)
        except Exception:
            continue

    object_matches = re.findall(r'\{\s*"id"\s*:\s*"[^"]*"\s*,\s*"bbox"\s*:\s*\[[^\]]*\]\s*\}', text, re.DOTALL)
    for obj_str in object_matches:
        try:
            obj = json.loads(obj_str)
            if isinstance(obj, dict) and "id" in obj and "bbox" in obj:
                result["objects"].append(obj)
        except Exception:
            continue

    rel_pattern = r'\{\s*"subject"\s*:\s*"([^"]+\.\d{1,2})"\s*,\s*"predicate"\s*:\s*"([^"]+)"\s*,\s*"object"\s*:\s*"([^"]+\.\d{1,2})"\s*\}'
    for rel_match in re.finditer(rel_pattern, text, re.DOTALL):
        result["relations"].append(
            {
                "subject": rel_match.group(1),
                "predicate": rel_match.group(2),
                "object": rel_match.group(3),
            }
        )

    return json.dumps(result, ensure_ascii=False)


def is_valid_bbox(bbox: List) -> bool:
    return isinstance(bbox, list) and len(bbox) == 4 and all(isinstance(e, (int, float)) for e in bbox)


def is_valid_object_item(obj: Dict) -> bool:
    return isinstance(obj, dict) and "id" in obj and "bbox" in obj and is_valid_bbox(obj["bbox"])


def is_valid_relation_item(rel: Dict) -> bool:
    required = ("subject", "predicate", "object")
    return isinstance(rel, dict) and all(k in rel for k in required) and all(isinstance(rel[k], str) for k in required)


def _parse_tagged_json(content, tag):
    match = re.search(rf"<{tag}>(.*?)</{tag}>", content, re.DOTALL)
    if not match:
        return None, "missing"
    try:
        return json.loads(match.group(1).strip()), None
    except Exception:
        return None, "json"


def check_category_stage(content: str) -> float:
    data, error = _parse_tagged_json(content, "CATEGORY")
    if error == "missing":
        return FORMAT_MISSING_TAG_SCORE
    if error == "json":
        return FORMAT_INVALID_JSON_SCORE
    if not isinstance(data, dict) or not isinstance(data.get("categories"), list):
        return FORMAT_INVALID_SCHEMA_SCORE
    for cat in data["categories"]:
        if not isinstance(cat, dict) or "id" not in cat:
            return FORMAT_INVALID_ITEM_SCORE
    return FORMAT_VALID_STAGE_SCORE


def check_object_stage(content: str) -> float:
    data, error = _parse_tagged_json(content, "OBJECT")
    if error == "missing":
        return FORMAT_MISSING_TAG_SCORE
    if error == "json":
        return FORMAT_INVALID_JSON_SCORE
    if not isinstance(data, dict) or not isinstance(data.get("objects"), list):
        return FORMAT_INVALID_SCHEMA_SCORE
    for obj in data["objects"]:
        if not is_valid_object_item(obj):
            return FORMAT_INVALID_ITEM_SCORE
    return FORMAT_VALID_STAGE_SCORE


def check_relation_stage(content: str, dataset_type: str = DATASET_TYPE) -> float:
    data, error = _parse_tagged_json(content, "RELATION")
    if error == "missing":
        return FORMAT_MISSING_TAG_SCORE
    if error == "json":
        return FORMAT_INVALID_JSON_SCORE
    if not isinstance(data, dict) or not isinstance(data.get("relations"), dict):
        return FORMAT_INVALID_SCHEMA_SCORE

    if dataset_type.lower() == "psg":
        required_keys = ["spatial_relations", "static_action_relations", "dynamic_action_relations"]
    else:
        required_keys = ["spatial_relations", "possession_relations", "interaction_relations"]

    relations = data["relations"]
    if not all(key in relations for key in required_keys):
        return FORMAT_INVALID_SCHEMA_SCORE
    for key in required_keys:
        if not isinstance(relations[key], list):
            return FORMAT_INVALID_ITEM_SCORE
        for rel in relations[key]:
            if not is_valid_relation_item(rel):
                return FORMAT_INVALID_ITEM_SCORE
    return FORMAT_VALID_STAGE_SCORE


def format_reward(completions: List, image_id: List, **kwargs) -> List[float]:
    rewards = []
    for completion, _ in zip(completions, image_id):
        content = completion[0]["content"].strip()
        total_score = check_category_stage(content) + check_object_stage(content) + check_relation_stage(content)
        rewards.append(total_score)
    return rewards


def stage1_category_reward(completions, solution, **kwargs):
    """Reward category-level F1 between predicted and ground-truth categories."""
    contents = [completion[0]["content"] for completion in completions]
    rewards = []

    for content, sol in zip(contents, solution):
        try:
            parsed_result = json.loads(extract_answer_content(content))
            category_match = parsed_result.get("categories", [])
            gt_categories = set(sol["categories"])
            pred_categories_list = [obj["id"].split(".")[0].lower() for obj in category_match]
            pred_categories = set(pred_categories_list)

            if len(pred_categories_list) > len(pred_categories):
                rewards.append(0.0)
                continue

            true_positives = len(gt_categories & pred_categories)
            false_positives = len(pred_categories - gt_categories)
            false_negatives = len(gt_categories - pred_categories)
            precision = true_positives / max(1, true_positives + false_positives)
            recall = true_positives / max(1, true_positives + false_negatives)
            f1_score = 2 * precision * recall / max(1e-9, precision + recall)
            rewards.append(f1_score)
        except Exception as e:
            _log_debug(f"Stage1 Error: {e}")
            rewards.append(0.0)

    return rewards


def stage2_node_reward(completions, solution, image_id, box_scale, **kwargs):
    """Reward object localization and category recall."""
    contents = [completion[0]["content"] for completion in completions]
    rewards = []

    for content, sol, _, box_wh in zip(contents, solution, image_id, box_scale):
        try:
            gt_objs = sol["objects"]
            parsed_result = json.loads(extract_answer_content(content))
            pred_objs = parsed_result.get("objects", [])

            normalized_preds = []
            for obj in pred_objs:
                obj["bbox"] = scale_box(obj.get("bbox", [0, 0, 0, 0]), (1.0 / box_wh[0], 1.0 / box_wh[1]))
                obj["id"] = refine_node_edge(obj["id"])
                normalized_preds.append(obj)

            if not gt_objs or not normalized_preds:
                rewards.append(0.0)
                continue

            assignments = bi_match(gt_objs, normalized_preds)
            box_reward = 0.0
            correct_count = 0.0

            for assign in assignments:
                gt = assign["groundtruth"]
                pred = assign["prediction"]
                if not pred or not pred.get("id"):
                    continue

                iou = compute_iou(gt["bbox"], pred["bbox"])
                l1 = box_L1(gt["bbox"], pred["bbox"])
                box_score = (iou * IOU_WEIGHT + np.exp(-l1) * BOX_L1_WEIGHT) / (IOU_WEIGHT + BOX_L1_WEIGHT)
                box_reward += box_score

                gt_category = refine_node_edge(gt["id"].split(".")[0])
                pred_category = refine_node_edge(pred["id"].split(".")[0])
                if gt_category == pred_category and iou >= 0.5:
                    correct_count += 1.0
                elif gt_category == pred_category or iou >= 0.5:
                    correct_count += 0.5

            box_reward /= max(1, len(gt_objs))
            recall_reward = correct_count / max(1, len(gt_objs))
            rewards.append(box_reward + recall_reward)
        except Exception as e:
            _log_debug(f"Stage2 Error: {e}")
            rewards.append(0.0)

    return rewards


class RelationWeightManager:
    """Dataset-aware relation weight manager."""

    VG_FREQUENCY = {
        "on": 29.49,
        "has": 17.98,
        "wearing": 11.96,
        "of": 9.15,
        "in": 5.66,
        "near": 4.56,
        "behind": 3.17,
        "with": 3.16,
        "holding": 2.48,
        "above": 1.63,
        "wears": 1.25,
        "sitting on": 1.19,
        "under": 1.10,
        "in front of": 0.98,
        "riding": 0.85,
        "standing on": 0.58,
        "at": 0.45,
        "attached to": 0.39,
        "carrying": 0.37,
        "walking on": 0.34,
        "over": 0.2470,
        "for": 0.2452,
        "belonging to": 0.2028,
        "hanging from": 0.1939,
        "looking at": 0.1805,
        "parked on": 0.1735,
        "laying on": 0.1687,
        "and": 0.1560,
        "covering": 0.1408,
        "eating": 0.1308,
        "between": 0.1296,
        "part of": 0.1259,
        "along": 0.1256,
        "covered in": 0.1215,
        "using": 0.1129,
        "watching": 0.1103,
        "on back of": 0.0973,
        "to": 0.0906,
        "lying on": 0.0776,
        "walking in": 0.0717,
        "mounted on": 0.0661,
        "against": 0.0591,
        "across": 0.0520,
        "from": 0.0505,
        "growing on": 0.0501,
        "painted on": 0.0457,
        "made of": 0.0301,
        "playing": 0.0193,
        "says": 0.0093,
        "flying in": 0.0015,
    }

    PSG_FREQUENCY = {
        "on": 20.245,
        "beside": 17.210,
        "over": 15.031,
        "attached to": 7.828,
        "standing on": 7.036,
        "in front of": 4.145,
        "holding": 3.901,
        "in": 3.755,
        "walking on": 2.723,
        "parked on": 2.531,
        "driving on": 2.144,
        "sitting on": 2.021,
        "looking at": 1.948,
        "hanging from": 1.475,
        "wearing": 1.094,
        "carrying": 0.881,
        "riding": 0.762,
        "playing": 0.713,
        "lying on": 0.561,
        "eating": 0.475,
        "running on": 0.436,
        "touching": 0.361,
        "flying over": 0.321,
        "leaning on": 0.307,
        "swinging": 0.278,
        "driving": 0.225,
        "about to hit": 0.215,
        "enclosing": 0.195,
        "talking to": 0.152,
        "playing with": 0.130,
        "pulling": 0.120,
        "painted on": 0.072,
        "crossing": 0.071,
        "throwing": 0.068,
        "jumping from": 0.062,
        "biting": 0.059,
        "slicing": 0.051,
        "catching": 0.050,
        "on back of": 0.047,
        "drinking": 0.044,
        "going down": 0.042,
        "guiding": 0.032,
        "jumping over": 0.028,
        "kicking": 0.028,
        "pushing": 0.026,
        "feeding": 0.024,
        "entering": 0.018,
        "chasing": 0.014,
        "exiting": 0.010,
        "cooking": 0.007,
        "kissing": 0.006,
        "cleaning": 0.006,
        "climbing": 0.004,
        "picking": 0.003,
        "falling off": 0.003,
        "opening": 0.003,
    }

    def __init__(self, dataset_type: str = "vg"):
        self.dataset_type = dataset_type.lower()
        if self.dataset_type not in ["vg", "psg"]:
            raise ValueError(f"Unsupported dataset_type: {dataset_type}")
        self.default_weight = BASE_WEIGHT
        self.weights = self._initialize_weights()

    def _calculate_weight(self, frequency_percent: float, min_frequency: float, max_frequency: float) -> float:
        freq = max(frequency_percent, 0.0001)
        log_inverse = math.log(1.0 / freq)
        min_log = math.log(1.0 / max_frequency)
        max_log = math.log(1.0 / min_frequency)
        normalized = (log_inverse - min_log) / max(max_log - min_log, 1e-9)
        weight = BASE_WEIGHT + normalized * INCREMENTAL_WEIGHT
        return round(weight, 2)

    def _initialize_weights(self) -> Dict[str, float]:
        if self.dataset_type == "vg":
            frequency_data = self.VG_FREQUENCY
        else:
            frequency_data = self.PSG_FREQUENCY

        min_frequency = min(frequency_data.values())
        max_frequency = max(frequency_data.values())
        return {
            predicate: self._calculate_weight(freq, min_frequency, max_frequency)
            for predicate, freq in frequency_data.items()
        }

    def get_weight(self, predicate: str) -> float:
        return self.weights.get(predicate, self.default_weight)


class FineGrainedReward:
    """Semantic similarity reward for relation triplets."""

    def __init__(self):
        self.model = word_model

    def calculate_fine_grained_reward(self, candidate_triplets, gt_triplet):
        if not candidate_triplets:
            return 0.0, {"error": "no_candidates"}

        all_triplets = [gt_triplet] + candidate_triplets
        all_texts = [
            f"{triplet['subject'].split('.')[0].lower()} {triplet['predicate']} {triplet['object'].split('.')[0].lower()}"
            for triplet in all_triplets
        ]
        all_predicates = [triplet["predicate"] for triplet in all_triplets]

        triplet_embeddings = self.model.encode(all_texts)
        predicate_embeddings = self.model.encode(all_predicates)
        triplet_similarity = self.model.similarity(triplet_embeddings, triplet_embeddings)
        predicate_similarity = self.model.similarity(predicate_embeddings, predicate_embeddings)
        gt_similarities = np.array(triplet_similarity[0][1:] * predicate_similarity[0][1:])
        best_similarity = float(np.max(gt_similarities))

        stats = {
            "n_candidates": len(candidate_triplets),
            "best_similarity": best_similarity,
            "all_similarities": gt_similarities.tolist(),
        }
        return best_similarity, stats


class CoarseGrainedReward:
    """Coverage reward over semantically clustered ground-truth triplets."""

    def __init__(self, threshold=0.7, min_cluster_size=1):
        self.threshold = threshold
        self.min_cluster_size = min_cluster_size

    def calculate_coarse_grained_reward(self, gt_triplets, pred_triplets):
        if not gt_triplets or not pred_triplets:
            return 0.0

        gt_clusters = self._cluster_gt_triplets(gt_triplets)
        if not gt_clusters:
            return 0.0

        cluster_matches = self._match_predicates_to_gt_clusters(gt_clusters, pred_triplets)
        diversity_reward = self._calculate_coverage_based_reward(gt_clusters, cluster_matches, len(pred_triplets))
        return min(1.0, max(0.0, diversity_reward))

    def _cluster_gt_triplets(self, gt_triplets):
        if len(gt_triplets) <= 1:
            return [
                {
                    "cluster_id": idx,
                    "triplets": [triplet],
                    "size": 1,
                    "center": None,
                    "is_valid": True,
                    "is_noise": False,
                }
                for idx, triplet in enumerate(gt_triplets)
            ]

        gt_embeddings = word_model.encode(gt_triplets)
        gt_embeddings_norm = gt_embeddings / np.linalg.norm(gt_embeddings, axis=1, keepdims=True)
        eps = math.sqrt(2 * (1 - self.threshold))

        try:
            dbscan = DBSCAN(eps=eps, min_samples=self.min_cluster_size, metric="cosine")
            gt_labels = dbscan.fit_predict(gt_embeddings_norm)
        except Exception:
            gt_labels = np.arange(len(gt_triplets))

        clusters = []
        for label in set(gt_labels):
            if label == -1:
                indices = np.where(gt_labels == -1)[0]
                for idx in indices:
                    clusters.append(
                        {
                            "cluster_id": f"noise_{idx}",
                            "triplets": [gt_triplets[idx]],
                            "size": 1,
                            "center": gt_embeddings_norm[idx],
                            "is_valid": True,
                            "is_noise": True,
                        }
                    )
                continue

            indices = np.where(gt_labels == label)[0]
            if len(indices) >= self.min_cluster_size:
                center = np.mean(gt_embeddings_norm[indices], axis=0)
                center = center / np.linalg.norm(center)
                clusters.append(
                    {
                        "cluster_id": f"cluster_{label}",
                        "triplets": [gt_triplets[i] for i in indices],
                        "size": len(indices),
                        "center": center,
                        "is_valid": True,
                        "is_noise": False,
                    }
                )

        return clusters

    def _match_predicates_to_gt_clusters(self, gt_clusters, pred_triplets):
        for cluster in gt_clusters:
            cluster["matched_preds"] = []
            cluster["matched_similarities"] = []
            cluster["matched_count"] = 0
            cluster["avg_similarity"] = 0

        valid_clusters = [cluster for cluster in gt_clusters if cluster["center"] is not None and cluster["is_valid"]]
        if not pred_triplets or not valid_clusters:
            return gt_clusters

        pred_embeddings = word_model.encode(pred_triplets)
        pred_embeddings_norm = pred_embeddings / np.linalg.norm(pred_embeddings, axis=1, keepdims=True)
        cluster_centers = np.array([cluster["center"] for cluster in valid_clusters])
        similarity_matrix = cosine_similarity(pred_embeddings_norm, cluster_centers)

        for pred_idx in range(len(pred_triplets)):
            best_similarity = np.max(similarity_matrix[pred_idx])
            best_cluster_idx = np.argmax(similarity_matrix[pred_idx])
            if best_similarity < self.threshold:
                continue

            matched_cluster = valid_clusters[best_cluster_idx]
            for cluster in gt_clusters:
                if cluster["cluster_id"] == matched_cluster["cluster_id"]:
                    cluster["matched_preds"].append(
                        {
                            "pred_idx": pred_idx,
                            "similarity": best_similarity,
                            "triplet": pred_triplets[pred_idx],
                        }
                    )
                    cluster["matched_similarities"].append(best_similarity)
                    break

        for cluster in gt_clusters:
            cluster["matched_count"] = len(cluster["matched_preds"])
            if cluster["matched_similarities"]:
                cluster["avg_similarity"] = np.mean(cluster["matched_similarities"])

        return gt_clusters

    def _calculate_coverage_based_reward(self, gt_clusters, cluster_matches, total_pred_count):
        if not gt_clusters or total_pred_count == 0:
            return 0.0

        valid_gt_clusters = [cluster for cluster in gt_clusters if cluster["is_valid"]]
        if not valid_gt_clusters:
            return 0.0

        covered_clusters = [cluster for cluster in valid_gt_clusters if cluster["matched_count"] > 0]
        coverage_reward = len(covered_clusters) / len(valid_gt_clusters)
        if not covered_clusters:
            return 0.0

        total_gt_in_covered = sum(len(cluster["triplets"]) for cluster in covered_clusters)
        total_pred_matches = sum(cluster["matched_count"] for cluster in covered_clusters)
        density_reward = min(1.0, total_pred_matches / max(1, total_gt_in_covered))
        diversity_reward = coverage_reward * density_reward

        if len(covered_clusters) > 1:
            relative_densities = [
                cluster["matched_count"] / len(cluster["triplets"])
                for cluster in covered_clusters
                if cluster["triplets"]
            ]
            if relative_densities:
                mean_density = np.mean(relative_densities)
                if mean_density > 0:
                    density_cv = np.std(relative_densities) / mean_density
                    if density_cv > 0.5:
                        diversity_reward = max(0.0, diversity_reward - min(0.5, density_cv * 0.1))

        return min(1.0, max(0.0, diversity_reward))


def stage3_edge_reward(completions, solution, image_id, box_scale, **kwargs):
    """Reward relation correctness and semantic diversity."""
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    fine_reward_processor = FineGrainedReward()
    coarse_reward_processor = CoarseGrainedReward(CLUSTER_THRESHOLD, MIN_CLUSTER_SIZE)
    weight_manager = RelationWeightManager(DATASET_TYPE)

    for content, sol, _, box_wh in zip(contents, solution, image_id, box_scale):
        try:
            preds = json.loads(extract_answer_content(content))
            gt_objs = sol["objects"]
            gt_rels = sol["relations"]
            pred_objs = preds.get("objects", [])
            pred_rels = preds.get("relations", [])

            for obj in pred_objs:
                obj["bbox"] = scale_box(obj.get("bbox", [0, 0, 0, 0]), (1.0 / box_wh[0], 1.0 / box_wh[1]))
                obj["id"] = refine_node_edge(obj["id"])

            assignments = bi_match(gt_objs, pred_objs)
            obj_map = {}
            for assign in assignments:
                gt_id = assign["groundtruth"]["id"]
                pred_entry = assign["prediction"]
                if pred_entry and pred_entry.get("id"):
                    obj_map[gt_id] = pred_entry["id"]

            pred_relations_dict = {}
            for rel in pred_rels:
                subj = refine_node_edge(rel["subject"])
                obj = refine_node_edge(rel["object"])
                predicate = refine_node_edge(rel["predicate"])
                pred_relations_dict.setdefault((subj, obj), []).append(
                    {"subject": subj, "predicate": predicate, "object": obj}
                )

            total_weight = 0.0
            achieved_reward = 0.0
            for gt_rel in gt_rels:
                subj = refine_node_edge(gt_rel["subject"])
                obj = refine_node_edge(gt_rel["object"])
                predicate = refine_node_edge(gt_rel["predicate"])
                weight = weight_manager.get_weight(predicate)
                total_weight += weight

                if subj not in obj_map or obj not in obj_map:
                    continue

                key = (obj_map[subj], obj_map[obj])
                if key not in pred_relations_dict:
                    continue

                gt_triplet = {"subject": subj, "predicate": predicate, "object": obj}
                best_similarity, _ = fine_reward_processor.calculate_fine_grained_reward(
                    pred_relations_dict[key],
                    gt_triplet,
                )
                achieved_reward += best_similarity * weight

            fine_reward = achieved_reward / total_weight if total_weight > 0 else 0.0
            coarse_reward = 0.0
            if gt_rels and pred_rels:
                gt_triplets = [
                    f"{r['subject'].split('.')[0].lower()} {r['predicate']} {r['object'].split('.')[0].lower()}"
                    for r in gt_rels
                ]
                pred_triplets = [
                    f"{r['subject'].split('.')[0].lower()} {r['predicate']} {r['object'].split('.')[0].lower()}"
                    for r in pred_rels
                ]
                coarse_reward = coarse_reward_processor.calculate_coarse_grained_reward(gt_triplets, pred_triplets)

            rewards.append(fine_reward + coarse_reward)
        except Exception as e:
            _log_debug(f"Stage3 Error: {e}")
            rewards.append(0.0)

    return rewards


def reward_funcs_registry(reward_config_path=None):
    """Return reward functions configured from the reward YAML file."""
    global DATASET_TYPE, REWARD_CONFIG, sentence_bert, category_semantic_similarity, word_model
    REWARD_CONFIG = load_reward_config(reward_config_path or os.getenv("REWARD_CONFIG_PATH"))
    DATASET_TYPE = REWARD_CONFIG["dataset"]["type"].lower()
    if DATASET_TYPE not in ["vg", "psg"]:
        raise ValueError(f"Unsupported reward dataset type: {DATASET_TYPE}")

    sentence_bert = SemanticSimilarityManager(dataset_type=DATASET_TYPE)
    category_semantic_similarity = sentence_bert.category_semantic_similarity
    word_model = sentence_bert.word_model
    apply_reward_config(REWARD_CONFIG)
    return [format_reward, stage1_category_reward, stage2_node_reward, stage3_edge_reward]


