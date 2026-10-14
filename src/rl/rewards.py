import json
import math
import os
import re
from typing import Dict, List

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity

from src.rl.utils.reward_tools import BipartiteMatcher, box_L1, compute_iou, refine_node_edge, scale_box
from src.rl.reward_config import load_relation_frequencies, load_reward_config
from src.rl.utils.sentence_bert import SemanticSimilarityManager


CONFIG = load_reward_config()
word_model = None
bi_match = None
relation_frequencies = None


def apply_reward_config(config):
    """Initialize the shared semantic model and matcher once per registry setup."""
    global CONFIG, word_model, bi_match, relation_frequencies
    frequencies = load_relation_frequencies(config)
    semantic = SemanticSimilarityManager(
        dataset_type=config["dataset"]["type"],
        model_name_or_path=config["semantic_model"]["name_or_path"],
    )
    matching = config["matching"]
    matcher = BipartiteMatcher(
        semantic.category_semantic_similarity,
        sem_weight=matching["semantic_weight"],
        iou_weight=matching["iou_weight"],
        box_l1_weight=matching["box_l1_weight"],
        padding_cost=matching["padding_cost"],
        cache_size=matching["cache_size"],
    )
    CONFIG = config
    word_model = semantic.word_model
    bi_match = matcher.bi_match
    relation_frequencies = frequencies


def _log_debug(message):
    if CONFIG["debug"]["enabled"]:
        log_path = CONFIG["debug"]["log_path"]
        os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as f:
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
        return CONFIG["format_reward"]["scores"]["missing_tag"]
    if error == "json":
        return CONFIG["format_reward"]["scores"]["invalid_json"]
    if not isinstance(data, dict) or not isinstance(data.get("categories"), list):
        return CONFIG["format_reward"]["scores"]["invalid_schema"]
    for cat in data["categories"]:
        if not isinstance(cat, dict) or "id" not in cat:
            return CONFIG["format_reward"]["scores"]["invalid_item"]
    return CONFIG["format_reward"]["scores"]["valid_stage"]


def check_object_stage(content: str) -> float:
    data, error = _parse_tagged_json(content, "OBJECT")
    if error == "missing":
        return CONFIG["format_reward"]["scores"]["missing_tag"]
    if error == "json":
        return CONFIG["format_reward"]["scores"]["invalid_json"]
    if not isinstance(data, dict) or not isinstance(data.get("objects"), list):
        return CONFIG["format_reward"]["scores"]["invalid_schema"]
    for obj in data["objects"]:
        if not is_valid_object_item(obj):
            return CONFIG["format_reward"]["scores"]["invalid_item"]
    return CONFIG["format_reward"]["scores"]["valid_stage"]


def check_relation_stage(content: str, dataset_type: str = None) -> float:
    dataset_type = dataset_type or CONFIG["dataset"]["type"]
    data, error = _parse_tagged_json(content, "RELATION")
    if error == "missing":
        return CONFIG["format_reward"]["scores"]["missing_tag"]
    if error == "json":
        return CONFIG["format_reward"]["scores"]["invalid_json"]
    if not isinstance(data, dict) or not isinstance(data.get("relations"), dict):
        return CONFIG["format_reward"]["scores"]["invalid_schema"]

    if dataset_type.lower() == "psg":
        required_keys = ["spatial_relations", "static_action_relations", "dynamic_action_relations"]
    else:
        required_keys = ["spatial_relations", "possession_relations", "interaction_relations"]

    relations = data["relations"]
    if not all(key in relations for key in required_keys):
        return CONFIG["format_reward"]["scores"]["invalid_schema"]
    for key in required_keys:
        if not isinstance(relations[key], list):
            return CONFIG["format_reward"]["scores"]["invalid_item"]
        for rel in relations[key]:
            if not is_valid_relation_item(rel):
                return CONFIG["format_reward"]["scores"]["invalid_item"]
    return CONFIG["format_reward"]["scores"]["valid_stage"]


def format_reward(completions: List, image_id: List, **kwargs) -> List[float]:
    rewards = []
    for completion, _ in zip(completions, image_id):
        content = completion[0]["content"].strip()
        weights = CONFIG["format_reward"]["stage_weights"]
        total_score = (
            weights["category"] * check_category_stage(content)
            + weights["object"] * check_object_stage(content)
            + weights["relation"] * check_relation_stage(content)
        )
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
                rewards.append(CONFIG["stage1_category"]["duplicate_score"])
                continue

            true_positives = len(gt_categories & pred_categories)
            false_positives = len(pred_categories - gt_categories)
            false_negatives = len(gt_categories - pred_categories)
            precision = true_positives / max(1, true_positives + false_positives)
            recall = true_positives / max(1, true_positives + false_negatives)
            f1_score = 2 * precision * recall / max(CONFIG["numerics"]["epsilon"], precision + recall)
            rewards.append(f1_score)
        except Exception as e:
            _log_debug(f"Stage1 Error: {e}")
            rewards.append(CONFIG["stage1_category"]["error_score"])

    return rewards


def stage2_node_reward(completions, solution, image_id, box_scale, **kwargs):
    """Reward object localization and category recall."""
    config = CONFIG["stage2_node"]
    box_config = config["box"]
    recall_config = config["recall"]
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
                rewards.append(config["empty_score"])
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
                box_score = (
                    iou * box_config["iou_weight"]
                    + np.exp(-box_config["l1_decay"] * l1) * box_config["l1_weight"]
                ) / (box_config["iou_weight"] + box_config["l1_weight"])
                box_reward += box_score

                gt_category = refine_node_edge(gt["id"].split(".")[0])
                pred_category = refine_node_edge(pred["id"].split(".")[0])
                if gt_category == pred_category and iou >= recall_config["iou_threshold"]:
                    correct_count += recall_config["full_match_score"]
                elif gt_category == pred_category or iou >= recall_config["iou_threshold"]:
                    correct_count += recall_config["partial_match_score"]

            box_reward /= max(1, len(gt_objs))
            recall_reward = correct_count / max(1, len(gt_objs))
            rewards.append(config["box_weight"] * box_reward + config["recall_weight"] * recall_reward)
        except Exception as e:
            _log_debug(f"Stage2 Error: {e}")
            rewards.append(config["error_score"])

    return rewards


class RelationWeightManager:
    """Dataset-aware relation weight manager."""

    def __init__(self):
        self.config = CONFIG["stage3_edge"]["relation_weight"]
        self.default_weight = self.config["default_weight"]
        self.weights = self._initialize_weights()

    def _calculate_weight(self, frequency_percent: float, min_frequency: float, max_frequency: float) -> float:
        freq = max(frequency_percent, self.config["frequency_floor"])
        log_inverse = math.log(1.0 / freq)
        min_log = math.log(1.0 / max_frequency)
        max_log = math.log(1.0 / min_frequency)
        normalized = (log_inverse - min_log) / max(max_log - min_log, CONFIG["numerics"]["epsilon"])
        weight = self.config["base_weight"] + normalized * self.config["incremental_weight"]
        return round(weight, self.config["rounding_digits"])

    def _initialize_weights(self) -> Dict[str, float]:
        frequency_data = relation_frequencies

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

    def __init__(self):
        self.config = CONFIG["stage3_edge"]["coarse"]
        self.min_cluster_size = self.config["min_cluster_size"]

    def calculate_coarse_grained_reward(self, gt_triplets, pred_triplets):
        if not gt_triplets or not pred_triplets:
            return 0.0

        gt_clusters = self._cluster_gt_triplets(gt_triplets)
        if not gt_clusters:
            return 0.0

        cluster_matches = self._match_predicates_to_gt_clusters(gt_clusters, pred_triplets)
        diversity_reward = self._calculate_coverage_based_reward(gt_clusters, cluster_matches, len(pred_triplets))
        return min(self.config["reward_cap"], max(0.0, diversity_reward))

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
        eps = math.sqrt(2 * (1 - self.config["cluster_threshold"]))

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
            if best_similarity < self.config["match_threshold"]:
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
        density_reward = min(self.config["density_cap"], total_pred_matches / max(1, total_gt_in_covered))
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
                    penalty = self.config["imbalance_penalty"]
                    if density_cv > penalty["cv_threshold"]:
                        diversity_reward = max(
                            0.0, diversity_reward - min(penalty["max_penalty"], density_cv * penalty["coefficient"])
                        )

        return min(self.config["reward_cap"], max(0.0, diversity_reward))


def stage3_edge_reward(completions, solution, image_id, box_scale, **kwargs):
    """Reward relation correctness and semantic diversity."""
    config = CONFIG["stage3_edge"]
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    fine_reward_processor = FineGrainedReward()
    coarse_reward_processor = CoarseGrainedReward()
    weight_manager = RelationWeightManager()

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

            rewards.append(config["fine_weight"] * fine_reward + config["coarse_weight"] * coarse_reward)
        except Exception as e:
            _log_debug(f"Stage3 Error: {e}")
            rewards.append(config["error_score"])

    return rewards


def reward_funcs_registry(reward_config_path=None):
    """Initialize rewards; the trainer applies the YAML aggregation weights."""
    config = load_reward_config(reward_config_path or os.getenv("REWARD_CONFIG_PATH"))
    apply_reward_config(config)
    return [format_reward, stage1_category_reward, stage2_node_reward, stage3_edge_reward]


