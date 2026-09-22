import json
import os
import sys
from typing import Dict, List, Optional

from datasets import load_dataset


current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(src_dir)
sys.path.insert(0, project_root)
sys.path.insert(0, src_dir)


RELATIONSHIP_DATA_CACHE = None
IMAGE_INDEX_CACHE = None

spatial_relations = []
possession_relations = []
interaction_relations = []
static_action_relations = []
dynamic_action_relations = []


def init_relation_categories(dataset_name: str) -> None:
    """Initialize dataset-specific relationship categories."""
    global spatial_relations, possession_relations, interaction_relations
    global static_action_relations, dynamic_action_relations

    if "psg" in dataset_name.lower():
        from src.prompt.prompt_cot import get_psg_categories

        categories = get_psg_categories()
        spatial_relations = categories["spatial_relations"]
        static_action_relations = categories["static_action_relations"]
        dynamic_action_relations = categories["dynamic_action_relations"]
    else:
        from src.prompt.prompt_cot import get_vg150_categories

        categories = get_vg150_categories()
        spatial_relations = categories["spatial_relations"]
        possession_relations = categories["possession_relations"]
        interaction_relations = categories["interaction_relations"]


def classify_relationship(predicate: str, dataset_name: str) -> str:
    """Map one predicate to its semantic relation group."""
    if "psg" in dataset_name.lower():
        if predicate in spatial_relations:
            return "spatial"
        if predicate in static_action_relations:
            return "static_action"
        if predicate in dynamic_action_relations:
            return "dynamic_action"
        return "other"

    if predicate in spatial_relations:
        return "spatial"
    if predicate in possession_relations:
        return "possession"
    if predicate in interaction_relations:
        return "interaction"
    return "other"


def should_normalize_bbox(model_type: str) -> bool:
    """Return whether bounding boxes should be normalized to the Qwen2-VL 0-1000 scale."""
    return model_type != "qwen2.5vl"


def normalize_bbox(box: List[float], iw: int, ih: int, model_type: str) -> List[float]:
    """Normalize a bounding box for the selected model family."""
    if should_normalize_bbox(model_type):
        return [
            int(box[0] / iw * 1000),
            int(box[1] / ih * 1000),
            int(box[2] / iw * 1000),
            int(box[3] / ih * 1000),
        ]
    return box


def process_objects_bboxes(objects: List[Dict], iw: int, ih: int, model_type: str) -> List[Dict]:
    """Normalize object bounding boxes without mutating the original objects."""
    processed_objects = []
    for obj in objects:
        processed_obj = obj.copy()
        processed_obj["bbox"] = normalize_bbox(obj["bbox"], iw, ih, model_type)
        processed_objects.append(processed_obj)
    return processed_objects


def sort_categories_by_first_letter(categories: List[Dict]) -> List[Dict]:
    """Sort categories by their first character for deterministic prompts."""
    return sorted(categories, key=lambda x: x["id"][0].lower() if x["id"] else "")


def _normalize_hf_dataset_id(source: str) -> str:
    """Convert a Hugging Face dataset URL into a dataset ID."""
    prefix = "https://huggingface.co/datasets/"
    if source.startswith(prefix):
        return source[len(prefix):].strip("/")
    return source


def _dataset_to_records(dataset_obj) -> List[Dict]:
    """Convert a Hugging Face Dataset or DatasetDict into a list of records."""
    if hasattr(dataset_obj, "keys"):
        split_name = "train" if "train" in dataset_obj else next(iter(dataset_obj.keys()))
        dataset_obj = dataset_obj[split_name]
    return [dict(item) for item in dataset_obj]


def load_relationship_data(source: str) -> List[Dict]:
    """Load augmented relationship data from a local JSON file or Hugging Face dataset."""
    try:
        if os.path.isfile(source):
            with open(source, "r", encoding="utf-8") as f:
                return json.load(f)

        dataset_id = _normalize_hf_dataset_id(source)
        dataset = load_dataset(dataset_id)
        return _dataset_to_records(dataset)
    except Exception as e:
        print(f"Error loading relationship data: {e}")
        return []


def get_sample_by_image_id_cached(image_id: str, source: str) -> Optional[Dict]:
    """Fetch one augmented sample by image ID using a process-level cache."""
    global RELATIONSHIP_DATA_CACHE, IMAGE_INDEX_CACHE

    if RELATIONSHIP_DATA_CACHE is None:
        print("Loading relationship data into cache...")
        try:
            RELATIONSHIP_DATA_CACHE = load_relationship_data(source)
            IMAGE_INDEX_CACHE = {str(item["image_id"]): item for item in RELATIONSHIP_DATA_CACHE}
            print(f"Loaded {len(RELATIONSHIP_DATA_CACHE)} samples into cache")
        except Exception as e:
            print(f"Error loading relationship data: {e}")
            return None

    return IMAGE_INDEX_CACHE.get(str(image_id))


def _collect_ordered_objects(objects: List[Dict]):
    categories = []
    seen = set()
    for obj in objects:
        category_name = obj["id"].split(".")[0]
        if category_name not in seen:
            seen.add(category_name)
            categories.append({"id": category_name})

    categories = sort_categories_by_first_letter(categories)
    category_to_objects = {}
    for obj in objects:
        category = obj["id"].split(".")[0]
        category_to_objects.setdefault(category, []).append(obj)

    ordered_objects = []
    obj_map = {}
    category_counters = {cat["id"]: 1 for cat in categories}
    for category in categories:
        cat_name = category["id"]
        for obj in category_to_objects.get(cat_name, []):
            original_id = obj["id"]
            new_id = f"{cat_name}.{category_counters[cat_name]}"
            category_counters[cat_name] += 1
            obj_map[original_id] = new_id

            new_obj = obj.copy()
            new_obj["id"] = new_id
            ordered_objects.append(new_obj)

    return categories, ordered_objects, obj_map


def _classify_relationships(relationships: List[Dict], dataset_name: str) -> Dict[str, List[Dict]]:
    if "psg" in dataset_name.lower():
        grouped = {
            "spatial_relations": [],
            "static_action_relations": [],
            "dynamic_action_relations": [],
        }
    else:
        grouped = {
            "spatial_relations": [],
            "possession_relations": [],
            "interaction_relations": [],
        }

    other_rels = []
    for rel in relationships:
        rel_type = classify_relationship(rel["predicate"], dataset_name)
        if rel_type == "spatial":
            grouped["spatial_relations"].append(rel)
        elif rel_type == "static_action":
            grouped["static_action_relations"].append(rel)
        elif rel_type == "dynamic_action":
            grouped["dynamic_action_relations"].append(rel)
        elif rel_type == "possession":
            grouped["possession_relations"].append(rel)
        elif rel_type == "interaction":
            grouped["interaction_relations"].append(rel)
        else:
            other_rels.append(rel)

    if other_rels:
        grouped["other_relations"] = other_rels
    return grouped


def sort_relationships_by_subject_order(relationships: Dict, normalized_objects: List[Dict], dataset_name: str) -> Dict:
    """Sort relationships by subject order from the object stage."""
    object_order = {obj["id"]: idx for idx, obj in enumerate(normalized_objects)}
    if "psg" in dataset_name.lower():
        categories = ["spatial_relations", "static_action_relations", "dynamic_action_relations"]
    else:
        categories = ["spatial_relations", "possession_relations", "interaction_relations"]

    sorted_relationships = {}
    for category in categories:
        rels = relationships.get(category, [])
        if isinstance(rels, list):
            sorted_relationships[category] = sorted(rels, key=lambda rel: object_order.get(rel["subject"], float("inf")))
        else:
            sorted_relationships[category] = []
    return sorted_relationships


def _format_stages(categories, objects, relationships):
    json_args = {"indent": None, "separators": (",", ":"), "ensure_ascii": False}
    stage1 = f"<CATEGORY>{json.dumps({'categories': categories}, **json_args)}</CATEGORY>"
    stage2 = f"<OBJECT>{json.dumps({'objects': objects}, **json_args)}</OBJECT>"
    stage3 = f"<RELATION>{json.dumps({'relations': relationships}, **json_args)}</RELATION>"
    return f"{stage1}\n{stage2}\n{stage3}"


def format_answer_with_json(image_id: str, file_path: str, iw: int, ih: int, dataset_name: str, model_type: str) -> str:
    """Format an answer using augmented relationship data."""
    sample = get_sample_by_image_id_cached(image_id, file_path)
    if not sample:
        return f"Error: No data found for image_id {image_id}"

    objects = sample.get("objects", [])
    relationships = sample.get("relations", sample.get("relationships", {}))
    if isinstance(objects, str):
        objects = json.loads(objects)
    if isinstance(relationships, str):
        relationships = json.loads(relationships)

    normalized_objects = process_objects_bboxes(objects, iw, ih, model_type)
    categories, ordered_objects, obj_map = _collect_ordered_objects(normalized_objects)

    if isinstance(relationships, list):
        relationships = _classify_relationships(relationships, dataset_name)

    for category in relationships:
        for rel in relationships[category]:
            rel["subject"] = obj_map.get(rel["subject"], rel["subject"])
            rel["object"] = obj_map.get(rel["object"], rel["object"])

    sorted_relationships = sort_relationships_by_subject_order(relationships, ordered_objects, dataset_name)
    return _format_stages(categories, ordered_objects, sorted_relationships)


def format_answer_original(objects: List[Dict], relationships: List[Dict], dataset_name: str, model_type: Optional[str] = None) -> str:
    """Format an answer from original dataset annotations."""
    categories, ordered_objects, obj_map = _collect_ordered_objects(objects)
    remapped_relationships = []
    for rel in relationships:
        new_rel = rel.copy()
        new_rel["subject"] = obj_map.get(rel["subject"], rel["subject"])
        new_rel["object"] = obj_map.get(rel["object"], rel["object"])
        remapped_relationships.append(new_rel)

    classified_relationships = _classify_relationships(remapped_relationships, dataset_name)
    return _format_stages(categories, [{"id": obj["id"], "bbox": obj["bbox"]} for obj in ordered_objects], classified_relationships)


def format_data(dataset_name: str, sample: Dict, model_type: Optional[str] = None, augmented_data_path: Optional[str] = None) -> Dict:
    """Prepare one dataset example for SFT training."""
    init_relation_categories(dataset_name)

    image = sample["image"].convert("RGB")
    iw, ih = image.size

    if "psg" in dataset_name.lower():
        from src.prompt.prompt_cot import PROMPT_PSG as prompt
    else:
        from src.prompt.prompt_cot import PROMPT_VG150 as prompt

    if augmented_data_path:
        answer = format_answer_with_json(str(sample["image_id"]), augmented_data_path, iw, ih, dataset_name, model_type)
    else:
        objects = process_objects_bboxes(json.loads(sample["objects"]), iw, ih, model_type)
        relationships = json.loads(sample["relationships"])
        answer = format_answer_original(objects, relationships, dataset_name, model_type)

    return {
        "images": [image],
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer},
        ],
    }
