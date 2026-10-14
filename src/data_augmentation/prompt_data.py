import json
import os
import re
import sys


current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(src_dir)
sys.path.insert(0, project_root)
sys.path.insert(0, src_dir)

from src.prompt.data_category import (
    PSG_OBJ_CATEGORIES,
    PSG_PREDICATES,
    VG150_OBJ_CATEGORIES,
    VG150_PREDICATES,
    get_psg_categories,
    get_vg150_categories,
)


class PromptBuilder:
    def __init__(self, dataset_type="vg"):
        self.dataset_type = dataset_type
        self._init_categories()

    def _init_categories(self):
        """Initialize predicate and object category metadata."""
        if self.dataset_type == "psg":
            categories = get_psg_categories()
            self.spatial_predicates = set(categories["spatial_relations"])
            self.cat2_predicates = set(categories["static_action_relations"])
            self.cat3_predicates = set(categories["dynamic_action_relations"])
            self.all_predicates = set(PSG_PREDICATES)
            self.obj_categories = PSG_OBJ_CATEGORIES
            self.category2_name = "static_action_relations"
            self.category3_name = "dynamic_action_relations"
        else:
            categories = get_vg150_categories()
            self.spatial_predicates = set(categories["spatial_relations"])
            self.cat2_predicates = set(categories["possession_relations"])
            self.cat3_predicates = set(categories["interaction_relations"])
            self.all_predicates = set(VG150_PREDICATES)
            self.obj_categories = VG150_OBJ_CATEGORIES[1:]
            self.category2_name = "possession_relations"
            self.category3_name = "interaction_relations"

    def preprocess_sample(self, sample):
        """Normalize one sample's object IDs and relationship endpoints."""
        try:
            objects = json.loads(sample["objects"])
            relationships = json.loads(sample["relationships"])
            new_objects, new_relationships = self._preprocess_objects_and_relationships(objects, relationships)

            new_sample = sample.copy()
            new_sample["objects"] = json.dumps(new_objects, ensure_ascii=False)
            new_sample["relationships"] = json.dumps(new_relationships, ensure_ascii=False)
            return new_sample
        except Exception as e:
            print(f"Error preprocessing sample: {e}")
            return sample

    def _preprocess_objects_and_relationships(self, objects, relationships):
        """Rename object IDs to category-local counters and update relationship endpoints."""
        obj_map = {}
        categories = []
        seen = set()

        for obj in objects:
            category_name = obj["id"].split(".")[0]
            if category_name not in seen:
                seen.add(category_name)
                categories.append({"id": category_name})

        category_to_objects = {}
        for obj in objects:
            category = obj["id"].split(".")[0]
            if category not in category_to_objects:
                category_to_objects[category] = []
            category_to_objects[category].append(obj)

        ordered_objects = []
        category_counters = {cat["id"]: 1 for cat in categories}

        for category in categories:
            cat_name = category["id"]
            if cat_name in category_to_objects:
                for obj in category_to_objects[cat_name]:
                    original_id = obj["id"]
                    new_id = f"{cat_name}.{category_counters[cat_name]}"
                    category_counters[cat_name] += 1
                    obj_map[original_id] = new_id

                    new_obj = obj.copy()
                    new_obj["id"] = new_id
                    ordered_objects.append(new_obj)

        new_relationships = []
        for rel in relationships:
            subject_old = rel["subject"]
            object_old = rel["object"]
            predicate = rel["predicate"]

            if subject_old in obj_map and object_old in obj_map:
                new_relationships.append(
                    {
                        "subject": obj_map[subject_old],
                        "predicate": predicate,
                        "object": obj_map[object_old],
                    }
                )
            else:
                print(f"Warning: Object ID {subject_old} or {object_old} not found in object mapping")

        return ordered_objects, new_relationships

    def construct_relationship_augmentation_prompt(self, sample):
        """Build the relationship augmentation prompt and return it with the image."""
        image = sample["image"].convert("RGB")
        preprocessed_sample = self.preprocess_sample(sample)
        objects = json.loads(preprocessed_sample["objects"])

        spatial_relations = list(self.spatial_predicates)
        category2_relations = list(self.cat2_predicates)
        category3_relations = list(self.cat3_predicates)

        if self.dataset_type == "psg":
            category2_display = "Static Action Relations"
            category3_display = "Dynamic Action Relations"
        else:
            category2_display = "Possession Relations"
            category3_display = "Interaction Relations"

        prompt = f'''# Two-Stage Visual Relation Augmentation

Execute a structured, two-stage relation augmentation analysis for the provided image with pre-annotated objects. Your goal is to generate plausible relations.

## Available Relationship Categories
- **Spatial Relations:** {json.dumps(spatial_relations)}
- **{category2_display}:** {json.dumps(category2_relations)}
- **{category3_display}:** {json.dumps(category3_relations)}

## Pre-annotated Objects ({len(objects)} objects, USE EXACT IDs):
Each object includes:
- id: Unique identifier (e.g., "person.1", "car.2")
- box: Bounding box coordinates [x1, y1, x2, y2] representing object position
{json.dumps(objects, indent=2)}

## Stage 1: Scene Context Understanding
- **Task:** Create a comprehensive natural language description for the image that incorporates ALL pre-annotated objects with their exact IDs.
- **Requirements:**
    - Naturally include all {len(objects)} objects using their specific IDs.
    - Describe spatial arrangements, potential interactions, and scene context.
    - Be descriptive but concise.

## Stage 2: Multi-Category Relation Extraction
- **Task:** Based on the global understanding from Stage 1, independently analyze relations in three categories, examining all object pairs within each category.
- ***Three Relation Categories:**
    1. **Spatial Relations:** Analyze object pairs for new spatial/topological relations.
    2. **{category2_display}:** Analyze for new {'static action/pose' if self.dataset_type == 'psg' else 'ownership/composition/part-whole'} relations.
    3. **{category3_display}:** Analyze for new {'dynamic action/interaction' if self.dataset_type == 'psg' else 'action-oriented/functional'} relations.
- **Requirements:**
    - Sequentially analyze each object in Stage 2 as the subject against all other objects in the scene to extract meaningful relations.
    - All relations must be between objects in pre-annotated objects.
    - All relation predicates MUST belong to their respective predefined category.

## Output Format Requirements
Return ONLY a JSON object with this exact structure:

{{
"caption": "A detailed scene description naturally incorporating all object IDs like person.1, car.1, etc.",
"spatial_relations": [
    {{"subject": "object.id", "predicate": "spatial_relations", "object": "object.id"}}
],
"{self.category2_name}": [
    {{"subject": "object.id", "predicate": {self.category2_name}, "object": "object.id"}}
],
"{self.category3_name}": [
    {{"subject": "object.id", "predicate": {self.category3_name}, "object": "object.id"}}
]
}}

Generate the two-stage relationship augmentation analysis, focusing on creating meaningful relationships:'''

        return prompt, image

    def parse_relationship_response(self, response_text, existing_relationships, available_objects):
        """Extract, merge, deduplicate, and categorize relationships from a model response."""
        print("=" * 60)
        print("Parsing model response...")

        try:
            model_relations = self._extract_relations_from_response(response_text, available_objects)

            existing_relations = []
            if existing_relationships:
                if isinstance(existing_relationships, str):
                    existing_relations = json.loads(existing_relationships)
                else:
                    existing_relations = existing_relationships

            all_relations = model_relations + existing_relations
            unique_relations = self._remove_duplicate_relations(all_relations)

            print(
                "Relationship statistics: "
                f"model={len(model_relations)}, existing={len(existing_relations)}, "
                f"unique={len(unique_relations)}"
            )

            result = self._categorize_relations(unique_relations)
            caption = self._extract_caption(response_text)
            if caption:
                result["caption"] = caption
                print(f"Extracted caption: {caption[:100]}...")

            total_relations = sum(len(rels) for rels in result.values() if isinstance(rels, list))
            print(f"Parsing completed. Total relationships: {total_relations}")
            return result
        except Exception as e:
            print(f"Parsing failed: {e}")
            print(f"Response preview: {response_text[:500]}...")
            print("Using fallback relationship categorization.")
            return self.fallback_categorize_relationships(
                existing_relationships if isinstance(existing_relationships, str) else json.dumps(existing_relationships),
                available_objects,
            )

    def _extract_relations_from_response(self, text, available_objects):
        """Extract valid relationship triples from the model response."""
        relations = []
        available_object_ids = {obj["id"] for obj in available_objects}
        pattern = r'{\s*"subject"\s*:\s*"([^"]+)"\s*,\s*"predicate"\s*:\s*"([^"]+)"\s*,\s*"object"\s*:\s*"([^"]+)"\s*}'

        matches = re.finditer(pattern, text, re.IGNORECASE | re.DOTALL)
        for match in matches:
            try:
                subject = match.group(1).strip()
                predicate = match.group(2).strip()
                obj = match.group(3).strip()

                if subject in available_object_ids and obj in available_object_ids and predicate in self.all_predicates:
                    relations.append(
                        {
                            "subject": subject,
                            "predicate": predicate,
                            "object": obj,
                        }
                    )
            except Exception:
                continue

        return relations

    def _categorize_relations(self, relations):
        """Group relationships into dataset-specific predicate categories."""
        if self.dataset_type == "psg":
            result = {
                "spatial_relations": [],
                "static_action_relations": [],
                "dynamic_action_relations": [],
            }
        else:
            result = {
                "spatial_relations": [],
                "possession_relations": [],
                "interaction_relations": [],
            }

        for rel in relations:
            predicate = rel["predicate"]

            if predicate in self.spatial_predicates:
                result["spatial_relations"].append(rel)
            elif predicate in self.cat2_predicates:
                key = "static_action_relations" if self.dataset_type == "psg" else "possession_relations"
                result[key].append(rel)
            elif predicate in self.cat3_predicates:
                key = "dynamic_action_relations" if self.dataset_type == "psg" else "interaction_relations"
                result[key].append(rel)

        return result

    def _remove_duplicate_relations(self, relations):
        """Remove duplicate subject-predicate-object triples."""
        unique_relations = []
        seen = set()

        for rel in relations:
            key = (rel["subject"], rel["predicate"], rel["object"])
            if key not in seen:
                seen.add(key)
                unique_relations.append(rel)

        return unique_relations

    def _extract_caption(self, text):
        """Extract the optional caption field from the model response."""
        pattern = r'"caption"\s*:\s*"([^"]+)"'
        match = re.search(pattern, text, re.IGNORECASE)
        return match.group(1) if match else None

    def fallback_categorize_relationships(self, existing_relationships, available_objects):
        """Fallback to categorizing only the existing relationships."""
        if not existing_relationships or existing_relationships == "[]":
            return self._create_empty_result()

        try:
            if isinstance(existing_relationships, str):
                existing_relations = json.loads(existing_relationships)
            else:
                existing_relations = existing_relationships

            return self._categorize_relations(existing_relations)
        except Exception:
            return self._create_empty_result()

    def _create_empty_result(self):
        """Create an empty category structure for the current dataset type."""
        if self.dataset_type == "psg":
            return {
                "spatial_relations": [],
                "static_action_relations": [],
                "dynamic_action_relations": [],
            }

        return {
            "spatial_relations": [],
            "possession_relations": [],
            "interaction_relations": [],
        }
