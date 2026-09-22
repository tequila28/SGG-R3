import json

from src.prompt.data_category import PSG_OBJ_CATEGORIES, VG150_OBJ_CATEGORIES, get_vg150_categories, get_psg_categories



PROMPT_TEMPLATE_vg = '''
# Three-Stage Visual Reasoning Analysis
Execute a structured, three-stage visual reasoning analysis for the provided image, strictly adhering to the predefined object and relation categories. Proceed through each stage sequentially and encapsulate your outputs using the specified tags.

## Available Categories
- **Object Categories:** `{OBJ_CLS}`
- **Relation Predicates:**
    - **Spatial Relations:** `{SPATIAL_REL}`
    - **Possession Relations:** `{POSSESSION_REL}`
    - **Interaction Relations:** `{SEMANTIC_REL}`

## Stage 1: Object Category Detection
- **Task:** Identify all unique object categories in the image from the provided "Available Object Categories".
- **Requirements:**
    - Only output categories that are clearly visible and identifiable in the image.
    - The categories must conform to the predefined object categories.
    - Maintain uniqueness (no duplicates).
- **Output Format:** Output as a JSON array within `<CATEGORY>` and `</CATEGORY>` tags.

## Stage 2: Instance-Level Object Localization
- **Task:** Detect and localize every individual instance of the categories identified in Stage 1 sequentially.
- **Requirements:**
    - Process categories in the order identified in Stage 1.
    - The object names must strictly correspond to the categories identified in Stage 1.
    - Assign sequential instance numbers to objects within each category (e.g., `man.1`, `man.2`, `car.1`).
    - Provide precise bounding boxes in `[x1, y1, x2, y2]` format (integer coordinates).
    - All categories listed in Stage 1 must have instances detected in Stage 2.
- **Output Format:** Output as a JSON array within `<OBJECT>` and `</OBJECT>` tags.

## Stage 3: Multi-Category Relation Extraction
- **Task:** Independently analyze relations in three categories, examining all object pairs within each category.
- **Three Relation Categories:**
    - **Spatial Relations:** Analyze all object pairs for spatial/topological relations.
    - **Possession Relations:** Analyze all object pairs for ownership, composition, and part-whole relations.  
    - **Interaction Relations:** Analyze all object pairs for action-oriented and functional relations.
- **Requirements:**
    - Sequentially analyze each object in Stage 2 as the subject against all other objects in the scene to extract meaningful relations.
    - All relations must be between objects localized in Stage 2.
    - All relation predicates MUST belong to their respective predefined category.  
- **Output Format:** Output as a JSON array within `<RELATION>` and `</RELATION>` tags.


## Complete Output Example

<CATEGORY>
{"categories":[{"id":"tree"},{"id":"sidewalk"},{"id":"building"},{"id":"man"},{"id":"bike"}]}
</CATEGORY>

<OBJECT>
{"objects": [
    {"id": "tree.1", "bbox": [0, 0, 799, 557]},
    {"id": "sidewalk.1", "bbox": [75, 306, 798, 596]},
    {"id": "building.1", "bbox": [0, 0, 222, 538]},
    {"id": "man.1", "bbox": [369, 262, 446, 512]},
    {"id": "man.2", "bbox": [236, 246, 296, 508]},
    {"id": "bike.1", "bbox": [335, 317, 362, 353]}
]}
</OBJECT>

<RELATION>
{"relations":{
"spatial_relations":[
  {"subject":"tree.1","predicate":"near","object":"sidewalk.1"},
  {"subject":"tree.1","predicate":"near","object":"building.1"},
  {"subject":"man.1","predicate":"on","object":""sidewalk.1"}
],
"possession_relations":[
  {"subject":"building.1","predicate":"with","object":"tree.1"}
],
"interaction_relations":[
  {"subject":"man.1","predicate":"standing on","object":"sidewalk.1"},
  {"subject":"man.2","predicate":"standing on","object":"sidewalk.1"},
  {"subject":"bike.1","predicate":"parked on","object":"sidewalk.1"}
]}
}
</RELATION>


## Generate the three-stage analysis for the image:'''





PROMPT_TEMPLATE_psg = '''
# Three-Stage Visual Reasoning Analysis
Execute a structured, three-stage visual reasoning analysis for the provided image, strictly adhering to the predefined object and relation categories. Proceed through each stage sequentially and encapsulate your outputs using the specified tags.

## Available Categories
- **Object Categories:** `{OBJ_CLS}`
- **Relation Predicates:**
    - **Spatial Relations:** `{SPATIAL_REL}`
    - **Static Action Relations:** `{STATIC_REL}`
    - **Dynamic Action Relations:** `{DYNAMIC_REL}`

## Stage 1: Object Category Detection
- **Task:** Identify all unique object categories in the image from the provided "Available Object Categories".
- **Requirements:**
    - Only output categories that are clearly visible and identifiable in the image.
    - The categories must conform to the predefined object categories.
    - Maintain uniqueness (no duplicates).
- **Output Format:** Output as a JSON array within `<CATEGORY>` and `</CATEGORY>` tags.

## Stage 2: Instance-Level Object Localization
- **Task:** Detect and localize every individual instance of the categories identified in Stage 1 sequentially.
- **Requirements:**
    - Process categories in the order identified in Stage 1.
    - The object names must strictly correspond to the categories identified in Stage 1.
    - Assign sequential instance numbers to objects within each category (e.g., `man.1`, `man.2`, `car.1`).
    - Provide precise bounding boxes in `[x1, y1, x2, y2]` format (integer coordinates).
    - All categories listed in Stage 1 must have instances detected in Stage 2.
- **Output Format:** Output as a JSON array within `<OBJECT>` and `</OBJECT>` tags.

## Stage 3: Multi-Category Relation Extraction
- **Task:** Independently analyze relations in three categories, examining all object pairs within each category.
- **Three Relation Categories:**
    - **Spatial Relations:** Analyze all object pairs for spatial/topological relations.
    - **Static Action Relations:** Analyze all object pairs for static actions, poses, and persistent states.
    - **Dynamic Action Relations:** Analyze all object pairs for dynamic actions, movements, and interactions.
- **Requirements:**
    - Sequentially analyze each object in Stage 2 as the subject against all other objects in the scene to extract meaningful relations.
    - All relations must be between objects localized in Stage 2.
    - All relation predicates MUST belong to their respective predefined category.  
- **Output Format:** Output as a JSON array within `<RELATION>` and `</RELATION>` tags.


## Complete Output Example

<CATEGORY>
{"categories":[{"id":"tree"},{"id":"sidewalk"},{"id":"building"},{"id":"man"},{"id":"bike"}]}
</CATEGORY>
<OBJECT>
{"objects": [
    {"id": "tree.1", "bbox": [0, 0, 799, 557]},
    {"id": "sidewalk.1", "bbox": [75, 306, 798, 596]},
    {"id": "building.1", "bbox": [0, 0, 222, 538]},
    {"id": "man.1", "bbox": [369, 262, 446, 512]},
    {"id": "man.2", "bbox": [236, 246, 296, 508]},
    {"id": "bike.1", "bbox": [335, 317, 362, 353]}
]}
</OBJECT>
<RELATION>
{"relations":{
"spatial_relations":[
  {"subject":"tree.1","predicate":"beside","object":"sidewalk.1"},
  {"subject":"tree.1","predicate":"in front of","object":"building.1"},
  {"subject":"sidewalk.1","predicate":"in front of","object":"building.1"}
],
"static_action_relations":[
  {"subject":"man.1","predicate":"standing on","object":"sidewalk.1"},
  {"subject":"man.2","predicate":"holding","object":"bike.1"},
  {"subject":"bike.1","predicate":"parked on","object":"sidewalk.1"}
],
"dynamic_action_relations":[
  {"subject":"man.2","predicate":"walking on","object":"sidewalk.1"},
  {"subject":"man.2","predicate":"talking to","object":"man.1"}
]}
}
</RELATION>



## Generate the three-stage analysis for the image:'''


categories_vg = get_vg150_categories()
categories_psg = get_psg_categories()


def format_prompt_close_sg(dataset, template, obj_cls, r1, r2, r3):
    """Format the prompt template, replacing placeholders with actual category lists."""
    if 'vg' == dataset:
        formatted_template = template.replace("{OBJ_CLS}", json.dumps(obj_cls))
        formatted_template = formatted_template.replace("{SPATIAL_REL}", json.dumps(r1))
        formatted_template = formatted_template.replace("{POSSESSION_REL}", json.dumps(r2))
        formatted_template = formatted_template.replace("{SEMANTIC_REL}", json.dumps(r3))

    else:
        formatted_template = template.replace("{OBJ_CLS}", json.dumps(obj_cls))
        formatted_template = formatted_template.replace("{SPATIAL_REL}", json.dumps(r1))
        formatted_template = formatted_template.replace("{STATIC_REL}", json.dumps(r2))
        formatted_template = formatted_template.replace("{DYNAMIC_REL}", json.dumps(r3))

    return formatted_template



PROMPT_PSG = format_prompt_close_sg('psg', PROMPT_TEMPLATE_psg, PSG_OBJ_CATEGORIES, categories_psg["spatial_relations"], categories_psg["static_action_relations"], categories_psg["dynamic_action_relations"])
PROMPT_VG150 = format_prompt_close_sg('vg', PROMPT_TEMPLATE_vg, VG150_OBJ_CATEGORIES[1:], categories_vg["spatial_relations"], categories_vg["possession_relations"], categories_vg["interaction_relations"])













