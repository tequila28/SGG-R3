import json

VG150_OBJ_CATEGORIES = ['__background__', 'airplane', 'animal', 'arm', 'bag', 'banana', 'basket', 'beach', 'bear', 'bed', 'bench', 'bike', 'bird', 'board', 'boat', 'book', 'boot', 'bottle', 'bowl', 'box', 'boy', 'branch', 'building', 'bus', 'cabinet', 'cap', 'car', 'cat', 'chair', 'child', 'clock', 'coat', 'counter', 'cow', 'cup', 'curtain', 'desk', 'dog', 'door', 'drawer', 'ear', 'elephant', 'engine', 'eye', 'face', 'fence', 'finger', 'flag', 'flower', 'food', 'fork', 'fruit', 'giraffe', 'girl', 'glass', 'glove', 'guy', 'hair', 'hand', 'handle', 'hat', 'head', 'helmet', 'hill', 'horse', 'house', 'jacket', 'jean', 'kid', 'kite', 'lady', 'lamp', 'laptop', 'leaf', 'leg', 'letter', 'light', 'logo', 'man', 'men', 'motorcycle', 'mountain', 'mouth', 'neck', 'nose', 'number', 'orange', 'pant', 'paper', 'paw', 'people', 'person', 'phone', 'pillow', 'pizza', 'plane', 'plant', 'plate', 'player', 'pole', 'post', 'pot', 'racket', 'railing', 'rock', 'roof', 'room', 'screen', 'seat', 'sheep', 'shelf', 'shirt', 'shoe', 'short', 'sidewalk', 'sign', 'sink', 'skateboard', 'ski', 'skier', 'sneaker', 'snow', 'sock', 'stand', 'street', 'surfboard', 'table', 'tail', 'tie', 'tile', 'tire', 'toilet', 'towel', 'tower', 'track', 'train', 'tree', 'truck', 'trunk', 'umbrella', 'vase', 'vegetable', 'vehicle', 'wave', 'wheel', 'window', 'windshield', 'wing', 'wire', 'woman', 'zebra']


VG150_PREDICATES = ['__background__', "above", "across", "against", "along", "and", "at", "attached to", "behind", "belonging to", "between", "carrying", "covered in", "covering", "eating", "flying in", "for", "from", "growing on", "hanging from", "has", "holding", "in", "in front of", "laying on", "looking at", "lying on", "made of", "mounted on", "near", "of", "on", "on back of", "over", "painted on", "parked on", "part of", "playing", "riding", "says", "sitting on", "standing on", "to", "under", "using", "walking in", "walking on", "watching", "wearing", "wears", "with"]


psg_categories = {"thing_classes": ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"], "stuff_classes": ["banner", "blanket", "bridge", "cardboard", "counter", "curtain", "door-stuff", "floor-wood", "flower", "fruit", "gravel", "house", "light", "mirror-stuff", "net", "pillow", "platform", "playingfield", "railroad", "river", "road", "roof", "sand", "sea", "shelf", "snow", "stairs", "tent", "towel", "wall-brick", "wall-stone", "wall-tile", "wall-wood", "water-other", "window-blind", "window-other", "tree-merged", "fence-merged", "ceiling-merged", "sky-other-merged", "cabinet-merged", "table-merged", "floor-other-merged", "pavement-merged", "mountain-merged", "grass-merged", "dirt-merged", "paper-merged", "food-other-merged", "building-other-merged", "rock-merged", "wall-other-merged", "rug-merged"], "predicate_classes": ["over", "in front of", "beside", "on", "in", "attached to", "hanging from", "on back of", "falling off", "going down", "painted on", "walking on", "running on", "crossing", "standing on", "lying on", "sitting on", "flying over", "jumping over", "jumping from", "wearing", "holding", "carrying", "looking at", "guiding", "kissing", "eating", "drinking", "feeding", "biting", "catching", "picking", "playing with", "chasing", "climbing", "cleaning", "playing", "touching", "pushing", "pulling", "opening", "cooking", "talking to", "throwing", "slicing", "driving", "riding", "parked on", "driving on", "about to hit", "kicking", "swinging", "entering", "exiting", "enclosing", "leaning on"]}

PSG_OBJ_CATEGORIES = psg_categories['thing_classes'] + psg_categories['stuff_classes']

PSG_REL_CATEGORIES = psg_categories['predicate_classes']



def get_vg150_categories():
    """
    将 VG150 谓词分为三类：空间、从属/组成、语义交互
    注意：'__background__' 已被忽略
    """
    categories = {
        # 1. 几何空间类 (Spatial)
        # 纯粹的位置描述，信息密度最低
        "spatial_relations": [
            "above", "across", "against", "along", "at", "behind", 
            "between", "in", "in front of", "near", "on", 
            "over", "to", "under", "from", "on back of"
        ],

        # 2. 从属与组成类 (Possession / Composition)
        # 逻辑关系，通常难以仅通过像素判断动作，需要常识推理
        "possession_relations": [
            "belonging to", "has", "made of", "of", "part of", 
            "with",  # 通常指 "Man with glasses" (拥有)
            "and",   # 弱连接，通常归为此类或忽略
            "for"    # 功能指向，归为此类
        ],

        # 3. 语义交互与动作类 (Semantic / Interaction)
        # 包含具体动作、特定姿态、物理附着。信息密度最高，是SGG的重点。
        "interaction_relations": [
            # 姿态 (Pose) - 比单纯的 'on' 更具体
            "sitting on", "standing on", "lying on", "laying on",
            
            # 穿戴 (Wearing)
            "wearing", "wears",
            
            # 动态/手部操作 (Action)
            "carrying", "eating", "holding", "playing", "riding", 
            "using", "walking in", "walking on", 
            
            # 视觉/感知 (Sensory)
            "looking at", "watching", "says",
            
            # 物理附着与状态 (Attachment/State)
            "attached to", "covered in", "covering", "flying in", 
            "growing on", "hanging from", "mounted on", "painted on", 
            "parked on"
        ]
    }
    
    return categories





def get_psg_categories():
    """
    将 PSG 谓词分为：空间、静态动作(状态)、动态动作 三类
    """
    categories = {
        # 1. 空间关系 (Spatial)
        # 特征：主要描述相对位置，不涉及生命体的意图，或者是非常稳固的物理连接。
        "spatial_relations": [
            "over", 
            "in front of", 
            "beside", 
            "on", 
            "in", 
            "on back of", 
            "enclosing"      # 空间包含
        ],

        # 2. 静态动作/状态 (Static Action / State)
        # 特征：涉及姿态(Pose)、持续性的接触或状态。虽然是动词，但在静态图中表现为一种“保持”的状态。
        "static_action_relations": [
            "painted on",    # 平面依附
            "attached to",   # 物理依附，视为空间结构
            "hanging from",  # 悬挂状态
            "standing on",   # 姿态
            "lying on",      # 姿态
            "sitting on",    # 姿态
            "leaning on",    # 姿态
            "parked on",     # 车辆的静止状态
            "wearing",       # 持续穿戴状态
            "holding",       # 静态持有
            "carrying",      # 搬运/携带（通常伴随持有状态，视为持续动作）
            "looking at",    # 视觉关注（非物理接触的静态交互）
            "touching",      # 轻微接触（视为状态）
            "driving",       # 驾驶（作为一种持续的操作状态，人-车关系）
            "riding"         # 骑乘（人-马/车关系）
        ],

        # 3. 动态动作 (Dynamic Action)
        # 特征：涉及位移、力的施加、具体的交互操作，具有明显的瞬间性或运动感。
        "dynamic_action_relations": [
            # 位移类 (Locomotion)
            "falling off", 
            "going down", 
            "walking on", 
            "running on", 
            "crossing", 
            "flying over", 
            "jumping over", 
            "jumping from", 
            "entering", 
            "exiting", 
            "climbing", 
            "driving on",    # 车-路关系（车辆正在行驶中，属于动态位移）
            "about to hit",  # 极短瞬间的状态

            # 交互与操作类 (Interaction & Manipulation)
            "guiding", 
            "kissing", 
            "eating", 
            "drinking", 
            "feeding", 
            "biting", 
            "catching", 
            "picking", 
            "playing with", 
            "chasing", 
            "cleaning", 
            "playing", 
            "pushing", 
            "pulling", 
            "opening", 
            "cooking", 
            "talking to", 
            "throwing", 
            "slicing", 
            "kicking", 
            "swinging"
        ]
    }
    return categories



VG150_BASE_OBJ_CATEGORIES = set(['tile', 'drawer', 'men', 'railing', 'stand', 'towel', 'sneaker', 'vegetable', 'screen', 'vehicle', 'animal', 'kite', 'cabinet', 'sink', 'wire', 'fruit', 'curtain', 'lamp', 'flag', 'pot', 'sock', 'boot', 'guy', 'kid', 'finger', 'basket', 'wave', 'lady', 'orange', 'number', 'toilet', 'post', 'room', 'paper', 'mountain', 'paw', 'banana', 'rock', 'cup', 'hill', 'house', 'airplane', 'plant', 'skier', 'fork', 'box', 'seat', 'engine', 'mouth', 'letter', 'windshield', 'desk', 'board', 'counter', 'branch', 'coat', 'logo', 'book', 'roof', 'tie', 'tower', 'glove', 'sheep', 'neck', 'shelf', 'bottle', 'cap', 'vase', 'racket', 'ski', 'phone', 'handle', 'boat', 'tire', 'flower', 'child', 'bowl', 'pillow', 'player', 'trunk', 'bag', 'wing', 'light', 'laptop', 'pizza', 'cow', 'truck', 'jean', 'eye', 'arm', 'leaf', 'bird', 'surfboard', 'umbrella', 'food', 'people', 'nose', 'beach', 'sidewalk', 'helmet', 'face', 'skateboard', 'motorcycle', 'clock', 'bear'])

VG150_BASE_PREDICATE = set(["between", "to", "made of", "looking at", "along", "laying on", "using", "carrying", "against", "mounted on", "sitting on", "flying in", "covering", "from", "over", "near", "hanging from", "across", "at", "above", "watching", "covered in", "wearing", "holding", "and", "standing on", "lying on", "growing on", "under", "on back of", "with", "has", "in front of", "behind", "parked on"])




PROMPT_CLOSE_TEMPLATE_vg = '''
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


Generate the three-stage analysis for the image:'''





PROMPT_CLOSE_TEMPLATE_psg = '''
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



Generate the three-stage analysis for the image:'''


categories_vg = get_vg150_categories()
categories_psg = get_psg_categories()


def format_prompt_close_sg(dataset, template, obj_cls, r1, r2, r3):
    """格式化prompt模板，替换占位符为实际的类别列表"""
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



PROMPT_CLOSE_PSG = format_prompt_close_sg('psg', PROMPT_CLOSE_TEMPLATE_psg, PSG_OBJ_CATEGORIES, categories_psg["spatial_relations"], categories_psg["static_action_relations"], categories_psg["dynamic_action_relations"])
PROMPT_CLOSE_VG150 = format_prompt_close_sg('vg', PROMPT_CLOSE_TEMPLATE_vg, VG150_OBJ_CATEGORIES[1:], categories_vg["spatial_relations"], categories_vg["possession_relations"], categories_vg["interaction_relations"])













