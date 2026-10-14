VG150_OBJ_CATEGORIES = ['__background__', 'airplane', 'animal', 'arm', 'bag', 'banana', 'basket', 'beach', 'bear', 'bed', 'bench', 'bike', 'bird', 'board', 'boat', 'book', 'boot', 'bottle', 'bowl', 'box', 'boy', 'branch', 'building', 'bus', 'cabinet', 'cap', 'car', 'cat', 'chair', 'child', 'clock', 'coat', 'counter', 'cow', 'cup', 'curtain', 'desk', 'dog', 'door', 'drawer', 'ear', 'elephant', 'engine', 'eye', 'face', 'fence', 'finger', 'flag', 'flower', 'food', 'fork', 'fruit', 'giraffe', 'girl', 'glass', 'glove', 'guy', 'hair', 'hand', 'handle', 'hat', 'head', 'helmet', 'hill', 'horse', 'house', 'jacket', 'jean', 'kid', 'kite', 'lady', 'lamp', 'laptop', 'leaf', 'leg', 'letter', 'light', 'logo', 'man', 'men', 'motorcycle', 'mountain', 'mouth', 'neck', 'nose', 'number', 'orange', 'pant', 'paper', 'paw', 'people', 'person', 'phone', 'pillow', 'pizza', 'plane', 'plant', 'plate', 'player', 'pole', 'post', 'pot', 'racket', 'railing', 'rock', 'roof', 'room', 'screen', 'seat', 'sheep', 'shelf', 'shirt', 'shoe', 'short', 'sidewalk', 'sign', 'sink', 'skateboard', 'ski', 'skier', 'sneaker', 'snow', 'sock', 'stand', 'street', 'surfboard', 'table', 'tail', 'tie', 'tile', 'tire', 'toilet', 'towel', 'tower', 'track', 'train', 'tree', 'truck', 'trunk', 'umbrella', 'vase', 'vegetable', 'vehicle', 'wave', 'wheel', 'window', 'windshield', 'wing', 'wire', 'woman', 'zebra']


VG150_PREDICATES = ['__background__', "above", "across", "against", "along", "and", "at", "attached to", "behind", "belonging to", "between", "carrying", "covered in", "covering", "eating", "flying in", "for", "from", "growing on", "hanging from", "has", "holding", "in", "in front of", "laying on", "looking at", "lying on", "made of", "mounted on", "near", "of", "on", "on back of", "over", "painted on", "parked on", "part of", "playing", "riding", "says", "sitting on", "standing on", "to", "under", "using", "walking in", "walking on", "watching", "wearing", "wears", "with"]



psg_categories = {"thing_classes": ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"], "stuff_classes": ["banner", "blanket", "bridge", "cardboard", "counter", "curtain", "door-stuff", "floor-wood", "flower", "fruit", "gravel", "house", "light", "mirror-stuff", "net", "pillow", "platform", "playingfield", "railroad", "river", "road", "roof", "sand", "sea", "shelf", "snow", "stairs", "tent", "towel", "wall-brick", "wall-stone", "wall-tile", "wall-wood", "water-other", "window-blind", "window-other", "tree-merged", "fence-merged", "ceiling-merged", "sky-other-merged", "cabinet-merged", "table-merged", "floor-other-merged", "pavement-merged", "mountain-merged", "grass-merged", "dirt-merged", "paper-merged", "food-other-merged", "building-other-merged", "rock-merged", "wall-other-merged", "rug-merged"], "predicate_classes": ["over", "in front of", "beside", "on", "in", "attached to", "hanging from", "on back of", "falling off", "going down", "painted on", "walking on", "running on", "crossing", "standing on", "lying on", "sitting on", "flying over", "jumping over", "jumping from", "wearing", "holding", "carrying", "looking at", "guiding", "kissing", "eating", "drinking", "feeding", "biting", "catching", "picking", "playing with", "chasing", "climbing", "cleaning", "playing", "touching", "pushing", "pulling", "opening", "cooking", "talking to", "throwing", "slicing", "driving", "riding", "parked on", "driving on", "about to hit", "kicking", "swinging", "entering", "exiting", "enclosing", "leaning on"]}

PSG_OBJ_CATEGORIES = psg_categories['thing_classes'] + psg_categories['stuff_classes']

PSG_PREDICATES = psg_categories['predicate_classes']




def get_vg150_categories():
    """
    Categorize VG150 predicates into three classes: spatial, possession/composition, semantic/interaction.
    """
    categories = {
        # 1. Spatial relations
        "spatial_relations": [
            "above", "across", "against", "along", "at", "behind", 
            "between", "in", "in front of", "near", "on", "over", 
            "to", "under", "from", "on back of"
        ],

        # 2. Possession/composition relations
        "possession_relations": [
            "belonging to", "has", "made of", "of", "part of", 
            "with", "and", "for"
        ],

        # 3. Semantic/interaction relations
        "interaction_relations": [
            "sitting on", "standing on", "lying on", "laying on",
            "wearing", "wears", "carrying", "eating", "holding", 
            "playing", "riding", "using", "walking in", "walking on",
            "looking at", "watching", "says", "attached to", "covered in",
            "covering", "flying in", "growing on", "hanging from", 
            "mounted on", "painted on", "parked on"
        ]
    }
    
    return categories


def get_psg_categories():
    """
    Categorize PSG predicates into: spatial, static action/state, dynamic action.
    """
    categories = {
        # 1. Spatial relations
        "spatial_relations": [
            "over", "in front of", "beside", "on", "in", 
            "on back of", "enclosing"
        ],

        # 2. Static action/state relations
        "static_action_relations": [
            "painted on", "attached to", "hanging from", "standing on", 
            "lying on", "sitting on", "leaning on", "parked on", "wearing", 
            "holding", "carrying", "looking at", "touching", "driving", "riding"
        ],

        # 3. Dynamic action relations
        "dynamic_action_relations": [
            "falling off", "going down", "walking on", "running on", 
            "crossing", "flying over", "jumping over", "jumping from", 
            "entering", "exiting", "climbing", "driving on", "about to hit", 
            "guiding", "kissing", "eating", "drinking", "feeding", "biting", 
            "catching", "picking", "playing with", "chasing", "cleaning", 
            "playing", "pushing", "pulling", "opening", "cooking", "talking to", 
            "throwing", "slicing", "kicking", "swinging"
        ]
    }
    
    return categories


def get_vg150_frequency_groups():

    groups = {
    "head": ["on", "has", "wearing", "of", "in", "near", "behind", "with", "holding", "above", "wears", "sitting on", "under", "in front of", "riding"],
    "body": ["standing on", "at", "attached to", "carrying", "walking on", "over", "for", "belonging to", "hanging from", "looking at", "parked on", "laying on", "and", "covering", "eating"],
    "tail": ["between", "part of", "along", "covered in", "using", "watching", "on back of", "to", "lying on", "walking in", "mounted on", "against", "across", "from", "growing on", "painted on", "made of", "playing", "says", "flying in"]
    }

    return groups


def get_psg_frequency_groups():

    groups = {
    "head": ["on", "beside", "over", "attached to", "standing on", "in front of", "holding", "in", "walking on", "parked on", "driving on", "sitting on", "looking at", "hanging from", "wearing", "carrying"],
    "body": ["riding", "playing", "lying on", "eating", "running on", "touching", "flying over", "leaning on", "swinging", "driving", "about to hit", "enclosing", "talking to", "playing with", "pulling", "painted on"],
    "tail": ["crossing", "throwing", "jumping from", "biting", "slicing", "catching", "on back of", "drinking", "going down", "guiding", "jumping over", "kicking", "pushing", "feeding", "entering", "chasing", "exiting", "cooking", "kissing", "cleaning", "climbing", "picking", "falling off", "opening"]
    }

    return groups
