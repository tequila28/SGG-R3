import os
import re
import json
from datetime import datetime
from typing import List, Dict
from functools import lru_cache
import numpy as np
from scipy.optimize import linear_sum_assignment
import numpy as np
import json



def refine_node_edge(obj):
    return obj.strip().lower()

def scale_box(box, scale):
    sw, sh = scale
    assert len(box) == 4, " len(box) != 4 "
    return [box[0]*sw, box[1]*sh, box[2]*sw, box[3]*sh]

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



class BipartiteMatcher:
    def __init__(
        self,
        category_semantic_similarity,
        sem_weight,
        iou_weight,
        box_l1_weight,
        padding_cost,
        cache_size,
    ):
        """
        
        
        Args:
            category_semantic_similarity: ?            sem_weight: ?            iou_weight: IoU
            box_l1_weight: 1
        """
        self.category_semantic_similarity = category_semantic_similarity
        self.sem_weight = sem_weight
        self.iou_weight = iou_weight
        self.box_l1_weight = box_l1_weight
        self.padding_cost = padding_cost
        
        # ?        self._freeze_objs = self._freeze_objs
        self._bi_match_impl = self._bi_match_impl
        self._bi_match_cached = lru_cache(maxsize=cache_size)(self._bi_match_cached_impl)
    
    def cost_function(self, pred, gt):
        assert len(pred['bbox']) == 4, f"Invalid bbox length: {len(pred['bbox'])}"
        
        # 
        iou = compute_giou(pred['bbox'], gt['bbox'])  # compute_giou
        sem_sim = self.category_semantic_similarity(pred['id'], gt['id'])
        
        return (self.sem_weight * (1.0 - sem_sim) + 
                self.iou_weight * (1.0 - iou) + 
                self.box_l1_weight * box_L1(pred['bbox'], gt['bbox']))  # box_L1
    
    def _freeze_objs(self, objs):
        return tuple(
            (o["id"], tuple(o["bbox"])) for o in objs
        )
    
    def _bi_match_impl(self, groundtruths, predictions):
        num_gt = len(groundtruths)
        num_pred = len(predictions)
        pad = max(0, num_gt - num_pred)
        cost_matrix = np.zeros((num_pred + pad, num_gt))

        for i, pred in enumerate(predictions):
            for j, gt in enumerate(groundtruths):
                cost_matrix[i, j] = self.cost_function(pred, gt)

        if pad:
            cost_matrix[num_pred:, :] = self.padding_cost

        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        return [
            {
                "groundtruth": groundtruths[c],
                "prediction": predictions[r],
                "cost": cost_matrix[r, c],
            }
            for r, c in zip(row_ind, col_ind)
            if r < num_pred
        ]
    
    def _bi_match_cached_impl(self, gt_key, pred_key):
        groundtruths = [
            {"id": obj_id, "bbox": list(bbox)}
            for obj_id, bbox in gt_key
        ]
        predictions = [
            {"id": obj_id, "bbox": list(bbox)}
            for obj_id, bbox in pred_key
        ]
        return self._bi_match_impl(groundtruths, predictions)
    
    def bi_match(self, groundtruths, predictions):
        """
        
        """
        return self._bi_match_cached(
            self._freeze_objs(groundtruths),
            self._freeze_objs(predictions)
        )



