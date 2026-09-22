import time
from functools import lru_cache

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from src.prompt.data_category import PSG_OBJ_CATEGORIES, PSG_PREDICATES, VG150_OBJ_CATEGORIES, VG150_PREDICATES


class SemanticSimilarityManager:
    """Manage Sentence-BERT embeddings for category and predicate similarity."""

    def __init__(self, dataset_type: str = "vg"):
        self.dataset_type = dataset_type.lower()
        if self.dataset_type not in ["vg", "psg"]:
            raise ValueError(f"Unsupported dataset_type: {self.dataset_type}; expected 'vg' or 'psg'")

        self.device_str, self.device_id, self.device_name = self._get_current_cuda_device()
        self.is_first_process = self._is_first_gpu_process()
        self.word_model = self._initialize_model()
        self.get_doc_cached = self._initialize_embeddings()

        if self.dataset_type == "vg":
            self.object_categories = VG150_OBJ_CATEGORIES
            self.predicates = VG150_PREDICATES
        else:
            self.object_categories = PSG_OBJ_CATEGORIES
            self.predicates = PSG_PREDICATES

    def _get_current_cuda_device(self):
        if torch.cuda.is_available():
            current_device = torch.cuda.current_device()
            device_name = torch.cuda.get_device_name(current_device)
            return f"cuda:{current_device}", current_device, device_name
        return "cpu", None, "CPU"

    def _is_first_gpu_process(self):
        if torch.cuda.is_available():
            return torch.cuda.current_device() == 0
        return True

    def _initialize_model(self):
        if self.is_first_process:
            print(f"Current device: {self.device_str} ({self.device_name})")
            print(f"Loading Sentence-BERT model for {self.dataset_type.upper()}...")

        start_time = time.time()
        model = SentenceTransformer("all-MiniLM-L6-v2")
        model = model.to(self.device_str)
        model.eval()

        if self.is_first_process:
            print(f"Sentence-BERT loaded in {time.time() - start_time:.2f}s")
            print(f"Model device: {next(model.parameters()).device}")

        return model

    def _initialize_embeddings(self):
        @lru_cache(maxsize=None)
        def get_doc_cached(word):
            return self.word_model.encode(word)

        if self.is_first_process:
            print(f"Initializing {self.dataset_type.upper()} category and predicate embeddings")

        if self.dataset_type == "vg":
            items = VG150_OBJ_CATEGORIES + VG150_PREDICATES
        else:
            items = PSG_OBJ_CATEGORIES + PSG_PREDICATES

        for item in items:
            get_doc_cached(item)

        if self.is_first_process:
            print("Embedding cache initialized")

        return get_doc_cached

    def category_semantic_similarity(self, pred_id, gt_id):
        pred_category = pred_id.split(".")[0].lower().strip()
        gt_category = gt_id.split(".")[0].lower().strip()

        if pred_category == gt_category:
            return 1.0
        if not pred_category or not gt_category:
            return 0.0

        try:
            pred_embedding = self.get_doc_cached(pred_category)
            gt_embedding = self.get_doc_cached(gt_category)
            similarity = np.dot(pred_embedding, gt_embedding) / (
                np.linalg.norm(pred_embedding) * np.linalg.norm(gt_embedding)
            )
            return max(0.0, min(1.0, float(similarity)))
        except Exception:
            return 0.0

    def get_dataset_type(self):
        return self.dataset_type

    def clear_cache(self):
        self.get_doc_cached.cache_clear()
        if self.is_first_process:
            print("Semantic similarity cache cleared")
