"""Dataset loading without treating the Hugging Face cache as a dataset."""

from pathlib import Path

from datasets import DatasetDict, load_dataset, load_from_disk


def load_scene_graph_dataset(source, split="train", name=None):
    path = Path(source)
    if path.is_dir() and any((path / marker).exists() for marker in ("dataset_dict.json", "state.json")):
        dataset = load_from_disk(str(path))
        if isinstance(dataset, DatasetDict):
            dataset = dataset[split]
        elif split != "train":
            raise ValueError("A single saved Dataset has no named splits; use split='train'.")
    else:
        dataset = load_dataset(source, name=name, split=split)
    missing = {"image_id", "image", "objects", "relationships"} - set(dataset.column_names)
    if missing:
        raise ValueError(f"Missing scene graph columns: {sorted(missing)}")
    if len(dataset) == 0:
        raise ValueError(f"Dataset {source!r}, split {split!r} is empty.")
    return dataset


def resolve_dataset_type(source, dataset_type="auto"):
    if dataset_type in ("vg", "psg"):
        return dataset_type
    source = source.lower()
    if "psg" in source:
        return "psg"
    if "vg" in source or "visual_genome" in source:
        return "vg"
    raise ValueError("Cannot infer the dataset type; pass --dataset_type vg or psg.")
