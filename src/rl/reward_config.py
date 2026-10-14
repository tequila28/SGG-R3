"""Load reward parameters without importing models or GPU libraries."""

import copy
import math
from pathlib import Path

import yaml


DEFAULT_REWARD_CONFIG_PATH = Path(__file__).parent / "configs" / "reward_config.yaml"
REWARD_ORDER = ("format", "category", "node", "edge")


def _read_yaml(path):
    with path.open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a YAML mapping.")
    return value


def _merge(default, override, prefix=""):
    result = copy.deepcopy(default)
    for key, value in override.items():
        name = f"{prefix}.{key}" if prefix else key
        if key not in default:
            raise ValueError(f"Unknown reward parameter: {name}. See src/rl/README.md.")
        if isinstance(default[key], dict):
            if not isinstance(value, dict):
                raise ValueError(f"{name} must be a mapping.")
            result[key] = _merge(default[key], value, name)
        else:
            result[key] = value
    return result


def _validate_values(value, template, prefix=""):
    for key, default in template.items():
        item = value[key]
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(default, dict):
            _validate_values(item, default, name)
        elif isinstance(default, bool):
            if not isinstance(item, bool):
                raise ValueError(f"{name} must be true or false.")
        elif isinstance(default, (int, float)):
            if isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(item) or item < 0:
                raise ValueError(f"{name} must be a finite non-negative number.")
            if isinstance(default, int) and not isinstance(item, int):
                raise ValueError(f"{name} must be an integer.")
        elif not isinstance(item, str) or not item.strip():
            raise ValueError(f"{name} must be a non-empty string.")


def load_reward_config(config_path=None):
    default = _read_yaml(DEFAULT_REWARD_CONFIG_PATH)
    path = Path(config_path) if config_path else DEFAULT_REWARD_CONFIG_PATH
    if not path.is_absolute() and not path.exists():
        path = Path(__file__).resolve().parents[2] / path
    override = _read_yaml(path)
    config = _merge(default, override)
    _validate_values(config, default)
    config["dataset"]["type"] = config["dataset"]["type"].lower()
    if config["dataset"]["type"] not in ("vg", "psg"):
        raise ValueError("dataset.type must be vg or psg.")
    if config["numerics"]["epsilon"] <= 0 or sum(config["weights"].values()) <= 0:
        raise ValueError("epsilon and the sum of reward weights must be positive.")
    matching = config["matching"]
    if sum(matching[k] for k in ("semantic_weight", "iou_weight", "box_l1_weight")) <= 0:
        raise ValueError("At least one matching weight must be positive.")
    if matching["padding_cost"] <= 0:
        raise ValueError("matching.padding_cost must be positive.")
    node = config["stage2_node"]
    if node["box"]["iou_weight"] + node["box"]["l1_weight"] <= 0:
        raise ValueError("stage2_node.box weights must have a positive sum.")
    if not 0 <= node["recall"]["iou_threshold"] <= 1:
        raise ValueError("stage2_node.recall.iou_threshold must be in [0, 1].")
    coarse = config["stage3_edge"]["coarse"]
    if not 0 <= coarse["cluster_threshold"] < 1 or not 0 <= coarse["match_threshold"] <= 1:
        raise ValueError("cluster_threshold must be in [0, 1); match_threshold must be in [0, 1].")
    if coarse["min_cluster_size"] < 1:
        raise ValueError("stage3_edge.coarse.min_cluster_size must be >= 1.")
    relation = config["stage3_edge"]["relation_weight"]
    if relation["base_weight"] <= 0 or relation["default_weight"] <= 0 or relation["frequency_floor"] <= 0:
        raise ValueError("Relation base_weight, default_weight and frequency_floor must be positive.")
    frequencies_path = Path(relation["frequencies_path"])
    # Partial override files can reuse the bundled table without copying it.
    override_relation = override.get("stage3_edge", {}).get("relation_weight", {})
    base = path.parent if "frequencies_path" in override_relation else DEFAULT_REWARD_CONFIG_PATH.parent
    if not frequencies_path.is_absolute():
        frequencies_path = base / frequencies_path
    relation["frequencies_path"] = str(frequencies_path.resolve())
    return config


def load_relation_frequencies(config):
    tables = _read_yaml(Path(config["stage3_edge"]["relation_weight"]["frequencies_path"]))
    data = tables.get(config["dataset"]["type"])
    if not isinstance(data, dict) or not data:
        raise ValueError("Frequency table must contain a non-empty mapping for the selected dataset.")
    for predicate, frequency in data.items():
        if not isinstance(predicate, str) or not predicate.strip():
            raise ValueError("Frequency keys must be quoted predicate strings (especially 'on').")
        if isinstance(frequency, bool) or not isinstance(frequency, (int, float)) or not math.isfinite(frequency) or frequency <= 0:
            raise ValueError(f"Frequency for {predicate!r} must be finite and positive.")
    if config["stage3_edge"]["relation_weight"]["frequency_floor"] > min(data.values()):
        raise ValueError("frequency_floor must not exceed the smallest frequency in the selected table.")
    return data


def get_reward_weights(config):
    return [config["weights"][name] for name in REWARD_ORDER]
