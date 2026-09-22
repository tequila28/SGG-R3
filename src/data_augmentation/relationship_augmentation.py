import argparse
import json
import os
import sys
from dataclasses import dataclass

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm


current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(src_dir)
sys.path.insert(0, project_root)
sys.path.insert(0, src_dir)

from src.data_augmentation.prompt_data import PromptBuilder
from src.common.data import load_scene_graph_dataset
from src.common.inference import (
    InferenceOptions, add_inference_arguments, build_sampling_params, load_inference_model,
)


@dataclass
class SimpleStatistics:
    """Track relationship augmentation progress."""

    total_processed: int = 0
    successful: int = 0
    failed: int = 0

    def to_dict(self):
        return {
            "total_processed": self.total_processed,
            "successful": self.successful,
            "failed": self.failed,
            "success_rate": round(self.successful / max(self.total_processed, 1) * 100, 2),
        }

    def merge(self, other):
        self.total_processed += other.total_processed
        self.successful += other.successful
        self.failed += other.failed


@dataclass
class InferenceConfig(InferenceOptions):
    dataset_path: str
    output_file: str
    temp_output_file: str
    stats_file: str
    model_name: str
    save_interval: int
    dataset_type: str


def extract_dataset_name(dataset_path):
    """Extract a stable dataset name from a local path or Hugging Face dataset ID."""
    if os.path.exists(dataset_path):
        return os.path.basename(dataset_path.rstrip("/"))

    if "/" in dataset_path:
        parts = dataset_path.split("/")
        if len(parts) >= 2:
            return parts[-1]

    return dataset_path


def adjust_output_paths(config):
    """Append the dataset name to output files and create parent directories."""
    dataset_name = extract_dataset_name(config.dataset_path)

    def get_base_path(file_path):
        directory = os.path.dirname(file_path)
        filename = os.path.basename(file_path)
        name, ext = os.path.splitext(filename)
        return directory, name, ext

    output_dir, output_name, output_ext = get_base_path(config.output_file)
    temp_dir, temp_name, temp_ext = get_base_path(config.temp_output_file)
    stats_dir, stats_name, stats_ext = get_base_path(config.stats_file)

    for directory in [output_dir, temp_dir, stats_dir]:
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

    config.output_file = os.path.join(output_dir if output_dir else ".", f"{output_name}_{dataset_name}{output_ext}")
    config.temp_output_file = os.path.join(temp_dir if temp_dir else ".", f"{temp_name}_{dataset_name}{temp_ext}")
    config.stats_file = os.path.join(stats_dir if stats_dir else ".", f"{stats_name}_{dataset_name}{stats_ext}")

    return config


def detect_dataset_type(dataset_path):
    """Infer whether the dataset follows the PSG or VG scene graph format."""
    dataset_path_lower = dataset_path.lower()

    if any(keyword in dataset_path_lower for keyword in ["psg", "panoptic", "scene_graph", "scene-graph"]):
        return "psg"
    if any(keyword in dataset_path_lower for keyword in ["vg", "visual_genome", "visual-genome"]):
        return "vg"

    try:
        dataset = load_dataset(dataset_path, split="train[:1]")
        sample = dataset[0]

        if "panoptic_segmentation" in sample or "scene_graph" in sample:
            return "psg"
        if "objects" in sample and "relationships" in sample:
            return "vg"

        if any("panoptic" in key.lower() for key in sample.keys()):
            return "psg"
        if any("relation" in key.lower() or "object" in key.lower() for key in sample.keys()):
            return "vg"
    except Exception as e:
        print(f"Warning: Could not automatically detect dataset type from path {dataset_path}: {e}")
        print("Defaulting to 'vg' dataset type")

    return "vg"


class VLInferencePipeline:
    def __init__(self, config):
        self.config = config
        self.prompt_builder = PromptBuilder(config.dataset_type)
        self.statistics = SimpleStatistics()

        print(f"Tensor parallel size: {config.tensor_parallel_size}")
        print(f"Batch size: {config.batch_size}")
        print(f"Max model length: {config.max_model_len}")
        print(
            "Sampling: "
            f"max_new_tokens={config.max_new_tokens}, "
            f"temperature={config.temperature}, "
            f"top_k={config.top_k}, "
            f"top_p={config.top_p}, "
            f"repetition_penalty={config.repetition_penalty}"
        )

        self.sampling_params = build_sampling_params(config)
        self.model, self.processor = load_inference_model(config.model_name, config)

        print("Model loading completed.")

    def collate_fn(self, batch):
        """Build a vLLM-ready batch from dataset samples."""
        llm_inputs = []
        image_ids = []
        objects_list = []
        relationships_list = []
        preprocessed_samples = []

        for sample in batch:
            preprocessed_sample = self.prompt_builder.preprocess_sample(sample)
            prompt, image = self.prompt_builder.construct_relationship_augmentation_prompt(preprocessed_sample)

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]

            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            llm_inputs.append(
                {
                    "prompt": text,
                    "multi_modal_data": {"image": image},
                }
            )
            image_ids.append(sample.get("image_id", "unknown"))
            objects_list.append(preprocessed_sample["objects"])
            relationships_list.append(preprocessed_sample["relationships"])
            preprocessed_samples.append(preprocessed_sample)

        return {
            "llm_inputs": llm_inputs,
            "image_ids": image_ids,
            "objects": objects_list,
            "relationships": relationships_list,
            "preprocessed_samples": preprocessed_samples,
        }

    def generate_relationships_batch(self, batch_data):
        """Generate augmented relationships for one batch."""
        try:
            llm_inputs = batch_data["llm_inputs"]
            relationships_list = batch_data["relationships"]
            objects_list = batch_data["objects"]

            outputs = self.model.generate(llm_inputs, sampling_params=self.sampling_params)
            responses = [output.outputs[0].text for output in outputs]
            parsed_results = []
            used_fallback = False

            for i, response in enumerate(responses):
                existing_relationships = json.loads(relationships_list[i])
                available_objects = json.loads(objects_list[i])
                parsed_result = self.prompt_builder.parse_relationship_response(
                    response,
                    existing_relationships,
                    available_objects,
                )
                parsed_results.append(parsed_result)

            torch.cuda.empty_cache()
            return parsed_results, used_fallback
        except Exception as e:
            print(f"Error generating batch relationships: {e}")
            import traceback

            traceback.print_exc()

            fallback_results = []
            for i in range(len(batch_data["llm_inputs"])):
                existing_relationships = json.loads(batch_data["relationships"][i])
                available_objects = json.loads(batch_data["objects"][i])
                fallback_result = self.prompt_builder.fallback_categorize_relationships(
                    existing_relationships,
                    available_objects,
                )
                fallback_results.append(fallback_result)

            return fallback_results, True

    def _save_results(self, results, file_path):
        """Save augmentation results."""
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write("[\n")

                for i, (image_id, data) in enumerate(results.items()):
                    f.write("  {\n")
                    f.write(f'    "image_id": "{image_id}",\n')

                    objects_str = json.dumps(data["objects"], ensure_ascii=False, separators=(",", ":"))
                    f.write(f'    "objects": {objects_str},\n')
                    f.write('    "relations": {\n')

                    if self.config.dataset_type == "psg":
                        categories = ["spatial_relations", "static_action_relations", "dynamic_action_relations"]
                    else:
                        categories = ["spatial_relations", "possession_relations", "interaction_relations"]

                    for j, category in enumerate(categories):
                        f.write(f'      "{category}": [\n')
                        relations = data["relations"].get(category, [])
                        for k, rel in enumerate(relations):
                            rel_str = json.dumps(rel, ensure_ascii=False, separators=(",", ":"))
                            if k < len(relations) - 1:
                                f.write(f"        {rel_str},\n")
                            else:
                                f.write(f"        {rel_str}\n")

                        if j < len(categories) - 1:
                            f.write("      ],\n")
                        else:
                            f.write("      ]\n")

                    f.write("    }\n")

                    if i < len(results) - 1:
                        f.write("  },\n")
                    else:
                        f.write("  }\n")

                f.write("]\n")

            print(f"Results saved to {file_path}, total {len(results)} samples")
        except Exception as e:
            print(f"Error saving results to {file_path}: {e}")
            output_data = []
            for image_id, data in results.items():
                output_data.append(
                    {
                        "image_id": image_id,
                        "objects": data["objects"],
                        "relations": data["relations"],
                    }
                )

            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
            print(f"Results saved with default format to {file_path}")

    def _save_statistics(self, statistics, file_path):
        """Save processing statistics."""
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(statistics.to_dict(), f, ensure_ascii=False, indent=2)
            print(f"Statistics saved to {file_path}")
        except Exception as e:
            print(f"Error saving statistics: {e}")

    def process_dataset(self):
        """Run augmentation over the full train split."""
        try:
            train_dataset = load_scene_graph_dataset(
                self.config.dataset_path, self.config.dataset_split, self.config.dataset_config
            )
            print(f"Dataset loaded successfully, containing {len(train_dataset)} samples")

            dataloader = DataLoader(
                train_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                collate_fn=self.collate_fn,
                num_workers=self.config.num_workers,
                pin_memory=False,
            )

            results = {}
            processed_count = 0
            pbar = tqdm(dataloader, desc="Augmenting relationships")

            for batch_idx, batch_data in enumerate(pbar):
                relationship_results, used_fallback = self.generate_relationships_batch(batch_data)
                batch_size = len(batch_data["image_ids"])
                self.statistics.total_processed += batch_size

                if used_fallback:
                    self.statistics.failed += batch_size
                    print(f"Batch {batch_idx + 1}: Used fallback for {batch_size} samples")
                else:
                    self.statistics.successful += batch_size

                for image_id, relation_data, objects_json, relationships_json in zip(
                    batch_data["image_ids"],
                    relationship_results,
                    batch_data["objects"],
                    batch_data["relationships"],
                ):
                    if relation_data:
                        preprocessed_sample = self.prompt_builder.preprocess_sample(
                            {
                                "objects": objects_json,
                                "relationships": relationships_json,
                            }
                        )
                        results[image_id] = {
                            "objects": json.loads(preprocessed_sample["objects"]),
                            "relations": relation_data,
                        }
                        processed_count += 1

                if (batch_idx + 1) % self.config.save_interval == 0:
                    self._save_results(results, self.config.temp_output_file)
                    self._save_statistics(self.statistics, self.config.stats_file)
                    print(f"Intermediate save: {len(results)} samples processed")

                pbar.set_postfix(
                    {
                        "Processed": processed_count,
                        "Success": self.statistics.successful,
                        "Failed": self.statistics.failed,
                    }
                )

            self._save_results(results, self.config.output_file)
            self._save_statistics(self.statistics, self.config.stats_file)

            stats_dict = self.statistics.to_dict()
            print("\n=== Processing Completed ===")
            print(f"Total processed: {stats_dict['total_processed']}")
            print(f"Successful: {stats_dict['successful']}")
            print(f"Failed: {stats_dict['failed']}")
            print(f"Success rate: {stats_dict['success_rate']}%")
            print(f"Generated {len(results)} samples")
            return results
        except Exception as e:
            print(f"Error in process_dataset: {e}")
            import traceback

            traceback.print_exc()
            raise


def parse_args():
    parser = argparse.ArgumentParser(description="Run relationship augmentation with vLLM.")
    parser.add_argument("--dataset_type", type=str, default="auto", choices=["vg", "psg", "auto"])
    parser.add_argument("--dataset_path", type=str, default="JosephZ/vg150_train_sgg_prompt")
    parser.add_argument("--output_file", type=str, default="./augmentation_results/relationship_augmentation.json")
    parser.add_argument("--temp_output_file", type=str, default="temp_relationship_augmentation.json")
    parser.add_argument("--stats_file", type=str, default="./augmentation_results/processing_statistics.json")
    parser.add_argument("--model_name", type=str, required=True)
    add_inference_arguments(
        parser, max_new_tokens=1024, temperature=0.2, top_p=0.9,
        repetition_penalty=1.0, batch_size=16, max_model_len=8192,
    )
    parser.add_argument("--save_interval", type=int, default=2, help="Save interval in batches")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.dataset_type == "auto":
        args.dataset_type = detect_dataset_type(args.dataset_path)
        print(f"Auto-detected dataset type: {args.dataset_type} from path: {args.dataset_path}")

    if args.save_interval <= 0:
        raise ValueError("save_interval must be positive.")
    config = InferenceConfig(**vars(args))

    config = adjust_output_paths(config)

    print(f"Using dataset type: {config.dataset_type}")
    print(f"Dataset path: {config.dataset_path}")
    print(f"Output file: {config.output_file}")
    print(f"Temp output file: {config.temp_output_file}")
    print(f"Statistics file: {config.stats_file}")

    try:
        pipeline = VLInferencePipeline(config)
        pipeline.process_dataset()
    except Exception as e:
        print(f"Error in main: {e}")
        import traceback

        traceback.print_exc()
        raise
    finally:
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
