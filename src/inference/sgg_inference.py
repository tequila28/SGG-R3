import argparse
import glob
import json
import os
import sys

from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoConfig


current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(src_dir)
sys.path.insert(0, project_root)
sys.path.insert(0, src_dir)

from src.prompt.prompt_cot import PROMPT_PSG, PROMPT_VG150
from src.common.data import load_scene_graph_dataset, resolve_dataset_type
from src.common.inference import add_inference_arguments, build_sampling_params, load_inference_model


def parse_args():
    parser = argparse.ArgumentParser(description="Run vLLM scene graph inference.")
    parser.add_argument("--dataset", required=True, help="Hugging Face dataset name or local dataset path")
    parser.add_argument("--model", required=True, help="Qwen2-VL/Qwen2.5-VL model name or local path")
    parser.add_argument("--output_dir", required=True, help="Directory for per-image JSON outputs")
    parser.add_argument("--dataset_type", choices=["auto", "vg", "psg"], default="auto")
    add_inference_arguments(parser)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Tensor parallel size: {args.tensor_parallel_size}")
    print(f"Batch size: {args.batch_size}")
    print(f"Max model length: {args.max_model_len}")
    print(
        "Sampling: "
        f"max_new_tokens={args.max_new_tokens}, "
        f"temperature={args.temperature}, "
        f"top_k={args.top_k}, "
        f"top_p={args.top_p}, "
        f"repetition_penalty={args.repetition_penalty}"
    )

    sampling_params = build_sampling_params(args)
    dataset_type = resolve_dataset_type(args.dataset, args.dataset_type)
    model_type = AutoConfig.from_pretrained(
        args.model, revision=args.model_revision, trust_remote_code=args.trust_remote_code
    ).model_type
    if model_type not in ("qwen2_vl", "qwen2_5_vl"):
        raise ValueError(f"Unsupported model_type: {model_type}")
    is_qwen2vl = model_type == "qwen2_vl"
    dataset = load_scene_graph_dataset(args.dataset, args.dataset_split, args.dataset_config)
    finished_ids = {
        os.path.basename(path).replace(".json", "")
        for path in glob.glob(os.path.join(args.output_dir, "*.json"))
    }

    dataset = dataset.filter(lambda x: str(x["image_id"]) not in finished_ids)
    print(f"Remaining images to infer: {len(dataset)}")
    if not len(dataset):
        return
    model, processor = load_inference_model(args.model, args)

    def collate_fn(example_list):
        batch_items = []

        for example in example_list:
            image = example["image"].convert("RGB")
            image_id = example["image_id"]
            gt_objs = example["objects"]
            gt_rels = example["relationships"]
            prompt = PROMPT_PSG if dataset_type == "psg" else PROMPT_VG150

            messages = [
                {
                    "role": "system",
                    "content": "You are a scene graph parsing expert. Analyze images step by step to detect objects, their positions, and relationships.",
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt},
                    ],
                },
            ]

            text = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            batch_items.append(
                {
                    "image_id": image_id,
                    "gt_objs": gt_objs,
                    "gt_rels": gt_rels,
                    "image_width": image.size[0],
                    "image_height": image.size[1],
                    "llm_input": {
                        "prompt": text,
                        "multi_modal_data": {"image": image},
                    },
                }
            )

        return batch_items

    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
    )

    for batch_idx, batch_items in enumerate(tqdm(data_loader)):
        batch_llm_inputs = [item["llm_input"] for item in batch_items]
        outputs = model.generate(batch_llm_inputs, sampling_params=sampling_params)

        for item, output in zip(batch_items, outputs):
            image_id = item["image_id"]
            response = output.outputs[0].text
            image_width = item["image_width"]
            image_height = item["image_height"]
            box_scale = [1000.0, 1000.0] if is_qwen2vl else [image_width, image_height]

            result = {
                "image_id": image_id,
                "response": response,
                "gt_objects": item["gt_objs"],
                "gt_relationships": item["gt_rels"],
                "box_scale": box_scale,
            }

            with open(os.path.join(args.output_dir, f"{image_id}.json"), "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)

        if batch_idx % 10 == 0:
            print(f"\nProgress: {batch_idx * args.batch_size}/{len(dataset)} | Batch size: {len(batch_items)}")
            for i, item in enumerate(batch_items):
                response = outputs[i].outputs[0].text
                print(f"  Image {i + 1} ID: {item['image_id']}")
                print(f"  Model output: {response[:100]}...")

    print(f"Inference completed. Processed {len(dataset)} images.")


if __name__ == "__main__":
    main()
