import os
import json
import torch
import torch.distributed as dist
from PIL import Image
from tqdm import tqdm
from dataclasses import dataclass
from transformers import AutoProcessor, AutoConfig
from datasets import load_dataset
import argparse
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from vllm import LLM, SamplingParams
from huggingface_hub import snapshot_download
import base64
from io import BytesIO

from prompt_data import PromptBuilder

# 设置NCCL超时参数
os.environ["NCCL_SOCKET_TIMEOUT"] = "3600"  
os.environ["NCCL_BLOCKING_WAIT"] = "1"

@dataclass
class SimpleStatistics:
    """简化统计信息类"""
    total_processed: int = 0
    successful: int = 0
    failed: int = 0
    
    def to_dict(self):
        return {
            "total_processed": self.total_processed,
            "successful": self.successful,
            "failed": self.failed,
            "success_rate": round(self.successful / max(self.total_processed, 1) * 100, 2)
        }
    
    def merge(self, other):
        """合并统计信息"""
        self.total_processed += other.total_processed
        self.successful += other.successful
        self.failed += other.failed

@dataclass
class InferenceConfig:
    dataset_path: str
    output_file: str
    temp_output_file: str
    stats_file: str
    model_name: str
    max_new_tokens: int
    temperature: float
    top_p: float
    batch_size_per_gpu: int
    num_gpus: int
    max_model_len: int
    save_interval: int
    dataset_type: str

def setup_distributed():
    """初始化分布式环境"""
    try:
        if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
            rank = int(os.environ["RANK"])
            world_size = int(os.environ['WORLD_SIZE'])
            local_rank = int(os.environ['LOCAL_RANK'])
            dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
            torch.cuda.set_device(local_rank)
            return rank, world_size, local_rank
        return 0, 1, 0
    except Exception as e:
        print(f"Distributed setup failed: {e}")
        return 0, 1, 0

def load_local_dataset(dataset_path):
    """加载本地数据集"""
    try:
        dataset = load_dataset(dataset_path)
        print(f"Dataset loaded successfully, containing {len(dataset['train'])} samples")
        return dataset
    except Exception as e:
        print(f"Dataset loading failed: {e}")
        return None

class VLInferencePipeline:
    def __init__(self, config, local_rank):
        self.config = config
        self.local_rank = local_rank
        self.device = torch.device(f"cuda:{local_rank}")
        self.prompt_builder = PromptBuilder(config.dataset_type)
        self.statistics = SimpleStatistics()  # 初始化统计信息
        
        # 基本CUDA设置
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        if local_rank == 0:
            print(f"Loading Qwen2.5-VL-7B-Instruct model...")
        
        # 使用vLLM加载模型
        min_pixels = 4 * 28 * 28
        max_pixels = 1024 * 28 * 28
        
        try:
            local_model_path = snapshot_download(config.model_name)
            print(f"Set model:{config.model_name} to local path:", local_model_path)
            config.model_name = local_model_path
        except:
            pass
        
        self.processor = AutoProcessor.from_pretrained(
            config.model_name,
            min_pixels=min_pixels,
            max_pixels=max_pixels
        )
        
        # 使用vLLM加载模型
        self.model = LLM(
            model=config.model_name,
            limit_mm_per_prompt={"image": 1},
            dtype='bfloat16',
            device=f"cuda:{local_rank}",
            max_model_len=config.max_model_len,
            mm_processor_kwargs={"max_pixels": max_pixels, "min_pixels": min_pixels},
        )
        
        # 配置采样参数
        self.sampling_params = SamplingParams(
            temperature=config.temperature,
            top_p=config.top_p,
            max_tokens=config.max_new_tokens,
            repetition_penalty=1.0
        )
        
        if local_rank == 0:
            print(f"Model loaded on GPU {local_rank}")
            print("Model loading completed!")

    def collate_fn(self, batch):
        """修复后的批处理函数"""
        images = []
        texts = []
        image_ids = []
        objects_list = []      # 存储预处理后的物体
        relationships_list = [] # 存储预处理后的关系
        preprocessed_samples = [] # 存储完整的预处理样本
        
        for sample in batch:
            # 预处理样本（与prompt构建一致）
            preprocessed_sample = self.prompt_builder.preprocess_sample(sample)
            
            # 使用预处理后的数据构建prompt
            prompt, image = self.prompt_builder.construct_relationship_augmentation_prompt(preprocessed_sample)
            
            images.append(image)
            texts.append(prompt)
            image_ids.append(sample.get('image_id', 'unknown'))
            objects_list.append(preprocessed_sample['objects'])  # ✅ 预处理后的物体
            relationships_list.append(preprocessed_sample['relationships'])  # ✅ 预处理后的关系
            preprocessed_samples.append(preprocessed_sample)
            
        return {
            'images': images,
            'texts': texts,
            'image_ids': image_ids,
            'objects': objects_list,
            'relationships': relationships_list,
            'preprocessed_samples': preprocessed_samples  # 新增，用于后续处理
        }

    def generate_relationships_batch(self, batch_data):
        """修复后的批量生成关系增强"""
        try:
            torch.cuda.empty_cache()
            
            images = batch_data['images']
            texts = batch_data['texts']
            relationships_list = batch_data['relationships']  # ✅ 现在已经是预处理后的数据
            objects_list = batch_data['objects']              # ✅ 现在已经是预处理后的数据
            
            # 构建vLLM输入格式
            messages_batch = []
            for image, text in zip(images, texts):
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{self._encode_image_to_base64(image)}"}},
                            {"type": "text", "text": text},
                        ],
                    }
                ]
                messages_batch.append(messages)
            
            # 使用vLLM进行推理
            outputs = self.model.chat(messages_batch, sampling_params=self.sampling_params)
            responses = [output.outputs[0].text for output in outputs]

            # 解析响应 - 直接使用预处理后的数据
            parsed_results = []
            used_fallback = False
            
            for i, response in enumerate(responses):
                # 直接使用预处理后的数据（与prompt构建时一致）
                existing_relationships = json.loads(relationships_list[i])
                available_objects = json.loads(objects_list[i])
                
                parsed_result = self.prompt_builder.parse_relationship_response(
                    response, 
                    existing_relationships,
                    available_objects
                )
                parsed_results.append(parsed_result)
            
            torch.cuda.empty_cache()
            return parsed_results, used_fallback
            
        except Exception as e:
            print(f"Error generating batch relationships on GPU {self.local_rank}: {e}")
            # 回退方案也使用预处理后的数据
            fallback_results = []
            for i in range(len(batch_data['images'])):
                existing_relationships = json.loads(batch_data['relationships'][i])
                available_objects = json.loads(batch_data['objects'][i])
                
                fallback_result = self.prompt_builder.fallback_categorize_relationships(
                    existing_relationships, 
                    available_objects
                )
                fallback_results.append(fallback_result)
            
            return fallback_results, True
    

    def _encode_image_to_base64(self, image):
        """将PIL图像编码为base64"""
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def _save_results(self, results, file_path):
        """保存结果到文件"""
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write("[\n")
                
                for i, (image_id, data) in enumerate(results.items()):
                    f.write("  {\n")
                    f.write(f'    "image_id": "{image_id}",\n')
                    
                    objects_str = json.dumps(data["objects"], ensure_ascii=False, separators=(',', ':'))
                    f.write(f'    "objects": {objects_str},\n')
                    
                    # 修改这里：将"relationships"改为"relation"
                    f.write('    "relations": {\n')
                    
                    if self.config.dataset_type == "psg":
                        categories = ["spatial_relations", "static_action_relations", "dynamic_action_relations"]
                    else:
                        categories = ["spatial_relations", "possession_relations", "interaction_relations"]
                    
                    for j, category in enumerate(categories):
                        f.write(f'      "{category}": [\n')
                        relations = data["relations"].get(category, [])
                        for k, rel in enumerate(relations):
                            rel_str = json.dumps(rel, ensure_ascii=False, separators=(',', ':'))
                            if k < len(relations) - 1:
                                f.write(f'        {rel_str},\n')
                            else:
                                f.write(f'        {rel_str}\n')
                        
                        if j < len(categories) - 1:
                            f.write('      ],\n')
                        else:
                            f.write('      ]\n')
                    
                    f.write('    }\n')
                    
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
                output_data.append({
                    "image_id": image_id,
                    "objects": data["objects"],
                    "relations": data["relations"]  # 修改这里
                })
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
            print(f"Results saved with default format to {file_path}")

    def _save_statistics(self, statistics, file_path):
        """保存统计信息"""
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(statistics.to_dict(), f, ensure_ascii=False, indent=2)
            print(f"Statistics saved to {file_path}")
        except Exception as e:
            print(f"Error saving statistics: {e}")

    def _gather_results(self, local_results, rank):
        """收集所有GPU的结果到主GPU"""
        if not dist.is_initialized():
            return local_results
            
        gathered_list = [None] * dist.get_world_size() if rank == 0 else None
        dist.gather_object(local_results, gathered_list, dst=0)
        
        if rank == 0:
            final_results = {}
            for result in gathered_list:
                if result:
                    final_results.update(result)
            return final_results
        return None

    def _gather_statistics(self, local_stats, rank):
        """收集所有GPU的统计信息到主GPU"""
        if not dist.is_initialized():
            return local_stats
            
        gathered_list = [None] * dist.get_world_size() if rank == 0 else None
        dist.gather_object(local_stats, gathered_list, dst=0)
        
        if rank == 0:
            final_stats = SimpleStatistics()
            for stats in gathered_list:
                if stats:
                    final_stats.merge(stats)
            return final_stats
        return None

    def process_dataset(self, rank, world_size):
        """处理整个数据集（分布式版本）"""
        try:
            dataset = load_dataset(self.config.dataset_path)
            if not dataset:
                raise ValueError("Dataset loading failed")
            
            sampler = DistributedSampler(
                dataset['train'], 
                num_replicas=world_size, 
                rank=rank,
                shuffle=False
            )
            
            dataloader = torch.utils.data.DataLoader(
                dataset['train'],
                batch_size=self.config.batch_size_per_gpu,
                sampler=sampler,
                collate_fn=self.collate_fn,
                num_workers=4,
                pin_memory=True
            )
            
            local_results = {}  # 当前GPU的结果
            processed_count = 0
            
            if rank == 0:
                pbar = tqdm(dataloader, desc="Augmenting relationships")
            else:
                pbar = dataloader
            
            for batch_idx, batch_data in enumerate(pbar):
                if rank == 0:
                    print(f"Processing batch {batch_idx + 1}/{len(dataloader)}")
                
                # 生成关系
                relationship_results, used_fallback = self.generate_relationships_batch(batch_data)
                
                # 更新统计信息
                batch_size = len(batch_data['image_ids'])
                self.statistics.total_processed += batch_size
                
                if used_fallback:
                    self.statistics.failed += batch_size  # 回滚就是失败
                    if rank == 0:
                        print(f"Batch {batch_idx + 1}: Used fallback for {batch_size} samples")
                else:
                    self.statistics.successful += batch_size  # 没回滚就是成功
                
                # 处理批结果 - 修改这里：将"relationships"改为"relation"
                for i, (image_id, relation_data) in enumerate(zip(
                    batch_data['image_ids'], relationship_results
                )):
                    if relation_data:
                        preprocessed_sample = self.prompt_builder.preprocess_sample({
                            'objects': batch_data['objects'][i],
                            'relationships': batch_data['relationships'][i]
                        })
                        objects_data = preprocessed_sample['objects']
                        
                        local_results[image_id] = {
                            "objects": json.loads(objects_data),
                            "relations": relation_data  # 修改这里
                        }
                        processed_count += 1
                
                # 所有进程都检查保存间隔
                if (batch_idx + 1) % self.config.save_interval == 0:
                    # 所有进程都参与数据收集
                    gathered_results = self._gather_results(local_results, rank)
                    gathered_stats = self._gather_statistics(self.statistics, rank)
                    
                    # 只有rank 0执行保存操作
                    if rank == 0:
                        if gathered_results is not None:
                            self._save_results(gathered_results, self.config.temp_output_file)
                            print(f"✓ Successfully saved {len(gathered_results)} samples")
                        if gathered_stats is not None:
                            self._save_statistics(gathered_stats, self.config.stats_file)
                            print(f"Intermediate statistics: {gathered_stats.to_dict()}")
                            
            # 最终收集和保存
            final_results = self._gather_results(local_results, rank)
            final_stats = self._gather_statistics(self.statistics, rank)
            
            if rank == 0 and final_results is not None and final_stats is not None:
                self._save_results(final_results, self.config.output_file)
                self._save_statistics(final_stats, self.config.stats_file)
                
                # 打印最终统计信息
                stats_dict = final_stats.to_dict()
                print(f"\n=== Processing Completed ===")
                print(f"Total processed: {stats_dict['total_processed']}")
                print(f"Successful: {stats_dict['successful']}")
                print(f"Failed: {stats_dict['failed']}")
                print(f"Success rate: {stats_dict['success_rate']}%")
                print(f"Generated {len(final_results)} samples")
            
            dist.barrier()
            return local_results
            
        except Exception as e:
            print(f"Error in process_dataset (rank {rank}): {str(e)}")
            import traceback
            traceback.print_exc()
            if dist.is_initialized():
                dist.barrier()
            return None

def main():
    # 初始化分布式环境
    rank, world_size, local_rank = setup_distributed()
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="Visual Language Inference Pipeline")
    parser.add_argument('--local_rank', type=int, default=0, help='Local rank for distributed training')
    parser.add_argument('--dataset_type', type=str, default='vg', choices=['vg', 'psg'],                 #数据集类别
                       help='Dataset type: vg or psg')
    parser.add_argument('--dataset_path', type=str, default='JosephZ/vg150_train_sgg_prompt',            #数据集路径
                       help='Path to the dataset')
    parser.add_argument('--output_file', type=str, default='/root/SGG-R3/relationship_augmentation.json',  #增强后数据集保存路径
                       help='Output file path for final results')
    parser.add_argument('--temp_output_file', type=str, default='temp_relationship_augmentation.json',
                       help='Temporary output file path for intermediate results')
    parser.add_argument('--stats_file', type=str, default='/root/R1-SGG/processing_statistics.json',
                       help='Statistics file path')
    parser.add_argument('--model_name', type=str, default='/root/R1-SGG/models/Qwen2.5-VL-7B-Instruct',     #使用模型路径
                       help='Model name or path')
    parser.add_argument('--max_new_tokens', type=int, default=1024,
                       help='Maximum number of new tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.2,
                       help='Temperature for sampling')
    parser.add_argument('--top_p', type=float, default=0.9,
                       help='Top-p for nucleus sampling')
    parser.add_argument('--batch_size_per_gpu', type=int, default=16,
                       help='Batch size per GPU')
    parser.add_argument('--num_gpus', type=int, default=8,
                       help='Number of GPUs to use')
    parser.add_argument('--max_model_len', type=int, default=8192,
                       help='Maximum model length')
    parser.add_argument('--save_interval', type=int, default=100,
                       help='Save interval for intermediate results')
    
    args = parser.parse_args()
    
    # 初始化配置
    config = InferenceConfig(
        dataset_path=args.dataset_path,
        output_file=args.output_file,
        temp_output_file=args.temp_output_file,
        stats_file=args.stats_file,
        model_name=args.model_name,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        batch_size_per_gpu=args.batch_size_per_gpu,
        num_gpus=args.num_gpus,
        max_model_len=args.max_model_len,
        save_interval=args.save_interval,
        dataset_type=args.dataset_type
    )
    
    # 根据数据集类型调整输出文件路径
    if config.dataset_type == "psg":
        config.output_file = config.output_file.replace(".json", "_psg.json")
        config.temp_output_file = config.temp_output_file.replace(".json", "_psg.json")
        config.stats_file = config.stats_file.replace(".json", "_psg.json")
    
    if rank == 0:
        print(f"Using dataset type: {config.dataset_type}")
        print(f"Output file: {config.output_file}")
        print(f"Statistics file: {config.stats_file}")
    
    try:
        # 运行推理管道
        pipeline = VLInferencePipeline(config, local_rank)
        pipeline.process_dataset(rank, world_size)
    finally:
        # 清理分布式环境
        if dist.is_initialized():
            dist.destroy_process_group()
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()