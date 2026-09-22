# Worker protocol adapted from Hugging Face TRL, Copyright 2020-2026
# The HuggingFace Team. Licensed under the Apache License, Version 2.0.
# https://www.apache.org/licenses/LICENSE-2.0
# Distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND.
# Modifications: SGG image-processing and engine configuration.
"""TRL 1.7.1 server with SGG image-processing and engine options.

The HTTP API and weight synchronization remain upstream TRL implementations.
The worker protocol below follows trl/scripts/vllm_serve.py (Apache-2.0).
"""

import os
from dataclasses import dataclass

from trl import TrlParser
from trl.scripts import vllm_serve


@dataclass
class ServerArguments(vllm_serve.ScriptArguments):
    min_pixels: int = 4 * 28 * 28
    max_pixels: int = 512 * 28 * 28
    seed: int = 42
    max_num_seqs: int = 128
    max_num_batched_tokens: int | None = None


def llm_worker(args, data_parallel_rank, master_port, connection):
    from vllm import LLM

    os.environ["VLLM_DP_RANK"] = str(data_parallel_rank)
    os.environ["VLLM_DP_RANK_LOCAL"] = str(data_parallel_rank)
    os.environ["VLLM_DP_SIZE"] = str(args.data_parallel_size)
    os.environ["VLLM_DP_MASTER_PORT"] = str(master_port)
    extra = {}
    if args.max_num_batched_tokens is not None:
        extra["max_num_batched_tokens"] = args.max_num_batched_tokens
    llm = LLM(
        model=args.model, revision=args.revision, dtype=args.dtype, seed=args.seed,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len, max_num_seqs=args.max_num_seqs,
        enable_prefix_caching=args.enable_prefix_caching, enforce_eager=args.enforce_eager,
        kv_cache_dtype=args.kv_cache_dtype, trust_remote_code=args.trust_remote_code,
        model_impl=args.vllm_model_impl,
        distributed_executor_backend=args.distributed_executor_backend,
        worker_extension_cls="trl.scripts.vllm_serve.WeightSyncWorkerExtension",
        logprobs_mode="processed_logprobs", generation_config="vllm",
        limit_mm_per_prompt={"image": 1, "video": 0},
        mm_processor_kwargs={"min_pixels": args.min_pixels, "max_pixels": args.max_pixels},
        **extra,
    )
    connection.send({"status": "ready"})
    try:
        while True:
            command = connection.recv()
            if command["type"] == "shutdown":
                break
            if command["type"] in ("call", "fire_and_forget"):
                result = getattr(llm, command["method"])(*command.get("args", ()), **command.get("kwargs", {}))
                if command["type"] == "call":
                    connection.send(result)
    except (EOFError, KeyboardInterrupt):
        pass
    finally:
        connection.close()


def main():
    args, = TrlParser(ServerArguments).parse_args_and_config()
    if args.min_pixels <= 0 or args.max_pixels < args.min_pixels:
        raise ValueError("Require 0 < min_pixels <= max_pixels.")
    if args.data_parallel_size != 1:
        raise ValueError("Qwen2-VL/Qwen2.5-VL are dense models; use data_parallel_size=1.")
    if args.speculative_config is not None:
        raise ValueError("Speculative decoding is not configured for this training server.")
    # TRL spawns this top-level worker, preserving its HTTP and weight-sync lifecycle.
    vllm_serve.llm_worker = llm_worker
    vllm_serve.main(args)


if __name__ == "__main__":
    main()
