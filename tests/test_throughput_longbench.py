"""
LongBench-v2 throughput smoke test with optional multi-batch (e.g. bs=8).

Truncates inputs, forces a fixed decode length via disable_early_stop.
"""

import argparse
import os
import sys

import torch
from termcolor import colored

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _REPO_ROOT)

from datasets import load_dataset
from transformers import AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_type",
        type=str,
        default="llama",
        choices=["llama", "qwen"],
        help="llama or qwen model hub entrypoint",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="HF repo id or local model directory (required unless defaults are set below)",
    )
    parser.add_argument("--batch_size", type=int, default=1, help="Parallel samples in one batch")
    parser.add_argument("--max_input_tokens", type=int, default=64000, help="Truncate input length")
    parser.add_argument("--max_new_tokens", type=int, default=128, help="Timed decode length")
    parser.add_argument("--attention_type", type=str, default="PolarANN")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--length", type=str, default="long", help="long/medium/short")
    parser.add_argument("--difficulty", type=str, default="easy")
    parser.add_argument("--sink_size", type=int, default=128)
    parser.add_argument("--local_size", type=int, default=512)
    parser.add_argument("--dynamic_update_interval", type=int, default=512)
    parser.add_argument("--enable_offload", type=bool, default=True, help="CPU KV offload")
    parser.add_argument("--warmup_steps", type=int, default=32, help="Warmup decode steps before timed region")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.model_name is None:
        args.model_name = os.environ.get(
            "PARISKV_MODEL_NAME",
            "meta-llama/Llama-3.1-8B-Instruct"
            if args.model_type == "llama"
            else "Qwen/Qwen3-8B",
        )

    print(colored("=" * 60, "cyan"))
    print(colored("  LongBench-v2 throughput (multi-batch)", "cyan"))
    print(colored("=" * 60, "cyan"))
    print(f"  model_type={args.model_type}")
    print(f"  model_name={args.model_name}")
    print(f"  batch_size={args.batch_size}")
    print(f"  max_input_tokens={args.max_input_tokens:,}")
    print(f"  max_new_tokens={args.max_new_tokens}")
    print(f"  length={args.length}")
    print(f"  attention_type={args.attention_type}")
    if args.attention_type == "PolarANN":
        print(f"  enable_offload={args.enable_offload}")
    print(colored("=" * 60, "cyan"))

    print("\n[1] Tokenizer...")
    model_path = args.model_name
    looks_like_path = model_path.startswith(("/", "./"))
    is_local_dir = os.path.isdir(model_path)
    if looks_like_path and not is_local_dir:
        raise FileNotFoundError(
            f"Model path does not exist: {model_path}. "
            "Use a valid directory, HF repo id, or set PARISKV_MODEL_NAME."
        )
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=is_local_dir,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    print("\n[2] Dataset...")
    dataset = load_dataset("zai-org/LongBench-v2", split="train")

    filtered_data = [
        s
        for s in dataset
        if s.get("length", "") == args.length and s.get("difficulty", "") == args.difficulty
    ]
    print(f"  filtered n={len(filtered_data)} (length={args.length}, difficulty={args.difficulty})")

    samples = filtered_data[: args.batch_size]
    print(f"  using {len(samples)} rows (batch_size={args.batch_size})")

    print("\n[3] Tokenize batch...")
    prompts = []
    for s in samples:
        context = s.get("context", "")
        question = s.get("question", "")
        prompt = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
        prompts.append(prompt)

    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=args.max_input_tokens,
    )
    input_ids = inputs.input_ids.to(args.device)
    attention_masks = inputs.attention_mask.to(args.device)

    bs, seq_len = input_ids.shape
    print(f"  input shape [{bs}, {seq_len}]")
    print(f"  total input tokens {bs * seq_len:,}")

    print("\n[4] Model...")
    max_length = seq_len + args.max_new_tokens + args.warmup_steps + 100

    if args.model_type == "llama":
        from model_hub.llama import LlamaModel

        llm = LlamaModel(
            model_name=args.model_name,
            max_length=max_length,
            dtype=torch.bfloat16,
            device_map=args.device,
        )
    else:
        from model_hub.qwen import QwenModel

        llm = QwenModel(
            model_name=args.model_name,
            max_length=max_length,
            dtype=torch.bfloat16,
            device_map=args.device,
        )

    print("\n[5] Attention config...")
    if args.attention_type == "Full_Flash_Attn":
        config = {
            "prefill_chunk_size": 1,
            "Full_Flash_Attn": {
                "enable_decode_timing": True,
            },
        }
    elif args.attention_type == "PolarANN":
        config = {
            "prefill_chunk_size": 1,
            "PolarANN": {
                "sink_size": args.sink_size,
                "local_size": args.local_size,
                "dynamic_update_interval": args.dynamic_update_interval,
                "enable_offload": args.enable_offload,
                "enable_recall": False,
                "enable_rerank": True,
                "core": 1,
                "nprobe": 1,
                "cache_unit_size": 1,
                "cache_cluster_num": 1,
            },
        }
    else:
        raise ValueError(f"Unsupported attention_type: {args.attention_type}")

    with torch.no_grad():
        print(
            colored(
                f"\n[Benchmark] warmup={args.warmup_steps}, timed={args.max_new_tokens}",
                "green",
            )
        )
        llm.generate(
            attention_type=args.attention_type,
            inputs_ids=input_ids,
            attention_masks=attention_masks,
            max_new_length=args.warmup_steps + args.max_new_tokens,
            attn_config=config,
            temperature=0.0,
            top_p=0.0,
            top_k=0,
            disable_early_stop=True,
        )

    print(f"  batch={bs} seq_len={seq_len:,} out_tokens/sample={args.max_new_tokens}")


if __name__ == "__main__":
    main()
