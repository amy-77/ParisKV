#!/usr/bin/env python3
"""LongBench-v2 evaluation with Qwen3-8B and PolarANN or full flash attention."""

import os

os.environ["TORCH_CUDA_ARCH_LIST"] = "8.0"
import numpy as np
import torch
import json
import sys
import argparse
from termcolor import colored
import time
from tqdm import tqdm
from transformers import AutoTokenizer
from datasets import load_dataset
import warnings
import traceback
import faulthandler

try:
    import nvtx
except ImportError:  # pragma: no cover - optional profiling dependency
    nvtx = None

try:
    import torch.cuda.nvtx as torch_nvtx
except ImportError:  # pragma: no cover
    torch_nvtx = None

try:
    import torch.cuda.cudart as cudart
except ImportError:  # pragma: no cover
    cudart = None

warnings.filterwarnings(
    "ignore", message="Token indices sequence length is longer than the specified maximum"
)
import re

# Ensure crash traces from native/CUDA faults are visible in logs.
faulthandler.enable(all_threads=True)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from model_hub.qwen import QwenModel


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def parse_args():
    parser = argparse.ArgumentParser(description="Test LongBench-v2 with Qwen3-8B")
    parser.add_argument("--max_new_tokens", type=int, default=2048, help="Maximum new tokens to generate")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["fp16", "bf16"], help="Data type")
    parser.add_argument("--model_name", type=str, 
                        default="Qwen/Qwen3-8B",
                        help="Model name or path (default: Qwen3-8B, supports 32K native, 131K with YaRN)")
    
    parser.add_argument("--dataset_name", type=str,
                        default="zai-org/LongBench-v2",
                        help="LongBench-v2 dataset name (zai-org/LongBench-v2 or THUDM/LongBench-v2)")
    parser.add_argument("--split_name", type=str, default="train",
                        help="Dataset split name (default: train)")
    parser.add_argument("--num_samples", type=int, default=10, 
                        help="Number of samples to test (default: 10, -1 for all)")
    parser.add_argument("--start_idx", type=int, default=0,
                        help="Start index for testing")
    parser.add_argument("--sample_offset", type=int, default=None,
                        help="Alias for start_idx when resuming partially processed splits")
    parser.add_argument("--output_file", type=str, default=None,
                        help="Output file for results (auto-generate if None)")
    
    parser.add_argument("--domain", type=str, default=None,
                        help="Filter by domain (e.g., single-document_qa)")
    parser.add_argument("--difficulty", type=str, default=None,
                        choices=["easy", "hard"],
                        help="Filter by difficulty")
    parser.add_argument("--length", type=str, default=None,
                        choices=["short", "medium", "long"],
                        help="Filter by length category")
    parser.add_argument("--enable_cot", action="store_true",
                        help="Enable Chain-of-Thought reasoning")
    parser.add_argument("--cot_max_new_tokens", type=int, default=1024,
                        help="CoT Stage 1 max tokens (default: 1024)")
    parser.add_argument("--answer_max_new_tokens", type=int, default=128,
                        help="CoT Stage 2 max tokens (default: 128)")
    parser.add_argument("--enable_recall", type=lambda x: x.lower() == 'true', default=None,
                        help="Enable Recall@K computation (true/false, default: use config file)")
    parser.add_argument("--enable_decode_timing", action="store_true",
                        help="Enable decode timing statistics for Full_Flash_Attn")
    parser.add_argument("--max_context_tokens", type=int, default=None,
                        help="Maximum context tokens to process (skip samples exceeding this, default: no limit)")
    parser.add_argument("--force_process", action="store_true",
                        help="Force process samples even if they exceed max_context_tokens (will truncate)")
    parser.add_argument("--no_truncate", action="store_true",
                        help="Skip truncation and only test samples that fit within max context (overrides --force_process)")
    parser.add_argument("--sink_size", type=int, default=None,
                        help="Override sink zone size (default: use config value)")
    parser.add_argument("--local_size", type=int, default=None,
                        help="Override local zone size (default: use config value)")
    
    parser.add_argument(
        "--attention_type",
        type=str,
        default="Full_Flash_Attn",
        choices=["PolarANN", "Full_Flash_Attn"],
        help="Attention backend",
    )
    parser.add_argument("--static_pattern_end", type=int, default=None,
                        help="Static pattern end (local tokens size) for PolarANN (default: use config file)")
    
    parser.add_argument("--final_topk", type=int, default=None,
                        help="Final top-K for reranking stage (default: use cache default)")
    parser.add_argument("--enable_offload", type=int, default=None, choices=[0, 1],
                        help="Enable KV cache offloading (0=off, 1=on)")
    parser.add_argument("--polar_cache_module", type=str, default=None,
                        help="PolarANN cache module path (also settable via POLAR_CACHE_MODULE env var)")
    parser.add_argument("--no_resume", action="store_true",
                        help="Disable resume from existing output file")
    parser.add_argument("--temperature", type=float, default=0.6,
                        help="Sampling temperature (0.6 for CoT, 0.7 for non-CoT, DO NOT use 0.0 with Qwen3)")
    parser.add_argument("--top_p", type=float, default=0.95,
                        help="Nucleus sampling top_p")
    parser.add_argument("--top_k", type=int, default=20,
                        help="Top-k sampling")
    
    args = parser.parse_args()
    if getattr(args, "sample_offset", None) is not None:
        args.start_idx = args.sample_offset
    return args


def load_longbench_v2_data(
    dataset_name,
    split_name,
    num_samples=-1,
    start_idx=0,
    domain=None,
    difficulty=None,
    length=None,
):
    print(colored("\nLoading LongBench-v2", "yellow"))
    print(colored(f"  dataset: {dataset_name}", "cyan"))
    print(colored(f"  split: {split_name}", "cyan"))

    try:
        dataset = load_dataset(dataset_name, split=split_name)
        print(colored(f"  rows (raw): {len(dataset)}", "cyan"))

        if domain or difficulty or length:
            filtered_data = []
            for item in dataset:
                if domain and item.get("domain") != domain:
                    continue
                if difficulty and item.get("difficulty") != difficulty:
                    continue
                if length and item.get("length") != length:
                    continue
                filtered_data.append(item)
            dataset = dataset.from_list(filtered_data)
            print(colored(f"  rows (after filters): {len(dataset)}", "cyan"))

        if num_samples > 0:
            end_idx = min(start_idx + num_samples, len(dataset))
            dataset = dataset.select(range(start_idx, end_idx))
        elif start_idx > 0:
            dataset = dataset.select(range(start_idx, len(dataset)))

        if len(dataset) == 0:
            print(colored("No rows after filter/sample; exiting.", "yellow"))
            sys.exit(0)

        print(colored(f"  loaded {len(dataset)} samples", "green"))

        if len(dataset) > 0:
            sample = dataset[0]
            print(colored("\nExample row:", "yellow"))
            print(f"  ID: {sample.get('_id', 'N/A')}")
            print(f"  Domain: {sample.get('domain', 'N/A')}")
            print(f"  Sub-domain: {sample.get('sub_domain', 'N/A')}")
            print(f"  Difficulty: {sample.get('difficulty', 'N/A')}")
            print(f"  Length: {sample.get('length', 'N/A')}")
            question = sample.get("question", "")
            qprev = question[:100] + "..." if len(question) > 100 else question
            print(f"  Question: {qprev}")
            print(f"  Gold answer: {sample.get('answer', 'N/A')}")
            context_len = len(sample.get("context", ""))
            print(f"  Context chars: {context_len}")

        return dataset

    except Exception as e:
        print(colored(f"Failed to load dataset: {e}", "red"))
        sys.exit(1)


def build_longbench_v2_prompt(sample, enable_cot=False):
    """Official-style 0-shot prompts (see LongBench prompts/)."""
    context = sample.get('context', '')
    question = sample.get('question', '')
    
    choice_a = sample.get('choice_A', '')
    choice_b = sample.get('choice_B', '')
    choice_c = sample.get('choice_C', '')
    choice_d = sample.get('choice_D', '')
    
    if enable_cot:
        prompt = f"""Please read the following text and answer the questions below.

<text>
{context}
</text>

What is the correct answer to this question: {question}
Choices:
(A) {choice_a}
(B) {choice_b}
(C) {choice_c}
(D) {choice_d}

Let's think step by step:"""
    else:
        prompt = f"""Please read the following text and answer the question below.

<text>
{context}
</text>

What is the correct answer to this question: {question}
Choices:
(A) {choice_a}
(B) {choice_b}
(C) {choice_c}
(D) {choice_d}

Format your response as follows: "The correct answer is (insert answer here)"."""
    
    return prompt


def detect_repetition(response):
    if not response or len(response) < 100:
        return {
            "is_repetitive": False,
            "max_sentence_repetition": 0,
            "answer_pattern_count": 0,
            "uniqueness_ratio": 1.0,
        }

    sentences = response.split(".")
    sentence_counts = {}
    for sent in sentences:
        sent = sent.strip()
        if len(sent) > 20:
            sentence_counts[sent] = sentence_counts.get(sent, 0) + 1

    max_repetition = max(sentence_counts.values()) if sentence_counts else 0

    answer_pattern_count = len(
        re.findall(r"(?:correct\s+)?answer\s+is\s+\([ABCD]\)", response, re.IGNORECASE)
    )

    unique_chars = len(set(response))
    total_chars = len(response)
    uniqueness_ratio = unique_chars / total_chars if total_chars > 0 else 0

    is_repetitive = (
        max_repetition >= 3
        or answer_pattern_count >= 5
        or (uniqueness_ratio < 0.05 and total_chars > 500)
    )
    
    return {
        "is_repetitive": is_repetitive,
        "max_sentence_repetition": max_repetition,
        "answer_pattern_count": answer_pattern_count,
        "uniqueness_ratio": uniqueness_ratio,
    }


def extract_answer_from_response(response, enable_cot=False):
    if not response:
        return None

    response_upper = response.upper()

    answer_patterns = [
        r"(?:correct\s+)?answer\s+is\s+\(([ABCD])\)",
        r"(?:correct\s+)?answer\s*[:：]\s*\(([ABCD])\)",
        r"(?:correct\s+)?answer\s+is\s+([ABCD])\b",
        r"(?:correct\s+)?answer\s*[:：]\s*([ABCD])\b",
    ]
    for pattern in answer_patterns:
        matches = list(re.finditer(pattern, response_upper, re.IGNORECASE))
        if matches:
            first_match = matches[0]
            if first_match.start() < 200 and len(matches) > 1:
                for match in matches:
                    if match.start() >= 200:
                        return match.group(1)
                return first_match.group(1)
            return first_match.group(1)

    matches = list(re.finditer(r"\(([ABCD])\)", response_upper))
    if matches:
        last_match = matches[-1]
        if len(response_upper) - last_match.end() < 10:
            if len(matches) > 1:
                return matches[-2].group(1)
        return last_match.group(1)

    if enable_cot:
        lines = response.strip().split("\n")
        for line in reversed(lines[-10:]):
            line = line.strip().upper()
            if not line:
                continue
            match = re.search(r"\(([ABCD])\)", line)
            if match:
                return match.group(1)
            match = re.search(r"\b([ABCD])\b(?!\s+single)", line)
            if match:
                return match.group(1)

    half_point = len(response_upper) // 2
    response_second_half = response_upper[half_point:]

    matches = list(re.finditer(r"\b([ABCD])\b(?!\s+single)", response_second_half))
    if matches:
        return matches[-1].group(1)

    matches = list(re.finditer(r"\b([ABCD])\b", response_upper))
    if matches:
        for match in reversed(matches):
            end = match.end()
            if end < len(response_upper) - 7:
                if response_upper[end : end + 7] == " SINGLE":
                    continue
            return match.group(1)

    return None


def generate_config(args, context_len):
    import json
    import os

    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    config_dir = os.path.join(project_root, "config")
    model_name = args.model_name.split("/")[-1] + ".json"
    config_file = os.path.join(config_dir, model_name)

    try:
        with open(config_file, "r") as f:
            config = json.load(f)
    except FileNotFoundError:
        config = {}

    if args.attention_type == "Full_Flash_Attn":
        if "Full_Flash_Attn" not in config:
            config["Full_Flash_Attn"] = {}
        config["Full_Flash_Attn"]["enable_decode_timing"] = args.enable_decode_timing
        
    elif args.attention_type == "PolarANN" and "PolarANN" in config:
        if args.sink_size is not None:
            config["PolarANN"]["static_pattern_start"] = args.sink_size
        if args.static_pattern_end is not None:
            config["PolarANN"]["static_pattern_end"] = args.static_pattern_end
        if args.local_size is not None:
            config["PolarANN"]["static_pattern_end"] = args.local_size

        if args.enable_recall is not None:
            config["PolarANN"]["enable_recall"] = args.enable_recall
        if getattr(args, "final_topk", None) is not None:
            config["PolarANN"]["final_topk"] = args.final_topk
        if getattr(args, "enable_offload", None) is not None:
            config["PolarANN"]["enable_offload"] = args.enable_offload == 1

    return config


def load_model(args):
    print(colored(f"\nLoading model: {args.model_name}", "yellow"))
    print(colored(f"  Attention: {args.attention_type}", "cyan"))
    print(colored(f"  Device: {args.device}", "cyan"))
    print(colored(f"  Dtype: {args.dtype}", "cyan"))
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    max_length = 131072
    llm = QwenModel(
        model_name=args.model_name,
        max_length=max_length,
        dtype=dtype,
        device_map=args.device,
    )
    print(colored("Model ready.", "green"))
    return llm


def main():
    args = parse_args()
    set_seed(42)

    if args.polar_cache_module:
        os.environ["POLAR_CACHE_MODULE"] = args.polar_cache_module

    if args.output_file is None:
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_file = f"longbench_v2_{args.attention_type}_{timestamp}.jsonl"

    output_dir = os.path.dirname(args.output_file) or "../results/longbench_v2"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, os.path.basename(args.output_file))

    print(colored("=" * 80, "yellow"))
    print(colored("LongBench-v2", "yellow", attrs=["bold"]))
    print(colored(f"Model: {args.model_name}", "cyan"))
    print(colored(f"Dataset: {args.dataset_name}", "cyan"))
    print(colored(f"Split: {args.split_name}", "cyan"))
    print(colored(f"Attention: {args.attention_type}", "cyan"))

    data = load_longbench_v2_data(
        args.dataset_name,
        args.split_name,
        args.num_samples,
        args.start_idx,
        args.domain,
        args.difficulty,
        args.length,
    )

    llm = load_model(args)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    tokenizer.model_max_length = 131072

    results = []
    correct = 0
    total = 0
    total_time = 0
    
    processed_ids = set()
    mode = "w"
    if not args.no_resume and os.path.exists(output_path):
        print(colored(f"Found {output_path}, resuming...", "yellow"))
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        res = json.loads(line)
                        if "sample_id" in res:
                            processed_ids.add(res["sample_id"])
                    except Exception:
                        pass
            if processed_ids:
                mode = "a"
                print(
                    colored(
                        f"Skipping {len(processed_ids)} already-written samples (append).",
                        "green",
                    )
                )
        except Exception as e:
            print(colored(f"Could not read output file: {e}", "red"))

    output_fp = open(output_path, mode, encoding="utf-8")
    profiler_active = False

    try:
        if cudart is not None:
            cudart.cudaProfilerStart()
            profiler_active = True

        print(colored(f"\nEvaluating {len(data)} samples...", "yellow"))
        print(colored("=" * 80, "yellow"))

        for idx, sample in enumerate(tqdm(data, desc="Testing", ncols=100)):
            if sample.get("_id") in processed_ids:
                continue

            try:
                if idx == 0 and torch.cuda.is_available():
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()

                prompt = build_longbench_v2_prompt(sample, enable_cot=args.enable_cot)
                correct_answer = sample.get("answer", "").strip().upper()

                inputs = tokenizer([prompt], return_tensors="pt", padding=True)
                input_ids = inputs.input_ids
                attention_masks = inputs.attention_mask
                input_ids = input_ids.to(llm.layers[0].device)
                attention_masks = attention_masks.to(llm.layers[0].device)
                context_len = input_ids.shape[1]
                
                model_max_length = llm.max_length
                max_input_tokens = model_max_length - (
                    args.cot_max_new_tokens if args.enable_cot else args.max_new_tokens
                )

                if args.max_context_tokens:
                    user_max_input = args.max_context_tokens - args.max_new_tokens
                    if user_max_input < args.max_new_tokens or user_max_input <= 0:
                        if idx == 0:
                            print(
                                colored(
                                    f"\n  Warning: max_context_tokens={args.max_context_tokens} is inconsistent.",
                                    "yellow",
                                )
                            )
                            print(
                                colored(
                                    f"    max_context_tokens - max_new_tokens = {user_max_input} (need > 0)",
                                    "yellow",
                                )
                            )
                            print(
                                colored(
                                    f"    Using model cap: {model_max_length} - {args.max_new_tokens} = {max_input_tokens}",
                                    "green",
                                )
                            )
                    else:
                        max_input_tokens = min(max_input_tokens, user_max_input)

                if max_input_tokens <= 0:
                    print(
                        colored(f"\n  Error: max_input_tokens = {max_input_tokens} <= 0", "red")
                    )
                    print(
                        colored(
                            f"    model_max_length={model_max_length}, max_new_tokens={args.max_new_tokens}",
                            "red",
                        )
                    )
                    raise ValueError(f"Invalid max_input_tokens: {max_input_tokens}")

                if context_len > max_input_tokens:
                    if args.no_truncate:
                        print(
                            colored(
                                f"\nSample {idx + 1}/{len(data)} skip (--no_truncate, context too long)",
                                "yellow",
                            )
                        )
                        print(
                            f"  context_len={context_len:,} tokens (limit {max_input_tokens:,})"
                        )
                        result = {
                            'sample_id': sample.get('_id', idx),
                            'domain': sample.get('domain', ''),
                            'sub_domain': sample.get('sub_domain', ''),
                            'difficulty': sample.get('difficulty', ''),
                            'length': sample.get('length', ''),
                            'question': sample.get('question', ''),
                            'correct_answer': correct_answer,
                            'predicted_answer': None,
                            'response': None,
                            'is_correct': False,
                            'context_length': context_len,
                            'num_output_tokens': 0,
                            'prefill_time': 0,
                            'decode_time': 0,
                            'generation_time': 0,
                            'tpot': 0,
                            'skipped': True,
                            'skip_reason': 'no_truncate_enabled'
                        }
                        results.append(result)
                        output_fp.write(json.dumps(result, ensure_ascii=False) + '\n')
                        output_fp.flush()
                        continue
                    
                    if args.force_process:
                        print(
                            colored(
                                f"\nSample {idx + 1}/{len(data)}: truncating (decode + re-tokenize)",
                                "yellow",
                            )
                        )
                        print(f"  before: {context_len:,} tokens")

                        safe_max_input = max_input_tokens - 10
                        stage1_reserve = (
                            args.cot_max_new_tokens if args.enable_cot else args.max_new_tokens
                        )
                        print(
                            f"  strategy: first {safe_max_input // 2} + last {safe_max_input // 2} tokens "
                            f"(reserve {stage1_reserve} + 10)"
                        )

                        half = safe_max_input // 2
                        prefix_text = tokenizer.decode(
                            input_ids[0, :half], skip_special_tokens=True
                        )
                        suffix_text = tokenizer.decode(
                            input_ids[0, -half:], skip_special_tokens=True
                        )
                        truncated_text = prefix_text + suffix_text
                        new_inputs = tokenizer([truncated_text], return_tensors="pt", padding=True)
                        input_ids = new_inputs.input_ids.to(llm.layers[0].device)
                        attention_masks = new_inputs.attention_mask.to(llm.layers[0].device)

                        context_len = input_ids.shape[1]

                        print(f"  after: {context_len:,} tokens (contiguous positions 0..{context_len - 1})")

                        length_diff = abs(context_len - max_input_tokens)
                        length_tolerance = 32
                        if length_diff > length_tolerance:
                            print(
                                colored(
                                    f"  length mismatch: got {context_len:,} vs target {max_input_tokens:,} "
                                    f"(diff {length_diff})",
                                    "yellow",
                                )
                            )
                        elif length_diff > 0:
                            print(f"  small re-tokenize drift: ±{length_diff} tokens")

                        stage1_reserve = (
                            args.cot_max_new_tokens if args.enable_cot else args.max_new_tokens
                        )
                        if context_len + stage1_reserve > model_max_length:
                            print(
                                colored(
                                    f"  still over limit: {context_len} + {stage1_reserve} > {model_max_length}",
                                    "red",
                                )
                            )
                            raise ValueError(
                                f"Truncation failed: {context_len} + {stage1_reserve} > {model_max_length}"
                            )
                    else:
                        print(
                            colored(
                                f"\nSample {idx + 1}/{len(data)} skip (context too long)",
                                "yellow",
                            )
                        )
                        print(
                            f"  context_len={context_len:,} (limit {max_input_tokens:,}, "
                            f"need {args.max_new_tokens} for output)"
                        )
                        result = {
                            'sample_id': sample.get('_id', idx),
                            'domain': sample.get('domain', ''),
                            'sub_domain': sample.get('sub_domain', ''),
                            'difficulty': sample.get('difficulty', ''),
                            'length': sample.get('length', ''),
                            'question': sample.get('question', ''),
                            'correct_answer': correct_answer,
                            'predicted_answer': None,
                            'response': None,
                            'is_correct': False,
                            'context_length': context_len,
                            'num_output_tokens': 0,
                            'prefill_time': 0,
                            'decode_time': 0,
                            'generation_time': 0,
                            'tpot': 0,
                            'skipped': True,
                            'skip_reason': f'exceeds_max_length_{args.max_context_tokens}'
                        }
                        results.append(result)
                        continue
                
                if torch.cuda.is_available():
                    import gc

                    gc.collect()
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()

                config = generate_config(args, context_len)

                if args.attention_type == "PolarANN" and hasattr(llm.layers[0], "kv_cache"):
                    if hasattr(llm.layers[0].kv_cache, "reset_recall_history"):
                        llm.layers[0].kv_cache.reset_recall_history()

                with torch.no_grad():
                    if args.enable_cot:
                        print(
                            colored(
                                f"\n  [CoT Stage 1] reasoning (max_new_tokens={args.cot_max_new_tokens}) ...",
                                "cyan",
                            )
                        )
                        config_s1 = generate_config(args, context_len)
                        s1_out, s1_stats = llm.generate(
                            attention_type=args.attention_type,
                            inputs_ids=input_ids,
                            attention_masks=attention_masks,
                            max_new_length=args.cot_max_new_tokens + 1,
                            attn_config=config_s1,
                            temperature=args.temperature,
                            top_p=args.top_p,
                            top_k=args.top_k,
                        )
                        s1_ids = (
                            s1_out[0].tolist()
                            if hasattr(s1_out[0], "tolist")
                            else s1_out[0]
                        )
                        reasoning_text = tokenizer.decode(s1_ids, skip_special_tokens=True)
                        s1_time = s1_stats["prefill_time"] + s1_stats["decode_time"]
                        print(
                            colored(
                                f"  [CoT Stage 1] done: {len(s1_ids)} tok, "
                                f"prefill {s1_stats['prefill_time']:.2f}s + decode {s1_stats['decode_time']:.2f}s = {s1_time:.2f}s",
                                "green",
                            )
                        )

                        s2_prompt = prompt + reasoning_text
                        s2_inputs = tokenizer([s2_prompt], return_tensors="pt", padding=True)
                        s2_input_ids = s2_inputs.input_ids.to(llm.layers[0].device)
                        s2_attention_masks = s2_inputs.attention_mask.to(llm.layers[0].device)
                        s2_context_len = s2_input_ids.shape[1]
                        s2_max_input = model_max_length - args.answer_max_new_tokens - 10

                        if s2_context_len > s2_max_input:
                            print(
                                colored(
                                    f"  [CoT Stage 2] truncating: {s2_context_len:,} > {s2_max_input:,}",
                                    "yellow",
                                )
                            )
                            half = s2_max_input // 2
                            prefix_t = tokenizer.decode(
                                s2_input_ids[0, :half], skip_special_tokens=True
                            )
                            suffix_t = tokenizer.decode(
                                s2_input_ids[0, -half:], skip_special_tokens=True
                            )
                            s2_prompt = prefix_t + suffix_t
                            s2_inputs = tokenizer([s2_prompt], return_tensors="pt", padding=True)
                            s2_input_ids = s2_inputs.input_ids.to(llm.layers[0].device)
                            s2_attention_masks = s2_inputs.attention_mask.to(llm.layers[0].device)
                            s2_context_len = s2_input_ids.shape[1]
                            print(colored(f"    after truncate: {s2_context_len:,} tokens", "yellow"))

                        print(
                            colored(
                                f"  [CoT Stage 2] answer (max_new_tokens={args.answer_max_new_tokens}) ...",
                                "cyan",
                            )
                        )
                        config_s2 = generate_config(args, s2_context_len)
                        s2_out, s2_stats = llm.generate(
                            attention_type=args.attention_type,
                            inputs_ids=s2_input_ids,
                            attention_masks=s2_attention_masks,
                            max_new_length=args.answer_max_new_tokens + 1,
                            attn_config=config_s2,
                            temperature=args.temperature,
                            top_p=args.top_p,
                            top_k=args.top_k,
                        )
                        s2_ids = (
                            s2_out[0].tolist()
                            if hasattr(s2_out[0], "tolist")
                            else s2_out[0]
                        )
                        response = tokenizer.decode(s2_ids, skip_special_tokens=True)
                        s2_time = s2_stats["prefill_time"] + s2_stats["decode_time"]
                        print(
                            colored(
                                f"  [CoT Stage 2] done: prefill {s2_stats['prefill_time']:.2f}s + "
                                f"decode {s2_stats['decode_time']:.2f}s = {s2_time:.2f}s",
                                "green",
                            )
                        )
                        
                        prefill_time = s1_stats["prefill_time"] + s2_stats["prefill_time"]
                        decode_time = s1_stats["decode_time"] + s2_stats["decode_time"]
                        gen_time = s1_time + s2_time
                        total_time += gen_time
                        num_output_tokens = len(s1_ids) + len(s2_ids)
                    else:
                        output_ids, timing_stats = llm.generate(
                            attention_type=args.attention_type,
                            inputs_ids=input_ids,
                            attention_masks=attention_masks,
                            max_new_length=args.max_new_tokens + 1,
                            attn_config=config,
                            temperature=args.temperature,
                            top_p=args.top_p,
                            top_k=args.top_k,
                        )
                        prefill_time = timing_stats["prefill_time"]
                        decode_time = timing_stats["decode_time"]
                        gen_time = timing_stats["total_time"]
                        total_time += gen_time
                        generated_ids = (
                            output_ids[0].tolist()
                            if hasattr(output_ids[0], "tolist")
                            else output_ids[0]
                        )
                        response = tokenizer.decode(generated_ids, skip_special_tokens=True)
                        num_output_tokens = len(generated_ids)

                repetition_info = detect_repetition(response)

                predicted_answer = extract_answer_from_response(
                    response, enable_cot=args.enable_cot
                )
                is_correct = (predicted_answer == correct_answer) if predicted_answer else False

                if is_correct:
                    correct += 1
                total += 1

                tpot = (decode_time / num_output_tokens) if num_output_tokens > 0 else 0

                result = {
                    "sample_id": sample.get("_id", idx),
                    "domain": sample.get("domain", ""),
                    "sub_domain": sample.get("sub_domain", ""),
                    "difficulty": sample.get("difficulty", ""),
                    "length": sample.get("length", ""),
                    "question": sample.get("question", ""),
                    "correct_answer": correct_answer,
                    "predicted_answer": predicted_answer,
                    "response": response,
                    "is_correct": is_correct,
                    "context_length": context_len,
                    "num_output_tokens": num_output_tokens,
                    "prefill_time": prefill_time,
                    "decode_time": decode_time,
                    "generation_time": gen_time,
                    "tpot": tpot,
                    "is_repetitive": repetition_info["is_repetitive"],
                    "repetition_stats": repetition_info,
                }
                results.append(result)

                status = colored("OK", "green") if is_correct else colored("XX", "red")
                repetition_warning = (
                    colored(" [repetition]", "yellow")
                    if repetition_info["is_repetitive"]
                    else ""
                )
                print(f"\nSample {idx + 1}/{len(data)} {status}{repetition_warning}")
                print(
                    f"  domain={sample.get('domain', 'N/A')} "
                    f"difficulty={sample.get('difficulty', 'N/A')} "
                    f"length={sample.get('length', 'N/A')}"
                )
                print(f"  context_len={context_len} tokens")
                print(f"  pred={predicted_answer} gold={correct_answer}")
                print(
                    f"  time: prefill {prefill_time:.2f}s + decode {decode_time:.2f}s = {gen_time:.2f}s"
                )
                print(
                    f"  out_tokens={num_output_tokens} TPOT(decode)={tpot * 1000:.2f}ms"
                )
                if repetition_info["is_repetitive"]:
                    print(
                        colored(
                            f"  repetition: answer-pattern={repetition_info['answer_pattern_count']}, "
                            f"max_sentence_rep={repetition_info['max_sentence_repetition']}",
                            "yellow",
                        )
                    )

                if args.attention_type == "PolarANN" and hasattr(llm.layers[0], "kv_cache"):
                    kv_cache = llm.layers[0].kv_cache
                    if hasattr(kv_cache, "print_recall_summary"):
                        kv_cache.print_recall_summary()
                    if hasattr(kv_cache, "plot_recall_curve"):
                        recall_plots_dir = os.path.join(
                            os.path.dirname(args.output_file), "recall_plots"
                        )
                        os.makedirs(recall_plots_dir, exist_ok=True)
                        kv_cache.plot_recall_curve(save_dir=recall_plots_dir)

                if torch.cuda.is_available():
                    import gc

                    gc.collect()
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()

                output_fp.write(json.dumps(result, ensure_ascii=False) + "\n")
                output_fp.flush()

            except Exception as e:
                print(colored(f"\nError on sample {idx + 1}: {e}", "red"))
                print(colored("Traceback:", "red"))
                traceback.print_exc(file=sys.stderr)
                sys.stderr.flush()
                sys.stdout.flush()
                if torch.cuda.is_available():
                    import gc

                    gc.collect()
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                continue
    finally:
        if profiler_active and cudart is not None:
            cudart.cudaProfilerStop()
        output_fp.close()

    accuracy = (correct / total * 100) if total > 0 else 0
    avg_time = total_time / total if total > 0 else 0

    print(colored("Summary", "yellow", attrs=["bold"]))
    print(f"total={total} correct={correct}")
    print(f"accuracy={accuracy:.2f}% ({correct}/{total})")
    print(f"avg_gen_time={avg_time:.2f}s")
    print(colored("=" * 80, "yellow"))
    print(f"\nWrote: {output_path}")

    if results:
        domain_stats = {}
        for r in results:
            if r.get("skipped", False):
                continue

            domain = r.get("domain", "unknown")
            if domain not in domain_stats:
                domain_stats[domain] = {"correct": 0, "total": 0}
            domain_stats[domain]["total"] += 1
            if r.get("is_correct"):
                domain_stats[domain]["correct"] += 1

        print(colored("\nBy domain:", "cyan"))
        for domain, stats in sorted(domain_stats.items()):
            acc = (stats['correct'] / stats['total'] * 100) if stats['total'] > 0 else 0
            print(f"  {domain}: {acc:.2f}% ({stats['correct']}/{stats['total']})")
    
    if results:
        combo_stats = {}
        for r in results:
            if r.get("skipped", False):
                continue

            difficulty = r.get("difficulty", "unknown")
            length = r.get("length", "unknown")
            combo_key = f"{difficulty}_{length}"

            if combo_key not in combo_stats:
                combo_stats[combo_key] = {
                    "correct": 0,
                    "total": 0,
                    "context_lengths": [],
                    "tpots": [],
                    "prefill_times": [],
                    "decode_times": [],
                    "gen_times": [],
                }

            combo_stats[combo_key]["total"] += 1
            if r.get("is_correct"):
                combo_stats[combo_key]["correct"] += 1

            if r.get("context_length"):
                combo_stats[combo_key]["context_lengths"].append(r["context_length"])
            if r.get("tpot"):
                combo_stats[combo_key]["tpots"].append(r["tpot"])
            if r.get("prefill_time"):
                combo_stats[combo_key]["prefill_times"].append(r["prefill_time"])
            if r.get("decode_time"):
                combo_stats[combo_key]["decode_times"].append(r["decode_time"])
            if r.get("generation_time"):
                combo_stats[combo_key]["gen_times"].append(r["generation_time"])

        if combo_stats:
            print(colored("\nBy difficulty x length:", "cyan"))
            print(colored("=" * 80, "cyan"))
            hdr = (
                f"{'combo':<14}{'acc%':>10}{'n':>8}"
                f"{'avg_len':>14}{'prefill':>12}{'decode':>12}{'TPOT(ms)':>12}"
            )
            print(hdr)

            for combo_key in sorted(combo_stats.keys()):
                stats = combo_stats[combo_key]
                acc = (
                    (stats["correct"] / stats["total"] * 100)
                    if stats["total"] > 0
                    else 0
                )

                avg_len = (
                    np.mean(stats["context_lengths"]) if stats["context_lengths"] else 0
                )
                avg_prefill = (
                    np.mean(stats["prefill_times"]) if stats["prefill_times"] else 0
                )
                avg_decode = (
                    np.mean(stats["decode_times"]) if stats["decode_times"] else 0
                )
                avg_tpot_ms = np.mean(stats["tpots"]) * 1000 if stats["tpots"] else 0

                print(
                    f"{combo_key:<16}"
                    f"{acc:>10.2f}%({stats['correct']}/{stats['total']})"
                    f"{stats['total']:>8}"
                    f"{avg_len:>14,.0f}"
                    f"{avg_prefill:>12.2f}s"
                    f"{avg_decode:>12.2f}s"
                    f"{avg_tpot_ms:>12.2f}"
                )

    print(colored("\n" + "=" * 80, "cyan"))
    if args.attention_type == "PolarANN":
        print(colored("PolarANN timing", "cyan", attrs=["bold"]))

        if hasattr(llm, "timing_stats"):
            print(colored("\n[LLM]", "yellow"))
            print(
                f"  decode_update_kv_cache:  {llm.timing_stats.get('decode_update_kv_cache', 0):.3f}s"
            )
            print(
                f"  decode_attention:        {llm.timing_stats.get('decode_attention', 0):.3f}s"
            )

        if hasattr(llm.layers[0], "kv_cache") and hasattr(
            llm.layers[0].kv_cache, "timing_stats"
        ):
            kv_stats = llm.layers[0].kv_cache.timing_stats
            print(colored("\n[PolarANN KV]", "yellow"))
            print(
                f"  collision_based_topk_batch:           {kv_stats.get('collision_based_topk_batch', 0):.3f}s"
            )
            print(f"    get_q_score:                      {kv_stats.get('get_q_score', 0):.3f}s")
            print(
                f"    get_topk_clusters:                {kv_stats.get('get_topk_clusters', 0):.3f}s"
            )
            print(
                f"    update_cache_cnt:                 {kv_stats.get('update_cache_cnt', 0):.3f}s"
            )
            print(
                f"    get_candidate_cache_rerank_slow:  {kv_stats.get('get_candidate_cache_rerank_slow', 0):.3f}s"
            )
            print(
                f"  flash_attn_with_kvcache_compat:       {kv_stats.get('flash_attn_with_kvcache_compat', 0):.3f}s"
            )

        print(colored("=" * 80, "cyan"))

    if args.enable_decode_timing and args.attention_type == "Full_Flash_Attn":
        if (
            hasattr(llm.layers[0], "kv_cache")
            and hasattr(llm.layers[0].kv_cache, "decode_times")
            and llm.layers[0].kv_cache.decode_times
        ):
            decode_times = llm.layers[0].kv_cache.decode_times

            print(colored("\n" + "=" * 80, "cyan"))
            print(colored("Full flash decode timing", "cyan", attrs=["bold"]))
            print(f"steps={len(decode_times)}")
            print(f"mean={np.mean(decode_times):.4f} ms")
            print(f"median={np.median(decode_times):.4f} ms")
            print(f"min={np.min(decode_times):.4f} ms")
            print(f"max={np.max(decode_times):.4f} ms")
            print(f"std={np.std(decode_times):.4f} ms")
            print(
                f"sum={np.sum(decode_times):.2f} ms ({np.sum(decode_times) / 1000:.2f} s)"
            )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(colored(f"\nFatal: {e}", "red"))
        print(colored("Traceback:", "red"))
        traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()
        sys.stdout.flush()
        raise
