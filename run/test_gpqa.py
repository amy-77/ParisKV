#!/usr/bin/env python3
"""GPQA Diamond（JSONL）四选一 MC 评测；选项按 Record ID 做可复现打乱。"""

import argparse
import hashlib
import json
import os
import random
import sys
import time

import torch
from termcolor import colored
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from test_aime2025 import generate_config, load_model, set_seed
from test_longbench_v2 import extract_answer_from_response
from test_math500 import apply_polar_layout_overrides


def parse_args():
    p = argparse.ArgumentParser(description="GPQA Diamond benchmark")
    # 默认改本机；GPQA 数据集本机暂无，需要自行 hf-cli 下载或拷过来
    p.add_argument("--model_name", type=str,
                   default="/home/secure/pariskv/models/Qwen3-4B")
    p.add_argument(
        "--data_path",
        type=str,
        default="/home/secure/pariskv/data/GPQA/gpqa_diamond.jsonl",
        help="GPQA Diamond JSONL; 本机无需自备",
    )
    # YaRN rope_scaling（与 test_longbench_v2 / test_aime2025 同名）
    p.add_argument("--rope_scaling_factor", type=float, default=None,
                   help="YaRN rope_scaling factor (e.g. 4.0)")
    p.add_argument("--rope_scaling_orig_max", type=int, default=None,
                   help="YaRN rope_scaling original_max_position_embeddings (e.g. 32768)")
    p.add_argument("--num_samples", type=int, default=-1, help="-1 = all rows")
    p.add_argument("--start_idx", type=int, default=0)
    p.add_argument("--max_new_tokens", type=int, default=38912)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--dtype", type=str, default="bf16", choices=["fp16", "bf16", "fp32"])
    p.add_argument(
        "--attention_type",
        type=str,
        default="PolarANN",
        choices=["SuCo", "PQ", "PolarANN", "Polar_PQ", "RetroInfer", "Full_Flash_Attn"],
    )
    p.add_argument("--collision_ratio", type=float, default=None)
    p.add_argument("--candidate_ratio", type=float, default=None)
    p.add_argument("--cluster_num_per_subspace", type=int, default=32)
    p.add_argument("--num_subspaces", type=int, default=32)
    p.add_argument("--static_pattern_end", type=int, default=None)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--top_k", type=int, default=20)
    p.add_argument("--output_file", type=str, default=None)
    p.add_argument("--polar_cache_module", type=str,
                   default="cache_hub.polar_cache_pure_rabitq_quest_density",
                   help="Import path of polar_cache module (density release default)")
    p.add_argument(
        "--codebook_path",
        type=str,
        default=None,
        help="Override PolarANN codebook_path from the model config.",
    )
    p.add_argument("--final_topk", type=int, default=None)
    p.add_argument("--adaptive_topk_threshold", type=float, default=None)
    p.add_argument("--adaptive_topk_enabled", type=int, default=None, choices=[0, 1])
    p.add_argument("--sink_size", type=int, default=None)
    p.add_argument("--local_size", type=int, default=None)
    p.add_argument("--dynamic_update_interval", type=int, default=None)
    p.add_argument("--full_attention_threshold", type=int, default=None)
    p.add_argument("--enable_offload", type=int, default=None, choices=[0, 1])
    return p.parse_args()


def load_gpqa_diamond(path: str, num_samples: int, start_idx: int):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if start_idx > 0:
        rows = rows[start_idx:]
    if num_samples >= 0:
        rows = rows[:num_samples]
    return rows


def shuffle_options(row: dict):
    correct = str(row["Correct Answer"]).strip()
    incorrect = [
        str(row["Incorrect Answer 1"]).strip(),
        str(row["Incorrect Answer 2"]).strip(),
        str(row["Incorrect Answer 3"]).strip(),
    ]
    options = [correct] + incorrect
    rid = str(row.get("Record ID") or row.get("id") or "")
    h = hashlib.sha256(rid.encode("utf-8")).digest()
    seed = int.from_bytes(h[:8], "big")
    rng = random.Random(seed)
    order = list(range(4))
    rng.shuffle(order)
    shuffled = [options[i] for i in order]
    correct_letter = "ABCD"[shuffled.index(correct)]
    return shuffled, correct_letter


def build_gpqa_user_text(question: str, a: str, b: str, c: str, d: str) -> str:
    return f"""What is the correct answer to this question: {question}
Choices:
(A) {a}
(B) {b}
(C) {c}
(D) {d}

Format your response as follows: "The correct answer is (insert answer here)"."""


def build_chat_prompt(tokenizer, user_text: str) -> str:
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": user_text},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def decode_generation(out, tokenizer):
    gen_payload = out
    if isinstance(gen_payload, tuple):
        gen_payload = gen_payload[0]
    if isinstance(gen_payload, list) and len(gen_payload) > 0:
        if isinstance(gen_payload[0], list):
            token_ids = gen_payload[0]
        else:
            token_ids = gen_payload
        token_ids = [int(tid) for tid in token_ids if tid is not None]
    else:
        token_ids = []
    if not token_ids:
        return ""
    return tokenizer.decode(token_ids, skip_special_tokens=True).strip()


def main():
    args = parse_args()
    set_seed(42)

    pcm = args.polar_cache_module or os.environ.get("POLAR_CACHE_MODULE")
    if pcm:
        os.environ["POLAR_CACHE_MODULE"] = pcm
        print(colored(f"[PolarANN] POLAR_CACHE_MODULE = {pcm}", "cyan"))

    if args.dtype == "bf16":
        dtype = torch.bfloat16
    elif args.dtype == "fp16":
        dtype = torch.float16
    else:
        dtype = torch.float32

    data = load_gpqa_diamond(args.data_path, args.num_samples, args.start_idx)
    print(colored(f"加载 GPQA Diamond: {len(data)} 条 (path={args.data_path})", "yellow"))

    from model_hub.tokenizer_utils import load_tokenizer
    tokenizer = load_tokenizer(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model_config = AutoConfig.from_pretrained(args.model_name, trust_remote_code=True)
    model_max_length = model_config.max_position_embeddings
    input_reserve = max(args.max_new_tokens, 8192)
    required_max_len = args.max_new_tokens + input_reserve
    if model_max_length >= 100000:
        max_len = min(model_max_length, max(required_max_len, 65536))
    else:
        max_len = min(model_max_length, max(required_max_len, 32000))

    rope_scaling = None
    if getattr(args, 'rope_scaling_factor', None) is not None:
        rope_scaling = {
            "rope_type": "yarn",
            "factor": float(args.rope_scaling_factor),
            "original_max_position_embeddings": int(
                args.rope_scaling_orig_max if args.rope_scaling_orig_max is not None else 32768
            ),
        }
    llm = load_model(args.model_name, max_len, dtype, args.device, rope_scaling=rope_scaling)

    if args.output_file is None:
        args.output_file = f"gpqa_diamond_{args.attention_type}_{time.strftime('%Y%m%d_%H%M%S')}.jsonl"
    if os.path.isabs(args.output_file):
        output_path = args.output_file
    else:
        output_path = os.path.join(PROJECT_ROOT, "results", args.output_file)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(colored(f"输出: {os.path.abspath(output_path)}", "cyan"))

    correct_n = 0
    total_n = 0

    with open(output_path, "w", encoding="utf-8") as out_fp:
        for idx, row in enumerate(tqdm(data, desc="GPQA", ncols=100)):
            qtext = str(row.get("Question", "")).strip()
            sid = row.get("Record ID", row.get("id", idx))
            shuffled, correct_letter = shuffle_options(row)
            ca, cb, cc, cd = shuffled
            user_text = build_gpqa_user_text(qtext, ca, cb, cc, cd)
            prompt = build_chat_prompt(tokenizer, user_text)
            inputs = tokenizer([prompt], return_tensors="pt", padding=True)
            input_ids = inputs.input_ids.to(llm.layers[0].device)
            attention_masks = inputs.attention_mask.to(llm.layers[0].device)
            input_len = int(input_ids.shape[1])

            if args.attention_type in ["PolarANN", "Polar_PQ", "SuCo", "PQ"]:
                kvc = getattr(llm.layers[0], "kv_cache", None)
                if kvc is not None and hasattr(kvc, "reset_recall_history"):
                    kvc.reset_recall_history()

            config = generate_config(args.model_name, input_len, args.attention_type, args)
            if args.codebook_path and args.attention_type in config:
                config[args.attention_type]["codebook_path"] = args.codebook_path
            apply_polar_layout_overrides(config, args.attention_type, args)
            t0 = time.time()
            out = llm.generate(
                attention_type=args.attention_type,
                inputs_ids=input_ids,
                attention_masks=attention_masks,
                max_new_length=args.max_new_tokens,
                attn_config=config,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
            )
            elapsed = time.time() - t0
            response = decode_generation(out, tokenizer)
            predicted = extract_answer_from_response(response, enable_cot=False)
            pred_letter = predicted.upper().strip() if predicted else None
            if pred_letter and pred_letter not in "ABCD":
                pred_letter = None
            is_ok = pred_letter == correct_letter
            if is_ok:
                correct_n += 1
            total_n += 1

            rec = {
                "sample_id": sid,
                "question": qtext,
                "correct_letter": correct_letter,
                "predicted_letter": pred_letter,
                "is_correct": is_ok,
                "response": response,
                "generation_time": elapsed,
                "input_len": input_len,
            }
            out_fp.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out_fp.flush()

    acc = 100.0 * correct_n / total_n if total_n else 0.0
    print(colored(f"完成: {correct_n}/{total_n} = {acc:.2f}%", "green", attrs=["bold"]))
    print(colored(f"结果文件: {os.path.abspath(output_path)}", "cyan"))


if __name__ == "__main__":
    main()
