#!/usr/bin/env python3
"""AIME2025 evaluation with pass@k support (PolarANN / Full_Flash_Attn)."""

import torch
import json
import os
import sys
import argparse
import re
import time
import traceback

from termcolor import colored
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from model_hub.qwen import QwenModel


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    import numpy as np
    import random
    np.random.seed(seed)
    random.seed(seed)


def parse_args():
    parser = argparse.ArgumentParser(description="Test AIME2025 with pass@k evaluation")

    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size for inference")
    parser.add_argument("--max_new_tokens", type=int, default=2048,
                        help="Maximum new tokens to generate per sample")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="CUDA device (e.g. cuda:0)")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["fp16", "bf16"],
                        help="Model precision (fp16 or bf16)")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-8B",
                        help="HuggingFace model name or local path")

    parser.add_argument("--data_path", type=str, required=True,
                        help="AIME2025 data file path (JSON or JSONL)")
    parser.add_argument("--num_samples", type=int, default=-1,
                        help="Number of problems to test (-1 for all)")
    parser.add_argument("--start_idx", type=int, default=0,
                        help="Start index for testing (skip first N problems)")
    parser.add_argument("--output_file", type=str, default=None,
                        help="Output filename (saved under results/)")

    parser.add_argument("--attention_type", type=str, default="Full_Flash_Attn",
                        choices=["PolarANN", "Full_Flash_Attn"])
    parser.add_argument("--static_pattern_end", type=int, default=None,
                        help="Local-window size override for PolarANN (default: use config)")
    parser.add_argument("--final_topk", type=int, default=None,
                        help="Top-K after RaBitQ reranking (default: use config)")
    parser.add_argument("--enable_offload", type=int, default=None, choices=[0, 1],
                        help="Offload retrieval-zone KV to CPU pinned memory (0=off, 1=on)")
    parser.add_argument("--budget_ratio", type=float, default=0.2,
                        help="Budget ratio (used for display only)")

    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature (0.0=greedy for pass@1, 0.6-0.8 for pass@k)")
    parser.add_argument("--top_p", type=float, default=0.95,
                        help="Nucleus sampling top-p")
    parser.add_argument("--top_k", type=int, default=20,
                        help="Top-k sampling")

    parser.add_argument("--n_generate", type=int, default=1,
                        help="Answers to generate per problem (1=pass@1, 8=pass@8)")
    parser.add_argument("--pass_k", type=int, default=None,
                        help="Evaluate pass@k (default: same as n_generate)")
    parser.add_argument("--early_stop_on_first_correct", type=int, default=0,
                        help="Stop sampling a problem once any sample is correct (0/1)")

    args = parser.parse_args()
    if args.pass_k is None:
        args.pass_k = args.n_generate
    return args


def load_aime2025_data(data_path, num_samples=-1, start_idx=0):
    print(colored(f"Loading AIME2025 data: {data_path}", 'yellow'))

    data = []
    with open(data_path, 'r') as f:
        try:
            f.seek(0)
            data = json.load(f)
        except json.JSONDecodeError:
            f.seek(0)
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))

    for i, item in enumerate(data):
        if 'id' not in item:
            item['id'] = f"aime2025_{i+1}"
        if 'question' in item and 'problem' not in item:
            item['problem'] = item['question']

    if start_idx > 0:
        data = data[start_idx:]
        print(colored(f"  starting from index {start_idx}", 'yellow'))

    if num_samples > 0:
        data = data[:num_samples]

    print(colored(f"  loaded {len(data)} samples (index {start_idx}..{start_idx + len(data) - 1})", 'cyan'))
    return data


def build_aime_prompt(problem):
    return f"""You are an expert mathematical problem solver. Solve the following problem with complete step-by-step reasoning.

Problem: {problem}

Instructions:
1. Work through the problem step-by-step
2. Show your key reasoning and calculations
3. When you reach the final answer, write it in this EXACT format: \\boxed{{your_answer}}
4. STOP immediately after providing the boxed answer - do not verify or explore alternatives

Solve the problem:"""


def extract_answer_from_response(response):
    """Extract integer answer from model response (last complete \\boxed preferred)."""

    def extract_complete_boxed_contents(text):
        contents = []
        boxed_starts = [m.start() for m in re.finditer(r'\\boxed\{', text)]
        for start_pos in boxed_starts:
            pos = start_pos + 7
            brace_count = 1
            content_start = pos
            while pos < len(text) and brace_count > 0:
                if text[pos] == '{':
                    brace_count += 1
                elif text[pos] == '}':
                    brace_count -= 1
                pos += 1
            if brace_count == 0:
                content = re.sub(r'\s+', '', text[content_start:pos - 1]).strip()
                if content:
                    contents.append(content)
        return contents

    def to_strict_integer(candidate):
        if candidate is None:
            return None
        s = str(candidate).strip().replace("$", "").replace(",", "").replace(" ", "")
        if re.fullmatch(r"-?\d+", s):
            return str(int(s))
        if re.fullmatch(r"-?\d+\.0+", s):
            return str(int(float(s)))
        return None

    boxed_contents = extract_complete_boxed_contents(response)
    for content in reversed(boxed_contents):
        integer_answer = to_strict_integer(content)
        if integer_answer is not None:
            return integer_answer

    result_pattern = r'(?:S|sum|total|answer|result)\s*=\s*(-?\d+)'
    result_matches = list(re.finditer(result_pattern, response, re.IGNORECASE))
    if result_matches:
        return str(int(result_matches[-1].group(1)))

    match = re.search(
        r'[Tt]he (?:answer|solution) is[:\s]+(-?\d+)(?:\b|\.|$)',
        response, re.IGNORECASE,
    )
    if match:
        return str(int(match.group(1)))

    equals_matches = list(re.finditer(r'=\s*(-?\d+)\s*\.', response))
    if equals_matches:
        return str(int(equals_matches[-1].group(1)))

    lines = [line.strip() for line in response.strip().split('\n') if line.strip()]
    if lines and re.fullmatch(r"-?\d+", lines[-1]):
        return str(int(lines[-1]))

    return None


def normalize_answer(answer):
    if not answer:
        return ""
    answer = str(answer).strip()
    answer = re.sub(r'\s+', '', answer)
    answer = answer.replace('\\%', '').replace('%', '')
    answer = answer.replace('\\circ', '').replace('^\\circ', '').replace('^\circ', '')
    answer = re.sub(r'\\[dt]?frac', r'\\frac', answer)
    answer = re.sub(r'\\left\(', '(', answer)
    answer = re.sub(r'\\right\)', ')', answer)
    answer = re.sub(r'\\left\[', '[', answer)
    answer = re.sub(r'\\right\]', ']', answer)
    answer = re.sub(r'\\left\\{', '{', answer)
    answer = re.sub(r'\\right\\}', '}', answer)
    answer = answer.replace('\\,', '')
    answer = re.sub(r'\\text[a-z]*\{([^}]+)\}', r'\1', answer)
    return answer.lower()


def compare_answers(predicted, ground_truth):
    if not predicted or not ground_truth:
        return False
    pred_norm = normalize_answer(predicted)
    gt_norm = normalize_answer(ground_truth)
    if pred_norm == gt_norm:
        return True
    try:
        import sympy
        pred_expr = sympy.sympify(predicted.replace('\\', ''))
        gt_expr = sympy.sympify(ground_truth.replace('\\', ''))
        return sympy.simplify(pred_expr - gt_expr) == 0
    except Exception:
        pass
    return False


def generate_config(model_name, context_len, attention_type, args):
    model_name_lower = model_name.lower()
    if "qwen3-30b-a3b-thinking" in model_name_lower:
        config_file = "Qwen3-30B-A3B-Thinking-2507.json"
    elif "qwen3-4b-thinking" in model_name_lower:
        config_file = "Qwen3-4B-Thinking-2507.json"
    elif "qwen" in model_name_lower:
        config_file = "Qwen3-8B.json"
    elif "deepseek" in model_name_lower or "llama" in model_name_lower:
        config_file = "DeepSeek-R1-Distill-Llama-8B.json"
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    config_path = os.path.join(PROJECT_ROOT, "config", config_file)
    with open(config_path, "r") as f:
        config = json.load(f)

    if attention_type == "Full_Flash_Attn":
        return config

    if attention_type not in config:
        print(colored(f"WARNING: {attention_type} not found in {config_file}", 'yellow'))
        return config

    if args.static_pattern_end is not None:
        config[attention_type]['static_pattern_end'] = args.static_pattern_end

    if attention_type == "PolarANN":
        if getattr(args, 'final_topk', None) is not None:
            config[attention_type]['final_topk'] = args.final_topk
        if getattr(args, 'enable_offload', None) is not None:
            config[attention_type]['enable_offload'] = bool(args.enable_offload)

    return config


def load_model(model_name, max_len, dtype, device):
    print(colored(f"\nLoading model: {model_name}", 'yellow'))
    if "qwen" in model_name.lower():
        llm = QwenModel(model_name, max_length=max_len, dtype=dtype, device_map=device)
    else:
        raise ValueError(f"Unsupported model architecture: {model_name}. Only Qwen models are supported.")
    return llm


def main():
    args = parse_args()
    set_seed(2025)

    model_name = args.model_name
    dtype = torch.bfloat16 if args.dtype == 'bf16' else torch.float16
    device = args.device
    data_path = args.data_path

    data = load_aime2025_data(data_path, args.num_samples, args.start_idx)

    from transformers import AutoTokenizer, AutoConfig
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    model_max_length = model_config.max_position_embeddings
    input_reserve = max(args.max_new_tokens, 10000)
    required_max_len = args.max_new_tokens + input_reserve
    if model_max_length >= 100000:
        max_len = min(model_max_length, max(required_max_len, 65536))
    else:
        max_len = min(model_max_length, max(required_max_len, 32000))

    print(f"\nLength config:")
    print(f"  model max_position_embeddings: {model_max_length:,}")
    print(f"  max_new_tokens: {args.max_new_tokens:,}")
    print(f"  max_length: {max_len:,}")
    print(f"  available input space: {max_len - args.max_new_tokens:,}")

    if max_len < args.max_new_tokens:
        raise ValueError(f"max_length ({max_len}) must be >= max_new_tokens ({args.max_new_tokens})")

    llm = load_model(model_name, max_len, dtype, device)

    if args.output_file is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        pass_mode = f"pass@{args.pass_k}" if args.n_generate > 1 else "pass@1"
        args.output_file = f"aime2025_{args.attention_type}_{pass_mode}_{timestamp}.jsonl"

    output_path = (args.output_file if os.path.isabs(args.output_file)
                   else os.path.join(PROJECT_ROOT, "results", args.output_file))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    output_file = open(output_path, 'w', buffering=1)

    print(colored("\n" + "=" * 80, 'cyan'))
    print(colored("AIME2025 Evaluation", 'cyan', attrs=['bold']))
    print(colored("=" * 80, 'cyan'))
    print(f"Model: {model_name}")
    print(f"Attention: {args.attention_type}")
    print(f"Dataset: AIME2025 ({len(data)} problems)")
    print(f"Mode: pass@{args.pass_k} ({'Greedy' if args.temperature == 0 else 'Sampling'})")
    if args.n_generate > 1:
        print(f"  {args.n_generate} answers per problem")
        print(f"  sampling: temperature={args.temperature}, top_p={args.top_p}, top_k={args.top_k}")
        if bool(args.early_stop_on_first_correct):
            print("  early stop on first correct: enabled")
    print(f"Output: {output_path}")
    print(colored("=" * 80 + "\n", 'cyan'))

    results = []
    total_correct = 0
    total_problems = 0
    total_generated = 0

    for idx, sample in enumerate(tqdm(data, desc="Processing")):
        problem_id = sample.get('id', f"problem_{idx+1}")
        problem = sample['problem']
        ground_truth = sample['answer']

        print(colored(f"\n{'=' * 80}", 'cyan'))
        print(colored(f"Problem {idx+1}/{len(data)} | ID: {problem_id}", 'cyan'))
        print(f"\n{problem[:200]}..." if len(problem) > 200 else f"\n{problem}")
        print(f"Ground truth: {ground_truth}")

        base_prompt = build_aime_prompt(problem)
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": base_prompt}
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        inputs = tokenizer([prompt], return_tensors="pt", padding=True)
        input_ids = inputs.input_ids
        attention_masks = inputs.attention_mask
        input_len = input_ids.shape[1]
        print(colored(f"\nInput tokens: {input_len}", 'yellow'))

        config = generate_config(model_name, input_len, args.attention_type, args)

        all_generated_texts = []
        all_predicted_answers = []
        all_generation_times = []
        problem_is_correct = False
        early_stop_enabled = bool(args.early_stop_on_first_correct)

        print(colored(f"\nGenerating {args.n_generate} answer(s)...", 'yellow'))

        for gen_idx in range(args.n_generate):
            try:
                if args.attention_type == "PolarANN":
                    if hasattr(llm.layers[0], 'kv_cache') and hasattr(llm.layers[0].kv_cache, 'reset_recall_history'):
                        llm.layers[0].kv_cache.reset_recall_history()

                t_start = time.time()
                out = llm.generate(
                    attention_type=args.attention_type,
                    inputs_ids=input_ids.to(llm.layers[0].device),
                    attention_masks=attention_masks.to(llm.layers[0].device),
                    max_new_length=args.max_new_tokens,
                    attn_config=config,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,
                )
                t_end = time.time()

                if isinstance(out, tuple):
                    token_ids = out[0]
                elif isinstance(out, list) and len(out) > 0:
                    token_ids = out[0] if isinstance(out[0], list) else out
                else:
                    token_ids = []
                if token_ids and isinstance(token_ids[0], list):
                    token_ids = token_ids[0]
                token_ids = [int(tid) for tid in token_ids if tid is not None]

                generated_text = tokenizer.decode(token_ids, skip_special_tokens=True).strip() if token_ids else ""
                predicted_answer = extract_answer_from_response(generated_text)
                is_correct = compare_answers(predicted_answer, ground_truth)
                if is_correct:
                    problem_is_correct = True

                all_generated_texts.append(generated_text)
                all_predicted_answers.append(predicted_answer)
                all_generation_times.append(t_end - t_start)

                status = "CORRECT" if is_correct else "WRONG"
                print(colored(f"\n  [{gen_idx+1}/{args.n_generate}] done ({t_end - t_start:.2f}s, {len(token_ids)} tokens)", 'cyan'))
                print(f"    predicted: {predicted_answer}  [{status}]")

                single_result = {
                    'id': problem_id,
                    'problem': problem,
                    'ground_truth': ground_truth,
                    'sample_index': gen_idx + 1,
                    'n_generate': args.n_generate,
                    'predicted_answer': predicted_answer,
                    'generated_text': generated_text,
                    'is_correct': is_correct,
                    'generation_time': t_end - t_start,
                    'num_tokens': len(token_ids),
                    'input_length': input_len,
                    'type': 'single_sample',
                }
                output_file.write(json.dumps(single_result, ensure_ascii=False) + '\n')
                output_file.flush()
                os.fsync(output_file.fileno())

                del out, token_ids, generated_text

                if early_stop_enabled and is_correct:
                    print(colored(f"  [Early Stop] correct at sample {gen_idx+1}, skipping remaining", 'green'))
                    break

            except Exception as e:
                print(colored(f"\n  [{gen_idx+1}/{args.n_generate}] failed: {e}", 'red'))
                traceback.print_exc()
                all_generated_texts.append("")
                all_predicted_answers.append(None)
                all_generation_times.append(0)

        if problem_is_correct:
            total_correct += 1
        total_problems += 1
        total_generated += len(all_predicted_answers)

        correct_count = sum(1 for pred in all_predicted_answers if compare_answers(pred, ground_truth))
        avg_time = sum(all_generation_times) / len(all_generation_times) if all_generation_times else 0

        print(colored(f"\n{'=' * 80}", 'yellow'))
        print(colored(f"Problem {idx+1}/{len(data)} complete", 'yellow', attrs=['bold']))
        print(colored(f"{'=' * 80}", 'yellow'))
        print(f"Ground truth: {ground_truth}")
        print(f"Predictions: {all_predicted_answers}")
        print(f"Correct: {correct_count}/{len(all_predicted_answers)}")
        status = "PASS" if problem_is_correct else "FAIL"
        print(colored(f"pass@{args.pass_k}: {status}", 'green' if problem_is_correct else 'red'))
        print(f"Avg time: {avg_time:.2f}s")
        print(colored(f"\nRunning pass@{args.pass_k}: {total_correct}/{total_problems} = {100*total_correct/total_problems:.2f}%", 'cyan'))
        print(colored(f"{'=' * 80}\n", 'yellow'))

        result = {
            'id': problem_id,
            'problem': problem,
            'ground_truth': ground_truth,
            'n_generate': args.n_generate,
            'attempted_generate': len(all_predicted_answers),
            'all_predicted_answers': all_predicted_answers,
            'all_generated_texts': all_generated_texts,
            'all_generation_times': all_generation_times,
            'correct_count': correct_count,
            'pass_at_k': problem_is_correct,
            'input_length': input_len,
            'avg_generation_time': avg_time,
            'type': 'summary',
        }
        results.append(result)
        output_file.write(json.dumps(result, ensure_ascii=False) + '\n')
        output_file.flush()
        os.fsync(output_file.fileno())

        if hasattr(llm, 'kv_cache'):
            del llm.kv_cache
            llm.kv_cache = None
        for layer in llm.layers:
            if hasattr(layer, 'kv_cache'):
                del layer.kv_cache
                layer.kv_cache = None
        torch.cuda.empty_cache()

    output_file.close()

    print(colored("\n" + "=" * 80, 'green'))
    print(colored("Done", 'green', attrs=['bold']))
    print(colored("=" * 80, 'green'))
    print(f"\nTotal problems: {total_problems}")
    print(f"Total answers generated: {total_generated}")
    print(f"pass@{args.pass_k} correct: {total_correct}")
    if total_problems > 0:
        print(colored(f"\npass@{args.pass_k} accuracy: {100*total_correct/total_problems:.2f}%", 'cyan', attrs=['bold']))

    if results:
        avg_correct_per_problem = sum(r['correct_count'] for r in results) / len(results)
        avg_at_k = avg_correct_per_problem / args.n_generate
        print(f"Avg correct per problem: {avg_correct_per_problem:.2f}/{args.n_generate}")
        print(colored(f"avg@{args.pass_k}: {100*avg_at_k:.2f}%", 'cyan', attrs=['bold']))

    print(f"\nResults saved to: {output_path}")
    print(colored("=" * 80 + "\n", 'green'))


if __name__ == "__main__":
    main()
