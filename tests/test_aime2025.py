#!/usr/bin/env python3
"""
测试 AIME2025 数据集 + 支持 pass@k 评估

数据集: AIME2025 (American Invitational Mathematics Examination 2025)
模型: 支持 Qwen3-8B / DeepSeek-R1-Distill-Llama-8B
KV Cache: Full_Flash_Attn / PolarANN / Polar_PQ / SuCo / PQ / RetroInfer
评估模式: pass@1 (greedy) 或 pass@k (sampling)
"""

import torch
import json
import os
import sys
import argparse
from termcolor import colored
import time
from tqdm import tqdm

# 添加路径（Python 文件在 run/ 目录下，需要指向父目录）


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)  # 父目录（RetrievalAttention）
sys.path.insert(0, PROJECT_ROOT)

# 导入模型（density release 包剥离了 llama，import 失败时兜底；走 if "llama" 分支再报错）
from model_hub.qwen import QwenModel
try:
    from model_hub.llama import LlamaModel  # type: ignore
except ImportError:  # pragma: no cover - density release stripped llama support
    LlamaModel = None  # type: ignore


def set_seed(seed):
    """设置随机种子 + 确定性后端（用于 KV cache 复现实验）"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    import numpy as np
    import random
    np.random.seed(seed)
    random.seed(seed)
    # 确定性后端：仅当 POLAR_DETERMINISTIC=1 时启用，避免影响其他基准 benchmark
    if os.environ.get("POLAR_DETERMINISTIC", "0") in ("1", "true", "True"):
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            print("[set_seed] deterministic algorithms ENABLED (warn_only=True)", flush=True)
        except Exception as e:
            print(f"[set_seed] deterministic setup failed: {e}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Test AIME2025 with pass@k evaluation")
    
    # 基本参数
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--max_new_tokens", type=int, default=2048, help="Maximum new tokens to generate")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["fp16", "bf16", "fp32"], help="Data type")
    parser.add_argument("--model_name", type=str,
                        default="/home/secure/pariskv/models/Qwen3-4B",
                        help="Model name or path (默认本机 Qwen3-4B; 也可写成 HF 名 Qwen/Qwen3-8B)")

    # 数据参数
    parser.add_argument("--data_path", type=str,
                        default="/home/secure/pariskv/data/AIME2025/test.jsonl",
                        help="AIME2025 data file path (JSON 或 JSONL)")
    # YaRN: 当模型自身 max_position_embeddings 不够 max_length 时启用 (与 test_longbench_v2 同名)
    parser.add_argument("--rope_scaling_factor", type=float, default=None,
                        help="YaRN rope_scaling factor (e.g. 4.0 for Qwen3-4B native 32K -> 128K)")
    parser.add_argument("--rope_scaling_orig_max", type=int, default=None,
                        help="YaRN rope_scaling original_max_position_embeddings (e.g. 32768)")
    # ---- pass@k retry on failed problems with early-exit ----
    parser.add_argument("--retry_failed_from", type=str, default=None,
                        help="Path to a previous run's JSONL. Will pick up rows where is_correct=False "
                             "(based on 'id' field) and run only those, overriding --sample_indices/--num_samples.")
    parser.add_argument("--early_exit_on_correct", action="store_true",
                        help="During pass@k generation, stop the inner loop as soon as one attempt is correct. "
                             "Use with --n_generate 8 --pass_k 8 for early-exit pass@8.")
    parser.add_argument("--num_samples", type=int, default=-1, 
                        help="Number of problems to test (default: -1 for all 30)")
    parser.add_argument("--start_idx", type=int, default=0,
                        help="Start index for testing (default: 0)")
    parser.add_argument("--sample_indices", type=str, default=None,
                        help="Comma-separated explicit sample indices, **1-based** "
                             "(e.g. '7,9,15,30' selects the 7th/9th/15th/30th problem). "
                             "If set, overrides --start_idx / --num_samples.")
    parser.add_argument("--output_file", type=str, default=None,
                        help="Output file for results (auto-generated if not specified)")
    
    # Attention 类型
    parser.add_argument("--attention_type", type=str, default="Full_Flash_Attn",
                        choices=["SuCo", "PQ", "PolarANN", "Polar_PQ", "RetroInfer", "Full_Flash_Attn"],
                        help="Attention type")
    parser.add_argument("--collision_ratio", type=float, default=None,
                        help="Collision search ratio for Polar ANN (default: use config file)")
    parser.add_argument("--candidate_ratio", type=float, default=None,
                        help="Candidate filtering ratio for Polar ANN (default: use config file)")
    parser.add_argument("--cluster_num_per_subspace", type=int, default=32,
                        help="Number of clusters per subspace for SuCo/PQ (default: 32)")
    parser.add_argument("--num_subspaces", type=int, default=32,
                        help="Number of subspaces (B) for SuCo/PQ (default: 32)")
    parser.add_argument("--static_pattern_end", type=int, default=None,
                        help="Static pattern end (local tokens size) for PolarANN/Polar_PQ (default: use config file)")
    
    # 采样参数
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature (0.0=greedy for pass@1, 0.6-0.8 for pass@k)")
    parser.add_argument("--top_p", type=float, default=0.95,
                        help="Nucleus sampling top_p")
    parser.add_argument("--top_k", type=int, default=20,
                        help="Top-k sampling")
    
    # 🔥 pass@k 参数
    parser.add_argument("--n_generate", type=int, default=1,
                        help="Number of answers to generate per problem (1 for pass@1, 8 for pass@8)")
    parser.add_argument("--pass_k", type=int, default=None,
                        help="If set, will evaluate pass@k (e.g., 8 for pass@8). If None, uses n_generate value.")

    # density release 默认走 polar_cache_pure_rabitq_quest_density (与 README 主入口一致);
    # 若想退回 base 模块, 手动 --polar_cache_module cache_hub.polar_cache 即可
    parser.add_argument("--polar_cache_module", type=str,
                        default="cache_hub.polar_cache_pure_rabitq_quest_density",
                        help="Import path exposing `polar_cache` (PolarANN backend). CLI wins over "
                             "$POLAR_CACHE_MODULE when both set (values merged into env before model load). "
                             "Effective module in model_hub/qwen.py: config PolarANN['cache_module'] > "
                             "$POLAR_CACHE_MODULE > <this default>. "
                             "Density variant default: cache_hub.polar_cache_pure_rabitq_quest_density")

    parser.add_argument("--adaptive_topk_threshold", type=float, default=None,
                        help="Override config[attention_type]['adaptive_topk_threshold'] for adaptive-K cache modules "
                             "(e.g. 0.85). When None, uses the value from the config JSON.")
    parser.add_argument("--adaptive_topk_min_k", type=int, default=None,
                        help="Override PolarANN adaptive_topk_min_k (per-head floor before global max).")
    parser.add_argument("--adaptive_topk_max_k", type=int, default=None,
                        help="Override PolarANN adaptive_topk_max_k (per-head cap before global max).")
    parser.add_argument("--adaptive_topk_enabled", type=int, default=None, choices=[0, 1],
                        help="Override PolarANN adaptive_topk_enabled (1=on). For polar_cache_pure_rabitq_adaptiveK.")

    parser.add_argument("--enable_offload", type=int, default=None, choices=[0, 1],
                        help="Override config[PolarANN]['enable_offload'] (1=offload to CPU pinned memory, "
                             "required by polar_cache_value_adaptive_k_cuda / _cuda2). Default: use config JSON.")
    parser.add_argument("--sink_size", type=int, default=None,
                        help="Override sink zone tokens (PolarANN/Polar_PQ blocks in config)")
    parser.add_argument("--local_size", type=int, default=None,
                        help="Override local window tokens")
    parser.add_argument("--dynamic_update_interval", type=int, default=None,
                        help="Override PolarANN steady-zone update cadence")
    parser.add_argument("--final_topk", type=int, default=None,
                        help="Override rerank retrieval final_topk")
    parser.add_argument(
        "--enable_coarse_recall_ablation",
        type=lambda x: str(x).lower() in ("1", "true", "yes"),
        default=None,
        help=(
            "Forward to PolarANN config: Stage-1 coarse Recall@K ablation (collision path). "
            "Pure-RaBitQ default path skips collision and won't print [Stage1]; useful when "
            "POLAR_PURE_RABITQ=0 or backend uses collision."
        ),
    )
    parser.add_argument(
        "--enable_rerank_recall_ablation",
        type=lambda x: str(x).lower() in ("1", "true", "yes"),
        default=None,
        help=(
            "Forward to PolarANN config: Rerank Recall@100 ablation (Layer 4 / Head 4). "
            "Triggers [Rerank] lines in polar_cache_pure_rabitq._pure_rabitq_topk_full once "
            "decode enters RETRIEVAL mode."
        ),
    )

    args = parser.parse_args()
    
    # 如果 pass_k 未指定，默认使用 n_generate 的值
    if args.pass_k is None:
        args.pass_k = args.n_generate
    
    return args


def load_aime2025_data(data_path, num_samples=-1, start_idx=0, sample_indices=None):
    """加载 AIME2025 数据集（支持 JSON 和 JSONL 格式）

    sample_indices: 可选 list[int]，如果提供则忽略 start_idx/num_samples，
    严格按给定下标列表抽取（保留顺序，去重）。
    """
    print(colored(f"加载 AIME2025 数据: {data_path}", 'yellow'))
    
    data = []
    with open(data_path, 'r') as f:
        # 先尝试作为完整的 JSON 文件加载
        try:
            f.seek(0)
            data = json.load(f)
        except json.JSONDecodeError:
            # 如果失败，尝试作为 JSONL 格式逐行读取
            f.seek(0)
            for line in f:
                line = line.strip()
                if line:  # 跳过空行
                    data.append(json.loads(line))
    
    # 为数据添加 ID（如果没有）
    for i, item in enumerate(data):
        if 'id' not in item:
            item['id'] = f"aime2025_{i+1}"
        # 统一字段名
        if 'question' in item and 'problem' not in item:
            item['problem'] = item['question']
    
    # 优先：显式 sample_indices
    if sample_indices is not None and len(sample_indices) > 0:
        seen = set()
        ordered_unique = []
        for idx in sample_indices:
            if idx in seen:
                continue
            seen.add(idx)
            ordered_unique.append(idx)
        n_total = len(data)
        out_of_range = [i for i in ordered_unique if i < 0 or i >= n_total]
        if out_of_range:
            raise IndexError(
                f"sample_indices 中存在越界下标 {out_of_range}（数据集大小 {n_total}）"
            )
        data = [data[i] for i in ordered_unique]
        # 这里 ordered_unique 是内部 0-based；同时打 1-based 方便对照用户输入
        one_based = [i + 1 for i in ordered_unique]
        print(colored(
            f"按 sample_indices 加载 {len(data)} 个样本: 1-based={one_based} (0-based={ordered_unique})",
            'cyan',
        ))
        return data

    # 否则：start_idx / num_samples 走老路径
    if start_idx > 0:
        data = data[start_idx:]
        print(colored(f"从索引 {start_idx} 开始", 'yellow'))

    if num_samples > 0:
        data = data[:num_samples]

    print(colored(f"加载了 {len(data)} 个样本 (索引 {start_idx} 到 {start_idx + len(data) - 1})", 'cyan'))
    return data


def build_aime_prompt(problem):
    """构建 AIME 问题的 prompt"""
    prompt = f"""You are an expert mathematical problem solver. Solve the following problem with complete step-by-step reasoning.

Problem: {problem}

Instructions:
1. Work through the problem step-by-step
2. Show your key reasoning and calculations
3. When you reach the final answer, put only the final integer inside a LaTeX boxed expression
4. STOP immediately after providing the boxed answer - do not verify or explore alternatives

Solve the problem:"""
    
    return prompt


def extract_answer_from_response(response):
    """从模型响应中提取答案（支持复杂 LaTeX 格式）"""
    import re

    # Some long generations can drift into a new chat turn and repeat the
    # original prompt.  Do not let prompt placeholders after that boundary
    # override an earlier valid final answer.  Cut at the EARLIEST role/prompt
    # boundary among several markers (not just the first one in list order).
    _boundary_markers = (
        "\nHuman:", "\nUser:", "\nAssistant:", "\nSystem:",
        "Human:", "User:", "Assistant:", "System:",
        "You are an expert mathematical problem solver",
        "<|im_start|>", "<|im_end|>",
    )
    cut = len(response)
    for marker in _boundary_markers:
        mp = response.find(marker)
        if mp != -1:
            cut = min(cut, mp)
    if cut < len(response):
        response = response[:cut]

    def _is_placeholder_boxed(content):
        norm = re.sub(r'[\s{}\\]+', '', str(content).strip().lower())
        if not norm:
            return True
        if "your_answer" in norm or "youranswer" in norm:
            return True
        return norm in {
            "answer",
            "finalanswer",
            "theanswer",
            "yourfinalanswer",
            "integeranswer",
            "finalinteger",
        }

    def _prefer_int(content):
        """AIME answers are integers 0-999. After a real boxed hit, prefer a
        clean integer if the content reduces to one (handles ``\\text{70}``,
        ``70.``, ``1,000`` etc.), else return None to keep raw content."""
        cleaned = re.sub(r'\\text[a-z]*\{([^}]*)\}', r'\1', str(content))
        cleaned = cleaned.replace('$', '').replace(',', '').strip()
        cleaned = cleaned.rstrip('.')
        m = re.fullmatch(r'[+-]?\d{1,7}', cleaned)
        if m:
            return str(int(cleaned))
        return None
    
    # 格式1: "\\boxed{...}" - 取最后一个完整的 boxed 作为最终答案。
    # thinking 模型（如 Qwen3-Thinking）会在推理链里反复写中间的 \boxed{}，
    # 取第一个会抓到推理早期的错误试探值，因此必须从最后一个往前找。
    boxed_starts = [m.start() for m in re.finditer(r'\\boxed\{', response)]
    if boxed_starts:
        for start_pos in reversed(boxed_starts):
            pos = start_pos + 7  # len('\\boxed{')
            brace_count = 1
            content_start = pos
            while pos < len(response) and brace_count > 0:
                if response[pos] == '{':
                    brace_count += 1
                elif response[pos] == '}':
                    brace_count -= 1
                pos += 1
            if brace_count == 0:
                content = response[content_start:pos-1]
                # 清理答案：去除多余空格，提取纯数字（如果是纯数字答案）
                content = re.sub(r'\s+', ' ', content).strip()
                if _is_placeholder_boxed(content):
                    continue
                # AIME 答案是整数：boxed 命中后优先归一化成整数。
                int_ans = _prefer_int(content)
                if int_ans is not None:
                    return int_ans
                # 如果内容是简单的数字表达式，直接返回
                if re.match(r'^-?\d+(\.\d+)?$', content):
                    return content
                return content
    
    # 格式2: "#### answer" - MATH 数据集格式
    match = re.search(r'####\s*(.+)', response)
    if match:
        return match.group(1).strip()
    
    # 格式3: "The answer/solution is ..."
    match = re.search(r'[Tt]he (?:answer|solution) is[:\s]+(.+?)(?:\.|$)', response, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    
    # 格式4: 查找结果变量模式 "S = 数字" 或 "answer = 数字"（排除中间变量如 N=1）
    result_pattern = r'(?:S|sum|total|answer|result)\s*=\s*(-?\d+(?:\.\d+)?)'
    result_matches = list(re.finditer(result_pattern, response, re.IGNORECASE))
    if result_matches:
        return result_matches[-1].group(1)
    
    # 格式5: 句尾的 "= 数字." 格式（更安全，避免提取中间步骤）
    equals_matches = list(re.finditer(r'=\s*(-?\d+(?:\.\d+)?)\s*\.', response))
    if equals_matches:
        return equals_matches[-1].group(1)
    
    # 格式6: 最后一行的内容（作为后备）
    lines = [line.strip() for line in response.strip().split('\n') if line.strip()]
    if lines:
        last_line = lines[-1]
        # 排除验证过程和不完整的行
        if not any(keyword in last_line.lower() for keyword in ['user:', 'assistant:', 'question:', 'problem:', 'let', 'we have', 'therefore']):
            # 尝试从最后一行提取数字
            num_match = re.search(r'(-?\d+(?:\.\d+)?)', last_line)
            if num_match:
                return num_match.group(1)
    
    return None


def normalize_answer(answer):
    """标准化答案以便比对（处理 LaTeX 格式差异）"""
    if not answer:
        return ""
    
    import re
    
    # 转为字符串
    answer = str(answer).strip()
    
    # 1. 统一空格
    answer = re.sub(r'\s+', '', answer)
    
    # 2. 移除百分号、度数符号
    answer = answer.replace('\\%', '').replace('%', '')
    answer = answer.replace('\\circ', '').replace('^\\circ', '').replace('^\circ', '')
    
    # 3. 统一分数格式
    answer = re.sub(r'\\[dt]?frac', r'\\frac', answer)
    
    # 4. 统一括号
    answer = re.sub(r'\\left\(', '(', answer)
    answer = re.sub(r'\\right\)', ')', answer)
    answer = re.sub(r'\\left\[', '[', answer)
    answer = re.sub(r'\\right\]', ']', answer)
    answer = re.sub(r'\\left\\{', '{', answer)
    answer = re.sub(r'\\right\\}', '}', answer)
    
    # 5. 移除 \, (thin space)
    answer = answer.replace('\\,', '')
    
    # 6. 统一文本格式
    answer = re.sub(r'\\text[a-z]*\{([^}]+)\}', r'\1', answer)
    
    # 7. 转为小写
    answer_lower = answer.lower()
    
    return answer_lower


def compare_answers(predicted, ground_truth):
    """比较预测答案和真实答案"""
    if not predicted or not ground_truth:
        return False

    import re

    # 兜底：若提取结果前缀是纯数值（如 "588? But ..."），优先补一个数值候选用于判分。
    pred_text = str(predicted).strip()
    pred_candidates = [pred_text]
    leading_num = re.match(r'^\s*([+-]?\d+(?:\.\d+)?)', pred_text)
    if leading_num:
        num_prefix = leading_num.group(1)
        if num_prefix != pred_text:
            pred_candidates.append(num_prefix)

    gt_norm = normalize_answer(ground_truth)

    for pred_candidate in pred_candidates:
        # 标准化后比较
        pred_norm = normalize_answer(pred_candidate)
        if pred_norm == gt_norm:
            return True

        # 尝试数值比较
        try:
            import sympy
            pred_expr = sympy.sympify(str(pred_candidate).replace('\\', ''))
            gt_expr = sympy.sympify(str(ground_truth).replace('\\', ''))
            if sympy.simplify(pred_expr - gt_expr) == 0:
                return True
        except:
            pass

    return False


def generate_config(model_name, context_len, attention_type, args):
    """生成配置（通用函数）"""
    # 判断模型架构: 1) 优先 <basename>.json 精确命中; 2) 没有再走关键字 fallback
    CONFIG_DIR = os.path.join(PROJECT_ROOT, "config")
    basename_json = os.path.basename(model_name.rstrip('/')) + ".json"
    if os.path.exists(os.path.join(CONFIG_DIR, basename_json)):
        config_file = basename_json
    else:
        model_name_lower = model_name.lower()
        if "qwen3-30b-a3b-thinking" in model_name_lower:
            config_file = "Qwen3-30B-A3B-Thinking-2507.json"
        elif "qwen3-4b-thinking" in model_name_lower:
            config_file = "Qwen3-4B-Thinking-2507.json"
        elif "qwen3-4b" in model_name_lower:                   # <- 新增: 普通 Qwen3-4B
            config_file = "Qwen3-4B.json"
        elif "qwen" in model_name_lower:
            config_file = "Qwen3-8B.json"
        elif "deepseek" in model_name_lower or "llama" in model_name_lower:
            config_file = "DeepSeek-R1-Distill-Llama-8B.json"
        else:
            raise ValueError(f"Unsupported model: {model_name}")
    CONFIG_FILE = os.path.join(CONFIG_DIR, config_file)
    print(colored(f"  config: {config_file}", 'cyan'))
    
    with open(CONFIG_FILE, "r") as f:
        config = json.load(f)
    
    if attention_type == "Full_Flash_Attn":
        return config
    
    # 根据 attention_type 更新配置
    if attention_type not in config:
        print(colored(f"⚠️ 配置文件中没有 {attention_type} 配置", 'yellow'))
        return config
    
    # 只在用户显式提供参数时才覆盖配置文件
    if args.collision_ratio is not None:
        config[attention_type]['collision_ratio'] = args.collision_ratio
    if args.candidate_ratio is not None:
        config[attention_type]['candidate_ratio'] = args.candidate_ratio
    if args.static_pattern_end is not None:
        config[attention_type]['static_pattern_end'] = args.static_pattern_end
        print(colored(f"使用命令行参数覆盖 static_pattern_end (local tokens): {args.static_pattern_end}", 'yellow'))
    if getattr(args, "adaptive_topk_threshold", None) is not None:
        old_thr = config[attention_type].get('adaptive_topk_threshold', None)
        config[attention_type]['adaptive_topk_threshold'] = args.adaptive_topk_threshold
        print(colored(
            f"使用命令行参数覆盖 adaptive_topk_threshold: {old_thr} -> {args.adaptive_topk_threshold}",
            'yellow',
        ))
    if getattr(args, "adaptive_topk_min_k", None) is not None:
        old_v = config[attention_type].get("adaptive_topk_min_k", None)
        config[attention_type]["adaptive_topk_min_k"] = int(args.adaptive_topk_min_k)
        print(colored(
            f"使用命令行参数覆盖 adaptive_topk_min_k: {old_v} -> {args.adaptive_topk_min_k}",
            "yellow",
        ))
    if getattr(args, "adaptive_topk_max_k", None) is not None:
        old_v = config[attention_type].get("adaptive_topk_max_k", None)
        config[attention_type]["adaptive_topk_max_k"] = int(args.adaptive_topk_max_k)
        print(colored(
            f"使用命令行参数覆盖 adaptive_topk_max_k: {old_v} -> {args.adaptive_topk_max_k}",
            "yellow",
        ))
    if getattr(args, "adaptive_topk_enabled", None) is not None:
        old_v = config[attention_type].get("adaptive_topk_enabled", None)
        config[attention_type]["adaptive_topk_enabled"] = bool(args.adaptive_topk_enabled)
        print(colored(
            f"使用命令行参数覆盖 adaptive_topk_enabled: {old_v} -> {bool(args.adaptive_topk_enabled)}",
            "yellow",
        ))
    if getattr(args, "enable_offload", None) is not None:
        old_off = config[attention_type].get('enable_offload', None)
        config[attention_type]['enable_offload'] = (args.enable_offload == 1)
        print(colored(
            f"使用命令行参数覆盖 enable_offload: {old_off} -> {bool(args.enable_offload)}",
            'yellow',
        ))
    blk = config[attention_type]
    if attention_type in ("PolarANN", "Polar_PQ") and isinstance(blk, dict):
        _overrides = []
        if getattr(args, 'sink_size', None) is not None:
            blk['sink_size'] = args.sink_size
            _overrides.append(f"sink_size={args.sink_size}")
        if getattr(args, 'local_size', None) is not None:
            blk['local_size'] = args.local_size
            _overrides.append(f"local_size={args.local_size}")
        if getattr(args, 'dynamic_update_interval', None) is not None:
            blk['dynamic_update_interval'] = args.dynamic_update_interval
            _overrides.append(f"dynamic_update_interval={args.dynamic_update_interval}")
        if getattr(args, 'final_topk', None) is not None:
            blk['final_topk'] = args.final_topk
            _overrides.append(f"final_topk={args.final_topk}")
        if getattr(args, 'enable_coarse_recall_ablation', None) is not None:
            blk['enable_coarse_recall_ablation'] = bool(args.enable_coarse_recall_ablation)
            _overrides.append(
                f"enable_coarse_recall_ablation={bool(args.enable_coarse_recall_ablation)}"
            )
        if getattr(args, 'enable_rerank_recall_ablation', None) is not None:
            blk['enable_rerank_recall_ablation'] = bool(args.enable_rerank_recall_ablation)
            _overrides.append(
                f"enable_rerank_recall_ablation={bool(args.enable_rerank_recall_ablation)}"
            )
        if _overrides:
            print(colored(f"使用命令行参数覆盖 PolarANN/Polar_PQ: {', '.join(_overrides)}", 'yellow'))

    # 计算 nprobe（使用配置中的 candidate_ratio，与 polar_cache.py 逻辑一致）
    candidate_ratio = config[attention_type].get('candidate_ratio', 0.2)
    MIN_TOPK = 30
    calculated_topk = int(context_len * candidate_ratio)
    config[attention_type]['nprobe'] = max(MIN_TOPK, min(calculated_topk, context_len))
    
    # SuCo/PQ 特殊处理
    if attention_type in ["SuCo", "PQ"]:
        config[attention_type]['num_subspaces'] = args.num_subspaces
        config[attention_type]['cluster_num_per_subspace'] = args.cluster_num_per_subspace
    
    # PolarANN/Polar_PQ: 不传递 polar_K_r 和 polar_K_omega
    # 这些参数将从 codebook 文件中自动读取
    
    return config


def load_model(model_name, max_len, dtype, device, rope_scaling=None):
    """加载模型 (rope_scaling: 可选 dict, 启用 YaRN 等; None 时走模型 config 内置)"""
    print(colored(f"\n加载模型: {model_name}", 'yellow'))
    if rope_scaling is not None:
        print(colored(f"  rope_scaling: {rope_scaling}", 'cyan'))

    if "qwen" in model_name.lower():
        llm = QwenModel(
            model_name,
            max_length=max_len,
            dtype=dtype,
            device_map=device,
            rope_scaling=rope_scaling,
        )
    elif "deepseek" in model_name.lower() or "llama" in model_name.lower():
        if LlamaModel is None:
            raise ImportError(
                "model_hub.llama 在本 release 包中被剥离 (PolarANN density variant). "
                "如需跑 Llama/DeepSeek, 请用完整版仓库或自行补回 model_hub/llama.py"
            )
        llm = LlamaModel(
            model_name,
            max_length=max_len,
            dtype=dtype,
            device_map=device,
        )
    else:
        raise ValueError(f"Unsupported model architecture: {model_name}")

    return llm


def main():
    args = parse_args()
    set_seed(2025)

    # Mirror test_gpqa / shell launchers: honor $POLAR_CACHE_MODULE when CLI omits --polar_cache_module.
    pcm = getattr(args, "polar_cache_module", None) or os.environ.get("POLAR_CACHE_MODULE")
    if pcm:
        os.environ["POLAR_CACHE_MODULE"] = pcm
    if args.attention_type in ("PolarANN", "Polar_PQ"):
        if pcm:
            print(colored(
                f'[PolarANN] POLAR_CACHE_MODULE (CLI/env; JSON PolarANN["cache_module"] overrides) = {pcm}',
                "cyan",
            ))
        else:
            print(colored(
                '[PolarANN] POLAR_CACHE_MODULE unset → Qwen uses PolarANN["cache_module"] in config '
                'if present, else cache_hub.polar_cache (see later: [PolarANN] using cache module: ...)',
                "cyan",
            ))
    model_name = args.model_name
    if args.dtype == 'bf16':
        dtype = torch.bfloat16
    elif args.dtype == 'fp16':
        dtype = torch.float16
    else:  # fp32
        dtype = torch.float32
    device = args.device
    data_path = args.data_path
    
    # 加载数据
    # NOTE: --sample_indices 是 **1-based** 用户编号（与题目"第 N 题"一致）；
    # 这里在边界处统一减 1 转成内部 0-based 下标，再传给 load_aime2025_data。
    sample_indices_list = None
    if getattr(args, "sample_indices", None):
        raw_1based = [int(x.strip()) for x in args.sample_indices.split(",") if x.strip()]
        bad = [v for v in raw_1based if v < 1]
        if bad:
            raise ValueError(
                f"--sample_indices 必须是 1-based 正整数，收到非法值: {bad}"
            )
        sample_indices_list = [v - 1 for v in raw_1based]
        print(colored(
            f"--sample_indices (1-based)={raw_1based} -> internal 0-based={sample_indices_list}",
            'cyan',
        ))

    # ---- retry_failed_from: 自动从 source jsonl 推断错题 sample_indices ----
    # 逻辑：扫 source jsonl 所有 type=summary（每条对应 1 道题最终判定）；
    # 找 correct_count == 0 (即 pass@k=0, 全部 attempt 都错) 的 problem_id；
    # 这些 id 转成 1-based index 塞给 sample_indices_list。
    if getattr(args, "retry_failed_from", None):
        source_path = args.retry_failed_from
        if not os.path.exists(source_path):
            raise FileNotFoundError(f"--retry_failed_from path not found: {source_path}")
        # 先把数据集按顺序读一遍拿到 (id -> 0-based-index) 映射
        full_data = load_aime2025_data(data_path, num_samples=-1, start_idx=0, sample_indices=None)
        id_to_idx = {str(d.get('id', f'aime2025_{i+1}')): i for i, d in enumerate(full_data)}
        # 再扫 source jsonl 收集错题 id
        wrong_ids = []
        seen = set()
        for line in open(source_path, 'r'):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            pid = str(rec.get('id', ''))
            if not pid or pid in seen:
                continue
            # 优先看 summary 行 (含 correct_count); 否则看 single_sample 的 is_correct
            if rec.get('type') == 'summary':
                if int(rec.get('correct_count', 0)) == 0:
                    wrong_ids.append(pid)
                    seen.add(pid)
            elif rec.get('type') == 'single_sample':
                # summary 缺失时回退: 任一 sample is_correct=True 就算对
                if pid not in seen:
                    if not bool(rec.get('is_correct', False)):
                        wrong_ids.append(pid)
                    else:
                        seen.add(pid)  # 已经至少有一次对, 不进 retry
                        if pid in wrong_ids:
                            wrong_ids.remove(pid)
        # 去重 + 转 0-based 下标
        wrong_ids = list(dict.fromkeys(wrong_ids))
        miss = [pid for pid in wrong_ids if pid not in id_to_idx]
        if miss:
            print(colored(f"WARN: source jsonl 的 id {miss[:5]} 在数据集里找不到, 跳过", 'yellow'))
        wrong_zero_based = [id_to_idx[pid] for pid in wrong_ids if pid in id_to_idx]
        if not wrong_zero_based:
            print(colored(f"--retry_failed_from {source_path} 里没有 wrong 题目, 退出", 'green'))
            sys.exit(0)
        sample_indices_list = sorted(set(wrong_zero_based))
        wrong_1based = [i + 1 for i in sample_indices_list]
        print(colored(
            f"--retry_failed_from: 从 {source_path} 推断错题 ids={wrong_ids}, "
            f"对应 1-based={wrong_1based}, 共 {len(sample_indices_list)} 题",
            'cyan',
        ))
    data = load_aime2025_data(
        data_path,
        args.num_samples,
        args.start_idx,
        sample_indices=sample_indices_list,
    )
    
    # 加载 tokenizer
    print(colored("\n加载 Tokenizer...", 'yellow'))
    from model_hub.tokenizer_utils import load_tokenizer
    tokenizer = load_tokenizer(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    # 动态计算最大长度：确保 max_len >= max_new_tokens + 输入空间
    # 从模型配置获取 max_position_embeddings
    from transformers import AutoConfig
    model_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    model_max_length = model_config.max_position_embeddings 
    
    # max_len 需要至少能容纳输入 + 输出
    # 策略：至少为输入预留 max_new_tokens 的空间，或者至少 10K tokens
    # 这样可以确保输入长度 + 输出长度不会超过 max_len
    input_reserve = max(args.max_new_tokens, 10000)
    required_max_len = args.max_new_tokens + input_reserve
    
    # 使用模型支持的最大值，但至少满足需求
    # 对于大模型（如 Qwen3-30B），可以使用更大的 max_length
    if model_max_length >= 100000:
        # 大模型：可以使用更大的 max_length，但不超过模型限制
        max_len = min(model_max_length, max(required_max_len, 65536))
    else:
        # 小模型：使用较小的 max_length
        max_len = min(model_max_length, max(required_max_len, 32000))
    
    print(colored(f"\n最大长度设置:", 'yellow'))
    print(colored(f"  模型 max_position_embeddings: {model_max_length:,}", 'cyan'))
    print(colored(f"  max_new_tokens: {args.max_new_tokens:,}", 'cyan'))
    print(colored(f"  设置的 max_length: {max_len:,}", 'cyan'))
    print(colored(f"  可用输入空间: {max_len - args.max_new_tokens:,}", 'cyan'))
    
    # 验证设置是否合理
    if max_len < args.max_new_tokens:
        raise ValueError(f"max_length ({max_len}) 必须 >= max_new_tokens ({args.max_new_tokens})")
    
    # 加载模型（构造可选 YaRN dict，仅当用户显式 --rope_scaling_factor 时启用）
    rope_scaling = None
    if getattr(args, 'rope_scaling_factor', None) is not None:
        rope_scaling = {
            "rope_type": "yarn",
            "factor": float(args.rope_scaling_factor),
            "original_max_position_embeddings": int(
                args.rope_scaling_orig_max if args.rope_scaling_orig_max is not None else 32768
            ),
        }
    llm = load_model(model_name, max_len, dtype, device, rope_scaling=rope_scaling)
    
    # 准备输出文件
    if args.output_file is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        pass_mode = f"pass@{args.pass_k}" if args.n_generate > 1 else "pass@1"
        args.output_file = f"aime2025_{args.attention_type}_{pass_mode}_results_{timestamp}.jsonl"
    
    output_path = os.path.join(PROJECT_ROOT, "results", args.output_file)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    output_file = open(output_path, 'w')
    
    # 打印配置
    print(colored("\n" + "="*80, 'cyan'))
    print(colored("AIME2025 测试配置", 'cyan', attrs=['bold']))
    print(colored("="*80, 'cyan'))
    print(f"模型: {model_name}")
    print(f"Attention: {args.attention_type}")
    print(f"数据集: AIME2025 ({len(data)} 个问题)")
    print(f"评估模式: pass@{args.pass_k} ({'Greedy' if args.temperature == 0 else 'Sampling'})")
    if args.n_generate > 1:
        print(colored(f"  每个问题生成 {args.n_generate} 个答案", 'yellow'))
        print(colored(f"  采样参数: temperature={args.temperature}, top_p={args.top_p}, top_k={args.top_k}", 'yellow'))
    print(f"输出文件: {output_path}")
    print(colored("="*80 + "\n", 'cyan'))
    
    # 测试开始
    results = []
    total_correct = 0  # pass@k 下有任意一个答案正确的问题数
    total_problems = 0
    total_generated = 0  # 总共生成的答案数
    
    for idx, sample in enumerate(tqdm(data, desc="Processing")):
        problem_id = sample.get('id', f"problem_{idx+1}")
        problem = sample['problem']
        ground_truth = sample['answer']
        
        print(colored(f"\n{'='*80}", 'cyan'))
        print(colored(f"问题 {idx+1}/{len(data)} | ID: {problem_id}", 'cyan'))
        print(f"\n问题: {problem[:200]}..." if len(problem) > 200 else f"\n问题: {problem}")
        print(f"正确答案: {ground_truth}")
        
        # 构建 prompt
        base_prompt = build_aime_prompt(problem)
        
        # 应用聊天格式
        if "deepseek" in model_name.lower() or "llama" in model_name.lower():
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": base_prompt}
            ]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        elif "qwen" in model_name.lower():
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": base_prompt}
            ]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            prompt = base_prompt
        
        # Tokenize
        inputs = tokenizer([prompt], return_tensors="pt", padding=True)
        input_ids = inputs.input_ids
        attention_masks = inputs.attention_mask
        
        input_len = input_ids.shape[1]
        print(colored(f"\nInput tokens: {input_len}", 'yellow'))
        
        # 生成配置
        config = generate_config(model_name, input_len, args.attention_type, args)
        
        # 🔥 pass@k 评估：生成多个答案
        all_generated_texts = []
        all_predicted_answers = []
        all_generation_times = []
        problem_is_correct = False  # 该问题是否至少有1个答案正确
        
        print(colored(f"\n生成 {args.n_generate} 个答案...", 'yellow'))
        
        for gen_idx in range(args.n_generate):
            try:
                # Reset recall history (if applicable)
                if args.attention_type in ["PolarANN", "SuCo", "PQ"]:
                    if hasattr(llm.layers[0], 'kv_cache') and hasattr(llm.layers[0].kv_cache, 'reset_recall_history'):
                        llm.layers[0].kv_cache.reset_recall_history()
                
                t_start = time.time()
                
                # 生成
                out = llm.generate(
                    attention_type=args.attention_type,
                    inputs_ids=input_ids.to(llm.layers[0].device),
                    attention_masks=attention_masks.to(llm.layers[0].device),
                    max_new_length=args.max_new_tokens,
                    attn_config=config,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k
                )
                
                t_end = time.time()
                
                # Decode 结果
                # NOTE: LLM.generate() may return either:
                #   - a plain list of token ids (older code path), or
                #   - a list of lists (batched), or
                #   - a (outputs_ids, timing_stats) tuple from the current
                #     LLM.inference() implementation.
                # Normalize all cases to a flat list of ints.
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
                
                if len(token_ids) > 0:
                    generated_text = tokenizer.decode(token_ids, skip_special_tokens=True).strip()
                else:
                    generated_text = ""
                
                # 提取答案
                predicted_answer = extract_answer_from_response(generated_text)
                
                # 判断正确性
                is_correct = compare_answers(predicted_answer, ground_truth)
                if is_correct:
                    problem_is_correct = True
                
                all_generated_texts.append(generated_text)
                all_predicted_answers.append(predicted_answer)
                all_generation_times.append(t_end - t_start)
                
                # 打印每次生成的结果
                print(colored(f"\n  [{gen_idx+1}/{args.n_generate}] 生成完成 ({t_end - t_start:.2f}s, {len(token_ids)} tokens)", 'cyan'))
                print(f"    预测答案: {predicted_answer}")
                print(f"    结果: {'✓ 正确' if is_correct else '✗ 错误'}")
                
                # 🔥 实时写入：每完成一次采样就写一条记录
                single_result = {
                    'id': problem_id,
                    'problem': problem,
                    'ground_truth': ground_truth,
                    'sample_index': gen_idx + 1,  # 第几次采样 (1-indexed)
                    'n_generate': args.n_generate,
                    'predicted_answer': predicted_answer,
                    'generated_text': generated_text,
                    'is_correct': is_correct,
                    'generation_time': t_end - t_start,
                    'num_tokens': len(token_ids),
                    'input_length': input_len,
                    'type': 'single_sample'  # 标记这是单次采样结果
                }
                output_file.write(json.dumps(single_result, ensure_ascii=False) + '\n')
                output_file.flush()
                
                # 🔥 清理生成过程中的临时变量
                del out, token_ids, generated_text

                # ---- early-exit pass@k: 答对就停, 避免无谓的 7×38912 token 重生成 ----
                if args.early_exit_on_correct and is_correct:
                    print(colored(
                        f"  [early-exit] 第 {gen_idx+1}/{args.n_generate} 次生成命中, 跳过余下 {args.n_generate - gen_idx - 1} 次",
                        'green', attrs=['bold'],
                    ))
                    break

            except Exception as e:
                print(colored(f"\n  [{gen_idx+1}/{args.n_generate}] 生成失败: {e}", 'red'))
                import traceback
                print(colored("\n完整错误追踪:", 'red'))
                traceback.print_exc()
                
                # 打印详细的错误信息
                import sys
                exc_type, exc_value, exc_traceback = sys.exc_info()
                print(colored("\n详细错误信息:", 'yellow'))
                for frame in traceback.extract_tb(exc_traceback):
                    print(f"  文件: {frame.filename}")
                    print(f"  行号: {frame.lineno}")
                    print(f"  函数: {frame.name}")
                    print(f"  代码: {frame.line}")
                    print()
                
                all_generated_texts.append("")
                all_predicted_answers.append(None)
                all_generation_times.append(0)
        
        # 统计结果
        if problem_is_correct:
            total_correct += 1
        total_problems += 1
        total_generated += args.n_generate
        
        # 计算统计信息
        correct_count = sum(1 for pred in all_predicted_answers if compare_answers(pred, ground_truth))
        avg_time = sum(all_generation_times) / len(all_generation_times) if all_generation_times else 0
        
        # 打印问题汇总
        print(colored(f"\n{'='*80}", 'yellow'))
        print(colored(f"问题 {idx+1}/{len(data)} 完成", 'yellow', attrs=['bold']))
        print(colored(f"{'='*80}", 'yellow'))
        print(f"\n正确答案: {ground_truth}")
        print(f"生成的答案: {all_predicted_answers}")
        print(f"正确数量: {correct_count}/{args.n_generate}")
        print(colored(f"pass@{args.pass_k} 结果: {'✓ PASS' if problem_is_correct else '✗ FAIL'}", 
                     'green' if problem_is_correct else 'red'))
        print(f"平均生成时间: {avg_time:.2f}s")
        print(colored(f"\n当前 pass@{args.pass_k} 准确率: {total_correct}/{total_problems} = {100*total_correct/total_problems:.2f}%", 'cyan'))
        print(colored(f"{'='*80}\n", 'yellow'))
        
        # 保存结果（汇总）
        result = {
            'id': problem_id,
            'problem': problem,
            'ground_truth': ground_truth,
            'n_generate': args.n_generate,
            'all_predicted_answers': all_predicted_answers,
            'all_generated_texts': all_generated_texts,
            'all_generation_times': all_generation_times,
            'correct_count': correct_count,
            'pass_at_k': problem_is_correct,
            'input_length': input_len,
            'avg_generation_time': avg_time,
            'type': 'summary'  # 标记这是汇总结果
        }
        results.append(result)
        
        # 写入汇总结果
        output_file.write(json.dumps(result, ensure_ascii=False) + '\n')
        output_file.flush()
        
        # 🔥 清理 KV Cache 和 GPU 内存，避免 OOM
        if hasattr(llm, 'kv_cache'):
            del llm.kv_cache
            llm.kv_cache = None
        # 清理每一层的 kv_cache
        for layer in llm.layers:
            if hasattr(layer, 'kv_cache'):
                del layer.kv_cache
                layer.kv_cache = None
        torch.cuda.empty_cache()
        print(colored("[Debug] KV Cache cleared after problem", 'green'))
    
    # 关闭输出文件
    output_file.close()
    
    # 最终统计
    print(colored("\n" + "="*80, 'green'))
    print(colored("测试完成！", 'green', attrs=['bold']))
    print(colored("="*80, 'green'))
    print(f"\n总问题数: {total_problems}")
    print(f"总生成答案数: {total_generated}")
    print(f"pass@{args.pass_k} 通过数: {total_correct}")
    if total_problems > 0:
        print(colored(f"\npass@{args.pass_k} 准确率: {100*total_correct/total_problems:.2f}%", 'cyan', attrs=['bold']))
    
    # 计算 avg@k: 所有样本的平均准确率
    if results:
        avg_correct_per_problem = sum(r['correct_count'] for r in results) / len(results)
        avg_at_k = avg_correct_per_problem / args.n_generate
        print(f"平均每题正确答案数: {avg_correct_per_problem:.2f}/{args.n_generate}")
        print(colored(f"avg@{args.pass_k} (平均准确率): {100*avg_at_k:.2f}%", 'cyan', attrs=['bold']))
    
    print(f"\n结果已保存到: {output_path}")
    print(colored("="*80 + "\n", 'green'))


if __name__ == "__main__":
    main()

