
"""
4-bit RaBitQ rerank reference and CUDA entry points.

Pipeline: coarse filter by collision count → gather 4-bit codes → unpack sign+magnitude
→ approximate inner product → top-k rerank.

Fused-kernel goal: combine gather/unpack/score to cut memory traffic (see profiling
notes in project docs if needed).
"""

import math
import os
import warnings
import importlib.util
from functools import lru_cache
from typing import Any, Iterable

import torch
import numpy as np


@lru_cache()
def _prepare_for_load() -> str:
    """Directory of this module (CUDA extension build path)."""
    warnings.filterwarnings(
        "ignore", category=UserWarning, module="torch.utils.cpp_extension"
    )
    return os.path.dirname(os.path.abspath(__file__))


def generate_test_input(
    bs: int = 1,
    kv_heads: int = 8,
    B: int = 16,
    m: int = 8,
    kv_len: int = 128 * 1024,
    candidate_ratio: float = 0.1,
    final_topk: int = 100,
    seed: int = 42,
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda:0")

    cache_cnt = torch.randint(
        0, B + 1, (bs, kv_heads, kv_len), dtype=torch.int32, device=device
    )

    sign = torch.randint(0, 2, (bs, kv_heads, B, kv_len, m), device=device)
    mag = torch.randint(0, 8, (bs, kv_heads, B, kv_len, m), device=device)
    v4bit = (sign << 3) | mag

    v_even = v4bit[..., 0::2]
    v_odd = v4bit[..., 1::2]
    key_4bit = ((v_odd << 4) | v_even).to(torch.uint8)

    key_block_weight = torch.randn(
        bs, kv_heads, B, kv_len, dtype=torch.bfloat16, device=device
    ).abs()

    q_blocks = torch.randn(bs, kv_heads, B, m, dtype=torch.bfloat16, device=device)

    q_norm = torch.randn(bs, kv_heads, 1, dtype=torch.bfloat16, device=device).abs() + 0.1

    mag_centers = torch.tensor(
        [0.0420, 0.1266, 0.2130, 0.3025, 0.3973, 0.5001, 0.6169, 0.7633],
        dtype=torch.bfloat16,
        device=device,
    )

    return {
        "cache_cnt": cache_cnt,
        "key_4bit": key_4bit,
        "key_block_weight": key_block_weight,
        "q_blocks": q_blocks,
        "q_norm": q_norm,
        "mag_centers": mag_centers,
        "candidate_ratio": candidate_ratio,
        "final_topk": final_topk,
        "B": B,
        "m": m,
    }


def _coarse_filter(cache_cnt, candidate_ratio, final_topk):
    kv_len = cache_cnt.shape[-1]
    candidate_len = max(1, math.ceil(kv_len * candidate_ratio))
    candidate_len = min(candidate_len, kv_len)
    candidate_len = max(candidate_len, final_topk)

    vals, idx = torch.topk(cache_cnt, k=candidate_len, dim=-1)
    thr = vals[..., -1]

    thr_count = (cache_cnt == thr.unsqueeze(-1)).sum(dim=-1)
    gt_count = (cache_cnt > thr.unsqueeze(-1)).sum(dim=-1)
    ge_count = gt_count + thr_count
    max_candidates = min(int(ge_count.max().item()), kv_len)

    if max_candidates > candidate_len:
        _, coarse_indices = torch.topk(cache_cnt, k=max_candidates, dim=-1)
    else:
        coarse_indices = idx
        max_candidates = candidate_len

    return coarse_indices, max_candidates


def _unpack_4bit_codes(packed: torch.Tensor, m: int = 8) -> tuple:
    v_even = packed & 0x0F
    v_odd = (packed >> 4) & 0x0F
    v4bit = torch.stack([v_even, v_odd], dim=-1).view(*packed.shape[:-1], m)
    sign_bits = (v4bit >> 3) & 1
    s_sign = (2 * sign_bits - 1).to(torch.int8)
    t_codes = (v4bit & 0x7).to(torch.int64)
    return s_sign, t_codes


def _coarse_filter_truncate(cache_cnt: torch.Tensor, candidate_ratio: float) -> tuple:
    kv_len = cache_cnt.shape[-1]
    candidate_len = max(1, math.ceil(kv_len * candidate_ratio))

    _, coarse_indices = torch.topk(cache_cnt, k=candidate_len, dim=-1)
    return coarse_indices, candidate_len


def rerank_rabitq_ref(
    cache_cnt: torch.Tensor,
    key_4bit: torch.Tensor,
    key_block_weight: torch.Tensor,
    q_blocks: torch.Tensor,
    q_norm: torch.Tensor,
    mag_centers: torch.Tensor,
    candidate_ratio: float,
    final_topk: int,
    B: int,
    m: int,
) -> torch.Tensor:
    coarse_indices, max_candidates = _coarse_filter(
        cache_cnt, candidate_ratio, final_topk
    )

    if max_candidates <= final_topk:
        return coarse_indices[:, :, :final_topk]

    idx_code = coarse_indices.unsqueeze(2).expand(-1, -1, B, -1)
    cand_weight = torch.gather(key_block_weight, dim=3, index=idx_code)
    idx_4bit = coarse_indices.unsqueeze(2).unsqueeze(-1).expand(
        -1, -1, B, -1, key_4bit.shape[-1]
    )
    cand_4bit = torch.gather(key_4bit, dim=3, index=idx_4bit)

    orig_shape = cand_4bit.shape[:-1]
    s_sign, t_codes = _unpack_4bit_codes(
        cand_4bit.view(-1, cand_4bit.shape[-1]), m=m
    )
    s_sign = s_sign.view(*orig_shape, m)
    t_codes = t_codes.view(*orig_shape, m)
    a_vals = mag_centers[t_codes]

    v4b = a_vals * s_sign.to(a_vals.dtype)
    q_exp = q_blocks.unsqueeze(3)
    per_block_scores = (q_exp * v4b).sum(dim=-1)
    cosine_proxy = (per_block_scores * cand_weight).sum(dim=2)
    approx_scores = cosine_proxy * q_norm

    _, rerank_indices = torch.topk(approx_scores, k=final_topk, dim=-1)
    final_indices = torch.gather(coarse_indices, dim=2, index=rerank_indices)
    return final_indices


@lru_cache()
def load_kernel_module(
    path: str | Iterable[str],
    name: str,
    *,
    build: str = "build",
    cflags: Iterable[str] | None = None,
    cuda_flags: Iterable[str] | None = None,
    ldflags: Iterable[str] | None = None,
) -> Any:
    from torch.utils.cpp_extension import load

    if isinstance(path, str):
        path = (path,)

    abs_path = _prepare_for_load()
    build_dir = f"{abs_path}/{build}"
    os.makedirs(build_dir, exist_ok=True)
    module_path = os.path.join(build_dir, f"{name}.so")
    source_paths = [f"{abs_path}/{p}" for p in path]

    if os.path.exists(module_path):
        module_mtime = os.path.getmtime(module_path)
        latest_source_mtime = max(os.path.getmtime(src) for src in source_paths)
        if module_mtime >= latest_source_mtime:
            spec = importlib.util.spec_from_file_location(name, module_path)
            if spec is not None and spec.loader is not None:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                return module

    return load(
        name=name,
        sources=source_paths,
        extra_cflags=list(cflags or []) or ["-O3", "-std=c++17"],
        extra_cuda_cflags=list(cuda_flags or []) or ["-O3", "-std=c++17"],
        extra_ldflags=list(ldflags or []) or None,
        build_directory=build_dir,
        verbose=True,
    )


def partial_rerank_cuda(
    coarse_indices: torch.Tensor,
    key_4bit: torch.Tensor,
    key_block_weight: torch.Tensor,
    q_blocks: torch.Tensor,
    q_norm: torch.Tensor,
    mag_centers: torch.Tensor,
    B: int,
    m: int,
    kernel=None,
) -> tuple:
    bs, kv_heads, _, kv_len, _ = key_4bit.shape
    candidate_len = coarse_indices.shape[-1]
    approx_scores = torch.zeros(
        bs, kv_heads, candidate_len, dtype=torch.bfloat16, device=coarse_indices.device
    )
    v4b = torch.zeros(
        bs, kv_heads, B, candidate_len, m, dtype=torch.bfloat16, device=coarse_indices.device
    )
    assert coarse_indices.dtype == torch.int64, (
        f"coarse_indices must be int64, got {coarse_indices.dtype}"
    )
    assert coarse_indices.is_contiguous()
    assert key_4bit.stride(-1) == 1
    assert key_block_weight.stride(-1) == 1
    assert q_blocks.is_contiguous()
    assert q_norm.is_contiguous()
    assert mag_centers.is_contiguous()
    assert coarse_indices.is_cuda
    assert key_4bit.is_cuda
    assert key_block_weight.is_cuda
    assert q_blocks.is_cuda
    assert q_norm.is_cuda
    assert mag_centers.is_cuda

    if kernel is None:
        kernel = load_kernel_module("rerank.cu", "rerank")
    kernel.partial_rerank(
        coarse_indices,
        key_4bit,
        key_block_weight,
        q_blocks,
        q_norm,
        mag_centers,
        B,
        m,
        approx_scores,
        v4b,
    )
    return approx_scores, v4b


def rerank_rabitq_fused(
    cache_cnt: torch.Tensor,
    key_4bit: torch.Tensor,
    key_block_weight: torch.Tensor,
    q_blocks: torch.Tensor,
    q_norm: torch.Tensor,
    mag_centers: torch.Tensor,
    candidate_ratio: float,
    final_topk: int,
    B: int,
    m: int,
) -> tuple:
    coarse_indices, max_candidates = _coarse_filter_truncate(cache_cnt, candidate_ratio)
    if max_candidates <= final_topk:
        return coarse_indices[:, :, :final_topk], None, None

    approx_scores, v4b = partial_rerank_cuda(
        coarse_indices,
        key_4bit,
        key_block_weight,
        q_blocks,
        q_norm,
        mag_centers,
        B,
        m,
    )
    _, rerank_indices = torch.topk(approx_scores, k=final_topk, dim=-1)
    final_indices = torch.gather(coarse_indices, dim=2, index=rerank_indices)
    return final_indices, approx_scores, v4b
