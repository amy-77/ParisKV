"""
Adaptive Top-k Selection (value-weighted) — CUDA Kernel Interface
被融合的源函数：
    cache_hub/polar_cache_accurate_attn_value_adaptive_k_value_indices.py
        ::PolarCacheAccurateAttnValueAdaptiveKValueIndices._compute_adaptive_topk

------------------------------------------------------------------
功能:
在每个 decode step / layer 中，retrieval 候选集已通过 RaBitQ rerank 拿到了
近似 QK 内积分数 (`cand_scores`) 与候选位置 (`cand_indices`)。
现在需要把这些候选按"对最终 attention 输出的贡献"重新评估，再按累积贡献占比
阈值 τ 自适应地挑出每个 (batch, kv_head) 真正需要的 top-K：

    benefit_i = exp(qk_i - max_qk) * ||v_i||₂
    sort by benefit desc, take prefix until prefix_sum >= τ * total_benefit
    K_(b,h)  = 该 prefix 长度（每个 head 独立）
    global_K = max_(b,h) K_(b,h)        ← dense tensor 第 3 维

输出的 topk_indices 直接喂给下游 sparse attention / h2d_gather_kv (UVA 拷贝)。

------------------------------------------------------------------
接口约定（Tensor 全部 GPU resident, 全部 contiguous 除非显式说明）
输入：
    query_avg        : [bs, kv_heads, head_dim]    bfloat16
                       group 内 query 在 head 维度求均值后的代理 query。
                       含义：用它和 sink/local 的真 key 计算精确 QK 内积。

    sink_local_keys  : [bs, kv_heads, sl_len, head_dim]  bfloat16
                       sink + local + update_buffer 的真实 key cache。
                       **sl_len 不是常量**： sl_len = sink_size + local_size + update_buffer_len = 64 + 256 + update_buffer_len
                       update_buffer_len 从 0 单调递增到 dynamic_update_interval (=512)，
                       到顶后整个 update_buffer 被 flush 进 retrieval zone 并清零，下一步又从 0 重新增长。所以 sl_len 在 [320, 832] 之间反复振荡，

    sl_vnorm         : [bs, kv_heads, sl_len]      float32（来自增量缓存）
                       sink_local 段对应的 ||v||₂，已在外部预先维护好。

    cand_scores      : [bs, kv_heads, cand_len]    bfloat16
                       retrieval zone 候选的 RaBitQ 近似 QK 内积，
                       注意：**未除以 sqrt(head_dim)**，kernel 内部要乘 attn_scale。
                       **cand_len 也不是常量**：cand_len ≈ retrieval_zone_len × candidate_ratio。

    cand_indices     : [bs, kv_heads, cand_len]    int64
                       候选在 retrieval zone 中的相对索引（0-based, < retrieval_zone_len）。
                       这个就是最终输出的 topk_indices 的来源（要按 benefit 顺序 gather 出来）。

    cand_vnorm       : [bs, kv_heads, cand_len]    float32
                       和 cand_indices 一一对应的候选 ||v||₂。
                       已经由调用方按 (cand_indices + sink_size) 在 value_norm 上 gather 过，

    attn_scale       : float (Python scalar, e.g. 1.0 / sqrt(head_dim))
    threshold        : float (Python scalar, e.g. 0.85)
    topk_cap         : int   (Python scalar, e.g. 2048)
                       每个 head 的 adaptive_k 上限，dense 输出的第 3 维。
                       

输出：
    adaptive_k       : [bs, kv_heads]              int64
                       每个 (b, h) 真正需要的 K，范围 [1, min(topk_cap, cand_len)]。

    topk_indices     : [bs, kv_heads, K_out]       int64, contiguous
                       K_out = min(topk_cap, cand_len)
                       前 adaptive_k[b, h] 项是 cand_indices 中按 benefit 降序的 top-K，
                       即 retrieval zone 相对索引；后面 padding 内容 unspecified
                       （调用方只 slice 到 [:, :, :global_k]）。
                       该 tensor 必须 long & contiguous，下游 h2d_gather_kv 强校验。

调用方：
    global_k = int(adaptive_k.max().item())   # 1 次 CPU sync
    topk_indices = topk_indices[:, :, :global_k]   # view, 0 cost

------------------------------------------------------------------
数值/边界约定
------------------------------------------------------------------
1. fp32 内部精度
   - exp 前必须做 logits - max(logits) 数值稳定化（max 对 sl 与 cand 一起取）。
   - benefit / cumsum / total 全部 fp32 累加，否则长上下文 fp16 累加会丢小项。

2. threshold 判定
   sl_benefit + cumsum_topk[k-1] >= threshold * (sl_benefit + sum_all_cand)
   注意分母 total = sl_benefit + sum(全部 cand)，**不是** sum(top-cap)
   （否则 threshold 永远在 cap 处刚好达成，K 总等于 cap）。

3. fall-back
   - 若没有任何前缀达到阈值（threshold * total），返回 K = min(topk_cap, cand_len)。
   - 若 total <= 0（理论上不会发生，数值噪声情况下可能），返回 K = 1。

4. 对每个 (b, h) 独立处理；batch 维 + head 维可完全并行，
   block 划分推荐：grid = (bs * kv_heads,) 或 (bs, kv_heads)。

5. tie-break：当多个候选 benefit 相等时，结果可与 ref 不一致（torch.topk
   的 tie-break 也未定义），只要 K 值与 ref 相等、所选索引集合等价（同 benefit
   的可互换）即可视为正确。

------------------------------------------------------------------
形状量级（用于 kernel 调参）
------------------------------------------------------------------
    bs        ∈ {1, 2, 4}            常量    （单卡推理常见 1）
    kv_heads  ∈ {4, 8, 16}           常量    （Qwen3-4B: kv_heads=8, GQA=4; 7B/14B: 8）
    head_dim  ∈ {64, 128}            常量    （默认 128）
    sl_len    ∈ [320, 832]           **每步都变** = 64 + 256 + update_buffer_len(0..512)
                                              update_buffer 满 512 → flush + 清零 → 重新从 0 涨
    cand_len  ∈ [5000, 15000]        **每步都变** ≈ retrieval_zone_len × candidate_ratio
                                              retrieval_zone_len 在每次 flush 后阶跃 +512
    topk_cap  = 2048                 常量    （adaptive_k 的硬上限，dense 输出第 3 维）

    → kernel 接口必须接受 runtime shape：grid 维按 (bs, kv_heads) 划分，
      sl_len / cand_len 仅作为 kernel 参数（int），不要做编译期 template 化。
"""

from __future__ import annotations

import math
import os
import warnings
import importlib.util
from functools import lru_cache
from typing import Any, Iterable, Tuple

import torch


# ============================================================================
# CUDA kernel loader（与 rerank / trans_h2d 同一套约定）
# ============================================================================
@lru_cache()
def _prepare_for_load() -> str:
    warnings.filterwarnings(
        "ignore", category=UserWarning, module="torch.utils.cpp_extension"
    )
    return os.path.dirname(os.path.abspath(__file__))


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


# ============================================================================
# 测试输入生成
# ============================================================================
def generate_test_input(
    bs: int = 1,
    kv_heads: int = 8,
    head_dim: int = 128,
    sl_len: int = 320,           # sink(64) + local(256) + update_buffer_len(0..512) ∈ [320, 832]
    cand_len: int = 12500,       # 130k ctx 下典型 retrieval 候选数 ∈ [5000, 15000]
    topk_cap: int = 2048,        # 固定 = adaptive_topk_max_k
    threshold: float = 0.95,     # 固定
    seed: int = 42,
    device: str = "cuda:0",
):
    """
    构造与生产环境同分布的合成数据，用于算子单元测试。

    重要：sl_len / cand_len 在生产里 **不是固定值**，每个 decode step 都可能不同：
        sl_len    = sink(64) + local(256) + update_buffer_len(0..512)
                    update_buffer 满 512 后 flush 进 retrieval zone 并清零，下一步又从 0 涨
        cand_len ≈ retrieval_zone_len × candidate_ratio
                    retrieval_zone_len 在每次 flush 后阶跃增长 +512，所以 cand_len 也跟着变
    确保 kernel 不依赖编译期固定 shape。

    Returns:
        dict with all kernel inputs + (topk_cap, threshold).
        cand_indices 用 0..cand_len 的 random permutation 模拟"在 retrieval zone
        里位置随机分布"的情形。
    """
    torch.manual_seed(seed)
    dev = torch.device(device)

    query_avg = torch.randn(bs, kv_heads, head_dim, dtype=torch.bfloat16, device=dev) * 0.1
    sink_local_keys = torch.randn(bs, kv_heads, sl_len, head_dim, dtype=torch.bfloat16, device=dev) * 0.1
    sl_vnorm = torch.rand(bs, kv_heads, sl_len, dtype=torch.float32, device=dev) + 0.5

    # cand_scores 是 RaBitQ 近似的 <q, k>，未除 sqrt(d)。
    # 真实 attention 分布：post-scale 大多数 logits 集中在 0 附近（softmax 后接近 0），
    # 少数 head-specific 尖峰落在 5~10 之间支配 attention mass。
    # 这里用低噪声（post-scale σ ≈ 0.3）+ 每个 head 不同数量的高尖峰，让每个 head 的 K 各异。
    base_std = math.sqrt(head_dim)
    cand_scores = torch.randn(bs, kv_heads, cand_len, dtype=torch.float32, device=dev) * (0.3 * base_std)
    rng = torch.Generator(device='cpu').manual_seed(seed)
    n_heavy_per_head = torch.randint(16, 1025, (bs, kv_heads), generator=rng).tolist()
    boost_scale = 5.0 * base_std  # post-scale ≈ 5，相对于噪声 0.3 完全支配
    for b in range(bs):
        for h in range(kv_heads):
            n_h = n_heavy_per_head[b][h]
            cand_scores[b, h, :n_h] += torch.rand(n_h, device=dev) * boost_scale + (3.0 * base_std)
    cand_scores = cand_scores.to(torch.bfloat16)

    # cand_indices: 每个 (b, h) 是 0..cand_len 的随机 perm 截断
    cand_indices = torch.stack([
        torch.stack([
            torch.randperm(cand_len, device=dev, dtype=torch.int64)
            for _ in range(kv_heads)
        ])
        for _ in range(bs)
    ])

    cand_vnorm = torch.rand(bs, kv_heads, cand_len, dtype=torch.float32, device=dev) + 0.5

    attn_scale = 1.0 / math.sqrt(head_dim)

    return {
        "query_avg":       query_avg,
        "sink_local_keys": sink_local_keys,
        "sl_vnorm":        sl_vnorm,
        "cand_scores":     cand_scores,
        "cand_indices":    cand_indices,
        "cand_vnorm":      cand_vnorm,
        "attn_scale":      attn_scale,
        "threshold":       threshold,
        "topk_cap":        topk_cap,
    }


# ============================================================================
# PyTorch 参考实现（数值真值；CUDA kernel 需对齐这个）
# ============================================================================
def adaptive_topk_ref(
    query_avg:        torch.Tensor,   # [bs, kv_heads, head_dim]    bf16
    sink_local_keys:  torch.Tensor,   # [bs, kv_heads, sl_len, head_dim]  bf16
    sl_vnorm:         torch.Tensor,   # [bs, kv_heads, sl_len]      fp32
    cand_scores:      torch.Tensor,   # [bs, kv_heads, cand_len]    bf16  (raw <q,k>, 未乘 1/sqrt(d))
    cand_indices:     torch.Tensor,   # [bs, kv_heads, cand_len]    int64
    cand_vnorm:       torch.Tensor,   # [bs, kv_heads, cand_len]    fp32
    attn_scale:       float,          # 1.0 / sqrt(head_dim)
    threshold:        float,          # 0.95
    topk_cap:         int,            # 2048
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    PyTorch 参考实现。CUDA kernel 必须与该函数的数值结果一致（tie-break 除外）。

    Returns:
        adaptive_k:   [bs, kv_heads]                 int64
        topk_indices: [bs, kv_heads, K_out]          int64, contiguous
                      K_out = min(topk_cap, cand_len)
                      前 adaptive_k[b, h] 项有效，其余 padding。
    """
    bs, kv_heads, head_dim = query_avg.shape
    sl_len   = sink_local_keys.shape[2]
    cand_len = cand_scores.shape[-1]
    K_out    = min(topk_cap, cand_len)

    # 1. QK logits（fp32），sink_local 走精确 QK，cand 是 RaBitQ 近似分数
    logits_sl   = torch.einsum('bhd,bhkd->bhk', query_avg, sink_local_keys).to(torch.float32) * attn_scale
    logits_cand = cand_scores.to(torch.float32) * attn_scale

    # 2. 减全局 max（数值稳定；对相对大小无影响）
    joint_logits = torch.cat([logits_sl, logits_cand], dim=-1)              # [bs, kv, sl+cand]
    joint_max    = joint_logits.amax(dim=-1, keepdim=True)
    exp_sl       = torch.exp(logits_sl   - joint_max)                       # [bs, kv, sl]
    exp_cand     = torch.exp(logits_cand - joint_max)                       # [bs, kv, cand]

    # 3. benefit = exp * ||v||
    sl_benefit   = (exp_sl   * sl_vnorm  ).sum(dim=-1, keepdim=True)        # [bs, kv, 1]
    cand_benefit =  exp_cand * cand_vnorm                                   # [bs, kv, cand]

    # 4. 按 benefit 降序选 top-cap，cumsum 看哪一步累积达阈值
    cand_benefit_topk, sort_perm = torch.topk(
        cand_benefit, k=K_out, dim=-1, largest=True, sorted=True
    )                                                                       # [bs, kv, K_out]
    cumsum_topk = torch.cumsum(cand_benefit_topk, dim=-1)                   # [bs, kv, K_out]

    # 5. 阈值判定（注意分母用 cand_benefit 整段 sum，不是 cumsum_topk[..., -1]）
    total_benefit = sl_benefit + cand_benefit.sum(dim=-1, keepdim=True)     # [bs, kv, 1]
    reached = (sl_benefit + cumsum_topk >= threshold * total_benefit)       # [bs, kv, K_out]
    adaptive_k = reached.long().argmax(dim=-1) + 1                          # [bs, kv]
    never_reached = ~reached.any(dim=-1)
    adaptive_k[never_reached] = K_out
    bad_total = (total_benefit.squeeze(-1) <= 0)
    if bad_total.any():
        adaptive_k = torch.where(bad_total, torch.ones_like(adaptive_k), adaptive_k)

    # 6. 把 sort_perm（cand 内部相对位置）映射回 retrieval zone 相对位置
    topk_indices = torch.gather(cand_indices, dim=2, index=sort_perm)       # [bs, kv, K_out]

    return adaptive_k, topk_indices.contiguous()


# ============================================================================
# CUDA kernel 调用入口
# ============================================================================
#
# 设计原则：不刻意融合所有操作，只融合适合融合的部分。
#
#   PyTorch / cuBLAS 负责（它们做得更好的部分）:
#     - sl_logits  = einsum('bhd,bhkd->bhk', q, slk).float() * attn_scale
#       → cuBLAS tensor core matmul，比手写循环快一个数量级
#     - cand_logits = cand_scores.float() * attn_scale
#       → trivial element-wise，1 次 kernel launch
#
#   CUDA kernel 融合（适合融合：数据依赖链，拆开需 ~10 次 launch + 中间 tensor）:
#     global_max → exp → benefit → sum → radix_select → bitonic_sort → cumsum → adaptive_k → gather
#
def adaptive_topk_cuda(
    query_avg:        torch.Tensor,
    sink_local_keys:  torch.Tensor,
    sl_vnorm:         torch.Tensor,
    cand_scores:      torch.Tensor,
    cand_indices:     torch.Tensor,
    cand_vnorm:       torch.Tensor,
    attn_scale:       float,
    threshold:        float,
    topk_cap:         int,
    kernel: Any = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    bs, kv_heads, head_dim = query_avg.shape
    sl_len   = sink_local_keys.shape[2]
    cand_len = cand_scores.shape[-1]
    K_out    = min(topk_cap, cand_len)
    device   = query_avg.device

    # ===== 正确性验证 =====
    # dtype
    assert query_avg.dtype == torch.bfloat16,       f"query_avg dtype: expected bf16, got {query_avg.dtype}"
    assert sink_local_keys.dtype == torch.bfloat16,  f"sink_local_keys dtype: expected bf16, got {sink_local_keys.dtype}"
    assert sl_vnorm.dtype == torch.float32,          f"sl_vnorm dtype: expected fp32, got {sl_vnorm.dtype}"
    assert cand_scores.dtype == torch.bfloat16,      f"cand_scores dtype: expected bf16, got {cand_scores.dtype}"
    assert cand_indices.dtype == torch.int64,         f"cand_indices dtype: expected int64, got {cand_indices.dtype}"
    assert cand_vnorm.dtype == torch.float32,         f"cand_vnorm dtype: expected fp32, got {cand_vnorm.dtype}"
    # device & contiguous
    for name, t in [("query_avg", query_avg), ("sink_local_keys", sink_local_keys),
                    ("sl_vnorm", sl_vnorm), ("cand_scores", cand_scores),
                    ("cand_indices", cand_indices), ("cand_vnorm", cand_vnorm)]:
        assert t.is_cuda,       f"{name} must be on CUDA, got {t.device}"
        assert t.is_contiguous(), f"{name} must be contiguous"
    # shape
    assert sink_local_keys.shape == (bs, kv_heads, sl_len, head_dim), \
        f"sink_local_keys shape mismatch: {sink_local_keys.shape}"
    assert sl_vnorm.shape == (bs, kv_heads, sl_len), \
        f"sl_vnorm shape mismatch: {sl_vnorm.shape}"
    assert cand_scores.shape == (bs, kv_heads, cand_len), \
        f"cand_scores shape mismatch: {cand_scores.shape}"
    assert cand_indices.shape == (bs, kv_heads, cand_len), \
        f"cand_indices shape mismatch: {cand_indices.shape}"
    assert cand_vnorm.shape == (bs, kv_heads, cand_len), \
        f"cand_vnorm shape mismatch: {cand_vnorm.shape}"
    # scalar params
    assert 0.0 < threshold <= 1.0,  f"threshold must be in (0, 1], got {threshold}"
    assert topk_cap > 0,            f"topk_cap must be > 0, got {topk_cap}"
    assert attn_scale > 0,          f"attn_scale must be > 0, got {attn_scale}"

    # ===== Step 1: PyTorch matmul + scale（不融合进 CUDA kernel）=====
    # sl_logits: [bs, kv_heads, sl_len], fp32
    sl_logits = torch.einsum(
        'bhd,bhkd->bhk', query_avg, sink_local_keys
    ).to(torch.float32) * attn_scale

    # cand_logits: [bs, kv_heads, cand_len], fp32
    cand_logits = cand_scores.to(torch.float32) * attn_scale

    # ===== Step 2: fused CUDA kernel（适合融合的部分）=====
    adaptive_k   = torch.empty(bs, kv_heads,        dtype=torch.int64, device=device)
    topk_indices = torch.empty(bs, kv_heads, K_out, dtype=torch.int64, device=device)

    if kernel is None:
        kernel = load_kernel_module("adaptive_topk.cu", "adaptive_topk")
    kernel.adaptive_topk(
        sl_logits.contiguous(),
        cand_logits.contiguous(),
        sl_vnorm,
        cand_indices,
        cand_vnorm,
        float(threshold),
        int(K_out),
        adaptive_k,
        topk_indices,
    )
    return adaptive_k, topk_indices


# ============================================================================
# 单元测试 / 验收脚本 — ref vs CUDA 对比
# ============================================================================

def _call_ref(inp):
    return adaptive_topk_ref(
        inp["query_avg"], inp["sink_local_keys"], inp["sl_vnorm"],
        inp["cand_scores"], inp["cand_indices"], inp["cand_vnorm"],
        attn_scale=inp["attn_scale"],
        threshold=inp["threshold"],
        topk_cap=inp["topk_cap"],
    )


def _call_cuda(inp, kernel=None):
    return adaptive_topk_cuda(
        inp["query_avg"], inp["sink_local_keys"], inp["sl_vnorm"],
        inp["cand_scores"], inp["cand_indices"], inp["cand_vnorm"],
        attn_scale=inp["attn_scale"],
        threshold=inp["threshold"],
        topk_cap=inp["topk_cap"],
        kernel=kernel,
    )


def verify_correctness(ref_k, ref_idx, cuda_k, cuda_idx, label=""):
    """Compare CUDA results against reference.
    - adaptive_k must match exactly.
    - topk_indices: for each (b,h), the first adaptive_k[b,h] indices must form
      the same *set* (tie-break order may differ for equal-benefit items).
    """
    bs, kv_heads = ref_k.shape
    k_match = (ref_k == cuda_k).all().item()
    print(f"  [{label}] adaptive_k exact match : {'PASS' if k_match else 'FAIL'}")
    if not k_match:
        print(f"    ref  adaptive_k = {ref_k.cpu().tolist()}")
        print(f"    cuda adaptive_k = {cuda_k.cpu().tolist()}")
        print(f"    diff            = {(ref_k - cuda_k).cpu().tolist()}")

    # Check index sets — for each (b, h), the top adaptive_k indices should be
    # the same set (order may differ due to tie-breaking).
    idx_pass = True
    for b_i in range(bs):
        for h_i in range(kv_heads):
            ak = int(ref_k[b_i, h_i].item())
            ref_set = set(ref_idx[b_i, h_i, :ak].cpu().tolist())
            cuda_set = set(cuda_idx[b_i, h_i, :ak].cpu().tolist())
            if ref_set != cuda_set:
                idx_pass = False
                diff = ref_set.symmetric_difference(cuda_set)
                print(f"    (b={b_i}, h={h_i}) index set mismatch, "
                      f"symmetric_diff size={len(diff)}, ak={ak}")
    print(f"  [{label}] topk index set match   : {'PASS' if idx_pass else 'FAIL'}")
    return k_match and idx_pass


def benchmark(fn, inp, n_warmup=10, n_iter=100, label=""):
    """Benchmark a function with CUDA event timing."""
    import time
    torch.cuda.synchronize()
    # Warmup
    for _ in range(n_warmup):
        fn(inp)
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event   = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(n_iter):
        fn(inp)
    end_event.record()
    torch.cuda.synchronize()

    elapsed_ms = start_event.elapsed_time(end_event)
    avg_us = elapsed_ms * 1000.0 / n_iter
    print(f"  [{label}] avg latency = {avg_us:.1f} us  ({n_iter} iters)")
    return avg_us


if __name__ == "__main__":
    # ------------------------------------------------------------------
    # Test configurations covering different shapes
    # ------------------------------------------------------------------
    test_configs = [
        dict(bs=1, kv_heads=8,  head_dim=128, sl_len=320,  cand_len=12500, topk_cap=2048, threshold=0.95, seed=42),
        dict(bs=1, kv_heads=8,  head_dim=128, sl_len=832,  cand_len=5000,  topk_cap=2048, threshold=0.95, seed=7),
        dict(bs=2, kv_heads=8,  head_dim=128, sl_len=576,  cand_len=8000,  topk_cap=2048, threshold=0.85, seed=123),
        dict(bs=1, kv_heads=16, head_dim=128, sl_len=320,  cand_len=15000, topk_cap=2048, threshold=0.95, seed=99),
    ]

    # ------------------------------------------------------------------
    # 1. Reference only
    # ------------------------------------------------------------------
    print("=" * 72)
    print(" Phase 1: adaptive_topk_ref (PyTorch reference)")
    print("=" * 72)
    for i, cfg in enumerate(test_configs):
        inp = generate_test_input(**cfg)
        ref_k, ref_idx = _call_ref(inp)
        print(f"  config[{i}] adaptive_k = {ref_k.cpu().tolist()}")
        print(f"  config[{i}] global_k   = {int(ref_k.max().item())}")

    # ------------------------------------------------------------------
    # 2. Build & load CUDA kernel
    # ------------------------------------------------------------------
    print()
    print("=" * 72)
    print(" Phase 2: Building adaptive_topk CUDA kernel ...")
    print("=" * 72)
    kernel = load_kernel_module(
        "adaptive_topk.cu", "adaptive_topk",
        cuda_flags=("-O3", "-std=c++17", "--use_fast_math"),
    )
    print("  Kernel module loaded:", kernel)

    # ------------------------------------------------------------------
    # 3. Correctness verification
    # ------------------------------------------------------------------
    print()
    print("=" * 72)
    print(" Phase 3: Correctness — ref vs CUDA")
    print("=" * 72)
    all_pass = True
    for i, cfg in enumerate(test_configs):
        inp = generate_test_input(**cfg)
        ref_k, ref_idx = _call_ref(inp)
        cuda_k, cuda_idx = _call_cuda(inp, kernel=kernel)
        ok = verify_correctness(ref_k, ref_idx, cuda_k, cuda_idx, label=f"cfg{i}")
        all_pass = all_pass and ok

    print()
    if all_pass:
        print("  >>> ALL correctness checks PASSED <<<")
    else:
        print("  >>> SOME correctness checks FAILED <<<")

    # ------------------------------------------------------------------
    # 4. Performance comparison
    # ------------------------------------------------------------------
    print()
    print("=" * 72)
    print(" Phase 4: Performance comparison (ref vs CUDA)")
    print("=" * 72)
    perf_cfg = test_configs[0]  # typical config
    inp = generate_test_input(**perf_cfg)
    ref_us  = benchmark(_call_ref,  inp, n_warmup=5, n_iter=50, label="ref ")
    cuda_us = benchmark(lambda x: _call_cuda(x, kernel=kernel), inp,
                        n_warmup=10, n_iter=100, label="cuda")
    if cuda_us > 0:
        print(f"  Speedup: {ref_us / cuda_us:.2f}x")