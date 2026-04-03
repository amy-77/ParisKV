from typing import Any, Iterable

import numpy as np
import torch
import triton

import os
from functools import lru_cache


def generate_test_input(bs, kv_heads, B, num_clusters, kv_len):
    torch.manual_seed(42)
    np.random.seed(42)

    sorted_cluster_ids = torch.stack(
        [
            torch.stack(
                [torch.stack([torch.randperm(num_clusters) for _ in range(B)]) for _ in range(kv_heads)]
            )
            for _ in range(bs)
        ]
    ).to(torch.int32)

    topk_cluster_cnt = torch.randint(1, num_clusters, (bs, kv_heads, B), dtype=torch.int32)
    codebook = torch.randint(0, num_clusters, (bs, kv_heads, B, kv_len), dtype=torch.int32)
    return sorted_cluster_ids.cuda(), topk_cluster_cnt.cuda(), codebook.cuda()


def update_cache_cnt_ref(sorted_cluster_ids, topk_cluster_cnt, codebook):
    """Reference collision counts: count how often each key's cluster falls in the top-k per subspace."""
    bs, kv_heads, B, kv_len = codebook.shape
    num_clusters = sorted_cluster_ids.shape[-1]
    cache_cnt = torch.zeros(bs, kv_heads, kv_len, dtype=torch.int32, device=codebook.device)
    cluster_range = torch.arange(num_clusters, device=codebook.device)
    cluster_range_expanded = cluster_range.view(1, 1, 1, -1).expand(bs, kv_heads, B, -1)
    topk_cnt_expanded = topk_cluster_cnt.unsqueeze(-1)
    valid_cluster_mask = cluster_range_expanded < topk_cnt_expanded
    selected_clusters = sorted_cluster_ids.clone()
    selected_clusters[~valid_cluster_mask] = -1
    codebook_expanded = codebook.unsqueeze(-1)
    selected_clusters_expanded = selected_clusters.unsqueeze(-2)
    matches = codebook_expanded == selected_clusters_expanded
    collisions_per_subspace = matches.sum(dim=-1).int()
    cache_cnt = collisions_per_subspace.sum(dim=2)
    return cache_cnt


@lru_cache()
def _prepare_for_load() -> str:
    import warnings

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
    return load(
        name=name,
        sources=[f"{abs_path}/{p}" for p in path],
        extra_cflags=list(cflags or []) or ["-O3", "-std=c++17"],
        extra_cuda_cflags=list(cuda_flags or []) or ["-O3", "-std=c++17"],
        extra_ldflags=list(ldflags or []) or None,
        build_directory=build_dir,
        verbose=True,
    )


def update_cache_cnt_cuda(sorted_cluster_ids, topk_cluster_cnt, codebook) -> torch.Tensor:
    """CUDA path: same semantics as `update_cache_cnt_ref` after transposing codebook layout for the kernel."""
    codebook = codebook.transpose(-1, -2).contiguous()
    bs, kv_heads, kv_len, B = codebook.shape
    cache_cnt = torch.empty(bs, kv_heads, kv_len, dtype=torch.int32, device=codebook.device)
    load_kernel_module("collision.cu", "update_cache_cnt").update_cache_cnt(
        sorted_cluster_ids,
        topk_cluster_cnt,
        codebook,
        cache_cnt,
    )
    return cache_cnt


if __name__ == "__main__":
    batchsize = 1
    kv_heads = 8
    B = 16
    num_clusters = 256
    kv_len = 32 * 1024 + 3
    test_accuracy = False

    print("Testing...")
    print("Performance comparison:\n")
    print("batchsize kv_len subspace cluster_num ref_perf cuda_perf")
    for B, num_clusters in [(16, 256)]:
        sorted_cluster_ids, topk_cluster_cnt, codebook = generate_test_input(
            batchsize, kv_heads, B, num_clusters, kv_len
        )

        if test_accuracy:
            cnt_cuda = update_cache_cnt_cuda(sorted_cluster_ids, topk_cluster_cnt, codebook)

        for bs in [1]:
            for kv_len in [i for i in [128 * 1024]]:
                print(f"{bs} {kv_len} {B} {num_clusters} ", end="")
                sorted_cluster_ids, topk_cluster_cnt, codebook = generate_test_input(
                    batchsize, kv_heads, B, num_clusters, kv_len
                )

                def cuda():
                    update_cache_cnt_cuda(sorted_cluster_ids, topk_cluster_cnt, codebook)

                with torch.cuda.nvtx.range("collision"):
                    cauda_perf = triton.testing.do_bench(cuda)

                print(f"{cauda_perf:.2f}")
