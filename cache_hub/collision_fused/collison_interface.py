import os
import importlib.util
from functools import lru_cache
from typing import Any, Iterable

import torch

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


def update_cache_cnt_cuda_interface(
    sorted_cluster_ids,
    cluster_key_counts,
    codebook,
    tier_counts,
) -> torch.Tensor:
    """Update per-key collision counts from cluster assignments.

    Args:
        sorted_cluster_ids: [bs, kv_heads, B, num_clusters]
        cluster_key_counts: per-cluster key counts (kernel input)
        codebook: [bs, kv_heads, B, kv_len] cluster id per key per subspace
        tier_counts: per-tier counts (kernel input)
    Returns:
        cache_cnt: [bs, kv_heads, kv_len] collision count summed over subspaces
    """
    bs, kv_heads, B, kv_len = codebook.shape
    cache_cnt = torch.empty(bs, kv_heads, kv_len, dtype=torch.int32, device=codebook.device)
    tier_bits_counts = load_kernel_module("collision.cu", "update_cache_cnt").update_cache_cnt(
        sorted_cluster_ids,
        cluster_key_counts,
        codebook,
        tier_counts,
        cache_cnt,
    )

    return cache_cnt
