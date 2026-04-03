from typing import Any

import torch
import numpy as np
import triton

import os
import importlib.util
from functools import lru_cache
from typing import Any, Iterable

@lru_cache()
def _prepare_for_load() -> str:
    import os
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


def h2d_gather_kv(
    src_k_caches, # bs, heads, kv_len, dim
    src_v_caches, # bs, heads, kv_len, dim
    selected_indices, # bs, heads, max_topk
    dst_k_caches, # bs, heads, topk, dim
    dst_v_caches, # bs, heads, topk, dim
) -> torch.Tensor:
    '''
    '''
    # check shape
    bs, heads, kv_len, dim = src_k_caches.shape
    assert src_k_caches.shape == src_v_caches.shape
    _, _, topk = selected_indices.shape
    assert dst_k_caches.shape == dst_v_caches.shape
    assert dst_k_caches.shape[-2] >= topk
    assert dim == 128
    # check device
    assert src_k_caches.is_cpu
    assert src_v_caches.is_cpu
    assert src_k_caches.is_pinned()
    assert src_v_caches.is_pinned()
    assert dst_k_caches.is_cuda
    assert dst_v_caches.is_cuda
    # check type 
    assert src_k_caches.dtype == src_v_caches.dtype
    assert dst_k_caches.dtype == dst_v_caches.dtype
    assert src_k_caches.dtype == dst_k_caches.dtype
    assert src_k_caches.dtype == torch.bfloat16
    # check continguous
    assert src_k_caches.is_contiguous()
    assert dst_k_caches.is_contiguous()
    assert src_v_caches.is_contiguous()
    assert dst_v_caches.is_contiguous()
    # run
    load_kernel_module("trans_h2d.cu", "h2d_gather_kv").h2d_gather_kv(
        src_k_caches,
        src_v_caches,
        selected_indices,
        dst_k_caches,
        dst_v_caches,
    )


if __name__ == "__main__":
    bs = 2
    heads = 8
    kv_len = 32 * 1024
    dim = 128
    topk = 100
    device = "cuda:0"
    max_topk = int(kv_len * 0.2)

    src_k_caches = torch.randn(bs, heads, kv_len, dim, dtype = torch.bfloat16, device = "cpu", pin_memory=True)
    src_v_caches = torch.randn(bs, heads, kv_len, dim, dtype = torch.bfloat16, device = "cpu", pin_memory=True)

    selected_indices = torch.randint(low=0, high=kv_len, size=(bs, heads, topk), dtype=torch.long).to(device)

    # accuracy
    def cuda():
        dst_k_caches = torch.zeros(bs, heads, max_topk, dim, dtype = torch.bfloat16, device = device)
        dst_v_caches = torch.zeros(bs, heads, max_topk, dim, dtype = torch.bfloat16, device = device)
        h2d_gather_kv(src_k_caches, src_v_caches, selected_indices, dst_k_caches, dst_v_caches)
        return dst_k_caches, dst_v_caches
    
    def ref():
        # Use different variable names to avoid shadowing
        src_k_cuda = src_k_caches.cuda()
        src_v_cuda = src_v_caches.cuda()
        dst_k_ref = torch.zeros(bs, heads, max_topk, dim, dtype = torch.bfloat16, device = device)
        dst_v_ref = torch.zeros(bs, heads, max_topk, dim, dtype = torch.bfloat16, device = device)
        for i in range(bs):
            for j in range(heads):
                dst_k_ref[i][j][:topk].copy_(src_k_cuda[i][j][selected_indices[i][j]])
                dst_v_ref[i][j][:topk].copy_(src_v_cuda[i][j][selected_indices[i][j]])
        return dst_k_ref, dst_v_ref

    dst_k_ref, dst_v_ref = ref()
    dst_k_cuda, dst_v_cuda = cuda()
    assert (dst_k_ref - dst_k_cuda).sum() == 0, "dst_k_ref and dst_k_cuda are not equal"
    assert (dst_v_ref - dst_v_cuda).sum() == 0, "dst_v_ref and dst_v_cuda are not equal"
    # print(dst_k_ref - dst_k_cuda)
    # assert dst_k_ref == dst_k_cuda, "dst_k_ref and dst_k_cuda are not equal"
    # assert dst_v_ref == dst_v_cuda, "dst_v_ref and dst_v_cuda are not equal"
