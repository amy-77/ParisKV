"""
Fast Top-K kernels for KV cache collision counts

Two implementations:
1. bucket_topk: Optimized for integer values in small range [0, 127]
   - Uses counting sort approach: O(n + range)
   - Perfect for collision counts [0, 96]
   
2. radix_topk: General-purpose float values
   - Uses radix select: O(n)
   - Better for arbitrary float distributions
"""

import os
import importlib.util
from functools import lru_cache
from typing import Tuple

import torch
from torch.utils.cpp_extension import load


@lru_cache()
def load_radix_topk_ext(verbose: bool = False):
    """
    Build & load the topk CUDA extension once per process.
    """
    this_dir = os.path.dirname(os.path.abspath(__file__))
    build_dir = os.path.join(this_dir, "build")
    module_path = os.path.join(build_dir, "radix_topk_ext.so")
    source_path = os.path.join(this_dir, "radix_topk.cu")
    os.makedirs(build_dir, exist_ok=True)

    if os.path.exists(module_path):
        if os.path.getmtime(module_path) >= os.path.getmtime(source_path):
            spec = importlib.util.spec_from_file_location("radix_topk_ext", module_path)
            if spec is not None and spec.loader is not None:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                return module
    
    return load(
        name="radix_topk_ext",
        sources=[source_path],
        extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo"],
        build_directory=build_dir,
        verbose=verbose or (os.environ.get("RADIX_TOPK_VERBOSE", "0") == "1"),
    )


def bucket_topk(
    input: torch.Tensor,
    k: int
) -> torch.Tensor:
    """
    Bucket-based Top-K for integer values in [0, 127]
    
    O(n + range) complexity - optimal for small integer ranges like collision counts.
    
    Args:
        input: [B, L] or [bs, kv_heads, kv_len] int32 tensor
        k: number of top elements to return
        
    Returns:
        indices: tensor of int32 indices
    """
    assert input.is_cuda, "input must be on CUDA"
    assert k > 0, f"k must be positive, got {k}"
    
    ext = load_radix_topk_ext()
    
    if input.dim() == 2:
        return ext.bucket_topk(input, k)
    elif input.dim() == 3:
        return ext.bucket_topk_3d(input, k)
    else:
        raise ValueError(f"input must be 2D or 3D, got {input.dim()}D")


def radix_topk(
    input: torch.Tensor,
    k: int
) -> torch.Tensor:
    """
    Radix Select based Top-K
    
    O(n) complexity vs O(n log k) for comparison-based topk.
    Best for float values or large integer ranges.
    
    Args:
        input: [B, L] or [bs, kv_heads, kv_len] tensor
        k: number of top elements to return
        
    Returns:
        indices: tensor of int32 indices
    """
    assert input.is_cuda, "input must be on CUDA"
    assert k > 0, f"k must be positive, got {k}"
    
    ext = load_radix_topk_ext()
    
    if input.dim() == 2:
        return ext.radix_topk(input, k)
    elif input.dim() == 3:
        return ext.radix_topk_3d(input, k)
    else:
        raise ValueError(f"input must be 2D or 3D, got {input.dim()}D")


def fast_topk(
    input: torch.Tensor,
    k: int
) -> torch.Tensor:
    """
    Auto-select the best topk method based on input type.
    
    - For int32 inputs: uses bucket_topk (optimized for collision counts)
    - For float inputs: uses radix_topk
    
    Args:
        input: [B, L] or [bs, kv_heads, kv_len] tensor
        k: number of top elements to return
        
    Returns:
        indices: tensor of int32 indices
    """
    if input.dtype in (torch.int32, torch.int64, torch.int16, torch.int8):
        return bucket_topk(input.int(), k)
    else:
        return radix_topk(input, k)


def fast_topk_with_ratio(
    input: torch.Tensor,
    ratio: float
) -> torch.Tensor:
    """
    Fast Top-K with ratio
    
    Args:
        input: [B, L] or [bs, kv_heads, kv_len] tensor
        ratio: percentage of elements to keep (e.g., 0.05 for 5%)
        
    Returns:
        indices: tensor of top-k indices
    """
    if input.dim() == 2:
        L = input.size(1)
    elif input.dim() == 3:
        L = input.size(2)
    else:
        raise ValueError(f"input must be 2D or 3D, got {input.dim()}D")
    
    k = max(1, int(L * ratio))
    return fast_topk(input, k)


# Benchmark function
def benchmark_topk(kv_len: int = 128000, k: int = 6400, num_runs: int = 100):
    """
    Benchmark different topk implementations
    """
    import time
    
    # Test with collision count style data (int in [0, 96])
    cache_cnt = torch.randint(0, 97, (1, 8, kv_len), device='cuda', dtype=torch.int32)
    
    # Warmup
    print("Warming up...")
    for _ in range(10):
        _ = torch.topk(cache_cnt.float(), k=k, dim=-1)
        _ = bucket_topk(cache_cnt, k=k)
        _ = radix_topk(cache_cnt, k=k)
    torch.cuda.synchronize()
    
    # Benchmark torch.topk
    print(f"\nBenchmarking (kv_len={kv_len}, k={k}, {num_runs} runs)...")
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(num_runs):
        _ = torch.topk(cache_cnt.float(), k=k, dim=-1)
    torch.cuda.synchronize()
    torch_time = (time.perf_counter() - start) / num_runs * 1000
    
    # Benchmark bucket_topk
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(num_runs):
        _ = bucket_topk(cache_cnt, k=k)
    torch.cuda.synchronize()
    bucket_time = (time.perf_counter() - start) / num_runs * 1000
    
    # Benchmark radix_topk
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(num_runs):
        _ = radix_topk(cache_cnt, k=k)
    torch.cuda.synchronize()
    radix_time = (time.perf_counter() - start) / num_runs * 1000
    
    print(f"\nResults:")
    print(f"  torch.topk:   {torch_time:.3f} ms")
    print(f"  bucket_topk:  {bucket_time:.3f} ms  ({torch_time / bucket_time:.2f}x speedup)")
    print(f"  radix_topk:   {radix_time:.3f} ms  ({torch_time / radix_time:.2f}x speedup)")
    
    # Verify correctness
    print("\nVerifying correctness...")
    torch_vals, torch_idx = torch.topk(cache_cnt.float(), k=k, dim=-1)
    bucket_idx = bucket_topk(cache_cnt, k=k)
    radix_idx = radix_topk(cache_cnt, k=k)
    
    # Get values at indices
    bucket_vals = torch.gather(cache_cnt.float(), 2, bucket_idx.long())
    radix_vals = torch.gather(cache_cnt.float(), 2, radix_idx.long())
    
    # Check if top-k values match (indices may differ for ties)
    torch_sum = torch_vals.sum().item()
    bucket_sum = bucket_vals.sum().item()
    radix_sum = radix_vals.sum().item()
    
    print(f"\n  Value sums (should be equal):")
    print(f"    torch:  {torch_sum:.0f}")
    print(f"    bucket: {bucket_sum:.0f}  (diff: {torch_sum - bucket_sum:.0f})")
    print(f"    radix:  {radix_sum:.0f}  (diff: {torch_sum - radix_sum:.0f})")
    
    if abs(torch_sum - bucket_sum) < 1:
        print("  ✓ bucket_topk correct!")
    else:
        print("  ✗ bucket_topk mismatch!")
        
    if abs(torch_sum - radix_sum) < 1:
        print("  ✓ radix_topk correct!")
    else:
        print("  ✗ radix_topk mismatch!")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "benchmark":
        benchmark_topk()
    else:
        # Quick test
        print("Quick test...")
        x = torch.randint(0, 97, (1, 8, 10000), device='cuda', dtype=torch.int32)
        
        print(f"Input shape: {x.shape}")
        
        bucket_idx = bucket_topk(x, k=500)
        print(f"bucket_topk output shape: {bucket_idx.shape}")
        
        radix_idx = radix_topk(x, k=500)
        print(f"radix_topk output shape: {radix_idx.shape}")
        
        print("Done!")
