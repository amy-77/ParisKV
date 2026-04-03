import argparse
import sys
import os
import torch
import numpy as np
import math

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
try:
    from .collison_interface import update_cache_cnt_cuda_interface
except ImportError:
    from cache_hub.collision_fused.collison_interface import update_cache_cnt_cuda_interface


def generate_test_input(
    bs=1,
    kv_heads=8,
    B=16,
    num_clusters=256,
    kv_len=2048,
    collision_ratio=0.2,
    tier_ratios=None,
):
    torch.manual_seed(42)
    np.random.seed(42)

    sorted_cluster_ids = torch.stack([
        torch.stack([
            torch.stack([
                torch.randperm(num_clusters) for _ in range(B)
            ]) for _ in range(kv_heads)
        ]) for _ in range(bs)
    ]).to(torch.int32)

    cluster_key_counts = torch.randint(
        1, max(2, kv_len // 2),
        (bs, kv_heads, B, num_clusters),
        dtype=torch.int32,
    )

    codebook = torch.randint(0, num_clusters, (bs, kv_heads, B, kv_len), dtype=torch.int32)

    if tier_ratios is None:
        tier_ratios = [0.05, 0.15, 0.30, 0.50, 0.75, 1.0]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sorted_cluster_ids = sorted_cluster_ids.to(device)
    cluster_key_counts = cluster_key_counts.to(device)
    codebook = codebook.to(device)

    return (sorted_cluster_ids, cluster_key_counts, codebook,
            tier_ratios, kv_len, collision_ratio)


def update_cache_cnt_overall_ref(
    sorted_cluster_ids,
    cluster_key_counts,
    codebook,
    tier_ratios,
    kv_len,
    collision_ratio,
):
    """Reference (CPU) implementation of multi-tier collision counting."""
    sorted_key_counts = torch.gather(cluster_key_counts, dim=-1, index=sorted_cluster_ids.long())
    cumsum_counts = torch.cumsum(sorted_key_counts, dim=-1)
    before_cumsum = cumsum_counts - sorted_key_counts

    cache_cnt = None
    for ratio in tier_ratios:
        target_keys = math.ceil(kv_len * collision_ratio * ratio)
        valid_mask = before_cumsum < target_keys
        topk_cluster_cnt = valid_mask.sum(dim=-1)
        tier_cnt = update_cache_cnt_ref(sorted_cluster_ids, topk_cluster_cnt, codebook)
        cache_cnt = tier_cnt if cache_cnt is None else cache_cnt + tier_cnt

    return cache_cnt


def update_cache_cnt_ref(sorted_cluster_ids, topk_cluster_cnt, codebook):
    """Reference: count collisions per key across all subspaces for a single tier."""
    bs, kv_heads, B, kv_len = codebook.shape
    num_clusters = sorted_cluster_ids.shape[-1]

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


def update_cache_cnt_cuda(
    sorted_cluster_ids,
    cluster_key_counts,
    codebook,
    tier_ratios,
    kv_len,
    collision_ratio,
):
    tier_nums = [math.ceil(kv_len * i * collision_ratio) for i in tier_ratios]
    return update_cache_cnt_cuda_interface(
        sorted_cluster_ids,
        cluster_key_counts,
        codebook,
        tier_nums,
    )


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bs", type=int, default=1)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--B", type=int, default=16)
    parser.add_argument("--num-clusters", type=int, default=256)
    parser.add_argument("--kv-len", type=int, default=2048)
    parser.add_argument("--collision-ratio", type=float, default=0.2)
    parser.add_argument("--skip-ref", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    import triton

    args = _parse_args()
    (sorted_cluster_ids, cluster_key_counts, codebook,
     tier_ratios, kv_len, collision_ratio) = generate_test_input(
        bs=args.bs,
        kv_heads=args.kv_heads,
        B=args.B,
        num_clusters=args.num_clusters,
        kv_len=args.kv_len,
        collision_ratio=args.collision_ratio,
    )

    def ref():
        return update_cache_cnt_overall_ref(
            sorted_cluster_ids, cluster_key_counts, codebook,
            tier_ratios, kv_len, collision_ratio,
        )

    codebook = codebook.to(torch.uint8)
    sorted_cluster_ids = sorted_cluster_ids.to(torch.int64)

    def cuda():
        return update_cache_cnt_cuda(
            sorted_cluster_ids, cluster_key_counts, codebook,
            tier_ratios, kv_len, collision_ratio,
        )

    if not args.skip_ref:
        print("[ref] running ...", flush=True)
        result_ref = ref()
        print("[ref] done", flush=True)

    print("[cuda] running ...", flush=True)
    result_cuda = cuda()
    print("[cuda] done", flush=True)

    if not args.skip_ref:
        assert (result_ref - result_cuda).sum() == 0, "Mismatch between ref and CUDA"

    running_time = triton.testing.do_bench(cuda)
    print(f"CUDA kernel time: {running_time:.3f} ms")
