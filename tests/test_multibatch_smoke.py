"""
Smoke test for multi-batch (bs=8) PolarANN cache; assumes equal-length lockstep batch.
"""

import os
import sys

import torch

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _REPO_ROOT)

from cache_hub.polar_cache import polar_cache

_DEFAULT_CODEBOOK = os.path.join(
    _REPO_ROOT,
    "turboquant",
    "codebooks",
    "codebook_d128_m8_Kr1_Kw256_rabitq_sign.json",
)


def test_multibatch_smoke():
    batch_size = 8
    layer_num = 28
    kv_head = 4
    num_heads = 32
    head_dim = 128
    prefill_len = 20480
    decode_steps = 50
    max_new_length = decode_steps + 100
    max_length = prefill_len + max_new_length
    device = "cuda:0"
    dtype = torch.bfloat16

    codebook_path = os.environ.get("PARISKV_CODEBOOK_PATH", _DEFAULT_CODEBOOK)
    if not os.path.isfile(codebook_path):
        raise FileNotFoundError(
            f"Missing codebook at {codebook_path}. Set PARISKV_CODEBOOK_PATH or add the file under turboquant/codebooks/."
        )

    print("=== Multi-batch smoke ===")
    print(f"  batch_size={batch_size} prefill_len={prefill_len} decode_steps={decode_steps}")
    print(f"  layers={layer_num} kv_heads={kv_head} head_dim={head_dim}")

    valid_start = [0] * batch_size

    cache = polar_cache(
        valid_start=valid_start,
        layer_num=layer_num,
        batch_size=batch_size,
        max_length=max_length,
        num_key_value_heads=kv_head,
        num_heads=num_heads,
        head_dim=head_dim,
        dtype=dtype,
        layer_mapping={i: i for i in range(layer_num)},
        max_new_length=max_new_length,
        sink_size=128,
        local_size=256,
        core=1,
        nprobe=1,
        cache_unit_size=1,
        cache_cluster_num=1,
        num_gpus=1,
        model_size=7,
        codebook_path=codebook_path,
        final_topk=100,
        dynamic_update_interval=128,
        enable_offload=True,
        enable_rerank=False,
        enable_recall=False,
    )

    print("\n[1] Prefill...")
    for layer_idx in range(layer_num):
        key_states = torch.randn(
            batch_size, prefill_len, kv_head, head_dim, device=device, dtype=dtype
        )
        value_states = torch.randn(
            batch_size, prefill_len, kv_head, head_dim, device=device, dtype=dtype
        )
        query_states = torch.randn(
            batch_size, prefill_len, kv_head * 8, head_dim, device=device, dtype=dtype
        )

        k_out, v_out = cache.prefill_update_kv_cache(
            query_states, key_states, value_states, layer_idx, batch_idx=0
        )

        if layer_idx == 0:
            print(f"  layer0 k_out={k_out.shape} v_out={v_out.shape}")

    print(f"  prefill done current_seq_len[0]={cache.current_seq_len[0]}")

    print("\n[2] Decode...")
    for step in range(decode_steps):
        for layer_idx in range(layer_num):
            key_states = torch.randn(batch_size, 1, kv_head, head_dim, device=device, dtype=dtype)
            value_states = torch.randn(batch_size, 1, kv_head, head_dim, device=device, dtype=dtype)
            query_states = torch.randn(
                batch_size, 1, num_heads, head_dim, device=device, dtype=dtype
            )

            cache.decode_update_kv_cache(key_states, value_states, layer_idx)
            attn_out = cache.compute(query_states, layer_idx)

        if step % 10 == 0 or step == decode_steps - 1:
            print(
                f"  step {step}: current_seq_len[0]={cache.current_seq_len[0]} "
                f"attn_out={attn_out.shape}"
            )

    print("\n=== PASSED ===")


if __name__ == "__main__":
    test_multibatch_smoke()
