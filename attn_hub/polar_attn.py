"""
PolarANN Attention Functions
"""

from attn_hub.flash_attn_compat import flash_attn_with_kvcache_compat as flash_attn_with_kvcache


def polar_prefill_attn(query_states, key_states, value_states, causal):
    """
    Prefill attention using standard Flash Attention.

    Args:
        query_states: [batch_size, seq_len, num_heads, head_dim]
        key_states: [batch_size, seq_len, num_kv_heads, head_dim]
        value_states: [batch_size, seq_len, num_kv_heads, head_dim]
        causal: whether to apply causal mask

    Returns:
        attn_out: [batch_size, seq_len, num_heads, head_dim]
    """
    attn_out = flash_attn_with_kvcache(
        q=query_states, 
        k_cache=key_states, 
        v_cache=value_states,
        causal=causal
    )
    
    return attn_out


def polar_decode_attn(query_states, key_states, value_states, layer_idx, kv_cache):
    """
    Decode attention using Polar ANN collision-based retrieval:
    1. Count collisions between query and keys across subspaces
    2. Select keys with highest collision counts as candidates
    3. Compute exact similarity on candidates to get final top-k tokens
    4. Run Flash Attention on steady zone + top-k retrieved tokens

    Args:
        query_states: [batch_size, 1, num_heads, head_dim]
        key_states: unused (kept for interface compatibility)
        value_states: unused (kept for interface compatibility)
        layer_idx: layer index
        kv_cache: polar_cache instance

    Returns:
        attn_out: [batch_size, 1, num_heads, head_dim]
    """
    if kv_cache is None or not hasattr(kv_cache, "compute") or not callable(getattr(kv_cache, "compute")):
        got_type = type(kv_cache).__name__
        got_module = type(kv_cache).__module__
        raise TypeError(
            "polar_decode_attn expects kv_cache to provide a callable compute(query_states, layer_idx), "
            f"but got {got_module}.{got_type}. "
            "Please ensure attention_type='PolarANN' and cache initialization are correctly configured."
        )
    
    attn_out = kv_cache.compute(
        query_states.contiguous(), layer_idx
    )
    
    return attn_out
