import torch
from attn_hub.flash_attn_compat import flash_attn_with_kvcache_compat as flash_attn_with_kvcache


def prefill_full_flash_attn(query_states, key_states, value_states, causal):
    """
    Prefill forward using flash attention with KV cache.
    For GQA, repeats K/V to match the number of Q heads.
    """
    bsz, seq_len, num_q_heads, head_dim = query_states.shape
    _, _, num_kv_heads, _ = key_states.shape
    
    if num_kv_heads != num_q_heads:
        num_key_value_groups = num_q_heads // num_kv_heads
        key_states = key_states.repeat_interleave(num_key_value_groups, dim=2)
        value_states = value_states.repeat_interleave(num_key_value_groups, dim=2)
    
    attn_out = flash_attn_with_kvcache(
        q=query_states, 
        k_cache=key_states, 
        v_cache=value_states, 
        causal=causal
    )
    
    return attn_out



def decode_full_flash_attn(query_states, key_states, value_states, layer_idx, full_attn_cache):
    """
    Decode forward using flash attention with KV cache.
    flash_attn_with_kvcache handles GQA automatically during decode.
    """
    valid_len = full_attn_cache.valid_length
    
    attn_out = flash_attn_with_kvcache(
        q=query_states, 
        k_cache=key_states, 
        v_cache=value_states, 
        cache_seqlens=valid_len,
    )
    
    return attn_out
