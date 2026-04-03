"""
Flash Attention Compatibility Layer

Cross-GPU architecture support:
- Ampere+ (compute capability >= 8.0): Flash Attention 2.x
- Turing (compute capability 7.5): Triton-based Flash Attention fallback
"""

import os
import torch

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('TRITON_CACHE_DIR', os.path.join(_project_root, '.triton', 'cache'))


def get_compute_capability():
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        return major + minor / 10
    return 0.0


def check_flash_attn_support():
    compute_cap = get_compute_capability()
    return compute_cap >= 8.0


_FLASH_ATTN_SUPPORTED = check_flash_attn_support()
_DISABLE_TRITON_FALLBACK = os.environ.get('DISABLE_TRITON_FALLBACK', 'false').lower() in ('true', '1', 'yes')


def flash_attn_with_kvcache_compat(
    q,
    k_cache,
    v_cache,
    causal=False,
    **kwargs
):
    """
    Flash Attention compatibility wrapper.

    Automatically selects the best implementation:
    - GPU >= Ampere: Flash Attention 2.x (native)
    - GPU < Ampere: Triton-based fallback (Turing compatible)

    Args:
        q: Query tensor [batch, seqlen_q, num_heads, head_dim]
        k_cache: Key cache [batch, seqlen_k, num_heads, head_dim]
        v_cache: Value cache [batch, seqlen_k, num_heads, head_dim]
        causal: Whether to apply causal mask
        **kwargs: Additional arguments passed to flash_attn_with_kvcache

    Returns:
        out: Attention output [batch, seqlen_q, num_heads, head_dim]
    """
    if _FLASH_ATTN_SUPPORTED:
        try:
            from flash_attn import flash_attn_with_kvcache
            return flash_attn_with_kvcache(
                q=q,
                k_cache=k_cache,
                v_cache=v_cache,
                causal=causal,
                **kwargs
            )
        except Exception as e:
            print(f"[Flash Attention] Warning: Flash Attention 2.x failed: {e}")
            if _DISABLE_TRITON_FALLBACK:
                raise RuntimeError(
                    f"Flash Attention 2.x failed and Triton fallback is disabled. "
                    f"Error: {e}. "
                    f"To enable Triton fallback, unset DISABLE_TRITON_FALLBACK environment variable."
                )
            return _triton_attention_fallback(q, k_cache, v_cache, causal)
    else:
        if _DISABLE_TRITON_FALLBACK:
            raise RuntimeError(
                f"GPU compute capability {get_compute_capability()} < 8.0 does not support Flash Attention 2.x, "
                f"and Triton fallback is disabled. "
                f"To enable Triton fallback, unset DISABLE_TRITON_FALLBACK environment variable."
            )
        return _triton_attention_fallback(q, k_cache, v_cache, causal)


def _triton_attention_fallback(q, k, v, causal=False):
    """
    Triton Flash Attention fallback for GPUs without Flash Attention 2.x support.
    Turing GPUs do not support BFloat16; automatic conversion to Float16 is applied.

    Args:
        q: [batch, seqlen_q, num_heads, head_dim]
        k: [batch, seqlen_k, num_heads, head_dim]
        v: [batch, seqlen_k, num_heads, head_dim]
        causal: Whether to apply causal mask

    Returns:
        out: [batch, seqlen_q, num_heads, head_dim]
    """
    try:
        original_dtype = q.dtype
        need_conversion = (original_dtype == torch.bfloat16)
        
        if need_conversion:
            q = q.to(torch.float16)
            k = k.to(torch.float16)
            v = v.to(torch.float16)
        
        from attn_hub.triton_flash_attn import triton_flash_attention
        out = triton_flash_attention(q, k, v, causal=causal)
        
        if need_conversion:
            out = out.to(original_dtype)
        
        return out
    except Exception as e:
        print(f"[Flash Attention] Triton implementation failed: {e}")
        import traceback
        traceback.print_exc()
        raise RuntimeError(
            f"Triton Flash Attention failed. GPU compute capability: {get_compute_capability()}. "
            f"Please check Triton installation or GPU compatibility."
        )


compute_cap = get_compute_capability()
if compute_cap > 0:
    if _FLASH_ATTN_SUPPORTED:
        print(f"[Flash Attention Compat] GPU CC={compute_cap}, using Flash Attention 2.x")
    else:
        if _DISABLE_TRITON_FALLBACK:
            print(f"[Flash Attention Compat] GPU CC={compute_cap} < 8.0, Triton fallback DISABLED")
        else:
            print(f"[Flash Attention Compat] GPU CC={compute_cap} < 8.0, using Triton fallback")
