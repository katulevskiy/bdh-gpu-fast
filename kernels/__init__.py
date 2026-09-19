"""BDH custom kernels (attention + RoPE; Triton, blocked PyTorch, optional CUDA)."""

from .attention_dispatch import backend_info, bdh_attn, bdh_attn_decode, resolve_attn_impl
from .rope_dispatch import (
    backend_info as rope_backend_info,
    bdh_rope_rotate,
    resolve_rope_impl,
)
from .attention_bwd import StrictTrilAttnFn, analytic_tril_attn_backward, strict_tril_attn
from .cuda_attn import (
    has_cuda_ext,
    has_cuda_kernel,
    tril_decode,
    tril_decode_ref,
    tril_score_v,
    tril_score_v_ref,
)

__all__ = [
    "bdh_attn",
    "bdh_attn_decode",
    "resolve_attn_impl",
    "backend_info",
    "strict_tril_attn",
    "StrictTrilAttnFn",
    "analytic_tril_attn_backward",
    "tril_score_v",
    "tril_score_v_ref",
    "tril_decode",
    "tril_decode_ref",
    "has_cuda_ext",
    "has_cuda_kernel",
    "bdh_rope_rotate",
    "resolve_rope_impl",
    "rope_backend_info",
]
