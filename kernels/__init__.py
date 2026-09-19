"""BDH custom attention kernels (Triton, blocked PyTorch, optional CUDA ext)."""

from .attention_dispatch import backend_info, bdh_attn, bdh_attn_decode, resolve_attn_impl
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
]
