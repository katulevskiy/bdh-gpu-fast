"""BDH custom kernels (attention + RoPE; Triton, blocked PyTorch, optional CUDA)."""

from .attention_dispatch import (
    DEFAULT_ATTN_AUTO_THRESHOLD,
    attn_auto_enabled,
    attn_auto_threshold,
    backend_info,
    bdh_attn,
    bdh_attn_decode,
    resolve_attn_impl,
    resolve_decode_impl,
)
from .rope_dispatch import (
    backend_info as rope_backend_info,
    bdh_rope_rotate,
    resolve_rope_impl,
)
from .attention_bwd import (
    StrictTrilAttnFn,
    StrictTrilSelfAttnFn,
    analytic_tril_attn_backward,
    analytic_tril_attn_backward_blocked,
    strict_tril_attn,
)
from .cuda_attn import (
    CUDA_TILE_M,
    CUDA_TILE_N,
    ext_status,
    has_cuda_ext,
    has_cuda_kernel,
    tril_decode,
    tril_decode_ref,
    tril_decode_tiled_ref,
    tril_score_v,
    tril_score_v_ref,
    tril_score_v_tiled_ref,
)

__all__ = [
    "bdh_attn",
    "bdh_attn_decode",
    "resolve_attn_impl",
    "resolve_decode_impl",
    "attn_auto_threshold",
    "attn_auto_enabled",
    "DEFAULT_ATTN_AUTO_THRESHOLD",
    "backend_info",
    "strict_tril_attn",
    "StrictTrilAttnFn",
    "StrictTrilSelfAttnFn",
    "analytic_tril_attn_backward",
    "analytic_tril_attn_backward_blocked",
    "tril_score_v",
    "tril_score_v_ref",
    "tril_score_v_tiled_ref",
    "tril_decode",
    "tril_decode_ref",
    "tril_decode_tiled_ref",
    "has_cuda_ext",
    "has_cuda_kernel",
    "ext_status",
    "CUDA_TILE_M",
    "CUDA_TILE_N",
    "bdh_rope_rotate",
    "resolve_rope_impl",
    "rope_backend_info",
]
