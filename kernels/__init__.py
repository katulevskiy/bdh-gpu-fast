"""BDH custom attention kernels (Triton, blocked PyTorch, optional CUDA ext)."""

<<<<<<< HEAD
from .attention_dispatch import bdh_attn, resolve_attn_impl
from .attention_bwd import StrictTrilAttnFn, analytic_tril_attn_backward, strict_tril_attn
from .cuda_attn import tril_score_v, tril_score_v_ref, has_cuda_ext, has_cuda_kernel
=======
from .attention_dispatch import backend_info, bdh_attn, resolve_attn_impl
from .cuda_attn import has_cuda_ext, has_cuda_kernel, tril_score_v, tril_score_v_ref
>>>>>>> f30de27 (opt/attn-unify: single BDH_ATTN_IMPL dispatch (eager|blocked|triton|cuda))

__all__ = [
    "bdh_attn",
    "resolve_attn_impl",
<<<<<<< HEAD
    "strict_tril_attn",
    "StrictTrilAttnFn",
    "analytic_tril_attn_backward",
=======
    "backend_info",
>>>>>>> f30de27 (opt/attn-unify: single BDH_ATTN_IMPL dispatch (eager|blocked|triton|cuda))
    "tril_score_v",
    "tril_score_v_ref",
    "has_cuda_ext",
    "has_cuda_kernel",
]
