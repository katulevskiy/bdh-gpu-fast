"""Env-gated dispatch for BDH attention kernels.

Wire-up (already applied thinly in bdh.py Attention.forward cold path)::

    export BDH_ATTN_IMPL=triton   # blocked PyTorch on CPU; Triton on CUDA
    export BDH_ATTN_IMPL=eager    # default — full T×T then tril_(diagonal=-1)
    export BDH_ATTN_IMPL=blocked  # tiled pure-PyTorch, no full upper triangle

If bdh.py is contended in another branch, import from here instead::

    from kernels.attention_dispatch import bdh_attn
    out = bdh_attn(QR, QR, V)
"""

from __future__ import annotations

import os
from typing import Literal

import torch

from .attention import (
    _HAS_TRITON,
    _can_use_triton,
    blocked_tril_attn,
    eager_tril_attn,
    triton_tril_attn,
)

ImplName = Literal["eager", "triton", "blocked"]


def resolve_attn_impl(requested: str | None = None) -> ImplName:
    """Resolve BDH_ATTN_IMPL (or explicit override) to a concrete backend."""
    raw = (requested if requested is not None else os.environ.get("BDH_ATTN_IMPL", "eager"))
    raw = (raw or "eager").strip().lower()
    if raw not in ("eager", "triton", "blocked"):
        raise ValueError(
            f"BDH_ATTN_IMPL must be eager|triton|blocked, got {raw!r}"
        )
    return raw  # type: ignore[return-value]


def bdh_attn(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    *,
    impl: str | None = None,
) -> torch.Tensor:
    """Compute tril(Q @ K.T, diagonal=-1) @ V with the selected backend.

    - eager:   materialize full scores (reference; matches original bdh.py)
    - blocked: tiled pure PyTorch (no full upper triangle; CPU/CUDA)
    - triton:  Triton fused kernel on CUDA; else same as blocked
    """
    name = resolve_attn_impl(impl)
    if name == "eager":
        return eager_tril_attn(Q, K, V)
    if name == "blocked":
        return blocked_tril_attn(Q, K, V)
    # triton
    if _can_use_triton(Q):
        return triton_tril_attn(Q, K, V)
    # Honest CPU / no-CUDA path for BDH_ATTN_IMPL=triton
    return blocked_tril_attn(Q, K, V)


def backend_info() -> dict:
    """Diagnostics for benchmarks / OPT_NOTES."""
    impl = resolve_attn_impl()
    cuda = torch.cuda.is_available()
    return {
        "BDH_ATTN_IMPL": impl,
        "has_triton": _HAS_TRITON,
        "cuda": cuda,
        "effective": (
            "triton_kernel"
            if impl == "triton" and cuda and _HAS_TRITON
            else ("blocked" if impl in ("triton", "blocked") else "eager")
        ),
    }
