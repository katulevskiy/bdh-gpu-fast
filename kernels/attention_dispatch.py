"""Env-gated dispatch for BDH attention kernels.

Wire-up (already applied thinly in bdh.py Attention.forward cold path)::

    export BDH_ATTN_IMPL=triton   # blocked PyTorch on CPU; Triton on CUDA
    export BDH_ATTN_IMPL=eager    # default — full T×T then tril_(diagonal=-1)
    export BDH_ATTN_IMPL=blocked  # tiled pure-PyTorch, no full upper triangle
    export BDH_ATTN_AUTOGRAD=1    # optional — StrictTrilAttnFn + analytic bwd

If bdh.py is contended in another branch, import from here instead::

    from kernels.attention_dispatch import bdh_attn
    out = bdh_attn(QR, QR, V)
"""

from __future__ import annotations

import os
from typing import Literal, Optional

import torch

from .attention import (
    _HAS_TRITON,
    _can_use_triton,
    blocked_tril_attn,
    eager_tril_attn,
    triton_tril_attn,
)
from .attention_bwd import _env_autograd_enabled, strict_tril_attn

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
    use_autograd_fn: Optional[bool] = None,
) -> torch.Tensor:
    """Compute tril(Q @ K.T, diagonal=-1) @ V with the selected backend.

    - eager:   materialize full scores (reference; matches original bdh.py)
    - blocked: tiled pure PyTorch (no full upper triangle; CPU/CUDA)
    - triton:  Triton fused kernel on CUDA; else same as blocked

    When ``use_autograd_fn`` is True (or BDH_ATTN_AUTOGRAD=1 and the arg is
    None), wraps the forward in ``StrictTrilAttnFn`` with analytic Q/K/V
    backward. Default is False / env-off so the eager training path is
    unchanged.
    """
    name = resolve_attn_impl(impl)
    if use_autograd_fn is None:
        use_autograd_fn = _env_autograd_enabled()
    if use_autograd_fn:
        return strict_tril_attn(Q, K, V, impl=name, use_fn=True)
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
        "BDH_ATTN_AUTOGRAD": _env_autograd_enabled(),
        "has_triton": _HAS_TRITON,
        "cuda": cuda,
        "effective": (
            "triton_kernel"
            if impl == "triton" and cuda and _HAS_TRITON
            else ("blocked" if impl in ("triton", "blocked") else "eager")
        ),
    }
