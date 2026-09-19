"""Unified env-gated dispatch for BDH attention backends.

Backends (``BDH_ATTN_IMPL``)::

<<<<<<< HEAD
    export BDH_ATTN_IMPL=triton   # blocked PyTorch on CPU; Triton on CUDA
    export BDH_ATTN_IMPL=eager    # default — full T×T then tril_(diagonal=-1)
    export BDH_ATTN_IMPL=blocked  # tiled pure-PyTorch, no full upper triangle
    export BDH_ATTN_AUTOGRAD=1    # optional — StrictTrilAttnFn + analytic bwd
=======
    eager    — full T×T scores then tril_(diagonal=-1)  [DEFAULT]
    blocked  — tiled pure PyTorch (no full upper triangle)
    triton   — Triton fused kernel on CUDA; blocked fallback otherwise
    cuda     — kernels.cuda_attn (native ext if built, else CPU/CUDA ref)
>>>>>>> f30de27 (opt/attn-unify: single BDH_ATTN_IMPL dispatch (eager|blocked|triton|cuda))

Wire-up in ``bdh.Attention.forward`` cold path (``past_kr is None``)::

    from kernels.attention_dispatch import bdh_attn
    out = bdh_attn(QR, QR, V)

Incremental / KV-cache path (``past_kr is not None``) always uses eager
PyTorch matmuls — custom kernels do not yet support incremental decode.
``generate()`` therefore ignores ``BDH_ATTN_IMPL`` after the cold prefill
step (prefill itself respects the env when cache starts empty).
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

ImplName = Literal["eager", "blocked", "triton", "cuda"]
_VALID = ("eager", "blocked", "triton", "cuda")


def resolve_attn_impl(requested: str | None = None) -> ImplName:
    """Resolve BDH_ATTN_IMPL (or explicit override) to a concrete backend name."""
    raw = requested if requested is not None else os.environ.get("BDH_ATTN_IMPL", "eager")
    raw = (raw or "eager").strip().lower()
    if raw not in _VALID:
        raise ValueError(
            f"BDH_ATTN_IMPL must be {'|'.join(_VALID)}, got {raw!r}"
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
<<<<<<< HEAD
    - triton:  Triton fused kernel on CUDA; else same as blocked

    When ``use_autograd_fn`` is True (or BDH_ATTN_AUTOGRAD=1 and the arg is
    None), wraps the forward in ``StrictTrilAttnFn`` with analytic Q/K/V
    backward. Default is False / env-off so the eager training path is
    unchanged.
=======
    - triton:  Triton fused kernel on CUDA; else blocked
    - cuda:    native ext via ``kernels.cuda_attn.tril_score_v`` when present,
               else that module's pure-PyTorch reference (same math)
>>>>>>> f30de27 (opt/attn-unify: single BDH_ATTN_IMPL dispatch (eager|blocked|triton|cuda))
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
    if name == "triton":
        if _can_use_triton(Q):
            return triton_tril_attn(Q, K, V)
        return blocked_tril_attn(Q, K, V)
    # cuda
    from .cuda_attn import tril_score_v

    return tril_score_v(Q, K, V)


def backend_info() -> dict:
    """Diagnostics for benchmarks / OPT_NOTES."""
    from .cuda_attn import has_cuda_ext, has_cuda_kernel

    impl = resolve_attn_impl()
    cuda_dev = torch.cuda.is_available()
    if impl == "eager":
        effective = "eager"
    elif impl == "blocked":
        effective = "blocked"
    elif impl == "triton":
        effective = (
            "triton_kernel"
            if cuda_dev and _HAS_TRITON
            else "blocked"
        )
    else:  # cuda
        if has_cuda_ext():
            effective = "cuda_ext" if (not cuda_dev or has_cuda_kernel()) else "cuda_ext_cpu"
        else:
            effective = "cuda_ref"
    return {
        "BDH_ATTN_IMPL": impl,
        "BDH_ATTN_AUTOGRAD": _env_autograd_enabled(),
        "has_triton": _HAS_TRITON,
        "has_cuda_ext": has_cuda_ext(),
        "has_cuda_kernel": has_cuda_kernel(),
        "cuda": cuda_dev,
        "effective": effective,
    }
