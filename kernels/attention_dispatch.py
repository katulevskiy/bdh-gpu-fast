"""Unified env-gated dispatch for BDH attention backends.

Backends (``BDH_ATTN_IMPL``)::

    export BDH_ATTN_IMPL=triton   # blocked PyTorch on CPU; Triton on CUDA
    export BDH_ATTN_IMPL=eager    # default — full T×T then tril_(diagonal=-1)
    export BDH_ATTN_IMPL=blocked  # online fused tiles, no full T×T scores
    export BDH_ATTN_IMPL=cuda    # native ext if built, else CPU/CUDA ref
    export BDH_ATTN_AUTOGRAD=1    # optional — StrictTrilAttnFn + analytic bwd

Backends: eager, blocked, triton, or cuda.

Wire-up in ``bdh.Attention.forward``:

- Cold path (``past_kr is None``): ``bdh_attn`` respects ``BDH_ATTN_IMPL``.
- T=1 decode (packed past KR/V): ``bdh_attn_decode`` — eager two-GEMM;
  blocked/triton/cuda use their decode paths (cuda → ``kernels.cuda_attn.tril_decode``).
"""

from __future__ import annotations

import os
from typing import Literal, Optional

import torch

from .attention import (
    _HAS_TRITON,
    _can_use_triton,
    DEFAULT_BLOCK_DECODE,
    blocked_decode_attn,
    blocked_tril_attn,
    eager_decode_attn,
    eager_tril_attn,
    triton_decode_attn,
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
    - blocked: online fused tiles (no full T×T scores; CPU/CUDA)
    - triton:  Triton fused kernel on CUDA; else blocked
    - cuda:    native ext via ``kernels.cuda_attn.tril_score_v`` when present,
               else that module's pure-PyTorch reference (same math)

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
    if name == "triton":
        if _can_use_triton(Q):
            return triton_tril_attn(Q, K, V)
        return blocked_tril_attn(Q, K, V)
    # cuda
    from .cuda_attn import tril_score_v

    return tril_score_v(Q, K, V)



def bdh_attn_decode(
    Q: torch.Tensor,
    K_past: torch.Tensor,
    V_past: torch.Tensor,
    *,
    impl: str | None = None,
    block_size: int = DEFAULT_BLOCK_DECODE,
) -> torch.Tensor:
    """Attend new queries to packed past KR/V only (no full TxT).

    Used by ``Attention.forward`` incremental decode (Tq typically 1).
    Preserves tril(diagonal=-1): past slices exclude the new token, so the
    query never attends to itself.

    - eager:   single ``(Q @ K.mT) @ V`` (reference)
    - blocked: tiled over past (shared ``_tiled_score_v``; larger default tile)
    - triton:  fused decode kernel on CUDA; blocked fallback on CPU
    - cuda:    ``kernels.cuda_attn.tril_decode`` (native ext if built, else ref)
    """
    name = resolve_attn_impl(impl)
    if name == "eager":
        return eager_decode_attn(Q, K_past, V_past)
    if name == "blocked":
        return blocked_decode_attn(Q, K_past, V_past, block_size=block_size)
    if name == "cuda":
        from .cuda_attn import tril_decode

        return tril_decode(Q, K_past, V_past)
    # triton → fused decode on CUDA; blocked (_tiled_score_v) otherwise
    return triton_decode_attn(Q, K_past, V_past, block_size=block_size)


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
