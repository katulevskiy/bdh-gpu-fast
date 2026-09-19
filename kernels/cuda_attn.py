"""Strict lower-triangular score×V for BDH attention.

Semantics (must match bdh.Attention cold path):

    scores = (Q @ K.transpose(-2, -1)).tril(diagonal=-1)
    out    = scores @ V

No softmax, no 1/sqrt(d) scale, diagonal EXCLUDED.

Decode (packed past KR/V — matches incremental / CacheManager hot path)::

    out = (Q @ K_past.transpose(-2, -1)) @ V_past

Past slices exclude the new token, so the query never attends to itself
(same as ``tril(diagonal=-1)``).

Public API
----------
tril_score_v_ref(q, k, v)       — pure PyTorch full tril score×V (always)
tril_score_v(q, k, v)           — native ext if built+compatible, else ref
tril_decode_ref(q, k_past, v)   — pure PyTorch decode vs packed past (always)
tril_decode(q, k_past, v)       — native ext if built, else ref

Optional native build (``csrc/``)::

    pip install -e .                                 # pure Python (default)
    BDH_BUILD_EXT=1 pip install -e . --no-build-isolation
    BDH_BUILD_EXT=1 BDH_BUILD_CUDA=1 pip install -e . --no-build-isolation

Cold CUDA kernel (``tril_score_v_cuda``) uses **tiled online** accumulation
(shared-mem Q/K/V tiles; no global T×T scores). Decode remains a separate
packed-past scaffold. Without the extension, training and tests still work
via the pure-PyTorch reference (which may materialize T×T — that is OK for
correctness, not the GPU path).

Wire-up: ``BDH_ATTN_IMPL=cuda`` → cold ``bdh_attn`` / decode ``bdh_attn_decode``.
Default remains ``eager``.
"""

from __future__ import annotations

from typing import Optional

import torch

_ext = None
_ext_load_error: Optional[BaseException] = None

try:
    import bdh_cuda_ext as _ext  # type: ignore
except Exception as exc:  # ImportError or missing symbols
    _ext_load_error = exc
    _ext = None


def has_cuda_ext() -> bool:
    """True if the optional native module imported successfully."""
    return _ext is not None


def has_cuda_kernel() -> bool:
    """True if the native module was built with a CUDA kernel."""
    if _ext is None:
        return False
    return bool(getattr(_ext, "has_cuda", False))


def tril_score_v_ref(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Pure PyTorch reference: always works on CPU or CUDA tensors.

    Shapes
    ------
    q, k : (B, H, T, Dk)
    v    : (B, H, T, Dv) or (B, 1, T, Dv)  — head dim broadcasts like bdh.Attention
    out  : (B, H, T, Dv)
    """
    if q.dim() != 4 or k.dim() != 4 or v.dim() != 4:
        raise ValueError("q, k, v must be 4D (B, H, T, D)")
    if q.shape != k.shape:
        raise ValueError(f"q and k shapes must match, got {tuple(q.shape)} vs {tuple(k.shape)}")
    if v.size(0) != q.size(0) or v.size(2) != q.size(2):
        raise ValueError("v batch/seq must match q")
    if v.size(1) not in (1, q.size(1)):
        raise ValueError("v heads must equal q heads or 1")

    scores = q @ k.transpose(-2, -1)
    scores = scores.tril(diagonal=-1)
    return scores @ v


def tril_score_v(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Dispatch: native CUDA/C++ ext when available and tensors match, else ref."""
    if _ext is not None:
        # Prefer CUDA kernel for CUDA tensors; CPU ext for CPU tensors.
        try:
            if q.is_cuda and has_cuda_kernel():
                return _ext.tril_score_v(q, k, v)
            if not q.is_cuda:
                return _ext.tril_score_v(q, k, v)
        except Exception:
            # Fall through to reference on any runtime failure.
            pass
    return tril_score_v_ref(q, k, v)


def tril_decode_ref(
    q: torch.Tensor, k_past: torch.Tensor, v_past: torch.Tensor
) -> torch.Tensor:
    """Pure PyTorch decode vs packed past KR/V (always available).

    Shapes
    ------
    q      : (B, H, Tq, Dk)  — typically Tq=1 for autoregressive decode
    k_past : (B, H, S, Dk)   — packed past keys (length S); excludes q positions
    v_past : (B, H, S, Dv) or (B, 1, S, Dv)
    out    : (B, H, Tq, Dv)

    Computes ``(Q @ K_past.mT) @ V_past``. Because K/V are the live cache
    prefix, every key is strictly earlier than the new queries → no self-attend
    (same as ``tril(diagonal=-1)``).
    """
    if q.dim() != 4 or k_past.dim() != 4 or v_past.dim() != 4:
        raise ValueError("q, k_past, v_past must be 4D")
    B, H, Tq, Dk = q.shape
    S = k_past.size(2)
    if k_past.shape[:2] != (B, H) or k_past.size(-1) != Dk:
        raise ValueError(
            f"k_past shape {tuple(k_past.shape)} incompatible with q {tuple(q.shape)}"
        )
    if v_past.size(0) != B or v_past.size(2) != S:
        raise ValueError(f"v_past shape {tuple(v_past.shape)} incompatible with k_past S={S}")
    if v_past.size(1) not in (1, H):
        raise ValueError("v_past heads must equal q heads or 1")

    Dv = v_past.size(-1)
    if S == 0:
        return q.new_zeros(B, H, Tq, Dv)

    # Broadcast V heads via matmul — no expand copy into (B,H,S,Dv).
    return (q @ k_past.transpose(-2, -1)) @ v_past


def tril_decode(
    q: torch.Tensor, k_past: torch.Tensor, v_past: torch.Tensor
) -> torch.Tensor:
    """Dispatch decode: native CUDA/C++ ext when available, else ref."""
    if _ext is not None and hasattr(_ext, "tril_decode"):
        try:
            if q.is_cuda and has_cuda_kernel():
                return _ext.tril_decode(q, k_past, v_past)
            if not q.is_cuda:
                return _ext.tril_decode(q, k_past, v_past)
        except Exception:
            pass
    return tril_decode_ref(q, k_past, v_past)


def ext_status() -> str:
    """Human-readable status for logs / OPT_NOTES."""
    if _ext is None:
        return f"native=unavailable ({type(_ext_load_error).__name__}: {_ext_load_error})"
    cuda = "yes" if has_cuda_kernel() else "no"
    decode = "yes" if hasattr(_ext, "tril_decode") else "no"
    return f"native=loaded has_cuda_kernel={cuda} has_decode={decode}"
