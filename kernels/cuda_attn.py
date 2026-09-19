"""Strict lower-triangular score×V for BDH attention.

Semantics (must match bdh.Attention cold path):

    scores = (Q @ K.transpose(-2, -1)).tril(diagonal=-1)
    out    = scores @ V

No softmax, no 1/sqrt(d) scale, diagonal EXCLUDED.

Public API
----------
tril_score_v_ref(q, k, v)  — pure PyTorch CPU/GPU reference (always available)
tril_score_v(q, k, v)      — native ext if built+compatible, else reference

Optional native build (scaffold in ``csrc/``)::

    pip install -e .                                 # pure Python (default)
    BDH_BUILD_EXT=1 pip install -e . --no-build-isolation
    BDH_BUILD_EXT=1 BDH_BUILD_CUDA=1 pip install -e . --no-build-isolation

Without the extension, training and tests still work via the reference path.
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


def ext_status() -> str:
    """Human-readable status for logs / OPT_NOTES."""
    if _ext is None:
        return f"native=unavailable ({type(_ext_load_error).__name__}: {_ext_load_error})"
    cuda = "yes" if has_cuda_kernel() else "no"
    return f"native=loaded has_cuda_kernel={cuda}"
