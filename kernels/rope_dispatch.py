"""Env-gated dispatch for BDH RoPE rotate backends.

Backends (``BDH_ROPE_IMPL``)::

    export BDH_ROPE_IMPL=eager   # default — strided even/odd Attention.rope math
    export BDH_ROPE_IMPL=fused   # pair-contiguous PyTorch; Triton on CUDA

``rope_cos_sin`` caching remains in ``bdh.Attention``; this module only picks
how (cos, sin) are applied to ``v``.
"""

from __future__ import annotations

import os
from typing import Literal, Optional

import torch

from .rope import (
    _HAS_TRITON,
    _can_use_triton_rope,
    eager_rope_rotate,
    fused_rope_rotate,
    fused_rope_rotate_pytorch,
    fused_rope_rotate_triton,
)

ImplName = Literal["eager", "fused"]
_VALID = ("eager", "fused")


# Env resolve cache: rope runs every layer/step in generate.
_ROPE_IMPL_ENV: object | None = object()
_ROPE_IMPL_RESOLVED: ImplName = "eager"


def resolve_rope_impl(requested: str | None = None) -> ImplName:
    """Resolve BDH_ROPE_IMPL (or explicit override) to a concrete backend name."""
    global _ROPE_IMPL_ENV, _ROPE_IMPL_RESOLVED
    if requested is not None:
        raw = (requested or "eager").strip().lower()
        if raw not in _VALID:
            raise ValueError(
                f"BDH_ROPE_IMPL must be {'|'.join(_VALID)}, got {raw!r}"
            )
        return raw  # type: ignore[return-value]
    env = os.environ.get("BDH_ROPE_IMPL", "eager")
    if env is _ROPE_IMPL_ENV or env == _ROPE_IMPL_ENV:
        return _ROPE_IMPL_RESOLVED
    raw = (env or "eager").strip().lower()
    if raw not in _VALID:
        raise ValueError(
            f"BDH_ROPE_IMPL must be {'|'.join(_VALID)}, got {raw!r}"
        )
    _ROPE_IMPL_ENV = env
    _ROPE_IMPL_RESOLVED = raw  # type: ignore[assignment]
    return _ROPE_IMPL_RESOLVED


def bdh_rope_rotate(
    v: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    *,
    impl: str | None = None,
) -> torch.Tensor:
    """Apply RoPE cos/sin to ``v`` with the selected backend.

    - eager: historical strided even/odd path (bit-identical to prior ``Attention.rope``)
    - fused: Triton on CUDA when usable; else pure-PyTorch pair-contiguous path
    """
    name = resolve_rope_impl(impl)
    if name == "eager":
        return eager_rope_rotate(v, cos, sin, out=out)
    return fused_rope_rotate(v, cos, sin, out=out)


def backend_info(device: torch.device | None = None) -> dict:
    """Small diagnostic for notes / tests."""
    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    probe = torch.empty(1, device=dev)
    return {
        "BDH_ROPE_IMPL": resolve_rope_impl(),
        "has_triton": _HAS_TRITON,
        "triton_usable": _can_use_triton_rope(probe),
        "device": str(dev),
    }


__all__ = [
    "resolve_rope_impl",
    "bdh_rope_rotate",
    "backend_info",
    "eager_rope_rotate",
    "fused_rope_rotate",
    "fused_rope_rotate_pytorch",
    "fused_rope_rotate_triton",
]
