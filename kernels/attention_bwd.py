"""Autograd for strict-tril BDH attention: tril(Q @ K.T, -1) @ V.

No softmax, no 1/sqrt(d). Analytic backward for Q, K, V so custom forward
kernels (blocked / Triton / CUDA) can train without relying on PyTorch's
graph through the forward implementation.

Opt-in via BDH_ATTN_AUTOGRAD=1 or strict_tril_attn(..., use_fn=True).
Default eager path in bdh.py is unchanged when the env var is unset.
"""

from __future__ import annotations

import os
from typing import Optional

import torch

from .attention import blocked_tril_attn, eager_tril_attn, triton_tril_attn, _can_use_triton


def analytic_tril_attn_backward(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    dO: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Analytic gradients of O = tril(Q @ K.T, diagonal=-1) @ V.

    Q, K: (B, H, T, N)
    V:    (B, 1, T, D) or (B, H, T, D)
    dO:   (B, H, T, D)

    Returns (dQ, dK, dV) with dV shaped like V.
    """
    B, H, T, _N = Q.shape
    D = V.shape[-1]
    v_broadcast = V.size(1) == 1 and H != 1
    if v_broadcast:
        Vh = V.expand(B, H, T, D)
    else:
        Vh = V

    # Recompute strict-lower scores (same as forward; no need to save T×T).
    M = (Q @ K.transpose(-2, -1)).tril(diagonal=-1)

    dM = dO @ Vh.transpose(-2, -1)
    dS = dM.tril(diagonal=-1)
    dQ = dS @ K
    dK = dS.transpose(-2, -1) @ Q
    dVh = M.transpose(-2, -1) @ dO
    if v_broadcast:
        dV = dVh.sum(dim=1, keepdim=True)
    else:
        dV = dVh
    return dQ, dK, dV


def _forward_impl(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    impl: str,
) -> torch.Tensor:
    name = (impl or "eager").strip().lower()
    if name == "eager":
        return eager_tril_attn(Q, K, V)
    if name == "blocked":
        return blocked_tril_attn(Q, K, V)
    if name == "triton":
        if _can_use_triton(Q):
            return triton_tril_attn(Q, K, V)
        return blocked_tril_attn(Q, K, V)
    raise ValueError(f"unknown attn impl={impl!r}")


class StrictTrilAttnFn(torch.autograd.Function):
    """Forward via eager/blocked/triton; analytic backward for Q, K, V."""

    @staticmethod
    def forward(
        ctx,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        impl: str,
    ) -> torch.Tensor:
        # Save inputs for analytic bwd; run forward without building a graph
        # through the (possibly non-differentiable) kernel.
        ctx.impl = impl
        ctx.save_for_backward(Q, K, V)
        with torch.no_grad():
            out = _forward_impl(Q, K, V, impl)
        # Detach so the Function owns the graph; analytic bwd recomputes M.
        return out

    @staticmethod
    def backward(ctx, dO: torch.Tensor):
        Q, K, V = ctx.saved_tensors
        dQ, dK, dV = analytic_tril_attn_backward(Q, K, V, dO)
        # impl has no grad
        return dQ, dK, dV, None


def strict_tril_attn(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    *,
    impl: Optional[str] = None,
    use_fn: Optional[bool] = None,
) -> torch.Tensor:
    """Strict-tril score@V with optional autograd.Function.

    use_fn:
      - True  → always StrictTrilAttnFn (analytic bwd)
      - False → plain forward (PyTorch autograd if ops support it)
      - None  → True when BDH_ATTN_AUTOGRAD is truthy, else False
    """
    if use_fn is None:
        use_fn = _env_autograd_enabled()
    name = impl if impl is not None else os.environ.get("BDH_ATTN_IMPL", "eager")
    name = (name or "eager").strip().lower()
    if use_fn:
        return StrictTrilAttnFn.apply(Q, K, V, name)
    return _forward_impl(Q, K, V, name)


def _env_autograd_enabled() -> bool:
    raw = os.environ.get("BDH_ATTN_AUTOGRAD", "").strip().lower()
    return raw in ("1", "true", "yes", "on")
