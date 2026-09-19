"""Autograd for strict-tril BDH attention: tril(Q @ K.T, -1) @ V.

No softmax, no 1/sqrt(d). Analytic backward for Q, K, V so custom forward
kernels (blocked / Triton / CUDA) can train without relying on PyTorch's
graph through the forward implementation.

Opt-in via BDH_ATTN_AUTOGRAD=1 or strict_tril_attn(..., use_fn=True).
Default remains OFF (eager train uses PyTorch autograd through GEMMs).
When set, cold / multi-token paths go through StrictTrilAttnFn (or
StrictTrilSelfAttnFn when Q is K — Dynamo-safe) so blocked / Triton / CUDA
forwards can train without differentiating the kernel graph.
T=1 CacheManager decode stays on the decode GEMM path (generate is no_grad).

Backward memory:
  * ``analytic_tril_attn_backward`` — dense recompute of full T×T scores M
    (reference; used for eager+AUTOGRAD).
  * ``analytic_tril_attn_backward_blocked`` — tiled recompute matching the
    blocked/online forward: never allocates a full T×T score tensor. Used
    automatically when ``BDH_ATTN_IMPL=blocked|online|triton|cuda``.

Why recompute (not save) scores: O does not retain M, and saving M would
defeat the blocked forward's peak-memory win. Tile-wise recompute of
``M_ij = Q_i·K_j`` (j<i) and ``dS_ij = dO_i·V_j`` is mathematically exact
for this linear (no-softmax) tril attention — no FlashAttention-style
softmax statistics are required.
"""

from __future__ import annotations

import os
from typing import Optional

import torch

from .attention import (
    DEFAULT_BLOCK_COLD,
    _SCORE_ELEMS_BUDGET,
    blocked_tril_attn,
    eager_tril_attn,
    triton_tril_attn,
    _can_use_triton,
)


def analytic_tril_attn_backward(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    dO: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Analytic gradients of O = tril(Q @ K.T, diagonal=-1) @ V.

    Dense reference: recomputes full T×T scores ``M``. Prefer
    ``analytic_tril_attn_backward_blocked`` when training with a blocked /
    online / fused forward so the backward does not force a full T×T either.

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


def analytic_tril_attn_backward_blocked(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    dO: torch.Tensor,
    block_size: int = DEFAULT_BLOCK_COLD,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Tiled analytic bwd — same grads as dense, no full T×T score buffer.

    Mirrors ``blocked_tril_attn`` tiling:

    * **Past** (key block entirely before query block): ephemeral ``Bi×Bj``
      score tiles for ``dS = dO @ V.T`` and ``M = Q @ K.T``.
    * **Diagonal**: ``Bi×Bi`` with ``tril(diagonal=-1)``.

    Peak score elems bound ≪ ``T×T`` for mid T (same budget as forward).
    Exact for float64 / fp32 up to roundoff; used by ``StrictTrilAttnFn`` when
    ``impl`` is blocked / online / triton / cuda.
    """
    B, H, T, _N = Q.shape
    D = V.shape[-1]
    v_broadcast = V.size(1) == 1 and H != 1
    if v_broadcast:
        Vh = V.expand(B, H, T, D)
    else:
        Vh = V

    if Q.dtype in (torch.float16, torch.bfloat16):
        acc_dtype = torch.float32
    else:
        acc_dtype = Q.dtype

    Qf = Q.to(dtype=acc_dtype)
    Kf = K.to(dtype=acc_dtype)
    Vhf = Vh.to(dtype=acc_dtype)
    dOf = dO.to(dtype=acc_dtype)

    dQ = torch.zeros(B, H, T, Q.size(-1), device=Q.device, dtype=acc_dtype)
    dK = torch.zeros(B, H, T, K.size(-1), device=K.device, dtype=acc_dtype)
    dVh = torch.zeros(B, H, T, D, device=V.device, dtype=acc_dtype)

    BS = max(1, int(block_size))
    score_budget = max(BS * BS, _SCORE_ELEMS_BUDGET)

    for i0 in range(0, T, BS):
        i1 = min(i0 + BS, T)
        Qi = Qf[:, :, i0:i1, :]
        dOi = dOf[:, :, i0:i1, :]
        Bi = i1 - i0

        # Past: every key index j < i0 is strictly before all queries in Qi.
        if i0 > 0:
            if Bi * i0 <= score_budget:
                Kj = Kf[:, :, :i0, :]
                Vj = Vhf[:, :, :i0, :]
                dS = dOi @ Vj.transpose(-2, -1)
                dQ[:, :, i0:i1, :] = dQ[:, :, i0:i1, :] + dS @ Kj
                dK[:, :, :i0, :] = dK[:, :, :i0, :] + dS.transpose(-2, -1) @ Qi
                M = Qi @ Kj.transpose(-2, -1)
                dVh[:, :, :i0, :] = dVh[:, :, :i0, :] + M.transpose(-2, -1) @ dOi
            else:
                tile = max(BS, score_budget // max(Bi, 1))
                for j0 in range(0, i0, tile):
                    j1 = min(j0 + tile, i0)
                    Kj = Kf[:, :, j0:j1, :]
                    Vj = Vhf[:, :, j0:j1, :]
                    dS = dOi @ Vj.transpose(-2, -1)
                    dQ[:, :, i0:i1, :] = dQ[:, :, i0:i1, :] + dS @ Kj
                    dK[:, :, j0:j1, :] = (
                        dK[:, :, j0:j1, :] + dS.transpose(-2, -1) @ Qi
                    )
                    M = Qi @ Kj.transpose(-2, -1)
                    dVh[:, :, j0:j1, :] = (
                        dVh[:, :, j0:j1, :] + M.transpose(-2, -1) @ dOi
                    )

        # Diagonal: Bi×Bi strict lower triangle.
        if Bi > 1:
            Ki = Kf[:, :, i0:i1, :]
            Vi = Vhf[:, :, i0:i1, :]
            dS = (dOi @ Vi.transpose(-2, -1)).tril(diagonal=-1)
            dQ[:, :, i0:i1, :] = dQ[:, :, i0:i1, :] + dS @ Ki
            dK[:, :, i0:i1, :] = dK[:, :, i0:i1, :] + dS.transpose(-2, -1) @ Qi
            M = (Qi @ Ki.transpose(-2, -1)).tril(diagonal=-1)
            dVh[:, :, i0:i1, :] = dVh[:, :, i0:i1, :] + M.transpose(-2, -1) @ dOi

    if v_broadcast:
        dV = dVh.sum(dim=1, keepdim=True)
    else:
        dV = dVh

    return (
        dQ.to(dtype=Q.dtype),
        dK.to(dtype=K.dtype),
        dV.to(dtype=V.dtype),
    )


def _normalize_impl(impl: str) -> str:
    name = (impl or "eager").strip().lower()
    if name == "online":
        return "blocked"
    return name


def _use_blocked_analytic_bwd(impl: str) -> bool:
    """Blocked/online/triton/cuda forwards avoid full T×T — match in bwd."""
    return _normalize_impl(impl) in ("blocked", "triton", "cuda")


def _forward_impl(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    impl: str,
) -> torch.Tensor:
    name = _normalize_impl(impl)
    if name == "eager":
        return eager_tril_attn(Q, K, V)
    if name == "blocked":
        return blocked_tril_attn(Q, K, V)
    if name == "triton":
        if _can_use_triton(Q):
            return triton_tril_attn(Q, K, V)
        return blocked_tril_attn(Q, K, V)
    if name == "cuda":
        from .cuda_attn import tril_score_v

        return tril_score_v(Q, K, V)
    raise ValueError(f"unknown attn impl={impl!r}")


class StrictTrilAttnFn(torch.autograd.Function):
    """Forward via eager/blocked/triton/cuda; analytic backward for Q, K, V.

    When ``impl`` is blocked/online/triton/cuda, backward uses the tiled
    analytic kernel (no full T×T). Eager keeps the dense M-recompute path
    (same asymptotic as the eager forward).

    Prefer ``StrictTrilSelfAttnFn`` when ``Q is K`` (BDH cold path): Dynamo
    graph-breaks on ``Function.apply`` with the same tensor twice
    (``gb6297``). Distinct Q/K keep this three-tensor form.
    """

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
        # Detach so the Function owns the graph; analytic bwd recomputes
        # scores (dense or tiled) from saved Q/K/V — never saves full M.
        return out

    @staticmethod
    def backward(ctx, dO: torch.Tensor):
        Q, K, V = ctx.saved_tensors
        if _use_blocked_analytic_bwd(ctx.impl):
            dQ, dK, dV = analytic_tril_attn_backward_blocked(Q, K, V, dO)
        else:
            dQ, dK, dV = analytic_tril_attn_backward(Q, K, V, dO)
        # impl has no grad
        return dQ, dK, dV, None


class StrictTrilSelfAttnFn(torch.autograd.Function):
    """Self-attn form (K is Q) — Dynamo-safe; sums dQ+dK into Q's grad.

    BDH always passes ``bdh_attn(QR, QR, V)``. Calling
    ``StrictTrilAttnFn.apply(Q, Q, V, impl)`` trips Dynamo
    ``autograd.Function.apply: duplicate tensor input`` and forces ~1 graph
    break per layer. This Function takes ``(Q, V, impl)`` only and returns
    ``dQ + dK`` for Q (same accumulation PyTorch would do for duplicate
    inputs).
    """

    @staticmethod
    def forward(
        ctx,
        Q: torch.Tensor,
        V: torch.Tensor,
        impl: str,
    ) -> torch.Tensor:
        ctx.impl = impl
        ctx.save_for_backward(Q, V)
        with torch.no_grad():
            out = _forward_impl(Q, Q, V, impl)
        return out

    @staticmethod
    def backward(ctx, dO: torch.Tensor):
        Q, V = ctx.saved_tensors
        if _use_blocked_analytic_bwd(ctx.impl):
            dQ, dK, dV = analytic_tril_attn_backward_blocked(Q, Q, V, dO)
        else:
            dQ, dK, dV = analytic_tril_attn_backward(Q, Q, V, dO)
        # K is Q → accumulate both score grads into Q
        return dQ + dK, dV, None


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
      - True  → always analytic Function (self-attn when ``Q is K``)
      - False → plain forward (PyTorch autograd if ops support it)
      - None  → True when BDH_ATTN_AUTOGRAD is truthy, else False

    When ``Q is K``, routes through ``StrictTrilSelfAttnFn`` so
    ``torch.compile`` does not graph-break on duplicate Function inputs.
    """
    if use_fn is None:
        use_fn = _env_autograd_enabled()
    name = impl if impl is not None else os.environ.get("BDH_ATTN_IMPL", "eager")
    name = _normalize_impl(name)
    if use_fn:
        if Q is K:
            return StrictTrilSelfAttnFn.apply(Q, V, name)
        return StrictTrilAttnFn.apply(Q, K, V, name)
    return _forward_impl(Q, K, V, name)


def _env_autograd_enabled() -> bool:
    raw = os.environ.get("BDH_ATTN_AUTOGRAD", "").strip().lower()
    return raw in ("1", "true", "yes", "on")
