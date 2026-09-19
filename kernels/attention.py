"""Strict lower-triangular BDH attention: tril(Q @ K.T, diagonal=-1) @ V.

No softmax, no 1/sqrt(d). Position i attends only to j < i.
"""

from __future__ import annotations

from typing import Optional

import torch

# Optional Triton (may be installed without a usable CUDA device).
try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except Exception:  # pragma: no cover - import guard
    triton = None  # type: ignore
    tl = None  # type: ignore
    _HAS_TRITON = False


def eager_tril_attn(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """Reference: materialize full T×T scores, then tril_(diagonal=-1), then @ V.

    Q, K: (B, H, T, N)
    V:    (B, 1, T, D) or (B, H, T, D)
    Returns: (B, H, T, D)
    """
    scores = Q @ K.transpose(-2, -1)
    scores = scores.tril(diagonal=-1)
    return scores @ V


# Default tile widths. Decode uses a larger default so Tq=1 past scans need
# fewer Python trips; cold blocked keeps 64 for score-tile memory.
DEFAULT_BLOCK_COLD = 64
DEFAULT_BLOCK_DECODE = 256
# Soft cap on score elements (Tq * tile) before preferring another tile split.
_SCORE_ELEMS_BUDGET = 256 * 256


def _expand_v_heads(V: torch.Tensor, B: int, H: int, S: int, D: int) -> torch.Tensor:
    """Broadcast V from (B,1,S,D) to (B,H,S,D) when heads share values."""
    if V.size(1) == 1 and H != 1:
        return V.expand(B, H, S, D)
    return V


def _pick_tile_size(S: int, Tq: int, block_size: int) -> int:
    """Choose a past-axis tile width: larger tiles ⇒ fewer Python loop trips.

    Still bounds peak score memory roughly by ``Tq * tile`` via ``_SCORE_ELEMS_BUDGET``.
    """
    BS = max(1, int(block_size))
    if S <= BS:
        return S if S > 0 else BS
    # Aim for a modest number of tiles on the hot Tq=1 decode path.
    if Tq <= 4:
        target_tiles = 8
        want = max(BS, (S + target_tiles - 1) // target_tiles)
        max_by_budget = max(BS, _SCORE_ELEMS_BUDGET // max(Tq, 1))
        return min(S, want, max_by_budget)
    return min(S, BS)


def _tiled_score_v(
    Q: torch.Tensor,
    K: torch.Tensor,
    Vh: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    """``(Q @ K.mT) @ Vh`` tiled over the past axis (no tril — all keys valid).

    Shared by ``blocked_decode_attn`` and the past region of ``blocked_tril_attn``.
    One-shots when the full ``(Tq × S)`` score fits the budget; otherwise loops
    over adaptive tiles (fewer trips than a fixed small BS).
    """
    B, H, Tq, _N = Q.shape
    S = K.size(2)
    D = Vh.size(-1)
    if S == 0:
        return Q.new_zeros(B, H, Tq, D)

    BS = _pick_tile_size(S, Tq, block_size)
    # Modest past / small score: single two-GEMM (same as eager decode).
    if S <= BS or Tq * S <= _SCORE_ELEMS_BUDGET:
        return (Q @ K.transpose(-2, -1)) @ Vh

    out = Q.new_zeros(B, H, Tq, D)
    for j0 in range(0, S, BS):
        j1 = min(j0 + BS, S)
        Kj = K[:, :, j0:j1, :]
        Vj = Vh[:, :, j0:j1, :]
        out = out + (Q @ Kj.transpose(-2, -1)) @ Vj
    return out


def blocked_tril_attn(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    block_size: int = DEFAULT_BLOCK_COLD,
) -> torch.Tensor:
    """Pure-PyTorch strict-tril attention without a full upper triangle.

    Tiles the sequence so score tiles are at most (BS × past) / (BS × BS).
    Past-only regions reuse ``_tiled_score_v`` (same helper as decode).
    Semantically identical to eager_tril_attn (up to fp roundoff on long T).
    Works on CPU and CUDA; used as the non-Triton "triton" path fallback and
    as a lower-memory eager alternative.
    """
    B, H, T, N = Q.shape
    D = V.shape[-1]
    Vh = _expand_v_heads(V, B, H, T, D)

    out = Q.new_zeros(B, H, T, D)
    BS = max(1, int(block_size))

    for i0 in range(0, T, BS):
        i1 = min(i0 + BS, T)
        Qi = Q[:, :, i0:i1, :]

        # Full past: all j < i0 are strictly before every query in [i0, i1).
        # Shared with decode — one helper, adaptive tiles, often a single GEMM.
        if i0 > 0:
            out[:, :, i0:i1, :] = _tiled_score_v(
                Qi, K[:, :, :i0, :], Vh[:, :, :i0, :], BS
            )

        # Diagonal block: apply tril(diagonal=-1) within the block
        Bi = i1 - i0
        if Bi > 1:
            Kj = K[:, :, i0:i1, :]
            Vj = Vh[:, :, i0:i1, :]
            scores = Qi @ Kj.transpose(-2, -1)
            scores = scores.tril(diagonal=-1)
            out[:, :, i0:i1, :] = out[:, :, i0:i1, :] + scores @ Vj

    return out


def _can_use_triton(Q: torch.Tensor) -> bool:
    if not _HAS_TRITON:
        return False
    if not Q.is_cuda:
        return False
    if not torch.cuda.is_available():
        return False
    return True


if _HAS_TRITON:

    @triton.jit
    def _bdh_attn_fwd_kernel(
        Q_ptr,
        K_ptr,
        V_ptr,
        Out_ptr,
        stride_qb,
        stride_qh,
        stride_qt,
        stride_qn,
        stride_kb,
        stride_kh,
        stride_kt,
        stride_kn,
        stride_vb,
        stride_vh,
        stride_vt,
        stride_vd,
        stride_ob,
        stride_oh,
        stride_ot,
        stride_od,
        T,
        N,
        D,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """Fused strict-tril score@V: out[i] = sum_{j<i} (q_i·k_j) * v_j.

        Grid: (cdiv(T, BLOCK_M), B * H)
        """
        pid_m = tl.program_id(0)
        pid_bh = tl.program_id(1)
        # Unpack batch / head from linear bh index via host-passed H? We use
        # flat BH and strides that already fold B*H — caller sets nh_total.
        # Here pid_bh indexes the combined (B,H) plane; strides are per-plane.
        bh = pid_bh

        start_m = pid_m * BLOCK_M
        offs_m = start_m + tl.arange(0, BLOCK_M)
        mask_m = offs_m < T

        # Base pointers for this (b,h)
        q_bh = Q_ptr + bh * stride_qh  # stride_qh is actually stride over H within flat BH
        k_bh = K_ptr + bh * stride_kh
        v_bh = V_ptr + bh * stride_vh
        o_bh = Out_ptr + bh * stride_oh

        # Accumulators over D in tiles
        # We stream D in BLOCK_D chunks to keep registers bounded.
        for d0 in range(0, D, BLOCK_D):
            offs_d = d0 + tl.arange(0, BLOCK_D)
            mask_d = offs_d < D
            acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

            # Iterate key blocks with j_end <= start_m + BLOCK_M, but only j < i
            # For simplicity: all key blocks with j0 < start_m + BLOCK_M
            end_n = start_m + BLOCK_M  # keys with j < end_n may contribute
            for j0 in range(0, end_n, BLOCK_N):
                offs_n = j0 + tl.arange(0, BLOCK_N)
                mask_n = offs_n < T

                # Compute QK scores for this tile: (BLOCK_M, BLOCK_N)
                qk = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
                for k0 in range(0, N, BLOCK_K):
                    offs_k = k0 + tl.arange(0, BLOCK_K)
                    mask_k = offs_k < N
                    q = tl.load(
                        q_bh
                        + offs_m[:, None] * stride_qt
                        + offs_k[None, :] * stride_qn,
                        mask=mask_m[:, None] & mask_k[None, :],
                        other=0.0,
                    ).to(tl.float32)
                    k = tl.load(
                        k_bh
                        + offs_n[:, None] * stride_kt
                        + offs_k[None, :] * stride_kn,
                        mask=mask_n[:, None] & mask_k[None, :],
                        other=0.0,
                    ).to(tl.float32)
                    qk += tl.dot(q, tl.trans(k))

                # Strict lower: keep only j < i (exclude diagonal and above)
                causal = offs_n[None, :] < offs_m[:, None]
                qk = tl.where(causal & mask_m[:, None] & mask_n[None, :], qk, 0.0)

                v = tl.load(
                    v_bh
                    + offs_n[:, None] * stride_vt
                    + offs_d[None, :] * stride_vd,
                    mask=mask_n[:, None] & mask_d[None, :],
                    other=0.0,
                ).to(tl.float32)
                acc += tl.dot(qk.to(v.dtype), v)

            tl.store(
                o_bh
                + offs_m[:, None] * stride_ot
                + offs_d[None, :] * stride_od,
                acc.to(Out_ptr.dtype.element_ty),
                mask=mask_m[:, None] & mask_d[None, :],
            )


def triton_tril_attn(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    block_m: int = 32,
    block_n: int = 32,
) -> torch.Tensor:
    """Triton fused strict-tril attention. Requires CUDA + Triton.

    Falls back to blocked_tril_attn if Triton cannot run.
    """
    if not _can_use_triton(Q):
        return blocked_tril_attn(Q, K, V)

    assert Q.is_contiguous() or True
    B, H, T, N = Q.shape
    D = V.shape[-1]

    # Expand V to (B,H,T,D) for uniform strides (head broadcast).
    if V.size(1) == 1 and H != 1:
        Vh = V.expand(B, H, T, D).contiguous()
    else:
        Vh = V.contiguous()

    Qc = Q.contiguous()
    Kc = K.contiguous()
    out = torch.empty(B, H, T, D, device=Q.device, dtype=Q.dtype)

    # Flatten (B,H) into one grid dimension; use H-stride as plane stride.
    # Pointer arithmetic: element (b,h,t,n) at
    #   b*stride_qb + h*stride_qh + t*stride_qt + n*stride_qn
    # We launch pid_bh in [0, B*H) and treat stride_qh' = stride over consecutive
    # (b,h) planes = stride_qh when scanning h innermost... easier: view as (BH,...)
    Qf = Qc.view(B * H, T, N)
    Kf = Kc.view(B * H, T, N)
    Vf = Vh.view(B * H, T, D)
    Of = out.view(B * H, T, D)

    BH = B * H
    BLOCK_M = block_m
    BLOCK_N = block_n
    BLOCK_D = min(64, triton.next_power_of_2(D) if D > 0 else 1)
    BLOCK_K = min(64, triton.next_power_of_2(N) if N > 0 else 1)
    # Cap BLOCK_D/K to reasonable sizes
    BLOCK_D = min(BLOCK_D, 64)
    BLOCK_K = min(BLOCK_K, 64)

    grid = (triton.cdiv(T, BLOCK_M), BH)

    _bdh_attn_fwd_kernel[grid](
        Qf,
        Kf,
        Vf,
        Of,
        Qf.stride(0),
        Qf.stride(0),
        Qf.stride(1),
        Qf.stride(2),
        Kf.stride(0),
        Kf.stride(0),
        Kf.stride(1),
        Kf.stride(2),
        Vf.stride(0),
        Vf.stride(0),
        Vf.stride(1),
        Vf.stride(2),
        Of.stride(0),
        Of.stride(0),
        Of.stride(1),
        Of.stride(2),
        T,
        N,
        D,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
        BLOCK_K=BLOCK_K,
    )
    return out


def tril_attn(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    *,
    impl: Optional[str] = None,
) -> torch.Tensor:
    """Dispatch helper used by attention_dispatch / tests."""
    if impl is None:
        impl = "triton" if _can_use_triton(Q) else "blocked"
    if impl == "eager":
        return eager_tril_attn(Q, K, V)
    if impl == "blocked":
        return blocked_tril_attn(Q, K, V)
    if impl == "triton":
        return triton_tril_attn(Q, K, V)
    raise ValueError(f"unknown attn impl={impl!r}")


def blocked_decode_attn(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    block_size: int = DEFAULT_BLOCK_DECODE,
) -> torch.Tensor:
    """Single-chunk decode against past KR/V only (strict tril diagonal=-1).

    Q: (B, H, Tq, N) — typically Tq=1 for autoregressive decode
    K: (B, H, S, N)  — packed past keys (length S); does **not** include Q's positions
    V: (B, 1, S, D) or (B, H, S, D) — packed past values

    Computes ``(Q @ K.mT) @ V`` via shared ``_tiled_score_v``. Because K/V are
    the live cache prefix, every key is strictly earlier than the new queries,
    so the new token never attends to itself (same as ``tril(diagonal=-1)``).

    Never materializes an ``(S+Tq) x (S+Tq)`` score matrix (unlike concat + full
    tril attn). Default tile is larger than cold blocked (fewer Python trips).
    """
    B, H, Tq, N = Q.shape
    S = K.size(2)
    D = V.size(-1)
    if K.shape[:2] != (B, H) or K.size(-1) != N:
        raise ValueError(f"K shape {tuple(K.shape)} incompatible with Q {tuple(Q.shape)}")
    if V.size(0) != B or V.size(2) != S:
        raise ValueError(f"V shape {tuple(V.shape)} incompatible with K S={S}")

    if S == 0:
        return Q.new_zeros(B, H, Tq, D)

    Vh = _expand_v_heads(V, B, H, S, D)
    return _tiled_score_v(Q, K, Vh, block_size)


if _HAS_TRITON:

    @triton.jit
    def _bdh_decode_fwd_kernel(
        Q_ptr,
        K_ptr,
        V_ptr,
        Out_ptr,
        stride_qh,
        stride_qt,
        stride_qn,
        stride_kh,
        stride_kt,
        stride_kn,
        stride_vh,
        stride_vt,
        stride_vd,
        stride_oh,
        stride_ot,
        stride_od,
        Tq,
        S,
        N,
        D,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """Fused decode: out[t] = sum_{j<S} (q_t·k_j) * v_j (all past keys valid).

        Grid: (cdiv(Tq, 1) effectively one row-group per program, B*H).
        pid_m indexes query rows in steps of 1..BLOCK_M with BLOCK_M=1 typical.
        """
        pid_m = tl.program_id(0)
        bh = tl.program_id(1)

        # One query row per program when Tq is small; still masked for safety.
        offs_m = pid_m + tl.arange(0, 1)
        mask_m = offs_m < Tq

        q_bh = Q_ptr + bh * stride_qh
        k_bh = K_ptr + bh * stride_kh
        v_bh = V_ptr + bh * stride_vh
        o_bh = Out_ptr + bh * stride_oh

        for d0 in range(0, D, BLOCK_D):
            offs_d = d0 + tl.arange(0, BLOCK_D)
            mask_d = offs_d < D
            acc = tl.zeros((1, BLOCK_D), dtype=tl.float32)

            for j0 in range(0, S, BLOCK_N):
                offs_n = j0 + tl.arange(0, BLOCK_N)
                mask_n = offs_n < S

                qk = tl.zeros((1, BLOCK_N), dtype=tl.float32)
                for k0 in range(0, N, BLOCK_K):
                    offs_k = k0 + tl.arange(0, BLOCK_K)
                    mask_k = offs_k < N
                    q = tl.load(
                        q_bh
                        + offs_m[:, None] * stride_qt
                        + offs_k[None, :] * stride_qn,
                        mask=mask_m[:, None] & mask_k[None, :],
                        other=0.0,
                    ).to(tl.float32)
                    k = tl.load(
                        k_bh
                        + offs_n[:, None] * stride_kt
                        + offs_k[None, :] * stride_kn,
                        mask=mask_n[:, None] & mask_k[None, :],
                        other=0.0,
                    ).to(tl.float32)
                    qk += tl.dot(q, tl.trans(k))

                qk = tl.where(mask_m[:, None] & mask_n[None, :], qk, 0.0)
                v = tl.load(
                    v_bh
                    + offs_n[:, None] * stride_vt
                    + offs_d[None, :] * stride_vd,
                    mask=mask_n[:, None] & mask_d[None, :],
                    other=0.0,
                ).to(tl.float32)
                acc += tl.dot(qk.to(v.dtype), v)

            tl.store(
                o_bh
                + offs_m[:, None] * stride_ot
                + offs_d[None, :] * stride_od,
                acc.to(Out_ptr.dtype.element_ty),
                mask=mask_m[:, None] & mask_d[None, :],
            )


def triton_decode_attn(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    block_size: int = DEFAULT_BLOCK_DECODE,
    block_n: int = 64,
) -> torch.Tensor:
    """Decode against past KR/V. Triton fused on CUDA; else blocked.

    On CPU (this box) and whenever Triton cannot run, uses
    ``blocked_decode_attn`` so ``BDH_ATTN_IMPL=triton`` still gets the
    no-full-TxT decode path. Dedicated decode kernel skips causal masking
    (past keys are all valid under tril(-1)).
    """
    if not _can_use_triton(Q):
        return blocked_decode_attn(Q, K, V, block_size=block_size)

    B, H, Tq, N = Q.shape
    S = K.size(2)
    D = V.size(-1)
    if S == 0:
        return Q.new_zeros(B, H, Tq, D)

    if V.size(1) == 1 and H != 1:
        Vh = V.expand(B, H, S, D).contiguous()
    else:
        Vh = V.contiguous()

    Qc = Q.contiguous()
    Kc = K.contiguous()
    out = torch.empty(B, H, Tq, D, device=Q.device, dtype=Q.dtype)

    Qf = Qc.view(B * H, Tq, N)
    Kf = Kc.view(B * H, S, N)
    Vf = Vh.view(B * H, S, D)
    Of = out.view(B * H, Tq, D)

    BH = B * H
    BLOCK_N = min(block_n, triton.next_power_of_2(max(S, 1)))
    BLOCK_N = max(16, min(BLOCK_N, 128))
    BLOCK_D = min(64, triton.next_power_of_2(D) if D > 0 else 1)
    BLOCK_K = min(64, triton.next_power_of_2(N) if N > 0 else 1)

    grid = (Tq, BH)
    _bdh_decode_fwd_kernel[grid](
        Qf,
        Kf,
        Vf,
        Of,
        Qf.stride(0),
        Qf.stride(1),
        Qf.stride(2),
        Kf.stride(0),
        Kf.stride(1),
        Kf.stride(2),
        Vf.stride(0),
        Vf.stride(1),
        Vf.stride(2),
        Of.stride(0),
        Of.stride(1),
        Of.stride(2),
        Tq,
        S,
        N,
        D,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
        BLOCK_K=BLOCK_K,
    )
    return out


def eager_decode_attn(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
) -> torch.Tensor:
    """Reference decode: ``(Q @ K.mT) @ V`` (past-only; no self-attention)."""
    B, H, Tq, _ = Q.shape
    S = K.size(2)
    D = V.size(-1)
    if S == 0:
        return Q.new_zeros(B, H, Tq, D)
    Vh = _expand_v_heads(V, B, H, S, D)
    return (Q @ K.transpose(-2, -1)) @ Vh
