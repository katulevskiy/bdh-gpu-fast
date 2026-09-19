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
# Cold blocked past-region keeps this large budget (BLAS-friendly oneshot).
_SCORE_ELEMS_BUDGET = 256 * 256
# Decode-online-v2: tighter oneshot for T=1 vs packed KR/V. Above this many
# score elements (Tq*S), blocked/online decode *tiles* so peak score mem is
# ~Tq×tile (not Tq×S). Also avoids a CPU broadcast-V oneshot cliff at long S
# (measured ~8–10× vs tiled at S=4096, B=4 H=4). Per-head V still oneshots
# under the large cold budget (BH-bmm likes big GEMMs).
_DECODE_ONESHOT_ELEMS = 2048


def _expand_v_heads(V: torch.Tensor, B: int, H: int, S: int, D: int) -> torch.Tensor:
    """Broadcast V from (B,1,S,D) to (B,H,S,D) when heads share values."""
    if V.size(1) == 1 and H != 1:
        return V.expand(B, H, S, D)
    return V


def _as_contiguous(t: torch.Tensor) -> torch.Tensor:
    """Contiguous copy only when needed (shared cold/decode staging helper)."""
    return t if t.is_contiguous() else t.contiguous()


def _pick_triton_cold_tiles(
    T: int,
    N: int,
    D: int,
    *,
    block_m: int | None = None,
    block_n: int | None = None,
) -> tuple[int, int, int, int]:
    """Power-of-2 Triton tiles for cold strict-tril, aligned with online blocked.

    Prefers ``DEFAULT_BLOCK_COLD``-sized query/key tiles (larger than the old
    fixed 32×32) so each program covers more of the lower triangle. Caps keep
    register pressure bounded; ``BLOCK_D``/``BLOCK_K`` track head dims.
    """
    def _p2_cap(x: int, lo: int, hi: int) -> int:
        x = max(lo, min(int(x), hi))
        # next_power_of_2 without requiring Triton at import time on CPU
        p = 1
        while p < x:
            p <<= 1
        return min(p, hi)

    # Default query/key tile ~ online blocked BS (64), grow slightly for long T.
    if block_m is None:
        want_m = DEFAULT_BLOCK_COLD if T <= 256 else min(128, DEFAULT_BLOCK_COLD * 2)
        block_m = _p2_cap(min(T, want_m) if T > 0 else want_m, 16, 128)
    else:
        block_m = _p2_cap(block_m, 16, 128)
    if block_n is None:
        want_n = DEFAULT_BLOCK_COLD if T <= 256 else min(128, DEFAULT_BLOCK_COLD * 2)
        block_n = _p2_cap(min(max(T, 1), want_n), 16, 128)
    else:
        block_n = _p2_cap(block_n, 16, 128)

    block_d = _p2_cap(D if D > 0 else 1, 16, 64)
    block_k = _p2_cap(N if N > 0 else 1, 16, 64)
    return block_m, block_n, block_d, block_k


def _pick_tile_size(
    S: int,
    Tq: int,
    block_size: int,
    *,
    score_budget: int | None = None,
) -> int:
    """Choose a past-axis tile width: larger tiles ⇒ fewer Python loop trips.

    Bounds peak score memory roughly by ``Tq * tile`` via ``score_budget``
    (default ``_SCORE_ELEMS_BUDGET``; decode passes ``_DECODE_ONESHOT_ELEMS``).
    """
    BS = max(1, int(block_size))
    budget = _SCORE_ELEMS_BUDGET if score_budget is None else max(1, int(score_budget))
    if S <= BS:
        return S if S > 0 else BS
    # Aim for a modest number of tiles on the hot Tq=1 decode path.
    # Long past: fewer tiles (larger GEMMs) — better CPU wall + still ≪ Tq×S peak.
    if Tq <= 4:
        target_tiles = 4 if S >= 4096 else (6 if S >= 2048 else 8)
        want = max(BS, (S + target_tiles - 1) // target_tiles)
        max_by_budget = max(BS, budget // max(Tq, 1))
        return min(S, want, max_by_budget)
    return min(S, BS)


def _two_gemm_decode(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
) -> torch.Tensor:
    """``(Q @ K.mT) @ V`` — shared eager / small-tile decode GEMM.

    Keeps broadcast ``V=(B,1,S,D)`` (no expand). For the hot ``Tq=1`` path with
    head-matched ``V=(B,H,S,D)``, flattens to ``bmm`` over ``B*H`` so BLAS sees
    dense ``(1,N)×(N,S)`` and ``(1,S)×(S,D)`` without a 4D batch broadcast.
    """
    B, H, Tq, N = Q.shape
    S = K.size(2)
    D = V.size(-1)
    if S == 0:
        return Q.new_zeros(B, H, Tq, D)

    # Hot Tq=1 + per-head V: BH-flattened bmm (same math as 4D @).
    if Tq == 1 and V.size(1) == H:
        Qf = Q.reshape(B * H, 1, N)
        Kf = K.reshape(B * H, S, N)
        Vf = V.reshape(B * H, S, D)
        scores = torch.bmm(Qf, Kf.transpose(1, 2))
        return torch.bmm(scores, Vf).view(B, H, 1, D)

    # Broadcast V or Tq>1: 4D matmul (broadcasts heads; no expand copy).
    return (Q @ K.transpose(-2, -1)) @ V


def _tiled_score_v(
    Q: torch.Tensor,
    K: torch.Tensor,
    Vh: torch.Tensor,
    block_size: int,
    *,
    oneshot_elems: int | None = None,
) -> torch.Tensor:
    """``(Q @ K.mT) @ Vh`` tiled over the past axis (no tril — all keys valid).

    Shared by ``blocked_decode_attn`` and the past region of ``blocked_tril_attn``.
    One-shots when the full ``(Tq × S)`` score fits ``oneshot_elems`` (default
    ``_SCORE_ELEMS_BUDGET`` for cold); otherwise loops over adaptive tiles.

    ``Vh`` may be ``(B, H, S, D)`` or broadcast ``(B, 1, S, D)`` — matmul
    broadcasts heads, so decode can skip an expand copy into ``(B, H, S, D)``.
    Tile loop accumulates with ``out.add_`` (no ``out + x`` temporary).
    """
    B, H, Tq, _N = Q.shape
    S = K.size(2)
    D = Vh.size(-1)
    if S == 0:
        return Q.new_zeros(B, H, Tq, D)

    budget = (
        _SCORE_ELEMS_BUDGET if oneshot_elems is None else max(1, int(oneshot_elems))
    )
    BS = _pick_tile_size(S, Tq, block_size, score_budget=budget)
    # Modest past / small score: single two-GEMM (same as eager decode).
    # Ephemeral ``(Tq × S)`` only — never ``(S+Tq)×(S+Tq)``.
    if S <= BS or Tq * S <= budget:
        return _two_gemm_decode(Q, K, Vh)

    out = Q.new_zeros(B, H, Tq, D)
    # Tq=1: flatten Q once; reuse across past tiles (decode-online-v2).
    if Tq == 1 and Vh.size(1) == H:
        Qf = Q.reshape(B * H, 1, _N)
        for j0 in range(0, S, BS):
            j1 = min(j0 + BS, S)
            tile = j1 - j0
            Kj = K[:, :, j0:j1, :].reshape(B * H, tile, _N)
            Vj = Vh[:, :, j0:j1, :].reshape(B * H, tile, D)
            scores = torch.bmm(Qf, Kj.transpose(1, 2))
            out.add_(torch.bmm(scores, Vj).view(B, H, 1, D))
        return out

    for j0 in range(0, S, BS):
        j1 = min(j0 + BS, S)
        Kj = K[:, :, j0:j1, :]
        Vj = Vh[:, :, j0:j1, :]
        # Score tile discarded after ×V — peak ~ Tq×BS, not Tq×S.
        out.add_(_two_gemm_decode(Q, Kj, Vj))
    return out


# Peak score elements for a single tile before the ×V epilogue discards it.
# Used by tests / OPT_NOTES — blocked never allocates a full T×T score tensor.
_STREAM_SCORE_ELEMS = 4096  # if Bi*Bj exceeds this, stream query rows (bound peak)


def max_score_tile_elems(T: int, block_size: int = DEFAULT_BLOCK_COLD) -> int:
    """Upper bound on score elements materialized at once by blocked/online.

    Eager allocates ``T*T`` (full matrix). Vectorized blocked keeps at most:

    * **Past** — one ``Bi × i0`` score tile (or a budget-capped past chunk),
      never ``T × T``. Cap is ``max(BS*BS, _SCORE_ELEMS_BUDGET)``.
    * **Diagonal** — one ``Bi × Bi`` tile with ``tril(diagonal=-1)``.

    For mid ``T`` with default ``BS=64`` this is still ≪ ``T*T`` (e.g. T=256
    → peak ≤ 64·192 or budget, vs 65536).
    """
    BS = max(1, int(block_size))
    if T <= 0:
        return 0
    budget = max(BS * BS, _SCORE_ELEMS_BUDGET)
    # Past: Bi * i0 with i0 < T, capped by budget (tiling kicks in above).
    past = min(BS * max(T - 1, 0), budget)
    diag = BS * BS
    return max(past, diag, max(0, BS - 1))


def max_decode_score_elems(
    S: int,
    Tq: int = 1,
    block_size: int = DEFAULT_BLOCK_DECODE,
    *,
    oneshot_elems: int = _DECODE_ONESHOT_ELEMS,
) -> int:
    """Upper bound on score elements for blocked/online T=1 decode vs packed KR/V.

    Eager decode allocates ``Tq * S``. Blocked/online oneshots while
    ``Tq * S <= oneshot_elems``, else tiles with peak ``Tq * tile`` where
    ``tile = _pick_tile_size(...)``. Never ``(S+Tq)×(S+Tq)``.
    """
    if S <= 0 or Tq <= 0:
        return 0
    budget = max(1, int(oneshot_elems))
    if Tq * S <= budget:
        return Tq * S
    tile = _pick_tile_size(S, Tq, block_size, score_budget=budget)
    return Tq * tile


def _accumulate_qk_v(
    out: torch.Tensor,
    Qi: torch.Tensor,
    Kj: torch.Tensor,
    Vj: torch.Tensor,
    *,
    i0: int,
    i1: int,
) -> None:
    """Online fuse: ``out[:,:,i0:i1] += (Qi @ Kj.mT) @ Vj``.

    Computes ``sum_j (Qi · Kj) Vj`` for the tile and accumulates into ``out``.
    The score tile is not retained after the ×V epilogue. When ``Bi*Bj`` is
    large, streams one query row at a time so peak score storage is ``O(Bj)``
    rather than ``O(Bi*Bj)`` (still never a full ``T×T``).
    """
    Bi = i1 - i0
    Bj = Kj.size(2)
    if Bi == 0 or Bj == 0:
        return

    if Bi * Bj > _STREAM_SCORE_ELEMS:
        # Stream query rows: peak scores (B,H,1,Bj)
        for r in range(Bi):
            q = Qi[:, :, r : r + 1, :]
            out[:, :, i0 + r : i0 + r + 1, :] = out[
                :, :, i0 + r : i0 + r + 1, :
            ] + (q @ Kj.transpose(-2, -1)) @ Vj
        return

    # Single tile: score ephemeral in the fused expression
    out[:, :, i0:i1, :] = out[:, :, i0:i1, :] + (Qi @ Kj.transpose(-2, -1)) @ Vj


def _accumulate_diag_online(
    out: torch.Tensor,
    Q: torch.Tensor,
    K: torch.Tensor,
    Vh: torch.Tensor,
    *,
    i0: int,
    i1: int,
) -> None:
    """Diagonal block with strict ``tril(diagonal=-1)`` via online rows.

    Retained as a lower-peak-memory alternative to the vectorized ``Bi×Bi``
    ``tril`` used by ``blocked_tril_attn``. For local row ``r``, only keys
    ``j in [i0, i0+r)`` contribute — never materializes a ``Bi×Bi`` score
    matrix (peak ``O(r) <= O(BS)`` per row).
    """
    Bi = i1 - i0
    if Bi <= 1:
        return
    for r in range(1, Bi):
        # Global query index i0+r; within-block keys with j < i0+r
        q = Q[:, :, i0 + r : i0 + r + 1, :]
        Kj = K[:, :, i0 : i0 + r, :]
        Vj = Vh[:, :, i0 : i0 + r, :]
        out[:, :, i0 + r : i0 + r + 1, :] = out[
            :, :, i0 + r : i0 + r + 1, :
        ] + (q @ Kj.transpose(-2, -1)) @ Vj


def blocked_tril_attn(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    block_size: int = DEFAULT_BLOCK_COLD,
) -> torch.Tensor:
    """Pure-PyTorch strict-tril attention without a full T×T score matrix.

    Vectorized block accumulation of ``sum_{j<i} (Q_i·K_j) V_j`` (fewer Python
    tile/row loops than the older online-row path):

    * **Past** — for query block ``[i0,i1)``, one fused
      ``(Qi @ K[:,:,:i0].mT) @ V[:,:,:i0]`` when ``Bi·i0`` fits the score
      budget; otherwise chunked bmm under the same budget. Ephemeral scores
      are ``Bi × chunk``, never ``T × T``.
    * **Diagonal** — ``Bi × Bi`` scores with ``tril(diagonal=-1)`` then ``@ Vi``
      (torch ops, not a Python row loop).

    Semantically identical to ``eager_tril_attn`` (up to fp roundoff on long T).
    Works on CPU and CUDA; used as the non-Triton ``triton`` path fallback and
    as a lower-peak-score-memory alternative to eager.

    Shares ``_expand_v_heads`` / ``DEFAULT_BLOCK_COLD`` with the Triton cold
    launcher so CPU fallback and CUDA tiles stay aligned.

    On CPU this path is measured much faster than the prior Python online-row
    blocked loop for mid ``T``, but still typically slower than eager (which
    pays a full ``T×T``). Default ``BDH_ATTN_IMPL`` remains eager.
    """
    B, H, T, N = Q.shape
    D = V.shape[-1]
    Vh = _expand_v_heads(V, B, H, T, D)

    # Accumulate in a stable float: widen half/bfloat16 to fp32; keep f32/f64
    # so gradcheck / analytic bwd on float64 stay exact-dtype.
    if Q.dtype in (torch.float16, torch.bfloat16):
        acc_dtype = torch.float32
    else:
        acc_dtype = Q.dtype
    Qf = Q.to(dtype=acc_dtype)
    Kf = K.to(dtype=acc_dtype)
    Vhf = Vh.to(dtype=acc_dtype)
    out = torch.zeros(B, H, T, D, device=Q.device, dtype=acc_dtype)
    BS = max(1, int(block_size))
    # Allow a Bi×past score tile up to this many elems before chunking.
    # Still ≪ T×T for mid T (e.g. 64×192 vs 256²).
    score_budget = max(BS * BS, _SCORE_ELEMS_BUDGET)

    for i0 in range(0, T, BS):
        i1 = min(i0 + BS, T)
        Qi = Qf[:, :, i0:i1, :]
        Bi = i1 - i0

        # Past: every key index j < i0 is strictly before all queries in Qi.
        if i0 > 0:
            if Bi * i0 <= score_budget:
                out[:, :, i0:i1, :] = (
                    Qi @ Kf[:, :, :i0, :].transpose(-2, -1)
                ) @ Vhf[:, :, :i0, :]
            else:
                # Chunk past so each score tile has ≤ score_budget elems.
                tile = max(BS, score_budget // max(Bi, 1))
                acc = out.new_zeros(B, H, Bi, D)
                for j0 in range(0, i0, tile):
                    j1 = min(j0 + tile, i0)
                    acc = acc + (
                        Qi @ Kf[:, :, j0:j1, :].transpose(-2, -1)
                    ) @ Vhf[:, :, j0:j1, :]
                out[:, :, i0:i1, :] = acc

        # Diagonal: vectorized Bi×Bi strict lower triangle (no Python row loop).
        if Bi > 1:
            scores = Qi @ Kf[:, :, i0:i1, :].transpose(-2, -1)
            scores = scores.tril(diagonal=-1)
            out[:, :, i0:i1, :] = out[:, :, i0:i1, :] + scores @ Vhf[:, :, i0:i1, :]

    return out.to(dtype=Q.dtype)


def online_tril_attn(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    block_size: int = DEFAULT_BLOCK_COLD,
) -> torch.Tensor:
    """Alias for deepened blocked fusion (explicit name for OPT / benches)."""
    return blocked_tril_attn(Q, K, V, block_size=block_size)


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
        stride_qh,
        stride_qt,
        stride_qn,
        stride_kh,
        stride_kt,
        stride_kn,
        stride_vb,
        stride_vt,
        stride_vd,
        stride_oh,
        stride_ot,
        stride_od,
        T,
        N,
        D,
        H,
        V_BROADCAST: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """Fused strict-tril score×V: out[i] = sum_{j<i} (q_i·k_j) * v_j.

        Grid: (cdiv(T, BLOCK_M), B * H)

        Host pairs this with larger power-of-2 tiles and optional V broadcast
        (``V_BROADCAST``: values indexed by ``bh // H``, staging ``B×T×D`` only).
        """
        pid_m = tl.program_id(0)
        bh = tl.program_id(1)

        start_m = pid_m * BLOCK_M
        offs_m = start_m + tl.arange(0, BLOCK_M)
        mask_m = offs_m < T

        q_bh = Q_ptr + bh * stride_qh
        k_bh = K_ptr + bh * stride_kh
        o_bh = Out_ptr + bh * stride_oh
        if V_BROADCAST:
            v_bh = V_ptr + (bh // H) * stride_vb
        else:
            v_bh = V_ptr + bh * stride_vb

        end_n = start_m + BLOCK_M

        for d0 in range(0, D, BLOCK_D):
            offs_d = d0 + tl.arange(0, BLOCK_D)
            mask_d = offs_d < D
            acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

            for j0 in range(0, end_n, BLOCK_N):
                offs_n = j0 + tl.arange(0, BLOCK_N)
                mask_n = offs_n < T

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

                # Strict lower-triangular: j < i (exclude diagonal)
                causal = offs_n[None, :] < offs_m[:, None]
                qk = tl.where(
                    causal & mask_m[:, None] & mask_n[None, :], qk, 0.0
                )

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
    block_m: int | None = None,
    block_n: int | None = None,
) -> torch.Tensor:
    """Triton fused strict-tril attention (cold / full T). Requires CUDA.

    Falls back to ``blocked_tril_attn`` (online fused, ``DEFAULT_BLOCK_COLD``)
    when Triton cannot run — same helpers as ``BDH_ATTN_IMPL=blocked``.

    Host staging: Q/K contiguous only if needed; V head-broadcast stages
    ``(B,T,D)`` rather than materializing ``(B,H,T,D)``. Tiles from
    ``_pick_triton_cold_tiles`` (aligned with online blocked BS=64).
    """
    if not _can_use_triton(Q):
        return blocked_tril_attn(Q, K, V, block_size=DEFAULT_BLOCK_COLD)

    B, H, T, N = Q.shape
    D = V.shape[-1]
    v_broadcast = bool(V.size(1) == 1 and H != 1)

    Qc = _as_contiguous(Q)
    Kc = _as_contiguous(K)
    out = torch.empty(B, H, T, D, device=Q.device, dtype=Q.dtype)

    Qf = Qc.view(B * H, T, N)
    Kf = Kc.view(B * H, T, N)
    Of = out.view(B * H, T, D)

    if v_broadcast:
        V1 = _as_contiguous(V.squeeze(1))  # (B, T, D) — not B*H*T*D
        V_ptr = V1
        stride_vb, stride_vt, stride_vd = V1.stride(0), V1.stride(1), V1.stride(2)
    else:
        Vh = _as_contiguous(V)
        Vf = Vh.view(B * H, T, D)
        V_ptr = Vf
        stride_vb, stride_vt, stride_vd = Vf.stride(0), Vf.stride(1), Vf.stride(2)

    BLOCK_M, BLOCK_N, BLOCK_D, BLOCK_K = _pick_triton_cold_tiles(
        T, N, D, block_m=block_m, block_n=block_n
    )
    BH = B * H
    grid = (triton.cdiv(max(T, 1), BLOCK_M), max(BH, 1))

    _bdh_attn_fwd_kernel[grid](
        Qf,
        Kf,
        V_ptr,
        Of,
        Qf.stride(0),
        Qf.stride(1),
        Qf.stride(2),
        Kf.stride(0),
        Kf.stride(1),
        Kf.stride(2),
        stride_vb,
        stride_vt,
        stride_vd,
        Of.stride(0),
        Of.stride(1),
        Of.stride(2),
        T,
        N,
        D,
        H,
        V_BROADCAST=1 if v_broadcast else 0,
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

    Decode-online-v2: broadcast-V (CacheManager layout ``(B,1,S,D)``) uses a
    tight ``_DECODE_ONESHOT_ELEMS`` so long ``S`` tiles — lower peak score mem
    than eager ``Tq×S``, and better CPU wall than eager oneshot when ``S`` is
    long. Per-head ``V`` keeps the large oneshot budget (BH-bmm).
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

    # Keep broadcast V as (B,1,S,D) — matmul broadcasts heads; no expand copy.
    # Broadcast-V (generate hot path): tight oneshot → tile long S.
    # Per-head V: large budget → BH-bmm oneshot (decode-mm).
    broadcast_v = bool(V.size(1) == 1 and H != 1)
    oneshot = _DECODE_ONESHOT_ELEMS if broadcast_v else _SCORE_ELEMS_BUDGET
    return _tiled_score_v(Q, K, V, block_size, oneshot_elems=oneshot)


def online_decode_attn(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    block_size: int = DEFAULT_BLOCK_DECODE,
) -> torch.Tensor:
    """Alias for deepened blocked T=1 decode (OPT / bench name)."""
    return blocked_decode_attn(Q, K, V, block_size=block_size)


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
        stride_vb,
        stride_vt,
        stride_vd,
        stride_oh,
        stride_ot,
        stride_od,
        Tq,
        S,
        N,
        D,
        H,
        V_BROADCAST: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """Fused decode: out[t] = sum_{j<S} (q_t·k_j) * v_j (all past keys valid).

        Grid: (Tq, B*H). ``V_BROADCAST`` indexes values by ``bh // H`` so host
        stages ``(B,S,D)`` instead of materializing ``(B,H,S,D)``.
        """
        pid_m = tl.program_id(0)
        bh = tl.program_id(1)

        offs_m = pid_m + tl.arange(0, 1)
        mask_m = offs_m < Tq

        q_bh = Q_ptr + bh * stride_qh
        k_bh = K_ptr + bh * stride_kh
        o_bh = Out_ptr + bh * stride_oh
        if V_BROADCAST:
            v_bh = V_ptr + (bh // H) * stride_vb
        else:
            v_bh = V_ptr + bh * stride_vb

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


def _pick_triton_decode_tiles(
    S: int,
    N: int,
    D: int,
    *,
    block_n: int | None = None,
) -> tuple[int, int, int]:
    """Power-of-2 past/Dk/Dv tiles for fused T=1 decode (decode-mm: up to 256)."""
    def _p2_cap(x: int, lo: int, hi: int) -> int:
        x = max(lo, min(int(x), hi))
        p = 1
        while p < x:
            p <<= 1
        return min(p, hi)

    if block_n is None:
        # Prefer larger past tiles on long packed caches (decode has no causal
        # diagonal — every key is valid). Cap 256 for register pressure.
        if S <= 128:
            want = 64
        elif S <= 512:
            want = 128
        else:
            want = min(256, DEFAULT_BLOCK_DECODE)
        block_n = _p2_cap(min(max(S, 1), want), 16, 256)
    else:
        block_n = _p2_cap(block_n, 16, 256)
    block_d = _p2_cap(D if D > 0 else 1, 16, 64)
    block_k = _p2_cap(N if N > 0 else 1, 16, 64)
    return block_n, block_d, block_k


def triton_decode_attn(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    block_size: int = DEFAULT_BLOCK_DECODE,
    block_n: int | None = None,
) -> torch.Tensor:
    """Decode against past KR/V. Triton fused on CUDA; else blocked.

    On CPU (this box) and whenever Triton cannot run, uses
    ``blocked_decode_attn`` so ``BDH_ATTN_IMPL=triton`` still gets the
    no-full-TxT decode path. Dedicated decode kernel skips causal masking
    (past keys are all valid under tril(-1)) and supports broadcast-V
    staging (``(B,S,D)`` when ``V`` is ``(B,1,S,D)``).
    """
    if not _can_use_triton(Q):
        return blocked_decode_attn(Q, K, V, block_size=block_size)

    B, H, Tq, N = Q.shape
    S = K.size(2)
    D = V.size(-1)
    if S == 0:
        return Q.new_zeros(B, H, Tq, D)

    v_broadcast = bool(V.size(1) == 1 and H != 1)
    Qc = _as_contiguous(Q)
    Kc = _as_contiguous(K)
    out = torch.empty(B, H, Tq, D, device=Q.device, dtype=Q.dtype)

    Qf = Qc.view(B * H, Tq, N)
    Kf = Kc.view(B * H, S, N)
    Of = out.view(B * H, Tq, D)

    if v_broadcast:
        V1 = _as_contiguous(V.squeeze(1))  # (B, S, D) — not B*H*S*D
        V_ptr = V1
        stride_vb, stride_vt, stride_vd = V1.stride(0), V1.stride(1), V1.stride(2)
    else:
        Vh = _as_contiguous(V)
        Vf = Vh.view(B * H, S, D)
        V_ptr = Vf
        stride_vb, stride_vt, stride_vd = Vf.stride(0), Vf.stride(1), Vf.stride(2)

    BH = B * H
    BLOCK_N, BLOCK_D, BLOCK_K = _pick_triton_decode_tiles(
        S, N, D, block_n=block_n
    )
    grid = (Tq, BH)
    _bdh_decode_fwd_kernel[grid](
        Qf,
        Kf,
        V_ptr,
        Of,
        Qf.stride(0),
        Qf.stride(1),
        Qf.stride(2),
        Kf.stride(0),
        Kf.stride(1),
        Kf.stride(2),
        stride_vb,
        stride_vt,
        stride_vd,
        Of.stride(0),
        Of.stride(1),
        Of.stride(2),
        Tq,
        S,
        N,
        D,
        H,
        V_BROADCAST=1 if v_broadcast else 0,
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
    """Reference decode: ``(Q @ K.mT) @ V`` (past-only; no self-attention).

    Delegates to ``_two_gemm_decode`` (Tq=1 BH-bmm when V is per-head; else
    4D matmul with broadcast-V — no expand).
    """
    return _two_gemm_decode(Q, K, V)
