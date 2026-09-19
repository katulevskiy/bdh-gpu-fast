"""Strict lower-triangular score×V for BDH attention.

Semantics (must match bdh.Attention cold path)::

    scores = (Q @ K.transpose(-2, -1)).tril(diagonal=-1)
    out    = scores @ V

No softmax, no 1/sqrt(d), diagonal EXCLUDED.

Decode (packed past KR/V — matches incremental / CacheManager hot path)::

    out = (Q @ K_past.transpose(-2, -1)) @ V_past

Past slices exclude the new token, so the query never attends to itself
(same as ``tril(diagonal=-1)``).

Public API
----------
tril_score_v_ref(q, k, v)         — golden eager full tril (bit-identical)
tril_score_v_tiled_ref(q, k, v)   — CPU mirror of CUDA tiled online cold
tril_score_v(q, k, v)             — native ext if built+compatible, else tiled/eager ref
tril_decode_ref(q, k_past, v)     — pure PyTorch decode (vectorized; tiles large S)
tril_decode_tiled_ref(...)        — CPU mirror of CUDA tiled online decode
tril_decode(q, k_past, v)         — native ext if built, else decode ref

Optional native build (``csrc/``)::

    pip install -e .                                 # pure Python (default)
    BDH_BUILD_EXT=1 pip install -e . --no-build-isolation
    BDH_BUILD_EXT=1 BDH_BUILD_CUDA=1 pip install -e . --no-build-isolation

Cold CUDA kernel (``tril_score_v_cuda``) uses **tiled online** accumulation
(shared-mem Q/K/V tiles; no global T×T scores) with **adaptive** query/key
tiles (``pick_cuda_cold_tiles``; base ``TILE_M/N=16``, long-T up to 32/64 on
CUDA smem / 128 on CPU refs — pairs ``#75`` ``pick_cold_block_size`` @T≥256).
Decode is a separate packed-past tiled scaffold with **adaptive past tiles**
(``pick_cuda_decode_tile_n``; base ``DECODE_TILE_N=32``, long-S up to 128 on
CUDA / 512 on CPU refs) and a dedicated ``Tq=1`` kernel. Without the
extension, training and tests still work via pure-PyTorch refs. Import of
this module **never** raises if the native ext is missing —
``has_cuda_ext()`` is False and dispatch falls back.

Wire-up: ``BDH_ATTN_IMPL=cuda`` → cold ``bdh_attn`` / decode ``bdh_attn_decode``.
Default remains ``eager``.

Tile sizes below match ``csrc/tril_attn_cuda.cu`` (TILE_M / TILE_N / TILE_D;
cold adaptive via ``pick_cuda_cold_tiles``).
"""

from __future__ import annotations

from typing import Optional

import torch

# Match csrc/tril_attn_cuda.cu — CPU tiled refs mirror these for drop-in parity.
CUDA_TILE_M = 16  # base query rows per tile (cold; back-compat)
CUDA_TILE_N = 16  # base key cols per online tile (cold; back-compat)
CUDA_TILE_D = 32  # Dv columns (CUDA blockDim.x; CPU uses full Dv via matmul)
# cuda-cold-v2: adaptive long-T cold tiles (pair #75 pick_cold_block_size @T≥256).
CUDA_TILE_M_MAX = 32  # CUDA smem-aware query-tile cap (blockDim.y)
CUDA_TILE_N_MAX = 64  # CUDA smem-aware key-tile cap (float Dk≈64 fits ≤48 KiB)
CUDA_TILE_M_CPU_MAX = 128  # CPU refs: same long-T preference as #75 BS=128
CUDA_TILE_N_CPU_MAX = 128
# Decode past tiles: base / max (no causal diagonal — all keys valid).
# cuda-decode-v3: adaptive long-S tiles (pair #55/#62); CUDA smem caps at MAX.
CUDA_DECODE_TILE_N = 32  # base / mid-S default (kept for back-compat asserts)
CUDA_DECODE_TILE_N_MAX = 128  # CUDA smem-aware cap (float Dk≈64 fits ≤48 KiB)
CUDA_DECODE_TILE_N_CPU_MAX = 512  # CPU refs: same long-S preference as Triton #62
# Soft smem budget matching csrc SMEM_CAP (bytes) for float-size estimates.
_CUDA_DECODE_SMEM_CAP = 48 * 1024
_CUDA_COLD_SMEM_CAP = 48 * 1024

# Soft budget: below this, cold/decode refs prefer a single vectorized two-GEMM
# (bit-identical to eager). Above it, use tiled online to avoid a full T×T /
# Tq×S score materialization — same structure the CUDA scaffold will run.
_SCORE_ELEMS_EAGER_OK = 256 * 256
# cuda-cold-v2 / #75: long-T oneshot past budget (Bi × past) before key-chunking.
_COLD_SCORE_ELEMS_BUDGET = 256 * 256

_ext = None
_ext_load_error: Optional[BaseException] = None

try:
    import bdh_cuda_ext as _ext  # type: ignore
except Exception as exc:  # ImportError or missing symbols — soft fail
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


def _check_cold_shapes(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> None:
    if q.dim() != 4 or k.dim() != 4 or v.dim() != 4:
        raise ValueError("q, k, v must be 4D (B, H, T, D)")
    if q.shape != k.shape:
        raise ValueError(f"q and k shapes must match, got {tuple(q.shape)} vs {tuple(k.shape)}")
    if v.size(0) != q.size(0) or v.size(2) != q.size(2):
        raise ValueError("v batch/seq must match q")
    if v.size(1) not in (1, q.size(1)):
        raise ValueError("v heads must equal q heads or 1")


def _check_decode_shapes(
    q: torch.Tensor, k_past: torch.Tensor, v_past: torch.Tensor
) -> tuple[int, int, int, int, int]:
    if q.dim() != 4 or k_past.dim() != 4 or v_past.dim() != 4:
        raise ValueError("q, k_past, v_past must be 4D")
    B, H, Tq, Dk = q.shape
    S = k_past.size(2)
    if k_past.shape[:2] != (B, H) or k_past.size(-1) != Dk:
        raise ValueError(
            f"k_past shape {tuple(k_past.shape)} incompatible with q {tuple(q.shape)}"
        )
    if v_past.size(0) != B or v_past.size(2) != S:
        raise ValueError(
            f"v_past shape {tuple(v_past.shape)} incompatible with k_past S={S}"
        )
    if v_past.size(1) not in (1, H):
        raise ValueError("v_past heads must equal q heads or 1")
    return B, H, Tq, S, v_past.size(-1)


def pick_cuda_cold_tiles(
    T: int,
    Dk: int = 64,
    *,
    Dv: int | None = None,
    for_smem: bool = False,
    tile_m: int | None = None,
    tile_n: int | None = None,
) -> tuple[int, int]:
    """Power-of-2 query/key tiles for CUDA cold tril score×V (long-T deepen).

    Pairs with ``#75`` ``pick_cold_block_size`` (BS 64→128 at ``T≥256``) and
    Triton ``_pick_triton_cold_tiles``: larger tiles on long prefill so fewer
    online trips while peak scores stay ≪ ``T×T``. CUDA launch uses
    ``for_smem=True`` (cap ``CUDA_TILE_M/N_MAX``, shrink for 48 KiB smem).
    CPU tiled refs use ``for_smem=False`` so long-T can reach 128 like blocked.
    When ``Dk > 64`` or ``Dv > 128``, default long-T tiles stay at the
    mid-size width (32/32 for CUDA, 64/64 for CPU), mirroring #108's
    wide-head Triton guard. Explicit overrides still win.
    """
    def _p2_cap(x: int, lo: int, hi: int) -> int:
        x = max(lo, min(int(x), hi))
        p = lo
        while p < x:
            nxt = p << 1
            if nxt > hi:
                break
            p = nxt
        return p

    hi_m = CUDA_TILE_M_MAX if for_smem else CUDA_TILE_M_CPU_MAX
    hi_n = CUDA_TILE_N_MAX if for_smem else CUDA_TILE_N_CPU_MAX
    lo = 16
    T = max(int(T), 1)
    wide_head = int(Dk) > 64 or (Dv is not None and int(Dv) > 128)

    if tile_m is not None:
        tm = _p2_cap(int(tile_m), lo, hi_m)
    else:
        # Mirror #75: base 16 for short; grow at mid/long T (fewer query tiles).
        if T <= 64:
            want_m = 16
        elif T <= 256:
            want_m = 32
        else:
            want_m = 64 if for_smem else 128
            if wide_head:
                want_m = min(want_m, 32 if for_smem else 64)
        tm = _p2_cap(min(T, want_m), lo, hi_m)

    if tile_n is not None:
        tn = _p2_cap(int(tile_n), lo, hi_n)
    else:
        if T <= 64:
            want_n = 16
        elif T <= 256:
            want_n = 32
        else:
            want_n = 64 if for_smem else 128
            if wide_head:
                want_n = min(want_n, 32 if for_smem else 64)
        tn = _p2_cap(min(T, want_n), lo, hi_n)

    if for_smem:
        Dk = max(int(Dk), 1)
        esz = 4

        def _smem(tm_: int, tn_: int) -> int:
            # Qs[TM*Dk] + Ks[TN*Dk] + Vs[TN*TILE_D]
            return esz * (tm_ * Dk + tn_ * Dk + tn_ * CUDA_TILE_D)

        while (tm > CUDA_TILE_M or tn > CUDA_TILE_N) and _smem(tm, tn) > _CUDA_COLD_SMEM_CAP:
            if tn > tm and tn > CUDA_TILE_N:
                tn >>= 1
            elif tm > CUDA_TILE_M:
                tm >>= 1
            elif tn > CUDA_TILE_N:
                tn >>= 1
            else:
                break
        tm = min(tm, CUDA_TILE_M_MAX)
        tn = min(tn, CUDA_TILE_N_MAX)
    return max(lo, tm), max(lo, tn)


def _cold_score_budget(T: int, BS: int) -> int:
    """Score-elem budget for one past tile (never a full ``T×T``).

    Mirrors ``kernels.attention._cold_score_budget`` (#75): base
    ``max(BS², _COLD_SCORE_ELEMS_BUDGET)``; for ``T≥256`` allow a larger
    oneshot up to ``BS * min(T, 1024)``.
    """
    budget = max(BS * BS, _COLD_SCORE_ELEMS_BUDGET)
    if int(T) >= 256:
        budget = max(budget, BS * min(int(T), 1024))
    return budget


def pick_cuda_decode_tile_n(
    S: int,
    Dk: int = 64,
    *,
    for_smem: bool = False,
    tile_n: int | None = None,
) -> int:
    """Power-of-2 past tiles for CUDA T=1 decode (long-S: up to 128 GPU / 512 CPU).

    Pairs with #55 blocked long-S wins and #62 Triton tile picker: larger past
    tiles on long packed caches (decode has no causal diagonal). CUDA launch
    uses ``for_smem=True`` (cap ``CUDA_DECODE_TILE_N_MAX``, shrink for 48 KiB
    smem). CPU tiled refs use ``for_smem=False`` so long-S can reach 512 like
    Triton while still matching blocked/eager numerically.
    """
    def _p2_cap(x: int, lo: int, hi: int) -> int:
        x = max(lo, min(int(x), hi))
        p = lo
        while p < x:
            nxt = p << 1
            if nxt > hi:
                break
            p = nxt
        return p

    hi = CUDA_DECODE_TILE_N_MAX if for_smem else CUDA_DECODE_TILE_N_CPU_MAX
    lo = 16
    if tile_n is not None:
        tn = _p2_cap(int(tile_n), lo, hi)
    else:
        S = max(int(S), 1)
        # Mirror #62 _pick_triton_decode_tiles long-S preference (CPU refs).
        if S <= 64:
            want = 32
        elif S <= 256:
            want = 64
        elif S <= 1024:
            want = 128
        elif S <= 2048:
            want = 256
        else:
            want = 512
        tn = _p2_cap(min(S, want), lo, hi)

    if for_smem:
        Dk = max(int(Dk), 1)
        # float-sized estimate: Qs[Dk] + Ks[TN*Dk] + Vs[TN*TILE_D]
        esz = 4

        def _smem(tn: int) -> int:
            return esz * (Dk + tn * Dk + tn * CUDA_TILE_D)

        while tn > CUDA_DECODE_TILE_N and _smem(tn) > _CUDA_DECODE_SMEM_CAP:
            tn >>= 1
        tn = max(tn, CUDA_DECODE_TILE_N) if _smem(CUDA_DECODE_TILE_N) <= _CUDA_DECODE_SMEM_CAP else max(16, tn)
        # Final clamp: never exceed MAX; if even base overflows, caller falls naive.
        tn = min(tn, CUDA_DECODE_TILE_N_MAX)
    return max(lo, tn)



def tril_score_v_ref(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Golden eager reference: always bit-identical to ``(Q@K.T).tril(-1)@V``.

    Shapes
    ------
    q, k : (B, H, T, Dk)
    v    : (B, H, T, Dv) or (B, 1, T, Dv)  — head dim broadcasts like bdh.Attention
    out  : (B, H, T, Dv)

    Materializes a T×T score matrix (fine for correctness / small T). For a
    CUDA-structure CPU path without full T×T, use ``tril_score_v_tiled_ref``.
    V stays broadcast ``(B,1,T,Dv)`` — no expand copy into ``(B,H,T,Dv)``.
    """
    _check_cold_shapes(q, k, v)
    # Vectorized two-GEMM + tril — no expand of V (matmul broadcasts heads).
    scores = q @ k.transpose(-2, -1)
    scores = scores.tril(diagonal=-1)
    return scores @ v


def tril_score_v_tiled_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    tile_m: int | None = None,
    tile_n: int | None = None,
) -> torch.Tensor:
    """CPU mirror of the CUDA **tiled online** cold kernel (no global T×T).

    Matches ``csrc/tril_attn_cuda.cu`` tile structure (cuda-cold-v2 adaptive):

    * query tiles of ``tile_m`` (default via ``pick_cuda_cold_tiles``;
      base 16, long-T up to 128 on CPU — pair ``#75`` BS)
    * key tiles of ``tile_n`` (same picker; past region may oneshot under
      ``_cold_score_budget`` like blocked)
    * past tiles: all ``j < i0`` for query block ``[i0, i0+tile_m)``
    * diagonal: ``Bi×Bi`` with ``tril(diagonal=-1)`` then ``@ Vi`` (vectorized)

    Accumulates ``sum_{j<i} (Q_i·K_j) V_j`` in fp32 for half/bf16 inputs.
    Bit-close to ``tril_score_v_ref`` / ``blocked_tril_attn`` (same math; tile
    order may differ in ulps on long T). V broadcast without expand.
    """
    _check_cold_shapes(q, k, v)
    B, H, T, Dk = q.shape
    Dv = v.size(-1)

    if T == 0:
        return q.new_zeros(B, H, 0, Dv)

    TM, TN = pick_cuda_cold_tiles(
        T, Dk, Dv=Dv, tile_m=tile_m, tile_n=tile_n
    )
    score_budget = _cold_score_budget(T, TM)

    # Acc dtype: widen half/bf16 like the CUDA kernel's float accumulators.
    if q.dtype in (torch.float16, torch.bfloat16):
        acc_dtype = torch.float32
    else:
        acc_dtype = q.dtype
    # Skip .to() when already in acc dtype (hot fp32 path).
    Qf = q if q.dtype == acc_dtype else q.to(dtype=acc_dtype)
    Kf = k if k.dtype == acc_dtype else k.to(dtype=acc_dtype)
    # Keep V as (B,1|H,T,Dv) — matmul broadcasts; no expand staging.
    Vf = v if v.dtype == acc_dtype else v.to(dtype=acc_dtype)
    out = torch.zeros(B, H, T, Dv, device=q.device, dtype=acc_dtype)

    for i0 in range(0, T, TM):
        i1 = min(i0 + TM, T)
        Bi = i1 - i0
        Qi = Qf[:, :, i0:i1, :]

        # Past: oneshot fused when Bi·i0 fits budget (pair #75); else chunk.
        if i0 > 0:
            if Bi * i0 <= score_budget:
                out[:, :, i0:i1, :].add_(
                    (Qi @ Kf[:, :, :i0, :].transpose(-2, -1)) @ Vf[:, :, :i0, :]
                )
            else:
                # Chunk past so each score tile has ≤ score_budget elems.
                # Prefer TN (CUDA key tile) but grow to budget//Bi like blocked.
                tile = max(TN, score_budget // max(Bi, 1))
                for j0 in range(0, i0, tile):
                    j1 = min(j0 + tile, i0)
                    out[:, :, i0:i1, :].add_(
                        (Qi @ Kf[:, :, j0:j1, :].transpose(-2, -1))
                        @ Vf[:, :, j0:j1, :]
                    )

        # Diagonal tile: vectorized Bi×Bi with tril(diagonal=-1) — not a row loop.
        if Bi > 1:
            scores = Qi @ Kf[:, :, i0:i1, :].transpose(-2, -1)
            scores = scores.tril(diagonal=-1)
            out[:, :, i0:i1, :].add_(scores @ Vf[:, :, i0:i1, :])

    return out.to(dtype=q.dtype) if out.dtype != q.dtype else out


def tril_score_v(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Dispatch: native CUDA/C++ ext when available and tensors match, else ref.

    When the native extension is missing, prefers the **tiled online** CPU ref
    for large T (mirrors CUDA scaffold; avoids full T×T staging). Small T uses
    the golden eager ref (bit-identical, fewer Python trips).
    """
    # Validate before optional native dispatch so malformed inputs have the
    # same CPU-safe contract regardless of extension availability.
    _check_cold_shapes(q, k, v)
    if _ext is not None:
        try:
            if q.is_cuda and has_cuda_kernel():
                return _ext.tril_score_v(q, k, v)
            if not q.is_cuda:
                return _ext.tril_score_v(q, k, v)
        except Exception:
            # Fall through to reference on any runtime failure.
            pass
    # Soft fallback — no native: choose tiled vs eager by score footprint.
    T = q.size(2) if q.dim() == 4 else 0
    if T * T > _SCORE_ELEMS_EAGER_OK:
        return tril_score_v_tiled_ref(q, k, v)
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

    Small ``Tq×S``: single vectorized two-GEMM (bit-identical to eager decode).
    Large past: tiles over ``pick_cuda_decode_tile_n`` key chunks (CUDA-mirror;
    no full ``Tq×S`` retained). V broadcast without expand.
    """
    B, H, Tq, S, Dv = _check_decode_shapes(q, k_past, v_past)
    if S == 0:
        return q.new_zeros(B, H, Tq, Dv)

    # Modest past: one fused two-GEMM (same as eager_decode_attn / _two_gemm_decode).
    if Tq * S <= _SCORE_ELEMS_EAGER_OK:
        from .attention import _two_gemm_decode

        return _two_gemm_decode(q, k_past, v_past)

    return tril_decode_tiled_ref(q, k_past, v_past)


def tril_decode_tiled_ref(
    q: torch.Tensor,
    k_past: torch.Tensor,
    v_past: torch.Tensor,
    *,
    tile_n: int | None = None,
) -> torch.Tensor:
    """CPU mirror of the CUDA **tiled online** decode kernel (no Tq×S retained).

    Tiles the past axis in chunks of ``tile_n``. When omitted, uses
    ``pick_cuda_decode_tile_n(S, Dk)`` (cuda-decode-v3: long-S up to 512 on
    CPU — same preference as #62 Triton / #55 blocked). Accumulates with
    ``out.add_`` via ``_two_gemm_decode`` — same structure as
    ``tril_decode_tiled_kernel`` / ``tril_decode_tq1_kernel`` in
    ``csrc/tril_attn_cuda.cu``. Numerically matches ``blocked_decode_attn``.
    """
    from .attention import _two_gemm_decode

    B, H, Tq, S, Dv = _check_decode_shapes(q, k_past, v_past)
    if S == 0:
        return q.new_zeros(B, H, Tq, Dv)

    Dk = q.size(-1)
    TN = pick_cuda_decode_tile_n(S, Dk, tile_n=tile_n)
    if q.dtype in (torch.float16, torch.bfloat16):
        acc_dtype = torch.float32
    else:
        acc_dtype = q.dtype
    Qf = q.to(dtype=acc_dtype)
    Kf = k_past.to(dtype=acc_dtype)
    Vf = v_past.to(dtype=acc_dtype)
    out = torch.zeros(B, H, Tq, Dv, device=q.device, dtype=acc_dtype)

    for j0 in range(0, S, TN):
        j1 = min(j0 + TN, S)
        out.add_(_two_gemm_decode(Qf, Kf[:, :, j0:j1, :], Vf[:, :, j0:j1, :]))

    return out.to(dtype=q.dtype)


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
    """Human-readable status for logs / OPT_NOTES / build smoke."""
    if _ext is None:
        err = _ext_load_error
        detail = f"{type(err).__name__}: {err}" if err is not None else "not attempted"
        return f"native=unavailable ({detail})"
    cuda = "yes" if has_cuda_kernel() else "no"
    decode = "yes" if hasattr(_ext, "tril_decode") else "no"
    return f"native=loaded has_cuda_kernel={cuda} has_decode={decode}"
