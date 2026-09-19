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
(shared-mem Q/K/V tiles; no global T×T scores). Decode is a separate
packed-past tiled scaffold. Without the extension, training and tests still
work via pure-PyTorch refs. Import of this module **never** raises if the
native ext is missing — ``has_cuda_ext()`` is False and dispatch falls back.

Wire-up: ``BDH_ATTN_IMPL=cuda`` → cold ``bdh_attn`` / decode ``bdh_attn_decode``.
Default remains ``eager``.

Tile sizes below match ``csrc/tril_attn_cuda.cu`` (TILE_M / TILE_N / TILE_D).
"""

from __future__ import annotations

from typing import Optional

import torch

# Match csrc/tril_attn_cuda.cu — CPU tiled refs mirror these for drop-in parity.
CUDA_TILE_M = 16  # query rows per tile
CUDA_TILE_N = 16  # key cols per online tile
CUDA_TILE_D = 32  # Dv columns (CUDA blockDim.x; CPU uses full Dv via matmul)

# Soft budget: below this, cold/decode refs prefer a single vectorized two-GEMM
# (bit-identical to eager). Above it, use tiled online to avoid a full T×T /
# Tq×S score materialization — same structure the CUDA scaffold will run.
_SCORE_ELEMS_EAGER_OK = 256 * 256

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
    tile_m: int = CUDA_TILE_M,
    tile_n: int = CUDA_TILE_N,
) -> torch.Tensor:
    """CPU mirror of the CUDA **tiled online** cold kernel (no global T×T).

    Matches ``csrc/tril_attn_cuda.cu`` tile structure:

    * query tiles of ``tile_m`` (default 16 = CUDA TILE_M)
    * key tiles of ``tile_n`` (default 16 = CUDA TILE_N)
    * past tiles: all ``j < i0`` for query block ``[i0, i0+tile_m)``
    * diagonal: row-wise online strict lower triangle (``j < i`` only)

    Accumulates ``sum_{j<i} (Q_i·K_j) V_j`` in fp32 for half/bf16 inputs.
    Bit-close to ``tril_score_v_ref`` / eager (same math; tile order may differ
    in ulps on long T). V broadcast without expand.
    """
    _check_cold_shapes(q, k, v)
    B, H, T, Dk = q.shape
    Dv = v.size(-1)
    TM = max(1, int(tile_m))
    TN = max(1, int(tile_n))

    if T == 0:
        return q.new_zeros(B, H, 0, Dv)

    # Acc dtype: widen half/bf16 like the CUDA kernel's float accumulators.
    if q.dtype in (torch.float16, torch.bfloat16):
        acc_dtype = torch.float32
    else:
        acc_dtype = q.dtype
    Qf = q.to(dtype=acc_dtype)
    Kf = k.to(dtype=acc_dtype)
    # Keep V as (B,1|H,T,Dv) — matmul broadcasts; no expand staging.
    Vf = v.to(dtype=acc_dtype)
    out = torch.zeros(B, H, T, Dv, device=q.device, dtype=acc_dtype)

    for i0 in range(0, T, TM):
        i1 = min(i0 + TM, T)
        Qi = Qf[:, :, i0:i1, :]

        # Past key tiles: every j < i0 is strictly before all queries in [i0,i1).
        for j0 in range(0, i0, TN):
            j1 = min(j0 + TN, i0)
            Kj = Kf[:, :, j0:j1, :]
            Vj = Vf[:, :, j0:j1, :]
            # Ephemeral (Bi × Bj) score discarded after ×V — never full T×T.
            out[:, :, i0:i1, :] = out[:, :, i0:i1, :] + (Qi @ Kj.transpose(-2, -1)) @ Vj

        # Diagonal tile: online rows with j < i inside the block (tril -1).
        Bi = i1 - i0
        if Bi > 1:
            for r in range(1, Bi):
                qi = Qf[:, :, i0 + r : i0 + r + 1, :]
                Kj = Kf[:, :, i0 : i0 + r, :]
                Vj = Vf[:, :, i0 : i0 + r, :]
                out[:, :, i0 + r : i0 + r + 1, :] = (
                    out[:, :, i0 + r : i0 + r + 1, :]
                    + (qi @ Kj.transpose(-2, -1)) @ Vj
                )

    return out.to(dtype=q.dtype)


def tril_score_v(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Dispatch: native CUDA/C++ ext when available and tensors match, else ref.

    When the native extension is missing, prefers the **tiled online** CPU ref
    for large T (mirrors CUDA scaffold; avoids full T×T staging). Small T uses
    the golden eager ref (bit-identical, fewer Python trips).
    """
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
    Large past: tiles over ``CUDA_TILE_N`` key chunks (CUDA-mirror; no full
    ``Tq×S`` retained). V broadcast without expand.
    """
    B, H, Tq, S, Dv = _check_decode_shapes(q, k_past, v_past)
    if S == 0:
        return q.new_zeros(B, H, Tq, Dv)

    # Modest past: one fused two-GEMM (same as eager_decode_attn).
    if Tq * S <= _SCORE_ELEMS_EAGER_OK:
        return (q @ k_past.transpose(-2, -1)) @ v_past

    return tril_decode_tiled_ref(q, k_past, v_past)


def tril_decode_tiled_ref(
    q: torch.Tensor,
    k_past: torch.Tensor,
    v_past: torch.Tensor,
    *,
    tile_n: int = CUDA_TILE_N,
) -> torch.Tensor:
    """CPU mirror of the CUDA **tiled online** decode kernel (no Tq×S retained).

    Tiles the past axis in chunks of ``tile_n`` (default 16 = CUDA TILE_N),
    accumulating ``(Q @ K_tile.mT) @ V_tile`` — same structure as
    ``tril_decode_tiled_kernel`` in ``csrc/tril_attn_cuda.cu``.
    """
    B, H, Tq, S, Dv = _check_decode_shapes(q, k_past, v_past)
    if S == 0:
        return q.new_zeros(B, H, Tq, Dv)

    TN = max(1, int(tile_n))
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
        Kj = Kf[:, :, j0:j1, :]
        Vj = Vf[:, :, j0:j1, :]
        out = out + (Qf @ Kj.transpose(-2, -1)) @ Vj

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
