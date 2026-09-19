"""RoPE rotate: eager (strided) vs fused (pair-contiguous / optional Triton).

BDH RoPE is *not* classic half-dim shared-(cos,sin). Each even/odd index has
its own phase from ``freqs`` of length ``N``:

    y0 = x0 * c0 - x1 * s0
    y1 = x1 * c1 + x0 * s1

``rope_cos_sin`` caching stays in ``bdh.Attention``; this module only applies
an already-computed (cos, sin) table to ``v``.
"""

from __future__ import annotations

from typing import Optional

import torch

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except Exception:  # pragma: no cover - import guard
    triton = None  # type: ignore
    tl = None  # type: ignore
    _HAS_TRITON = False


def _validate_rope_inputs(
    v: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    out: Optional[torch.Tensor],
) -> None:
    if v.shape[-1] % 2 != 0:
        raise ValueError(f"RoPE last dim must be even, got {v.shape[-1]}")
    if cos.shape != sin.shape:
        raise ValueError(f"cos/sin shape mismatch: {cos.shape} vs {sin.shape}")
    if cos.shape[-1] != v.shape[-1]:
        raise ValueError(
            f"cos/sin last dim {cos.shape[-1]} != v last dim {v.shape[-1]}"
        )
    if out is not None:
        # dtype may differ: CacheManager.reserve can be fp32 storage while AMP
        # compute is fp16/bf16 — historical rope wrote via assignment cast.
        if out.shape != v.shape or out.device != v.device:
            raise ValueError(
                f"rope out= must match v shape/device, got out={tuple(out.shape)}/"
                f"{out.device} vs v={tuple(v.shape)}/{v.device}"
            )
        if out is v:
            raise ValueError("rope out= must not alias v")


def _validate_paired_cis(
    v: torch.Tensor,
    cos_p: torch.Tensor,
    sin_p: torch.Tensor,
    out: Optional[torch.Tensor],
) -> None:
    """cos_p/sin_p are pair views: ``(..., N/2, 2)`` with N = v.shape[-1]."""
    if v.shape[-1] % 2 != 0:
        raise ValueError(f"RoPE last dim must be even, got {v.shape[-1]}")
    n_pairs = v.shape[-1] // 2
    if cos_p.shape != sin_p.shape:
        raise ValueError(f"paired cos/sin shape mismatch: {cos_p.shape} vs {sin_p.shape}")
    if cos_p.shape[-2:] != (n_pairs, 2):
        raise ValueError(
            f"paired cis trailing dims must be ({n_pairs}, 2), got {cos_p.shape[-2:]}"
        )
    if out is not None:
        if out.shape != v.shape or out.device != v.device:
            raise ValueError(
                f"rope out= must match v shape/device, got out={tuple(out.shape)}/"
                f"{out.device} vs v={tuple(v.shape)}/{v.device}"
            )
        if out is v:
            raise ValueError("rope out= must not alias v")


def _is_t1_seq(v: torch.Tensor) -> bool:
    """True when the sequence axis (dim -2) is length 1 — incremental decode."""
    return v.dim() >= 2 and v.shape[-2] == 1


def _rotate_pair_views(
    vp: torch.Tensor,
    cp: torch.Tensor,
    sp: torch.Tensor,
    *,
    v_dtype: torch.dtype,
    cis_dtype: torch.dtype,
):
    """Compute (y0, y1) from pair views; cis may broadcast over vp leading dims."""
    x0, x1 = vp[..., 0], vp[..., 1]
    c0, c1 = cp[..., 0], cp[..., 1]
    s0, s1 = sp[..., 0], sp[..., 1]
    if v_dtype != cis_dtype:
        y0 = (x0 * c0).to(v_dtype) + ((-x1) * s0).to(v_dtype)
        y1 = (x1 * c1).to(v_dtype) + (x0 * s1).to(v_dtype)
    else:
        y0 = x0 * c0 - x1 * s0
        y1 = x1 * c1 + x0 * s1
    return y0, y1


def _store_pairs(
    y0: torch.Tensor,
    y1: torch.Tensor,
    v: torch.Tensor,
    out: Optional[torch.Tensor],
) -> torch.Tensor:
    """Write interleaved pairs into ``out`` (or a fresh buffer) — no ``stack``.

    Pair-axis stores are contiguous when ``out``'s last dim is contiguous; avoids
    the extra ``stack→reshape`` temporary that previously taxed fuse/T=1 alloc.
    """
    if out is None:
        out = torch.empty_like(v)
    op = out.reshape(*v.shape[:-1], -1, 2)
    op[..., 0] = y0
    op[..., 1] = y1
    return out


def rope_rotate_paired(
    v: torch.Tensor,
    cos_p: torch.Tensor,
    sin_p: torch.Tensor,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Rotate using already-paired cis ``(..., N/2, 2)`` — no cis reshape tax.

    Used by T=1 decode when ``Attention`` narrows ``_rope_table_pairs`` (built
    once in ``ensure_rope_table``). Same math as ``eager_rope_rotate``.
    """
    _validate_paired_cis(v, cos_p, sin_p, out)
    vp = v.reshape(*v.shape[:-1], -1, 2)
    y0, y1 = _rotate_pair_views(
        vp, cos_p, sin_p, v_dtype=v.dtype, cis_dtype=cos_p.dtype
    )
    return _store_pairs(y0, y1, v, out)


def rope_rotate_t1(
    v: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """T=1 decode rotate: pair-contiguous writes, no strided ``0::2`` stores.

    Cos/sin are typically ``(1,1,1,N)`` narrows from the generate table. Pair
    reshape lets mul broadcast without ``expand(v.shape)`` materialization.
    ``out=None`` uses empty + pair store (no ``stack``). Same math as
    ``eager_rope_rotate`` (bit-identical on CPU fp32 / cast path).
    """
    _validate_rope_inputs(v, cos, sin, out)
    if not _is_t1_seq(v):
        raise ValueError(
            f"rope_rotate_t1 expects sequence length 1, got shape {tuple(v.shape)}"
        )

    # (..., 1, N) → (..., 1, N/2, 2). Cos/sin keep their leading 1s; mul broadcasts.
    vp = v.reshape(*v.shape[:-1], -1, 2)
    cp = cos.reshape(*cos.shape[:-1], -1, 2)
    sp = sin.reshape(*sin.shape[:-1], -1, 2)
    y0, y1 = _rotate_pair_views(vp, cp, sp, v_dtype=v.dtype, cis_dtype=cos.dtype)
    return _store_pairs(y0, y1, v, out)


def eager_rope_rotate(
    v: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Reference rotate: strided even/odd slices (matches historical ``Attention.rope``).

    When ``v`` has sequence length 1 (decode), uses ``rope_rotate_t1`` — same
    math, pair-contiguous stores into ``out`` / empty+pair store when ``out`` is
    None. Multi-token keeps the historical strided path.
    """
    if _is_t1_seq(v):
        return rope_rotate_t1(v, cos, sin, out=out)
    _validate_rope_inputs(v, cos, sin, out)
    if out is None:
        out = torch.empty_like(v)

    ve = v[..., 0::2]
    vo = v[..., 1::2]
    ce = cos[..., 0::2]
    se = sin[..., 0::2]
    co = cos[..., 1::2]
    so = sin[..., 1::2]

    if v.dtype != cos.dtype:
        # Match baseline cast-then-add rounding when phases are fp32 and v is not.
        out[..., 0::2] = (ve * ce).to(v.dtype) + ((-vo) * se).to(v.dtype)
        out[..., 1::2] = (vo * co).to(v.dtype) + (ve * so).to(v.dtype)
    else:
        out[..., 0::2] = ve * ce - vo * se
        out[..., 1::2] = vo * co + ve * so
    return out


def fused_rope_rotate_pytorch(
    v: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Pure-PyTorch fused rotate via last-dim pair views (fewer strided stores).

    Reshapes ``(..., N) → (..., N/2, 2)`` so even/odd live in a contiguous pair
    axis. Cos/sin are reshaped in their *native* leading shape and broadcast on
    mul — no ``expand(v.shape)``. Writes via empty+pair store (or into ``out``).
    Same math as ``eager_rope_rotate``.

    T=1 decode shares ``rope_rotate_t1``.
    """
    if _is_t1_seq(v):
        return rope_rotate_t1(v, cos, sin, out=out)
    _validate_rope_inputs(v, cos, sin, out)

    vp = v.reshape(*v.shape[:-1], -1, 2)
    # Native-shape pair views broadcast over v's batch/head dims (cos often 1×1×T×N).
    cp = cos.reshape(*cos.shape[:-1], -1, 2)
    sp = sin.reshape(*sin.shape[:-1], -1, 2)
    y0, y1 = _rotate_pair_views(vp, cp, sp, v_dtype=v.dtype, cis_dtype=cos.dtype)
    return _store_pairs(y0, y1, v, out)


def fused_rope_rotate_blocked(
    v: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    *,
    block: int = 64,
) -> torch.Tensor:
    """CPU / scaffold rotate mirroring Triton pair tiling (parity + fallback).

    Flattens nothing extra: walks the pair axis in tiles of ``block``, same
    ``y0/y1`` math as eager/fused. Used as the Triton entry's CPU fallback and
    for tile-structure parity tests. Not a claimed CPU wall win vs pytorch fuse.
    """
    if _is_t1_seq(v):
        return rope_rotate_t1(v, cos, sin, out=out)
    _validate_rope_inputs(v, cos, sin, out)
    if block < 1:
        raise ValueError(f"block must be >= 1, got {block}")

    n_pairs = v.shape[-1] // 2
    vp = v.reshape(*v.shape[:-1], n_pairs, 2)
    cp = cos.reshape(*cos.shape[:-1], -1, 2)
    sp = sin.reshape(*sin.shape[:-1], -1, 2)
    if out is None:
        out = torch.empty_like(v)
    op = out.reshape(*v.shape[:-1], n_pairs, 2)

    for start in range(0, n_pairs, block):
        end = min(start + block, n_pairs)
        y0, y1 = _rotate_pair_views(
            vp[..., start:end, :],
            cp[..., start:end, :],
            sp[..., start:end, :],
            v_dtype=v.dtype,
            cis_dtype=cos.dtype,
        )
        op[..., start:end, 0] = y0
        op[..., start:end, 1] = y1
    return out


def _can_use_triton_rope(v: torch.Tensor) -> bool:
    if not _HAS_TRITON:
        return False
    if not v.is_cuda:
        return False
    if not torch.cuda.is_available():
        return False
    return True


if _HAS_TRITON:

    @triton.jit
    def _rope_rotate_kernel(
        V_ptr,
        Cos_ptr,
        Sin_ptr,
        Out_ptr,
        stride_v_row,
        stride_c_row,
        stride_s_row,
        stride_o_row,
        n_pairs,
        BLOCK: tl.constexpr,
    ):
        """One program = one row (flattened leading dims); tiles over N/2 pairs."""
        row = tl.program_id(0)
        v_row = V_ptr + row * stride_v_row
        c_row = Cos_ptr + row * stride_c_row
        s_row = Sin_ptr + row * stride_s_row
        o_row = Out_ptr + row * stride_o_row

        for start in range(0, n_pairs, BLOCK):
            offs = start + tl.arange(0, BLOCK)
            mask = offs < n_pairs
            # Pair index p → elements 2p, 2p+1
            offs0 = offs * 2
            offs1 = offs0 + 1
            x0 = tl.load(v_row + offs0, mask=mask, other=0.0)
            x1 = tl.load(v_row + offs1, mask=mask, other=0.0)
            c0 = tl.load(c_row + offs0, mask=mask, other=0.0)
            c1 = tl.load(c_row + offs1, mask=mask, other=0.0)
            s0 = tl.load(s_row + offs0, mask=mask, other=0.0)
            s1 = tl.load(s_row + offs1, mask=mask, other=0.0)
            y0 = x0 * c0 - x1 * s0
            y1 = x1 * c1 + x0 * s1
            tl.store(o_row + offs0, y0, mask=mask)
            tl.store(o_row + offs1, y1, mask=mask)


def _triton_cis_rows(v: torch.Tensor, cis: torch.Tensor) -> torch.Tensor:
    """Stage cis for the row-wise Triton kernel with less host tax when possible.

    Same-shape cis: contig view only. Broadcast cis (e.g. 1×1×T×N vs B×H×T×N):
    still expand+contig on host — kernel takes per-row pointers today; a future
    stride-0 broadcast load would drop this copy (scaffold note, no GPU here).
    """
    if cis.shape == v.shape and cis.dtype == v.dtype:
        return cis.contiguous() if not cis.is_contiguous() else cis
    return cis.expand(v.shape).to(dtype=v.dtype).contiguous()


def _triton_rope_forward(
    v: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    out: Optional[torch.Tensor],
) -> torch.Tensor:
    """Launch Triton rotate; cos/sin broadcast-staged to v rows on host."""
    assert _HAS_TRITON
    _validate_rope_inputs(v, cos, sin, out)
    vc = v.contiguous()
    cos_b = _triton_cis_rows(v, cos)
    sin_b = _triton_cis_rows(v, sin)
    if out is None:
        out_t = torch.empty_like(vc)
    else:
        out_t = out.contiguous() if not out.is_contiguous() else out

    n_elem = vc.shape[-1]
    n_pairs = n_elem // 2
    rows = vc.numel() // n_elem
    v2 = vc.view(rows, n_elem)
    c2 = cos_b.view(rows, n_elem)
    s2 = sin_b.view(rows, n_elem)
    o2 = out_t.view(rows, n_elem)

    BLOCK = 64 if n_pairs >= 64 else max(1, 1 << (n_pairs - 1).bit_length())
    BLOCK = min(128, max(1, BLOCK))
    _rope_rotate_kernel[(rows,)](
        v2,
        c2,
        s2,
        o2,
        v2.stride(0),
        c2.stride(0),
        s2.stride(0),
        o2.stride(0),
        n_pairs,
        BLOCK=BLOCK,
    )
    if out is not None and out.data_ptr() != out_t.data_ptr():
        out.copy_(out_t)
        return out
    return out_t.view(v.shape)


class _TritonRopeFn(torch.autograd.Function):
    """Triton forward + analytic backward (cos/sin treated as constant)."""

    @staticmethod
    def forward(ctx, v, cos, sin):  # type: ignore[override]
        # Detach cis for bwd; training tables from rope_cos_sin are already detached.
        cos_d = cos.detach()
        sin_d = sin.detach()
        ctx.save_for_backward(cos_d, sin_d)
        ctx.shape = v.shape
        return _triton_rope_forward(v, cos_d, sin_d, None)

    @staticmethod
    def backward(ctx, grad_out):  # type: ignore[override]
        cos, sin = ctx.saved_tensors
        # ∂L/∂x0 = g0*c0 + g1*s1 ; ∂L/∂x1 = -g0*s0 + g1*c1
        go = grad_out.reshape(*ctx.shape[:-1], -1, 2)
        # Native-shape pair views broadcast (no expand-to-full materialize).
        cos_b = cos.reshape(*cos.shape[:-1], -1, 2)
        sin_b = sin.reshape(*sin.shape[:-1], -1, 2)
        g0, g1 = go[..., 0], go[..., 1]
        c0, c1 = cos_b[..., 0], cos_b[..., 1]
        s0, s1 = sin_b[..., 0], sin_b[..., 1]
        if grad_out.dtype != cos.dtype:
            dx0 = (g0 * c0).to(grad_out.dtype) + (g1 * s1).to(grad_out.dtype)
            dx1 = ((-g0) * s0).to(grad_out.dtype) + (g1 * c1).to(grad_out.dtype)
        else:
            dx0 = g0 * c0 + g1 * s1
            dx1 = -g0 * s0 + g1 * c1
        return _store_pairs(dx0, dx1, grad_out, None), None, None


def fused_rope_rotate_triton(
    v: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """CUDA Triton fused rotate; CPU → blocked tile scaffold (then same math)."""
    if not _can_use_triton_rope(v):
        # Scaffold deepen: tile-structured CPU path mirrors kernel BLOCK loop.
        return fused_rope_rotate_blocked(v, cos, sin, out=out)
    # out= + autograd: run Function then copy if needed
    if v.requires_grad or (isinstance(v, torch.Tensor) and v.grad_fn is not None):
        y = _TritonRopeFn.apply(v, cos, sin)
        if out is not None:
            out.copy_(y)
            return out
        return y
    return _triton_rope_forward(v, cos, sin, out)


def fused_rope_rotate(
    v: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Fused entry: Triton on CUDA when usable, else pure-PyTorch pair path.

    CPU default for ``BDH_ROPE_IMPL=fused`` stays ``fused_rope_rotate_pytorch``
    (no expand / no stack). Triton entry falls back to ``blocked`` for tile
    parity when CUDA is unavailable.
    """
    if _can_use_triton_rope(v):
        return fused_rope_rotate_triton(v, cos, sin, out=out)
    return fused_rope_rotate_pytorch(v, cos, sin, out=out)
