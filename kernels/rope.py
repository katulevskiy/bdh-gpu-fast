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


def eager_rope_rotate(
    v: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Reference rotate: strided even/odd slices (matches historical ``Attention.rope``)."""
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
    axis, computes both outputs, then writes via a single ``stack→flatten`` (or
    into ``out``'s pair view). Same math as ``eager_rope_rotate``.
    """
    _validate_rope_inputs(v, cos, sin, out)

    # Contiguous last-dim helps reshape without an extra copy; leave non-contig
    # as-is (reshape may still view when strides allow).
    vp = v.reshape(*v.shape[:-1], -1, 2)
    # Broadcast cos/sin to v's leading dims for the pair view (cos often 1×1×T×N).
    cos_b = cos.expand(v.shape) if cos.shape != v.shape else cos
    sin_b = sin.expand(v.shape) if sin.shape != v.shape else sin
    cp = cos_b.reshape(*v.shape[:-1], -1, 2)
    sp = sin_b.reshape(*v.shape[:-1], -1, 2)

    x0, x1 = vp[..., 0], vp[..., 1]
    c0, c1 = cp[..., 0], cp[..., 1]
    s0, s1 = sp[..., 0], sp[..., 1]

    if v.dtype != cos.dtype:
        y0 = (x0 * c0).to(v.dtype) + ((-x1) * s0).to(v.dtype)
        y1 = (x1 * c1).to(v.dtype) + (x0 * s1).to(v.dtype)
    else:
        y0 = x0 * c0 - x1 * s0
        y1 = x1 * c1 + x0 * s1

    if out is None:
        return torch.stack((y0, y1), dim=-1).reshape(v.shape)

    op = out.reshape(*v.shape[:-1], -1, 2)
    op[..., 0] = y0
    op[..., 1] = y1
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


def _triton_rope_forward(
    v: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    out: Optional[torch.Tensor],
) -> torch.Tensor:
    """Launch Triton rotate; cos/sin broadcast-expanded to v shape on host."""
    assert _HAS_TRITON
    _validate_rope_inputs(v, cos, sin, out)
    # Host staging: contiguous rows, broadcast cis to v (cache tables stay small
    # until expand). Same dtype as v for the kernel.
    vc = v.contiguous()
    cos_b = cos.expand(v.shape).to(dtype=v.dtype).contiguous()
    sin_b = sin.expand(v.shape).to(dtype=v.dtype).contiguous()
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
        cos_b = cos.expand(ctx.shape).reshape(*ctx.shape[:-1], -1, 2)
        sin_b = sin.expand(ctx.shape).reshape(*ctx.shape[:-1], -1, 2)
        g0, g1 = go[..., 0], go[..., 1]
        c0, c1 = cos_b[..., 0], cos_b[..., 1]
        s0, s1 = sin_b[..., 0], sin_b[..., 1]
        if grad_out.dtype != cos.dtype:
            dx0 = (g0 * c0).to(grad_out.dtype) + (g1 * s1).to(grad_out.dtype)
            dx1 = ((-g0) * s0).to(grad_out.dtype) + (g1 * c1).to(grad_out.dtype)
        else:
            dx0 = g0 * c0 + g1 * s1
            dx1 = -g0 * s0 + g1 * c1
        dx = torch.stack((dx0, dx1), dim=-1).reshape(ctx.shape)
        return dx, None, None


def fused_rope_rotate_triton(
    v: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """CUDA Triton fused rotate; falls back to pure-PyTorch fused if unavailable."""
    if not _can_use_triton_rope(v):
        return fused_rope_rotate_pytorch(v, cos, sin, out=out)
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
    """Fused entry: Triton on CUDA when usable, else pure-PyTorch pair path."""
    if _can_use_triton_rope(v):
        return fused_rope_rotate_triton(v, cos, sin, out=out)
    return fused_rope_rotate_pytorch(v, cos, sin, out=out)
