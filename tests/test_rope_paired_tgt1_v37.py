"""Deepen T>1 paired RoPE CPU contract: strided out parity + rejection.

Paired cis is the generate-table shape ``(..., N/2, 2)``. Decode usually
narrows T=1, but the rotate entrypoints accept any sequence length. Lock the
multi-token path so fused GPU readiness keeps the same CPU-safe out-buffer
and alias contracts as the flat T>1 dispatcher.

CPU-only: no claimed fused-RoPE wall-time wins; default BDH_ROPE_IMPL stays
eager unless a test temporarily selects fused.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import bdh
from kernels.rope import (
    eager_rope_rotate,
    fused_rope_rotate_paired,
    rope_rotate_paired,
)
from kernels.rope_dispatch import bdh_rope_rotate_paired


def _small_cfg(**kwargs) -> bdh.BDHConfig:
    defaults = dict(
        n_layer=2,
        n_embd=64,
        n_head=4,
        mlp_internal_dim_multiplier=8,
        dropout=0.0,
        vocab_size=256,
    )
    defaults.update(kwargs)
    return bdh.BDHConfig(**defaults)


def _t_gt1_paired(T: int = 7, seed: int = 370):
    cfg = _small_cfg()
    attn = bdh.Attention(cfg)
    N = cfg.mlp_internal_dim_multiplier * cfg.n_embd // cfg.n_head
    device = torch.device("cpu")
    cos, sin = attn.rope_cos_sin(T, 0, device)
    cos_p = cos.reshape(*cos.shape[:-1], -1, 2)
    sin_p = sin.reshape(*sin.shape[:-1], -1, 2)
    torch.manual_seed(seed)
    v = torch.randn(2, cfg.n_head, T, N)
    return cfg, cos, sin, cos_p, sin_p, v


@pytest.mark.parametrize("impl", ["eager", "fused"])
def test_public_paired_t_gt1_matches_flat_eager(impl):
    """T>1 paired dispatch stays bit-identical to flat eager rotate."""
    _, cos, sin, cos_p, sin_p, v = _t_gt1_paired(T=7, seed=371)
    assert v.shape[-2] > 1
    ref = eager_rope_rotate(v, cos, sin)
    got = bdh_rope_rotate_paired(v, cos_p, sin_p, impl=impl)
    assert torch.equal(got, ref), impl
    assert torch.equal(rope_rotate_paired(v, cos_p, sin_p), ref)
    assert torch.equal(fused_rope_rotate_paired(v, cos_p, sin_p), ref)


@pytest.mark.parametrize("impl", ["eager", "fused"])
def test_public_paired_t_gt1_strided_out_preserves_parity(impl):
    """T>1 paired writes only the requested non-contiguous cache-style slot."""
    _, cos, sin, cos_p, sin_p, v = _t_gt1_paired(T=7, seed=372)
    sentinel = torch.tensor(-777.0)

    def make_out():
        backing = torch.full((*v.shape[:-1], v.shape[-1] * 2), sentinel.item())
        return backing, backing[..., ::2]

    ref_backing, ref_out = make_out()
    ref = eager_rope_rotate(v, cos, sin, out=ref_out)
    assert ref is ref_out and not ref.is_contiguous()

    got_backing, got_out = make_out()
    got = bdh_rope_rotate_paired(v, cos_p, sin_p, out=got_out, impl=impl)

    assert got is got_out and not got.is_contiguous(), impl
    assert torch.equal(got, ref), impl
    assert torch.equal(
        got_backing[..., 1::2],
        torch.full_like(got_backing[..., 1::2], sentinel),
    ), impl
    assert torch.equal(got_backing[..., 1::2], ref_backing[..., 1::2]), impl


@pytest.mark.parametrize("impl", ["eager", "fused"])
def test_public_paired_t_gt1_rejects_cis_alias_and_bad_out_shape(impl):
    """T>1 paired rejects cis-aliased / wrong-shaped out before pair stores."""
    _, _, _, cos_p, sin_p, v = _t_gt1_paired(T=5, seed=373)

    for cis_p in (cos_p, sin_p):
        out = cis_p.reshape(*cis_p.shape[:-2], -1).expand_as(v)
        before = out.clone()
        with pytest.raises(ValueError, match="alias"):
            bdh_rope_rotate_paired(v, cos_p, sin_p, out=out, impl=impl)
        assert torch.equal(out, before), impl

    bad = torch.full((*v.shape[:-1], v.shape[-1] - 2), -55.0)
    before_bad = bad.clone()
    with pytest.raises(ValueError, match="rope out"):
        bdh_rope_rotate_paired(v, cos_p, sin_p, out=bad, impl=impl)
    assert torch.equal(bad, before_bad), impl


@pytest.mark.parametrize("impl", ["eager", "fused"])
def test_public_paired_t_gt1_mixed_dtype_strided_out_preserves_parity(impl):
    """fp16 T>1 V into fp32 strided cache slots stays parity-safe via paired."""
    cfg = _small_cfg()
    attn = bdh.Attention(cfg)
    N = cfg.mlp_internal_dim_multiplier * cfg.n_embd // cfg.n_head
    T = 6
    cos, sin = attn.rope_cos_sin(T, 0, torch.device("cpu"))
    cos_p = cos.reshape(*cos.shape[:-1], -1, 2)
    sin_p = sin.reshape(*sin.shape[:-1], -1, 2)
    torch.manual_seed(374)
    v = torch.randn(2, cfg.n_head, T, N, dtype=torch.float16)
    assert cos.dtype == torch.float32 and sin.dtype == torch.float32

    sentinel = torch.tensor(-123.0, dtype=torch.float32)

    def make_out():
        backing = torch.full((*v.shape[:-1], N * 2), sentinel.item())
        return backing, backing[..., ::2]

    ref_backing, ref_out = make_out()
    ref = eager_rope_rotate(v, cos, sin, out=ref_out)
    got_backing, got_out = make_out()
    got = bdh_rope_rotate_paired(v, cos_p, sin_p, out=got_out, impl=impl)

    assert ref is ref_out and got is got_out
    assert got.dtype == torch.float32 and not got.is_contiguous(), impl
    assert torch.equal(got, ref), impl
    assert torch.equal(got_backing[..., 1::2], ref_backing[..., 1::2]), impl
