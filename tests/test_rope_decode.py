"""T=1 decode RoPE apply deepen: parity vs strided reference; cis reuse."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import bdh
import bdh_baseline as baseline
from kernels.rope import (
    eager_rope_rotate,
    fused_rope_rotate_pytorch,
    rope_rotate_t1,
)
from kernels.rope_dispatch import bdh_rope_rotate


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


def _strided_ref(v, cos, sin, out=None):
    """Historical strided even/odd rotate — does not route through rope_rotate_t1."""
    if out is None:
        out = torch.empty_like(v)
    ve, vo = v[..., 0::2], v[..., 1::2]
    ce, se = cos[..., 0::2], sin[..., 0::2]
    co, so = cos[..., 1::2], sin[..., 1::2]
    if v.dtype != cos.dtype:
        out[..., 0::2] = (ve * ce).to(v.dtype) + ((-vo) * se).to(v.dtype)
        out[..., 1::2] = (vo * co).to(v.dtype) + (ve * so).to(v.dtype)
    else:
        out[..., 0::2] = ve * ce - vo * se
        out[..., 1::2] = vo * co + ve * so
    return out


def test_rope_rotate_t1_bit_identical_to_strided():
    cfg = _small_cfg()
    attn = bdh.Attention(cfg)
    device = torch.device("cpu")
    N = cfg.mlp_internal_dim_multiplier * cfg.n_embd // cfg.n_head
    attn.ensure_rope_table(32, device)
    cos, sin = attn.rope_cos_sin(1, 9, device)
    torch.manual_seed(0)
    v = torch.randn(2, cfg.n_head, 1, N)
    ref = _strided_ref(v, cos, sin)
    out_t1 = rope_rotate_t1(v, cos, sin)
    out_eager = eager_rope_rotate(v, cos, sin)
    out_fused = fused_rope_rotate_pytorch(v, cos, sin)
    assert torch.equal(out_t1, ref)
    assert torch.equal(out_eager, ref)
    assert torch.equal(out_fused, ref)


def test_rope_rotate_t1_out_inplace_into_kr_slot():
    cfg = _small_cfg()
    attn = bdh.Attention(cfg)
    device = torch.device("cpu")
    N = cfg.mlp_internal_dim_multiplier * cfg.n_embd // cfg.n_head
    attn.ensure_rope_table(16, device)
    cos, sin = attn.rope_cos_sin(1, 3, device)
    torch.manual_seed(1)
    v = torch.randn(1, cfg.n_head, 1, N)
    # Simulate CacheManager.reserve KR write view (contig slot).
    kr = torch.empty(1, cfg.n_head, 1, N)
    ref = _strided_ref(v, cos, sin)
    out = rope_rotate_t1(v, cos, sin, out=kr)
    assert out is kr
    assert torch.equal(kr, ref)


def test_rope_rotate_t1_matches_baseline_phases():
    cfg = _small_cfg()
    attn = bdh.Attention(cfg)
    device = torch.device("cpu")
    N = cfg.mlp_internal_dim_multiplier * cfg.n_embd // cfg.n_head
    rope_start = 5
    cos, sin = attn.phases_cos_sin(attn._rope_phases(1, rope_start, device))
    torch.manual_seed(2)
    v = torch.randn(2, cfg.n_head, 1, N)
    phases = attn._rope_phases(1, rope_start, device)
    out_base = baseline.Attention.rope(phases, v)
    out_t1 = rope_rotate_t1(v, cos, sin)
    assert torch.equal(out_t1, out_base)


def test_rope_t1_cis_reuse_same_object():
    """Single-slot last-position cache: same rope_start reuses the narrow."""
    cfg = _small_cfg()
    attn = bdh.Attention(cfg)
    device = torch.device("cpu")
    attn.ensure_rope_table(24, device)
    cos1, sin1 = attn.rope_cos_sin(1, 11, device)
    cos2, sin2 = attn.rope_cos_sin(1, 11, device)
    assert cos1 is cos2 and sin1 is sin2
    # Different position → new narrow (overwrites the single-slot cache).
    cos3, sin3 = attn.rope_cos_sin(1, 12, device)
    assert cos3 is not cos1
    assert cos3.shape[-2] == 1
    # Back to 11 → values match; new narrow object (single-slot, last-pos only).
    cos4, sin4 = attn.rope_cos_sin(1, 11, device)
    assert torch.equal(cos4, cos1) and torch.equal(sin4, sin1)
    cos5, sin5 = attn.rope_cos_sin(1, 11, device)
    assert cos5 is cos4 and sin5 is sin4


def test_rope_table_pairs_built_and_match_flat():
    cfg = _small_cfg()
    attn = bdh.Attention(cfg)
    device = torch.device("cpu")
    assert attn._rope_table_pairs is None
    attn.ensure_rope_table(20, device)
    assert attn._rope_table_pairs is not None
    cos, sin = attn._rope_table
    cp, sp = attn._rope_table_pairs
    assert cp.shape == (*cos.shape[:-1], cos.shape[-1] // 2, 2)
    assert torch.equal(cp.reshape(cos.shape), cos)
    assert torch.equal(sp.reshape(sin.shape), sin)
    # Second ensure is no-op (same pairs object).
    pairs0 = attn._rope_table_pairs
    attn.ensure_rope_table(20, device)
    assert attn._rope_table_pairs is pairs0


def test_t1_dispatch_default_eager_matches_strided(monkeypatch):
    monkeypatch.delenv("BDH_ROPE_IMPL", raising=False)
    cfg = _small_cfg()
    attn = bdh.Attention(cfg)
    device = torch.device("cpu")
    N = cfg.mlp_internal_dim_multiplier * cfg.n_embd // cfg.n_head
    attn.ensure_rope_table(16, device)
    cos, sin = attn.rope_cos_sin(1, 4, device)
    torch.manual_seed(4)
    v = torch.randn(2, cfg.n_head, 1, N)
    ref = _strided_ref(v, cos, sin)
    out = bdh_rope_rotate(v, cos, sin)
    assert torch.equal(out, ref)


def test_attention_forward_t1_decode_parity_vs_baseline():
    """Incremental T=1 Attention vs baseline phases path (tril(-1) decode)."""
    cfg = _small_cfg(n_layer=1)
    torch.manual_seed(0)
    attn = bdh.Attention(cfg).eval()
    base = baseline.Attention(cfg).eval()
    base.load_state_dict(attn.state_dict(), strict=False)
    device = torch.device("cpu")
    B, nh, S, N = 1, cfg.n_head, 8, cfg.mlp_internal_dim_multiplier * cfg.n_embd // cfg.n_head
    D = cfg.n_embd
    attn.ensure_rope_table(S + 1, device)
    # Past KR/V already RoPE'd at positions 0..S-1
    torch.manual_seed(10)
    past_q = torch.randn(B, nh, S, N)
    past_v = torch.randn(B, 1, S, D)
    cos_p, sin_p = attn.rope_cos_sin(S, 0, device)
    past_kr = eager_rope_rotate(past_q, cos_p, sin_p)
    # New token at absolute index S
    q = torch.randn(B, nh, 1, N)
    v = torch.randn(B, 1, 1, D)
    cos_sin = attn.rope_cos_sin(1, S, device)
    out, kr, vv = attn(
        Q=q, K=q, V=v, rope_start=S, past_kr=past_kr, past_v=past_v, cos_sin=cos_sin
    )
    # Reference: RoPE new Q with baseline, then two-GEMM decode (no self).
    phases = attn._rope_phases(1, S, device)
    qr_base = baseline.Attention.rope(phases, q)
    assert torch.equal(kr, qr_base)
    scores = torch.matmul(qr_base, past_kr.transpose(-2, -1))  # (B,nh,1,S)
    out_ref = torch.matmul(scores, past_v.expand(B, nh, S, D))
    assert torch.allclose(out, out_ref, rtol=0, atol=0)
    assert out.shape == (B, nh, 1, D)


def test_generate_tokens_match_baseline_after_t1_rope():
    cfg = _small_cfg(n_layer=2)
    torch.manual_seed(0)
    m = bdh.BDH(cfg).eval()
    torch.manual_seed(0)
    b = baseline.BDH(cfg).eval()
    b.load_state_dict(m.state_dict())
    prompt = torch.randint(0, cfg.vocab_size, (1, 4))
    torch.manual_seed(42)
    out_m = m.generate(prompt.clone(), max_new_tokens=8, temperature=1.0)
    torch.manual_seed(42)
    out_b = b.generate(prompt.clone(), max_new_tokens=8, temperature=1.0)
    assert torch.equal(out_m, out_b)


def test_rope_rotate_t1_rejects_multi_token():
    v = torch.randn(1, 2, 4, 8)
    cos = torch.randn(1, 1, 4, 8)
    sin = torch.randn(1, 1, 4, 8)
    with pytest.raises(ValueError, match="sequence length 1"):
        rope_rotate_t1(v, cos, sin)
