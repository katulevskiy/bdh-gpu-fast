"""copy-tax-v1: contiguous RoPE alloc + in-place tril_; parity vs baseline."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from torch.profiler import ProfilerActivity, profile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import bdh
import bdh_baseline as baseline
from kernels.attention import eager_tril_attn
from kernels.rope import (
    _alloc_rope_out,
    eager_rope_rotate,
    fused_rope_rotate_pytorch,
)


def _cfg(**kwargs) -> bdh.BDHConfig:
    defaults = dict(
        n_layer=3,
        n_embd=64,
        n_head=4,
        mlp_internal_dim_multiplier=8,
        dropout=0.0,
        vocab_size=256,
    )
    defaults.update(kwargs)
    return bdh.BDHConfig(**defaults)


def _twin(cfg: bdh.BDHConfig, seed: int = 0):
    torch.manual_seed(seed)
    m_base = baseline.BDH(cfg)
    m_opt = bdh.BDH(cfg)
    missing, unexpected = m_opt.load_state_dict(m_base.state_dict(), strict=True)
    assert not missing and not unexpected
    return m_base, m_opt


def test_alloc_rope_out_contiguous_despite_permute_view():
    """empty_like(permute) preserves strides; helper must return dense layout."""
    B, T, H, N = 2, 16, 4, 32
    x = torch.randn(B, T, H, N).contiguous()
    v = x.permute(0, 2, 1, 3)  # (B,H,T,N) non-contig — train encoder view
    assert not v.is_contiguous()
    like = torch.empty_like(v)
    assert not like.is_contiguous()
    out = _alloc_rope_out(v)
    assert out.shape == v.shape
    assert out.dtype == v.dtype
    assert out.device == v.device
    assert out.is_contiguous()


def test_eager_rope_on_permute_view_yields_contiguous_qr():
    cfg = _cfg()
    attn = bdh.Attention(cfg)
    N = cfg.mlp_internal_dim_multiplier * cfg.n_embd // cfg.n_head
    B, T, nh = 2, 24, cfg.n_head
    x = torch.randn(B, T, nh, N).contiguous()
    v = x.permute(0, 2, 1, 3)
    cos, sin = attn.rope_cos_sin(T, 0, v.device)
    qr = eager_rope_rotate(v, cos, sin)
    assert qr.is_contiguous()
    # Bit-identical to rotating a contiguous clone of the same values.
    qr2 = eager_rope_rotate(v.contiguous(), cos, sin)
    assert torch.equal(qr, qr2)


def test_fused_rope_on_permute_view_yields_contiguous_qr():
    cfg = _cfg()
    attn = bdh.Attention(cfg)
    N = cfg.mlp_internal_dim_multiplier * cfg.n_embd // cfg.n_head
    x = torch.randn(2, 20, cfg.n_head, N).contiguous()
    v = x.permute(0, 2, 1, 3)
    cos, sin = attn.rope_cos_sin(20, 0, v.device)
    qr = fused_rope_rotate_pytorch(v, cos, sin)
    assert qr.is_contiguous()
    assert torch.equal(qr, eager_rope_rotate(v.contiguous(), cos, sin))


def test_eager_tril_attn_matches_out_of_place_tril():
    B, H, T, N, D = 2, 4, 16, 32, 64
    torch.manual_seed(0)
    Q = torch.randn(B, H, T, N)
    V = torch.randn(B, 1, T, D)
    out = eager_tril_attn(Q, Q, V)
    scores = Q @ Q.transpose(-2, -1)
    ref = scores.tril(diagonal=-1) @ V
    assert torch.equal(out, ref)


def test_train_logits_and_grads_match_baseline_dropout_zero():
    cfg = _cfg(dropout=0.0, n_layer=2)
    m_base, m_opt = _twin(cfg, seed=41)
    m_base.train()
    m_opt.train()
    torch.manual_seed(42)
    x = torch.randint(0, cfg.vocab_size, (3, 20))
    y = torch.randint(0, cfg.vocab_size, (3, 20))
    lb, lossb = m_base(x, y)
    lo, losso = m_opt(x, y)
    assert torch.equal(lb, lo)
    assert torch.equal(lossb, losso)
    lossb.backward()
    losso.backward()
    for (nb, pb), (no, po) in zip(m_base.named_parameters(), m_opt.named_parameters()):
        assert nb == no
        assert torch.allclose(pb.grad, po.grad, rtol=1e-5, atol=1e-5), nb


def test_generate_cache_parity_untouched():
    """Generate tokens match baseline (RoPE out= cache path still correct)."""
    cfg = _cfg(n_layer=2, dropout=0.0)
    m_base, m_opt = _twin(cfg, seed=9)
    m_base.eval()
    m_opt.eval()
    torch.manual_seed(10)
    prompt = torch.randint(0, cfg.vocab_size, (1, 8))
    torch.manual_seed(11)
    out_b = m_base.generate(prompt.clone(), max_new_tokens=6)
    torch.manual_seed(11)
    out_o = m_opt.generate(prompt.clone(), max_new_tokens=6)
    assert torch.equal(out_b, out_o)


def test_forward_drops_qr_gemm_contig_copies():
    """Profile: permute-fed RoPE must not clone QR / QR.mT for the score GEMM.

    Before: empty_like(permute) → non-contig QR → 2× clone+copy_ of (B,H,T,N)
    and (B,H,N,T) per layer. After: those shapes absent from copy_/clone.
    """
    cfg = _cfg(n_layer=2, dropout=0.0)
    m = bdh.BDH(cfg).eval()
    B, T = 2, 32
    x = torch.randint(0, cfg.vocab_size, (B, T))
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as p:
        with torch.no_grad():
            m(x)
    nh = cfg.n_head
    N = cfg.mlp_internal_dim_multiplier * cfg.n_embd // nh
    bad_q = [B, nh, T, N]
    bad_kt = [B, nh, N, T]
    offenders = []
    for e in p.key_averages(group_by_input_shape=True):
        if e.key not in ("aten::copy_", "aten::clone"):
            continue
        for sh in e.input_shapes:
            if list(sh) == bad_q or list(sh) == bad_kt:
                offenders.append((e.key, e.count, sh))
    assert not offenders, f"unexpected QR GEMM contig copies: {offenders}"


def test_attn_module_qr_contiguous_from_encoder_layout():
    """Attention cold path: RoPE output contiguous when Q is encoder permute."""
    cfg = _cfg()
    attn = bdh.Attention(cfg)
    N = cfg.mlp_internal_dim_multiplier * cfg.n_embd // cfg.n_head
    x = torch.randn(2, 12, cfg.n_head, N).contiguous()
    Q = x.permute(0, 2, 1, 3)
    V = torch.randn(2, 1, 12, cfg.n_embd)
    out, qr, v = attn(Q=Q, K=Q, V=V)
    assert qr.is_contiguous()
    assert out.shape == (2, cfg.n_head, 12, cfg.n_embd)
