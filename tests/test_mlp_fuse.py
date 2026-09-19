"""MLP fuse: bias+ReLU in-place, decoder view+F.linear, no permute→contiguous."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import bdh
import bdh_baseline as baseline


def _cfg(**kwargs) -> bdh.BDHConfig:
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


def test_bias_relu_bit_identical_to_add_then_relu():
    torch.manual_seed(7)
    out = torch.randn(2, 5, 4, 8)
    bias = torch.randn(4, 8)
    ref = F.relu(out.clone() + bias, inplace=True)
    got = bdh.BDH._bias_relu_(out.clone(), bias)
    assert torch.equal(got, ref)


def test_bias_relu_none_is_inplace_relu():
    torch.manual_seed(8)
    out = torch.randn(3, 2, 2, 4)
    ref = F.relu(out.clone(), inplace=True)
    got = bdh.BDH._bias_relu_(out.clone(), None)
    assert torch.equal(got, ref)


def test_encoder_relu_with_bias_matches_reference():
    torch.manual_seed(9)
    B, T, D, nh, N = 2, 6, 32, 4, 16
    x = torch.randn(B, T, D)
    w = torch.randn(nh, D, N)
    bias = torch.randn(nh, N)
    got = bdh.BDH._encoder_relu(x, w, bias)
    ref = F.relu(torch.einsum("btd,hdn->bthn", x, w) + bias)
    assert torch.equal(got, ref)
    assert got.is_contiguous()
    assert got.shape == (B, T, nh, N)


def test_mlp_merge_view_no_contiguous_copy():
    """Contiguous (B,T,nh,N) → decoder via view; never calls Tensor.contiguous."""
    cfg = _cfg()
    torch.manual_seed(12)
    m = bdh.BDH(cfg).eval()
    B, T, nh = 2, 7, cfg.n_head
    N = cfg.n_embd * cfg.mlp_internal_dim_multiplier // nh
    x_bthn = torch.randn(B, T, nh, N)
    y_bthn = torch.randn(B, T, nh, N)
    assert x_bthn.is_contiguous() and y_bthn.is_contiguous()

    calls = {"n": 0}
    orig = torch.Tensor.contiguous

    def hooked(self):
        calls["n"] += 1
        return orig(self)

    torch.Tensor.contiguous = hooked
    try:
        y = m._mlp_merge(x_bthn, y_bthn, B, T, nh, N)
    finally:
        torch.Tensor.contiguous = orig

    assert calls["n"] == 0
    ref = F.linear(
        (x_bthn * y_bthn).view(B, T, nh * N),
        m.decoder.transpose(0, 1),
        m.decoder_bias,
    )
    assert torch.equal(y, ref)


def test_mlp_merge_with_decoder_bias_fuses():
    cfg = _cfg()
    torch.manual_seed(13)
    m = bdh.BDH(cfg).eval()
    D = cfg.n_embd
    nh = cfg.n_head
    N = D * cfg.mlp_internal_dim_multiplier // nh
    bias = torch.randn(D)
    m.decoder_bias = torch.nn.Parameter(bias)
    B, T = 2, 5
    x_bthn = F.relu(torch.randn(B, T, nh, N))
    y_bthn = F.relu(torch.randn(B, T, nh, N))
    got = m._mlp_merge(x_bthn, y_bthn, B, T, nh, N)
    ref = F.linear((x_bthn * y_bthn).view(B, T, nh * N), m.decoder.T, bias)
    assert torch.equal(got, ref)


def test_forward_with_optional_biases_runs():
    """Optional encoder/decoder biases fuse; attention tril(-1) path unchanged."""
    cfg = _cfg()
    torch.manual_seed(21)
    m = bdh.BDH(cfg)
    nh, D = cfg.n_head, cfg.n_embd
    N = D * cfg.mlp_internal_dim_multiplier // nh
    m.encoder_bias = torch.nn.Parameter(torch.randn(nh, N) * 0.01)
    m.encoder_v_bias = torch.nn.Parameter(torch.randn(nh, N) * 0.01)
    m.decoder_bias = torch.nn.Parameter(torch.randn(D) * 0.01)
    m.eval()
    idx = torch.randint(0, cfg.vocab_size, (2, 11))
    logits, _ = m(idx)
    assert logits.shape == (2, 11, cfg.vocab_size)
    assert torch.isfinite(logits).all()


def test_default_path_still_matches_baseline():
    cfg = _cfg()
    torch.manual_seed(42)
    base = baseline.BDH(cfg)
    opt = bdh.BDH(cfg)
    opt.load_state_dict(base.state_dict(), strict=True)
    base.eval()
    opt.eval()
    x = torch.randint(0, cfg.vocab_size, (3, 16))
    lb, lossb = base(x, x)
    lo, losso = opt(x, x)
    assert torch.equal(lb, lo)
    assert torch.equal(lossb, losso)


def test_attn_pos0_still_zero_with_mlp_fuse():
    """tril(diagonal=-1) preserved: position 0 attention output is zero."""
    cfg = _cfg(n_layer=1)
    torch.manual_seed(1)
    m = bdh.BDH(cfg).eval()
    B, T, nh = 1, 8, cfg.n_head
    N = cfg.n_embd * cfg.mlp_internal_dim_multiplier // nh
    Q = torch.randn(B, nh, T, N)
    V = torch.randn(B, 1, T, cfg.n_embd)
    out, _, _ = m.attn(Q, Q, V)
    assert torch.equal(out[:, :, 0, :], torch.zeros_like(out[:, :, 0, :]))
