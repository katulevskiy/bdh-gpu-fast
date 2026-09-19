"""Encoder fuse: einsum default path + optional F.linear bias epilogue."""

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


def test_hdn_as_linear_weight_shape_and_math():
    torch.manual_seed(0)
    nh, D, N = 4, 32, 16
    w = torch.randn(nh, D, N)
    wl = bdh.BDH._hdn_as_linear_weight(w)
    assert wl.shape == (nh * N, D)
    x = torch.randn(2, 5, D)
    ref = torch.einsum("btd,hdn->bthn", x, w)
    got = F.linear(x, wl).view(2, 5, nh, N)
    assert torch.equal(got, ref)


def test_encoder_relu_default_matches_einsum_relu():
    torch.manual_seed(1)
    B, T, D, nh, N = 2, 7, 32, 4, 16
    x = torch.randn(B, T, D)
    w = torch.randn(nh, D, N)
    got = bdh.BDH._encoder_relu(x, w, None)
    ref = F.relu(torch.einsum("btd,hdn->bthn", x, w))
    assert torch.equal(got, ref)
    assert got.is_contiguous()
    assert got.shape == (B, T, nh, N)


def test_encoder_relu_bias_matches_einsum_add_relu():
    """F.linear fused-bias path bit-identical to einsum + bias + ReLU."""
    torch.manual_seed(2)
    B, T, D, nh, N = 2, 6, 32, 4, 16
    x = torch.randn(B, T, D)
    w = torch.randn(nh, D, N)
    bias = torch.randn(nh, N)
    got = bdh.BDH._encoder_relu(x, w, bias)
    ref = F.relu(torch.einsum("btd,hdn->bthn", x, w) + bias)
    assert torch.equal(got, ref)
    assert got.is_contiguous()


def test_encoder_v_relu_bit_identical_and_decoder_view():
    torch.manual_seed(3)
    B, T, D, nh, N = 2, 5, 32, 4, 16
    y = torch.randn(B, nh, T, D)
    w = torch.randn(nh, D, N)
    bias = torch.randn(nh, N)
    got0 = bdh.BDH._encoder_v_relu(y, w, None)
    ref0 = F.relu(torch.einsum("bhtd,hdn->bthn", y, w))
    assert torch.equal(got0, ref0)
    got = bdh.BDH._encoder_v_relu(y, w, bias)
    ref = F.relu(torch.einsum("bhtd,hdn->bthn", y, w) + bias)
    assert torch.equal(got, ref)
    # einsum("bhtd,hdn->bthn") may be a non-contig permute-view; product with
    # contiguous encoder output still yields a free decoder view.
    x = torch.randn(B, T, D)
    x_bthn = bdh.BDH._encoder_relu(x, w, None)
    prod = x_bthn * got
    assert prod.is_contiguous()
    assert prod.view(B, T, nh * N).shape == (B, T, nh * N)


def test_forward_dropout0_bit_identical_vs_baseline():
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


def test_forward_with_encoder_biases_finite():
    cfg = _cfg()
    torch.manual_seed(21)
    m = bdh.BDH(cfg)
    nh, D = cfg.n_head, cfg.n_embd
    N = D * cfg.mlp_internal_dim_multiplier // nh
    m.encoder_bias = torch.nn.Parameter(torch.randn(nh, N) * 0.01)
    m.encoder_v_bias = torch.nn.Parameter(torch.randn(nh, N) * 0.01)
    m.eval()
    idx = torch.randint(0, cfg.vocab_size, (2, 11))
    logits, _ = m(idx)
    assert logits.shape == (2, 11, cfg.vocab_size)
    assert torch.isfinite(logits).all()


def test_state_dict_shapes_unchanged():
    cfg = _cfg()
    m = bdh.BDH(cfg)
    sd = m.state_dict()
    N = cfg.n_embd * cfg.mlp_internal_dim_multiplier // cfg.n_head
    assert sd["encoder"].shape == (cfg.n_head, cfg.n_embd, N)
    assert sd["encoder_v"].shape == sd["encoder"].shape
    assert "encoder_bias" not in sd  # registered None — absent from state_dict
    assert "encoder_v_bias" not in sd


def test_attn_pos0_still_zero():
    """tril(diagonal=-1) preserved after encoder-fuse."""
    cfg = _cfg(n_layer=1)
    torch.manual_seed(1)
    m = bdh.BDH(cfg).eval()
    B, T, nh = 1, 8, cfg.n_head
    N = cfg.n_embd * cfg.mlp_internal_dim_multiplier // nh
    Q = torch.randn(B, nh, T, N)
    V = torch.randn(B, 1, T, cfg.n_embd)
    out, _, _ = m.attn(Q, Q, V)
    assert torch.equal(out[:, :, 0, :], torch.zeros_like(out[:, :, 0, :]))
