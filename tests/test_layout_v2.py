"""Layout-v2: eval cached contiguous encoder F.linear; train keeps einsum."""

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


def test_encoder_w_lin_cached_shape_and_version():
    torch.manual_seed(0)
    m = bdh.BDH(_cfg()).eval()  # warms cache via train(False)
    assert m._encoder_w_lin is not None
    wl = m._encoder_w_lin
    nh, D, N = m.encoder.shape
    assert wl.shape == (nh * N, D)
    assert wl.is_contiguous()
    assert torch.equal(wl, bdh.BDH._hdn_as_linear_weight(m.encoder))
    ptr1 = wl.data_ptr()
    assert m._encoder_w_lin_cached().data_ptr() == ptr1
    with torch.no_grad():
        m.encoder.add_(0.0)
    m.eval()  # refresh after in-place bump
    wl2 = m._encoder_w_lin
    assert wl2 is not None and wl2.data_ptr() != ptr1
    assert torch.equal(wl2, bdh.BDH._hdn_as_linear_weight(m.encoder))
    m.train()
    assert m._encoder_w_lin is None


def test_encoder_relu_weight_lin_bit_identical_to_einsum():
    torch.manual_seed(1)
    B, T, D, nh, N = 2, 7, 32, 4, 16
    x = torch.randn(B, T, D)
    w = torch.randn(nh, D, N)
    bias = torch.randn(nh, N)
    wl = bdh.BDH._hdn_as_linear_weight(w).contiguous()
    got = bdh.BDH._encoder_relu(x, w, None, weight_lin=wl)
    ref = F.relu(torch.einsum("btd,hdn->bthn", x, w))
    assert torch.equal(got, ref)
    got_b = bdh.BDH._encoder_relu(x, w, bias, weight_lin=wl)
    ref_b = F.relu(torch.einsum("btd,hdn->bthn", x, w) + bias)
    assert torch.equal(got_b, ref_b)


def test_eval_fwd_matches_train_einsum_dropout0():
    cfg = _cfg()
    torch.manual_seed(42)
    m = bdh.BDH(cfg)
    x = torch.randint(0, cfg.vocab_size, (3, 16))
    m.train()
    lt, loss_t = m(x, x)
    m.eval()
    le, loss_e = m(x, x)
    assert torch.equal(lt, le)
    assert torch.equal(loss_t, loss_e)
    assert m._encoder_w_lin is not None


def test_state_dict_shapes_unchanged_vs_baseline():
    cfg = _cfg()
    torch.manual_seed(0)
    base = baseline.BDH(cfg)
    opt = bdh.BDH(cfg)
    opt.load_state_dict(base.state_dict(), strict=True)
    sd = opt.state_dict()
    assert sd["encoder"].shape == base.encoder.shape
    assert sd["encoder_v"].shape == base.encoder_v.shape
    assert sd["decoder"].shape == base.decoder.shape
    assert sd["lm_head"].shape == base.lm_head.shape
    assert not any("lin" in k for k in sd)
    base.eval()
    opt.eval()
    x = torch.randint(0, cfg.vocab_size, (2, 12))
    lb, _ = base(x, x)
    lo, _ = opt(x, x)
    assert torch.equal(lb, lo)


def test_tril_neg1_pos0_smoke():
    cfg = _cfg(n_layer=1)
    m = bdh.BDH(cfg).eval()
    x = torch.randint(0, cfg.vocab_size, (1, 8))
    with torch.no_grad():
        logits, _ = m(x)
    assert logits.shape == (1, 8, cfg.vocab_size)
    assert torch.isfinite(logits).all()


def test_channels_last_not_default_path():
    cfg = _cfg()
    m = bdh.BDH(cfg).eval()
    x = torch.randint(0, cfg.vocab_size, (2, 8))
    outs = []
    real = m._encoder_relu_fwd

    def hook(x_btd):
        y = real(x_btd)
        outs.append(y)
        return y

    m._encoder_relu_fwd = hook
    with torch.no_grad():
        m(x)
    assert outs
    y = outs[0]
    assert y.is_contiguous()
    assert not y.is_contiguous(memory_format=torch.channels_last)


def test_load_state_dict_refreshes_encoder_cache():
    """load_state_dict must not leave a stale (nh*N, D) eval cache."""
    cfg = _cfg()
    torch.manual_seed(0)
    a = bdh.BDH(cfg).eval()
    torch.manual_seed(1)
    b = bdh.BDH(cfg).eval()
    # Different init → different caches
    assert not torch.equal(a.encoder, b.encoder)
    old_ptr = b._encoder_w_lin.data_ptr()
    b.load_state_dict(a.state_dict())
    assert torch.equal(a.encoder, b.encoder)
    assert b._encoder_w_lin is not None
    assert torch.equal(b._encoder_w_lin, bdh.BDH._hdn_as_linear_weight(b.encoder))
    x = torch.randint(0, cfg.vocab_size, (2, 8))
    with torch.no_grad():
        la, _ = a(x)
        lb, _ = b(x)
    assert torch.equal(la, lb)
