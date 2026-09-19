"""Embedding + LM-head path: untied default, optional tie, contiguous vocab proj."""

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


def test_baseline_does_not_tie_embed_lm_head():
    """Published semantics: separate embed (V,D) and lm_head (D,V) Parameters."""
    cfg = _cfg()
    m = baseline.BDH(cfg)
    assert "embed.weight" in dict(m.named_parameters())
    assert "lm_head" in dict(m.named_parameters())
    assert m.embed.weight.shape == (cfg.vocab_size, cfg.n_embd)
    assert m.lm_head.shape == (cfg.n_embd, cfg.vocab_size)
    # Distinct storages (not GPT-tied).
    assert m.embed.weight.data_ptr() != m.lm_head.data_ptr()


def test_default_untied_matches_baseline_logits():
    cfg = _cfg()
    assert cfg.tie_weights is False
    torch.manual_seed(11)
    m_base = baseline.BDH(cfg)
    m_opt = bdh.BDH(cfg)
    m_opt.load_state_dict(m_base.state_dict(), strict=True)
    m_base.eval()
    m_opt.eval()
    x = torch.randint(0, cfg.vocab_size, (3, 20))
    lb, lossb = m_base(x, x)
    lo, losso = m_opt(x, x)
    assert torch.equal(lb, lo)
    assert torch.equal(lossb, losso)
    assert m_opt.lm_head is not None
    assert m_opt.lm_head_bias is None


def test_vocab_logits_bit_identical_to_matmul():
    cfg = _cfg()
    torch.manual_seed(3)
    m = bdh.BDH(cfg).eval()
    x4 = torch.randn(2, 1, 17, cfg.n_embd)
    via_helper = m._vocab_logits(x4)
    via_mm = x4.view(2, 17, cfg.n_embd) @ m.lm_head
    assert torch.equal(via_helper, via_mm)
    assert via_helper.is_contiguous()
    assert via_helper.shape == (2, 17, cfg.vocab_size)


def test_embed_tokens_matches_unsqueeze_then_ln():
    cfg = _cfg()
    torch.manual_seed(5)
    m = bdh.BDH(cfg).eval()
    idx = torch.randint(0, cfg.vocab_size, (2, 9))
    got = m._embed_tokens(idx)
    ref = m._ln(m.embed(idx).unsqueeze(1))
    # Helper does LN then unsqueeze; equivalent last-dim LN.
    alt = m._ln(m.embed(idx)).unsqueeze(1)
    assert torch.equal(got, alt)
    assert torch.equal(got, ref)
    assert got.shape == (2, 1, 9, cfg.n_embd)


def test_optional_tie_weights_shares_embed():
    cfg = _cfg(tie_weights=True)
    m = bdh.BDH(cfg)
    assert m.lm_head is None
    names = dict(m.named_parameters())
    assert "embed.weight" in names
    assert "lm_head" not in names

    m.eval()
    idx = torch.randint(0, cfg.vocab_size, (2, 8))
    logits, _ = m(idx)
    # Manual GPT-style projection from final hidden would need a full forward;
    # check helper directly on a fake hidden.
    h = torch.randn(2, 1, 8, cfg.n_embd)
    got = m._vocab_logits(h)
    ref = F.linear(h.squeeze(1), m.embed.weight)
    assert torch.equal(got, ref)


def test_tied_forward_runs_and_shapes():
    cfg = _cfg(tie_weights=True, n_layer=1)
    m = bdh.BDH(cfg).eval()
    x = torch.randint(0, cfg.vocab_size, (2, 12))
    logits, loss = m(x, x)
    assert logits.shape == (2, 12, cfg.vocab_size)
    assert loss is not None and loss.ndim == 0
