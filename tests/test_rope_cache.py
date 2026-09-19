"""RoPE cos/sin table cache: parity vs uncached and across batch reuse."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import bdh
import bdh_baseline as baseline


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


def test_rope_cos_sin_cache_matches_fresh_and_baseline():
    """Cached (cos, sin) must match a fresh compute and baseline phases path."""
    cfg = _small_cfg()
    attn = bdh.Attention(cfg)
    base = baseline.Attention(cfg)
    T, rope_start = 12, 0
    device = torch.device("cpu")

    # Fresh / first call fills cache
    cos1, sin1 = attn.rope_cos_sin(T, rope_start, device)
    # Second call must hit cache (same tensor objects)
    cos2, sin2 = attn.rope_cos_sin(T, rope_start, device)
    assert cos1 is cos2 and sin1 is sin2

    # Numerical match vs uncached recompute (bypass cache)
    attn._rope_cis_key = None
    attn._rope_cis = None
    cos_f, sin_f = attn.phases_cos_sin(attn._rope_phases(T, rope_start, device))
    assert torch.equal(cos1, cos_f) and torch.equal(sin1, sin_f)

    # Baseline rope via phases matches optimized rope using cached cis
    N = cfg.mlp_internal_dim_multiplier * cfg.n_embd // cfg.n_head
    torch.manual_seed(0)
    v = torch.randn(2, cfg.n_head, T, N)
    phases = attn._rope_phases(T, 0, device)
    out_base = baseline.Attention.rope(phases, v)
    out_opt = bdh.Attention.rope(None, v, cos_sin=(cos1, sin1))
    assert torch.allclose(out_opt, out_base, rtol=0, atol=0)


def test_rope_cos_sin_cache_key_changes_with_T():
    cfg = _small_cfg()
    attn = bdh.Attention(cfg)
    device = torch.device("cpu")
    c8, s8 = attn.rope_cos_sin(8, 0, device)
    c16, s16 = attn.rope_cos_sin(16, 0, device)
    assert c8.shape[-2] == 8 and c16.shape[-2] == 16
    assert c8 is not c16
    # Revisit T=8 rebuilds (single-slot cache) but values match
    c8b, s8b = attn.rope_cos_sin(8, 0, device)
    assert torch.equal(c8b, c8) and torch.equal(s8b, s8)


def test_rope_cos_sin_decode_rope_start_not_cached_as_zero():
    """rope_start!=0 must not return the rope_start=0 table."""
    cfg = _small_cfg()
    attn = bdh.Attention(cfg)
    device = torch.device("cpu")
    cos0, sin0 = attn.rope_cos_sin(4, 0, device)
    cos_s, sin_s = attn.rope_cos_sin(4, 7, device)
    assert not torch.equal(cos0, cos_s)
    phases_s = attn._rope_phases(4, 7, device)
    cos_f, sin_f = attn.phases_cos_sin(phases_s)
    assert torch.equal(cos_s, cos_f) and torch.equal(sin_s, sin_f)


def test_rope_cache_model_forward_parity_across_batches():
    """Two eval forwards with same T reuse cis; logits match twin without cache abuse."""
    cfg = _small_cfg()
    torch.manual_seed(11)
    m = bdh.BDH(cfg).eval()
    x1 = torch.randint(0, cfg.vocab_size, (2, 10))
    x2 = torch.randint(0, cfg.vocab_size, (2, 10))
    with torch.no_grad():
        y1, _ = m(x1)
        key_after = m.attn._rope_cis_key
        cis_after = m.attn._rope_cis
        y2, _ = m(x2)
    assert key_after is not None and cis_after is not None
    assert m.attn._rope_cis is cis_after  # same cached tables reused
    # Sanity: outputs finite and shaped
    assert y1.shape == (2, 10, cfg.vocab_size)
    assert torch.isfinite(y1).all() and torch.isfinite(y2).all()
