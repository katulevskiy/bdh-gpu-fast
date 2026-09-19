"""Host-path generate opts: RoPE table slices, cat=0, tokens match tip semantics."""

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


def _model(cfg, seed=0):
    torch.manual_seed(seed)
    return bdh.BDH(cfg).eval()


def test_ensure_rope_table_decode_slices_match_fresh():
    """ensure_rope_table + narrow must match fresh rope_start!=0 cis."""
    cfg = _small_cfg()
    attn = bdh.Attention(cfg)
    device = torch.device("cpu")
    attn.ensure_rope_table(32, device)
    for rope_start, T in [(0, 8), (7, 4), (16, 1), (31, 1)]:
        cos_s, sin_s = attn.rope_cos_sin(T, rope_start, device)
        # Bypass table for fresh reference
        cos_f, sin_f = attn.phases_cos_sin(attn._rope_phases(T, rope_start, device))
        assert torch.equal(cos_s, cos_f) and torch.equal(sin_s, sin_f)
        assert cos_s.shape[-2] == T


def test_ensure_rope_table_noop_on_second_call():
    cfg = _small_cfg()
    attn = bdh.Attention(cfg)
    device = torch.device("cpu")
    attn.ensure_rope_table(16, device)
    t0 = attn._rope_table
    attn.ensure_rope_table(16, device)
    assert attn._rope_table is t0


def test_generate_rope_arange_once_per_call():
    """With warmed table, generate should arange once (table build), not per step."""
    cfg = _small_cfg(n_layer=2)
    m = _model(cfg, seed=0)
    prompt = torch.randint(0, cfg.vocab_size, (1, 4))
    n_new = 8
    counts = {"n": 0}
    orig = torch.arange

    def hooked(*a, **k):
        counts["n"] += 1
        return orig(*a, **k)

    torch.arange = hooked  # type: ignore[assignment]
    try:
        torch.manual_seed(0)
        out = m.generate(prompt.clone(), max_new_tokens=n_new, temperature=1.0)
    finally:
        torch.arange = orig  # type: ignore[assignment]

    assert out.shape == (1, 4 + n_new)
    # One arange for ensure_rope_table(max_seq); decode/prefill must not re-arange.
    assert counts["n"] == 1, f"expected 1 torch.arange in generate, got {counts['n']}"


def test_generate_zero_cat_and_matches_baseline_tokens():
    cfg = _small_cfg()
    m_opt = _model(cfg, seed=42)
    torch.manual_seed(42)
    m_base = baseline.BDH(cfg).eval()
    m_base.load_state_dict(m_opt.state_dict())

    prompt = torch.randint(0, cfg.vocab_size, (1, 5))
    cats = {"n": 0}
    orig = torch.cat

    def hooked(*a, **k):
        cats["n"] += 1
        return orig(*a, **k)

    torch.cat = hooked  # type: ignore[assignment]
    try:
        torch.manual_seed(0)
        g_opt = m_opt.generate(prompt.clone(), max_new_tokens=6, temperature=1.0)
    finally:
        torch.cat = orig  # type: ignore[assignment]

    torch.manual_seed(0)
    g_base = m_base.generate(prompt.clone(), max_new_tokens=6, temperature=1.0)
    assert cats["n"] == 0
    assert torch.equal(g_opt, g_base)


def test_generate_topk_path_runs():
    cfg = _small_cfg()
    m = _model(cfg, seed=1)
    prompt = torch.randint(0, cfg.vocab_size, (2, 3))
    torch.manual_seed(3)
    out = m.generate(prompt.clone(), max_new_tokens=4, temperature=0.8, top_k=10)
    assert out.shape == (2, 7)


def test_resolve_attn_impl_env_cache(monkeypatch):
    from kernels.attention_dispatch import resolve_attn_impl
    import kernels.attention_dispatch as ad

    monkeypatch.setenv("BDH_ATTN_IMPL", "eager")
    ad._ATTN_IMPL_ENV = object()  # force refresh
    a = resolve_attn_impl()
    assert a == "eager"
    # Second call hits cache (same env string)
    b = resolve_attn_impl()
    assert b == "eager"
    monkeypatch.setenv("BDH_ATTN_IMPL", "blocked")
    assert resolve_attn_impl() == "blocked"
    # Explicit override bypasses env cache
    assert resolve_attn_impl("cuda") == "cuda"


def test_generate_hoists_attn_impl_clears_after():
    """generate sets _attn_impl_override for the loop and clears it after."""
    cfg = _small_cfg()
    m = _model(cfg, seed=0)
    assert m.attn._attn_impl_override is None
    prompt = torch.randint(0, cfg.vocab_size, (1, 3))
    torch.manual_seed(0)
    out = m.generate(prompt.clone(), max_new_tokens=3, temperature=1.0)
    assert out.shape == (1, 6)
    assert m.attn._attn_impl_override is None


def test_generate_environ_get_reduced_vs_unhoisted(monkeypatch):
    """With hoist, os.environ.get during generate should be far below n_layer*steps*2."""
    import os
    cfg = _small_cfg(n_layer=2)
    m = _model(cfg, seed=0)
    prompt = torch.randint(0, cfg.vocab_size, (1, 4))
    n_new = 8
    counts = {"n": 0}
    orig = os.environ.get

    def hooked(*a, **k):
        counts["n"] += 1
        return orig(*a, **k)

    monkeypatch.setattr(os.environ, "get", hooked)
    torch.manual_seed(0)
    m.generate(prompt.clone(), max_new_tokens=n_new, temperature=1.0)
    # Unhoisted tip: ~2 backends × n_layer × (1 prefill + n_new) ≈ 36+.
    # Hoisted: attn override removes decode attn gets; rope still cached-get
    # per layer. Expect well below the unhoisted floor.
    unhoisted_floor = 2 * cfg.n_layer * (1 + n_new)  # 36 for this cfg
    assert counts["n"] < unhoisted_floor, (
        f"environ.get {counts['n']} not below unhoisted floor {unhoisted_floor}"
    )
