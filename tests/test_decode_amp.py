"""Decode-path CacheManager + optional AMP correctness vs fp32.

Growth/no-realloc is owned by bdh_cache.CacheManager (cache-pack). This suite
covers single-token decode under BDH_ATTN_IMPL and AMP autocast.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import bdh
from bdh_cache import CacheManager


# Documented tolerances for AMP vs fp32 (CPU autocast). Not bit-identical.
AMP_ATOL = {
    torch.float16: 5e-2,
    torch.bfloat16: 5e-2,
}
AMP_RTOL = {
    torch.float16: 5e-2,
    torch.bfloat16: 5e-2,
}
# Tokenwise decode accumulates error; observed bf16 max ~0.3 on this box.
AMP_DECODE_ATOL = {
    torch.float16: 5e-1,
    torch.bfloat16: 5e-1,
}
AMP_DECODE_RTOL = {
    torch.float16: 5e-1,
    torch.bfloat16: 5e-1,
}


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


def test_cache_manager_prealloc_no_step_realloc():
    """Packed cache buffers keep identity across decode appends."""
    cfg = _small_cfg()
    cm = CacheManager.from_config(cfg, batch_size=1, max_seq=32, device="cpu")
    N = cfg.mlp_internal_dim_multiplier * cfg.n_embd // cfg.n_head
    buf_id = id(cm._kr[0])
    for step in range(8):
        for level in range(cfg.n_layer):
            cm.append(
                level,
                torch.randn(1, cfg.n_head, 1, N),
                torch.randn(1, 1, 1, cfg.n_embd),
            )
        cm.commit()
        assert id(cm._kr[0]) == buf_id
    assert cm.seq_len == 8


def test_tokenwise_packed_decode_matches_full():
    cfg = _small_cfg()
    torch.manual_seed(31)
    model = bdh.BDH(cfg).eval()
    tokens = torch.randint(0, cfg.vocab_size, (2, 20))
    with torch.no_grad():
        full, _ = model(tokens)
        cm = CacheManager.from_config(cfg, tokens.size(0), tokens.size(1), tokens.device)
        parts = []
        for t in range(tokens.size(1)):
            lg, _ = model(tokens[:, t : t + 1], cache=cm)
            parts.append(lg)
        got = torch.cat(parts, dim=1)
    assert torch.allclose(full, got, rtol=1e-5, atol=1e-5)
    assert cm.seq_len == 20


def test_prefill_then_decode_with_cache_manager():
    cfg = _small_cfg()
    torch.manual_seed(32)
    model = bdh.BDH(cfg).eval()
    tokens = torch.randint(0, cfg.vocab_size, (1, 12))
    with torch.no_grad():
        full, _ = model(tokens)
        cm = CacheManager.from_config(cfg, 1, 12, tokens.device)
        pre, _ = model(tokens[:, :5], cache=cm)
        assert cm.seq_len == 5
        rest = []
        for t in range(5, 12):
            lg, _ = model(tokens[:, t : t + 1], cache=cm)
            rest.append(lg)
        got = torch.cat([pre] + rest, dim=1)
    assert torch.allclose(full, got, rtol=1e-5, atol=1e-5)
    assert cm.seq_len == 12


@pytest.mark.parametrize("impl", ["eager", "blocked", "triton"])
def test_single_token_decode_matches_full_under_attn_impl(impl, monkeypatch):
    """Incremental T=1 steps stay correct for every BDH_ATTN_IMPL."""
    monkeypatch.setenv("BDH_ATTN_IMPL", impl)
    cfg = _small_cfg()
    torch.manual_seed(40 + hash(impl) % 50)
    model = bdh.BDH(cfg).eval()
    tokens = torch.randint(0, cfg.vocab_size, (1, 14))
    with torch.no_grad():
        full, _ = model(tokens)
        cm = CacheManager.from_config(cfg, 1, tokens.size(1), tokens.device)
        parts = []
        for t in range(tokens.size(1)):
            lg, _ = model(tokens[:, t : t + 1], cache=cm)
            parts.append(lg)
        got = torch.cat(parts, dim=1)
    assert torch.allclose(full, got, rtol=1e-5, atol=1e-5), (
        f"impl={impl} max={(full - got).abs().max().item()}"
    )


@pytest.mark.parametrize("impl", ["eager", "blocked"])
def test_chunked_cache_continuation_under_attn_impl(impl, monkeypatch):
    """Prefill + multi-token continuation with past (blocked uses concat path)."""
    monkeypatch.setenv("BDH_ATTN_IMPL", impl)
    cfg = _small_cfg()
    torch.manual_seed(55)
    model = bdh.BDH(cfg).eval()
    tokens = torch.randint(0, cfg.vocab_size, (1, 16))
    S, chunk = 6, 4
    with torch.no_grad():
        full, _ = model(tokens)
        cm = CacheManager.from_config(cfg, 1, tokens.size(1), tokens.device)
        pre, _ = model(tokens[:, :S], cache=cm)
        mid, _ = model(tokens[:, S : S + chunk], cache=cm)
        rest_parts = []
        for t in range(S + chunk, tokens.size(1)):
            lg, _ = model(tokens[:, t : t + 1], cache=cm)
            rest_parts.append(lg)
        rest = torch.cat(rest_parts, dim=1)
        got = torch.cat([pre, mid, rest], dim=1)
    assert torch.allclose(full, got, rtol=1e-5, atol=1e-5), (
        f"impl={impl} max={(full - got).abs().max().item()}"
    )


@pytest.mark.parametrize("amp_dtype", [torch.float16, torch.bfloat16])
def test_amp_forward_close_to_fp32(amp_dtype):
    """Autocast forward vs fp32 reference; documented atol/rtol."""
    cfg = _small_cfg()
    torch.manual_seed(70)
    model = bdh.BDH(cfg).eval()
    tokens = torch.randint(0, cfg.vocab_size, (2, 16))
    with torch.no_grad():
        ref, _ = model(tokens)
        with bdh._autocast_context(tokens.device, amp_dtype):
            amp_logits, _ = model(tokens)
        amp_f = amp_logits.float()
    atol, rtol = AMP_ATOL[amp_dtype], AMP_RTOL[amp_dtype]
    max_diff = (ref - amp_f).abs().max().item()
    assert torch.allclose(ref, amp_f, rtol=rtol, atol=atol), (
        f"{amp_dtype} vs fp32 max abs diff={max_diff} (atol={atol}, rtol={rtol})"
    )


@pytest.mark.parametrize("amp_dtype", [torch.float16, torch.bfloat16])
def test_amp_cached_decode_close_to_fp32(amp_dtype):
    cfg = _small_cfg()
    torch.manual_seed(71)
    model = bdh.BDH(cfg).eval()
    tokens = torch.randint(0, cfg.vocab_size, (1, 12))
    atol, rtol = AMP_DECODE_ATOL[amp_dtype], AMP_DECODE_RTOL[amp_dtype]
    with torch.no_grad():
        ref, _ = model(tokens)
        cm = CacheManager.from_config(cfg, 1, tokens.size(1), tokens.device)
        parts = []
        with bdh._autocast_context(tokens.device, amp_dtype):
            for t in range(tokens.size(1)):
                lg, _ = model(tokens[:, t : t + 1], cache=cm)
                parts.append(lg.float())
        got = torch.cat(parts, dim=1)
    assert torch.allclose(ref, got, rtol=rtol, atol=atol), (
        f"amp decode {amp_dtype} max={(ref - got).abs().max().item()}"
    )


@pytest.mark.parametrize("amp_dtype", [torch.float16, torch.bfloat16])
def test_generate_amp_dtype_runs_and_extends(amp_dtype):
    cfg = _small_cfg()
    torch.manual_seed(72)
    model = bdh.BDH(cfg).eval()
    prompt = torch.randint(0, cfg.vocab_size, (1, 4))
    out = model.generate(prompt.clone(), max_new_tokens=6, amp_dtype=amp_dtype)
    assert out.shape == (1, 10)
    assert torch.equal(out[:, :4], prompt)


def test_generate_default_fp32_unchanged_vs_baseline_rng():
    """Default amp_dtype=None must keep prior generate semantics (fp32)."""
    import bdh_baseline as baseline

    cfg = _small_cfg()
    torch.manual_seed(11)
    m_base = baseline.BDH(cfg)
    m_opt = bdh.BDH(cfg)
    m_opt.load_state_dict(m_base.state_dict())
    m_base.eval()
    m_opt.eval()
    prompt = torch.randint(0, cfg.vocab_size, (1, 4))
    torch.manual_seed(99)
    g1 = m_base.generate(prompt.clone(), max_new_tokens=8, temperature=1.0)
    torch.manual_seed(99)
    g2 = m_opt.generate(prompt.clone(), max_new_tokens=8, temperature=1.0)
    assert torch.equal(g1, g2)


def test_generate_cache_dtype_and_amp_compose():
    cfg = _small_cfg()
    torch.manual_seed(80)
    model = bdh.BDH(cfg).eval()
    prompt = torch.randint(0, cfg.vocab_size, (1, 3))
    out = model.generate(
        prompt.clone(),
        max_new_tokens=4,
        cache_dtype=torch.float16,
        amp_dtype=torch.bfloat16,
    )
    assert out.shape == (1, 7)


def test_amp_dtype_rejects_invalid():
    with pytest.raises(ValueError):
        with bdh._autocast_context(torch.device("cpu"), torch.float64):
            pass


def test_attention_excludes_diagonal_unchanged():
    cfg = _small_cfg()
    attn = bdh.Attention(cfg)
    N = cfg.mlp_internal_dim_multiplier * cfg.n_embd // cfg.n_head
    Q = torch.randn(2, cfg.n_head, 8, N)
    V = torch.randn(2, 1, 8, cfg.n_embd)
    out, _, _ = attn(Q, Q, V)
    assert torch.count_nonzero(out[:, :, 0, :]) == 0
