"""Opt-in BDH_ATTN_AUTO: long-T cold + long-S decode → triton|blocked; default eager.

Optional ``BDH_ATTN_AUTO_COLD_THRESHOLD`` (defaults to ``AUTO_THRESHOLD``).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import bdh
from kernels.attention import (
    blocked_decode_attn,
    eager_decode_attn,
    triton_decode_attn,
    triton_decode_available,
)
from kernels.attention_dispatch import (
    DEFAULT_ATTN_AUTO_THRESHOLD,
    attn_auto_cold_threshold,
    attn_auto_enabled,
    attn_auto_threshold,
    backend_info,
    bdh_attn,
    bdh_attn_decode,
    resolve_attn_impl,
    resolve_cold_impl,
    resolve_decode_impl,
)


def _bump_caches():
    """Force env-cache re-read after monkeypatch setenv/delenv."""
    import kernels.attention_dispatch as d

    d._ATTN_IMPL_ENV = object()
    d._ATTN_AUTO_ENV = object()
    d._ATTN_AUTO_THR_ENV = object()
    d._ATTN_AUTO_COLD_THR_ENV = object()


def _make_decode(S, B=2, H=4, N=16, D=32, seed=0):
    g = torch.Generator().manual_seed(seed)
    Q = torch.randn(B, H, 1, N, generator=g)
    K = torch.randn(B, H, S, N, generator=g)
    V = torch.randn(B, 1, S, D, generator=g)
    return Q, K, V


def _auto_long_s_impl() -> str:
    """What AUTO selects above threshold: triton on CUDA+Triton, else blocked."""
    return "triton" if triton_decode_available() else "blocked"



def test_default_auto_off(monkeypatch):
    monkeypatch.delenv("BDH_ATTN_AUTO", raising=False)
    monkeypatch.delenv("BDH_ATTN_AUTO_THRESHOLD", raising=False)
    monkeypatch.delenv("BDH_ATTN_AUTO_COLD_THRESHOLD", raising=False)
    monkeypatch.delenv("BDH_ATTN_IMPL", raising=False)
    _bump_caches()
    assert not attn_auto_enabled()
    assert attn_auto_threshold() == DEFAULT_ATTN_AUTO_THRESHOLD == 512
    assert attn_auto_cold_threshold() == 512
    assert resolve_attn_impl() == "eager"
    assert resolve_decode_impl(4096) == "eager"
    assert resolve_cold_impl(4096) == "eager"
    info = backend_info()
    assert info["BDH_ATTN_AUTO"] is False
    assert info["BDH_ATTN_AUTO_THRESHOLD"] == 512
    assert info["BDH_ATTN_AUTO_COLD_THRESHOLD"] == 512


@pytest.mark.parametrize(
    "flag,expect",
    [("1", True), ("true", True), ("YES", True), ("on", True), ("0", False), ("", False)],
)
def test_auto_truthy_parsing(monkeypatch, flag, expect):
    if flag == "":
        monkeypatch.delenv("BDH_ATTN_AUTO", raising=False)
    else:
        monkeypatch.setenv("BDH_ATTN_AUTO", flag)
    monkeypatch.delenv("BDH_ATTN_IMPL", raising=False)
    _bump_caches()
    assert attn_auto_enabled() is expect


def test_auto_switches_eager_decode_past_threshold(monkeypatch):
    monkeypatch.setenv("BDH_ATTN_AUTO", "1")
    monkeypatch.delenv("BDH_ATTN_AUTO_THRESHOLD", raising=False)
    monkeypatch.delenv("BDH_ATTN_IMPL", raising=False)
    _bump_caches()
    thr = attn_auto_threshold()
    assert resolve_decode_impl(thr) == "eager"
    assert resolve_decode_impl(thr + 1) == _auto_long_s_impl()
    assert resolve_cold_impl(thr + 1) == _auto_long_s_impl()
    # base IMPL env still eager (AUTO is length-gated, not an IMPL flip)
    assert resolve_attn_impl() == "eager"


def test_auto_custom_threshold(monkeypatch):
    monkeypatch.setenv("BDH_ATTN_AUTO", "1")
    monkeypatch.setenv("BDH_ATTN_AUTO_THRESHOLD", "128")
    monkeypatch.delenv("BDH_ATTN_AUTO_COLD_THRESHOLD", raising=False)
    monkeypatch.delenv("BDH_ATTN_IMPL", raising=False)
    _bump_caches()
    assert attn_auto_threshold() == 128
    assert attn_auto_cold_threshold() == 128
    assert resolve_decode_impl(128) == "eager"
    assert resolve_decode_impl(129) == _auto_long_s_impl()
    assert resolve_cold_impl(129) == _auto_long_s_impl()


def test_auto_does_not_override_explicit_non_eager(monkeypatch):
    monkeypatch.setenv("BDH_ATTN_AUTO", "1")
    monkeypatch.delenv("BDH_ATTN_AUTO_THRESHOLD", raising=False)
    for impl in ("blocked", "triton", "cuda", "online"):
        monkeypatch.setenv("BDH_ATTN_IMPL", impl)
        _bump_caches()
        want = "blocked" if impl == "online" else impl
        assert resolve_decode_impl(4096) == want
        assert resolve_decode_impl(4096, requested=impl) == want


def test_auto_decode_parity_vs_eager_and_blocked(monkeypatch):
    monkeypatch.setenv("BDH_ATTN_AUTO", "1")
    monkeypatch.delenv("BDH_ATTN_AUTO_THRESHOLD", raising=False)
    monkeypatch.delenv("BDH_ATTN_IMPL", raising=False)
    _bump_caches()
    # below threshold → eager path
    Q, K, V = _make_decode(S=64, seed=11)
    got = bdh_attn_decode(Q, K, V)
    assert torch.allclose(got, eager_decode_attn(Q, K, V), rtol=1e-5, atol=1e-5)
    # above threshold → triton|blocked (CPU: blocked via triton fallback; parity)
    Q, K, V = _make_decode(S=600, seed=12)
    got = bdh_attn_decode(Q, K, V)
    ref_e = eager_decode_attn(Q, K, V)
    ref_b = blocked_decode_attn(Q, K, V)
    ref_t = triton_decode_attn(Q, K, V)
    assert torch.allclose(got, ref_e, rtol=1e-4, atol=1e-5)
    assert torch.allclose(got, ref_b, rtol=1e-5, atol=1e-5)
    assert torch.allclose(got, ref_t, rtol=1e-5, atol=1e-5)
    assert resolve_decode_impl(600) == _auto_long_s_impl()


def test_auto_off_long_s_stays_eager(monkeypatch):
    monkeypatch.delenv("BDH_ATTN_AUTO", raising=False)
    monkeypatch.delenv("BDH_ATTN_IMPL", raising=False)
    _bump_caches()
    Q, K, V = _make_decode(S=600, seed=13)
    got = bdh_attn_decode(Q, K, V)
    assert torch.allclose(got, eager_decode_attn(Q, K, V), atol=0)
    assert resolve_decode_impl(600) == "eager"


def test_cold_path_short_stays_eager_under_auto(monkeypatch):
    """Short cold/prefill (T ≤ thr) stays eager under AUTO — prior semantics."""
    monkeypatch.setenv("BDH_ATTN_AUTO", "1")
    monkeypatch.delenv("BDH_ATTN_IMPL", raising=False)
    monkeypatch.delenv("BDH_ATTN_AUTO_THRESHOLD", raising=False)
    _bump_caches()
    g = torch.Generator().manual_seed(21)
    Q = torch.randn(2, 4, 16, 16, generator=g)
    V = torch.randn(2, 1, 16, 32, generator=g)
    out = bdh_attn(Q, Q, V)
    from kernels.attention import eager_tril_attn

    assert resolve_cold_impl(16) == "eager"
    assert torch.allclose(out, eager_tril_attn(Q, Q, V), atol=0)


def test_cold_path_auto_long_t(monkeypatch):
    """Long cold/prefill (T > thr) switches under AUTO — same knobs as decode."""
    monkeypatch.setenv("BDH_ATTN_AUTO", "1")
    monkeypatch.delenv("BDH_ATTN_IMPL", raising=False)
    monkeypatch.delenv("BDH_ATTN_AUTO_THRESHOLD", raising=False)
    _bump_caches()
    thr = attn_auto_threshold()
    assert resolve_cold_impl(thr) == "eager"
    assert resolve_cold_impl(thr + 1) == _auto_long_s_impl()
    # Parity vs eager at long T (fp tile reorder).
    from kernels.attention import eager_tril_attn, blocked_tril_attn

    g = torch.Generator().manual_seed(22)
    T = thr + 1
    Q = torch.randn(1, 2, T, 16, generator=g)
    V = torch.randn(1, 1, T, 32, generator=g)
    out = bdh_attn(Q, Q, V)
    ref = eager_tril_attn(Q, Q, V)
    assert torch.allclose(out, ref, rtol=1e-3, atol=1e-3)
    assert torch.allclose(out, blocked_tril_attn(Q, Q, V), rtol=1e-5, atol=1e-5)


def test_attention_module_auto_decode(monkeypatch):
    monkeypatch.setenv("BDH_ATTN_AUTO", "1")
    monkeypatch.delenv("BDH_ATTN_AUTO_THRESHOLD", raising=False)
    monkeypatch.delenv("BDH_ATTN_IMPL", raising=False)
    _bump_caches()
    cfg = bdh.BDHConfig(
        n_layer=1,
        n_embd=64,
        n_head=4,
        mlp_internal_dim_multiplier=8,
        dropout=0.0,
        vocab_size=256,
    )
    attn = bdh.Attention(cfg)
    N = cfg.mlp_internal_dim_multiplier * cfg.n_embd // cfg.n_head
    B, S = 1, 600
    torch.manual_seed(3)
    Q = torch.randn(B, cfg.n_head, 1, N)
    past_kr = torch.randn(B, cfg.n_head, S, N)
    past_v = torch.randn(B, 1, S, cfg.n_embd)
    V = torch.randn(B, 1, 1, cfg.n_embd)
    out, _, _ = attn(Q, Q, V, rope_start=S, past_kr=past_kr, past_v=past_v)
    # poison V_new must not affect (tril -1)
    out2, _, _ = attn(
        Q, Q, torch.randn_like(V) * 99, rope_start=S, past_kr=past_kr, past_v=past_v
    )
    assert torch.allclose(out, out2, atol=0)
    assert resolve_decode_impl(S) == _auto_long_s_impl()



def test_cold_threshold_defaults_to_decode_thr(monkeypatch):
    """Unset COLD_THRESHOLD mirrors AUTO_THRESHOLD."""
    monkeypatch.setenv("BDH_ATTN_AUTO", "1")
    monkeypatch.setenv("BDH_ATTN_AUTO_THRESHOLD", "128")
    monkeypatch.delenv("BDH_ATTN_AUTO_COLD_THRESHOLD", raising=False)
    monkeypatch.delenv("BDH_ATTN_IMPL", raising=False)
    _bump_caches()
    assert attn_auto_threshold() == 128
    assert attn_auto_cold_threshold() == 128
    assert resolve_cold_impl(128) == "eager"
    assert resolve_cold_impl(129) == _auto_long_s_impl()
    assert resolve_decode_impl(129) == _auto_long_s_impl()


def test_independent_cold_threshold(monkeypatch):
    """COLD_THRESHOLD can be lower than decode thr for peak-mem mid-T (#73)."""
    monkeypatch.setenv("BDH_ATTN_AUTO", "1")
    monkeypatch.setenv("BDH_ATTN_AUTO_THRESHOLD", "512")
    monkeypatch.setenv("BDH_ATTN_AUTO_COLD_THRESHOLD", "256")
    monkeypatch.delenv("BDH_ATTN_IMPL", raising=False)
    _bump_caches()
    assert attn_auto_threshold() == 512
    assert attn_auto_cold_threshold() == 256
    # Mid-T: cold switches, decode still eager
    assert resolve_cold_impl(256) == "eager"
    assert resolve_cold_impl(257) == _auto_long_s_impl()
    assert resolve_decode_impl(257) == "eager"
    assert resolve_decode_impl(513) == _auto_long_s_impl()
    info = backend_info()
    assert info["BDH_ATTN_AUTO_COLD_THRESHOLD"] == 256
    assert info["BDH_ATTN_AUTO_THRESHOLD"] == 512


def test_invalid_cold_threshold_raises(monkeypatch):
    monkeypatch.setenv("BDH_ATTN_AUTO_COLD_THRESHOLD", "nope")
    _bump_caches()
    with pytest.raises(ValueError, match="BDH_ATTN_AUTO_COLD_THRESHOLD"):
        attn_auto_cold_threshold()


def test_invalid_threshold_raises(monkeypatch):
    monkeypatch.setenv("BDH_ATTN_AUTO_THRESHOLD", "nope")
    _bump_caches()
    with pytest.raises(ValueError, match="BDH_ATTN_AUTO_THRESHOLD"):
        attn_auto_threshold()


def test_auto_prefers_triton_when_available(monkeypatch):
    """Document AUTO → triton on CUDA+Triton, else blocked (#55 CPU path)."""
    monkeypatch.setenv("BDH_ATTN_AUTO", "1")
    monkeypatch.delenv("BDH_ATTN_AUTO_THRESHOLD", raising=False)
    monkeypatch.delenv("BDH_ATTN_IMPL", raising=False)
    _bump_caches()
    info = backend_info()
    assert info["auto_decode_prefers"] == _auto_long_s_impl()
    assert info["triton_decode_available"] == triton_decode_available()
    # On this CPU box: False → blocked; GPU boxes with Triton: True → triton.
    if triton_decode_available():
        assert resolve_decode_impl(4096) == "triton"
    else:
        assert resolve_decode_impl(4096) == "blocked"
        assert not torch.cuda.is_available() or not info["has_triton"]


def test_backend_info_auto_decode_fields(monkeypatch):
    monkeypatch.delenv("BDH_ATTN_AUTO", raising=False)
    monkeypatch.delenv("BDH_ATTN_IMPL", raising=False)
    _bump_caches()
    info = backend_info()
    assert "triton_decode_available" in info
    assert "auto_decode_prefers" in info
    assert info["auto_decode_prefers"] in ("triton", "blocked")
    assert "auto_cold_prefers" in info
    assert info["auto_cold_prefers"] == info["auto_decode_prefers"]

