"""Incremental single-token decode vs full forward; blocked/triton + CacheManager."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import bdh
from bdh_cache import CacheManager
from kernels.attention import (
    DEFAULT_BLOCK_DECODE,
    _pick_tile_size,
    _tiled_score_v,
    blocked_decode_attn,
    blocked_tril_attn,
    eager_decode_attn,
    eager_tril_attn,
    triton_decode_attn,
)
from kernels.attention_dispatch import bdh_attn_decode, resolve_attn_impl


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


def _model(cfg, seed: int = 0):
    torch.manual_seed(seed)
    return bdh.BDH(cfg).eval()


def _make_decode_qkv(B=2, H=4, S=32, N=16, D=64, Tq=1, seed=0):
    g = torch.Generator().manual_seed(seed)
    Q = torch.randn(B, H, Tq, N, generator=g)
    K = torch.randn(B, H, S, N, generator=g)
    V = torch.randn(B, 1, S, D, generator=g)
    return Q, K, V


def test_blocked_decode_matches_eager_ref():
    Q, K, V = _make_decode_qkv(S=48, Tq=1)
    ref = eager_decode_attn(Q, K, V)
    got = blocked_decode_attn(Q, K, V, block_size=8)
    assert torch.allclose(got, ref, rtol=1e-5, atol=1e-5)


def test_blocked_decode_tiled_path_matches_eager():
    """Force the Python tile loop (S > block_size and Tq*S > BS^2)."""
    Q, K, V = _make_decode_qkv(B=1, H=2, S=40, N=8, D=16, Tq=1, seed=7)
    ref = eager_decode_attn(Q, K, V)
    got = blocked_decode_attn(Q, K, V, block_size=4)
    assert torch.allclose(got, ref, rtol=1e-5, atol=1e-5)


def test_triton_decode_cpu_fallback_matches_eager():
    Q, K, V = _make_decode_qkv(S=24)
    got = triton_decode_attn(Q, K, V, block_size=8)
    ref = eager_decode_attn(Q, K, V)
    assert torch.allclose(got, ref, rtol=1e-5, atol=1e-5)


def test_decode_matches_full_tril_excluding_self():
    """Concat Q into K/V and run tril(-1); last query row must match decode."""
    B, H, S, N, D = 2, 4, 16, 8, 32
    g = torch.Generator().manual_seed(11)
    past_k = torch.randn(B, H, S, N, generator=g)
    past_v = torch.randn(B, 1, S, D, generator=g)
    q = torch.randn(B, H, 1, N, generator=g)
    # Full sequence: past + new; V for new is unused by last query under tril(-1)
    K_all = torch.cat([past_k, q], dim=2)
    V_new = torch.randn(B, 1, 1, D, generator=g)
    V_all = torch.cat([past_v, V_new], dim=2)
    full = eager_tril_attn(K_all, K_all, V_all)
    last = full[:, :, -1:, :]
    dec = blocked_decode_attn(q, past_k, past_v, block_size=5)
    assert torch.allclose(dec, last, rtol=1e-5, atol=1e-5)
    # Changing V_new must not affect last row (no self-attention)
    V_all2 = torch.cat([past_v, torch.randn_like(V_new)], dim=2)
    full2 = eager_tril_attn(K_all, K_all, V_all2)
    assert torch.allclose(full[:, :, -1:, :], full2[:, :, -1:, :], atol=0)


def test_decode_s0_returns_zeros():
    Q = torch.randn(1, 2, 1, 8)
    K = torch.zeros(1, 2, 0, 8)
    V = torch.zeros(1, 1, 0, 16)
    out = blocked_decode_attn(Q, K, V)
    assert out.shape == (1, 2, 1, 16)
    assert torch.all(out == 0)


@pytest.mark.parametrize("impl", ["eager", "blocked", "triton", "cuda"])
def test_dispatch_decode(impl):
    Q, K, V = _make_decode_qkv(S=20, seed=3)
    out = bdh_attn_decode(Q, K, V, impl=impl, block_size=7)
    ref = eager_decode_attn(Q, K, V)
    assert torch.allclose(out, ref, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("impl", ["eager", "blocked", "triton"])
def test_incremental_vs_full_forward(impl, monkeypatch):
    monkeypatch.setenv("BDH_ATTN_IMPL", impl)
    cfg = _small_cfg()
    model = _model(cfg, seed=41)
    tokens = torch.randint(0, cfg.vocab_size, (2, 12))
    with torch.no_grad():
        full, _ = model(tokens)
        cache = [None] * cfg.n_layer
        parts = []
        for t in range(tokens.size(1)):
            lg, _ = model(tokens[:, t : t + 1], cache=cache)
            parts.append(lg)
        got = torch.cat(parts, dim=1)
    assert torch.allclose(full, got, rtol=1e-5, atol=1e-5), (
        f"impl={impl} max abs={ (full - got).abs().max().item() }"
    )


@pytest.mark.parametrize("impl", ["eager", "blocked", "triton"])
def test_cachemanager_tokenwise_matches_full(impl, monkeypatch):
    monkeypatch.setenv("BDH_ATTN_IMPL", impl)
    cfg = _small_cfg()
    model = _model(cfg, seed=42)
    x = torch.randint(0, cfg.vocab_size, (1, 16))
    with torch.no_grad():
        full, _ = model(x)
        cm = CacheManager.from_config(cfg, 1, x.size(1), x.device)
        parts = []
        for t in range(x.size(1)):
            lg, _ = model(x[:, t : t + 1], cache=cm)
            parts.append(lg)
        got = torch.cat(parts, dim=1)
    assert cm.seq_len == x.size(1)
    assert torch.allclose(full, got, rtol=1e-5, atol=1e-5), (
        f"impl={impl} max abs={ (full - got).abs().max().item() }"
    )


@pytest.mark.parametrize("impl", ["eager", "blocked", "triton"])
def test_prefill_then_decode_matches_full(impl, monkeypatch):
    monkeypatch.setenv("BDH_ATTN_IMPL", impl)
    cfg = _small_cfg()
    model = _model(cfg, seed=43)
    x = torch.randint(0, cfg.vocab_size, (2, 10))
    S = 6
    with torch.no_grad():
        full, _ = model(x)
        cm = CacheManager.from_config(cfg, x.size(0), x.size(1), x.device)
        pre, _ = model(x[:, :S], cache=cm)
        assert torch.allclose(full[:, :S], pre, rtol=1e-5, atol=1e-5)
        parts = [pre]
        for t in range(S, x.size(1)):
            lg, _ = model(x[:, t : t + 1], cache=cm)
            parts.append(lg)
        got = torch.cat(parts, dim=1)
    assert torch.allclose(full, got, rtol=1e-5, atol=1e-5)


def test_default_eager_unchanged_without_env(monkeypatch):
    monkeypatch.delenv("BDH_ATTN_IMPL", raising=False)
    cfg = _small_cfg()
    model = _model(cfg, seed=44)
    x = torch.randint(0, cfg.vocab_size, (1, 8))
    with torch.no_grad():
        full, _ = model(x)
        cm = CacheManager.from_config(cfg, 1, 8, x.device)
        parts = []
        for t in range(8):
            lg, _ = model(x[:, t : t + 1], cache=cm)
            parts.append(lg)
        got = torch.cat(parts, dim=1)
    assert torch.allclose(full, got, rtol=1e-5, atol=1e-5)
    assert os.environ.get("BDH_ATTN_IMPL", "eager") == "eager"


def test_attention_module_t1_no_self_attend():
    """Direct Attention call: T=1 against past must ignore a poisoned V_new."""
    cfg = _small_cfg()
    attn = bdh.Attention(cfg)
    B, nh, S = 1, cfg.n_head, 5
    N = cfg.mlp_internal_dim_multiplier * cfg.n_embd // nh
    D = cfg.n_embd
    torch.manual_seed(9)
    Q = torch.randn(B, nh, 1, N)
    past_kr = torch.randn(B, nh, S, N)
    past_v = torch.randn(B, 1, S, D)
    V = torch.randn(B, 1, 1, D)
    out1, _, _ = attn(Q, Q, V, rope_start=S, past_kr=past_kr, past_v=past_v)
    out2, _, _ = attn(
        Q, Q, torch.randn_like(V) * 100, rope_start=S, past_kr=past_kr, past_v=past_v
    )
    assert torch.allclose(out1, out2, atol=0)


def test_pick_tile_size_prefers_fewer_trips_for_tq1():
    """Decode default tile is larger than cold 64; adaptive grows toward fewer tiles."""
    assert DEFAULT_BLOCK_DECODE >= 128
    # Fixed BS honored when S is small
    assert _pick_tile_size(40, 1, 256) == 40
    # Long past, Tq=1: tile grows above a small requested BS (fewer Python loops)
    big = _pick_tile_size(2048, 1, 64)
    assert big >= 64
    assert big <= 2048
    # Roughly ~8 tiles worth
    assert big >= 2048 // 8


def test_tiled_score_v_shared_with_cold_past_region():
    """Cold blocked past region == decode helper against K[:, :i0]."""
    B, H, T, N, D = 1, 2, 20, 8, 16
    g = torch.Generator().manual_seed(21)
    Q = torch.randn(B, H, T, N, generator=g)
    V = torch.randn(B, 1, T, D, generator=g)
    i0, i1 = 8, 12
    Qi = Q[:, :, i0:i1, :]
    K_past = Q[:, :, :i0, :]
    V_past = V[:, :, :i0, :]
    # Expand V like blocked paths
    Vh = V.expand(B, H, T, D)
    shared = _tiled_score_v(Qi, K_past, Vh[:, :, :i0, :], block_size=4)
    # Full cold blocked, slice the query block — past-only contribution equals
    # full output minus diagonal-block contribution.
    full_blocked = blocked_tril_attn(Q, Q, V, block_size=4)
    # Diagonal-only reference for [i0,i1)
    scores = Qi @ Q[:, :, i0:i1, :].transpose(-2, -1)
    scores = scores.tril(diagonal=-1)
    diag = scores @ Vh[:, :, i0:i1, :]
    past_only = full_blocked[:, :, i0:i1, :] - diag
    assert torch.allclose(shared, past_only, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("S,BS", [(17, 5), (64, 256), (100, 16), (3, 64)])
def test_blocked_decode_vs_eager_tril_last_row(S, BS):
    """blocked_decode ≡ last row of eager tril(-1) over concat(past, new)."""
    B, H, N, D = 2, 4, 8, 16
    g = torch.Generator().manual_seed(100 + S + BS)
    past_k = torch.randn(B, H, S, N, generator=g)
    past_v = torch.randn(B, 1, S, D, generator=g)
    q = torch.randn(B, H, 1, N, generator=g)
    v_new = torch.randn(B, 1, 1, D, generator=g)
    K_all = torch.cat([past_k, q], dim=2)
    V_all = torch.cat([past_v, v_new], dim=2)
    last = eager_tril_attn(K_all, K_all, V_all)[:, :, -1:, :]
    got = blocked_decode_attn(q, past_k, past_v, block_size=BS)
    tri = triton_decode_attn(q, past_k, past_v, block_size=BS)
    assert torch.allclose(got, last, rtol=1e-5, atol=1e-5)
    assert torch.allclose(tri, last, rtol=1e-5, atol=1e-5)


def test_default_impl_is_eager(monkeypatch):
    monkeypatch.delenv("BDH_ATTN_IMPL", raising=False)
    assert resolve_attn_impl() == "eager"


def test_blocked_decode_no_expand_broadcast_v():
    """Broadcast V stays (B,1,S,D); result matches eager last-row of tril(-1)."""
    B, H, S, N, D = 2, 4, 33, 8, 16
    g = torch.Generator().manual_seed(99)
    past_k = torch.randn(B, H, S, N, generator=g)
    past_v = torch.randn(B, 1, S, D, generator=g)
    q = torch.randn(B, H, 1, N, generator=g)
    v_new = torch.randn(B, 1, 1, D, generator=g)
    K_all = torch.cat([past_k, q], dim=2)
    V_all = torch.cat([past_v, v_new], dim=2)
    last = eager_tril_attn(K_all, K_all, V_all)[:, :, -1:, :]
    got = blocked_decode_attn(q, past_k, past_v, block_size=9)
    assert past_v.size(1) == 1
    assert torch.allclose(got, last, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("S", [1, 7, 64, 257])
def test_long_past_tiled_decode_vs_eager_last_row(S):
    """Long packed past: blocked/triton decode ≡ eager tril(-1) last row."""
    B, H, N, D = 1, 2, 8, 16
    g = torch.Generator().manual_seed(200 + S)
    past_k = torch.randn(B, H, S, N, generator=g)
    past_v = torch.randn(B, 1, S, D, generator=g)
    q = torch.randn(B, H, 1, N, generator=g)
    v_new = torch.randn(B, 1, 1, D, generator=g)
    last = eager_tril_attn(
        torch.cat([past_k, q], dim=2),
        torch.cat([past_k, q], dim=2),
        torch.cat([past_v, v_new], dim=2),
    )[:, :, -1:, :]
    # Force tiles when S is large (budget / small BS)
    got = blocked_decode_attn(q, past_k, past_v, block_size=16)
    tri = triton_decode_attn(q, past_k, past_v, block_size=16)
    assert torch.allclose(got, last, rtol=1e-5, atol=1e-5)
    assert torch.allclose(tri, last, rtol=1e-5, atol=1e-5)


def test_cuda_dispatch_decode_vs_eager_last_row():
    """BDH_ATTN_IMPL=cuda decode path matches eager tril(-1) last row."""
    B, H, S, N, D = 2, 4, 24, 8, 16
    g = torch.Generator().manual_seed(55)
    past_k = torch.randn(B, H, S, N, generator=g)
    past_v = torch.randn(B, 1, S, D, generator=g)
    q = torch.randn(B, H, 1, N, generator=g)
    v_new = torch.randn(B, 1, 1, D, generator=g)
    last = eager_tril_attn(
        torch.cat([past_k, q], dim=2),
        torch.cat([past_k, q], dim=2),
        torch.cat([past_v, v_new], dim=2),
    )[:, :, -1:, :]
    got = bdh_attn_decode(q, past_k, past_v, impl="cuda")
    assert torch.allclose(got, last, rtol=1e-5, atol=1e-5)
