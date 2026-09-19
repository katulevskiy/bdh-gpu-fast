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
import kernels.attention as attention_impl
from bdh_cache import CacheManager
from kernels.attention import (
    DEFAULT_BLOCK_DECODE,
    _DECODE_ONESHOT_ELEMS,
    _HAS_TRITON,
    _can_use_triton,
    _pick_tile_size,
    _pick_triton_decode_tiles,
    _tiled_score_v,
    blocked_decode_attn,
    blocked_tril_attn,
    eager_decode_attn,
    eager_tril_attn,
    max_decode_score_elems,
    online_decode_attn,
    triton_decode_attn,
    triton_decode_available,
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


def test_t1_shared_v_b1_direct_epilogue_matches_eager(monkeypatch):
    """B=1 shared-V decode writes score×V directly into the output tile."""
    B, H, S, N, D = 1, 4, _DECODE_ONESHOT_ELEMS + 513, 8, 16
    Q, K, V = _make_decode_qkv(B=B, H=H, S=S, N=N, D=D, seed=101)
    ref = eager_decode_attn(Q, K, V)
    seen = []
    orig = torch.baddbmm

    def spy(input, batch1, batch2, *, beta=1, alpha=1, out=None):
        if out is not None:
            seen.append((input.data_ptr(), out.data_ptr(), batch2.stride(0)))
        return orig(input, batch1, batch2, beta=beta, alpha=alpha, out=out)

    monkeypatch.setattr(torch, "baddbmm", spy)
    got = blocked_decode_attn(Q, K, V, block_size=64)

    assert seen
    assert all(input_ptr == out_ptr for input_ptr, out_ptr, _ in seen)
    assert all(head_stride == 0 for _, _, head_stride in seen)
    assert torch.allclose(got, ref, rtol=1e-5, atol=1e-5)


def test_t1_shared_v_batched_epilogue_matches_eager(monkeypatch):
    """B>1 shared-V tiles use one broadcast matmul without staging heads."""
    B, H, S, N, D = 2, 4, _DECODE_ONESHOT_ELEMS * 2 + 1, 8, 16
    Q, K, V = _make_decode_qkv(B=B, H=H, S=S, N=N, D=D, seed=102)
    ref = eager_decode_attn(Q, K, V)
    seen = []
    orig = torch.matmul

    def spy(input, other, *, out=None):
        if input.dim() == 4 and other.dim() == 4 and input.size(0) == B:
            seen.append((input.shape, other.shape, out is not None))
        if out is None:
            return orig(input, other)
        return orig(input, other, out=out)

    monkeypatch.setattr(torch, "matmul", spy)
    got = blocked_decode_attn(Q, K, V, block_size=64)

    assert seen
    assert all(input_shape[:2] == (B, H) for input_shape, _, _ in seen)
    assert all(other_shape[:2] == (B, 1) for _, other_shape, _ in seen)
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


@pytest.mark.parametrize("impl", ["eager", "blocked", "online", "triton", "cuda"])
def test_decode_dispatch_pos0_is_zero(impl):
    """Every decode backend keeps position zero at exact zero."""
    Q = torch.randn(2, 4, 1, 8)
    K = torch.empty(2, 4, 0, 8)
    V = torch.empty(2, 1, 0, 16)
    out = bdh_attn_decode(Q, K, V, impl=impl)
    assert out.shape == (2, 4, 1, 16)
    assert torch.count_nonzero(out) == 0


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


@pytest.mark.parametrize("impl", ["eager", "blocked", "triton", "cuda"])
def test_padded_cache_t1_decode_views_match_eager(impl, monkeypatch):
    """T=1 decode stays parity-safe with capacity-padded packed KR/V views."""
    cfg = _small_cfg()
    model = _model(cfg, seed=420)
    x = torch.randint(0, cfg.vocab_size, (2, 12))
    split = 7

    # Establish the reference and cache prefix with default eager semantics.
    monkeypatch.setenv("BDH_ATTN_IMPL", "eager")
    with torch.inference_mode():
        full, _ = model(x)
        cm = CacheManager.from_config(cfg, x.size(0), max_seq=32, device=x.device)
        model(x[:, :split], cache=cm)

        for level in range(cfg.n_layer):
            past_kr, past_v = cm.get_past(level)
            assert past_kr is not None and past_v is not None
            assert not past_kr.is_contiguous()
            assert not past_v.is_contiguous()

        monkeypatch.setenv("BDH_ATTN_IMPL", impl)
        got, _ = model(x[:, split : split + 1], cache=cm)

    assert torch.allclose(
        full[:, split : split + 1], got, rtol=1e-5, atol=1e-5
    ), f"impl={impl} max abs={(full[:, split : split + 1] - got).abs().max().item()}"


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


def test_t1_decode_tight_oneshot_tiles_mid_past():
    """The tighter T=1 budget tiles a mid-size past scan without drift."""
    assert _DECODE_ONESHOT_ELEMS <= 1024
    S = _DECODE_ONESHOT_ELEMS + 512
    Q, K, V = _make_decode_qkv(B=1, H=2, S=S, N=8, D=16, seed=73)
    ref = eager_decode_attn(Q, K, V)
    got = blocked_decode_attn(Q, K, V, block_size=64)
    assert max_decode_score_elems(S, block_size=64) < S
    assert torch.allclose(got, ref, rtol=1e-5, atol=1e-5)


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


def test_two_gemm_decode_tq1_bh_bmm_matches_4d():
    """Tq=1 + per-head V uses BH-bmm path; must match broadcast 4D @."""
    from kernels.attention import _two_gemm_decode

    B, H, S, N, D = 2, 4, 48, 8, 16
    g = torch.Generator().manual_seed(77)
    Q = torch.randn(B, H, 1, N, generator=g)
    K = torch.randn(B, H, S, N, generator=g)
    Vh = torch.randn(B, H, S, D, generator=g)
    got = _two_gemm_decode(Q, K, Vh)
    ref = (Q @ K.transpose(-2, -1)) @ Vh
    assert torch.allclose(got, ref, rtol=1e-5, atol=1e-5)


def test_two_gemm_decode_broadcast_v_no_expand():
    """Broadcast V=(B,1,S,D) stays unexpanded; ≡ eager last-row tril(-1)."""
    from kernels.attention import _two_gemm_decode

    B, H, S, N, D = 2, 4, 17, 8, 16
    g = torch.Generator().manual_seed(78)
    Q = torch.randn(B, H, 1, N, generator=g)
    K = torch.randn(B, H, S, N, generator=g)
    V = torch.randn(B, 1, S, D, generator=g)
    v_new = torch.randn(B, 1, 1, D, generator=g)
    got = _two_gemm_decode(Q, K, V)
    last = eager_tril_attn(
        torch.cat([K, Q], dim=2),
        torch.cat([K, Q], dim=2),
        torch.cat([V, v_new], dim=2),
    )[:, :, -1:, :]
    assert torch.allclose(got, last, rtol=1e-5, atol=1e-5)


def test_tiled_score_v_add_inplace_matches_eager():
    """Forced tile loop (out.add_) ≡ eager two-GEMM."""
    Q, K, V = _make_decode_qkv(B=1, H=2, S=80, N=8, D=16, Tq=1, seed=12)
    ref = eager_decode_attn(Q, K, V)
    # Small BS forces tiles under budget path
    got = _tiled_score_v(Q, K, V, block_size=4)
    assert torch.allclose(got, ref, rtol=1e-5, atol=1e-5)


def test_cuda_decode_tiled_ref_uses_decode_tile_n():
    """CUDA decode tiled ref (DECODE_TILE_N / adaptive) ≡ eager last row."""
    from kernels.cuda_attn import (
        CUDA_DECODE_TILE_N,
        pick_cuda_decode_tile_n,
        tril_decode_tiled_ref,
    )

    assert CUDA_DECODE_TILE_N >= 32
    assert pick_cuda_decode_tile_n(100) >= 32
    B, H, S, N, D = 1, 2, 100, 8, 16
    g = torch.Generator().manual_seed(88)
    q = torch.randn(B, H, 1, N, generator=g)
    k = torch.randn(B, H, S, N, generator=g)
    v = torch.randn(B, 1, S, D, generator=g)
    v_new = torch.randn(B, 1, 1, D, generator=g)
    last = eager_tril_attn(
        torch.cat([k, q], dim=2),
        torch.cat([k, q], dim=2),
        torch.cat([v, v_new], dim=2),
    )[:, :, -1:, :]
    got = tril_decode_tiled_ref(q, k, v, tile_n=CUDA_DECODE_TILE_N)
    assert torch.allclose(got, last, rtol=1e-5, atol=1e-5)


def test_attention_t1_unified_decode_dispatch(monkeypatch):
    """Attention T=1 routes all impls through bdh_attn_decode (eager included)."""
    monkeypatch.delenv("BDH_ATTN_IMPL", raising=False)
    cfg = _small_cfg()
    attn = bdh.Attention(cfg)
    B, nh, S = 1, cfg.n_head, 9
    N = cfg.mlp_internal_dim_multiplier * cfg.n_embd // nh
    D = cfg.n_embd
    torch.manual_seed(3)
    Q = torch.randn(B, nh, 1, N)
    past_kr = torch.randn(B, nh, S, N)
    past_v = torch.randn(B, 1, S, D)
    V = torch.randn(B, 1, 1, D)
    out, _, _ = attn(Q, Q, V, rope_start=S, past_kr=past_kr, past_v=past_v)
    # Poison V_new must not affect (tril -1 / past-only)
    out2, _, _ = attn(
        Q, Q, torch.randn_like(V) * 99, rope_start=S, past_kr=past_kr, past_v=past_v
    )
    assert torch.allclose(out, out2, atol=0)
    # pos0-style: S=0 → zeros
    out0, _, _ = attn(
        Q, Q, V, rope_start=0, past_kr=past_kr[:, :, :0], past_v=past_v[:, :, :0]
    )
    assert torch.all(out0 == 0)



def test_online_decode_alias_matches_blocked():
    Q, K, V = _make_decode_qkv(S=96, Tq=1, seed=5)
    assert torch.allclose(
        online_decode_attn(Q, K, V, block_size=32),
        blocked_decode_attn(Q, K, V, block_size=32),
        atol=0,
    )


@pytest.mark.parametrize("S", [64, 256, 1024, 4096])
def test_blocked_decode_long_s_broadcast_v_parity(S):
    """Decode-online-v2: long packed past, broadcast V ≡ eager (tril -1 last row)."""
    B, H, N, D = 2, 4, 16, 32
    g = torch.Generator().manual_seed(300 + S)
    past_k = torch.randn(B, H, S, N, generator=g)
    past_v = torch.randn(B, 1, S, D, generator=g)  # CacheManager layout
    q = torch.randn(B, H, 1, N, generator=g)
    v_new = torch.randn(B, 1, 1, D, generator=g)
    last = eager_tril_attn(
        torch.cat([past_k, q], dim=2),
        torch.cat([past_k, q], dim=2),
        torch.cat([past_v, v_new], dim=2),
    )[:, :, -1:, :]
    got = blocked_decode_attn(q, past_k, past_v)
    online = online_decode_attn(q, past_k, past_v)
    assert torch.allclose(got, last, rtol=1e-4, atol=1e-5)
    assert torch.allclose(online, last, rtol=1e-4, atol=1e-5)


def test_max_decode_score_elems_bound_vs_eager():
    """Blocked/online decode peak score elems ≤ eager Tq×S; tiles when long S."""
    assert _DECODE_ONESHOT_ELEMS >= DEFAULT_BLOCK_DECODE
    for S in (64, 256, 1024, 2048, 4096):
        peak = max_decode_score_elems(S, Tq=1)
        eager_peak = 1 * S
        assert peak <= eager_peak
        if S > _DECODE_ONESHOT_ELEMS:
            assert peak < eager_peak
            assert peak <= _DECODE_ONESHOT_ELEMS


def test_decode_tiles_when_past_exceeds_oneshot_budget():
    """S > _DECODE_ONESHOT_ELEMS with broadcast V must take the tile path (parity)."""
    S = _DECODE_ONESHOT_ELEMS + 512
    B, H, N, D = 1, 2, 8, 16
    g = torch.Generator().manual_seed(401)
    Q = torch.randn(B, H, 1, N, generator=g)
    K = torch.randn(B, H, S, N, generator=g)
    V = torch.randn(B, 1, S, D, generator=g)
    ref = eager_decode_attn(Q, K, V)
    # Default block_size — policy must tile (peak bound) and match eager
    got = blocked_decode_attn(Q, K, V)
    assert torch.allclose(got, ref, rtol=1e-4, atol=1e-5)
    assert max_decode_score_elems(S) < S


def test_pick_tile_size_honors_decode_score_budget():
    """Decode budget caps tile width so Tq*tile stays within oneshot elems."""
    tile = _pick_tile_size(8192, 1, 256, score_budget=_DECODE_ONESHOT_ELEMS)
    assert tile <= _DECODE_ONESHOT_ELEMS
    assert tile >= 256


# --- opt/triton-decode-v3: deepen Triton T=1 scaffold + CPU honesty ---

def test_pick_triton_decode_tiles_long_s():
    """Long S prefers larger past tiles (up to 512); mid-S stays moderate."""
    bn64, bd, bk = _pick_triton_decode_tiles(64, 64, 128)
    assert bn64 == 64 and bd >= 16 and bk >= 16
    bn512, _, _ = _pick_triton_decode_tiles(512, 64, 128)
    assert bn512 == 128
    bn2k, _, _ = _pick_triton_decode_tiles(2048, 64, 128)
    assert bn2k == 256
    bn4k, _, bk4 = _pick_triton_decode_tiles(4096, 64, 128)
    assert bn4k == 512
    # N<=64 → BLOCK_K covers N → host will set HOIST_Q=1
    assert bk4 >= 64
    # explicit override still power-of-2 capped
    bn_o, _, _ = _pick_triton_decode_tiles(4096, 32, 32, block_n=300)
    assert bn_o == 512


def test_triton_decode_available_cpu_honest():
    """No GPU on this box → triton_decode_available is False (AUTO → blocked)."""
    avail = triton_decode_available()
    if not torch.cuda.is_available() or not _HAS_TRITON:
        assert avail is False
    else:
        assert avail is True


@pytest.mark.parametrize("S", [64, 256, 1024, 4096])
def test_triton_decode_long_s_cpu_fallback_matches_blocked(S):
    """CPU triton_decode_attn → blocked_decode (#55 oneshot); ≡ eager last-row math."""
    Q, K, V = _make_decode_qkv(B=2, H=4, S=S, N=16, D=32, Tq=1, seed=50 + S)
    tri = triton_decode_attn(Q, K, V)
    blk = blocked_decode_attn(Q, K, V)
    ref = eager_decode_attn(Q, K, V)
    assert torch.allclose(tri, blk, atol=0), "CPU fallback must be blocked_decode"
    assert torch.allclose(tri, ref, rtol=1e-4, atol=1e-5)
    assert not _can_use_triton(Q)  # CPU tensor → no kernel launch


@pytest.mark.skipif(
    not torch.cuda.is_available() or not _HAS_TRITON,
    reason="CUDA+Triton required for fused decode kernel",
)
@pytest.mark.parametrize("S", [32, 128, 1024])
def test_triton_decode_kernel_matches_eager_cuda(S):
    """Fused Triton decode ≡ eager on CUDA (skip cleanly on CPU boxes)."""
    device = "cuda"
    g = torch.Generator(device="cpu").manual_seed(77 + S)
    B, H, N, D = 2, 4, 64, 64
    Q = torch.randn(B, H, 1, N, generator=g).to(device)
    K = torch.randn(B, H, S, N, generator=g).to(device)
    V = torch.randn(B, 1, S, D, generator=g).to(device)
    assert _can_use_triton(Q)
    got = triton_decode_attn(Q, K, V)
    ref = eager_decode_attn(Q, K, V)
    assert torch.allclose(got, ref, rtol=1e-3, atol=1e-3), (
        f"S={S} maxdiff={(got - ref).abs().max().item()}"
    )


def test_triton_decode_vs_full_tril_minus_one():
    """triton decode (CPU→blocked) ≡ last row of eager tril(diagonal=-1)."""
    B, H, S, N, D = 2, 4, 48, 16, 32
    g = torch.Generator().manual_seed(91)
    past_k = torch.randn(B, H, S, N, generator=g)
    past_v = torch.randn(B, 1, S, D, generator=g)
    q = torch.randn(B, H, 1, N, generator=g)
    K_all = torch.cat([past_k, q], dim=2)
    V_new = torch.randn(B, 1, 1, D, generator=g)
    V_all = torch.cat([past_v, V_new], dim=2)
    last = eager_tril_attn(K_all, K_all, V_all)[:, :, -1:, :]
    dec = triton_decode_attn(q, past_k, past_v)
    assert torch.allclose(dec, last, rtol=1e-5, atol=1e-5)
    # poison new V — must not affect (tril -1)
    V_all2 = torch.cat([past_v, torch.randn_like(V_new) * 9], dim=2)
    last2 = eager_tril_attn(K_all, K_all, V_all2)[:, :, -1:, :]
    assert torch.allclose(last, last2, atol=0)


def test_pick_cuda_decode_tile_n_pairs_long_s():
    """cuda-decode-v3 tile picker mirrors #62 long-S preference (CPU path)."""
    from kernels.cuda_attn import (
        CUDA_DECODE_TILE_N_MAX,
        pick_cuda_decode_tile_n,
    )

    assert pick_cuda_decode_tile_n(64) == 32
    assert pick_cuda_decode_tile_n(256) >= 64
    assert pick_cuda_decode_tile_n(2048) >= 128
    assert pick_cuda_decode_tile_n(4096) == 512
    assert pick_cuda_decode_tile_n(4096, for_smem=True) <= CUDA_DECODE_TILE_N_MAX


def test_cuda_tiled_ref_blocked_parity_inc():
    """Inc-decode: CUDA tiled ref ≡ blocked on broadcast-V long S."""
    from kernels.cuda_attn import tril_decode_tiled_ref

    Q, K, V = _make_decode_qkv(B=2, H=4, S=1024, N=16, D=32, Tq=1, seed=91)
    assert torch.allclose(
        tril_decode_tiled_ref(Q, K, V),
        blocked_decode_attn(Q, K, V),
        rtol=1e-4,
        atol=1e-4,
    )

def test_triton_decode_launcher_preserves_packed_strides(monkeypatch):
    """Host launcher passes packed views without staging copies.

    The fake launcher makes this a CPU-safe contract test: real Triton remains
    covered by the CUDA-only tests below, while the no-copy stride contract is
    exercised even when Triton is not installed on the test host.
    """
    B, H, S, N, D = 2, 4, 7, 8, 16
    cm = CacheManager(
        n_layer=1,
        max_seq=32,
        batch_size=B,
        n_head=H,
        n_latent=N,
        n_embd=D,
        device="cpu",
    )
    cm._kr_buf[0].normal_()
    cm._v_buf[0].normal_()
    cm.seq_len = S
    K, V = cm.get_past(0)
    assert K is not None and V is not None
    Q = cm._kr_buf[0].narrow(2, S, 1)
    captured = {}

    class _FakeKernel:
        def __getitem__(self, grid):
            def launch(qf, kf, vptr, out, *args, **kwargs):
                captured.update(
                    qf=qf, kf=kf, vptr=vptr, out=out, args=args, kwargs=kwargs
                )
                out.copy_(eager_decode_attn(Q, K, V).reshape_as(out))

            return launch

    monkeypatch.setattr(attention_impl, "_can_use_triton", lambda _: True)
    monkeypatch.setattr(
        attention_impl, "_bdh_decode_fwd_kernel", _FakeKernel(), raising=False
    )
    got = attention_impl.triton_decode_attn(Q, K, V)
    ref = eager_decode_attn(Q, K, V)

    assert torch.allclose(got, ref, rtol=1e-5, atol=1e-5)
    assert captured["kf"].data_ptr() == K.data_ptr()
    assert captured["vptr"].data_ptr() == V.squeeze(1).data_ptr()
    assert captured["kf"].stride() == K.reshape(B * H, S, N).stride()
    assert captured["vptr"].stride() == V.squeeze(1).stride()


@pytest.mark.parametrize("S", [_DECODE_ONESHOT_ELEMS, _DECODE_ONESHOT_ELEMS + 1])
@pytest.mark.parametrize("B", [1, 2])
def test_packed_cache_t1_decode_oneshot_boundary_cpu_parity(S, B):
    """Packed padded KR/V views stay exact across the oneshot budget boundary."""
    H, N, D = 4, 8, 16
    cm = CacheManager(
        n_layer=1,
        max_seq=S + 17,
        batch_size=B,
        n_head=H,
        n_latent=N,
        n_embd=D,
        device="cpu",
    )
    torch.manual_seed(500 + B + S)
    cm._kr_buf[0].normal_()
    cm._v_buf[0].normal_()
    cm.seq_len = S
    K, V = cm.get_past(0)
    assert K is not None and V is not None
    # A singleton B=1 V dimension can make ``is_contiguous()`` true despite
    # the live prefix retaining the larger capacity stride. Check the actual
    # packed strides instead of relying on that ambiguous predicate.
    assert K.stride() == (H * cm.capacity * N, cm.capacity * N, N, 1)
    assert V.stride() == (cm.capacity * D, cm.capacity * D, D, 1)

    Q = torch.randn(B, H, 1, N)
    ref = eager_decode_attn(Q, K, V)
    for impl in ("blocked", "online", "triton"):
        got = bdh_attn_decode(Q, K, V, impl=impl)
        assert torch.allclose(got, ref, rtol=1e-4, atol=1e-5), (
            f"impl={impl} B={B} S={S} "
            f"maxdiff={(got - ref).abs().max().item()}"
        )
    assert max_decode_score_elems(S) <= S
    if S > _DECODE_ONESHOT_ELEMS:
        assert max_decode_score_elems(S) < S
