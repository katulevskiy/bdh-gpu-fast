"""Correctness of Triton / blocked BDH attention vs eager tril(diagonal=-1)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kernels.attention import (  # noqa: E402
    DEFAULT_BLOCK_COLD,
    _HAS_TRITON,
    _as_contiguous,
    _can_use_triton,
    _expand_v_heads,
    _pick_triton_cold_tiles,
    blocked_tril_attn,
    eager_tril_attn,
    online_tril_attn,
    pick_cold_block_size,
    pick_triton_cold_tiles,
    triton_decode_available,
    triton_tril_attn,
)
from kernels.attention_dispatch import (  # noqa: E402
    bdh_attn,
    resolve_attn_impl,
    resolve_cold_impl,
)


def _make_qkv(B=2, H=4, T=16, N=32, D=64, seed=0, device="cpu", dtype=torch.float32):
    g = torch.Generator(device="cpu").manual_seed(seed)
    Q = torch.randn(B, H, T, N, generator=g, dtype=dtype)
    K = Q.clone()  # BDH shares Q/K latent; still test general K
    V = torch.randn(B, 1, T, D, generator=g, dtype=dtype)
    return Q.to(device), K.to(device), V.to(device)


@pytest.mark.parametrize("T", [1, 2, 7, 16, 33])
@pytest.mark.parametrize("block_size", [4, 8, 64])
def test_blocked_matches_eager(T, block_size):
    Q, K, V = _make_qkv(T=T, seed=T * 10 + block_size)
    ref = eager_tril_attn(Q, K, V)
    got = blocked_tril_attn(Q, K, V, block_size=block_size)
    assert got.shape == ref.shape
    assert torch.allclose(got, ref, rtol=1e-4, atol=1e-4), (
        f"T={T} BS={block_size} max diff={(got - ref).abs().max().item()}"
    )


def test_position_zero_exactly_zero():
    Q, K, V = _make_qkv(T=8, seed=1)
    for fn in (eager_tril_attn, blocked_tril_attn):
        out = fn(Q, K, V)
        assert torch.equal(out[:, :, 0, :], torch.zeros_like(out[:, :, 0, :]))
        assert torch.count_nonzero(out[:, :, 0, :]) == 0


def test_no_softmax_no_scale_semantics():
    """Sanity: output equals raw score@V, not softmax-normalized attention."""
    Q, K, V = _make_qkv(B=1, H=1, T=4, N=8, D=8, seed=2)
    scores = (Q @ K.transpose(-2, -1)).tril(diagonal=-1)
    expected = scores @ V
    assert torch.allclose(eager_tril_attn(Q, K, V), expected, rtol=0, atol=0)
    assert torch.allclose(blocked_tril_attn(Q, K, V), expected, rtol=1e-4, atol=1e-4)
    # Softmax path would differ
    sm = torch.softmax(scores, dim=-1)  # includes -inf? no — zeros on upper become e^0
    # Upper is 0 not -inf so softmax ≠ causal softmax; either way ≠ raw scores@V
    sm_out = sm @ V
    assert not torch.allclose(expected, sm_out, rtol=1e-3, atol=1e-3)


def test_dispatch_eager_default(monkeypatch):
    monkeypatch.delenv("BDH_ATTN_IMPL", raising=False)
    assert resolve_attn_impl() == "eager"
    Q, K, V = _make_qkv(T=8, seed=3)
    out = bdh_attn(Q, K, V)
    assert torch.allclose(out, eager_tril_attn(Q, K, V), rtol=0, atol=0)


def test_dispatch_triton_falls_back_on_cpu(monkeypatch):
    monkeypatch.setenv("BDH_ATTN_IMPL", "triton")
    Q, K, V = _make_qkv(T=12, seed=4)
    assert not _can_use_triton(Q) or Q.device.type == "cpu"
    out = bdh_attn(Q, K, V, impl="triton")
    ref = eager_tril_attn(Q, K, V)
    assert torch.allclose(out, ref, rtol=1e-4, atol=1e-4)


def test_triton_api_matches_eager_on_cpu_fallback():
    """triton_tril_attn must CPU-fallback to blocked and match eager."""
    Q, K, V = _make_qkv(T=20, seed=5)
    got = triton_tril_attn(Q, K, V)
    ref = eager_tril_attn(Q, K, V)
    assert torch.allclose(got, ref, rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available() or not _HAS_TRITON, reason="CUDA+Triton required")
def test_triton_kernel_matches_eager_cuda():
    Q, K, V = _make_qkv(T=64, N=64, D=64, seed=6, device="cuda")
    ref = eager_tril_attn(Q, K, V)
    got = triton_tril_attn(Q, K, V)
    assert torch.allclose(got, ref, rtol=1e-4, atol=1e-4), (
        f"cuda max diff={(got - ref).abs().max().item()}"
    )


def test_bdh_attention_hook_respects_env(monkeypatch):
    """bdh.Attention cold path uses kernels when BDH_ATTN_IMPL != eager."""
    import bdh

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
    B, T = 2, 10
    torch.manual_seed(7)
    Q = torch.randn(B, cfg.n_head, T, N)
    V = torch.randn(B, 1, T, cfg.n_embd)

    monkeypatch.setenv("BDH_ATTN_IMPL", "eager")
    out_e, _, _ = attn(Q, Q, V)

    monkeypatch.setenv("BDH_ATTN_IMPL", "blocked")
    out_b, _, _ = attn(Q, Q, V)

    monkeypatch.setenv("BDH_ATTN_IMPL", "triton")
    out_t, _, _ = attn(Q, Q, V)

    assert torch.allclose(out_e, out_b, rtol=1e-4, atol=1e-4)
    assert torch.allclose(out_e, out_t, rtol=1e-4, atol=1e-4)
    assert torch.equal(out_e[:, :, 0, :], torch.zeros_like(out_e[:, :, 0, :]))


def test_v_head_broadcast():
    """V is (B,1,T,D) and must broadcast across heads like eager matmul."""
    Q, K, V = _make_qkv(B=2, H=4, T=9, N=16, D=32, seed=8)
    ref = eager_tril_attn(Q, K, V)
    got = blocked_tril_attn(Q, K, V, block_size=5)
    assert torch.allclose(got, ref, rtol=1e-4, atol=1e-4)


# --- opt/triton-cold: better tiles, less staging, share online blocked ---

def test_pick_triton_cold_tiles_aligned_with_online_block():
    """Cold Triton defaults track DEFAULT_BLOCK_COLD (online blocked BS)."""
    bm, bn, bd, bk = _pick_triton_cold_tiles(T=128, N=64, D=128)
    assert bm >= 16 and bn >= 16
    assert bm <= 128 and bn <= 128
    # Prefer ~64 like online blocked, not the old fixed 32.
    assert bm >= min(DEFAULT_BLOCK_COLD, 64)
    assert bn >= min(DEFAULT_BLOCK_COLD, 64)
    assert bd >= 16 and bk >= 16
    # Explicit overrides still honored (power-of-2 capped).
    bm2, bn2, _, _ = _pick_triton_cold_tiles(64, 32, 32, block_m=40, block_n=40)
    assert bm2 == 64 and bn2 == 64  # next pow2 of 40 capped


def test_as_contiguous_noop_when_already_contiguous():
    t = torch.randn(2, 3, 4)
    assert _as_contiguous(t) is t
    v = t.transpose(0, 1)
    assert not v.is_contiguous()
    c = _as_contiguous(v)
    assert c.is_contiguous()
    assert torch.equal(c, v.contiguous())


def test_triton_cold_fallback_matches_online_blocked():
    """CPU triton_tril_attn must use adaptive #75 online blocked."""
    Q, K, V = _make_qkv(T=37, N=32, D=64, seed=42)
    got = triton_tril_attn(Q, K, V)
    ref_eager = eager_tril_attn(Q, K, V)
    ref_online = online_tril_attn(Q, K, V)  # adaptive pick_cold_block_size
    assert torch.allclose(got, ref_eager, rtol=1e-4, atol=1e-4)
    assert torch.equal(got, ref_online)


def test_expand_v_heads_shared_no_copy_on_broadcast():
    """_expand_v_heads is a view (no B*H*T*D staging) — used by blocked + cold."""
    B, H, T, D = 2, 4, 16, 32
    V = torch.randn(B, 1, T, D)
    Vh = _expand_v_heads(V, B, H, T, D)
    assert Vh.shape == (B, H, T, D)
    assert Vh.data_ptr() == V.data_ptr()
    # Mutating base reflects in expand (true broadcast view)
    V[0, 0, 0, 0] = 123.0
    assert Vh[0, 0, 0, 0] == 123.0 and Vh[0, 3, 0, 0] == 123.0


@pytest.mark.parametrize("T", [1, 5, 17, 64])
def test_triton_path_vs_eager_tril_minus_one(T):
    """triton cold path (CPU→online blocked) ≡ eager (Q@K.T).tril(-1) @ V."""
    Q, K, V = _make_qkv(T=T, seed=100 + T)
    scores = (Q @ K.transpose(-2, -1)).tril(diagonal=-1)
    expected = scores @ V
    got = triton_tril_attn(Q, K, V)
    assert torch.allclose(got, expected, rtol=1e-4, atol=1e-4)
    assert torch.count_nonzero(got[:, :, 0, :]) == 0


def test_default_remains_eager_after_triton_cold(monkeypatch):
    monkeypatch.delenv("BDH_ATTN_IMPL", raising=False)
    assert resolve_attn_impl() == "eager"


# --- opt/triton-cold-v2: deepen long-T tiles (pair #75/#79) ---

def test_pick_triton_cold_tiles_long_t_pairs_with_blocked():
    """T≥256 grows to 128 like pick_cold_block_size (#75); short stays 64."""
    assert pick_cold_block_size(128) == DEFAULT_BLOCK_COLD
    assert pick_cold_block_size(256) == min(128, DEFAULT_BLOCK_COLD * 2)

    bm, bn, bd, bk = pick_triton_cold_tiles(T=128, N=64, D=128)
    assert bm == DEFAULT_BLOCK_COLD and bn == DEFAULT_BLOCK_COLD
    assert (bm, bn, bd, bk) == _pick_triton_cold_tiles(128, 64, 128)

    bm256, bn256, _, _ = pick_triton_cold_tiles(T=256, N=64, D=128)
    assert bm256 == 128 and bn256 == 128

    bm512, bn512, _, _ = pick_triton_cold_tiles(T=512, N=64, D=128)
    assert bm512 == 128 and bn512 == 128

    # Wide heads keep the 64×64 staging footprint; explicit overrides win.
    bm_wide, bn_wide, _, _ = pick_triton_cold_tiles(T=512, N=128, D=256)
    assert bm_wide == DEFAULT_BLOCK_COLD and bn_wide == DEFAULT_BLOCK_COLD
    bm_explicit, bn_explicit, _, _ = pick_triton_cold_tiles(
        512, 128, 256, block_m=128, block_n=128
    )
    assert bm_explicit == 128 and bn_explicit == 128

    # Explicit overrides still power-of-2 capped
    bm2, bn2, _, _ = pick_triton_cold_tiles(64, 32, 32, block_m=40, block_n=40)
    assert bm2 == 64 and bn2 == 64


@pytest.mark.parametrize("T", [256, 512])
def test_triton_cold_long_t_fallback_equiv_blocked_eager(T):
    """Long-T CPU triton→blocked adaptive ≡ eager tril(-1); no full T×T claim."""
    Q, K, V = _make_qkv(B=1, H=2, T=T, N=16, D=32, seed=200 + T)
    scores = (Q @ K.transpose(-2, -1)).tril(diagonal=-1)
    expected = scores @ V
    got = triton_tril_attn(Q, K, V)
    blocked = blocked_tril_attn(Q, K, V)
    assert torch.allclose(got, expected, rtol=1e-4, atol=1e-4)
    assert torch.equal(got, blocked)
    assert torch.count_nonzero(got[:, :, 0, :]) == 0


def test_triton_cold_auto_interaction_documented(monkeypatch):
    """AUTO cold: long-T → triton if CUDA+Triton else blocked; short stays eager.

    Documented in triton_tril_attn / OPT_NOTES; defaults unchanged (AUTO off).
    """
    monkeypatch.delenv("BDH_ATTN_IMPL", raising=False)
    monkeypatch.delenv("BDH_ATTN_AUTO", raising=False)
    monkeypatch.delenv("BDH_ATTN_AUTO_THRESHOLD", raising=False)
    monkeypatch.delenv("BDH_ATTN_AUTO_COLD_THRESHOLD", raising=False)
    assert resolve_cold_impl(1024) == "eager"  # AUTO off

    monkeypatch.setenv("BDH_ATTN_AUTO", "1")
    # On this CPU box triton_decode_available() is False → blocked
    assert not triton_decode_available() or torch.cuda.is_available()
    want = "triton" if triton_decode_available() else "blocked"
    assert resolve_cold_impl(1024) == want
    assert resolve_cold_impl(128) == "eager"  # short stays eager

    monkeypatch.setenv("BDH_ATTN_IMPL", "blocked")
    assert resolve_cold_impl(1024) == "blocked"  # explicit never overridden


@pytest.mark.skipif(
    not torch.cuda.is_available() or not _HAS_TRITON,
    reason="CUDA+Triton required (soft-skip; no GPU claims on CPU box)",
)
@pytest.mark.parametrize("T", [256, 512])
def test_triton_cold_kernel_long_t_vs_eager_soft_skip(T):
    """When CUDA+Triton available: fused cold long-T ≡ eager tril(-1)."""
    Q, K, V = _make_qkv(B=1, H=2, T=T, N=32, D=64, seed=300 + T, device="cuda")
    ref = eager_tril_attn(Q, K, V)
    got = triton_tril_attn(Q, K, V)
    assert torch.allclose(got, ref, rtol=1e-3, atol=1e-3)
    assert torch.count_nonzero(got[:, :, 0, :]) == 0
