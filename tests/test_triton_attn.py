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
    _HAS_TRITON,
    _can_use_triton,
    blocked_tril_attn,
    eager_tril_attn,
    triton_tril_attn,
)
from kernels.attention_dispatch import bdh_attn, resolve_attn_impl  # noqa: E402


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
    assert torch.allclose(got, ref, rtol=1e-5, atol=1e-5), (
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
    assert torch.allclose(blocked_tril_attn(Q, K, V), expected, rtol=1e-5, atol=1e-5)
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
    assert torch.allclose(out, ref, rtol=1e-5, atol=1e-5)


def test_triton_api_matches_eager_on_cpu_fallback():
    """triton_tril_attn must CPU-fallback to blocked and match eager."""
    Q, K, V = _make_qkv(T=20, seed=5)
    got = triton_tril_attn(Q, K, V)
    ref = eager_tril_attn(Q, K, V)
    assert torch.allclose(got, ref, rtol=1e-5, atol=1e-5)


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

    assert torch.allclose(out_e, out_b, rtol=1e-5, atol=1e-5)
    assert torch.allclose(out_e, out_t, rtol=1e-5, atol=1e-5)
    assert torch.equal(out_e[:, :, 0, :], torch.zeros_like(out_e[:, :, 0, :]))


def test_v_head_broadcast():
    """V is (B,1,T,D) and must broadcast across heads like eager matmul."""
    Q, K, V = _make_qkv(B=2, H=4, T=9, N=16, D=32, seed=8)
    ref = eager_tril_attn(Q, K, V)
    got = blocked_tril_attn(Q, K, V, block_size=5)
    assert torch.allclose(got, ref, rtol=1e-5, atol=1e-5)
