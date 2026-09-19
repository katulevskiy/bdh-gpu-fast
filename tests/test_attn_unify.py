"""Unified BDH_ATTN_IMPL dispatch: eager | blocked | triton | cuda.

Cold Attention.forward respects the env; KV-cache / generate decode stays eager.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kernels.attention import blocked_tril_attn, eager_tril_attn  # noqa: E402
from kernels.attention_dispatch import (  # noqa: E402
    backend_info,
    bdh_attn,
    resolve_attn_impl,
)
from kernels.cuda_attn import tril_score_v_ref  # noqa: E402
import bdh  # noqa: E402


def _make_qkv(B=2, H=4, T=12, N=32, D=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    Q = torch.randn(B, H, T, N, generator=g)
    K = Q.clone()
    V = torch.randn(B, 1, T, D, generator=g)
    return Q, K, V


def _small_attn():
    cfg = bdh.BDHConfig(
        n_layer=1,
        n_embd=64,
        n_head=4,
        mlp_internal_dim_multiplier=8,
        dropout=0.0,
        vocab_size=256,
    )
    return bdh.Attention(cfg), cfg


@pytest.mark.parametrize("impl", ["eager", "blocked", "triton", "cuda"])
def test_resolve_accepts_all_impls(monkeypatch, impl):
    monkeypatch.setenv("BDH_ATTN_IMPL", impl)
    assert resolve_attn_impl() == impl


def test_resolve_default_eager(monkeypatch):
    monkeypatch.delenv("BDH_ATTN_IMPL", raising=False)
    assert resolve_attn_impl() == "eager"


def test_resolve_rejects_unknown(monkeypatch):
    monkeypatch.setenv("BDH_ATTN_IMPL", "flash")
    with pytest.raises(ValueError, match="eager|blocked|triton|cuda"):
        resolve_attn_impl()


@pytest.mark.parametrize("impl", ["eager", "blocked", "triton", "cuda"])
def test_bdh_attn_matches_eager(monkeypatch, impl):
    monkeypatch.setenv("BDH_ATTN_IMPL", impl)
    Q, K, V = _make_qkv(seed=10 + hash(impl) % 100)
    ref = eager_tril_attn(Q, K, V)
    got = bdh_attn(Q, K, V)
    assert got.shape == ref.shape
    assert torch.allclose(got, ref, rtol=1e-5, atol=1e-5), (
        f"impl={impl} max diff={(got - ref).abs().max().item()}"
    )
    assert torch.equal(got[:, :, 0, :], torch.zeros_like(got[:, :, 0, :]))


def test_cuda_impl_matches_ref(monkeypatch):
    monkeypatch.setenv("BDH_ATTN_IMPL", "cuda")
    Q, K, V = _make_qkv(T=9, seed=42)
    got = bdh_attn(Q, K, V, impl="cuda")
    ref = tril_score_v_ref(Q, K, V)
    assert torch.allclose(got, ref, rtol=1e-5, atol=1e-5)


def test_backend_info_keys(monkeypatch):
    monkeypatch.delenv("BDH_ATTN_IMPL", raising=False)
    info = backend_info()
    assert info["BDH_ATTN_IMPL"] == "eager"
    assert info["effective"] == "eager"
    assert "has_triton" in info and "has_cuda_ext" in info and "cuda" in info


@pytest.mark.parametrize("impl", ["eager", "blocked", "triton", "cuda"])
def test_attention_cold_path_env_switch(monkeypatch, impl):
    """bdh.Attention cold path uses unified dispatch for every BDH_ATTN_IMPL."""
    attn, cfg = _small_attn()
    N = cfg.mlp_internal_dim_multiplier * cfg.n_embd // cfg.n_head
    B, T = 2, 10
    torch.manual_seed(7)
    Q = torch.randn(B, cfg.n_head, T, N)
    V = torch.randn(B, 1, T, cfg.n_embd)

    monkeypatch.setenv("BDH_ATTN_IMPL", "eager")
    out_eager, kr, vv = attn(Q, Q, V)

    monkeypatch.setenv("BDH_ATTN_IMPL", impl)
    out, kr2, vv2 = attn(Q, Q, V)

    assert torch.allclose(out, out_eager, rtol=1e-5, atol=1e-5)
    assert torch.equal(out[:, :, 0, :], torch.zeros_like(out[:, :, 0, :]))
    # RoPE'd keys / values store tensors unchanged by backend choice
    assert torch.equal(kr, kr2)
    assert torch.equal(vv, vv2)


def test_cached_path_ignores_attn_impl(monkeypatch):
    """Incremental decode stays eager even when BDH_ATTN_IMPL=blocked/cuda."""
    attn, cfg = _small_attn()
    N = cfg.mlp_internal_dim_multiplier * cfg.n_embd // cfg.n_head
    B, T = 1, 6
    torch.manual_seed(11)
    Q = torch.randn(B, cfg.n_head, T, N)
    V = torch.randn(B, 1, T, cfg.n_embd)

    # Cold prefill under eager → cache seed
    monkeypatch.setenv("BDH_ATTN_IMPL", "eager")
    out_full, past_kr, past_v = attn(Q, Q, V)

    # Single-token step with cache under blocked/cuda must match eager cache step
    Q1 = torch.randn(B, cfg.n_head, 1, N)
    V1 = torch.randn(B, 1, 1, cfg.n_embd)
    # Continue RoPE from T
    monkeypatch.setenv("BDH_ATTN_IMPL", "eager")
    out_e, _, _ = attn(Q1, Q1, V1, rope_start=T, past_kr=past_kr, past_v=past_v)

    monkeypatch.setenv("BDH_ATTN_IMPL", "blocked")
    out_b, _, _ = attn(Q1, Q1, V1, rope_start=T, past_kr=past_kr, past_v=past_v)

    monkeypatch.setenv("BDH_ATTN_IMPL", "cuda")
    out_c, _, _ = attn(Q1, Q1, V1, rope_start=T, past_kr=past_kr, past_v=past_v)

    assert torch.equal(out_e, out_b)
    assert torch.equal(out_e, out_c)
    # Sanity: full cold still sensible
    assert out_full.shape[2] == T


def test_no_softmax_no_scale_across_impls(monkeypatch):
    Q, K, V = _make_qkv(B=1, H=1, T=5, N=8, D=8, seed=99)
    scores = (Q @ K.transpose(-2, -1)).tril(diagonal=-1)
    expected = scores @ V
    for impl in ("eager", "blocked", "triton", "cuda"):
        monkeypatch.setenv("BDH_ATTN_IMPL", impl)
        got = bdh_attn(Q, K, V)
        assert torch.allclose(got, expected, rtol=1e-5, atol=1e-5)
    sm_out = torch.softmax(scores, dim=-1) @ V
    assert not torch.allclose(expected, sm_out, rtol=1e-3, atol=1e-3)
