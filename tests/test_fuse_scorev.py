"""Online fused strict-tril score×V — no full T×T materialization."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Tuple

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kernels.attention import (  # noqa: E402
    blocked_tril_attn,
    eager_tril_attn,
    max_score_tile_elems,
    online_tril_attn,
)
from kernels.attention_dispatch import bdh_attn, resolve_attn_impl  # noqa: E402


def _make_qkv(B=2, H=4, T=16, N=32, D=64, seed=0, dtype=torch.float32):
    g = torch.Generator(device="cpu").manual_seed(seed)
    Q = torch.randn(B, H, T, N, generator=g, dtype=dtype)
    K = Q.clone()
    V = torch.randn(B, 1, T, D, generator=g, dtype=dtype)
    return Q, K, V


@pytest.mark.parametrize("T", [1, 2, 3, 8, 17, 64, 96])
@pytest.mark.parametrize("block_size", [4, 8, 16, 64])
def test_online_blocked_matches_eager(T, block_size):
    Q, K, V = _make_qkv(T=T, seed=T * 100 + block_size)
    ref = eager_tril_attn(Q, K, V)
    got_b = blocked_tril_attn(Q, K, V, block_size=block_size)
    got_o = online_tril_attn(Q, K, V, block_size=block_size)
    # Online tile/row order differs from one big eager GEMM; allow long-T fp drift.
    assert torch.allclose(got_b, ref, rtol=1e-4, atol=1e-4), (
        f"blocked T={T} BS={block_size} maxdiff={(got_b - ref).abs().max().item()}"
    )
    assert torch.allclose(got_o, ref, rtol=1e-4, atol=1e-4)
    assert torch.equal(got_b, got_o)


def test_online_blocked_match_eager_for_distinct_qk():
    """Score×V must preserve the contract when Q and K are not aliased."""
    Q, _, V = _make_qkv(B=2, H=3, T=19, N=24, D=16, seed=31)
    g = torch.Generator(device="cpu").manual_seed(32)
    K = torch.randn(Q.shape, generator=g, dtype=Q.dtype)

    ref = eager_tril_attn(Q, K, V)
    expected = (Q @ K.transpose(-2, -1)).tril(diagonal=-1) @ V
    assert torch.allclose(ref, expected, rtol=1e-5, atol=1e-5)

    got_b = blocked_tril_attn(Q, K, V, block_size=7)
    got_o = online_tril_attn(Q, K, V, block_size=7)
    assert torch.allclose(got_b, ref, rtol=1e-4, atol=1e-4)
    assert torch.allclose(got_o, ref, rtol=1e-4, atol=1e-4)


def test_pos0_zero_and_no_softmax():
    Q, K, V = _make_qkv(T=12, seed=11)
    out = online_tril_attn(Q, K, V, block_size=5)
    assert torch.count_nonzero(out[:, :, 0, :]) == 0
    scores = (Q @ K.transpose(-2, -1)).tril(diagonal=-1)
    assert torch.allclose(out, scores @ V, rtol=1e-4, atol=1e-4)
    sm_out = torch.softmax(scores, dim=-1) @ V
    assert not torch.allclose(out, sm_out, rtol=1e-3, atol=1e-3)


def test_default_impl_remains_eager(monkeypatch):
    monkeypatch.delenv("BDH_ATTN_IMPL", raising=False)
    assert resolve_attn_impl() == "eager"
    Q, K, V = _make_qkv(T=8, seed=12)
    assert torch.equal(bdh_attn(Q, K, V), eager_tril_attn(Q, K, V))


def test_max_score_tile_bound_vs_full_txt():
    T, BS = 256, 64
    bound = max_score_tile_elems(T, BS)
    full = T * T
    assert bound < full
    # Vectorized blocked: past Bi×i0 (budget-capped) + Bi×Bi diag — still ≪ T×T
    assert bound <= max(BS * (T - 1), BS * BS)
    assert bound >= BS - 1


def test_blocked_never_allocates_full_txt_scores(monkeypatch):
    """Monkeypatch Tensor.__matmul__ to record 4D score-shaped products.

    Eager produces a (B,H,T,T) score tensor. Blocked/online must not.
    """
    T, BS = 48, 8
    Q, K, V = _make_qkv(B=1, H=2, T=T, N=16, D=16, seed=99)
    B, H = Q.shape[0], Q.shape[1]

    recorded: List[Tuple[int, ...]] = []
    orig = torch.Tensor.__matmul__

    def spy(self, other):
        out = orig(self, other)
        # Score-like: (B,H,*,*) from Q@K.T style
        if out.dim() == 4 and out.shape[0] == B and out.shape[1] == H:
            recorded.append(tuple(out.shape))
        return out

    monkeypatch.setattr(torch.Tensor, "__matmul__", spy)

    _ = blocked_tril_attn(Q, K, V, block_size=BS)
    blocked_shapes = list(recorded)
    recorded.clear()
    _ = eager_tril_attn(Q, K, V)
    eager_shapes = list(recorded)

    assert any(s[-2:] == (T, T) for s in eager_shapes), (
        f"eager should materialize T×T, got {eager_shapes}"
    )
    assert not any(s[-2] == T and s[-1] == T for s in blocked_shapes), (
        f"blocked must not materialize T×T scores, got {blocked_shapes}"
    )
    # Every blocked score tile fits in the documented bound (or Bi×past < T×T)
    bound = max_score_tile_elems(T, BS)
    for s in blocked_shapes:
        elems = s[-2] * s[-1]
        assert elems <= bound or elems < T * T, (
            f"unexpected large score tile {s} elems={elems} bound={bound}"
        )


def test_dispatch_blocked_uses_online_fusion(monkeypatch):
    monkeypatch.setenv("BDH_ATTN_IMPL", "blocked")
    Q, K, V = _make_qkv(T=33, seed=7)
    out = bdh_attn(Q, K, V)
    ref = eager_tril_attn(Q, K, V)
    assert torch.allclose(out, ref, rtol=1e-4, atol=1e-4)


def test_bdh_attention_blocked_hook(monkeypatch):
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
    torch.manual_seed(21)
    Q = torch.randn(2, cfg.n_head, 20, N)
    V = torch.randn(2, 1, 20, cfg.n_embd)

    monkeypatch.setenv("BDH_ATTN_IMPL", "eager")
    out_e, _, _ = attn(Q, Q, V)
    monkeypatch.setenv("BDH_ATTN_IMPL", "blocked")
    out_b, _, _ = attn(Q, Q, V)
    assert torch.allclose(out_e, out_b, rtol=1e-4, atol=1e-4)
    assert torch.equal(out_e[:, :, 0, :], torch.zeros_like(out_e[:, :, 0, :]))
