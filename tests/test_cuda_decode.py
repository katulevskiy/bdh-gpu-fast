"""CUDA/C++ scaffold: single-token decode vs packed KR/V.

CPU reference always runs. Native CUDA path is skipped when unavailable
(no extension build and/or no CUDA device) — expected on CPU boxes.

Semantics: out = (Q @ K_past.mT) @ V_past — no softmax, no scale; past-only
so the new token never attends to itself (tril diagonal=-1).
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
from bdh_cache import CacheManager
from kernels.attention import eager_decode_attn, eager_tril_attn
from kernels.attention_dispatch import bdh_attn_decode
from kernels.cuda_attn import (
    ext_status,
    has_cuda_ext,
    has_cuda_kernel,
    tril_decode,
    tril_decode_ref,
)


def _make_decode_qkv(B=2, H=4, S=32, N=16, D=64, Tq=1, seed=0):
    g = torch.Generator().manual_seed(seed)
    Q = torch.randn(B, H, Tq, N, generator=g)
    K = torch.randn(B, H, S, N, generator=g)
    V = torch.randn(B, 1, S, D, generator=g)
    return Q, K, V


def test_ext_status_mentions_decode():
    s = ext_status()
    assert isinstance(s, str) and len(s) > 0
    print("cuda_attn decode:", s)


def test_decode_ref_matches_eager():
    Q, K, V = _make_decode_qkv(S=48, Tq=1, seed=1)
    assert torch.allclose(tril_decode_ref(Q, K, V), eager_decode_attn(Q, K, V), atol=0)


def test_decode_ref_broadcast_v_heads():
    Q, K, V = _make_decode_qkv(B=1, H=4, S=20, N=8, D=16, seed=2)
    out = tril_decode_ref(Q, K, V)
    gold = eager_decode_attn(Q, K, V)
    assert out.shape == (1, 4, 1, 16)
    assert torch.allclose(out, gold, rtol=1e-5, atol=1e-5)


def test_decode_matches_full_tril_last_row():
    """Concat Q into K/V + tril(-1); last query row must match decode."""
    B, H, S, N, D = 2, 4, 16, 8, 32
    g = torch.Generator().manual_seed(11)
    past_k = torch.randn(B, H, S, N, generator=g)
    past_v = torch.randn(B, 1, S, D, generator=g)
    q = torch.randn(B, H, 1, N, generator=g)
    K_all = torch.cat([past_k, q], dim=2)
    V_new = torch.randn(B, 1, 1, D, generator=g)
    V_all = torch.cat([past_v, V_new], dim=2)
    full = eager_tril_attn(K_all, K_all, V_all)
    last = full[:, :, -1:, :]
    dec = tril_decode_ref(q, past_k, past_v)
    assert torch.allclose(dec, last, rtol=1e-5, atol=1e-5)
    # Poison V_new — last row unchanged (no self-attention)
    V_all2 = torch.cat([past_v, torch.randn_like(V_new)], dim=2)
    full2 = eager_tril_attn(K_all, K_all, V_all2)
    assert torch.allclose(full[:, :, -1:, :], full2[:, :, -1:, :], atol=0)


def test_decode_s0_zeros():
    Q = torch.randn(1, 2, 1, 8)
    K = torch.zeros(1, 2, 0, 8)
    V = torch.zeros(1, 1, 0, 16)
    out = tril_decode_ref(Q, K, V)
    assert out.shape == (1, 2, 1, 16)
    assert torch.all(out == 0)


def test_tril_decode_dispatch_cpu():
    """Public dispatcher matches ref on CPU (ext or fallback)."""
    Q, K, V = _make_decode_qkv(S=24, seed=3)
    assert torch.allclose(tril_decode(Q, K, V), tril_decode_ref(Q, K, V), rtol=1e-5, atol=1e-5)


def test_bdh_attn_decode_cuda_impl():
    Q, K, V = _make_decode_qkv(S=20, seed=4)
    out = bdh_attn_decode(Q, K, V, impl="cuda")
    ref = eager_decode_attn(Q, K, V)
    assert torch.allclose(out, ref, rtol=1e-5, atol=1e-5)


def test_attention_t1_cuda_no_self_attend(monkeypatch):
    monkeypatch.setenv("BDH_ATTN_IMPL", "cuda")
    cfg = bdh.BDHConfig(
        n_layer=1, n_embd=64, n_head=4, mlp_internal_dim_multiplier=8, dropout=0.0
    )
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


def test_incremental_vs_full_cuda(monkeypatch):
    monkeypatch.setenv("BDH_ATTN_IMPL", "cuda")
    cfg = bdh.BDHConfig(
        n_layer=2, n_embd=64, n_head=4, mlp_internal_dim_multiplier=8, dropout=0.0, vocab_size=256
    )
    torch.manual_seed(41)
    model = bdh.BDH(cfg).eval()
    tokens = torch.randint(0, cfg.vocab_size, (2, 12))
    with torch.no_grad():
        full, _ = model(tokens)
        cache = [None] * cfg.n_layer
        parts = []
        for t in range(tokens.size(1)):
            lg, _ = model(tokens[:, t : t + 1], cache=cache)
            parts.append(lg)
        got = torch.cat(parts, dim=1)
    assert torch.allclose(full, got, rtol=1e-5, atol=1e-5)


def test_cachemanager_cuda_tokenwise(monkeypatch):
    monkeypatch.setenv("BDH_ATTN_IMPL", "cuda")
    cfg = bdh.BDHConfig(
        n_layer=2, n_embd=64, n_head=4, mlp_internal_dim_multiplier=8, dropout=0.0, vocab_size=256
    )
    torch.manual_seed(42)
    model = bdh.BDH(cfg).eval()
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
    assert torch.allclose(full, got, rtol=1e-5, atol=1e-5)


def test_default_eager_unchanged(monkeypatch):
    monkeypatch.delenv("BDH_ATTN_IMPL", raising=False)
    assert os.environ.get("BDH_ATTN_IMPL", "eager") == "eager"
    Q, K, V = _make_decode_qkv(S=10, seed=5)
    # Without env, Attention T=1 uses eager two-GEMM (not cuda decode).
    cfg = bdh.BDHConfig(
        n_layer=1, n_embd=64, n_head=4, mlp_internal_dim_multiplier=8, dropout=0.0
    )
    attn = bdh.Attention(cfg)
    B, nh, S = 1, cfg.n_head, 8
    N = cfg.mlp_internal_dim_multiplier * cfg.n_embd // nh
    D = cfg.n_embd
    torch.manual_seed(0)
    Q = torch.randn(B, nh, 1, N)
    past_kr = torch.randn(B, nh, S, N)
    past_v = torch.randn(B, 1, S, D)
    V = torch.randn(B, 1, 1, D)
    out, _, _ = attn(Q, Q, V, rope_start=S, past_kr=past_kr, past_v=past_v)
    # RoPE applied inside Attention — just shape smoke for default path
    assert out.shape == (B, nh, 1, D)


@pytest.mark.skipif(not has_cuda_ext(), reason="bdh_cuda_ext not built")
def test_native_cpu_decode_matches_ref():
    Q, K, V = _make_decode_qkv(S=15, seed=6)
    import bdh_cuda_ext

    if not hasattr(bdh_cuda_ext, "tril_decode_cpu"):
        pytest.skip("ext built without tril_decode (rebuild needed)")
    out = bdh_cuda_ext.tril_decode_cpu(Q, K, V)
    assert torch.allclose(out, tril_decode_ref(Q, K, V), rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(
    not (has_cuda_kernel() and torch.cuda.is_available()),
    reason="CUDA kernel not built or no CUDA device",
)
def test_cuda_decode_kernel_matches_ref():
    device = torch.device("cuda")
    g = torch.Generator(device=device).manual_seed(7)
    Q = torch.randn(2, 4, 1, 32, device=device, generator=g)
    K = torch.randn(2, 4, 64, 32, device=device, generator=g)
    V = torch.randn(2, 4, 64, 48, device=device, generator=g)
    out = tril_decode(Q, K, V)
    gold = tril_decode_ref(Q, K, V)
    assert torch.allclose(out, gold, rtol=1e-3, atol=1e-3)


@pytest.mark.skipif(
    not (has_cuda_kernel() and torch.cuda.is_available()),
    reason="CUDA kernel not built or no CUDA device",
)
def test_cuda_decode_broadcast_v():
    device = torch.device("cuda")
    Q = torch.randn(1, 4, 1, 16, device=device)
    K = torch.randn(1, 4, 32, 16, device=device)
    V = torch.randn(1, 1, 32, 24, device=device)
    out = tril_decode(Q, K, V)
    gold = tril_decode_ref(Q, K, V)
    assert out.shape == gold.shape
    assert torch.allclose(out, gold, rtol=1e-3, atol=1e-3)
