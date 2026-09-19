"""opt/prefill-blocked: deepen blocked/online cold for T≥256; AUTO long-T cold."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kernels.attention import (  # noqa: E402
    DEFAULT_BLOCK_COLD,
    blocked_tril_attn,
    eager_tril_attn,
    max_score_tile_elems,
    online_tril_attn,
    pick_cold_block_size,
)
from kernels.attention_dispatch import (  # noqa: E402
    attn_auto_threshold,
    bdh_attn,
    resolve_attn_impl,
    resolve_cold_impl,
    resolve_decode_impl,
)


def _bump():
    import kernels.attention_dispatch as d

    d._ATTN_IMPL_ENV = object()
    d._ATTN_AUTO_ENV = object()
    d._ATTN_AUTO_THR_ENV = object()


def _qkv(T, *, B=1, H=4, N=32, D=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    Q = torch.randn(B, H, T, N, generator=g)
    V = torch.randn(B, 1, T, D, generator=g)
    return Q, Q.clone(), V


@pytest.mark.parametrize("T", [256, 512, 1024])
def test_blocked_online_parity_long_t(T):
    Q, K, V = _qkv(T, seed=T)
    ref = eager_tril_attn(Q, K, V)
    blk = blocked_tril_attn(Q, K, V)
    onl = online_tril_attn(Q, K, V)
    # Tile reorder vs one eager GEMM — allow mild fp drift at long T.
    assert torch.allclose(blk, ref, rtol=1e-3, atol=1e-3), (
        f"T={T} maxdiff={(blk - ref).abs().max().item()}"
    )
    assert torch.equal(blk, onl)
    assert torch.count_nonzero(blk[:, :, 0, :]) == 0


def test_pick_cold_block_size_adaptive():
    assert pick_cold_block_size(64) == DEFAULT_BLOCK_COLD
    assert pick_cold_block_size(255) == DEFAULT_BLOCK_COLD
    assert pick_cold_block_size(256) == min(128, DEFAULT_BLOCK_COLD * 2)
    assert pick_cold_block_size(1024) == min(128, DEFAULT_BLOCK_COLD * 2)
    assert pick_cold_block_size(1024, block_size=32) == 32


def test_peak_bound_still_below_txt():
    for T in (256, 512, 1024, 2048):
        bound = max_score_tile_elems(T)
        assert bound < T * T
        assert bound == max_score_tile_elems(T, pick_cold_block_size(T))


def test_default_impl_eager_auto_off(monkeypatch):
    monkeypatch.delenv("BDH_ATTN_IMPL", raising=False)
    monkeypatch.delenv("BDH_ATTN_AUTO", raising=False)
    _bump()
    assert resolve_attn_impl() == "eager"
    assert resolve_cold_impl(2048) == "eager"
    Q, K, V = _qkv(32, seed=1)
    assert torch.equal(bdh_attn(Q, K, V), eager_tril_attn(Q, K, V))


def test_auto_cold_mirrors_decode_threshold(monkeypatch):
    monkeypatch.setenv("BDH_ATTN_AUTO", "1")
    monkeypatch.delenv("BDH_ATTN_IMPL", raising=False)
    monkeypatch.delenv("BDH_ATTN_AUTO_THRESHOLD", raising=False)
    _bump()
    thr = attn_auto_threshold()
    assert resolve_cold_impl(thr) == "eager"
    assert resolve_decode_impl(thr) == "eager"
    assert resolve_cold_impl(thr + 1) == resolve_decode_impl(thr + 1)
    assert resolve_cold_impl(thr + 1) != "eager"


def test_auto_does_not_override_explicit_blocked(monkeypatch):
    monkeypatch.setenv("BDH_ATTN_AUTO", "1")
    monkeypatch.setenv("BDH_ATTN_IMPL", "blocked")
    _bump()
    assert resolve_cold_impl(2048) == "blocked"
    assert resolve_decode_impl(2048) == "blocked"
