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
    d._ATTN_AUTO_COLD_THR_ENV = object()


def _qkv(T, *, B=1, H=4, N=32, D=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    Q = torch.randn(B, H, T, N, generator=g)
    V = torch.randn(B, 1, T, D, generator=g)
    return Q, Q.clone(), V


def test_strict_tril_raw_score_contract_at_tile_boundary():
    """Blocked tiles use raw scores and exclude diagonal/future keys."""
    Q = torch.tensor([2.0, 3.0, 5.0, 7.0, 11.0], dtype=torch.float64).view(
        1, 1, 5, 1
    )
    K = torch.tensor([13.0, 17.0, 19.0, 23.0, 29.0], dtype=torch.float64).view(
        1, 1, 5, 1
    )
    V = torch.tensor(
        [[[[1.0, -1.0], [2.0, -2.0], [4.0, -4.0], [8.0, -8.0], [16.0, -16.0]]]],
        dtype=torch.float64,
    )
    expected = torch.tril(Q @ K.transpose(-2, -1), diagonal=-1) @ V

    for impl in (blocked_tril_attn, online_tril_attn):
        got = impl(Q, K, V, block_size=2)
        assert torch.equal(got, expected)


@pytest.mark.parametrize("value_heads", [1, 2])
def test_batched_strict_tril_raw_score_contract_partial_tile(value_heads):
    """Batched and head-matched tiles keep exact raw-score masking."""
    B, H, T, N, D = 2, 2, 7, 2, 2
    Q = torch.arange(-20, -20 + B * H * T * N, dtype=torch.float64).view(
        B, H, T, N
    )
    K = torch.arange(11, 11 + B * H * T * N, dtype=torch.float64).view(
        B, H, T, N
    )
    V = torch.arange(-7, -7 + B * value_heads * T * D, dtype=torch.float64).view(
        B, value_heads, T, D
    )
    expected = torch.tril(Q @ K.transpose(-2, -1), diagonal=-1) @ V

    got = blocked_tril_attn(Q, K, V, block_size=3)
    assert torch.equal(got, expected)


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


@pytest.mark.parametrize("value_heads", [1, 3])
def test_cpu_flattened_bmm_broadcast_and_head_matched(value_heads):
    """Long CPU cold tiles keep parity for broadcast and per-head V."""
    Q, K, _ = _qkv(256, B=2, H=3, N=7, D=5, seed=17 + value_heads)
    g = torch.Generator().manual_seed(23 + value_heads)
    V = torch.randn(2, value_heads, 256, 5, generator=g)
    ref = eager_tril_attn(Q, K, V)
    got = blocked_tril_attn(Q, K, V, block_size=128)
    assert torch.allclose(got, ref, rtol=1e-4, atol=1e-4)
    assert torch.count_nonzero(got[:, :, 0, :]) == 0


@pytest.mark.parametrize("value_heads", [1, 2])
def test_cpu_flattened_bmm_padded_v_view_preserves_layout_contract(value_heads):
    """Long CPU cold tiles preserve parity for capacity-padded V views."""
    T, B, H, N, D = 257, 2, 2, 5, 4
    g = torch.Generator().manual_seed(27 + value_heads)
    Q = torch.randn(B, H, T, N, generator=g)
    K = torch.randn(B, H, T, N, generator=g)
    V_storage = torch.randn(B, value_heads, T, D + 1, generator=g)
    V = V_storage[..., :D]

    assert not V.is_contiguous()
    assert V.stride(-2) == D + 1
    ref = eager_tril_attn(Q, K, V)
    got = blocked_tril_attn(Q, K, V, block_size=128)
    assert torch.allclose(got, ref, rtol=1e-4, atol=1e-4)
    assert torch.count_nonzero(got[:, :, 0, :]) == 0


@pytest.mark.parametrize("impl", ["blocked", "online"])
def test_cpu_flattened_bmm_padded_qk_views_preserve_layout_contract(impl):
    """Long CPU cold tiles preserve parity for padded Q/K views."""
    T, B, H, N, D = 257, 2, 2, 5, 4
    g = torch.Generator().manual_seed(28)
    Q_storage = torch.randn(B, H, T, N + 1, generator=g)
    K_storage = torch.randn(B, H, T, N + 1, generator=g)
    Q = Q_storage[..., :N]
    K = K_storage[..., :N]
    V = torch.randn(B, 1, T, D, generator=g)

    assert not Q.is_contiguous()
    assert not K.is_contiguous()
    assert Q.stride(-2) == N + 1
    assert K.stride(-2) == N + 1
    ref = eager_tril_attn(Q, K, V)
    tiled = blocked_tril_attn if impl == "blocked" else online_tril_attn
    got = tiled(Q, K, V, block_size=128)
    assert torch.allclose(got, ref, rtol=1e-4, atol=1e-4)
    assert torch.count_nonzero(got[:, :, 0, :]) == 0


@pytest.mark.parametrize("impl", ["blocked", "online"])
@pytest.mark.parametrize("value_heads", [1, 2])
def test_cpu_long_padded_v_autograd_matches_eager(impl, value_heads):
    """Long CPU tiles preserve gradients through a capacity-padded V view."""
    T, B, H, N, D = 257, 2, 2, 5, 4
    g = torch.Generator().manual_seed(121 + value_heads)
    Q0 = torch.randn(B, H, T, N, dtype=torch.float64, generator=g)
    K0 = torch.randn(B, H, T, N, dtype=torch.float64, generator=g)
    V_storage0 = torch.randn(
        B, value_heads, T, D + 1, dtype=torch.float64, generator=g
    )
    weight = torch.randn(B, H, T, D, dtype=torch.float64, generator=g)

    def run(fn):
        Q = Q0.clone().requires_grad_()
        K = K0.clone().requires_grad_()
        V_storage = V_storage0.clone().requires_grad_()
        V = V_storage[..., :D]
        assert not V.is_contiguous()
        assert V.stride(-2) == D + 1
        out = fn(Q, K, V)
        grads = torch.autograd.grad((out * weight).sum(), (Q, K, V_storage))
        return out, grads

    ref, ref_grads = run(eager_tril_attn)
    tiled = blocked_tril_attn if impl == "blocked" else online_tril_attn
    got, got_grads = run(lambda Q, K, V: tiled(Q, K, V, block_size=128))
    assert torch.allclose(got, ref, rtol=1e-9, atol=1e-9)
    for got_grad, ref_grad in zip(got_grads, ref_grads):
        assert torch.allclose(got_grad, ref_grad, rtol=1e-9, atol=1e-9)
    assert torch.count_nonzero(got[:, :, 0, :]) == 0
    assert torch.count_nonzero(got_grads[2][..., D]) == 0


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("value_heads", [1, 2])
def test_cpu_flattened_bmm_low_precision_padded_v_view_matches_eager(
    dtype, value_heads
):
    """Low-precision cold tiles preserve parity for capacity-padded V views."""
    T, B, H, N, D = 257, 2, 2, 5, 4
    g = torch.Generator().manual_seed(29 + value_heads)
    Q = torch.randn(B, H, T, N, dtype=dtype, generator=g) * 0.125
    K = torch.randn(B, H, T, N, dtype=dtype, generator=g) * 0.125
    V_storage = torch.randn(
        B, value_heads, T, D + 1, dtype=dtype, generator=g
    ) * 0.125
    V = V_storage[..., :D]

    assert not V.is_contiguous()
    assert V.stride(-2) == D + 1
    ref = eager_tril_attn(Q, K, V)
    got = blocked_tril_attn(Q, K, V, block_size=128)
    assert got.dtype == dtype
    assert torch.allclose(got, ref, rtol=2e-2, atol=2e-2)
    assert torch.count_nonzero(got[:, :, 0, :]) == 0


@pytest.mark.parametrize("value_heads", [1, 3])
def test_cpu_flattened_bmm_wide_head_parity(value_heads):
    """Wide N/D heads preserve CPU cold parity for both V layouts."""
    T, B, H, N, D = 512, 1, 3, 128, 256
    g = torch.Generator().manual_seed(31 + value_heads)
    Q = torch.randn(B, H, T, N, generator=g)
    K = torch.randn(B, H, T, N, generator=g)
    V = torch.randn(B, value_heads, T, D, generator=g)

    ref = eager_tril_attn(Q, K, V)
    got = blocked_tril_attn(Q, K, V)
    assert torch.allclose(got, ref, rtol=1e-3, atol=1e-3)
    assert torch.count_nonzero(got[:, :, 0, :]) == 0


@pytest.mark.parametrize("value_heads", [1, 3])
def test_cpu_wide_head_partial_tile_parity(value_heads):
    """Wide heads keep CPU cold parity across a partial 128-row tile."""
    T, B, H, N, D = 257, 2, 3, 160, 192
    g = torch.Generator().manual_seed(41 + value_heads)
    Q = torch.randn(B, H, T, N, generator=g)
    K = torch.randn(B, H, T, N, generator=g)
    V = torch.randn(B, value_heads, T, D, generator=g)

    ref = eager_tril_attn(Q, K, V)
    got = blocked_tril_attn(Q, K, V, block_size=128)
    assert torch.allclose(got, ref, rtol=1e-3, atol=1e-3)
    assert torch.count_nonzero(got[:, :, 0, :]) == 0


@pytest.mark.parametrize("impl", ["blocked", "online"])
@pytest.mark.parametrize("value_heads", [1, 2])
@pytest.mark.parametrize("batch_size", [1, 2])
def test_cpu_long_blocked_online_autograd_matches_eager(
    impl, value_heads, batch_size
):
    """Long flattened CPU tiles preserve raw-score forward and gradient parity."""
    T, B, H, N, D = 257, batch_size, 2, 5, 4
    g = torch.Generator().manual_seed(101 + value_heads + batch_size)
    Q0 = torch.randn(B, H, T, N, dtype=torch.float64, generator=g)
    K0 = torch.randn(B, H, T, N, dtype=torch.float64, generator=g)
    V0 = torch.randn(B, value_heads, T, D, dtype=torch.float64, generator=g)
    weight = torch.randn(B, H, T, D, dtype=torch.float64, generator=g)

    def run(fn):
        Q = Q0.clone().requires_grad_()
        K = K0.clone().requires_grad_()
        V = V0.clone().requires_grad_()
        out = fn(Q, K, V)
        grads = torch.autograd.grad((out * weight).sum(), (Q, K, V))
        return out, grads

    ref, ref_grads = run(eager_tril_attn)
    tiled = blocked_tril_attn if impl == "blocked" else online_tril_attn
    got, got_grads = run(lambda Q, K, V: tiled(Q, K, V, block_size=128))
    assert torch.allclose(got, ref, rtol=1e-9, atol=1e-9)
    for got_grad, ref_grad in zip(got_grads, ref_grads):
        assert torch.allclose(got_grad, ref_grad, rtol=1e-9, atol=1e-9)
    assert torch.count_nonzero(got[:, :, 0, :]) == 0


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("value_heads", [1, 2])
def test_cpu_long_blocked_low_precision_matches_eager(dtype, value_heads):
    """CPU cold tiles preserve raw-score parity while widening half inputs."""
    T, B, H, N, D = 257, 2, 2, 3, 4
    g = torch.Generator().manual_seed(131 + value_heads)
    Q = torch.randn(B, H, T, N, dtype=dtype, generator=g) * 0.125
    K = torch.randn(B, H, T, N, dtype=dtype, generator=g) * 0.125
    V = torch.randn(B, value_heads, T, D, dtype=dtype, generator=g) * 0.125

    ref = eager_tril_attn(Q, K, V)
    got = blocked_tril_attn(Q, K, V, block_size=128)
    assert got.dtype == dtype
    assert torch.allclose(got, ref, rtol=2e-2, atol=2e-2)
    assert torch.count_nonzero(got[:, :, 0, :]) == 0


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


@pytest.mark.parametrize("value_heads", [1, 2])
def test_auto_cold_dispatch_preserves_blocked_contract_cpu(monkeypatch, value_heads):
    """AUTO cold dispatch keeps raw-score parity for shared/head-matched V."""
    monkeypatch.setenv("BDH_ATTN_AUTO", "1")
    monkeypatch.setenv("BDH_ATTN_AUTO_COLD_THRESHOLD", "256")
    monkeypatch.delenv("BDH_ATTN_IMPL", raising=False)
    _bump()

    Q, K, V_shared = _qkv(257, B=2, H=2, N=5, D=4, seed=211)
    if value_heads == 1:
        V = V_shared
    else:
        g = torch.Generator().manual_seed(212)
        V = torch.randn(2, value_heads, 257, 4, generator=g)
    assert resolve_cold_impl(Q.size(2)) == "blocked"
    got = bdh_attn(Q, K, V)
    ref = eager_tril_attn(Q, K, V)
    assert torch.allclose(got, ref, rtol=1e-3, atol=1e-3)
    assert torch.count_nonzero(got[:, :, 0, :]) == 0


def test_auto_does_not_override_explicit_blocked(monkeypatch):
    monkeypatch.setenv("BDH_ATTN_AUTO", "1")
    monkeypatch.setenv("BDH_ATTN_IMPL", "blocked")
    _bump()
    assert resolve_cold_impl(2048) == "blocked"
    assert resolve_decode_impl(2048) == "blocked"
