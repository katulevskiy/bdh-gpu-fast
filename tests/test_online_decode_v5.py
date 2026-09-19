"""Focused CPU coverage for the post-#187 online decode path."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import bdh
from bdh_cache import CacheManager
from kernels.attention import eager_decode_attn, online_decode_attn
from kernels.attention_dispatch import bdh_attn_decode


def test_b3_long_s_packed_shared_v_decode_parity():
    """B>1 long-S packed KR/V views stay exact across CPU decode backends."""
    B, H, S, N, D = 3, 4, 2049, 16, 32
    cfg = bdh.BDHConfig(
        n_layer=1,
        n_embd=D,
        n_head=H,
        mlp_internal_dim_multiplier=8,
        dropout=0.0,
        vocab_size=128,
    )
    cm = CacheManager(
        n_layer=1,
        max_seq=S + 31,
        batch_size=B,
        n_head=H,
        n_latent=N,
        n_embd=D,
        device="cpu",
    )
    torch.manual_seed(905)
    cm._kr_buf[0].normal_()
    cm._v_buf[0].normal_()
    cm.seq_len = S
    K, V = cm.get_past(0)
    assert K is not None and V is not None
    assert K.stride() == (H * cm.capacity * N, cm.capacity * N, N, 1)
    assert V.stride() == (cm.capacity * D, cm.capacity * D, D, 1)

    Q = torch.randn(B, H, 1, N)
    ref = eager_decode_attn(Q, K, V)
    for impl in ("blocked", "online", "triton", "cuda"):
        got = bdh_attn_decode(Q, K, V, impl=impl)
        assert torch.allclose(got, ref, rtol=1e-4, atol=1e-5), (
            f"impl={impl} maxdiff={(got - ref).abs().max().item()}"
        )


def test_decode_uses_raw_scores_without_softmax_or_scale():
    """Every decode backend keeps raw QK scores: no softmax or scale."""
    Q = torch.tensor([[[[2.0, 0.0]]]])
    K_past = torch.tensor([[[[1.0, 0.0], [0.5, 0.0]]]])
    V_past = torch.tensor([[[[3.0], [4.0]]]])
    expected = torch.tensor([[[[10.0]]]])

    for impl in ("eager", "blocked", "online", "triton", "cuda"):
        got = bdh_attn_decode(Q, K_past, V_past, impl=impl)
        assert torch.allclose(got, expected, atol=0, rtol=0), (
            f"impl={impl} maxdiff={(got - expected).abs().max().item()}"
        )


def test_decode_preserves_signed_raw_scores_with_per_head_values():
    """Signed raw QK scores and independent per-head V stay unnormalized."""
    Q = torch.tensor(
        [
            [[[2.0, -1.0]], [[-1.0, 2.0]]],
            [[[1.0, 1.0]], [[2.0, 1.0]]],
        ]
    )
    K_past = torch.tensor(
        [
            [[[1.0, 0.0], [0.0, 2.0]], [[1.0, 1.0], [2.0, 0.0]]],
            [[[1.0, -1.0], [2.0, 1.0]], [[1.0, 0.0], [0.0, -1.0]]],
        ]
    )
    V_past = torch.tensor(
        [
            [[[3.0, 5.0], [-2.0, 4.0]], [[7.0, -1.0], [1.0, 2.0]]],
            [[[2.0, 3.0], [4.0, -1.0]], [[5.0, 6.0], [-3.0, 2.0]]],
        ]
    )
    expected = torch.tensor(
        [
            [[[10.0, 2.0]], [[5.0, -5.0]]],
            [[[12.0, -3.0]], [[13.0, 10.0]]],
        ]
    )

    for impl in ("eager", "blocked", "online", "triton", "cuda"):
        got = bdh_attn_decode(Q, K_past, V_past, impl=impl)
        assert torch.equal(got, expected), (
            f"impl={impl} maxdiff={(got - expected).abs().max().item()}"
        )


def test_multi_query_decode_keeps_signed_raw_scores_per_head():
    """Each query row uses raw scores against past-only, per-head values."""
    Q = torch.tensor(
        [[[[2.0, -1.0], [-1.0, 2.0]], [[1.0, 2.0], [2.0, -1.0]]]]
    )
    K_past = torch.tensor(
        [[[[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]]]]
    )
    V_past = torch.tensor(
        [[[[3.0, 5.0], [-2.0, 4.0]], [[7.0, -1.0], [1.0, 2.0]]]]
    )
    expected = torch.tensor(
        [[[[8.0, 6.0], [-7.0, 3.0]], [[9.0, 3.0], [13.0, -4.0]]]]
    )

    for impl in ("eager", "blocked", "online", "triton", "cuda"):
        got = bdh_attn_decode(Q, K_past, V_past, impl=impl, block_size=1)
        assert torch.equal(got, expected), (
            f"impl={impl} maxdiff={(got - expected).abs().max().item()}"
        )


def test_decode_dispatch_multi_query_keeps_signed_raw_scores_with_shared_v():
    """Tiled multi-query dispatch keeps raw scores with broadcast V heads."""
    Q = torch.tensor(
        [[[[2.0, -1.0], [-1.0, 2.0]], [[1.0, 2.0], [2.0, -1.0]]]]
    )
    K_past = torch.tensor(
        [[[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
          [[1.0, 0.0], [0.0, 1.0], [-1.0, 1.0]]]]
    )
    V_past = torch.tensor([[[[2.0], [3.0], [5.0]]]])
    expected = torch.tensor(
        [[[[6.0], [9.0]], [[13.0], [-14.0]]]]
    )

    for impl in ("eager", "blocked", "online", "triton", "cuda"):
        got = bdh_attn_decode(Q, K_past, V_past, impl=impl, block_size=1)
        assert torch.equal(got, expected), (
            f"impl={impl} maxdiff={(got - expected).abs().max().item()}"
        )


def test_online_decode_tiled_multi_query_keeps_raw_signed_scores():
    """Direct online decode keeps signed score×V semantics after tiling."""
    S = 513
    Q = torch.tensor([[[[2.0, -1.0], [-1.0, 2.0]]]])
    K_past = torch.zeros(1, 1, S, 2)
    K_past[:, :, 0, :] = torch.tensor([1.0, 0.0])
    K_past[:, :, 1, :] = torch.tensor([0.0, 1.0])
    V_past = torch.zeros(1, 1, S, 1)
    V_past[:, :, 0, 0] = 3.0
    V_past[:, :, 1, 0] = 4.0

    got = online_decode_attn(Q, K_past, V_past, block_size=1)
    expected = torch.tensor([[[[2.0], [5.0]]]])
    assert torch.equal(got, expected)


def test_online_decode_multi_query_packed_shared_v_tiled_parity():
    """Multi-query online decode preserves packed shared-V views across tiles."""
    B, H, S, Tq, N, D = 2, 3, 353, 3, 4, 2
    capacity = S + 11
    g = torch.Generator().manual_seed(811)
    Q = torch.randn(B, H, Tq, N, generator=g)
    K = torch.randn(B, H, S, N, generator=g)
    V_storage = torch.randn(B, 1, capacity, D, generator=g)
    V = V_storage.narrow(2, 5, S)

    assert V.stride() == (capacity * D, capacity * D, D, 1)
    ref = eager_decode_attn(Q, K, V)
    got = online_decode_attn(Q, K, V, block_size=64)

    assert torch.allclose(got, ref, rtol=1e-4, atol=1e-5), (
        f"maxdiff={(got - ref).abs().max().item()}"
    )


def test_online_decode_block_size_invariance_for_shared_v():
    """Shared-V decode stays past-only across single and split tiles."""
    B, H, S, Tq, N, D = 2, 3, 65, 3, 4, 2
    g = torch.Generator().manual_seed(1213)
    Q = torch.randn(B, H, Tq, N, generator=g)
    K = torch.randn(B, H, S, N, generator=g)
    V_storage = torch.randn(B, 1, S + 9, D, generator=g)
    V = V_storage.narrow(2, 4, S)

    ref = eager_decode_attn(Q, K, V)
    for block_size in (1, 16, 64, 128):
        got = online_decode_attn(Q, K, V, block_size=block_size)
        assert torch.allclose(got, ref, rtol=1e-4, atol=1e-5), (
            f"block_size={block_size} maxdiff={(got - ref).abs().max().item()}"
        )


def test_online_decode_block_size_invariance_for_per_head_v():
    """Per-head V decode keeps raw score×V results across tile boundaries."""
    B, H, S, Tq, N, D = 2, 3, 65, 3, 4, 2
    g = torch.Generator().manual_seed(2627)
    Q = torch.randn(B, H, Tq, N, generator=g)
    K = torch.randn(B, H, S, N, generator=g)
    V = torch.randn(B, H, S, D, generator=g)

    ref = eager_decode_attn(Q, K, V)
    for block_size in (1, 16, 64, 128):
        got = online_decode_attn(Q, K, V, block_size=block_size)
        assert torch.allclose(got, ref, rtol=1e-4, atol=1e-5), (
            f"block_size={block_size} maxdiff={(got - ref).abs().max().item()}"
        )


def test_online_decode_preserves_offset_packed_shared_v_views():
    """Long CPU decode keeps nonzero-offset packed K/V views exact and read-only."""
    B, H, S, N, D = 2, 3, 1025, 4, 2
    offset = 5
    capacity = S + 17
    g = torch.Generator().manual_seed(912)
    k_storage = torch.randn(B, H, offset + capacity, N, generator=g)
    v_storage = torch.randn(B, 1, offset + capacity, D, generator=g)
    K = k_storage.narrow(2, offset, S)
    V = v_storage.narrow(2, offset, S)
    Q = torch.randn(B, H, 1, N, generator=g)

    assert K.stride() == (
        H * (offset + capacity) * N, (offset + capacity) * N, N, 1
    )
    assert V.stride() == ((offset + capacity) * D, (offset + capacity) * D, D, 1)
    K_before, V_before = K.clone(), V.clone()
    ref = eager_decode_attn(Q, K, V)
    got = online_decode_attn(Q, K, V, block_size=64)

    assert torch.allclose(got, ref, rtol=1e-4, atol=1e-5)
    assert torch.equal(K, K_before)
    assert torch.equal(V, V_before)


def test_online_decode_multi_query_preserves_offset_packed_kv_views():
    """Multi-query tiled decode keeps offset packed K/V views exact and read-only."""
    B, H, S, Tq, N, D = 2, 3, 2049, 3, 4, 2
    offset = 7
    capacity = S + 19
    g = torch.Generator().manual_seed(1314)
    k_storage = torch.randn(B, H, offset + capacity, N, generator=g)
    v_storage = torch.randn(B, 1, offset + capacity, D, generator=g)
    K = k_storage.narrow(2, offset, S)
    V = v_storage.narrow(2, offset, S)
    Q = torch.randn(B, H, Tq, N, generator=g)

    assert K.stride() == (
        H * (offset + capacity) * N, (offset + capacity) * N, N, 1
    )
    assert V.stride() == ((offset + capacity) * D, (offset + capacity) * D, D, 1)
    K_before, V_before = K.clone(), V.clone()
    ref = eager_decode_attn(Q, K, V)
    got = online_decode_attn(Q, K, V, block_size=64)

    assert torch.allclose(got, ref, rtol=1e-4, atol=1e-5), (
        f"maxdiff={(got - ref).abs().max().item()}"
    )
    assert torch.equal(K, K_before)
    assert torch.equal(V, V_before)


def test_decode_empty_past_returns_typed_zeros_for_all_backends():
    """An empty past is a zero score×V result with the query's shape and dtype."""
    B, H, Tq, N, D = 2, 3, 4, 5, 7
    Q = torch.randn(B, H, Tq, N, dtype=torch.float32)
    K_past = torch.empty(B, H, 0, N, dtype=Q.dtype)
    V_past = torch.empty(B, 1, 0, D, dtype=Q.dtype)
    expected = torch.zeros(B, H, Tq, D, dtype=Q.dtype)

    for impl in ("eager", "blocked", "online", "triton", "cuda"):
        got = bdh_attn_decode(Q, K_past, V_past, impl=impl)
        assert got.shape == expected.shape
        assert got.dtype == expected.dtype
        assert torch.equal(got, expected), f"impl={impl} returned {got}"


def test_decode_empty_past_preserves_float64_query_dtype():
    """Empty decode keeps float64 instead of silently widening or narrowing zeros."""
    B, H, Tq, N, D = 1, 2, 3, 4, 5
    Q = torch.randn(B, H, Tq, N, dtype=torch.float64)
    K_past = torch.empty(B, H, 0, N, dtype=Q.dtype)
    V_past = torch.empty(B, 1, 0, D, dtype=Q.dtype)

    for impl in ("eager", "blocked", "online", "triton", "cuda"):
        got = bdh_attn_decode(Q, K_past, V_past, impl=impl)
        assert got.shape == (B, H, Tq, D)
        assert got.dtype == Q.dtype
        assert torch.equal(got, torch.zeros_like(got)), f"impl={impl} returned {got}"


def test_online_decode_tiled_nonempty_preserves_query_dtype():
    """A tiled nonempty CPU decode keeps score×V output in the query dtype."""
    B, H, S, Tq, N, D = 2, 3, 1025, 2, 4, 3
    g = torch.Generator().manual_seed(1516)
    Q = torch.randn(B, H, Tq, N, dtype=torch.float64, generator=g)
    K = torch.randn(B, H, S, N, dtype=torch.float64, generator=g)
    V = torch.randn(B, 1, S, D, dtype=torch.float64, generator=g)

    ref = eager_decode_attn(Q, K, V)
    got = online_decode_attn(Q, K, V, block_size=64)

    assert got.shape == (B, H, Tq, D)
    assert got.dtype == Q.dtype
    assert torch.allclose(got, ref, rtol=1e-10, atol=1e-10), (
        f"maxdiff={(got - ref).abs().max().item()}"
    )


def test_decode_dispatch_nonempty_preserves_float64_query_dtype():
    """Every CPU-safe decode dispatch keeps float64 on a packed cache view."""
    B, H, S, N, D = 2, 3, 1025, 4, 3
    offset = 3
    capacity = S + 11
    g = torch.Generator().manual_seed(1718)
    Q = torch.randn(B, H, 1, N, dtype=torch.float64, generator=g)
    K_storage = torch.randn(B, H, offset + capacity, N, dtype=Q.dtype, generator=g)
    V_storage = torch.randn(B, 1, offset + capacity, D, dtype=Q.dtype, generator=g)
    K = K_storage.narrow(2, offset, S)
    V = V_storage.narrow(2, offset, S)

    ref = eager_decode_attn(Q, K, V)
    for impl in ("eager", "blocked", "online", "triton", "cuda"):
        got = bdh_attn_decode(Q, K, V, impl=impl, block_size=64)
        assert got.shape == (B, H, 1, D)
        assert got.dtype == Q.dtype
        assert torch.allclose(got, ref, rtol=1e-10, atol=1e-10), (
            f"impl={impl} maxdiff={(got - ref).abs().max().item()}"
        )


def test_decode_dispatch_per_head_preserves_float64_query_dtype():
    """Per-head V keeps float64 through every CPU-safe decode dispatch."""
    B, H, S, Tq, N, D = 2, 3, 1025, 3, 4, 3
    offset = 3
    capacity = S + 11
    g = torch.Generator().manual_seed(1819)
    Q = torch.randn(B, H, Tq, N, dtype=torch.float64, generator=g)
    K_storage = torch.randn(
        B, H, offset + capacity, N, dtype=Q.dtype, generator=g
    )
    V_storage = torch.randn(
        B, H, offset + capacity, D, dtype=Q.dtype, generator=g
    )
    K = K_storage.narrow(2, offset, S)
    V = V_storage.narrow(2, offset, S)

    ref = eager_decode_attn(Q, K, V)
    for impl in ("eager", "blocked", "online", "triton", "cuda"):
        got = bdh_attn_decode(Q, K, V, impl=impl, block_size=64)
        assert got.shape == (B, H, Tq, D)
        assert got.dtype == Q.dtype
        assert torch.allclose(got, ref, rtol=1e-10, atol=1e-10), (
            f"impl={impl} maxdiff={(got - ref).abs().max().item()}"
        )


def test_decode_dispatch_multi_query_preserves_offset_packed_kv_views():
    """All decode dispatches preserve multi-query offset packed K/V semantics."""
    B, H, S, Tq, N, D = 2, 3, 1025, 3, 4, 2
    offset = 7
    capacity = S + 19
    g = torch.Generator().manual_seed(1920)
    K_storage = torch.randn(B, H, offset + capacity, N, generator=g)
    V_storage = torch.randn(B, 1, offset + capacity, D, generator=g)
    K = K_storage.narrow(2, offset, S)
    V = V_storage.narrow(2, offset, S)
    Q = torch.randn(B, H, Tq, N, generator=g)
    K_before, V_before = K.clone(), V.clone()

    ref = eager_decode_attn(Q, K, V)
    for impl in ("eager", "blocked", "online", "triton", "cuda"):
        got = bdh_attn_decode(Q, K, V, impl=impl, block_size=64)
        assert got.shape == (B, H, Tq, D)
        assert torch.allclose(got, ref, rtol=1e-4, atol=1e-5), (
            f"impl={impl} maxdiff={(got - ref).abs().max().item()}"
        )
        assert torch.equal(K, K_before)
        assert torch.equal(V, V_before)


def test_online_decode_tiled_multi_query_preserves_per_head_v_views():
    """Direct tiled online decode keeps per-head V semantics on packed views."""
    B, H, S, Tq, N, D = 2, 3, 2049, 3, 4, 2
    offset = 7
    capacity = S + 19
    g = torch.Generator().manual_seed(2122)
    K_storage = torch.randn(B, H, offset + capacity, N, generator=g)
    V_storage = torch.randn(B, H, offset + capacity, D, generator=g)
    K = K_storage.narrow(2, offset, S)
    V = V_storage.narrow(2, offset, S)
    Q = torch.randn(B, H, Tq, N, generator=g)
    K_before, V_before = K.clone(), V.clone()

    ref = eager_decode_attn(Q, K, V)
    got = online_decode_attn(Q, K, V, block_size=64)

    assert torch.allclose(got, ref, rtol=1e-4, atol=1e-5), (
        f"maxdiff={(got - ref).abs().max().item()}"
    )
    assert torch.equal(K, K_before)
    assert torch.equal(V, V_before)


def test_decode_dispatch_t1_preserves_per_head_v_tiled_views():
    """T=1 tiled dispatch keeps per-head V semantics on packed views."""
    B, H, S, N, D = 2, 3, 1025, 4, 2
    offset = 7
    capacity = S + 19
    g = torch.Generator().manual_seed(2223)
    K_storage = torch.randn(B, H, offset + capacity, N, generator=g)
    V_storage = torch.randn(B, H, offset + capacity, D, generator=g)
    K = K_storage.narrow(2, offset, S)
    V = V_storage.narrow(2, offset, S)
    Q = torch.randn(B, H, 1, N, generator=g)
    K_before, V_before = K.clone(), V.clone()

    assert K.stride() == (
        H * (offset + capacity) * N, (offset + capacity) * N, N, 1
    )
    assert V.stride() == (
        H * (offset + capacity) * D, (offset + capacity) * D, D, 1
    )
    ref = eager_decode_attn(Q, K, V)
    for impl in ("eager", "blocked", "online", "triton", "cuda"):
        got = bdh_attn_decode(Q, K, V, impl=impl, block_size=64)
        assert got.shape == (B, H, 1, D)
        assert torch.allclose(got, ref, rtol=1e-4, atol=1e-5), (
            f"impl={impl} maxdiff={(got - ref).abs().max().item()}"
        )
        assert torch.equal(K, K_before)
        assert torch.equal(V, V_before)


def test_decode_dispatch_multi_query_preserves_offset_packed_per_head_v_views():
    """Multi-query dispatch keeps per-head V semantics on packed views."""
    B, H, S, Tq, N, D = 2, 3, 2049, 3, 4, 2
    offset = 7
    capacity = S + 19
    g = torch.Generator().manual_seed(2425)
    K_storage = torch.randn(B, H, offset + capacity, N, generator=g)
    V_storage = torch.randn(B, H, offset + capacity, D, generator=g)
    K = K_storage.narrow(2, offset, S)
    V = V_storage.narrow(2, offset, S)
    Q = torch.randn(B, H, Tq, N, generator=g)
    K_before, V_before = K.clone(), V.clone()

    assert K.stride() == (
        H * (offset + capacity) * N, (offset + capacity) * N, N, 1
    )
    assert V.stride() == (
        H * (offset + capacity) * D, (offset + capacity) * D, D, 1
    )
    ref = eager_decode_attn(Q, K, V)
    for impl in ("eager", "blocked", "online", "triton", "cuda"):
        got = bdh_attn_decode(Q, K, V, impl=impl, block_size=64)
        assert got.shape == (B, H, Tq, D)
        assert torch.allclose(got, ref, rtol=1e-4, atol=1e-5), (
            f"impl={impl} maxdiff={(got - ref).abs().max().item()}"
        )
        assert torch.equal(K, K_before)
        assert torch.equal(V, V_before)


def test_online_decode_t1_long_per_head_v_tiled_parity():
    """Long T=1 per-head decode takes the tiled path without mutating cache views."""
    B, H, S, N, D = 1, 2, 65537, 3, 2
    offset = 5
    capacity = S + 7
    g = torch.Generator().manual_seed(2324)
    K_storage = torch.randn(B, H, offset + capacity, N, generator=g)
    V_storage = torch.randn(B, H, offset + capacity, D, generator=g)
    K = K_storage.narrow(2, offset, S)
    V = V_storage.narrow(2, offset, S)
    Q = torch.randn(B, H, 1, N, generator=g)
    K_before, V_before = K.clone(), V.clone()

    ref = eager_decode_attn(Q, K, V)
    got = online_decode_attn(Q, K, V, block_size=64)

    assert torch.allclose(got, ref, rtol=1e-4, atol=1e-5), (
        f"maxdiff={(got - ref).abs().max().item()}"
    )
    assert torch.equal(K, K_before)
    assert torch.equal(V, V_before)

def test_online_decode_empty_past_returns_typed_zeros_for_shared_and_per_head_v():
    """Direct online decode keeps empty shared/per-head past outputs typed."""
    B, H, Tq, N, D = 2, 3, 2, 4, 5
    Q = torch.randn(B, H, Tq, N, dtype=torch.float64)
    K_past = torch.empty(B, H, 0, N, dtype=Q.dtype)
    expected = torch.zeros(B, H, Tq, D, dtype=Q.dtype)

    for V_past in (
        torch.empty(B, 1, 0, D, dtype=Q.dtype),
        torch.empty(B, H, 0, D, dtype=Q.dtype),
    ):
        for block_size in (1, 64):
            got = online_decode_attn(Q, K_past, V_past, block_size=block_size)
            assert got.shape == expected.shape
            assert got.dtype == expected.dtype
            assert torch.equal(got, expected), (
                f"v_heads={V_past.size(1)} block_size={block_size} returned {got}"
            )


def test_online_decode_preserves_strided_query_views():
    """Tiled online decode accepts offset query views without mutating them."""
    B, H, S, Tq, N, D = 2, 3, 257, 2, 4, 2
    g = torch.Generator().manual_seed(2526)
    Q_storage = torch.randn(B, H, Tq + 5, N, generator=g)
    Q = Q_storage.narrow(2, 3, Tq)
    K = torch.randn(B, H, S, N, generator=g)
    V = torch.randn(B, 1, S, D, generator=g)
    Q_before = Q.clone()

    assert Q.stride() == (H * (Tq + 5) * N, (Tq + 5) * N, N, 1)
    ref = eager_decode_attn(Q, K, V)
    for block_size in (1, 64, 128):
        got = online_decode_attn(Q, K, V, block_size=block_size)
        assert got.shape == (B, H, Tq, D)
        assert torch.allclose(got, ref, rtol=1e-4, atol=1e-5), (
            f"block_size={block_size} maxdiff={(got - ref).abs().max().item()}"
        )
        assert torch.equal(Q, Q_before)


def test_online_decode_tiled_shared_v_preserves_autograd_raw_score_contract():
    """Tiled shared-V decode keeps raw score×V gradients graph-safe."""
    B, H, S, Tq, N, D = 1, 2, 513, 2, 3, 2
    g = torch.Generator().manual_seed(2728)
    Q0 = torch.randn(B, H, Tq, N, generator=g)
    K0 = torch.randn(B, H, S, N, generator=g)
    V0 = torch.randn(B, 1, S, D, generator=g)
    weights = torch.randn(B, H, Tq, D, generator=g)

    Q_ref = Q0.detach().clone().requires_grad_()
    K_ref = K0.detach().clone().requires_grad_()
    V_ref = V0.detach().clone().requires_grad_()
    ref = eager_decode_attn(Q_ref, K_ref, V_ref)
    (ref * weights).sum().backward()
    ref_grads = (
        Q_ref.grad.detach().clone(),
        K_ref.grad.detach().clone(),
        V_ref.grad.detach().clone(),
    )

    Q = Q0.detach().clone().requires_grad_()
    K = K0.detach().clone().requires_grad_()
    V = V0.detach().clone().requires_grad_()
    got = online_decode_attn(Q, K, V, block_size=64)
    assert S * Tq > 1024  # force the shared-V tiled decode path
    (got * weights).sum().backward()

    assert torch.allclose(got, ref.detach(), rtol=1e-4, atol=1e-5), (
        f"maxdiff={(got - ref.detach()).abs().max().item()}"
    )
    for actual, expected in zip((Q.grad, K.grad, V.grad), ref_grads):
        assert actual is not None
        assert torch.allclose(actual, expected, rtol=1e-4, atol=1e-5), (
            f"grad maxdiff={(actual - expected).abs().max().item()}"
        )


def test_online_decode_tiled_per_head_v_preserves_autograd_raw_score_contract():
    """Tiled per-head-V decode keeps raw score×V gradients graph-safe."""
    B, H, S, Tq, N, D = 1, 2, 513, 2, 3, 2
    g = torch.Generator().manual_seed(2829)
    Q0 = torch.randn(B, H, Tq, N, generator=g)
    K0 = torch.randn(B, H, S, N, generator=g)
    V0 = torch.randn(B, H, S, D, generator=g)
    weights = torch.randn(B, H, Tq, D, generator=g)

    Q_ref = Q0.detach().clone().requires_grad_()
    K_ref = K0.detach().clone().requires_grad_()
    V_ref = V0.detach().clone().requires_grad_()
    ref = eager_decode_attn(Q_ref, K_ref, V_ref)
    (ref * weights).sum().backward()
    ref_grads = (
        Q_ref.grad.detach().clone(),
        K_ref.grad.detach().clone(),
        V_ref.grad.detach().clone(),
    )

    Q = Q0.detach().clone().requires_grad_()
    K = K0.detach().clone().requires_grad_()
    V = V0.detach().clone().requires_grad_()
    got = online_decode_attn(Q, K, V, block_size=64)
    assert S * Tq > 1024  # force the per-head-V tiled decode path
    (got * weights).sum().backward()

    assert torch.allclose(got, ref.detach(), rtol=1e-4, atol=1e-5), (
        f"maxdiff={(got - ref.detach()).abs().max().item()}"
    )
    for actual, expected in zip((Q.grad, K.grad, V.grad), ref_grads):
        assert actual is not None
        assert torch.allclose(actual, expected, rtol=1e-4, atol=1e-5), (
            f"grad maxdiff={(actual - expected).abs().max().item()}"
        )


def test_online_decode_tiled_shared_v_offset_views_preserve_autograd_contract():
    """Tiled decode keeps raw-score gradients through packed cache views."""
    B, H, S, Tq, N, D = 2, 3, 513, 2, 4, 2
    offset = 5
    capacity = S + 17
    g = torch.Generator().manual_seed(2930)
    Q0 = torch.randn(B, H, Tq, N, generator=g)
    K_storage0 = torch.randn(B, H, offset + capacity, N, generator=g)
    V_storage0 = torch.randn(B, 1, offset + capacity, D, generator=g)
    weights = torch.randn(B, H, Tq, D, generator=g)

    Q_ref = Q0.detach().clone().requires_grad_()
    K_storage_ref = K_storage0.detach().clone().requires_grad_()
    V_storage_ref = V_storage0.detach().clone().requires_grad_()
    K_ref = K_storage_ref.narrow(2, offset, S)
    V_ref = V_storage_ref.narrow(2, offset, S)
    ref = eager_decode_attn(Q_ref, K_ref, V_ref)
    (ref * weights).sum().backward()
    ref_grads = (
        Q_ref.grad.detach().clone(),
        K_storage_ref.grad.detach().clone(),
        V_storage_ref.grad.detach().clone(),
    )

    Q = Q0.detach().clone().requires_grad_()
    K_storage = K_storage0.detach().clone().requires_grad_()
    V_storage = V_storage0.detach().clone().requires_grad_()
    K = K_storage.narrow(2, offset, S)
    V = V_storage.narrow(2, offset, S)
    got = online_decode_attn(Q, K, V, block_size=64)
    assert S * Tq > 1024  # force the shared-V tiled decode path
    (got * weights).sum().backward()

    assert torch.allclose(got, ref.detach(), rtol=1e-4, atol=1e-5), (
        f"maxdiff={(got - ref.detach()).abs().max().item()}"
    )
    for actual, expected in zip(
        (Q.grad, K_storage.grad, V_storage.grad), ref_grads
    ):
        assert actual is not None
        assert torch.allclose(actual, expected, rtol=1e-4, atol=1e-5), (
            f"grad maxdiff={(actual - expected).abs().max().item()}"
        )


def test_online_decode_tiled_per_head_v_offset_views_preserve_autograd_contract():
    """Tiled per-head-V decode keeps raw-score gradients through packed views."""
    B, H, S, Tq, N, D = 2, 3, 513, 2, 4, 2
    offset = 7
    capacity = S + 19
    g = torch.Generator().manual_seed(3031)
    Q0 = torch.randn(B, H, Tq, N, generator=g)
    K_storage0 = torch.randn(B, H, offset + capacity, N, generator=g)
    V_storage0 = torch.randn(B, H, offset + capacity, D, generator=g)
    weights = torch.randn(B, H, Tq, D, generator=g)

    Q_ref = Q0.detach().clone().requires_grad_()
    K_storage_ref = K_storage0.detach().clone().requires_grad_()
    V_storage_ref = V_storage0.detach().clone().requires_grad_()
    K_ref = K_storage_ref.narrow(2, offset, S)
    V_ref = V_storage_ref.narrow(2, offset, S)
    ref = eager_decode_attn(Q_ref, K_ref, V_ref)
    (ref * weights).sum().backward()
    ref_grads = (
        Q_ref.grad.detach().clone(),
        K_storage_ref.grad.detach().clone(),
        V_storage_ref.grad.detach().clone(),
    )

    Q = Q0.detach().clone().requires_grad_()
    K_storage = K_storage0.detach().clone().requires_grad_()
    V_storage = V_storage0.detach().clone().requires_grad_()
    K = K_storage.narrow(2, offset, S)
    V = V_storage.narrow(2, offset, S)
    got = online_decode_attn(Q, K, V, block_size=64)
    assert S * Tq > 1024  # force the per-head-V tiled decode path
    (got * weights).sum().backward()

    assert torch.allclose(got, ref.detach(), rtol=1e-4, atol=1e-5), (
        f"maxdiff={(got - ref.detach()).abs().max().item()}"
    )
    for actual, expected in zip(
        (Q.grad, K_storage.grad, V_storage.grad), ref_grads
    ):
        assert actual is not None
        assert torch.allclose(actual, expected, rtol=1e-4, atol=1e-5), (
            f"grad maxdiff={(actual - expected).abs().max().item()}"
        )

def test_online_decode_tiled_zero_stride_shared_v_view_matches_shared_layout():
    """A head-expanded shared-V view keeps raw score×V decode semantics."""
    B, H, S, Tq, N, D = 2, 3, 513, 3, 4, 2
    offset = 5
    capacity = S + 17
    g = torch.Generator().manual_seed(3132)
    Q = torch.randn(B, H, Tq, N, generator=g)
    K_storage = torch.randn(B, H, offset + capacity, N, generator=g)
    V_storage = torch.randn(B, 1, offset + capacity, D, generator=g)
    K = K_storage.narrow(2, offset, S)
    V_shared = V_storage.narrow(2, offset, S)
    V_expanded = V_shared.expand(B, H, S, D)

    assert V_expanded.stride() == ((offset + capacity) * D, 0, D, 1)
    ref = eager_decode_attn(Q, K, V_shared)
    for block_size in (1, 64, 256):
        got_shared = online_decode_attn(Q, K, V_shared, block_size=block_size)
        got_expanded = online_decode_attn(Q, K, V_expanded, block_size=block_size)
        assert torch.allclose(got_shared, ref, rtol=1e-4, atol=1e-5), (
            f"shared block_size={block_size} maxdiff="
            f"{(got_shared - ref).abs().max().item()}"
        )
        assert torch.allclose(got_expanded, ref, rtol=1e-4, atol=1e-5), (
            f"expanded block_size={block_size} maxdiff="
            f"{(got_expanded - ref).abs().max().item()}"
        )
