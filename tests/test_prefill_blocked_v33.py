"""CPU-safe contract for the explicit Triton cold fallback."""

import pytest
import torch

from kernels.attention import eager_tril_attn, triton_tril_attn


@pytest.mark.parametrize("value_heads", [1, 2])
def test_cpu_triton_cold_fallback_keeps_strict_raw_score_contract(value_heads):
    """CPU fallback preserves raw strict-tril math for both V layouts."""
    B, H, T, N, D = 2, 2, 257, 3, 2
    g = torch.Generator().manual_seed(401 + value_heads)
    Q = torch.randn(B, H, T, N, dtype=torch.float64, generator=g)
    K = torch.randn(B, H, T, N, dtype=torch.float64, generator=g)
    V = torch.randn(B, value_heads, T, D, dtype=torch.float64, generator=g)

    expected = torch.tril(Q @ K.transpose(-2, -1), diagonal=-1) @ V
    got = triton_tril_attn(Q, K, V)

    assert not Q.is_cuda
    assert torch.allclose(got, expected, rtol=1e-9, atol=1e-9)
    assert torch.count_nonzero(got[:, :, 0, :]) == 0


@pytest.mark.parametrize("value_heads", [1, 2])
def test_cpu_triton_cold_fallback_backward_matches_eager(value_heads):
    """CPU fallback preserves strict-tril gradients for both V layouts."""
    B, H, T, N, D = 2, 2, 257, 3, 2
    g = torch.Generator().manual_seed(411 + value_heads)
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
    got, got_grads = run(triton_tril_attn)

    assert not Q0.is_cuda
    assert torch.allclose(got, ref, rtol=1e-9, atol=1e-9)
    for got_grad, ref_grad in zip(got_grads, ref_grads):
        assert torch.allclose(got_grad, ref_grad, rtol=1e-9, atol=1e-9)
    assert torch.count_nonzero(got[:, :, 0, :]) == 0


@pytest.mark.parametrize("value_heads", [1, 2])
def test_cpu_triton_cold_fallback_preserves_padded_views(value_heads):
    """CPU fallback keeps parity through non-contiguous Q/K/V feature views."""
    B, H, T, N, D = 2, 2, 257, 3, 2
    g = torch.Generator().manual_seed(421 + value_heads)
    Q_storage = torch.randn(B, H, T, N + 1, dtype=torch.float64, generator=g)
    K_storage = torch.randn(B, H, T, N + 1, dtype=torch.float64, generator=g)
    V_storage = torch.randn(
        B, value_heads, T, D + 1, dtype=torch.float64, generator=g
    )
    Q = Q_storage[..., :N]
    K = K_storage[..., :N]
    V = V_storage[..., :D]

    assert not Q.is_contiguous()
    assert not K.is_contiguous()
    assert not V.is_contiguous()
    expected = eager_tril_attn(Q, K, V)
    got = triton_tril_attn(Q, K, V)

    assert not Q.is_cuda
    assert torch.allclose(got, expected, rtol=1e-9, atol=1e-9)
    assert torch.count_nonzero(got[:, :, 0, :]) == 0
