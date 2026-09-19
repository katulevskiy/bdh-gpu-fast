"""CPU-safe contract for the explicit Triton cold fallback."""

import pytest
import torch

from kernels.attention import triton_tril_attn


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
