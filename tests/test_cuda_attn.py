"""Tests for strict tril(diagonal=-1) score×V kernel API.

CPU reference always runs. Native CUDA path is skipped when unavailable
(no extension build and/or no CUDA device) — that is expected on CPU boxes.
"""

from __future__ import annotations

import pytest
import torch

from kernels.cuda_attn import (
    ext_status,
    has_cuda_ext,
    has_cuda_kernel,
    tril_score_v,
    tril_score_v_ref,
)


def _naive(q, k, v):
    """Inline golden: identical math to bdh.Attention cold path."""
    scores = (q @ k.transpose(-2, -1)).tril(diagonal=-1)
    return scores @ v


@pytest.fixture(params=["cpu"])
def device(request):
    return torch.device(request.param)


def test_ext_status_prints():
    # Smoke: status string is non-empty either way.
    s = ext_status()
    assert isinstance(s, str) and len(s) > 0
    print("cuda_attn:", s)


def test_cpu_ref_matches_naive(device):
    torch.manual_seed(0)
    B, H, T, Dk, Dv = 2, 4, 8, 16, 32
    q = torch.randn(B, H, T, Dk, device=device)
    k = torch.randn(B, H, T, Dk, device=device)
    v = torch.randn(B, H, T, Dv, device=device)
    out = tril_score_v_ref(q, k, v)
    gold = _naive(q, k, v)
    assert out.shape == (B, H, T, Dv)
    assert torch.allclose(out, gold, rtol=1e-5, atol=1e-5)


def test_cpu_ref_broadcast_v_heads(device):
    """Match bdh.Attention: V is (B, 1, T, D) and broadcasts over heads."""
    torch.manual_seed(1)
    B, H, T, Dk, Dv = 2, 4, 7, 8, 16
    q = torch.randn(B, H, T, Dk, device=device)
    k = q.clone()
    v = torch.randn(B, 1, T, Dv, device=device)
    out = tril_score_v_ref(q, k, v)
    gold = _naive(q, k, v)
    assert out.shape == (B, H, T, Dv)
    assert torch.allclose(out, gold, rtol=1e-5, atol=1e-5)


def test_diagonal_excluded_position0_zero(device):
    torch.manual_seed(2)
    B, H, T, Dk, Dv = 1, 2, 5, 8, 4
    q = torch.randn(B, H, T, Dk, device=device)
    k = torch.randn(B, H, T, Dk, device=device)
    v = torch.randn(B, H, T, Dv, device=device)
    out = tril_score_v_ref(q, k, v)
    assert torch.allclose(out[:, :, 0, :], torch.zeros_like(out[:, :, 0, :]))


def test_tril_score_v_dispatch_cpu(device):
    """Public dispatcher must match ref on CPU (ext or fallback)."""
    torch.manual_seed(3)
    q = torch.randn(1, 2, 6, 8, device=device)
    k = torch.randn(1, 2, 6, 8, device=device)
    v = torch.randn(1, 2, 6, 4, device=device)
    assert torch.allclose(tril_score_v(q, k, v), tril_score_v_ref(q, k, v), rtol=1e-5, atol=1e-5)


def test_t1_all_zeros(device):
    q = torch.randn(2, 3, 1, 8, device=device)
    k = torch.randn(2, 3, 1, 8, device=device)
    v = torch.randn(2, 3, 1, 5, device=device)
    out = tril_score_v_ref(q, k, v)
    assert torch.allclose(out, torch.zeros_like(out))


@pytest.mark.skipif(not has_cuda_ext(), reason="bdh_cuda_ext not built (pip install -e .)")
def test_native_cpu_matches_ref():
    torch.manual_seed(4)
    q = torch.randn(2, 2, 5, 8)
    k = torch.randn(2, 2, 5, 8)
    v = torch.randn(2, 2, 5, 4)
    import bdh_cuda_ext

    out = bdh_cuda_ext.tril_score_v_cpu(q, k, v)
    assert torch.allclose(out, tril_score_v_ref(q, k, v), rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(
    not (has_cuda_kernel() and torch.cuda.is_available()),
    reason="CUDA kernel not built or no CUDA device",
)
def test_cuda_kernel_matches_ref():
    torch.manual_seed(5)
    device = torch.device("cuda")
    q = torch.randn(2, 4, 16, 32, device=device)
    k = torch.randn(2, 4, 16, 32, device=device)
    v = torch.randn(2, 4, 16, 64, device=device)
    out = tril_score_v(q, k, v)
    gold = tril_score_v_ref(q, k, v)
    assert torch.allclose(out, gold, rtol=1e-3, atol=1e-3)


@pytest.mark.skipif(
    not (has_cuda_kernel() and torch.cuda.is_available()),
    reason="CUDA kernel not built or no CUDA device",
)
def test_cuda_broadcast_v():
    torch.manual_seed(6)
    device = torch.device("cuda")
    q = torch.randn(1, 4, 12, 16, device=device)
    k = q.clone()
    v = torch.randn(1, 1, 12, 32, device=device)
    out = tril_score_v(q, k, v)
    gold = tril_score_v_ref(q, k, v)
    assert out.shape == gold.shape
    assert torch.allclose(out, gold, rtol=1e-3, atol=1e-3)
