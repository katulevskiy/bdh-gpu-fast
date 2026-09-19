"""Focused v19 contract coverage for dispatched score×V gradients.

This stays CPU-safe: every public dispatch alias is checked against the eager
strict-tril reference with distinct Q/K tensors and per-head V values.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kernels.attention_dispatch import bdh_attn  # noqa: E402


@pytest.mark.parametrize("impl", ["blocked", "online", "triton", "cuda"])
def test_dispatch_preserves_distinct_qk_per_head_v_gradients(impl):
    """Dispatch aliases preserve raw strict-tril Q/K/V gradients on CPU."""
    B, H, T, N, D = 2, 3, 19, 7, 5
    generator = torch.Generator(device="cpu").manual_seed(1919)
    Q = torch.randn(B, H, T, N, generator=generator, dtype=torch.float64)
    K = torch.randn(B, H, T, N, generator=generator, dtype=torch.float64)
    V = torch.randn(B, H, T, D, generator=generator, dtype=torch.float64)
    dO = torch.randn(B, H, T, D, generator=generator, dtype=torch.float64)

    def run(name):
        q = Q.detach().clone().requires_grad_(True)
        k = K.detach().clone().requires_grad_(True)
        v = V.detach().clone().requires_grad_(True)
        out = bdh_attn(q, k, v, impl=name)
        out.backward(dO)
        return out.detach(), tuple(x.grad.detach() for x in (q, k, v))

    ref, ref_grads = run("eager")
    expected = (Q @ K.transpose(-2, -1)).tril(diagonal=-1) @ V
    assert torch.equal(ref, expected)

    got, got_grads = run(impl)
    assert torch.allclose(got, ref, rtol=1e-10, atol=1e-10)
    for got_grad, ref_grad in zip(got_grads, ref_grads):
        assert torch.allclose(got_grad, ref_grad, rtol=1e-9, atol=1e-9)
