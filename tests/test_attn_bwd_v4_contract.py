"""CPU-safe edge contract for strict-tril analytic attention backward."""

from __future__ import annotations

import pytest
import torch

from kernels.attention_bwd import strict_tril_attn


@pytest.mark.parametrize("impl", ["eager", "blocked", "online", "triton", "cuda"])
@pytest.mark.parametrize("v_heads", [1, 3])
def test_t1_excludes_self_attention_in_output_and_backward(impl, v_heads):
    """T=1 has no past keys, so output and every input gradient are zero."""
    generator = torch.Generator().manual_seed(2026)
    Q = torch.randn(
        2, 3, 1, 4, generator=generator, dtype=torch.float64, requires_grad=True
    )
    K = torch.randn(
        2, 3, 1, 4, generator=generator, dtype=torch.float64, requires_grad=True
    )
    V = torch.randn(
        2, v_heads, 1, 5, generator=generator, dtype=torch.float64, requires_grad=True
    )

    out = strict_tril_attn(Q, K, V, impl=impl, use_fn=True)
    assert torch.equal(out, torch.zeros_like(out))

    out.sum().backward()
    assert Q.grad is not None and K.grad is not None and V.grad is not None
    assert torch.equal(Q.grad, torch.zeros_like(Q))
    assert torch.equal(K.grad, torch.zeros_like(K))
    assert torch.equal(V.grad, torch.zeros_like(V))
