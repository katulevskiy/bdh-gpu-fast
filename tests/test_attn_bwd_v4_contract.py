"""CPU-safe edge contract for strict-tril analytic attention backward."""

from __future__ import annotations

import pytest
import torch

from kernels.attention import eager_tril_attn
from kernels.attention_bwd import strict_tril_attn


@pytest.mark.parametrize("impl", ["eager", "blocked", "online", "triton", "cuda"])
def test_raw_score_strict_tril_forward_and_backward_contract(impl):
    """Strict-tril attention is raw score@V, with no scale or softmax."""
    Q = torch.tensor(
        [[[[2.0, 0.0], [1.0, 3.0], [4.0, 5.0]]]], requires_grad=True
    )
    K = torch.tensor(
        [[[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]]], requires_grad=True
    )
    V = torch.tensor(
        [[[[2.0], [3.0], [100.0]]]], requires_grad=True
    )

    out = strict_tril_attn(Q, K, V, impl=impl, use_fn=True)
    expected = torch.tensor([[[[0.0], [14.0], [124.0]]]])
    assert torch.equal(out, expected)

    out.sum().backward()
    assert torch.equal(
        Q.grad, torch.tensor([[[[0.0, 0.0], [2.0, 4.0], [11.0, 16.0]]]])
    )
    assert torch.equal(
        K.grad, torch.tensor([[[[10.0, 16.0], [12.0, 15.0], [0.0, 0.0]]]])
    )
    assert torch.equal(V.grad, torch.tensor([[[[21.0], [32.0], [0.0]]]]))


@pytest.mark.parametrize("impl", ["eager", "blocked", "online", "triton", "cuda"])
def test_zero_upstream_gradient_produces_zero_input_gradients(impl):
    """A zero upstream gradient cannot create strict-tril input gradients."""
    generator = torch.Generator().manual_seed(2042)
    Q = torch.randn(
        2, 2, 4, 3, generator=generator, dtype=torch.float64, requires_grad=True
    )
    K = torch.randn(
        2, 2, 4, 3, generator=generator, dtype=torch.float64, requires_grad=True
    )
    V = torch.randn(
        2, 1, 4, 5, generator=generator, dtype=torch.float64, requires_grad=True
    )

    out = strict_tril_attn(Q, K, V, impl=impl, use_fn=True)
    out.backward(torch.zeros_like(out))

    assert Q.grad is not None and K.grad is not None and V.grad is not None
    assert torch.equal(Q.grad, torch.zeros_like(Q))
    assert torch.equal(K.grad, torch.zeros_like(K))
    assert torch.equal(V.grad, torch.zeros_like(V))


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


@pytest.mark.parametrize("impl", ["eager", "blocked", "online", "triton", "cuda"])
@pytest.mark.parametrize("v_heads", [1, 3])
def test_empty_sequence_preserves_zero_shape_and_backward_contract(impl, v_heads):
    """An empty strict-tril sequence stays empty with zero-shaped gradients."""
    Q = torch.empty(2, 3, 0, 4, dtype=torch.float64, requires_grad=True)
    K = torch.empty(2, 3, 0, 4, dtype=torch.float64, requires_grad=True)
    V = torch.empty(2, v_heads, 0, 5, dtype=torch.float64, requires_grad=True)

    out = strict_tril_attn(Q, K, V, impl=impl, use_fn=True)
    assert out.shape == (2, 3, 0, 5)
    assert out.numel() == 0

    out.sum().backward()
    assert Q.grad is not None and K.grad is not None and V.grad is not None
    assert Q.grad.shape == Q.shape
    assert K.grad.shape == K.shape
    assert V.grad.shape == V.shape
    assert Q.grad.numel() == K.grad.numel() == V.grad.numel() == 0


@pytest.mark.parametrize("impl", ["eager", "blocked", "online", "triton", "cuda"])
@pytest.mark.parametrize("v_heads", [1, 3])
@pytest.mark.parametrize("query", [0, 3, 4])
def test_single_query_backward_only_reaches_strict_past(impl, v_heads, query):
    """A loss at query q can reach Q[q], but only K/V positions j < q."""
    generator = torch.Generator().manual_seed(2027)
    Q = torch.randn(
        1, 3, 5, 4, generator=generator, dtype=torch.float64, requires_grad=True
    )
    K = torch.randn(
        1, 3, 5, 4, generator=generator, dtype=torch.float64, requires_grad=True
    )
    V = torch.randn(
        1, v_heads, 5, 6, generator=generator, dtype=torch.float64, requires_grad=True
    )

    out = strict_tril_attn(Q, K, V, impl=impl, use_fn=True)
    dO = torch.zeros_like(out)
    dO[:, :, query, :] = torch.randn(
        1, 3, 6, generator=generator, dtype=torch.float64
    )
    out.backward(dO)

    # With no strict-past keys, the selected output and query gradient are zero.
    if query == 0:
        assert torch.equal(out[:, :, query, :], torch.zeros_like(out[:, :, query, :]))
        assert torch.equal(Q.grad[:, :, query, :], torch.zeros_like(Q.grad[:, :, query, :]))
    # A single output row depends on its matching Q row only.
    assert torch.equal(Q.grad[:, :, :query, :], torch.zeros_like(Q.grad[:, :, :query, :]))
    assert torch.equal(Q.grad[:, :, query + 1 :, :], torch.zeros_like(Q.grad[:, :, query + 1 :, :]))
    # Diagonal and future keys/values are excluded by tril(diagonal=-1).
    assert torch.equal(K.grad[:, :, query:, :], torch.zeros_like(K.grad[:, :, query:, :]))
    assert torch.equal(V.grad[:, :, query:, :], torch.zeros_like(V.grad[:, :, query:, :]))


@pytest.mark.parametrize("impl", ["eager", "blocked", "online", "triton", "cuda"])
def test_single_query_shared_v_backward_reduces_head_grads(impl):
    """Shared V must reduce only the strict-past per-head gradients."""
    generator = torch.Generator().manual_seed(2028)
    query = 3
    Q = torch.randn(
        2, 3, 5, 4, generator=generator, dtype=torch.float64, requires_grad=True
    )
    K = torch.randn(
        2, 3, 5, 4, generator=generator, dtype=torch.float64, requires_grad=True
    )
    V = torch.randn(
        2, 1, 5, 6, generator=generator, dtype=torch.float64, requires_grad=True
    )
    dO = torch.zeros(2, 3, 5, 6, dtype=torch.float64)
    dO[:, :, query, :] = torch.randn(
        2, 3, 6, generator=generator, dtype=torch.float64
    )

    out = strict_tril_attn(Q, K, V, impl=impl, use_fn=True)
    out.backward(dO)

    Q_ref = Q.detach().clone().requires_grad_(True)
    K_ref = K.detach().clone().requires_grad_(True)
    V_full = V.detach().expand(2, 3, 5, 6).clone().requires_grad_(True)
    eager_tril_attn(Q_ref, K_ref, V_full).backward(dO)

    assert torch.allclose(Q.grad, Q_ref.grad, rtol=1e-8, atol=1e-8)
    assert torch.allclose(K.grad, K_ref.grad, rtol=1e-8, atol=1e-8)
    assert torch.allclose(
        V.grad, V_full.grad.sum(dim=1, keepdim=True), rtol=1e-8, atol=1e-8
    )
    assert torch.equal(
        V.grad[:, :, query:, :], torch.zeros_like(V.grad[:, :, query:, :])
    )


@pytest.mark.parametrize("impl", ["eager", "blocked", "online", "triton", "cuda"])
def test_self_attn_aliases_preserve_raw_strict_tril_backward(impl):
    """Self-attention aliases must keep duplicate-Q gradients exact."""
    generator = torch.Generator().manual_seed(2029)
    Q = torch.randn(1, 3, 5, 4, generator=generator, dtype=torch.float64)
    V = torch.randn(1, 1, 5, 6, generator=generator, dtype=torch.float64)
    dO = torch.randn(1, 3, 5, 6, generator=generator, dtype=torch.float64)

    Q_impl = Q.clone().requires_grad_(True)
    V_impl = V.clone().requires_grad_(True)
    out = strict_tril_attn(Q_impl, Q_impl, V_impl, impl=impl, use_fn=True)

    Q_ref = Q.clone().requires_grad_(True)
    V_ref = V.clone().requires_grad_(True)
    ref = eager_tril_attn(Q_ref, Q_ref, V_ref)

    assert torch.allclose(out, ref, rtol=1e-12, atol=1e-12)
    out.backward(dO)
    ref.backward(dO)
    assert torch.allclose(Q_impl.grad, Q_ref.grad, rtol=1e-10, atol=1e-10)
    assert torch.allclose(V_impl.grad, V_ref.grad, rtol=1e-10, atol=1e-10)
    assert torch.equal(out[:, :, 0, :], torch.zeros_like(out[:, :, 0, :]))


@pytest.mark.parametrize("impl", ["eager", "blocked", "online", "triton", "cuda"])
def test_noncontiguous_self_attn_alias_preserves_backward_contract(impl):
    """Aliased noncontiguous Q/K views preserve duplicate-path gradients."""
    generator = torch.Generator().manual_seed(2044)
    Q_storage = torch.randn(
        2, 3, 5, 8, generator=generator, dtype=torch.float64, requires_grad=True
    )
    Q = Q_storage[..., ::2]
    V = torch.randn(
        2, 1, 5, 6, generator=generator, dtype=torch.float64, requires_grad=True
    )
    dO = torch.randn(2, 3, 5, 6, generator=generator, dtype=torch.float64)

    out = strict_tril_attn(Q, Q, V, impl=impl, use_fn=True)
    Q_ref = Q.detach().clone().requires_grad_(True)
    V_ref = V.detach().clone().requires_grad_(True)
    ref = eager_tril_attn(Q_ref, Q_ref, V_ref)

    assert not Q.is_contiguous()
    assert torch.allclose(out, ref, rtol=1e-12, atol=1e-12)
    out.backward(dO)
    ref.backward(dO)

    assert torch.allclose(
        Q_storage.grad[..., ::2], Q_ref.grad, rtol=1e-12, atol=1e-12
    )
    assert torch.equal(
        Q_storage.grad[..., 1::2], torch.zeros_like(Q_storage.grad[..., 1::2])
    )
    assert torch.allclose(V.grad, V_ref.grad, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("impl", ["eager", "blocked", "online", "triton", "cuda"])
def test_self_attn_single_query_accumulates_only_valid_qk_paths(impl):
    """A selected output row exposes duplicate-Q strict-tril gradient routing."""
    Q = torch.tensor(
        [[[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]]], requires_grad=True
    )
    V = torch.tensor([[[[2.0], [3.0], [100.0]]]], requires_grad=True)
    dO = torch.zeros(1, 1, 3, 1)
    dO[:, :, 2, :] = 1.0

    out = strict_tril_attn(Q, Q, V, impl=impl, use_fn=True)
    assert torch.equal(out, torch.tensor([[[[0.0], [22.0], [151.0]]]]))
    out.backward(dO)

    # Q[2] is query-only; Q[0:2] are key-only; Q[2] is not a diagonal key.
    assert torch.equal(
        Q.grad, torch.tensor([[[[10.0, 12.0], [15.0, 18.0], [11.0, 16.0]]]])
    )
    assert torch.equal(V.grad, torch.tensor([[[[17.0], [39.0], [0.0]]]]))


@pytest.mark.parametrize("impl", ["eager", "blocked", "online", "triton", "cuda"])
def test_zero_stride_self_attn_q_view_reduces_duplicate_base_gradient(impl):
    """Self-attention must reduce duplicate Q/K paths through a shared view."""
    generator = torch.Generator().manual_seed(2039)
    Q_base = torch.randn(
        1, 1, 5, 4, generator=generator, dtype=torch.float64, requires_grad=True
    )
    Q = Q_base.expand(2, 3, 5, 4)
    V = torch.randn(
        2, 3, 5, 6, generator=generator, dtype=torch.float64, requires_grad=True
    )
    dO = torch.randn(2, 3, 5, 6, generator=generator, dtype=torch.float64)

    out = strict_tril_attn(Q, Q, V, impl=impl, use_fn=True)
    Q_ref = Q.detach().clone().requires_grad_(True)
    V_ref = V.detach().clone().requires_grad_(True)
    ref = eager_tril_attn(Q_ref, Q_ref, V_ref)

    assert Q.stride(0) == 0 and Q.stride(1) == 0
    assert torch.allclose(out, ref, rtol=1e-12, atol=1e-12)
    out.backward(dO)
    ref.backward(dO)
    assert torch.allclose(
        Q_base.grad, Q_ref.grad.sum(dim=(0, 1), keepdim=True), rtol=1e-12, atol=1e-12
    )
    assert torch.allclose(V.grad, V_ref.grad, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("impl", ["blocked", "online", "triton", "cuda"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_half_dtypes_preserve_strict_tril_backward_contract(impl, dtype):
    """Tiled analytic backward widens safely, then restores input grad dtypes."""
    generator = torch.Generator().manual_seed(2030)
    Q = torch.randn(1, 2, 5, 3, generator=generator, dtype=dtype, requires_grad=True)
    K = torch.randn(1, 2, 5, 3, generator=generator, dtype=dtype, requires_grad=True)
    V = torch.randn(1, 2, 5, 4, generator=generator, dtype=dtype, requires_grad=True)

    out = strict_tril_attn(Q, K, V, impl=impl, use_fn=True)
    Q_ref = Q.detach().clone().requires_grad_(True)
    K_ref = K.detach().clone().requires_grad_(True)
    V_ref = V.detach().clone().requires_grad_(True)
    ref = eager_tril_attn(Q_ref, K_ref, V_ref)
    dO = torch.randn_like(out)
    dO_ref = dO.to(dtype=ref.dtype)

    assert torch.allclose(out.float(), ref.float(), rtol=5e-2, atol=5e-2)
    out.backward(dO)
    ref.backward(dO_ref)
    assert Q.grad.dtype == dtype and K.grad.dtype == dtype and V.grad.dtype == dtype
    assert torch.allclose(Q.grad.float(), Q_ref.grad.float(), rtol=5e-2, atol=5e-2)
    assert torch.allclose(K.grad.float(), K_ref.grad.float(), rtol=5e-2, atol=5e-2)
    assert torch.allclose(V.grad.float(), V_ref.grad.float(), rtol=5e-2, atol=5e-2)


@pytest.mark.parametrize("impl", ["eager", "blocked", "online", "triton", "cuda"])
def test_noncontiguous_inputs_preserve_strict_tril_backward_contract(impl):
    """Analytic backward must preserve the contract for strided input views."""
    generator = torch.Generator().manual_seed(2032)

    def _strided(shape):
        padded = torch.randn(
            *shape[:-1],
            shape[-1] * 2,
            generator=generator,
            dtype=torch.float64,
        )
        return padded[..., ::2].detach().requires_grad_(True)

    Q = _strided((2, 3, 4, 3))
    K = _strided((2, 3, 4, 3))
    V = _strided((2, 1, 4, 2))
    dO = torch.randn(2, 3, 4, 2, generator=generator, dtype=torch.float64)

    out = strict_tril_attn(Q, K, V, impl=impl, use_fn=True)
    Q_ref = Q.detach().contiguous().requires_grad_(True)
    K_ref = K.detach().contiguous().requires_grad_(True)
    V_ref = V.detach().contiguous().requires_grad_(True)
    ref = eager_tril_attn(Q_ref, K_ref, V_ref)

    assert not Q.is_contiguous() and not K.is_contiguous() and not V.is_contiguous()
    assert torch.allclose(out, ref, rtol=1e-12, atol=1e-12)
    out.backward(dO)
    ref.backward(dO)
    assert torch.allclose(Q.grad, Q_ref.grad, rtol=1e-12, atol=1e-12)
    assert torch.allclose(K.grad, K_ref.grad, rtol=1e-12, atol=1e-12)
    assert torch.allclose(V.grad, V_ref.grad, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("impl", ["eager", "blocked", "online", "triton", "cuda"])
def test_zero_stride_v_input_reduces_base_gradient(impl):
    """A zero-stride per-head V view must reduce its gradient to the base."""
    generator = torch.Generator().manual_seed(2034)
    Q = torch.randn(
        2, 3, 5, 4, generator=generator, dtype=torch.float64, requires_grad=True
    )
    K = torch.randn(
        2, 3, 5, 4, generator=generator, dtype=torch.float64, requires_grad=True
    )
    V_base = torch.randn(
        2, 1, 5, 6, generator=generator, dtype=torch.float64, requires_grad=True
    )
    V = V_base.expand(2, 3, 5, 6)
    dO = torch.randn(2, 3, 5, 6, generator=generator, dtype=torch.float64)

    out = strict_tril_attn(Q, K, V, impl=impl, use_fn=True)
    Q_ref = Q.detach().clone().requires_grad_(True)
    K_ref = K.detach().clone().requires_grad_(True)
    V_ref = V.detach().clone().requires_grad_(True)
    ref = eager_tril_attn(Q_ref, K_ref, V_ref)

    assert V.stride(1) == 0
    assert torch.allclose(out, ref, rtol=1e-12, atol=1e-12)
    out.backward(dO)
    ref.backward(dO)
    assert torch.allclose(Q.grad, Q_ref.grad, rtol=1e-12, atol=1e-12)
    assert torch.allclose(K.grad, K_ref.grad, rtol=1e-12, atol=1e-12)
    assert torch.allclose(
        V_base.grad, V_ref.grad.sum(dim=1, keepdim=True), rtol=1e-12, atol=1e-12
    )


@pytest.mark.parametrize("impl", ["eager", "blocked", "online", "triton", "cuda"])
def test_zero_stride_v_batch_head_view_reduces_base_gradient(impl):
    """V views shared across batch and heads must reduce to the base."""
    generator = torch.Generator().manual_seed(2037)
    Q = torch.randn(
        2, 3, 5, 4, generator=generator, dtype=torch.float64, requires_grad=True
    )
    K = torch.randn(
        2, 3, 5, 4, generator=generator, dtype=torch.float64, requires_grad=True
    )
    V_base = torch.randn(
        1, 1, 5, 6, generator=generator, dtype=torch.float64, requires_grad=True
    )
    V = V_base.expand(2, 3, 5, 6)
    dO = torch.randn(2, 3, 5, 6, generator=generator, dtype=torch.float64)

    out = strict_tril_attn(Q, K, V, impl=impl, use_fn=True)
    Q_ref = Q.detach().clone().requires_grad_(True)
    K_ref = K.detach().clone().requires_grad_(True)
    V_ref = V.detach().clone().requires_grad_(True)
    ref = eager_tril_attn(Q_ref, K_ref, V_ref)

    assert V.stride(0) == 0 and V.stride(1) == 0
    assert torch.allclose(out, ref, rtol=1e-12, atol=1e-12)
    out.backward(dO)
    ref.backward(dO)
    assert torch.allclose(Q.grad, Q_ref.grad, rtol=1e-12, atol=1e-12)
    assert torch.allclose(K.grad, K_ref.grad, rtol=1e-12, atol=1e-12)
    assert torch.allclose(
        V_base.grad, V_ref.grad.sum(dim=(0, 1), keepdim=True), rtol=1e-12, atol=1e-12
    )


@pytest.mark.parametrize("impl", ["eager", "blocked", "online", "triton", "cuda"])
def test_zero_stride_qk_inputs_reduce_base_gradients(impl):
    """Zero-stride Q/K head views must reduce their gradients to the bases."""
    generator = torch.Generator().manual_seed(2035)
    Q_base = torch.randn(
        2, 1, 5, 4, generator=generator, dtype=torch.float64, requires_grad=True
    )
    K_base = torch.randn(
        2, 1, 5, 4, generator=generator, dtype=torch.float64, requires_grad=True
    )
    Q = Q_base.expand(2, 3, 5, 4)
    K = K_base.expand(2, 3, 5, 4)
    V = torch.randn(
        2, 3, 5, 6, generator=generator, dtype=torch.float64, requires_grad=True
    )
    dO = torch.randn(2, 3, 5, 6, generator=generator, dtype=torch.float64)

    out = strict_tril_attn(Q, K, V, impl=impl, use_fn=True)
    Q_ref = Q.detach().clone().requires_grad_(True)
    K_ref = K.detach().clone().requires_grad_(True)
    V_ref = V.detach().clone().requires_grad_(True)
    ref = eager_tril_attn(Q_ref, K_ref, V_ref)

    assert Q.stride(1) == 0 and K.stride(1) == 0
    assert torch.allclose(out, ref, rtol=1e-12, atol=1e-12)
    out.backward(dO)
    ref.backward(dO)

    assert torch.allclose(
        Q_base.grad, Q_ref.grad.sum(dim=1, keepdim=True), rtol=1e-12, atol=1e-12
    )
    assert torch.allclose(
        K_base.grad, K_ref.grad.sum(dim=1, keepdim=True), rtol=1e-12, atol=1e-12
    )
    assert torch.allclose(V.grad, V_ref.grad, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("impl", ["eager", "blocked", "online", "triton", "cuda"])
def test_zero_stride_qk_batch_head_views_reduce_base_gradients(impl):
    """Q/K views shared across batch and heads must reduce both axes."""
    generator = torch.Generator().manual_seed(2036)
    Q_base = torch.randn(
        1, 1, 5, 4, generator=generator, dtype=torch.float64, requires_grad=True
    )
    K_base = torch.randn(
        1, 1, 5, 4, generator=generator, dtype=torch.float64, requires_grad=True
    )
    Q = Q_base.expand(2, 3, 5, 4)
    K = K_base.expand(2, 3, 5, 4)
    V = torch.randn(
        2, 3, 5, 6, generator=generator, dtype=torch.float64, requires_grad=True
    )
    dO = torch.randn(2, 3, 5, 6, generator=generator, dtype=torch.float64)

    out = strict_tril_attn(Q, K, V, impl=impl, use_fn=True)
    Q_ref = Q.detach().clone().requires_grad_(True)
    K_ref = K.detach().clone().requires_grad_(True)
    V_ref = V.detach().clone().requires_grad_(True)
    ref = eager_tril_attn(Q_ref, K_ref, V_ref)

    assert Q.stride(0) == 0 and Q.stride(1) == 0
    assert K.stride(0) == 0 and K.stride(1) == 0
    assert torch.allclose(out, ref, rtol=1e-12, atol=1e-12)
    out.backward(dO)
    ref.backward(dO)

    assert torch.allclose(
        Q_base.grad, Q_ref.grad.sum(dim=(0, 1), keepdim=True), rtol=1e-12, atol=1e-12
    )
    assert torch.allclose(
        K_base.grad, K_ref.grad.sum(dim=(0, 1), keepdim=True), rtol=1e-12, atol=1e-12
    )
    assert torch.allclose(V.grad, V_ref.grad, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("impl", ["eager", "blocked", "online", "triton", "cuda"])
def test_noncontiguous_upstream_gradient_preserves_backward_contract(impl):
    """Analytic backward must accept a strided upstream gradient."""
    generator = torch.Generator().manual_seed(2033)
    Q = torch.randn(
        2, 3, 5, 4, generator=generator, dtype=torch.float64, requires_grad=True
    )
    K = torch.randn(
        2, 3, 5, 4, generator=generator, dtype=torch.float64, requires_grad=True
    )
    V = torch.randn(
        2, 1, 5, 6, generator=generator, dtype=torch.float64, requires_grad=True
    )
    dO_padded = torch.randn(
        2, 3, 5, 12, generator=generator, dtype=torch.float64
    )
    dO = dO_padded[..., ::2]

    out = strict_tril_attn(Q, K, V, impl=impl, use_fn=True)
    Q_ref = Q.detach().clone().requires_grad_(True)
    K_ref = K.detach().clone().requires_grad_(True)
    V_ref = V.detach().clone().requires_grad_(True)
    ref = eager_tril_attn(Q_ref, K_ref, V_ref)

    assert not dO.is_contiguous()
    assert torch.allclose(out, ref, rtol=1e-12, atol=1e-12)
    out.backward(dO)
    ref.backward(dO.contiguous())
    assert torch.allclose(Q.grad, Q_ref.grad, rtol=1e-12, atol=1e-12)
    assert torch.allclose(K.grad, K_ref.grad, rtol=1e-12, atol=1e-12)
    assert torch.allclose(V.grad, V_ref.grad, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("impl", ["eager", "blocked", "online", "triton", "cuda"])
def test_zero_stride_upstream_gradient_preserves_backward_contract(impl):
    """Analytic backward must accept a head-broadcast upstream gradient."""
    generator = torch.Generator().manual_seed(2038)
    Q = torch.randn(
        2, 3, 5, 4, generator=generator, dtype=torch.float64, requires_grad=True
    )
    K = torch.randn(
        2, 3, 5, 4, generator=generator, dtype=torch.float64, requires_grad=True
    )
    V = torch.randn(
        2, 1, 5, 6, generator=generator, dtype=torch.float64, requires_grad=True
    )
    dO_base = torch.randn(2, 1, 5, 6, generator=generator, dtype=torch.float64)
    dO = dO_base.expand(2, 3, 5, 6)

    out = strict_tril_attn(Q, K, V, impl=impl, use_fn=True)
    Q_ref = Q.detach().clone().requires_grad_(True)
    K_ref = K.detach().clone().requires_grad_(True)
    V_ref = V.detach().clone().requires_grad_(True)
    ref = eager_tril_attn(Q_ref, K_ref, V_ref)

    assert dO.stride(1) == 0
    assert torch.allclose(out, ref, rtol=1e-12, atol=1e-12)
    out.backward(dO)
    ref.backward(dO.contiguous())
    assert torch.allclose(Q.grad, Q_ref.grad, rtol=1e-12, atol=1e-12)
    assert torch.allclose(K.grad, K_ref.grad, rtol=1e-12, atol=1e-12)
    assert torch.allclose(V.grad, V_ref.grad, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("impl", ["eager", "blocked", "online", "triton", "cuda"])
def test_zero_stride_batch_head_upstream_gradient_preserves_backward_contract(impl):
    """Analytic backward must accept upstream gradients broadcast on batch and head."""
    generator = torch.Generator().manual_seed(2040)
    Q = torch.randn(
        2, 3, 5, 4, generator=generator, dtype=torch.float64, requires_grad=True
    )
    K = torch.randn(
        2, 3, 5, 4, generator=generator, dtype=torch.float64, requires_grad=True
    )
    V = torch.randn(
        2, 1, 5, 6, generator=generator, dtype=torch.float64, requires_grad=True
    )
    dO_base = torch.randn(1, 1, 5, 6, generator=generator, dtype=torch.float64)
    dO = dO_base.expand(2, 3, 5, 6)

    out = strict_tril_attn(Q, K, V, impl=impl, use_fn=True)
    Q_ref = Q.detach().clone().requires_grad_(True)
    K_ref = K.detach().clone().requires_grad_(True)
    V_ref = V.detach().clone().requires_grad_(True)
    ref = eager_tril_attn(Q_ref, K_ref, V_ref)

    assert dO.stride(0) == 0 and dO.stride(1) == 0
    assert torch.allclose(out, ref, rtol=1e-12, atol=1e-12)
    out.backward(dO)
    ref.backward(dO.contiguous())
    assert torch.allclose(Q.grad, Q_ref.grad, rtol=1e-12, atol=1e-12)
    assert torch.allclose(K.grad, K_ref.grad, rtol=1e-12, atol=1e-12)
    assert torch.allclose(V.grad, V_ref.grad, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("impl", ["eager", "blocked", "online", "triton", "cuda"])
def test_zero_stride_sequence_upstream_gradient_preserves_backward_contract(impl):
    """Analytic backward must accept an upstream gradient broadcast over queries."""
    generator = torch.Generator().manual_seed(2043)
    Q = torch.randn(
        2, 3, 5, 4, generator=generator, dtype=torch.float64, requires_grad=True
    )
    K = torch.randn(
        2, 3, 5, 4, generator=generator, dtype=torch.float64, requires_grad=True
    )
    V = torch.randn(
        2, 1, 5, 6, generator=generator, dtype=torch.float64, requires_grad=True
    )
    dO_base = torch.randn(2, 3, 1, 6, generator=generator, dtype=torch.float64)
    dO = dO_base.expand(2, 3, 5, 6)

    out = strict_tril_attn(Q, K, V, impl=impl, use_fn=True)
    Q_ref = Q.detach().clone().requires_grad_(True)
    K_ref = K.detach().clone().requires_grad_(True)
    V_ref = V.detach().clone().requires_grad_(True)
    ref = eager_tril_attn(Q_ref, K_ref, V_ref)

    assert dO.stride(2) == 0
    assert torch.allclose(out, ref, rtol=1e-12, atol=1e-12)
    out.backward(dO)
    ref.backward(dO.contiguous())
    assert torch.allclose(Q.grad, Q_ref.grad, rtol=1e-12, atol=1e-12)
    assert torch.allclose(K.grad, K_ref.grad, rtol=1e-12, atol=1e-12)
    assert torch.allclose(V.grad, V_ref.grad, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("impl", ["eager", "blocked", "online", "triton", "cuda"])
@pytest.mark.parametrize(
    ("sequence_length", "query"), [(65, 63), (65, 64), (128, 127)]
)
def test_backward_preserves_strict_past_at_default_tile_boundary(
    impl, sequence_length, query
):
    """Rows across the first and second 64-row tiles exclude self/future."""
    generator = torch.Generator().manual_seed(2031)
    Q = torch.randn(
        1,
        2,
        sequence_length,
        3,
        generator=generator,
        dtype=torch.float64,
        requires_grad=True,
    )
    K = torch.randn(
        1,
        2,
        sequence_length,
        3,
        generator=generator,
        dtype=torch.float64,
        requires_grad=True,
    )
    V = torch.randn(
        1,
        1,
        sequence_length,
        4,
        generator=generator,
        dtype=torch.float64,
        requires_grad=True,
    )
    dO = torch.zeros(1, 2, sequence_length, 4, dtype=torch.float64)
    dO[:, :, query, :] = torch.randn(
        1, 2, 4, generator=generator, dtype=torch.float64
    )

    out = strict_tril_attn(Q, K, V, impl=impl, use_fn=True)
    ref = eager_tril_attn(Q.detach(), K.detach(), V.detach())
    assert torch.allclose(out, ref, rtol=1e-10, atol=1e-10)
    out.backward(dO)

    Q_ref = Q.detach().clone().requires_grad_(True)
    K_ref = K.detach().clone().requires_grad_(True)
    V_ref = V.detach().clone().requires_grad_(True)
    eager_tril_attn(Q_ref, K_ref, V_ref).backward(dO)
    assert torch.allclose(Q.grad, Q_ref.grad, rtol=1e-10, atol=1e-10)
    assert torch.allclose(K.grad, K_ref.grad, rtol=1e-10, atol=1e-10)
    assert torch.allclose(V.grad, V_ref.grad, rtol=1e-10, atol=1e-10)
    assert torch.equal(K.grad[:, :, query:, :], torch.zeros_like(K.grad[:, :, query:, :]))
    assert torch.equal(V.grad[:, :, query:, :], torch.zeros_like(V.grad[:, :, query:, :]))

@pytest.mark.parametrize("impl", ["eager", "blocked", "online", "triton", "cuda"])
def test_multiple_query_tiles_accumulate_strict_past_gradients(impl):
    """Multiple upstream rows across tiles accumulate only strict-past paths."""
    generator = torch.Generator().manual_seed(2041)
    sequence_length = 129
    selected_queries = (64, 128)
    Q = torch.randn(
        1, 2, sequence_length, 3, generator=generator, dtype=torch.float64, requires_grad=True
    )
    K = torch.randn(
        1, 2, sequence_length, 3, generator=generator, dtype=torch.float64, requires_grad=True
    )
    V = torch.randn(
        1, 1, sequence_length, 4, generator=generator, dtype=torch.float64, requires_grad=True
    )
    dO = torch.zeros(1, 2, sequence_length, 4, dtype=torch.float64)
    for query in selected_queries:
        dO[:, :, query, :] = torch.randn(
            1, 2, 4, generator=generator, dtype=torch.float64
        )

    out = strict_tril_attn(Q, K, V, impl=impl, use_fn=True)
    Q_ref = Q.detach().clone().requires_grad_(True)
    K_ref = K.detach().clone().requires_grad_(True)
    V_ref = V.detach().clone().requires_grad_(True)
    ref = eager_tril_attn(Q_ref, K_ref, V_ref)

    assert torch.allclose(out, ref, rtol=1e-10, atol=1e-10)
    out.backward(dO)
    ref.backward(dO)
    assert torch.allclose(Q.grad, Q_ref.grad, rtol=1e-10, atol=1e-10)
    assert torch.allclose(K.grad, K_ref.grad, rtol=1e-10, atol=1e-10)
    assert torch.allclose(V.grad, V_ref.grad, rtol=1e-10, atol=1e-10)

    inactive_queries = torch.ones(sequence_length, dtype=torch.bool)
    inactive_queries[list(selected_queries)] = False
    assert torch.equal(
        Q.grad[:, :, inactive_queries, :],
        torch.zeros_like(Q.grad[:, :, inactive_queries, :]),
    )
    assert torch.equal(K.grad[:, :, 128:, :], torch.zeros_like(K.grad[:, :, 128:, :]))
    assert torch.equal(V.grad[:, :, 128:, :], torch.zeros_like(V.grad[:, :, 128:, :]))

@pytest.mark.parametrize("impl", ["eager", "blocked", "online", "triton", "cuda"])
def test_zero_stride_self_attn_q_and_v_views_reduce_base_gradients(impl):
    """Self-attn must reduce duplicate Q/K and broadcast-V paths together."""
    generator = torch.Generator().manual_seed(2045)
    Q_base = torch.randn(
        1, 1, 5, 4, generator=generator, dtype=torch.float64, requires_grad=True
    )
    Q = Q_base.expand(2, 3, 5, 4)
    V_base = torch.randn(
        1, 1, 5, 6, generator=generator, dtype=torch.float64, requires_grad=True
    )
    V = V_base.expand(2, 3, 5, 6)
    dO = torch.randn(2, 3, 5, 6, generator=generator, dtype=torch.float64)

    out = strict_tril_attn(Q, Q, V, impl=impl, use_fn=True)
    Q_ref = Q.detach().clone().requires_grad_(True)
    V_ref = V.detach().clone().requires_grad_(True)
    ref = eager_tril_attn(Q_ref, Q_ref, V_ref)

    assert Q.stride(0) == 0 and Q.stride(1) == 0
    assert V.stride(0) == 0 and V.stride(1) == 0
    assert torch.allclose(out, ref, rtol=1e-12, atol=1e-12)
    out.backward(dO)
    ref.backward(dO)

    assert torch.allclose(
        Q_base.grad, Q_ref.grad.sum(dim=(0, 1), keepdim=True), rtol=1e-12, atol=1e-12
    )
    assert torch.allclose(
        V_base.grad, V_ref.grad.sum(dim=(0, 1), keepdim=True), rtol=1e-12, atol=1e-12
    )


@pytest.mark.parametrize("impl", ["eager", "blocked", "online", "triton", "cuda"])
def test_sequence_strided_inputs_preserve_backward_contract(impl):
    """Analytic backward must scatter gradients through sequence-strided views."""
    generator = torch.Generator().manual_seed(2046)
    Q_base = torch.randn(
        2, 3, 8, 4, generator=generator, dtype=torch.float64, requires_grad=True
    )
    K_base = torch.randn(
        2, 3, 8, 4, generator=generator, dtype=torch.float64, requires_grad=True
    )
    V_base = torch.randn(
        2, 1, 8, 6, generator=generator, dtype=torch.float64, requires_grad=True
    )
    Q = Q_base[:, :, ::2, :]
    K = K_base[:, :, ::2, :]
    V = V_base[:, :, ::2, :]
    dO = torch.randn(2, 3, 4, 6, generator=generator, dtype=torch.float64)

    out = strict_tril_attn(Q, K, V, impl=impl, use_fn=True)
    Q_ref = Q.detach().clone().requires_grad_(True)
    K_ref = K.detach().clone().requires_grad_(True)
    V_ref = V.detach().clone().requires_grad_(True)
    ref = eager_tril_attn(Q_ref, K_ref, V_ref)

    assert Q.stride(2) == 2 * Q_base.stride(2)
    assert K.stride(2) == 2 * K_base.stride(2)
    assert V.stride(2) == 2 * V_base.stride(2)
    assert torch.allclose(out, ref, rtol=1e-12, atol=1e-12)
    out.backward(dO)
    ref.backward(dO)

    assert torch.allclose(
        Q_base.grad[:, :, ::2, :], Q_ref.grad, rtol=1e-12, atol=1e-12
    )
    assert torch.allclose(
        K_base.grad[:, :, ::2, :], K_ref.grad, rtol=1e-12, atol=1e-12
    )
    assert torch.allclose(
        V_base.grad[:, :, ::2, :], V_ref.grad, rtol=1e-12, atol=1e-12
    )
    assert torch.equal(
        Q_base.grad[:, :, 1::2, :], torch.zeros_like(Q_base.grad[:, :, 1::2, :])
    )
    assert torch.equal(
        K_base.grad[:, :, 1::2, :], torch.zeros_like(K_base.grad[:, :, 1::2, :])
    )
    assert torch.equal(
        V_base.grad[:, :, 1::2, :], torch.zeros_like(V_base.grad[:, :, 1::2, :])
    )


@pytest.mark.parametrize("impl", ["eager", "blocked", "online", "triton", "cuda"])
def test_sequence_strided_self_attn_alias_preserves_backward_contract(impl):
    """Aliased sequence-strided Q/K views scatter both backward paths to Q_base."""
    generator = torch.Generator().manual_seed(2047)
    Q_base = torch.randn(
        2, 3, 8, 4, generator=generator, dtype=torch.float64, requires_grad=True
    )
    Q = Q_base[:, :, ::2, :]
    V = torch.randn(
        2, 1, 4, 6, generator=generator, dtype=torch.float64, requires_grad=True
    )
    dO = torch.randn(2, 3, 4, 6, generator=generator, dtype=torch.float64)

    out = strict_tril_attn(Q, Q, V, impl=impl, use_fn=True)
    Q_ref = Q.detach().clone().requires_grad_(True)
    V_ref = V.detach().clone().requires_grad_(True)
    ref = eager_tril_attn(Q_ref, Q_ref, V_ref)

    assert not Q.is_contiguous()
    assert Q.stride(2) == 2 * Q_base.stride(2)
    assert torch.allclose(out, ref, rtol=1e-12, atol=1e-12)
    out.backward(dO)
    ref.backward(dO)

    assert torch.allclose(
        Q_base.grad[:, :, ::2, :], Q_ref.grad, rtol=1e-12, atol=1e-12
    )
    assert torch.equal(
        Q_base.grad[:, :, 1::2, :], torch.zeros_like(Q_base.grad[:, :, 1::2, :])
    )
    assert torch.allclose(V.grad, V_ref.grad, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("impl", ["eager", "blocked", "online", "triton", "cuda"])
def test_self_attn_alias_with_broadcast_upstream_preserves_backward_contract(impl):
    """Aliased Q/K must reduce duplicate paths with broadcast upstream gradients."""
    generator = torch.Generator().manual_seed(2048)
    Q = torch.randn(
        2, 3, 5, 4, generator=generator, dtype=torch.float64, requires_grad=True
    )
    V = torch.randn(
        2, 1, 5, 6, generator=generator, dtype=torch.float64, requires_grad=True
    )
    dO_base = torch.randn(1, 1, 5, 6, generator=generator, dtype=torch.float64)
    dO = dO_base.expand(2, 3, 5, 6)

    out = strict_tril_attn(Q, Q, V, impl=impl, use_fn=True)
    Q_ref = Q.detach().clone().requires_grad_(True)
    V_ref = V.detach().clone().requires_grad_(True)
    ref = eager_tril_attn(Q_ref, Q_ref, V_ref)

    assert dO.stride(0) == 0 and dO.stride(1) == 0
    assert torch.allclose(out, ref, rtol=1e-12, atol=1e-12)
    out.backward(dO)
    ref.backward(dO.contiguous())

    assert torch.allclose(Q.grad, Q_ref.grad, rtol=1e-12, atol=1e-12)
    assert torch.allclose(V.grad, V_ref.grad, rtol=1e-12, atol=1e-12)
    assert torch.equal(V.grad[:, :, -1, :], torch.zeros_like(V.grad[:, :, -1, :]))
