"""Focused v19 contract coverage for dispatched score×V gradients.

This stays CPU-safe: every public dispatch alias is checked against the eager
strict-tril reference with distinct Q/K tensors with both shared and per-head
V values.
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
@pytest.mark.parametrize("v_heads", [1, 3])
def test_dispatch_preserves_distinct_qk_shared_and_per_head_v_gradients(
    impl, v_heads
):
    """Dispatch aliases preserve raw strict-tril gradients for V layouts."""
    B, H, T, N, D = 2, 3, 19, 7, 5
    generator = torch.Generator(device="cpu").manual_seed(1919)
    Q = torch.randn(B, H, T, N, generator=generator, dtype=torch.float64)
    K = torch.randn(B, H, T, N, generator=generator, dtype=torch.float64)
    V = torch.randn(B, v_heads, T, D, generator=generator, dtype=torch.float64)
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


@pytest.mark.parametrize("impl", ["blocked", "online", "triton", "cuda"])
def test_dispatch_preserves_capacity_strided_shared_v_gradients(impl):
    """Dispatch aliases preserve gradients through capacity-strided shared V."""
    B, H, T, N, D = 2, 3, 67, 5, 4
    generator = torch.Generator(device="cpu").manual_seed(1920)
    Q_storage = torch.randn(
        B, H, T + 3, N, generator=generator, dtype=torch.float64
    )
    K_storage = torch.randn(
        B, H, T + 3, N, generator=generator, dtype=torch.float64
    )
    V_storage = torch.randn(
        B, 1, T + 3, D, generator=generator, dtype=torch.float64
    )
    Q = Q_storage[:, :, 1 : T + 1, :]
    K = K_storage[:, :, 2 : T + 2, :]
    V = V_storage[:, :, 1 : T + 1, :]
    dO = torch.randn(B, H, T, D, generator=generator, dtype=torch.float64)

    assert not Q.is_contiguous()
    assert not K.is_contiguous()
    assert not V.is_contiguous()

    def run(name):
        q = Q.detach().requires_grad_(True)
        k = K.detach().requires_grad_(True)
        v = V.detach().requires_grad_(True)
        out = bdh_attn(q, k, v, impl=name)
        out.backward(dO)
        return out.detach(), tuple(x.grad.detach() for x in (q, k, v))

    ref, ref_grads = run("eager")
    expected = (Q @ K.transpose(-2, -1)).tril(diagonal=-1) @ V
    assert torch.equal(ref, expected)

    got, got_grads = run(impl)
    assert torch.allclose(got, ref, rtol=1e-9, atol=1e-9)
    for got_grad, ref_grad in zip(got_grads, ref_grads):
        assert torch.allclose(got_grad, ref_grad, rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize("impl", ["blocked", "online", "triton", "cuda"])
def test_dispatch_preserves_capacity_strided_per_head_v_gradients(impl):
    """Dispatch aliases preserve gradients through capacity-strided per-head V."""
    B, H, T, N, D = 2, 3, 67, 5, 4
    generator = torch.Generator(device="cpu").manual_seed(1923)
    Q_storage = torch.randn(
        B, H, T + 3, N, generator=generator, dtype=torch.float64
    )
    K_storage = torch.randn(
        B, H, T + 3, N, generator=generator, dtype=torch.float64
    )
    V_storage = torch.randn(
        B, H, T + 3, D, generator=generator, dtype=torch.float64
    )
    Q = Q_storage[:, :, 1 : T + 1, :]
    K = K_storage[:, :, 2 : T + 2, :]
    V = V_storage[:, :, 1 : T + 1, :]
    dO = torch.randn(B, H, T, D, generator=generator, dtype=torch.float64)

    assert not Q.is_contiguous()
    assert not K.is_contiguous()
    assert not V.is_contiguous()

    def run(name):
        q = Q.detach().requires_grad_(True)
        k = K.detach().requires_grad_(True)
        v = V.detach().requires_grad_(True)
        out = bdh_attn(q, k, v, impl=name)
        out.backward(dO)
        return out.detach(), tuple(x.grad.detach() for x in (q, k, v))

    ref, ref_grads = run("eager")
    expected = (Q @ K.transpose(-2, -1)).tril(diagonal=-1) @ V
    assert torch.equal(ref, expected)

    got, got_grads = run(impl)
    assert torch.allclose(got, ref, rtol=1e-9, atol=1e-9)
    for got_grad, ref_grad in zip(got_grads, ref_grads):
        assert torch.allclose(got_grad, ref_grad, rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize("impl", ["blocked", "online", "triton", "cuda"])
@pytest.mark.parametrize("v_heads", [1, 3])
def test_dispatch_preserves_capacity_strided_v_storage_gradients(impl, v_heads):
    """Dispatch aliases accumulate V gradients in the backing storage."""
    B, H, T, N, D = 2, 3, 67, 5, 4
    generator = torch.Generator(device="cpu").manual_seed(1924)
    Q = torch.randn(B, H, T, N, generator=generator, dtype=torch.float64)
    K = torch.randn(B, H, T, N, generator=generator, dtype=torch.float64)
    V_storage = torch.randn(
        B, v_heads, T + 3, D, generator=generator, dtype=torch.float64
    )
    V = V_storage[:, :, 1 : T + 1, :]
    dO = torch.randn(B, H, T, D, generator=generator, dtype=torch.float64)

    assert not V.is_contiguous()

    def run(name):
        q = Q.detach().clone().requires_grad_(True)
        k = K.detach().clone().requires_grad_(True)
        v_storage = V_storage.detach().clone().requires_grad_(True)
        v = v_storage[:, :, 1 : T + 1, :]
        out = bdh_attn(q, k, v, impl=name)
        out.backward(dO)
        return out.detach(), v_storage.grad.detach()

    ref, ref_storage_grad = run("eager")
    expected = (Q @ K.transpose(-2, -1)).tril(diagonal=-1) @ V
    assert torch.equal(ref, expected)

    got, got_storage_grad = run(impl)
    assert torch.allclose(got, ref, rtol=1e-9, atol=1e-9)
    assert torch.allclose(
        got_storage_grad, ref_storage_grad, rtol=1e-9, atol=1e-9
    )


@pytest.mark.parametrize("impl", ["blocked", "online", "triton", "cuda"])
@pytest.mark.parametrize("v_heads", [1, 3])
def test_dispatch_preserves_capacity_strided_qk_storage_gradients(impl, v_heads):
    """Dispatch aliases accumulate Q/K storage gradients for shared/per-head V."""
    B, H, T, N, D = 2, 3, 67, 5, 4
    generator = torch.Generator(device="cpu").manual_seed(1925)
    Q_storage = torch.randn(
        B, H, T + 3, N, generator=generator, dtype=torch.float64
    )
    K_storage = torch.randn(
        B, H, T + 3, N, generator=generator, dtype=torch.float64
    )
    V = torch.randn(B, v_heads, T, D, generator=generator, dtype=torch.float64)
    Q = Q_storage[:, :, 1 : T + 1, :]
    K = K_storage[:, :, 2 : T + 2, :]
    dO = torch.randn(B, H, T, D, generator=generator, dtype=torch.float64)

    assert not Q.is_contiguous()
    assert not K.is_contiguous()

    def run(name):
        q_storage = Q_storage.detach().clone().requires_grad_(True)
        k_storage = K_storage.detach().clone().requires_grad_(True)
        q = q_storage[:, :, 1 : T + 1, :]
        k = k_storage[:, :, 2 : T + 2, :]
        out = bdh_attn(q, k, V, impl=name)
        out.backward(dO)
        return out.detach(), q_storage.grad.detach(), k_storage.grad.detach()

    ref, ref_q_storage_grad, ref_k_storage_grad = run("eager")
    expected = (Q @ K.transpose(-2, -1)).tril(diagonal=-1) @ V
    assert torch.equal(ref, expected)

    got, got_q_storage_grad, got_k_storage_grad = run(impl)
    assert torch.allclose(got, ref, rtol=1e-9, atol=1e-9)
    assert torch.allclose(
        got_q_storage_grad, ref_q_storage_grad, rtol=1e-9, atol=1e-9
    )
    assert torch.allclose(
        got_k_storage_grad, ref_k_storage_grad, rtol=1e-9, atol=1e-9
    )


@pytest.mark.parametrize("impl", ["blocked", "online", "triton", "cuda"])
def test_dispatch_preserves_feature_strided_shared_v_gradients(impl):
    """Dispatch aliases preserve gradients through a feature-strided shared V."""
    B, H, T, N, D = 2, 3, 29, 5, 4
    generator = torch.Generator(device="cpu").manual_seed(1921)
    Q = torch.randn(B, H, T, N, generator=generator, dtype=torch.float64)
    K = torch.randn(B, H, T, N, generator=generator, dtype=torch.float64)
    V_storage = torch.randn(
        B, 1, T, 2 * D + 1, generator=generator, dtype=torch.float64
    )
    V = V_storage[..., 1 : 2 * D : 2]
    dO = torch.randn(B, H, T, D, generator=generator, dtype=torch.float64)

    assert not V.is_contiguous()
    assert V.stride(-1) == 2

    def run(name):
        q = Q.detach().clone().requires_grad_(True)
        k = K.detach().clone().requires_grad_(True)
        v = V.detach().requires_grad_(True)
        out = bdh_attn(q, k, v, impl=name)
        out.backward(dO)
        return out.detach(), tuple(x.grad.detach() for x in (q, k, v))

    ref, ref_grads = run("eager")
    expected = (Q @ K.transpose(-2, -1)).tril(diagonal=-1) @ V
    assert torch.equal(ref, expected)

    got, got_grads = run(impl)
    assert torch.allclose(got, ref, rtol=1e-9, atol=1e-9)
    for got_grad, ref_grad in zip(got_grads, ref_grads):
        assert torch.allclose(got_grad, ref_grad, rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize("impl", ["blocked", "online", "triton", "cuda"])
def test_dispatch_preserves_feature_strided_per_head_v_gradients(impl):
    """Dispatch aliases preserve gradients through a feature-strided per-head V."""
    B, H, T, N, D = 2, 3, 29, 5, 4
    generator = torch.Generator(device="cpu").manual_seed(1922)
    Q = torch.randn(B, H, T, N, generator=generator, dtype=torch.float64)
    K = torch.randn(B, H, T, N, generator=generator, dtype=torch.float64)
    V_storage = torch.randn(
        B, H, T, 2 * D + 1, generator=generator, dtype=torch.float64
    )
    V = V_storage[..., 1 : 2 * D : 2]
    dO = torch.randn(B, H, T, D, generator=generator, dtype=torch.float64)

    assert not V.is_contiguous()
    assert V.stride(-1) == 2

    def run(name):
        q = Q.detach().clone().requires_grad_(True)
        k = K.detach().clone().requires_grad_(True)
        v = V.detach().requires_grad_(True)
        out = bdh_attn(q, k, v, impl=name)
        out.backward(dO)
        return out.detach(), tuple(x.grad.detach() for x in (q, k, v))

    ref, ref_grads = run("eager")
    expected = (Q @ K.transpose(-2, -1)).tril(diagonal=-1) @ V
    assert torch.equal(ref, expected)

    got, got_grads = run(impl)
    assert torch.allclose(got, ref, rtol=1e-9, atol=1e-9)
    for got_grad, ref_grad in zip(got_grads, ref_grads):
        assert torch.allclose(got_grad, ref_grad, rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize("impl", ["blocked", "online", "triton", "cuda"])
@pytest.mark.parametrize("v_heads", [1, 3])
def test_dispatch_preserves_feature_strided_v_storage_gradients(impl, v_heads):
    """Dispatch aliases accumulate feature-strided V gradients in storage."""
    B, H, T, N, D = 2, 3, 29, 5, 4
    generator = torch.Generator(device="cpu").manual_seed(1926)
    Q = torch.randn(B, H, T, N, generator=generator, dtype=torch.float64)
    K = torch.randn(B, H, T, N, generator=generator, dtype=torch.float64)
    V_storage = torch.randn(
        B, v_heads, T, 2 * D + 1, generator=generator, dtype=torch.float64
    )
    V = V_storage[..., 1 : 2 * D : 2]
    dO = torch.randn(B, H, T, D, generator=generator, dtype=torch.float64)

    assert not V.is_contiguous()
    assert V.stride(-1) == 2

    def run(name):
        q = Q.detach().clone().requires_grad_(True)
        k = K.detach().clone().requires_grad_(True)
        v_storage = V_storage.detach().clone().requires_grad_(True)
        v = v_storage[..., 1 : 2 * D : 2]
        out = bdh_attn(q, k, v, impl=name)
        out.backward(dO)
        return out.detach(), v_storage.grad.detach()

    ref, ref_storage_grad = run("eager")
    expected = (Q @ K.transpose(-2, -1)).tril(diagonal=-1) @ V
    assert torch.equal(ref, expected)

    got, got_storage_grad = run(impl)
    assert torch.allclose(got, ref, rtol=1e-9, atol=1e-9)
    assert torch.allclose(
        got_storage_grad, ref_storage_grad, rtol=1e-9, atol=1e-9
    )


@pytest.mark.parametrize("impl", ["blocked", "online", "triton", "cuda"])
@pytest.mark.parametrize("v_heads", [1, 3])
def test_dispatch_preserves_feature_strided_qk_storage_gradients(impl, v_heads):
    """Dispatch aliases accumulate feature-strided Q/K gradients in storage."""
    B, H, T, N, D = 2, 3, 29, 5, 4
    generator = torch.Generator(device="cpu").manual_seed(1927)
    Q_storage = torch.randn(
        B, H, T, 2 * N + 1, generator=generator, dtype=torch.float64
    )
    K_storage = torch.randn(
        B, H, T, 2 * N + 1, generator=generator, dtype=torch.float64
    )
    Q = Q_storage[..., 1 : 2 * N : 2]
    K = K_storage[..., 1 : 2 * N : 2]
    V = torch.randn(B, v_heads, T, D, generator=generator, dtype=torch.float64)
    dO = torch.randn(B, H, T, D, generator=generator, dtype=torch.float64)

    assert not Q.is_contiguous()
    assert not K.is_contiguous()
    assert Q.stride(-1) == 2
    assert K.stride(-1) == 2

    def run(name):
        q_storage = Q_storage.detach().clone().requires_grad_(True)
        k_storage = K_storage.detach().clone().requires_grad_(True)
        q = q_storage[..., 1 : 2 * N : 2]
        k = k_storage[..., 1 : 2 * N : 2]
        out = bdh_attn(q, k, V, impl=name)
        out.backward(dO)
        return out.detach(), q_storage.grad.detach(), k_storage.grad.detach()

    ref, ref_q_storage_grad, ref_k_storage_grad = run("eager")
    expected = (Q @ K.transpose(-2, -1)).tril(diagonal=-1) @ V
    assert torch.equal(ref, expected)

    got, got_q_storage_grad, got_k_storage_grad = run(impl)
    assert torch.allclose(got, ref, rtol=1e-9, atol=1e-9)
    assert torch.allclose(
        got_q_storage_grad, ref_q_storage_grad, rtol=1e-9, atol=1e-9
    )
    assert torch.allclose(
        got_k_storage_grad, ref_k_storage_grad, rtol=1e-9, atol=1e-9
    )


@pytest.mark.parametrize("impl", ["blocked", "online", "triton", "cuda"])
@pytest.mark.parametrize("v_heads", [1, 3])
def test_dispatch_preserves_composed_feature_strided_storage_gradients(
    impl, v_heads
):
    """Dispatch aliases preserve gradients when every Q/K/V view is strided."""
    B, H, T, N, D = 2, 3, 29, 5, 4
    generator = torch.Generator(device="cpu").manual_seed(1928)
    Q_storage = torch.randn(
        B, H, T, 2 * N + 1, generator=generator, dtype=torch.float64
    )
    K_storage = torch.randn(
        B, H, T, 2 * N + 1, generator=generator, dtype=torch.float64
    )
    V_storage = torch.randn(
        B, v_heads, T, 2 * D + 1, generator=generator, dtype=torch.float64
    )
    Q = Q_storage[..., 1 : 2 * N : 2]
    K = K_storage[..., 1 : 2 * N : 2]
    V = V_storage[..., 1 : 2 * D : 2]
    dO = torch.randn(B, H, T, D, generator=generator, dtype=torch.float64)

    assert not Q.is_contiguous()
    assert not K.is_contiguous()
    assert not V.is_contiguous()
    assert Q.stride(-1) == 2
    assert K.stride(-1) == 2
    assert V.stride(-1) == 2

    def run(name):
        q_storage = Q_storage.detach().clone().requires_grad_(True)
        k_storage = K_storage.detach().clone().requires_grad_(True)
        v_storage = V_storage.detach().clone().requires_grad_(True)
        q = q_storage[..., 1 : 2 * N : 2]
        k = k_storage[..., 1 : 2 * N : 2]
        v = v_storage[..., 1 : 2 * D : 2]
        out = bdh_attn(q, k, v, impl=name)
        out.backward(dO)
        return (
            out.detach(),
            q_storage.grad.detach(),
            k_storage.grad.detach(),
            v_storage.grad.detach(),
        )

    ref, ref_q_grad, ref_k_grad, ref_v_grad = run("eager")
    expected = (Q @ K.transpose(-2, -1)).tril(diagonal=-1) @ V
    assert torch.equal(ref, expected)

    got, got_q_grad, got_k_grad, got_v_grad = run(impl)
    assert torch.allclose(got, ref, rtol=1e-9, atol=1e-9)
    for got_grad, ref_grad in (
        (got_q_grad, ref_q_grad),
        (got_k_grad, ref_k_grad),
        (got_v_grad, ref_v_grad),
    ):
        assert torch.allclose(got_grad, ref_grad, rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize("impl", ["eager", "blocked", "online", "triton", "cuda"])
@pytest.mark.parametrize("v_heads", [1, 3])
def test_dispatch_preserves_self_qk_analytic_autograd_contract(impl, v_heads):
    """Analytic dispatch keeps strict-tril gradients when Q and K alias."""
    B, H, T, N, D = 2, 3, 23, 5, 4
    generator = torch.Generator(device="cpu").manual_seed(1929)
    Q = torch.randn(B, H, T, N, generator=generator, dtype=torch.float64)
    V = torch.randn(B, v_heads, T, D, generator=generator, dtype=torch.float64)
    dO = torch.randn(B, H, T, D, generator=generator, dtype=torch.float64)

    def run(name, use_autograd_fn):
        q = Q.detach().clone().requires_grad_(True)
        v = V.detach().clone().requires_grad_(True)
        out = bdh_attn(q, q, v, impl=name, use_autograd_fn=use_autograd_fn)
        out.backward(dO)
        return out.detach(), q.grad.detach(), v.grad.detach()

    ref, ref_q_grad, ref_v_grad = run("eager", use_autograd_fn=False)
    expected = (Q @ Q.transpose(-2, -1)).tril(diagonal=-1) @ V
    assert torch.equal(ref, expected)

    got, got_q_grad, got_v_grad = run(impl, use_autograd_fn=True)
    assert torch.allclose(got, ref, rtol=1e-9, atol=1e-9)
    assert torch.allclose(got_q_grad, ref_q_grad, rtol=1e-9, atol=1e-9)
    assert torch.allclose(got_v_grad, ref_v_grad, rtol=1e-9, atol=1e-9)
