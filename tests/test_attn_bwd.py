"""Gradient checks for strict-tril attention autograd.Function."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import bdh  # noqa: E402
import bdh_baseline as baseline  # noqa: E402
from kernels.attention import blocked_tril_attn, eager_tril_attn  # noqa: E402
from kernels.attention_bwd import (  # noqa: E402
    StrictTrilAttnFn,
    analytic_tril_attn_backward,
    strict_tril_attn,
)
from kernels.attention_dispatch import bdh_attn  # noqa: E402


def _qkv(B=2, H=3, T=5, N=4, D=6, seed=0, v_heads=1):
    g = torch.Generator().manual_seed(seed)
    Q = torch.randn(B, H, T, N, generator=g, dtype=torch.float64)
    K = torch.randn(B, H, T, N, generator=g, dtype=torch.float64)
    V = torch.randn(B, v_heads, T, D, generator=g, dtype=torch.float64)
    return Q, K, V


def test_analytic_matches_eager_autograd_v_broadcast():
    Q, K, V = _qkv(v_heads=1)
    Q.requires_grad_(True)
    K.requires_grad_(True)
    V.requires_grad_(True)
    O = eager_tril_attn(Q, K, V)
    dO = torch.randn_like(O)
    O.backward(dO)
    dQ, dK, dV = analytic_tril_attn_backward(Q.detach(), K.detach(), V.detach(), dO)
    assert torch.allclose(Q.grad, dQ, rtol=1e-8, atol=1e-8)
    assert torch.allclose(K.grad, dK, rtol=1e-8, atol=1e-8)
    assert torch.allclose(V.grad, dV, rtol=1e-8, atol=1e-8)


def test_analytic_matches_eager_autograd_v_full_heads():
    Q, K, V = _qkv(v_heads=3)
    Q.requires_grad_(True)
    K.requires_grad_(True)
    V.requires_grad_(True)
    O = eager_tril_attn(Q, K, V)
    dO = torch.randn_like(O)
    O.backward(dO)
    dQ, dK, dV = analytic_tril_attn_backward(Q.detach(), K.detach(), V.detach(), dO)
    assert torch.allclose(Q.grad, dQ, rtol=1e-8, atol=1e-8)
    assert torch.allclose(K.grad, dK, rtol=1e-8, atol=1e-8)
    assert torch.allclose(V.grad, dV, rtol=1e-8, atol=1e-8)


@pytest.mark.parametrize("impl", ["eager", "blocked"])
def test_function_grads_match_eager_baseline(impl):
    """StrictTrilAttnFn grads must match plain eager PyTorch autograd."""
    Q, K, V = _qkv(B=1, H=2, T=6, N=5, D=7, seed=11)
    Qe, Ke, Ve = Q.clone().requires_grad_(True), K.clone().requires_grad_(True), V.clone().requires_grad_(True)
    Qf, Kf, Vf = Q.clone().requires_grad_(True), K.clone().requires_grad_(True), V.clone().requires_grad_(True)

    Oe = eager_tril_attn(Qe, Ke, Ve)
    Of = strict_tril_attn(Qf, Kf, Vf, impl=impl, use_fn=True)
    assert torch.allclose(Of, Oe, rtol=1e-7, atol=1e-7)

    dO = torch.randn_like(Oe)
    Oe.backward(dO)
    Of.backward(dO.clone())

    assert torch.allclose(Qf.grad, Qe.grad, rtol=1e-6, atol=1e-6)
    assert torch.allclose(Kf.grad, Ke.grad, rtol=1e-6, atol=1e-6)
    assert torch.allclose(Vf.grad, Ve.grad, rtol=1e-6, atol=1e-6)


def test_gradcheck_small_shapes():
    """torch.autograd.gradcheck on small float64 tensors."""

    def f(Q, K, V):
        return StrictTrilAttnFn.apply(Q, K, V, "eager")

    Q, K, V = _qkv(B=1, H=2, T=4, N=3, D=3, seed=42)
    Q = Q.requires_grad_(True)
    K = K.requires_grad_(True)
    V = V.requires_grad_(True)
    assert torch.autograd.gradcheck(f, (Q, K, V), eps=1e-6, atol=1e-4, rtol=1e-3)


def test_gradcheck_blocked_impl():
    def f(Q, K, V):
        return StrictTrilAttnFn.apply(Q, K, V, "blocked")

    Q, K, V = _qkv(B=1, H=1, T=5, N=4, D=3, seed=7)
    Q = Q.requires_grad_(True)
    K = K.requires_grad_(True)
    V = V.requires_grad_(True)
    assert torch.autograd.gradcheck(f, (Q, K, V), eps=1e-6, atol=1e-4, rtol=1e-3)


def test_finite_diff_manual():
    """Central finite-diff spot-check on a few Q/K/V coordinates."""
    torch.manual_seed(0)
    Q, K, V = _qkv(B=1, H=1, T=3, N=2, D=2, seed=99)
    Q = Q.clone().requires_grad_(True)
    K = K.clone().requires_grad_(True)
    V = V.clone().requires_grad_(True)
    O = strict_tril_attn(Q, K, V, impl="eager", use_fn=True)
    loss = O.sum()
    loss.backward()

    eps = 1e-6
    # perturb Q[0,0,1,0]
    Q2 = Q.detach().clone()
    Q2[0, 0, 1, 0] += eps
    f_pos = strict_tril_attn(Q2, K.detach(), V.detach(), impl="eager", use_fn=True).sum()
    Q2[0, 0, 1, 0] -= 2 * eps
    f_neg = strict_tril_attn(Q2, K.detach(), V.detach(), impl="eager", use_fn=True).sum()
    fd = ((f_pos - f_neg) / (2 * eps)).item()
    assert abs(fd - Q.grad[0, 0, 1, 0].item()) < 1e-4


def test_attention_module_backward_matches_baseline_with_fn(monkeypatch):
    """When BDH_ATTN_AUTOGRAD=1, Attention cold path grads match baseline."""
    monkeypatch.setenv("BDH_ATTN_AUTOGRAD", "1")
    monkeypatch.delenv("BDH_ATTN_IMPL", raising=False)

    cfg = bdh.BDHConfig(
        n_layer=1,
        n_embd=32,
        n_head=2,
        mlp_internal_dim_multiplier=4,
        dropout=0.0,
        vocab_size=256,
    )
    torch.manual_seed(5)
    a_opt = bdh.Attention(cfg)
    a_base = baseline.Attention(cfg)
    a_opt.load_state_dict(a_base.state_dict())

    N = cfg.mlp_internal_dim_multiplier * cfg.n_embd // cfg.n_head
    Q = torch.randn(2, cfg.n_head, 8, N, dtype=torch.float64, requires_grad=True)
    V = torch.randn(2, 1, 8, cfg.n_embd, dtype=torch.float64, requires_grad=True)
    Qb = Q.detach().clone().requires_grad_(True)
    Vb = V.detach().clone().requires_grad_(True)

    # baseline Attention returns out only (older API) or tuple — handle both
    out_base = a_base(Qb, Qb, Vb)
    if isinstance(out_base, tuple):
        out_base = out_base[0]
    out_opt, _, _ = a_opt(Q, Q, V)

    assert torch.allclose(out_base, out_opt, rtol=1e-7, atol=1e-7)

    g = torch.randn_like(out_base)
    out_base.backward(g)
    out_opt.backward(g.clone())

    assert torch.allclose(Q.grad, Qb.grad, rtol=1e-5, atol=1e-5)
    assert torch.allclose(V.grad, Vb.grad, rtol=1e-5, atol=1e-5)


def test_default_eager_path_unaffected(monkeypatch):
    """Without BDH_ATTN_AUTOGRAD, dispatch/default stays plain eager."""
    monkeypatch.delenv("BDH_ATTN_AUTOGRAD", raising=False)
    monkeypatch.delenv("BDH_ATTN_IMPL", raising=False)
    Q, K, V = _qkv(T=4, seed=3)
    Q = Q.float()
    K = K.float()
    V = V.float()
    out = bdh_attn(Q, K, V)
    ref = eager_tril_attn(Q, K, V)
    assert torch.equal(out, ref)


def test_dispatch_autograd_env(monkeypatch):
    monkeypatch.setenv("BDH_ATTN_AUTOGRAD", "1")
    monkeypatch.setenv("BDH_ATTN_IMPL", "blocked")
    Q, K, V = _qkv(B=1, H=2, T=4, N=3, D=3, seed=2)
    Q = Q.float().requires_grad_(True)
    K = K.float().requires_grad_(True)
    V = V.float().requires_grad_(True)
    out = bdh_attn(Q, K, V)
    out.sum().backward()
    assert Q.grad is not None and K.grad is not None and V.grad is not None
    # Match eager grads
    Q2 = Q.detach().clone().requires_grad_(True)
    K2 = K.detach().clone().requires_grad_(True)
    V2 = V.detach().clone().requires_grad_(True)
    eager_tril_attn(Q2, K2, V2).sum().backward()
    assert torch.allclose(Q.grad, Q2.grad, rtol=1e-4, atol=1e-4)
    assert torch.allclose(K.grad, K2.grad, rtol=1e-4, atol=1e-4)
    assert torch.allclose(V.grad, V2.grad, rtol=1e-4, atol=1e-4)


def test_use_fn_false_blocked_still_forward():
    Q, K, V = _qkv(T=8, seed=1)
    Q, K, V = Q.float(), K.float(), V.float()
    out = strict_tril_attn(Q, K, V, impl="blocked", use_fn=False)
    assert torch.allclose(out, blocked_tril_attn(Q, K, V), rtol=1e-5, atol=1e-5)
