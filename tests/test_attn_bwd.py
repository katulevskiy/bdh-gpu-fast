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
    StrictTrilSelfAttnFn,
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


def _tiny_cfg(**kwargs) -> bdh.BDHConfig:
    defaults = dict(
        n_layer=1,
        n_embd=32,
        n_head=2,
        mlp_internal_dim_multiplier=4,
        dropout=0.0,
        vocab_size=256,
    )
    defaults.update(kwargs)
    return bdh.BDHConfig(**defaults)


def test_pos0_zero_strict_tril_fn():
    """tril(diagonal=-1) ⇒ attention output at position 0 is all zeros."""
    Q, K, V = _qkv(B=1, H=2, T=5, N=4, D=6, seed=2)
    Q, K, V = Q.float(), K.float(), V.float()
    out = strict_tril_attn(Q, K, V, impl="eager", use_fn=True)
    assert torch.allclose(out[:, :, 0, :], torch.zeros_like(out[:, :, 0, :]))
    out_b = strict_tril_attn(Q, K, V, impl="blocked", use_fn=True)
    assert torch.allclose(out_b[:, :, 0, :], torch.zeros_like(out_b[:, :, 0, :]))


def test_full_model_grad_parity_dropout0(monkeypatch):
    """Full BDH train grads: AUTOGRAD=1 matches eager autograd at dropout=0."""
    monkeypatch.delenv("BDH_ATTN_IMPL", raising=False)
    cfg = _tiny_cfg()
    torch.manual_seed(0)
    m0 = bdh.BDH(cfg)
    torch.manual_seed(0)
    m1 = bdh.BDH(cfg)
    m1.load_state_dict(m0.state_dict())

    torch.manual_seed(1)
    x = torch.randint(0, 256, (2, 16))
    y = torch.randint(0, 256, (2, 16))

    monkeypatch.delenv("BDH_ATTN_AUTOGRAD", raising=False)
    m0.zero_grad(set_to_none=True)
    _, loss0 = m0(x, y)
    loss0.backward()

    monkeypatch.setenv("BDH_ATTN_AUTOGRAD", "1")
    m1.zero_grad(set_to_none=True)
    _, loss1 = m1(x, y)
    loss1.backward()

    assert torch.allclose(loss0, loss1, rtol=1e-6, atol=1e-6)
    max_diff = 0.0
    for (n0, p0), (n1, p1) in zip(m0.named_parameters(), m1.named_parameters()):
        assert p0.grad is not None and p1.grad is not None, n0
        d = (p0.grad - p1.grad).abs().max().item()
        max_diff = max(max_diff, d)
        assert torch.allclose(p0.grad, p1.grad, rtol=1e-4, atol=1e-5), (
            f"{n0} grad mismatch max={d}"
        )
    assert max_diff < 1e-4


def test_generate_works_with_autograd_flag(monkeypatch):
    """CacheManager incremental decode must keep working with AUTOGRAD=1."""
    monkeypatch.setenv("BDH_ATTN_AUTOGRAD", "1")
    monkeypatch.delenv("BDH_ATTN_IMPL", raising=False)
    cfg = _tiny_cfg(n_layer=2)
    torch.manual_seed(3)
    m = bdh.BDH(cfg).eval()
    prompt = torch.randint(0, 256, (1, 8))
    out = m.generate(prompt.clone(), max_new_tokens=4, temperature=1.0)
    assert out.shape == (1, 12)
    # tokens after prompt should be written
    assert not torch.equal(out[:, 8:], prompt.new_zeros(1, 4))


def test_cuda_impl_with_autograd_fn(monkeypatch):
    """BDH_ATTN_IMPL=cuda + AUTOGRAD=1 must not raise (ref forward + analytic)."""
    monkeypatch.setenv("BDH_ATTN_AUTOGRAD", "1")
    monkeypatch.setenv("BDH_ATTN_IMPL", "cuda")
    Q, K, V = _qkv(B=1, H=2, T=4, N=3, D=3, seed=4)
    Q = Q.float().requires_grad_(True)
    K = K.float().requires_grad_(True)
    V = V.float().requires_grad_(True)
    out = bdh_attn(Q, K, V)
    assert out[:, :, 0, :].abs().max().item() == 0.0
    out.sum().backward()
    assert Q.grad is not None and K.grad is not None and V.grad is not None


# --- opt/blocked-autograd: tiled analytic bwd + blocked|online train path ---

from kernels.attention_bwd import (  # noqa: E402
    analytic_tril_attn_backward_blocked,
    _use_blocked_analytic_bwd,
)
from kernels.attention_dispatch import resolve_attn_impl  # noqa: E402


def test_online_alias_resolves_to_blocked():
    assert resolve_attn_impl("online") == "blocked"
    assert resolve_attn_impl("blocked") == "blocked"
    assert _use_blocked_analytic_bwd("online")
    assert _use_blocked_analytic_bwd("blocked")
    assert _use_blocked_analytic_bwd("triton")
    assert _use_blocked_analytic_bwd("cuda")
    assert not _use_blocked_analytic_bwd("eager")


@pytest.mark.parametrize("v_heads", [1, 3])
def test_blocked_analytic_matches_dense(v_heads):
    """Tiled analytic bwd ≡ dense M-recompute (float64)."""
    Q, K, V = _qkv(B=2, H=3, T=17, N=5, D=7, seed=13, v_heads=v_heads)
    dO = torch.randn(2, 3, 17, 7, dtype=torch.float64)
    dQ1, dK1, dV1 = analytic_tril_attn_backward(Q, K, V, dO)
    dQ2, dK2, dV2 = analytic_tril_attn_backward_blocked(Q, K, V, dO, block_size=4)
    assert torch.allclose(dQ1, dQ2, rtol=1e-10, atol=1e-10)
    assert torch.allclose(dK1, dK2, rtol=1e-10, atol=1e-10)
    assert torch.allclose(dV1, dV2, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("impl", ["blocked", "online"])
def test_blocked_online_fn_grads_match_eager_autograd(impl):
    """StrictTrilAttnFn(blocked|online) grads match eager autograd @ dropout=0 math."""
    Q, K, V = _qkv(B=1, H=2, T=9, N=4, D=5, seed=21)
    Qe = Q.clone().requires_grad_(True)
    Ke = K.clone().requires_grad_(True)
    Ve = V.clone().requires_grad_(True)
    Qf = Q.clone().requires_grad_(True)
    Kf = K.clone().requires_grad_(True)
    Vf = V.clone().requires_grad_(True)

    Oe = eager_tril_attn(Qe, Ke, Ve)
    Of = strict_tril_attn(Qf, Kf, Vf, impl=impl, use_fn=True)
    assert torch.allclose(Of, Oe, rtol=1e-7, atol=1e-7)
    assert torch.allclose(Of[:, :, 0, :], torch.zeros_like(Of[:, :, 0, :]))

    dO = torch.randn_like(Oe)
    Oe.backward(dO)
    Of.backward(dO.clone())
    assert torch.allclose(Qf.grad, Qe.grad, rtol=1e-6, atol=1e-6)
    assert torch.allclose(Kf.grad, Ke.grad, rtol=1e-6, atol=1e-6)
    assert torch.allclose(Vf.grad, Ve.grad, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("impl", ["blocked", "online"])
def test_full_model_blocked_autograd_parity(impl, monkeypatch):
    """Full BDH: IMPL=blocked|online + AUTOGRAD=1 matches eager+AUTOGRAD @ dropout=0."""
    cfg = _tiny_cfg()
    torch.manual_seed(0)
    m_eager = bdh.BDH(cfg)
    torch.manual_seed(0)
    m_blk = bdh.BDH(cfg)
    m_blk.load_state_dict(m_eager.state_dict())

    torch.manual_seed(2)
    x = torch.randint(0, 256, (2, 12))
    y = torch.randint(0, 256, (2, 12))

    monkeypatch.setenv("BDH_ATTN_AUTOGRAD", "1")
    monkeypatch.setenv("BDH_ATTN_IMPL", "eager")
    m_eager.zero_grad(set_to_none=True)
    _, loss_e = m_eager(x, y)
    loss_e.backward()

    monkeypatch.setenv("BDH_ATTN_IMPL", impl)
    m_blk.zero_grad(set_to_none=True)
    _, loss_b = m_blk(x, y)
    loss_b.backward()

    assert torch.allclose(loss_e, loss_b, rtol=1e-5, atol=1e-5)
    for (n0, p0), (n1, p1) in zip(m_eager.named_parameters(), m_blk.named_parameters()):
        assert p0.grad is not None and p1.grad is not None, n0
        assert torch.allclose(p0.grad, p1.grad, rtol=1e-4, atol=1e-5), (
            f"{n0} grad mismatch max={(p0.grad - p1.grad).abs().max().item()}"
        )


def test_dispatch_online_autograd_env(monkeypatch):
    """BDH_ATTN_IMPL=online + AUTOGRAD=1 routes and matches eager grads."""
    monkeypatch.setenv("BDH_ATTN_AUTOGRAD", "1")
    monkeypatch.setenv("BDH_ATTN_IMPL", "online")
    Q, K, V = _qkv(B=1, H=2, T=5, N=3, D=3, seed=8)
    Q = Q.float().requires_grad_(True)
    K = K.float().requires_grad_(True)
    V = V.float().requires_grad_(True)
    out = bdh_attn(Q, K, V)
    assert out[:, :, 0, :].abs().max().item() == 0.0
    out.sum().backward()
    Q2 = Q.detach().clone().requires_grad_(True)
    K2 = K.detach().clone().requires_grad_(True)
    V2 = V.detach().clone().requires_grad_(True)
    eager_tril_attn(Q2, K2, V2).sum().backward()
    assert torch.allclose(Q.grad, Q2.grad, rtol=1e-4, atol=1e-4)
    assert torch.allclose(K.grad, K2.grad, rtol=1e-4, atol=1e-4)
    assert torch.allclose(V.grad, V2.grad, rtol=1e-4, atol=1e-4)


def test_self_attn_fn_grads_match_duplicate_q():
    """StrictTrilSelfAttnFn (Q is K) grads == StrictTrilAttnFn with distinct clones.

    BDH cold path passes QR twice; SelfAttnFn sums dQ+dK. Distinct-Q/K Function
    on equal clones must match after summing reference grads onto Q.
    """
    g = torch.Generator().manual_seed(99)
    Q = torch.randn(1, 2, 6, 4, generator=g, dtype=torch.float64)
    V = torch.randn(1, 1, 6, 5, generator=g, dtype=torch.float64)

    # Reference: distinct clones through three-arg Function, then sum dQ+dK
    Qc = Q.clone().requires_grad_(True)
    Kc = Q.clone().requires_grad_(True)
    Vc = V.clone().requires_grad_(True)
    Or = StrictTrilAttnFn.apply(Qc, Kc, Vc, "blocked")
    dO = torch.randn_like(Or)
    Or.backward(dO)
    dQ_ref = Qc.grad + Kc.grad

    Qs = Q.clone().requires_grad_(True)
    Vs = V.clone().requires_grad_(True)
    Os = StrictTrilSelfAttnFn.apply(Qs, Vs, "blocked")
    assert torch.allclose(Os, Or.detach(), rtol=1e-8, atol=1e-8)
    Os.backward(dO.clone())
    assert torch.allclose(Qs.grad, dQ_ref, rtol=1e-6, atol=1e-6)
    assert torch.allclose(Vs.grad, Vc.grad, rtol=1e-6, atol=1e-6)


def test_strict_tril_attn_routes_self_when_q_is_k():
    """strict_tril_attn(Q, Q, V, use_fn=True) uses SelfAttnFn path (no raise)."""
    Q = torch.randn(1, 1, 4, 3, dtype=torch.float64, requires_grad=True)
    V = torch.randn(1, 1, 4, 2, dtype=torch.float64, requires_grad=True)
    O = strict_tril_attn(Q, Q, V, impl="eager", use_fn=True)
    O.sum().backward()
    assert Q.grad is not None and V.grad is not None
    assert torch.isfinite(Q.grad).all()

@pytest.mark.parametrize("impl", ["eager", "blocked", "online", "triton", "cuda"])
@pytest.mark.parametrize("use_fn", [False, True])
@pytest.mark.parametrize("v_heads", [1, 3])
def test_cpu_impl_autograd_matrix_parity(impl, use_fn, v_heads):
    """CPU parity covers aliases, Function modes, and shared/full-head V layouts."""
    Q, K, V = _qkv(B=2, H=3, T=9, N=5, D=7, seed=31, v_heads=v_heads)
    Q = Q.float().requires_grad_(True)
    K = K.float().requires_grad_(True)
    V = V.float().requires_grad_(True)
    out = strict_tril_attn(Q, K, V, impl=impl, use_fn=use_fn)
    ref = eager_tril_attn(Q.detach(), K.detach(), V.detach())
    assert torch.allclose(out, ref, rtol=1e-4, atol=1e-4)
    dO = torch.randn_like(out)
    out.backward(dO)

    Qr = Q.detach().clone().requires_grad_(True)
    Kr = K.detach().clone().requires_grad_(True)
    Vr = V.detach().clone().requires_grad_(True)
    eager_tril_attn(Qr, Kr, Vr).backward(dO)
    assert torch.allclose(Q.grad, Qr.grad, rtol=1e-4, atol=1e-4)
    assert torch.allclose(K.grad, Kr.grad, rtol=1e-4, atol=1e-4)
    assert torch.allclose(V.grad, Vr.grad, rtol=1e-4, atol=1e-4)
