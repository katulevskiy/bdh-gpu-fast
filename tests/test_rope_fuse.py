"""Fused RoPE rotate vs eager / baseline; BDH_ROPE_IMPL dispatch."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import bdh
import bdh_baseline as baseline
from kernels.rope import (
    _HAS_TRITON,
    eager_rope_rotate,
    fused_rope_rotate_blocked,
    fused_rope_rotate_paired,
    fused_rope_rotate_pytorch,
    fused_rope_rotate_triton,
    _can_use_triton_rope,
)
from kernels.rope_dispatch import backend_info, bdh_rope_rotate, resolve_rope_impl


def _small_cfg(**kwargs) -> bdh.BDHConfig:
    defaults = dict(
        n_layer=2,
        n_embd=64,
        n_head=4,
        mlp_internal_dim_multiplier=8,
        dropout=0.0,
        vocab_size=256,
    )
    defaults.update(kwargs)
    return bdh.BDHConfig(**defaults)


def _cis_and_v(T=8, seed=0):
    cfg = _small_cfg()
    attn = bdh.Attention(cfg)
    N = cfg.mlp_internal_dim_multiplier * cfg.n_embd // cfg.n_head
    device = torch.device("cpu")
    cos, sin = attn.rope_cos_sin(T, 0, device)
    torch.manual_seed(seed)
    v = torch.randn(2, cfg.n_head, T, N)
    phases = attn._rope_phases(T, 0, device)
    return cfg, attn, cos, sin, v, phases


def test_resolve_rope_impl_default_eager(monkeypatch):
    monkeypatch.delenv("BDH_ROPE_IMPL", raising=False)
    assert resolve_rope_impl() == "eager"
    monkeypatch.setenv("BDH_ROPE_IMPL", "fused")
    assert resolve_rope_impl() == "fused"
    monkeypatch.setenv("BDH_ROPE_IMPL", "nope")
    with pytest.raises(ValueError, match="BDH_ROPE_IMPL"):
        resolve_rope_impl()


def test_fused_pytorch_bit_identical_to_eager():
    _, _, cos, sin, v, _ = _cis_and_v()
    a = eager_rope_rotate(v, cos, sin)
    b = fused_rope_rotate_pytorch(v, cos, sin)
    assert torch.equal(a, b)


def test_fused_pytorch_matches_baseline_rope():
    _, attn, cos, sin, v, phases = _cis_and_v(seed=3)
    out_base = baseline.Attention.rope(phases, v)
    out_fused = fused_rope_rotate_pytorch(v, cos, sin)
    assert torch.equal(out_fused, out_base)


def test_eager_dispatch_matches_baseline(monkeypatch):
    monkeypatch.setenv("BDH_ROPE_IMPL", "eager")
    _, _, cos, sin, v, phases = _cis_and_v(seed=5)
    out = bdh_rope_rotate(v, cos, sin)
    out_base = baseline.Attention.rope(phases, v)
    assert torch.equal(out, out_base)


def test_fused_dispatch_matches_eager_atol(monkeypatch):
    monkeypatch.setenv("BDH_ROPE_IMPL", "fused")
    _, _, cos, sin, v, _ = _cis_and_v(seed=7)
    out_e = bdh_rope_rotate(v, cos, sin, impl="eager")
    out_f = bdh_rope_rotate(v, cos, sin, impl="fused")
    # Same contiguous math → bit-identical on CPU; atol kept for half/GPU.
    assert torch.allclose(out_f, out_e, rtol=0, atol=0)


def test_attention_rope_respects_env(monkeypatch):
    monkeypatch.setenv("BDH_ROPE_IMPL", "fused")
    _, attn, cos, sin, v, phases = _cis_and_v(seed=9)
    out = bdh.Attention.rope(None, v, cos_sin=(cos, sin))
    out_base = baseline.Attention.rope(phases, v)
    assert torch.equal(out, out_base)


def test_rope_cos_sin_cache_still_reused_under_fused(monkeypatch):
    """Fused rotate must not break the (T, head_dim, device, dtype) cis cache."""
    monkeypatch.setenv("BDH_ROPE_IMPL", "fused")
    cfg = _small_cfg()
    attn = bdh.Attention(cfg)
    device = torch.device("cpu")
    cos1, sin1 = attn.rope_cos_sin(12, 0, device)
    cos2, sin2 = attn.rope_cos_sin(12, 0, device)
    assert cos1 is cos2 and sin1 is sin2


def test_fused_out_param_and_no_alias():
    _, _, cos, sin, v, _ = _cis_and_v(seed=11)
    out = torch.empty_like(v)
    y = fused_rope_rotate_pytorch(v, cos, sin, out=out)
    assert y is out
    assert torch.equal(out, eager_rope_rotate(v, cos, sin))
    with pytest.raises(ValueError, match="alias"):
        fused_rope_rotate_pytorch(v, cos, sin, out=v)


def test_fused_backward_matches_eager():
    _, _, cos, sin, v, _ = _cis_and_v(seed=13)
    ve = v.detach().requires_grad_(True)
    vf = v.detach().requires_grad_(True)
    ye = eager_rope_rotate(ve, cos, sin)
    yf = fused_rope_rotate_pytorch(vf, cos, sin)
    ye.sum().backward()
    yf.sum().backward()
    assert torch.equal(ve.grad, vf.grad)


def test_half_dtype_cast_path_matches():
    """fp32 cis + fp16 v: cast-then-add rounding matches eager."""
    cfg = _small_cfg()
    attn = bdh.Attention(cfg)
    T, N = 6, cfg.mlp_internal_dim_multiplier * cfg.n_embd // cfg.n_head
    cos, sin = attn.rope_cos_sin(T, 0, torch.device("cpu"))
    torch.manual_seed(15)
    v = torch.randn(1, cfg.n_head, T, N, dtype=torch.float16)
    a = eager_rope_rotate(v, cos, sin)
    b = fused_rope_rotate_pytorch(v, cos, sin)
    assert torch.equal(a, b)


@pytest.mark.skipif(
    not (torch.cuda.is_available() and _HAS_TRITON),
    reason="CUDA + Triton required for Triton RoPE kernel",
)
def test_triton_rope_matches_eager_cuda():
    cfg = _small_cfg()
    attn = bdh.Attention(cfg)
    T = 8
    N = cfg.mlp_internal_dim_multiplier * cfg.n_embd // cfg.n_head
    device = torch.device("cuda")
    cos, sin = attn.rope_cos_sin(T, 0, device)
    torch.manual_seed(17)
    v = torch.randn(2, cfg.n_head, T, N, device=device)
    assert _can_use_triton_rope(v)
    out_e = eager_rope_rotate(v, cos, sin)
    out_t = fused_rope_rotate_triton(v, cos, sin)
    assert torch.allclose(out_t, out_e, rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(
    not (torch.cuda.is_available() and _HAS_TRITON),
    reason="CUDA + Triton required for paired T=1 RoPE kernel",
)
def test_triton_paired_t1_matches_eager_cuda():
    cfg = _small_cfg()
    attn = bdh.Attention(cfg)
    device = torch.device("cuda")
    N = cfg.mlp_internal_dim_multiplier * cfg.n_embd // cfg.n_head
    rope_start = 5
    attn.ensure_rope_table(16, device)
    paired = attn.t1_cis_pairs(rope_start, device)
    cos, sin = attn.rope_cos_sin(1, rope_start, device)
    assert paired is not None
    torch.manual_seed(18)
    v = torch.randn(2, cfg.n_head, 1, N, device=device)
    out_e = eager_rope_rotate(v, cos, sin)
    out_t = fused_rope_rotate_paired(v, paired[0], paired[1])
    assert torch.allclose(out_t, out_e, rtol=1e-5, atol=1e-5)


def test_default_impl_is_eager_no_env(monkeypatch):
    monkeypatch.delenv("BDH_ROPE_IMPL", raising=False)
    # Attention.rope with default env must match baseline bit-identically
    _, _, cos, sin, v, phases = _cis_and_v(seed=19)
    out = bdh.Attention.rope(None, v, cos_sin=(cos, sin))
    assert torch.equal(out, baseline.Attention.rope(phases, v))
    assert resolve_rope_impl() == "eager"


def test_fused_pytorch_no_expand_matches_eager_broadcast_cis():
    """Cold T>1: cis (1,1,T,N) broadcasts without expand(v.shape)."""
    _, _, cos, sin, v, _ = _cis_and_v(T=16, seed=21)
    assert cos.shape[0] == 1 and v.shape[0] == 2
    a = eager_rope_rotate(v, cos, sin)
    b = fused_rope_rotate_pytorch(v, cos, sin)
    assert torch.equal(a, b)


def test_blocked_tile_parity_vs_eager_and_fused():
    """CPU blocked (Triton tile scaffold) ≡ eager ≡ fused pytorch."""
    _, _, cos, sin, v, phases = _cis_and_v(T=12, seed=23)
    ref = eager_rope_rotate(v, cos, sin)
    blocked = fused_rope_rotate_blocked(v, cos, sin, block=8)
    fused = fused_rope_rotate_pytorch(v, cos, sin)
    assert torch.equal(blocked, ref)
    assert torch.equal(fused, ref)
    # Tiny block + large block
    assert torch.equal(fused_rope_rotate_blocked(v, cos, sin, block=1), ref)
    assert torch.equal(fused_rope_rotate_blocked(v, cos, sin, block=128), ref)
    # Baseline phases path
    assert torch.equal(blocked, baseline.Attention.rope(phases, v))


def test_triton_entry_cpu_falls_back_to_blocked(monkeypatch):
    """fused_rope_rotate_triton on CPU uses blocked scaffold (parity)."""
    monkeypatch.setenv("BDH_ROPE_IMPL", "fused")
    _, _, cos, sin, v, _ = _cis_and_v(T=8, seed=25)
    assert not _can_use_triton_rope(v)
    out_t = fused_rope_rotate_triton(v, cos, sin)
    out_b = fused_rope_rotate_blocked(v, cos, sin)
    out_e = eager_rope_rotate(v, cos, sin)
    assert torch.equal(out_t, out_b)
    assert torch.equal(out_t, out_e)


def test_triton_gate_and_backend_info_are_cpu_safe():
    """Unavailable CUDA or mismatched auxiliary tensors cleanly skip Triton."""
    _, _, cos, sin, v, _ = _cis_and_v(T=8, seed=26)
    assert not _can_use_triton_rope(v, cos, sin)
    info = backend_info(torch.device("cuda"))
    assert info["device"] == "cuda"
    assert info["triton_usable"] is False


def test_t_gt1_noncontiguous_cpu_parity_and_out():
    """T>1 pair views preserve parity for transposed leading dimensions."""
    _, _, cos, sin, v, _ = _cis_and_v(T=7, seed=27)
    v_nc = v.transpose(1, 2)
    cos_nc = cos.transpose(1, 2)
    sin_nc = sin.transpose(1, 2)
    ref = eager_rope_rotate(v_nc, cos_nc, sin_nc)
    out = torch.empty_like(v_nc)
    got = fused_rope_rotate_blocked(v_nc, cos_nc, sin_nc, out=out, block=3)
    assert got is out
    assert torch.equal(got, ref)
    assert torch.equal(fused_rope_rotate_pytorch(v_nc, cos_nc, sin_nc), ref)


def test_fused_out_none_pair_store_parity():
    _, _, cos, sin, v, _ = _cis_and_v(T=10, seed=27)
    # Explicit out=None path
    y = fused_rope_rotate_pytorch(v, cos, sin, out=None)
    assert torch.equal(y, eager_rope_rotate(v, cos, sin))


def test_fused_paired_t1_cpu_parity():
    cfg = _small_cfg()
    attn = bdh.Attention(cfg)
    device = torch.device("cpu")
    N = cfg.mlp_internal_dim_multiplier * cfg.n_embd // cfg.n_head
    rope_start = 4
    attn.ensure_rope_table(16, device)
    paired = attn.t1_cis_pairs(rope_start, device)
    cos, sin = attn.rope_cos_sin(1, rope_start, device)
    assert paired is not None
    torch.manual_seed(28)
    v = torch.randn(2, cfg.n_head, 1, N)
    ref = eager_rope_rotate(v, cos, sin)
    assert torch.equal(fused_rope_rotate_paired(v, paired[0], paired[1]), ref)


def test_blocked_out_param_and_t1():
    cfg = _small_cfg()
    attn = bdh.Attention(cfg)
    device = torch.device("cpu")
    N = cfg.mlp_internal_dim_multiplier * cfg.n_embd // cfg.n_head
    attn.ensure_rope_table(16, device)
    cos, sin = attn.rope_cos_sin(1, 5, device)
    torch.manual_seed(29)
    v = torch.randn(2, cfg.n_head, 1, N)
    out = torch.empty_like(v)
    y = fused_rope_rotate_blocked(v, cos, sin, out=out, block=16)
    assert y is out
    assert torch.equal(out, eager_rope_rotate(v, cos, sin))


def test_generate_cache_continuity_fused_and_eager(monkeypatch):
    """Generate tokens match baseline under both rope impls (cache phases)."""
    cfg = _small_cfg(n_layer=2)
    prompt = torch.randint(0, cfg.vocab_size, (1, 5))
    for impl in ("eager", "fused"):
        monkeypatch.setenv("BDH_ROPE_IMPL", impl)
        torch.manual_seed(0)
        m = bdh.BDH(cfg).eval()
        torch.manual_seed(0)
        b = baseline.BDH(cfg).eval()
        b.load_state_dict(m.state_dict())
        torch.manual_seed(99)
        out_m = m.generate(prompt.clone(), max_new_tokens=6, temperature=1.0)
        torch.manual_seed(99)
        out_b = b.generate(prompt.clone(), max_new_tokens=6, temperature=1.0)
        assert torch.equal(out_m, out_b), impl
