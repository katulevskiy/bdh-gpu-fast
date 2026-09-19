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
import kernels.rope_dispatch as rope_dispatch
from kernels.rope import (
    _HAS_TRITON,
    eager_rope_rotate,
    triton_rope_skip_reason,
    fused_rope_rotate_blocked,
    fused_rope_rotate_paired,
    fused_rope_rotate_pytorch,
    fused_rope_rotate_triton,
    _can_use_triton_rope,
)
from kernels.rope_dispatch import (
    backend_info,
    bdh_rope_rotate,
    bdh_rope_rotate_paired,
    resolve_rope_impl,
)


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


@pytest.mark.parametrize("entrypoint", [bdh_rope_rotate, bdh_rope_rotate_paired])
def test_public_dispatch_rejects_invalid_explicit_impl_before_input_validation(
    entrypoint,
):
    """Explicit backend overrides fail before touching malformed tensor inputs."""
    with pytest.raises(ValueError, match="BDH_ROPE_IMPL"):
        entrypoint(None, None, None, impl="triton")


@pytest.mark.parametrize(
    ("entrypoint", "eager_backend", "fused_backend"),
    [
        (bdh_rope_rotate, "eager_rope_rotate", "fused_rope_rotate"),
        (bdh_rope_rotate_paired, "rope_rotate_paired", "fused_rope_rotate_paired"),
    ],
)
@pytest.mark.parametrize(
    ("env_impl", "explicit_impl"),
    [("fused", "eager"), ("eager", "fused")],
)
def test_public_dispatch_explicit_impl_overrides_env(
    monkeypatch, entrypoint, eager_backend, fused_backend, env_impl, explicit_impl
):
    """Explicit backend selection wins over BDH_ROPE_IMPL for both entrypoints."""
    monkeypatch.setenv("BDH_ROPE_IMPL", env_impl)
    calls = []

    def mark_eager(v, *args, **kwargs):
        calls.append("eager")
        return v

    def mark_fused(v, *args, **kwargs):
        calls.append("fused")
        return v

    monkeypatch.setattr(rope_dispatch, eager_backend, mark_eager)
    monkeypatch.setattr(rope_dispatch, fused_backend, mark_fused)

    _, _, cos, sin, v, _ = _cis_and_v(
        T=1 if entrypoint is bdh_rope_rotate_paired else 7,
        seed=118 if entrypoint is bdh_rope_rotate_paired else 119,
    )
    if entrypoint is bdh_rope_rotate_paired:
        cos = cos.reshape(*cos.shape[:-1], -1, 2)
        sin = sin.reshape(*sin.shape[:-1], -1, 2)

    entrypoint(v, cos, sin, impl=explicit_impl)
    assert calls == [explicit_impl]


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

    view_alias = v.view_as(v)
    with pytest.raises(ValueError, match="alias"):
        fused_rope_rotate_pytorch(v, cos, sin, out=view_alias)

    backing = torch.empty(*v.shape[:-1], v.shape[-1] + 1)
    shifted_v = backing[..., :-1]
    shifted_out = backing[..., 1:]
    with pytest.raises(ValueError, match="alias"):
        fused_rope_rotate_pytorch(shifted_v, cos, sin, out=shifted_out)


def test_rope_out_param_rejects_cis_aliases_before_pair_store():
    """Output slots cannot overwrite broadcasted cosine/sine storage."""
    _, _, cos, sin, v, _ = _cis_and_v(T=4, seed=114)

    for cis in (cos, sin):
        out = cis.expand_as(v)
        before = out.clone()
        with pytest.raises(ValueError, match="alias"):
            fused_rope_rotate_blocked(v, cos, sin, out=out, block=1)
        assert torch.equal(out, before)


def test_paired_out_param_rejects_cis_aliases_before_pair_store():
    """Paired output slots cannot overwrite the cached cis pair views."""
    _, _, cos, sin, v, _ = _cis_and_v(T=1, seed=115)
    cos_p = cos.reshape(*cos.shape[:-1], -1, 2)
    sin_p = sin.reshape(*sin.shape[:-1], -1, 2)

    for cis_p in (cos_p, sin_p):
        out = cis_p.reshape(*cis_p.shape[:-2], -1).expand_as(v)
        before = out.clone()
        with pytest.raises(ValueError, match="alias"):
            fused_rope_rotate_paired(v, cos_p, sin_p, out=out)
        assert torch.equal(out, before)


@pytest.mark.parametrize("impl", ["eager", "fused"])
def test_public_dispatch_rejects_cis_aliases_before_store(impl):
    """Public T>1 dispatchers keep the no-cis-alias output contract."""
    _, _, cos, sin, v, _ = _cis_and_v(T=4, seed=116)
    for cis in (cos, sin):
        out = cis.expand_as(v)
        before = out.clone()
        with pytest.raises(ValueError, match="alias"):
            bdh_rope_rotate(v, cos, sin, out=out, impl=impl)
        assert torch.equal(out, before), impl


@pytest.mark.parametrize("impl", ["eager", "fused"])
def test_public_paired_dispatch_rejects_cis_aliases_before_store(impl):
    """Public paired dispatch keeps the no-cis-alias output contract."""
    _, _, cos, sin, v, _ = _cis_and_v(T=1, seed=117)
    cos_p = cos.reshape(*cos.shape[:-1], -1, 2)
    sin_p = sin.reshape(*sin.shape[:-1], -1, 2)

    for cis_p in (cos_p, sin_p):
        out = cis_p.reshape(*cis_p.shape[:-2], -1).expand_as(v)
        before = out.clone()
        with pytest.raises(ValueError, match="alias"):
            bdh_rope_rotate_paired(v, cos_p, sin_p, out=out, impl=impl)
        assert torch.equal(out, before), impl


def test_rope_shape_contracts_reject_malformed_inputs():
    """Rotation entrypoints fail clearly before doing partial math."""
    _, _, cos, sin, v, _ = _cis_and_v(T=2, seed=111)

    with pytest.raises(ValueError, match="must be even"):
        fused_rope_rotate_pytorch(v[..., :3], cos[..., :3], sin[..., :3])
    with pytest.raises(ValueError, match="shape mismatch"):
        fused_rope_rotate_pytorch(v, cos, sin[..., :-1])
    with pytest.raises(ValueError, match="last dim"):
        fused_rope_rotate_pytorch(v, cos[..., :-2], sin[..., :-2])
    with pytest.raises(ValueError, match="rope out"):
        fused_rope_rotate_pytorch(v, cos, sin, out=torch.empty_like(v[..., :-1]))


def test_paired_rope_shape_contract_rejects_malformed_inputs():
    """The paired T=1 contract validates its pair-axis geometry too."""
    _, _, cos, sin, v, _ = _cis_and_v(T=1, seed=112)
    cos_p = cos.reshape(*cos.shape[:-1], -1, 2)
    sin_p = sin.reshape(*sin.shape[:-1], -1, 2)

    with pytest.raises(ValueError, match="trailing dims"):
        fused_rope_rotate_paired(v, cos_p[..., :-1, :], sin_p[..., :-1, :])
    with pytest.raises(ValueError, match="shape mismatch"):
        fused_rope_rotate_paired(v, cos_p, sin_p[..., :-1, :])
    with pytest.raises(ValueError, match="must be even"):
        fused_rope_rotate_paired(v[..., :3], cos_p, sin_p)
    with pytest.raises(ValueError, match="alias"):
        fused_rope_rotate_paired(v, cos_p, sin_p, out=v)


def test_paired_rope_out_shape_contract_rejects_before_store():
    """Paired output slots must match v before pair stores begin."""
    _, _, cos, sin, v, _ = _cis_and_v(T=1, seed=118)
    cos_p = cos.reshape(*cos.shape[:-1], -1, 2)
    sin_p = sin.reshape(*sin.shape[:-1], -1, 2)
    out = torch.full((*v.shape[:-1], v.shape[-1] - 2), -77.0)
    before = out.clone()

    with pytest.raises(ValueError, match="rope out"):
        fused_rope_rotate_paired(v, cos_p, sin_p, out=out)
    assert torch.equal(out, before)


def test_rope_shape_contracts_reject_unbroadcastable_cis():
    """Cis leading dimensions must broadcast before pair math starts."""
    _, _, cos, sin, v, _ = _cis_and_v(T=2, seed=113)
    n = v.shape[-1]
    bad_cos = torch.empty(3, 1, 2, n)
    bad_sin = torch.empty_like(bad_cos)

    with pytest.raises(ValueError, match="not broadcastable"):
        fused_rope_rotate_pytorch(v, bad_cos, bad_sin)

    bad_cos_p = bad_cos.reshape(*bad_cos.shape[:-1], -1, 2)
    bad_sin_p = bad_sin.reshape(*bad_sin.shape[:-1], -1, 2)
    with pytest.raises(ValueError, match="not broadcastable"):
        fused_rope_rotate_paired(v, bad_cos_p, bad_sin_p)



@pytest.mark.parametrize("T", [1, 7])
def test_cpu_rope_out_param_mixed_dtype_preserves_parity(T):
    """CPU RoPE entries cast fp16 math into fp32 cache-style output slots."""
    cfg = _small_cfg()
    attn = bdh.Attention(cfg)
    N = cfg.mlp_internal_dim_multiplier * cfg.n_embd // cfg.n_head
    cos, sin = attn.rope_cos_sin(T, 0, torch.device("cpu"))
    torch.manual_seed(12 + T)
    v = torch.randn(2, cfg.n_head, T, N, dtype=torch.float16)
    assert cos.dtype == torch.float32 and sin.dtype == torch.float32

    ref_out = torch.empty_like(v, dtype=torch.float32)
    ref = eager_rope_rotate(v, cos, sin, out=ref_out)
    fused_out = torch.empty_like(ref_out)
    fused = fused_rope_rotate_pytorch(v, cos, sin, out=fused_out)
    blocked_out = torch.empty_like(ref_out)
    blocked = fused_rope_rotate_blocked(v, cos, sin, out=blocked_out, block=3)
    triton_out = torch.empty_like(ref_out)
    triton = fused_rope_rotate_triton(v, cos, sin, out=triton_out)

    assert ref is ref_out and fused is fused_out
    assert blocked is blocked_out and triton is triton_out
    assert all(result.dtype == torch.float32 for result in (ref, fused, blocked, triton))
    assert torch.equal(fused, ref)
    assert torch.equal(blocked, ref)
    assert torch.equal(triton, ref)


@pytest.mark.parametrize("T", [1, 7])
def test_cpu_rope_mixed_dtype_strided_out_preserves_parity(T):
    """fp16 RoPE writes only its non-contiguous fp32 cache slot."""
    cfg = _small_cfg()
    attn = bdh.Attention(cfg)
    N = cfg.mlp_internal_dim_multiplier * cfg.n_embd // cfg.n_head
    cos, sin = attn.rope_cos_sin(T, 0, torch.device("cpu"))
    torch.manual_seed(120 + T)
    v = torch.randn(2, cfg.n_head, T, N, dtype=torch.float16)
    sentinel = torch.tensor(-123.0, dtype=torch.float32)

    def make_out():
        backing = torch.full((*v.shape[:-1], N * 2), sentinel.item())
        return backing, backing[..., ::2]

    ref_backing, ref_out = make_out()
    ref = eager_rope_rotate(v, cos, sin, out=ref_out)
    assert ref is ref_out and not ref.is_contiguous()

    paths = (
        ("pytorch", fused_rope_rotate_pytorch, {}),
        ("blocked", fused_rope_rotate_blocked, {"block": 3}),
        ("triton", fused_rope_rotate_triton, {}),
    )
    for name, rotate, kwargs in paths:
        backing, out = make_out()
        got = rotate(v, cos, sin, out=out, **kwargs)
        assert got is out, name
        assert got.dtype == torch.float32 and not got.is_contiguous(), name
        assert torch.equal(got, ref), name
        assert torch.equal(backing[..., 1::2], ref_backing[..., 1::2]), name


def test_fused_backward_matches_eager():
    _, _, cos, sin, v, _ = _cis_and_v(seed=13)
    ve = v.detach().requires_grad_(True)
    vf = v.detach().requires_grad_(True)
    ye = eager_rope_rotate(ve, cos, sin)
    yf = fused_rope_rotate_pytorch(vf, cos, sin)
    ye.sum().backward()
    yf.sum().backward()
    assert torch.equal(ve.grad, vf.grad)


def test_fused_backward_matches_eager_for_arbitrary_upstream_gradient():
    """Fused T>1 backward preserves pair-specific upstream gradients."""
    _, _, cos, sin, v, _ = _cis_and_v(T=7, seed=131)
    ve = v.detach().requires_grad_(True)
    vf = v.detach().requires_grad_(True)
    ye = eager_rope_rotate(ve, cos, sin)
    yf = fused_rope_rotate_pytorch(vf, cos, sin)
    torch.manual_seed(132)
    grad = torch.randn_like(ye)
    ye.backward(grad)
    yf.backward(grad)
    assert torch.equal(ve.grad, vf.grad)


@pytest.mark.parametrize(
    ("name", "rotate", "kwargs"),
    [
        ("blocked", fused_rope_rotate_blocked, {"block": 3}),
        ("triton-cpu-fallback", fused_rope_rotate_triton, {}),
    ],
)
def test_cpu_fused_entrypoints_backward_match_eager_for_arbitrary_upstream_gradient(
    name, rotate, kwargs
):
    """CPU-safe fused entries preserve pair-specific upstream gradients."""
    _, _, cos, sin, v, _ = _cis_and_v(T=7, seed=133)
    ve = v.detach().requires_grad_(True)
    vf = v.detach().requires_grad_(True)
    ye = eager_rope_rotate(ve, cos, sin)
    yf = rotate(vf, cos, sin, **kwargs)
    torch.manual_seed(134)
    grad = torch.randn_like(ye)
    ye.backward(grad)
    yf.backward(grad)
    assert torch.equal(ve.grad, vf.grad), name


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


def test_triton_skip_reason_rejects_non_tensor_before_runtime_probe():
    """Invalid primary inputs get a stable contract reason on every host."""
    assert triton_rope_skip_reason(None) == "v-not-tensor"
    assert _can_use_triton_rope(None) is False


def test_triton_skip_reason_is_actionable_on_cpu():
    """The scaffold reports why CPU inputs stay on the safe fallback."""
    _, _, cos, sin, v, _ = _cis_and_v(T=8, seed=261)
    reason = triton_rope_skip_reason(v, cos, sin)
    assert reason in {"triton-not-installed", "v-not-cuda"}
    assert _can_use_triton_rope(v, cos, sin) is False
    info = backend_info(torch.device("cpu"))
    assert info["triton_usable"] is False
    assert info["triton_skip_reason"] in {"triton-not-installed", "device-not-cuda"}


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


def test_fused_strided_pair_input_and_out_preserve_parity():
    """Pair views preserve RoPE parity when both input and cache slot stride."""
    _, _, cos, sin, v, _ = _cis_and_v(T=7, seed=271)
    sentinel = torch.tensor(-321.0)

    input_backing = torch.full((*v.shape[:-1], v.shape[-1] * 2), sentinel.item())
    v_strided = input_backing[..., ::2]
    v_strided.copy_(v)
    ref = eager_rope_rotate(v_strided, cos, sin)
    assert not v_strided.is_contiguous()

    paths = (
        ("pytorch", fused_rope_rotate_pytorch, {}),
        ("blocked", fused_rope_rotate_blocked, {"block": 3}),
        ("triton-cpu-fallback", fused_rope_rotate_triton, {}),
    )
    for name, rotate, kwargs in paths:
        out_backing = torch.full(
            (*v.shape[:-1], v.shape[-1] * 2), sentinel.item()
        )
        out = out_backing[..., ::2]
        got = rotate(v_strided, cos, sin, out=out, **kwargs)
        assert got is out and not got.is_contiguous(), name
        assert torch.equal(got, ref), name
        assert torch.equal(
            out_backing[..., 1::2],
            torch.full_like(out_backing[..., 1::2], sentinel),
        ), name


@pytest.mark.parametrize("impl", ["eager", "fused"])
def test_public_dispatch_noncontiguous_t_gt1_backward_matches_eager(impl):
    """Public T>1 dispatch preserves input gradients for strided V views."""
    _, _, cos, sin, v, _ = _cis_and_v(T=7, seed=294)

    def make_strided_input():
        backing = torch.full((*v.shape[:-1], v.shape[-1] * 2), -321.0)
        backing[..., ::2].copy_(v)
        backing.requires_grad_()
        return backing, backing[..., ::2]

    eager_backing, eager_v = make_strided_input()
    impl_backing, impl_v = make_strided_input()
    eager = eager_rope_rotate(eager_v, cos, sin)
    got = bdh_rope_rotate(impl_v, cos, sin, impl=impl)
    torch.manual_seed(295)
    grad = torch.randn_like(eager)
    eager.backward(grad)
    got.backward(grad)

    assert torch.equal(impl_backing.grad, eager_backing.grad), impl


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


def test_fused_paired_t1_mixed_dtype_out_preserves_parity():
    """Paired decode casts fp16 math into fp32 cache-style output slots."""
    cfg = _small_cfg()
    attn = bdh.Attention(cfg)
    device = torch.device("cpu")
    N = cfg.mlp_internal_dim_multiplier * cfg.n_embd // cfg.n_head
    rope_start = 6
    attn.ensure_rope_table(16, device)
    paired = attn.t1_cis_pairs(rope_start, device)
    cos, sin = attn.rope_cos_sin(1, rope_start, device)
    assert paired is not None
    torch.manual_seed(283)
    v = torch.randn(2, cfg.n_head, 1, N, dtype=torch.float16)
    ref_out = torch.empty_like(v, dtype=torch.float32)
    ref = eager_rope_rotate(v, cos, sin, out=ref_out)
    fused_out = torch.empty_like(ref_out)
    got = fused_rope_rotate_paired(v, paired[0], paired[1], out=fused_out)

    assert ref is ref_out and got is fused_out
    assert got.dtype == torch.float32
    assert torch.equal(got, ref)


def test_fused_paired_t1_mixed_dtype_strided_out_preserves_parity():
    """Paired decode preserves mixed-dtype parity in a strided cache slot."""
    cfg = _small_cfg()
    attn = bdh.Attention(cfg)
    device = torch.device("cpu")
    N = cfg.mlp_internal_dim_multiplier * cfg.n_embd // cfg.n_head
    rope_start = 6
    attn.ensure_rope_table(16, device)
    paired = attn.t1_cis_pairs(rope_start, device)
    cos, sin = attn.rope_cos_sin(1, rope_start, device)
    assert paired is not None
    torch.manual_seed(284)
    v = torch.randn(2, cfg.n_head, 1, N, dtype=torch.float16)

    sentinel = torch.tensor(-123.0, dtype=torch.float32)

    def make_out():
        backing = torch.full((*v.shape[:-1], N * 2), sentinel.item())
        return backing, backing[..., ::2]

    ref_backing, ref_out = make_out()
    ref = eager_rope_rotate(v, cos, sin, out=ref_out)
    fused_backing, fused_out = make_out()
    got = fused_rope_rotate_paired(
        v, paired[0], paired[1], out=fused_out
    )

    assert ref is ref_out and got is fused_out
    assert got.dtype == torch.float32 and not got.is_contiguous()
    assert torch.equal(got, ref)
    assert torch.equal(fused_backing[..., 1::2], ref_backing[..., 1::2])


@pytest.mark.parametrize("impl", ["eager", "fused"])
def test_paired_dispatch_mixed_dtype_strided_out_preserves_parity(impl):
    """The public paired dispatcher preserves the v15 cache-slot contract."""
    cfg = _small_cfg()
    attn = bdh.Attention(cfg)
    device = torch.device("cpu")
    N = cfg.mlp_internal_dim_multiplier * cfg.n_embd // cfg.n_head
    rope_start = 6
    attn.ensure_rope_table(16, device)
    paired = attn.t1_cis_pairs(rope_start, device)
    cos, sin = attn.rope_cos_sin(1, rope_start, device)
    assert paired is not None
    torch.manual_seed(285)
    v = torch.randn(2, cfg.n_head, 1, N, dtype=torch.float16)

    sentinel = torch.tensor(-123.0, dtype=torch.float32)

    def make_out():
        backing = torch.full((*v.shape[:-1], N * 2), sentinel.item())
        return backing, backing[..., ::2]

    ref_backing, ref_out = make_out()
    ref = eager_rope_rotate(v, cos, sin, out=ref_out)
    got_backing, got_out = make_out()
    got = bdh_rope_rotate_paired(
        v, paired[0], paired[1], out=got_out, impl=impl
    )

    assert ref is ref_out and got is got_out
    assert got.dtype == torch.float32 and not got.is_contiguous()
    assert torch.equal(got, ref), impl
    assert torch.equal(got_backing[..., 1::2], ref_backing[..., 1::2]), impl


@pytest.mark.parametrize("impl", ["eager", "fused"])
def test_dispatch_mixed_dtype_strided_out_preserves_parity(impl):
    """The public T>1 dispatcher preserves the cache-slot contract."""
    cfg = _small_cfg()
    attn = bdh.Attention(cfg)
    device = torch.device("cpu")
    N = cfg.mlp_internal_dim_multiplier * cfg.n_embd // cfg.n_head
    T = 7
    cos, sin = attn.rope_cos_sin(T, 0, device)
    torch.manual_seed(286)
    v = torch.randn(2, cfg.n_head, T, N, dtype=torch.float16)

    sentinel = torch.tensor(-123.0, dtype=torch.float32)

    def make_out():
        backing = torch.full((*v.shape[:-1], N * 2), sentinel.item())
        return backing, backing[..., ::2]

    ref_backing, ref_out = make_out()
    ref = eager_rope_rotate(v, cos, sin, out=ref_out)
    got_backing, got_out = make_out()
    got = bdh_rope_rotate(v, cos, sin, out=got_out, impl=impl)

    assert ref is ref_out and got is got_out
    assert got.dtype == torch.float32 and not got.is_contiguous()
    assert torch.equal(got, ref), impl
    assert torch.equal(got_backing[..., 1::2], ref_backing[..., 1::2]), impl


@pytest.mark.parametrize("impl", ["eager", "fused"])
def test_dispatch_t1_mixed_dtype_strided_out_preserves_parity(impl):
    """The public T=1 dispatcher preserves the cache-slot contract."""
    cfg = _small_cfg()
    attn = bdh.Attention(cfg)
    device = torch.device("cpu")
    N = cfg.mlp_internal_dim_multiplier * cfg.n_embd // cfg.n_head
    rope_start = 6
    attn.ensure_rope_table(16, device)
    cos, sin = attn.rope_cos_sin(1, rope_start, device)
    torch.manual_seed(287)
    v = torch.randn(2, cfg.n_head, 1, N, dtype=torch.float16)

    sentinel = torch.tensor(-123.0, dtype=torch.float32)

    def make_out():
        backing = torch.full((*v.shape[:-1], N * 2), sentinel.item())
        return backing, backing[..., ::2]

    ref_backing, ref_out = make_out()
    ref = eager_rope_rotate(v, cos, sin, out=ref_out)
    got_backing, got_out = make_out()
    got = bdh_rope_rotate(v, cos, sin, out=got_out, impl=impl)

    assert ref is ref_out and got is got_out
    assert got.dtype == torch.float32 and not got.is_contiguous()
    assert torch.equal(got, ref), impl
    assert torch.equal(got_backing[..., 1::2], ref_backing[..., 1::2]), impl


def test_fused_paired_t1_strided_out_preserves_parity():
    """Paired decode writes only the requested non-contiguous cache slot."""
    cfg = _small_cfg()
    attn = bdh.Attention(cfg)
    device = torch.device("cpu")
    N = cfg.mlp_internal_dim_multiplier * cfg.n_embd // cfg.n_head
    rope_start = 5
    attn.ensure_rope_table(16, device)
    paired = attn.t1_cis_pairs(rope_start, device)
    cos, sin = attn.rope_cos_sin(1, rope_start, device)
    assert paired is not None
    torch.manual_seed(281)
    v = torch.randn(2, cfg.n_head, 1, N)

    sentinel = torch.tensor(-123.0)
    backing = torch.full((*v.shape[:-1], N * 2), sentinel.item())
    out = backing[..., ::2]
    ref = eager_rope_rotate(v, cos, sin)
    got = fused_rope_rotate_paired(v, paired[0], paired[1], out=out)

    assert got is out and not got.is_contiguous()
    assert torch.equal(got, ref)
    assert torch.equal(
        backing[..., 1::2], torch.full_like(backing[..., 1::2], sentinel)
    )


def test_fused_paired_t1_strided_pair_input_and_out_preserve_parity():
    """Paired decode preserves parity with strided input and cache slots."""
    cfg = _small_cfg()
    attn = bdh.Attention(cfg)
    device = torch.device("cpu")
    N = cfg.mlp_internal_dim_multiplier * cfg.n_embd // cfg.n_head
    rope_start = 5
    attn.ensure_rope_table(16, device)
    paired = attn.t1_cis_pairs(rope_start, device)
    cos, sin = attn.rope_cos_sin(1, rope_start, device)
    assert paired is not None
    torch.manual_seed(292)
    v = torch.randn(2, cfg.n_head, 1, N)
    sentinel = torch.tensor(-321.0)

    input_backing = torch.full((*v.shape[:-1], N * 2), sentinel.item())
    v_strided = input_backing[..., ::2]
    v_strided.copy_(v)
    assert not v_strided.is_contiguous()
    ref = eager_rope_rotate(v_strided, cos, sin)

    out_backing = torch.full((*v.shape[:-1], N * 2), sentinel.item())
    out = out_backing[..., ::2]
    got = fused_rope_rotate_paired(v_strided, paired[0], paired[1], out=out)

    assert got is out and not got.is_contiguous()
    assert torch.equal(got, ref)
    assert torch.equal(
        out_backing[..., 1::2], torch.full_like(out_backing[..., 1::2], sentinel)
    )


def test_fused_paired_t1_backward_matches_eager():
    """Paired T=1 CPU fallback preserves eager input gradients."""
    cfg = _small_cfg()
    attn = bdh.Attention(cfg)
    device = torch.device("cpu")
    N = cfg.mlp_internal_dim_multiplier * cfg.n_embd // cfg.n_head
    rope_start = 7
    attn.ensure_rope_table(16, device)
    paired = attn.t1_cis_pairs(rope_start, device)
    cos, sin = attn.rope_cos_sin(1, rope_start, device)
    assert paired is not None
    torch.manual_seed(282)
    v = torch.randn(2, cfg.n_head, 1, N)
    ve = v.detach().requires_grad_(True)
    vf = v.detach().requires_grad_(True)
    eager = eager_rope_rotate(ve, cos, sin)
    fused = fused_rope_rotate_paired(vf, paired[0], paired[1])
    grad = torch.randn_like(eager)
    eager.backward(grad)
    fused.backward(grad)
    assert torch.equal(vf.grad, ve.grad)


@pytest.mark.parametrize("impl", ["eager", "fused"])
def test_public_paired_t1_backward_matches_eager_for_strided_input(impl):
    """Public paired decode preserves input gradients for strided V views."""
    cfg = _small_cfg()
    attn = bdh.Attention(cfg)
    device = torch.device("cpu")
    N = cfg.mlp_internal_dim_multiplier * cfg.n_embd // cfg.n_head
    rope_start = 7
    attn.ensure_rope_table(16, device)
    paired = attn.t1_cis_pairs(rope_start, device)
    cos, sin = attn.rope_cos_sin(1, rope_start, device)
    assert paired is not None
    torch.manual_seed(293)
    v = torch.randn(2, cfg.n_head, 1, N)

    def make_strided_input():
        backing = torch.full((*v.shape[:-1], N * 2), -321.0)
        backing[..., ::2].copy_(v)
        backing.requires_grad_()
        return backing, backing[..., ::2]

    eager_backing, eager_v = make_strided_input()
    fused_backing, fused_v = make_strided_input()
    assert not eager_v.is_contiguous() and not fused_v.is_contiguous()
    eager = eager_rope_rotate(eager_v, cos, sin)
    fused = bdh_rope_rotate_paired(
        fused_v, paired[0], paired[1], impl=impl
    )
    grad = torch.randn_like(eager)
    eager.backward(grad)
    fused.backward(grad)
    assert torch.equal(fused_backing.grad, eager_backing.grad), impl


def test_t1_paired_cache_refreshes_across_positions(monkeypatch):
    """Paired decode cis follows each absolute position on the CPU path."""
    cfg = _small_cfg()
    attn = bdh.Attention(cfg)
    device = torch.device("cpu")
    N = cfg.mlp_internal_dim_multiplier * cfg.n_embd // cfg.n_head
    attn.ensure_rope_table(16, device)
    torch.manual_seed(281)
    v = torch.randn(2, cfg.n_head, 1, N)

    for impl in ("eager", "fused"):
        monkeypatch.setenv("BDH_ROPE_IMPL", impl)
        for rope_start in (2, 11, 2):
            cos, sin = attn.rope_cos_sin(1, rope_start, device)
            paired = attn.t1_cis_pairs(rope_start, device)
            assert paired is not None
            ref = eager_rope_rotate(v, cos, sin)
            got = bdh_rope_rotate_paired(
                v, paired[0], paired[1], impl=impl
            )
            assert torch.equal(got, ref), (impl, rope_start)


@pytest.mark.parametrize("T", [1, 12])
def test_blocked_rejects_nonpositive_block(T):
    """The blocked contract rejects invalid tiles before its T=1 shortcut."""
    _, _, cos, sin, v, _ = _cis_and_v(T=T, seed=30 + T)
    with pytest.raises(ValueError, match="block must be >= 1"):
        fused_rope_rotate_blocked(v, cos, sin, block=0)


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


def test_t1_noncontiguous_out_buffer_preserves_parity():
    """T=1 paired stores honor a strided cache-slot-like output buffer."""
    cfg = _small_cfg()
    attn = bdh.Attention(cfg)
    device = torch.device("cpu")
    N = cfg.mlp_internal_dim_multiplier * cfg.n_embd // cfg.n_head
    attn.ensure_rope_table(16, device)
    cos, sin = attn.rope_cos_sin(1, 5, device)
    torch.manual_seed(291)
    v = torch.randn(2, cfg.n_head, 1, N)
    backing = torch.empty(*v.shape[:-1], N * 2)
    out = backing[..., ::2]
    ref = eager_rope_rotate(v, cos, sin)
    got = fused_rope_rotate_blocked(v, cos, sin, out=out, block=3)
    assert got is out
    assert not out.is_contiguous()
    assert torch.equal(out, ref)


@pytest.mark.parametrize(
    ("name", "rotate", "kwargs"),
    [
        ("pytorch", fused_rope_rotate_pytorch, {}),
        ("blocked", fused_rope_rotate_blocked, {"block": 3}),
        ("triton-cpu-fallback", fused_rope_rotate_triton, {}),
    ],
)
def test_cpu_fused_fresh_output_is_contiguous_for_strided_input(
    name, rotate, kwargs
):
    """CPU-safe fused paths keep fresh outputs dense for strided inputs."""
    _, _, cos, sin, v, _ = _cis_and_v(T=7, seed=296)
    v_nc = v.transpose(1, 2)
    cos_nc = cos.transpose(1, 2)
    sin_nc = sin.transpose(1, 2)
    assert not v_nc.is_contiguous()

    ref = eager_rope_rotate(v_nc, cos_nc, sin_nc)
    got = rotate(v_nc, cos_nc, sin_nc, **kwargs)

    assert got.is_contiguous(), name
    assert torch.equal(got, ref), name


@pytest.mark.parametrize(
    ("name", "rotate", "paired", "kwargs"),
    [
        ("pytorch", fused_rope_rotate_pytorch, False, {}),
        ("blocked", fused_rope_rotate_blocked, False, {"block": 3}),
        ("triton-cpu-fallback", fused_rope_rotate_triton, False, {}),
        ("paired", fused_rope_rotate_paired, True, {}),
    ],
)
def test_cpu_fused_t1_fresh_output_is_contiguous_for_strided_input(
    name, rotate, paired, kwargs
):
    """CPU-safe T=1 fused paths keep fresh outputs dense for strided inputs."""
    cfg = _small_cfg()
    attn = bdh.Attention(cfg)
    device = torch.device("cpu")
    N = cfg.mlp_internal_dim_multiplier * cfg.n_embd // cfg.n_head
    rope_start = 6
    attn.ensure_rope_table(16, device)
    cos, sin = attn.rope_cos_sin(1, rope_start, device)
    cis = attn.t1_cis_pairs(rope_start, device)
    assert cis is not None

    torch.manual_seed(297)
    v = torch.randn(2, cfg.n_head, 1, N)
    v_nc = v.transpose(0, 1)
    assert not v_nc.is_contiguous()
    ref = eager_rope_rotate(v_nc, cos, sin)

    args = (v_nc, cis[0], cis[1]) if paired else (v_nc, cos, sin)
    got = rotate(*args, **kwargs)

    assert got.is_contiguous(), name
    assert torch.equal(got, ref), name


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
