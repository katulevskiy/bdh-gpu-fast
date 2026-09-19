"""CPU parity for the B>1 shared-V decode epilogue."""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bdh_cache import CacheManager  # noqa: E402
from kernels.attention import (  # noqa: E402
    _DECODE_ONESHOT_ELEMS,
    blocked_decode_attn,
    eager_decode_attn,
)


def _qkv(B=2, H=4, S=32, N=8, D=16, seed=0, requires_grad=False):
    g = torch.Generator(device="cpu").manual_seed(seed)
    Q = torch.randn(B, H, 1, N, generator=g)
    K = torch.randn(B, H, S, N, generator=g)
    V = torch.randn(B, 1, S, D, generator=g)
    if requires_grad:
        Q.requires_grad_()
        K.requires_grad_()
        V.requires_grad_()
    return Q, K, V


def test_b1_shared_v_epilogue_preserves_oneshot_parity():
    """The exact budget boundary remains on the direct shared-V path."""
    Q, K, V = _qkv(S=_DECODE_ONESHOT_ELEMS, seed=401)
    with torch.inference_mode():
        got = blocked_decode_attn(Q, K, V, block_size=64)
        ref = eager_decode_attn(Q, K, V)
    assert torch.allclose(got, ref, rtol=1e-5, atol=1e-5)


def test_batched_shared_v_accumulation_writes_output(monkeypatch):
    """B>1 beta=1 tiles reuse output via baddbmm without head staging."""
    B, H, S = 2, 4, _DECODE_ONESHOT_ELEMS * 2 + 1
    Q, K, V = _qkv(B=B, H=H, S=S, seed=402)
    calls = []
    original = torch.baddbmm

    def spy(input, batch1, batch2, *, beta=1, alpha=1, out=None):
        if out is not None:
            calls.append((beta, input.data_ptr(), out.data_ptr(), batch1.shape))
        return original(input, batch1, batch2, beta=beta, alpha=alpha, out=out)

    monkeypatch.setattr(torch, "baddbmm", spy)
    with torch.inference_mode():
        got = blocked_decode_attn(Q, K, V, block_size=64)
        ref = eager_decode_attn(Q, K, V)

    accumulated = [c for c in calls if c[0] == 1]
    assert accumulated
    assert all(input_ptr == out_ptr for _, input_ptr, out_ptr, _ in accumulated)
    assert all(shape[0] == B and shape[1] == H for _, _, _, shape in accumulated)
    assert torch.allclose(got, ref, rtol=1e-4, atol=1e-5)


def test_batched_shared_v_nonflat_key_view_keeps_4d_strides(monkeypatch):
    """Long B>1 decode retains a non-flattenable K view instead of copying."""
    B, H, S, N, D = 2, 4, _DECODE_ONESHOT_ELEMS * 2 + 129, 8, 16
    cm = CacheManager(
        n_layer=1,
        max_seq=S + 17,
        batch_size=B,
        n_head=H,
        n_latent=N,
        n_embd=D,
        device="cpu",
    )
    torch.manual_seed(404)
    cm._kr_buf[0].normal_()
    cm._v_buf[0].normal_()
    cm.seq_len = S
    K, V = cm.get_past(0)
    assert K is not None and V is not None
    assert K.stride() == (H * cm.capacity * N, cm.capacity * N, N, 1)

    # Reorder the backing storage so (B,H,...) cannot flatten as a view while
    # preserving the same logical K values and the shared-V cache layout.
    K = K.transpose(0, 1).contiguous().permute(1, 0, 2, 3)
    Q = torch.randn(B, H, 1, N)
    assert K.stride(0) != H * K.stride(1)
    bmm_calls = []
    original = torch.bmm

    def spy(input, batch1, batch2):
        bmm_calls.append((batch1.shape, batch2.shape))
        return original(input, batch1, batch2)

    monkeypatch.setattr(torch, "bmm", spy)
    with torch.inference_mode():
        got = blocked_decode_attn(Q, K, V, block_size=64)
        ref = eager_decode_attn(Q, K, V)

    # The non-flattenable K view takes the 4-D score path; the shared-V
    # epilogue may still use baddbmm, but never a score bmm staging copy.
    assert not bmm_calls
    assert torch.allclose(got, ref, rtol=1e-4, atol=1e-5)


def test_batched_shared_v_autograd_fallback_is_graph_safe():
    """Grad-enabled shared-V decode keeps the non-out= fallback."""
    Q, K, V = _qkv(
        B=2,
        H=3,
        S=_DECODE_ONESHOT_ELEMS + 129,
        N=5,
        D=7,
        seed=403,
        requires_grad=True,
    )
    got = blocked_decode_attn(Q, K, V, block_size=64)
    got.square().mean().backward()
    assert Q.grad is not None and K.grad is not None and V.grad is not None
    assert torch.isfinite(Q.grad).all()
    assert torch.isfinite(K.grad).all()
    assert torch.isfinite(V.grad).all()
