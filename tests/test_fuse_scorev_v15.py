"""Focused v15 contract coverage for capacity-strided score×V inputs."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kernels.attention import (  # noqa: E402
    blocked_tril_attn,
    eager_tril_attn,
    online_tril_attn,
)


def test_blocked_online_preserve_capacity_strided_qkv():
    """Strict score×V parity must survive capacity-strided Q/K/V views."""
    B, H, T, N, D = 2, 3, 257, 5, 4
    g = torch.Generator(device="cpu").manual_seed(115)
    Q = torch.randn(B, H, T + 3, N, generator=g)[:, :, 1 : T + 1, :]
    K = torch.randn(B, H, T + 3, N, generator=g)[:, :, 2 : T + 2, :]
    V = torch.randn(B, H, T + 2, D, generator=g)[:, :, 1 : T + 1, :]

    assert not Q.is_contiguous()
    assert not K.is_contiguous()
    assert not V.is_contiguous()

    ref = eager_tril_attn(Q, K, V)
    expected = (Q @ K.transpose(-2, -1)).tril(diagonal=-1) @ V
    assert torch.allclose(ref, expected, rtol=1e-5, atol=1e-5)

    for fn in (blocked_tril_attn, online_tril_attn):
        got = fn(Q, K, V, block_size=64)
        assert torch.allclose(got, ref, rtol=1e-4, atol=1e-4)
