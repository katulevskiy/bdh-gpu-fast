"""CPU-safe Triton cold diagnostics and blocked-fallback parity."""

from __future__ import annotations

import pytest
import torch

from kernels.attention import (
    _HAS_TRITON,
    blocked_tril_attn,
    eager_tril_attn,
    triton_cold_skip_reason,
    triton_tril_attn,
)


def test_triton_cold_skip_reason_marks_cpu_safe_contract():
    """Unavailable cold skips identify their gate without CUDA work."""
    reason = triton_cold_skip_reason()
    if torch.cuda.is_available() and _HAS_TRITON:
        assert reason is None
    else:
        assert reason is not None
        assert reason.endswith("(CPU-safe skip)")
        assert reason.startswith(("Triton unavailable:", "CUDA unavailable:"))


def test_triton_cold_skip_reason_covers_import_and_device_gates(monkeypatch):
    """Both skip gates are deterministic, and import failure avoids CUDA probes."""
    import kernels.attention as attention

    monkeypatch.setattr(attention, "_HAS_TRITON", False)

    def unexpected_cuda_probe():
        raise AssertionError("import failure must short-circuit before CUDA probing")

    monkeypatch.setattr(torch.cuda, "is_available", unexpected_cuda_probe)
    import_reason = attention.triton_cold_skip_reason()
    assert import_reason.startswith("Triton unavailable:")
    assert import_reason.endswith("(CPU-safe skip)")

    monkeypatch.setattr(attention, "_HAS_TRITON", True)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert attention.triton_cold_skip_reason() == (
        "CUDA unavailable: torch.cuda.is_available() is false (CPU-safe skip)"
    )

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert attention.triton_cold_skip_reason() is None


@pytest.mark.parametrize("v_heads", [1, 3])
def test_cpu_triton_fallback_matches_blocked_long_t(v_heads):
    """Long CPU fallback stays parity-equivalent for shared and per-head V."""
    B, H, T, N, D = 2, 3, 257, 17, 19
    generator = torch.Generator(device="cpu").manual_seed(500 + v_heads)
    Q = torch.randn(B, H, T, N, generator=generator)
    K = torch.randn(B, H, T, N, generator=generator)
    V = torch.randn(B, v_heads, T, D, generator=generator)

    got = triton_tril_attn(Q, K, V)
    blocked = blocked_tril_attn(Q, K, V)
    eager = eager_tril_attn(Q, K, V)

    assert torch.allclose(got, blocked, rtol=1e-4, atol=1e-4)
    assert torch.allclose(got, eager, rtol=1e-4, atol=1e-4)
    assert torch.count_nonzero(got[:, :, 0, :]) == 0
