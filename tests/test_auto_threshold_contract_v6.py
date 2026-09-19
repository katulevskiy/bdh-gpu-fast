"""CPU-safe parity coverage for explicit AUTO backend contracts."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kernels.attention_dispatch import (  # noqa: E402
    resolve_cold_impl,
    resolve_decode_impl,
)


@pytest.mark.parametrize(
    "impl,want",
    [
        ("blocked", "blocked"),
        ("triton", "triton"),
        ("cuda", "cuda"),
        ("online", "blocked"),
    ],
)
def test_auto_preserves_explicit_backend_for_both_gates(monkeypatch, impl, want):
    """AUTO never overrides an explicit backend on decode or cold paths."""
    monkeypatch.setenv("BDH_ATTN_AUTO", "1")
    monkeypatch.setenv("BDH_ATTN_AUTO_THRESHOLD", "0")
    monkeypatch.setenv("BDH_ATTN_AUTO_COLD_THRESHOLD", "0")
    monkeypatch.setenv("BDH_ATTN_IMPL", impl)

    assert resolve_decode_impl(1) == want
    assert resolve_cold_impl(1) == want
    assert resolve_decode_impl(1, requested=impl) == want
    assert resolve_cold_impl(1, requested=impl) == want
