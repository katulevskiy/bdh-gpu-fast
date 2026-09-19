"""CPU-safe coverage for scoped AUTO cold-threshold failures."""

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


def test_invalid_cold_threshold_does_not_poison_decode_gate(monkeypatch):
    """An invalid cold-only override leaves the shared decode gate usable."""
    monkeypatch.setenv("BDH_ATTN_AUTO", "1")
    monkeypatch.setenv("BDH_ATTN_IMPL", "eager")
    monkeypatch.setenv("BDH_ATTN_AUTO_THRESHOLD", "6")
    monkeypatch.setenv("BDH_ATTN_AUTO_COLD_THRESHOLD", "not-an-int")

    with pytest.raises(ValueError, match="must be an int"):
        resolve_cold_impl(7)

    assert resolve_decode_impl(6) == "eager"
    assert resolve_decode_impl(7) != "eager"

    monkeypatch.setenv("BDH_ATTN_AUTO_COLD_THRESHOLD", "8")
    assert resolve_cold_impl(8) == "eager"
    assert resolve_cold_impl(9) != "eager"
    assert resolve_decode_impl(7) != "eager"
