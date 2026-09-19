"""CPU-safe coverage for AUTO threshold override transitions."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kernels.attention_dispatch import (  # noqa: E402
    attn_auto_cold_threshold,
    attn_auto_threshold,
    resolve_cold_impl,
    resolve_decode_impl,
)


def test_explicit_cold_threshold_stays_independent_then_falls_back(monkeypatch):
    """An override stays fixed, then follows a later decode update when removed."""
    monkeypatch.setenv("BDH_ATTN_AUTO", "1")
    monkeypatch.setenv("BDH_ATTN_IMPL", "eager")
    monkeypatch.setenv("BDH_ATTN_AUTO_THRESHOLD", "4")
    monkeypatch.setenv("BDH_ATTN_AUTO_COLD_THRESHOLD", "2")

    assert attn_auto_threshold() == 4
    assert attn_auto_cold_threshold() == 2
    assert resolve_cold_impl(2) == "eager"
    assert resolve_cold_impl(3) != "eager"

    monkeypatch.setenv("BDH_ATTN_AUTO_THRESHOLD", "8")
    assert attn_auto_threshold() == 8
    assert attn_auto_cold_threshold() == 2
    assert resolve_decode_impl(8) == "eager"
    assert resolve_decode_impl(9) != "eager"
    assert resolve_cold_impl(3) != "eager"

    monkeypatch.delenv("BDH_ATTN_AUTO_COLD_THRESHOLD", raising=False)
    assert attn_auto_cold_threshold() == 8
    assert resolve_cold_impl(8) == "eager"
    assert resolve_cold_impl(9) != "eager"
