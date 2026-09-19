"""CPU-safe coverage for the AUTO threshold fallback contract."""

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


def test_unset_cold_threshold_tracks_decode_threshold_changes(monkeypatch):
    """The unset cold gate follows later decode-threshold updates, too."""
    monkeypatch.setenv("BDH_ATTN_AUTO", "1")
    monkeypatch.setenv("BDH_ATTN_IMPL", "eager")
    monkeypatch.setenv("BDH_ATTN_AUTO_THRESHOLD", "4")
    monkeypatch.delenv("BDH_ATTN_AUTO_COLD_THRESHOLD", raising=False)

    assert attn_auto_threshold() == 4
    assert attn_auto_cold_threshold() == 4
    assert resolve_decode_impl(4) == "eager"
    assert resolve_cold_impl(4) == "eager"
    assert resolve_decode_impl(5) != "eager"
    assert resolve_cold_impl(5) != "eager"

    monkeypatch.setenv("BDH_ATTN_AUTO_THRESHOLD", "8")
    assert attn_auto_threshold() == 8
    assert attn_auto_cold_threshold() == 8
    assert resolve_decode_impl(8) == "eager"
    assert resolve_cold_impl(8) == "eager"
    assert resolve_decode_impl(9) == resolve_cold_impl(9)
    assert resolve_decode_impl(9) != "eager"
