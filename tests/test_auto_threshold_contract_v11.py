"""CPU-safe coverage for invalid AUTO decode-threshold overrides."""

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
    ("raw", "message"),
    [
        ("not-an-int", "must be an int"),
        ("-1", "must be >= 0"),
    ],
)
def test_invalid_decode_threshold_fails_both_gates_and_recovers(
    monkeypatch, raw, message
):
    """Invalid shared thresholds fail closed and recover for both AUTO paths."""
    monkeypatch.setenv("BDH_ATTN_AUTO", "1")
    monkeypatch.setenv("BDH_ATTN_IMPL", "eager")
    monkeypatch.setenv("BDH_ATTN_AUTO_THRESHOLD", raw)
    monkeypatch.delenv("BDH_ATTN_AUTO_COLD_THRESHOLD", raising=False)

    with pytest.raises(ValueError, match=message):
        resolve_decode_impl(5)
    with pytest.raises(ValueError, match=message):
        resolve_cold_impl(5)

    monkeypatch.setenv("BDH_ATTN_AUTO_THRESHOLD", "6")
    assert resolve_decode_impl(6) == "eager"
    assert resolve_cold_impl(6) == "eager"
    assert resolve_decode_impl(7) != "eager"
    assert resolve_cold_impl(7) != "eager"
