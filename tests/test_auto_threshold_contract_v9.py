"""CPU-safe coverage for blank AUTO cold-threshold overrides."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kernels.attention_dispatch import (  # noqa: E402
    attn_auto_cold_threshold,
    attn_auto_threshold,
    resolve_cold_impl,
    resolve_decode_impl,
)


@pytest.mark.parametrize("raw", ["", "   "])
def test_blank_cold_threshold_tracks_decode_threshold(monkeypatch, raw):
    """Blank cold overrides mirror the live decode threshold at both gates."""
    monkeypatch.setenv("BDH_ATTN_AUTO", "1")
    monkeypatch.setenv("BDH_ATTN_IMPL", "eager")
    monkeypatch.setenv("BDH_ATTN_AUTO_THRESHOLD", "4")
    monkeypatch.setenv("BDH_ATTN_AUTO_COLD_THRESHOLD", raw)

    assert attn_auto_threshold() == 4
    assert attn_auto_cold_threshold() == 4
    assert resolve_cold_impl(4) == "eager"
    assert resolve_cold_impl(5) != "eager"
    assert resolve_decode_impl(4) == "eager"
    assert resolve_decode_impl(5) != "eager"

    monkeypatch.setenv("BDH_ATTN_AUTO_THRESHOLD", "8")
    assert attn_auto_threshold() == 8
    assert attn_auto_cold_threshold() == 8
    assert resolve_cold_impl(8) == "eager"
    assert resolve_cold_impl(9) != "eager"
    assert resolve_decode_impl(8) == "eager"
    assert resolve_decode_impl(9) != "eager"


def test_blank_cold_threshold_recovers_from_explicit_override(monkeypatch):
    """Blank cold overrides clear a prior value and resume live fallback."""
    monkeypatch.setenv("BDH_ATTN_AUTO", "1")
    monkeypatch.setenv("BDH_ATTN_IMPL", "eager")
    monkeypatch.setenv("BDH_ATTN_AUTO_THRESHOLD", "4")
    monkeypatch.setenv("BDH_ATTN_AUTO_COLD_THRESHOLD", "2")

    assert attn_auto_cold_threshold() == 2
    assert resolve_cold_impl(2) == "eager"
    assert resolve_cold_impl(3) != "eager"

    monkeypatch.setenv("BDH_ATTN_AUTO_COLD_THRESHOLD", "   ")
    assert attn_auto_cold_threshold() == 4
    assert resolve_cold_impl(4) == "eager"
    assert resolve_cold_impl(5) != "eager"

    monkeypatch.setenv("BDH_ATTN_AUTO_THRESHOLD", "8")
    assert attn_auto_cold_threshold() == 8
    assert resolve_cold_impl(8) == "eager"
    assert resolve_cold_impl(9) != "eager"


def test_unset_cold_threshold_recovers_from_explicit_override(monkeypatch):
    """Removing a cold override resumes the live decode-threshold fallback."""
    monkeypatch.setenv("BDH_ATTN_AUTO", "1")
    monkeypatch.setenv("BDH_ATTN_IMPL", "eager")
    monkeypatch.setenv("BDH_ATTN_AUTO_THRESHOLD", "4")
    monkeypatch.setenv("BDH_ATTN_AUTO_COLD_THRESHOLD", "2")

    assert attn_auto_cold_threshold() == 2
    assert resolve_cold_impl(2) == "eager"
    assert resolve_cold_impl(3) != "eager"

    monkeypatch.delenv("BDH_ATTN_AUTO_COLD_THRESHOLD")
    assert attn_auto_cold_threshold() == 4
    assert resolve_cold_impl(4) == "eager"
    assert resolve_cold_impl(5) != "eager"

    monkeypatch.setenv("BDH_ATTN_AUTO_THRESHOLD", "8")
    assert attn_auto_cold_threshold() == 8
    assert resolve_cold_impl(8) == "eager"
    assert resolve_cold_impl(9) != "eager"


def test_blank_cold_threshold_accepts_later_explicit_override(monkeypatch):
    """A blank fallback can transition back to an independent override."""
    monkeypatch.setenv("BDH_ATTN_AUTO", "1")
    monkeypatch.setenv("BDH_ATTN_IMPL", "eager")
    monkeypatch.setenv("BDH_ATTN_AUTO_THRESHOLD", "4")
    monkeypatch.setenv("BDH_ATTN_AUTO_COLD_THRESHOLD", " ")

    assert attn_auto_cold_threshold() == 4
    assert resolve_cold_impl(4) == "eager"
    assert resolve_cold_impl(5) != "eager"

    monkeypatch.setenv("BDH_ATTN_AUTO_COLD_THRESHOLD", "2")
    assert attn_auto_cold_threshold() == 2
    assert resolve_cold_impl(2) == "eager"
    assert resolve_cold_impl(3) != "eager"

    monkeypatch.setenv("BDH_ATTN_AUTO_THRESHOLD", "8")
    assert attn_auto_cold_threshold() == 2
    assert resolve_cold_impl(2) == "eager"
    assert resolve_cold_impl(3) != "eager"


def test_explicit_cold_threshold_does_not_shadow_decode_threshold(monkeypatch):
    """An explicit cold override stays scoped while decode follows the shared knob."""
    monkeypatch.setenv("BDH_ATTN_AUTO", "1")
    monkeypatch.setenv("BDH_ATTN_IMPL", "eager")
    monkeypatch.setenv("BDH_ATTN_AUTO_THRESHOLD", "4")
    monkeypatch.setenv("BDH_ATTN_AUTO_COLD_THRESHOLD", "2")

    assert attn_auto_cold_threshold() == 2
    assert resolve_cold_impl(2) == "eager"
    assert resolve_cold_impl(3) != "eager"
    assert resolve_decode_impl(4) == "eager"
    assert resolve_decode_impl(5) != "eager"

    monkeypatch.setenv("BDH_ATTN_AUTO_THRESHOLD", "8")
    assert attn_auto_cold_threshold() == 2
    assert resolve_cold_impl(2) == "eager"
    assert resolve_cold_impl(3) != "eager"
    assert resolve_decode_impl(8) == "eager"
    assert resolve_decode_impl(9) != "eager"
