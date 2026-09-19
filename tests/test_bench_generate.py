"""CPU-safe unit checks for the generate microbench helpers."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))

import bench_generate  # noqa: E402


def test_parse_threshold_sweep_is_stable_and_deduplicated():
    assert bench_generate._parse_threshold_sweep("256, 512,256,1024") == [
        256,
        512,
        1024,
    ]
    assert bench_generate._parse_threshold_sweep(None) == []


@pytest.mark.parametrize("raw", ["", "256,,512", "-1,512", "fast,512"])
def test_parse_threshold_sweep_rejects_invalid_values(raw):
    with pytest.raises(ValueError):
        bench_generate._parse_threshold_sweep(raw)


def test_attn_impl_restores_environment_after_exception(monkeypatch):
    """An interrupted impl run must not leak its dispatch override."""
    monkeypatch.setenv("BDH_ATTN_IMPL", "eager")

    with pytest.raises(RuntimeError, match="stop impl"):
        with bench_generate._attn_impl("blocked"):
            assert os.environ["BDH_ATTN_IMPL"] == "blocked"
            raise RuntimeError("stop impl")

    assert os.environ["BDH_ATTN_IMPL"] == "eager"


def test_attn_auto_restores_all_threshold_environment_after_exception(monkeypatch):
    """An interrupted AUTO A/B run must restore every temporary gate value."""
    monkeypatch.setenv("BDH_ATTN_AUTO", "0")
    monkeypatch.setenv("BDH_ATTN_AUTO_THRESHOLD", "old-decode")
    monkeypatch.setenv("BDH_ATTN_AUTO_COLD_THRESHOLD", "old-cold")

    with pytest.raises(RuntimeError, match="stop auto"):
        with bench_generate._attn_auto(True, threshold=256, cold_threshold=128):
            assert os.environ["BDH_ATTN_AUTO"] == "1"
            assert os.environ["BDH_ATTN_AUTO_THRESHOLD"] == "256"
            assert os.environ["BDH_ATTN_AUTO_COLD_THRESHOLD"] == "128"
            raise RuntimeError("stop auto")

    assert os.environ["BDH_ATTN_AUTO"] == "0"
    assert os.environ["BDH_ATTN_AUTO_THRESHOLD"] == "old-decode"
    assert os.environ["BDH_ATTN_AUTO_COLD_THRESHOLD"] == "old-cold"
