"""CPU-safe unit checks for the generate microbench helpers."""

from __future__ import annotations

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
