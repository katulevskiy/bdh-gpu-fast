"""AMP train v10: complete the forward-only environment toggle contract."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ.setdefault("BDH_AMP_DTYPE", "float32")
os.environ.setdefault("BDH_COMPILE", "0")

import train as tr


@pytest.fixture(autouse=True)
def _restore_fp32_amp():
    yield
    tr.configure_amp("float32")


def test_forward_only_env_zero_and_explicit_true(monkeypatch):
    """The env false value disables the mode, while an explicit true wins."""
    if tr.device.type == "cpu" and not tr.cpu_bf16_available():
        pytest.skip("CPU bf16 autocast unavailable")

    monkeypatch.setenv("BDH_AMP_DTYPE", "bfloat16")
    monkeypatch.setenv("BDH_AMP_FORWARD_ONLY", "0")
    assert tr.configure_amp() == "bfloat16"
    assert tr._amp_forward_only is False

    tr.configure_amp(forward_only=True)
    assert tr._amp_forward_only is True
