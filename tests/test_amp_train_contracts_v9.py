"""AMP train v9: environment/default contract for forward-only mode."""

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


def test_forward_only_env_default_and_explicit_override(monkeypatch):
    """Env defaults are honored, while an explicit argument remains authoritative."""
    if tr.device.type == "cpu" and not tr.cpu_bf16_available():
        pytest.skip("CPU bf16 autocast unavailable")

    monkeypatch.setenv("BDH_AMP_DTYPE", "bfloat16")
    monkeypatch.setenv("BDH_AMP_FORWARD_ONLY", "1")
    assert tr.configure_amp() == "bfloat16"
    assert tr._amp_forward_only is True
    assert tr._use_scaler is False

    tr.configure_amp(forward_only=False)
    assert tr.dtype == "bfloat16"
    assert tr._amp_forward_only is False
