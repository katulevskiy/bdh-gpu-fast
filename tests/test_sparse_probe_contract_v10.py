"""Contract tests for sparse probe gate normalization."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks import bench_sparse_probe as probe
import bdh_sparse as sp


def test_falsey_gate_tokens_preserve_hard_noop(monkeypatch, capsys):
    """False-like opt-in values must not start CPU probe work."""
    falsey_tokens = ("", "0", "false", "no", "off", "disabled")

    def unexpected_work(**_):
        raise AssertionError("falsey sparse probe gate must not do work")

    monkeypatch.setattr(probe, "short_train_density", unexpected_work)
    monkeypatch.setattr(probe, "bench_matmul", unexpected_work)

    for token in falsey_tokens:
        monkeypatch.setenv(sp.SPARSE_PROBE_ENV, token)
        assert probe.main([]) == probe.EXIT_OK
        assert "exit_code=0 reason=probe_disabled" in capsys.readouterr().out
