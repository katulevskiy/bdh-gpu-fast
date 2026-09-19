"""Contract tests for sparse probe truthy gate normalization."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks import bench_sparse_probe as probe
import bdh_sparse as sp


def test_truthy_gate_tokens_enable_density_only_probe(monkeypatch, capsys):
    """Supported truthy tokens must enter the opt-in density-only path."""
    truthy_tokens = ("1", "true", "yes", "on", " TRUE ")
    calls = []

    def fake_short_train_density(**_):
        calls.append(True)
        return [{"x": 0.20, "y": 0.20, "xy": 0.08}]

    def unexpected_crossover(**_):
        raise AssertionError("density-only mode must not run crossover work")

    monkeypatch.setattr(probe, "short_train_density", fake_short_train_density)
    monkeypatch.setattr(probe, "bench_matmul", unexpected_crossover)

    for token in truthy_tokens:
        monkeypatch.setenv(sp.SPARSE_PROBE_ENV, token)
        assert (
            probe.main(["--density-only", "--enforce-density-guardrail"])
            == probe.EXIT_OK
        )
        captured = capsys.readouterr()
        assert "density_guardrail=pass" in captured.out
        assert "exit_code=0 reason=probe_complete" in captured.out

    assert len(calls) == len(truthy_tokens)
