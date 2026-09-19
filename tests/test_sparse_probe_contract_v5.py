"""Contract tests for sparse probe exit guards."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks import bench_sparse_probe as probe
import bdh_sparse as sp


def test_enforced_guardrail_failure_is_terminal_before_crossover(
    monkeypatch, capsys
):
    """A failed density guardrail must not start the CPU crossover sweep."""
    monkeypatch.setenv(sp.SPARSE_PROBE_ENV, "1")
    monkeypatch.setattr(
        probe,
        "short_train_density",
        lambda **_: [{"x": 0.01, "y": 0.01, "xy": 0.01}],
    )

    def unexpected_crossover(**_):
        raise AssertionError("failed guardrail must stop before crossover work")

    monkeypatch.setattr(probe, "bench_matmul", unexpected_crossover)
    assert probe.main(["--enforce-density-guardrail"]) == probe.EXIT_DENSITY_GUARDRAIL

    captured = capsys.readouterr()
    assert "CPU sparse vs dense crossover" not in captured.out
    assert "exit_code=2 reason=guardrail_failed" in captured.err
