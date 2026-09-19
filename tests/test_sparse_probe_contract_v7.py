"""Contract tests for the sparse probe density-only boundary."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks import bench_sparse_probe as probe
import bdh_sparse as sp


def test_density_only_skips_crossover_without_enforcement(monkeypatch, capsys):
    """Density-only must remain crossover-free without guardrail enforcement."""
    monkeypatch.setenv(sp.SPARSE_PROBE_ENV, "1")
    monkeypatch.setattr(
        probe,
        "short_train_density",
        lambda **_: [{"x": 0.25, "y": 0.25, "xy": 0.10}],
    )

    def unexpected_crossover(**_):
        raise AssertionError("density-only mode must not run crossover work")

    monkeypatch.setattr(probe, "bench_matmul", unexpected_crossover)
    assert probe.main(["--density-only"]) == probe.EXIT_OK

    captured = capsys.readouterr()
    assert "density_guardrail=pass" in captured.out
    assert "CPU sparse vs dense crossover" not in captured.out
    assert "exit_code=0 reason=probe_complete" in captured.out


def test_failed_density_guardrail_stops_crossover(monkeypatch, capsys):
    """An enforced failure must return before any crossover work starts."""
    monkeypatch.setenv(sp.SPARSE_PROBE_ENV, "1")
    monkeypatch.setattr(
        probe,
        "short_train_density",
        lambda **_: [{"x": 0.01, "y": 0.25, "xy": 0.01}],
    )

    def unexpected_crossover(**_):
        raise AssertionError("failed density guardrail must stop before crossover")

    monkeypatch.setattr(probe, "bench_matmul", unexpected_crossover)
    assert (
        probe.main(["--enforce-density-guardrail"])
        == probe.EXIT_DENSITY_GUARDRAIL
    )

    captured = capsys.readouterr()
    assert "density_guardrail=fail" in captured.out
    assert "CPU sparse vs dense crossover" not in captured.out
    assert "density re-smoke guardrail failed" in captured.err
    assert "exit_code=2 reason=guardrail_failed" in captured.err
