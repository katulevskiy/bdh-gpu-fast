"""Contract tests for sparse probe guardrail boundaries."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks import bench_sparse_probe as probe
import bdh_sparse as sp


def test_density_guardrail_accepts_exact_thresholds(monkeypatch, capsys):
    """The inclusive minimum floors accept an exact boundary sample."""
    monkeypatch.setenv(sp.SPARSE_PROBE_ENV, "1")
    monkeypatch.setattr(
        probe,
        "short_train_density",
        lambda **_: [{"x": 0.20, "y": 0.25, "xy": 0.08}],
    )

    def unexpected_crossover(**_):
        raise AssertionError("density-only mode must not run crossover work")

    monkeypatch.setattr(probe, "bench_matmul", unexpected_crossover)
    assert (
        probe.main(["--density-only", "--enforce-density-guardrail"])
        == probe.EXIT_OK
    )

    captured = capsys.readouterr()
    assert "density_guardrail=pass" in captured.out
    assert "final_x=0.2000 final_xy=0.0800" in captured.out
    assert "exit_code=0 reason=probe_complete" in captured.out


def test_enforced_guardrail_without_samples_returns_distinct_exit(monkeypatch, capsys):
    """Enforcement without training samples must stop before crossover work."""
    monkeypatch.setenv(sp.SPARSE_PROBE_ENV, "1")

    def unexpected_work(**_):
        raise AssertionError("no-sample enforcement must not run probe work")

    monkeypatch.setattr(probe, "short_train_density", unexpected_work)
    monkeypatch.setattr(probe, "bench_matmul", unexpected_work)
    assert (
        probe.main(["--skip-train", "--enforce-density-guardrail"])
        == probe.EXIT_GUARDRAIL_NO_SAMPLES
    )

    captured = capsys.readouterr()
    assert "density_guardrail=unavailable" in captured.err
    assert "CPU sparse vs dense crossover" not in captured.out
    assert "exit_code=3 reason=guardrail_no_samples" in captured.err
