"""Contract tests for sparse probe mode boundaries."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks import bench_sparse_probe as probe
import bdh_sparse as sp


def test_density_only_skips_crossover_after_guardrail_pass(monkeypatch, capsys):
    """The density-only mode must not start the CPU crossover sweep."""
    monkeypatch.setenv(sp.SPARSE_PROBE_ENV, "1")
    monkeypatch.setattr(
        probe,
        "short_train_density",
        lambda **_: [{"x": 0.25, "y": 0.25, "xy": 0.10}],
    )

    def unexpected_crossover(**_):
        raise AssertionError("density-only mode must stop before crossover work")

    monkeypatch.setattr(probe, "bench_matmul", unexpected_crossover)
    assert (
        probe.main(["--density-only", "--enforce-density-guardrail"])
        == probe.EXIT_OK
    )

    captured = capsys.readouterr()
    assert "density_guardrail=pass" in captured.out
    assert "CPU sparse vs dense crossover" not in captured.out
    assert "exit_code=0 reason=probe_complete" in captured.out
