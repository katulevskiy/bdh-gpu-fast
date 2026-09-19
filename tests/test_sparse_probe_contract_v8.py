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


def test_enforced_guardrail_failure_returns_distinct_exit(monkeypatch, capsys):
    """A failing sample must stop before CPU crossover work."""
    monkeypatch.setenv(sp.SPARSE_PROBE_ENV, "1")
    monkeypatch.setattr(
        probe,
        "short_train_density",
        lambda **_: [{"x": 0.19, "y": 0.25, "xy": 0.07}],
    )

    def unexpected_crossover(**_):
        raise AssertionError("failed guardrail must not run crossover work")

    monkeypatch.setattr(probe, "bench_matmul", unexpected_crossover)
    assert (
        probe.main(["--enforce-density-guardrail"])
        == probe.EXIT_DENSITY_GUARDRAIL
    )

    captured = capsys.readouterr()
    assert "density_guardrail=fail" in captured.out
    assert "final_x=0.1900 final_xy=0.0700" in captured.out
    assert "CPU sparse vs dense crossover" not in captured.out
    assert "density re-smoke guardrail failed" in captured.err
    assert "exit_code=2 reason=guardrail_failed" in captured.err


def test_enforced_guardrail_checks_final_sample(monkeypatch, capsys):
    """An earlier passing sample must not hide a failing final sample."""
    monkeypatch.setenv(sp.SPARSE_PROBE_ENV, "1")
    monkeypatch.setattr(
        probe,
        "short_train_density",
        lambda **_: [
            {"x": 0.20, "y": 0.25, "xy": 0.08},
            {"x": 0.19, "y": 0.25, "xy": 0.07},
        ],
    )

    def unexpected_crossover(**_):
        raise AssertionError("a failing final sample must stop crossover work")

    monkeypatch.setattr(probe, "bench_matmul", unexpected_crossover)
    assert (
        probe.main(["--enforce-density-guardrail"])
        == probe.EXIT_DENSITY_GUARDRAIL
    )

    captured = capsys.readouterr()
    assert "density_guardrail=fail" in captured.out
    assert "final_x=0.1900 final_xy=0.0700" in captured.out
    assert "CPU sparse vs dense crossover" not in captured.out
    assert "exit_code=2 reason=guardrail_failed" in captured.err


def test_enforced_guardrail_fails_when_only_xy_misses_floor(monkeypatch, capsys):
    """Both final density floors are required; one miss must still stop work."""
    monkeypatch.setenv(sp.SPARSE_PROBE_ENV, "1")
    monkeypatch.setattr(
        probe,
        "short_train_density",
        lambda **_: [{"x": 0.21, "y": 0.25, "xy": 0.07}],
    )

    def unexpected_crossover(**_):
        raise AssertionError("a single missed floor must stop crossover work")

    monkeypatch.setattr(probe, "bench_matmul", unexpected_crossover)
    assert (
        probe.main(["--enforce-density-guardrail"])
        == probe.EXIT_DENSITY_GUARDRAIL
    )

    captured = capsys.readouterr()
    assert "density_guardrail=fail" in captured.out
    assert "final_x=0.2100 final_xy=0.0700" in captured.out
    assert "CPU sparse vs dense crossover" not in captured.out
    assert "exit_code=2 reason=guardrail_failed" in captured.err


def test_enforced_guardrail_pass_allows_crossover(monkeypatch, capsys):
    """A passing enforced sample must permit the optional crossover sweep."""
    monkeypatch.setenv(sp.SPARSE_PROBE_ENV, "1")
    monkeypatch.setattr(
        probe,
        "short_train_density",
        lambda **_: [{"x": 0.20, "y": 0.25, "xy": 0.08}],
    )
    calls = []

    def fake_bench_matmul(M, K, N, density, seed=0):
        calls.append((M, K, N, density, seed))
        return {
            "density_measured": density,
            "dense_ms": 1.0,
            "coo_ms": 2.0,
            "csr_ms": 2.0,
            "row_ms": 2.0,
            "col_ms": 2.0,
        }

    monkeypatch.setattr(probe, "bench_matmul", fake_bench_matmul)
    assert probe.main(["--enforce-density-guardrail"]) == probe.EXIT_OK

    captured = capsys.readouterr()
    assert "density_guardrail=pass" in captured.out
    assert "CPU sparse vs dense crossover" in captured.out
    assert len(calls) == 3 * 8
    assert "exit_code=0 reason=probe_complete" in captured.out
