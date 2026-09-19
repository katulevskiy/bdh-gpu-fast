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


def test_enforced_guardrail_empty_history_returns_distinct_exit(monkeypatch, capsys):
    """An empty training result must be treated as missing samples."""
    monkeypatch.setenv(sp.SPARSE_PROBE_ENV, "1")
    monkeypatch.setattr(probe, "short_train_density", lambda **_: [])

    def unexpected_crossover(**_):
        raise AssertionError("empty history must not run crossover work")

    monkeypatch.setattr(probe, "bench_matmul", unexpected_crossover)
    assert (
        probe.main(["--enforce-density-guardrail"])
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


def test_enforced_guardrail_accepts_passing_final_sample(monkeypatch, capsys):
    """A passing final sample must override an earlier transient miss."""
    monkeypatch.setenv(sp.SPARSE_PROBE_ENV, "1")
    monkeypatch.setattr(
        probe,
        "short_train_density",
        lambda **_: [
            {"x": 0.19, "y": 0.25, "xy": 0.07},
            {"x": 0.20, "y": 0.25, "xy": 0.08},
        ],
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
    assert "CPU sparse vs dense crossover" not in captured.out
    assert "exit_code=0 reason=probe_complete" in captured.out


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


def test_density_only_skips_crossover_after_passing_guardrail(monkeypatch, capsys):
    """A passing density-only probe must still honor its crossover skip."""
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
    assert "CPU sparse vs dense crossover" not in captured.out
    assert "exit_code=0 reason=probe_complete" in captured.out


def test_enforced_guardrail_honors_custom_cli_floors(monkeypatch, capsys):
    """CLI floors must override defaults before deciding enforcement."""
    monkeypatch.setenv(sp.SPARSE_PROBE_ENV, "1")
    monkeypatch.setattr(
        probe,
        "short_train_density",
        lambda **_: [{"x": 0.20, "y": 0.25, "xy": 0.08}],
    )

    def unexpected_crossover(**_):
        raise AssertionError("a custom-floor failure must stop crossover work")

    monkeypatch.setattr(probe, "bench_matmul", unexpected_crossover)
    assert (
        probe.main(
            [
                "--density-only",
                "--enforce-density-guardrail",
                "--min-final-x",
                "0.21",
            ]
        )
        == probe.EXIT_DENSITY_GUARDRAIL
    )

    captured = capsys.readouterr()
    assert "density_guardrail=fail" in captured.out
    assert "required_x>=0.2100 required_xy>=0.0800" in captured.out
    assert "CPU sparse vs dense crossover" not in captured.out
    assert "exit_code=2 reason=guardrail_failed" in captured.err


def test_enforced_guardrail_honors_custom_cli_xy_floor(monkeypatch, capsys):
    """The custom xy floor must participate in enforced guardrail decisions."""
    monkeypatch.setenv(sp.SPARSE_PROBE_ENV, "1")
    monkeypatch.setattr(
        probe,
        "short_train_density",
        lambda **_: [{"x": 0.20, "y": 0.25, "xy": 0.08}],
    )

    def unexpected_crossover(**_):
        raise AssertionError("a custom xy-floor failure must stop crossover work")

    monkeypatch.setattr(probe, "bench_matmul", unexpected_crossover)
    assert (
        probe.main(
            [
                "--density-only",
                "--enforce-density-guardrail",
                "--min-final-xy",
                "0.09",
            ]
        )
        == probe.EXIT_DENSITY_GUARDRAIL
    )

    captured = capsys.readouterr()
    assert "density_guardrail=fail" in captured.out
    assert "required_x>=0.2000 required_xy>=0.0900" in captured.out
    assert "CPU sparse vs dense crossover" not in captured.out
    assert "exit_code=2 reason=guardrail_failed" in captured.err


def test_density_only_honors_custom_cli_floors_on_pass(monkeypatch, capsys):
    """Passing custom floors must preserve density-only crossover isolation."""
    monkeypatch.setenv(sp.SPARSE_PROBE_ENV, "1")
    monkeypatch.setattr(
        probe,
        "short_train_density",
        lambda **_: [{"x": 0.23, "y": 0.25, "xy": 0.09}],
    )

    def unexpected_crossover(**_):
        raise AssertionError("density-only mode must not run crossover work")

    monkeypatch.setattr(probe, "bench_matmul", unexpected_crossover)
    assert (
        probe.main(
            [
                "--density-only",
                "--enforce-density-guardrail",
                "--min-final-x",
                "0.23",
                "--min-final-xy",
                "0.09",
            ]
        )
        == probe.EXIT_OK
    )

    captured = capsys.readouterr()
    assert "density_guardrail=pass" in captured.out
    assert "required_x>=0.2300 required_xy>=0.0900" in captured.out
    assert "CPU sparse vs dense crossover" not in captured.out
    assert "exit_code=0 reason=probe_complete" in captured.out


def test_enforced_guardrail_honors_explicit_skip_crossover(monkeypatch, capsys):
    """A passing enforced probe must honor an explicit crossover skip."""
    monkeypatch.setenv(sp.SPARSE_PROBE_ENV, "1")
    monkeypatch.setattr(
        probe,
        "short_train_density",
        lambda **_: [{"x": 0.20, "y": 0.25, "xy": 0.08}],
    )

    def unexpected_crossover(**_):
        raise AssertionError("--skip-crossover must not run crossover work")

    monkeypatch.setattr(probe, "bench_matmul", unexpected_crossover)
    assert (
        probe.main(["--skip-crossover", "--enforce-density-guardrail"])
        == probe.EXIT_OK
    )

    captured = capsys.readouterr()
    assert "density_guardrail=pass" in captured.out
    assert "CPU sparse vs dense crossover" not in captured.out
    assert "exit_code=0 reason=probe_complete" in captured.out


def test_explicit_skips_complete_without_probe_work(monkeypatch, capsys):
    """Opted-in all-skip mode must remain a CPU-safe successful no-op."""
    monkeypatch.setenv(sp.SPARSE_PROBE_ENV, "1")

    def unexpected_work(*_, **__):
        raise AssertionError("explicit skips must not run probe work")

    monkeypatch.setattr(probe, "short_train_density", unexpected_work)
    monkeypatch.setattr(probe, "bench_matmul", unexpected_work)
    assert probe.main(["--skip-train", "--skip-crossover"]) == probe.EXIT_OK

    captured = capsys.readouterr()
    assert "short train density" not in captured.out
    assert "CPU sparse vs dense crossover" not in captured.out
    assert "density_guardrail" not in captured.out
    assert "exit_code=0 reason=probe_complete" in captured.out


def test_density_only_implies_crossover_skip_without_training(monkeypatch, capsys):
    """The density-only flag must suppress crossover even when training is skipped."""
    monkeypatch.setenv(sp.SPARSE_PROBE_ENV, "1")

    def unexpected_work(*_, **__):
        raise AssertionError("density-only skip mode must not run probe work")

    monkeypatch.setattr(probe, "short_train_density", unexpected_work)
    monkeypatch.setattr(probe, "bench_matmul", unexpected_work)
    assert probe.main(["--density-only", "--skip-train"]) == probe.EXIT_OK

    captured = capsys.readouterr()
    assert "short train density" not in captured.out
    assert "CPU sparse vs dense crossover" not in captured.out
    assert "exit_code=0 reason=probe_complete" in captured.out


def test_density_only_skip_training_enforces_no_sample_guardrail(
    monkeypatch, capsys
):
    """Density-only skip mode must report missing samples when enforcement is on."""
    monkeypatch.setenv(sp.SPARSE_PROBE_ENV, "1")

    def unexpected_work(*_, **__):
        raise AssertionError("no-sample enforcement must not run probe work")

    monkeypatch.setattr(probe, "short_train_density", unexpected_work)
    monkeypatch.setattr(probe, "bench_matmul", unexpected_work)
    assert (
        probe.main(
            ["--density-only", "--skip-train", "--enforce-density-guardrail"]
        )
        == probe.EXIT_GUARDRAIL_NO_SAMPLES
    )

    captured = capsys.readouterr()
    assert "density_guardrail=unavailable" in captured.err
    assert "CPU sparse vs dense crossover" not in captured.out
    assert "exit_code=3 reason=guardrail_no_samples" in captured.err


def test_disabled_gate_overrides_enforcement_without_probe_work(monkeypatch, capsys):
    """The default-off gate must win even when guardrail enforcement is asked for."""
    monkeypatch.delenv(sp.SPARSE_PROBE_ENV, raising=False)

    def unexpected_work(*_, **__):
        raise AssertionError("disabled sparse probe must not do work")

    monkeypatch.setattr(probe, "short_train_density", unexpected_work)
    monkeypatch.setattr(probe, "bench_matmul", unexpected_work)
    assert probe.main(["--enforce-density-guardrail"]) == probe.EXIT_OK

    captured = capsys.readouterr()
    assert "density_guardrail" not in captured.out
    assert "CPU sparse vs dense crossover" not in captured.out
    assert "exit_code=0 reason=probe_disabled" in captured.out


def test_density_probe_forwards_cli_sampling_options(monkeypatch, capsys):
    """The CPU probe must pass its sampling options through to training."""
    monkeypatch.setenv(sp.SPARSE_PROBE_ENV, "1")
    calls = []

    def fake_short_train_density(**kwargs):
        calls.append(kwargs)
        return [{"x": 0.20, "y": 0.25, "xy": 0.08}]

    def unexpected_crossover(**_):
        raise AssertionError("density-only mode must not run crossover work")

    monkeypatch.setattr(probe, "short_train_density", fake_short_train_density)
    monkeypatch.setattr(probe, "bench_matmul", unexpected_crossover)
    assert (
        probe.main(
            [
                "--density-only",
                "--enforce-density-guardrail",
                "--steps",
                "7",
                "--log-every",
                "3",
                "--batch",
                "2",
                "--block",
                "16",
                "--seed",
                "42",
            ]
        )
        == probe.EXIT_OK
    )

    assert calls == [
        {"steps": 7, "log_every": 3, "B": 2, "T": 16, "seed": 42}
    ]
    captured = capsys.readouterr()
    assert "density_guardrail=pass" in captured.out
    assert "CPU sparse vs dense crossover" not in captured.out
    assert "exit_code=0 reason=probe_complete" in captured.out
