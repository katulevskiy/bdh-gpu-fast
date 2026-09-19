"""CPU-safe unit checks for the generate microbench helpers."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))

import bench_generate  # noqa: E402


def test_parse_threshold_sweep_is_stable_and_deduplicated():
    assert bench_generate._parse_threshold_sweep("256, 512,256,1024") == [
        256,
        512,
        1024,
    ]
    assert bench_generate._parse_threshold_sweep(None) == []


@pytest.mark.parametrize("raw", ["", "256,,512", "-1,512", "fast,512"])
def test_parse_threshold_sweep_rejects_invalid_values(raw):
    with pytest.raises(ValueError):
        bench_generate._parse_threshold_sweep(raw)


@pytest.mark.parametrize("raw", ["8,,16", "8,fast"])
def test_parse_prompt_lengths_rejects_malformed_values(raw):
    with pytest.raises(ValueError):
        bench_generate._parse_prompt_lengths(raw)


def test_run_auto_ab_sweep_deduplicates_and_mirrors_cold_threshold(monkeypatch):
    """Each unique decode threshold runs once with a mirrored cold gate."""
    calls = []

    def fake_run_auto_ab(args, device):
        calls.append(
            (
                args.auto_threshold,
                args.auto_cold_threshold,
                args.auto_threshold_sweep,
            )
        )
        return 1 if args.auto_threshold == 512 else 0

    monkeypatch.setattr(bench_generate, "run_auto_ab", fake_run_auto_ab)
    args = argparse.Namespace(
        auto_threshold=999,
        auto_cold_threshold=None,
        auto_threshold_sweep="256, 512,256",
    )

    assert bench_generate.run_auto_ab_sweep(args, object()) == 1
    assert calls == [
        (256, 256, None),
        (512, 512, None),
    ]
    assert args.auto_threshold == 999
    assert args.auto_cold_threshold is None
    assert args.auto_threshold_sweep == "256, 512,256"


def test_run_auto_ab_sweep_preserves_explicit_cold_threshold(monkeypatch):
    """An explicit cold gate stays independent across decode threshold runs."""
    calls = []

    def fake_run_auto_ab(args, device):
        calls.append(
            (
                args.auto_threshold,
                args.auto_cold_threshold,
                args.auto_threshold_sweep,
            )
        )
        return 0

    monkeypatch.setattr(bench_generate, "run_auto_ab", fake_run_auto_ab)
    args = argparse.Namespace(
        auto_threshold=999,
        auto_cold_threshold=64,
        auto_threshold_sweep="256,512",
    )

    assert bench_generate.run_auto_ab_sweep(args, object()) == 0
    assert calls == [
        (256, 64, None),
        (512, 64, None),
    ]


def test_run_auto_ab_sweep_continues_after_failed_threshold(monkeypatch):
    """A failed threshold run must not prevent later sweep values from running."""
    calls = []

    def fake_run_auto_ab(args, device):
        calls.append(args.auto_threshold)
        return 1 if args.auto_threshold == 256 else 0

    monkeypatch.setattr(bench_generate, "run_auto_ab", fake_run_auto_ab)
    args = argparse.Namespace(
        auto_threshold=999,
        auto_cold_threshold=64,
        auto_threshold_sweep="256,512",
    )

    assert bench_generate.run_auto_ab_sweep(args, object()) == 1
    assert calls == [256, 512]


def test_run_auto_ab_sweep_rejects_invalid_values_before_model_setup(
    monkeypatch, capsys
):
    """Malformed threshold sweeps fail closed before any AUTO run starts."""
    monkeypatch.setattr(
        bench_generate,
        "run_auto_ab",
        lambda args, device: pytest.fail("invalid sweep must not start an AUTO run"),
    )
    args = argparse.Namespace(
        auto_threshold=512,
        auto_cold_threshold=None,
        auto_threshold_sweep="256,,512",
    )

    assert bench_generate.run_auto_ab_sweep(args, None) == 2
    assert "empty item" in capsys.readouterr().out


def test_run_impls_rejects_invalid_selection_before_model_setup(
    monkeypatch, capsys
):
    """Invalid backend selections fail closed before device/model work."""
    monkeypatch.setattr(
        bench_generate,
        "_cfg",
        lambda args: pytest.fail("invalid impls must not build a model config"),
    )

    cases = (
        ("blocked", "--impls must include eager"),
        ("eager,warp", "unknown impl 'warp'"),
    )
    for raw, expected in cases:
        assert bench_generate.run_impls(argparse.Namespace(impls=raw), None) == 2
        assert expected in capsys.readouterr().out


def test_run_auto_ab_rejects_invalid_inputs_before_model_setup(
    monkeypatch, capsys
):
    """Invalid AUTO inputs fail closed before model/device work."""
    monkeypatch.setattr(
        bench_generate,
        "_cfg",
        lambda args: pytest.fail("invalid AUTO inputs must not build a model config"),
    )

    cases = (
        (dict(prompts="", auto_threshold=512, auto_cold_threshold=None), "--prompts"),
        (
            dict(prompts="8,fast", auto_threshold=512, auto_cold_threshold=None),
            "invalid prompt length",
        ),
        (
            dict(prompts="8,,16", auto_threshold=512, auto_cold_threshold=None),
            "empty item",
        ),
        (
            dict(prompts="8,-1", auto_threshold=512, auto_cold_threshold=None),
            "prompt lengths must be > 0",
        ),
        (
            dict(prompts="8", auto_threshold=-1, auto_cold_threshold=None),
            "AUTO thresholds must be >= 0",
        ),
        (
            dict(prompts="8", auto_threshold=512, auto_cold_threshold=-1),
            "AUTO thresholds must be >= 0",
        ),
    )
    for values, expected in cases:
        assert bench_generate.run_auto_ab(argparse.Namespace(**values), None) == 2
        assert expected in capsys.readouterr().out


def test_count_torch_cat_restores_hook_after_exception():
    """An interrupted cat probe must not leak its temporary torch.cat hook."""
    original_cat = bench_generate.torch.cat

    def fail_probe():
        raise RuntimeError("stop cat probe")

    with pytest.raises(RuntimeError, match="stop cat probe"):
        bench_generate.count_torch_cat(fail_probe)

    assert bench_generate.torch.cat is original_cat


def test_attn_impl_restores_environment_after_exception(monkeypatch):
    """An interrupted impl run must not leak its dispatch override."""
    monkeypatch.setenv("BDH_ATTN_IMPL", "eager")

    with pytest.raises(RuntimeError, match="stop impl"):
        with bench_generate._attn_impl("blocked"):
            assert os.environ["BDH_ATTN_IMPL"] == "blocked"
            raise RuntimeError("stop impl")

    assert os.environ["BDH_ATTN_IMPL"] == "eager"


def test_attn_impl_restores_unset_environment_after_exception(monkeypatch):
    """An unset impl override must stay unset after an interrupted run."""
    monkeypatch.delenv("BDH_ATTN_IMPL", raising=False)

    with pytest.raises(RuntimeError, match="stop impl"):
        with bench_generate._attn_impl("blocked"):
            assert os.environ["BDH_ATTN_IMPL"] == "blocked"
            raise RuntimeError("stop impl")

    assert "BDH_ATTN_IMPL" not in os.environ


def test_attn_auto_restores_all_threshold_environment_after_exception(monkeypatch):
    """An interrupted AUTO A/B run must restore every temporary gate value."""
    monkeypatch.setenv("BDH_ATTN_AUTO", "0")
    monkeypatch.setenv("BDH_ATTN_AUTO_THRESHOLD", "old-decode")
    monkeypatch.setenv("BDH_ATTN_AUTO_COLD_THRESHOLD", "old-cold")

    with pytest.raises(RuntimeError, match="stop auto"):
        with bench_generate._attn_auto(True, threshold=256, cold_threshold=128):
            assert os.environ["BDH_ATTN_AUTO"] == "1"
            assert os.environ["BDH_ATTN_AUTO_THRESHOLD"] == "256"
            assert os.environ["BDH_ATTN_AUTO_COLD_THRESHOLD"] == "128"
            raise RuntimeError("stop auto")

    assert os.environ["BDH_ATTN_AUTO"] == "0"
    assert os.environ["BDH_ATTN_AUTO_THRESHOLD"] == "old-decode"
    assert os.environ["BDH_ATTN_AUTO_COLD_THRESHOLD"] == "old-cold"


def test_attn_auto_only_overrides_requested_threshold(monkeypatch):
    """An omitted cold gate stays independent during a decode-only override."""
    monkeypatch.setenv("BDH_ATTN_AUTO", "0")
    monkeypatch.setenv("BDH_ATTN_AUTO_THRESHOLD", "old-decode")
    monkeypatch.setenv("BDH_ATTN_AUTO_COLD_THRESHOLD", "old-cold")

    with bench_generate._attn_auto(True, threshold=256):
        assert os.environ["BDH_ATTN_AUTO"] == "1"
        assert os.environ["BDH_ATTN_AUTO_THRESHOLD"] == "256"
        assert os.environ["BDH_ATTN_AUTO_COLD_THRESHOLD"] == "old-cold"

    assert os.environ["BDH_ATTN_AUTO"] == "0"
    assert os.environ["BDH_ATTN_AUTO_THRESHOLD"] == "old-decode"
    assert os.environ["BDH_ATTN_AUTO_COLD_THRESHOLD"] == "old-cold"


def test_attn_auto_disable_preserves_omitted_thresholds(monkeypatch):
    """Disabling AUTO alone must not reset either threshold gate."""
    monkeypatch.setenv("BDH_ATTN_AUTO", "1")
    monkeypatch.setenv("BDH_ATTN_AUTO_THRESHOLD", "old-decode")
    monkeypatch.setenv("BDH_ATTN_AUTO_COLD_THRESHOLD", "old-cold")

    with bench_generate._attn_auto(False):
        assert os.environ["BDH_ATTN_AUTO"] == "0"
        assert os.environ["BDH_ATTN_AUTO_THRESHOLD"] == "old-decode"
        assert os.environ["BDH_ATTN_AUTO_COLD_THRESHOLD"] == "old-cold"

    assert os.environ["BDH_ATTN_AUTO"] == "1"
    assert os.environ["BDH_ATTN_AUTO_THRESHOLD"] == "old-decode"
    assert os.environ["BDH_ATTN_AUTO_COLD_THRESHOLD"] == "old-cold"


def test_attn_auto_does_not_create_omitted_cold_threshold(monkeypatch):
    """A decode-only override must not invent an absent cold gate."""
    monkeypatch.delenv("BDH_ATTN_AUTO", raising=False)
    monkeypatch.delenv("BDH_ATTN_AUTO_THRESHOLD", raising=False)
    monkeypatch.delenv("BDH_ATTN_AUTO_COLD_THRESHOLD", raising=False)

    with bench_generate._attn_auto(True, threshold=256):
        assert os.environ["BDH_ATTN_AUTO"] == "1"
        assert os.environ["BDH_ATTN_AUTO_THRESHOLD"] == "256"
        assert "BDH_ATTN_AUTO_COLD_THRESHOLD" not in os.environ

    assert "BDH_ATTN_AUTO" not in os.environ
    assert "BDH_ATTN_AUTO_THRESHOLD" not in os.environ
    assert "BDH_ATTN_AUTO_COLD_THRESHOLD" not in os.environ


def test_attn_auto_restores_unset_environment_after_exception(monkeypatch):
    """AUTO must not leave newly introduced variables behind."""
    for name in (
        "BDH_ATTN_AUTO",
        "BDH_ATTN_AUTO_THRESHOLD",
        "BDH_ATTN_AUTO_COLD_THRESHOLD",
    ):
        monkeypatch.delenv(name, raising=False)

    with pytest.raises(RuntimeError, match="stop auto"):
        with bench_generate._attn_auto(True, threshold=256, cold_threshold=128):
            assert os.environ["BDH_ATTN_AUTO"] == "1"
            assert os.environ["BDH_ATTN_AUTO_THRESHOLD"] == "256"
            assert os.environ["BDH_ATTN_AUTO_COLD_THRESHOLD"] == "128"
            raise RuntimeError("stop auto")

    for name in (
        "BDH_ATTN_AUTO",
        "BDH_ATTN_AUTO_THRESHOLD",
        "BDH_ATTN_AUTO_COLD_THRESHOLD",
    ):
        assert name not in os.environ
