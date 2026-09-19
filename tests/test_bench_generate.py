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


def test_resolve_device_is_cpu_safe_when_cuda_is_unavailable(monkeypatch, capsys):
    """Auto uses CPU, while an explicit unavailable CUDA request cleanly skips."""
    monkeypatch.setattr(bench_generate.torch.cuda, "is_available", lambda: False)

    assert bench_generate._resolve_device(
        argparse.Namespace(device="auto")
    ) == bench_generate.torch.device("cpu")
    assert bench_generate._resolve_device(argparse.Namespace(device="cuda")) is None
    assert "torch.cuda.is_available() is False" in capsys.readouterr().out


def test_resolve_device_honors_explicit_cpu_when_cuda_is_available(monkeypatch):
    """Explicit CPU stays CPU instead of following CUDA availability."""
    monkeypatch.setattr(bench_generate.torch.cuda, "is_available", lambda: True)

    assert bench_generate._resolve_device(
        argparse.Namespace(device="cpu")
    ) == bench_generate.torch.device("cpu")


def test_should_run_labels_cpu_triton_fallback_without_gpu_claim(monkeypatch):
    """CPU Triton probes stay runnable but report the non-GPU fallback."""
    monkeypatch.setattr(
        bench_generate,
        "backend_info",
        lambda: {
            "effective": "blocked",
            "has_triton": False,
            "has_cuda_ext": False,
        },
    )

    assert bench_generate._should_run(
        "triton", bench_generate.torch.device("cpu")
    ) == (True, "fallback effective=blocked (no CUDA Triton kernel)")


def test_should_run_labels_cpu_cuda_fallback_without_gpu_claim(monkeypatch):
    """CPU CUDA probes stay runnable but report the pure-PyTorch fallback."""
    monkeypatch.setattr(
        bench_generate,
        "backend_info",
        lambda: {
            "effective": "eager",
            "has_triton": False,
            "has_cuda_ext": False,
        },
    )

    assert bench_generate._should_run(
        "cuda", bench_generate.torch.device("cpu")
    ) == (True, "effective=eager (pure-PyTorch ref; no native ext)")


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


def test_run_auto_ab_sweep_without_sweep_delegates_unchanged(monkeypatch):
    """No sweep keeps the single-run path and its original arguments."""
    calls = []
    args = argparse.Namespace(auto_threshold_sweep=None)
    device = object()

    def fake_run_auto_ab(got_args, got_device):
        calls.append((got_args, got_device))
        return 7

    monkeypatch.setattr(bench_generate, "run_auto_ab", fake_run_auto_ab)

    assert bench_generate.run_auto_ab_sweep(args, device) == 7
    assert calls == [(args, device)]


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


def test_attn_auto_nested_override_restores_outer_gates(monkeypatch):
    """Nested AUTO probes restore the outer gate values before final cleanup."""
    monkeypatch.setenv("BDH_ATTN_AUTO", "old-auto")
    monkeypatch.setenv("BDH_ATTN_AUTO_THRESHOLD", "old-decode")
    monkeypatch.setenv("BDH_ATTN_AUTO_COLD_THRESHOLD", "old-cold")

    with bench_generate._attn_auto(True, threshold=256, cold_threshold=128):
        assert os.environ["BDH_ATTN_AUTO"] == "1"
        assert os.environ["BDH_ATTN_AUTO_THRESHOLD"] == "256"
        assert os.environ["BDH_ATTN_AUTO_COLD_THRESHOLD"] == "128"

        with bench_generate._attn_auto(False, threshold=512):
            assert os.environ["BDH_ATTN_AUTO"] == "0"
            assert os.environ["BDH_ATTN_AUTO_THRESHOLD"] == "512"
            assert os.environ["BDH_ATTN_AUTO_COLD_THRESHOLD"] == "128"

        assert os.environ["BDH_ATTN_AUTO"] == "1"
        assert os.environ["BDH_ATTN_AUTO_THRESHOLD"] == "256"
        assert os.environ["BDH_ATTN_AUTO_COLD_THRESHOLD"] == "128"

    assert os.environ["BDH_ATTN_AUTO"] == "old-auto"
    assert os.environ["BDH_ATTN_AUTO_THRESHOLD"] == "old-decode"
    assert os.environ["BDH_ATTN_AUTO_COLD_THRESHOLD"] == "old-cold"


def test_attn_auto_nested_exception_restores_outer_gates(monkeypatch):
    """An interrupted nested AUTO probe must restore the outer override."""
    monkeypatch.setenv("BDH_ATTN_AUTO", "old-auto")
    monkeypatch.setenv("BDH_ATTN_AUTO_THRESHOLD", "old-decode")
    monkeypatch.setenv("BDH_ATTN_AUTO_COLD_THRESHOLD", "old-cold")

    with bench_generate._attn_auto(True, threshold=256, cold_threshold=128):
        with pytest.raises(RuntimeError, match="stop nested auto"):
            with bench_generate._attn_auto(False, threshold=512):
                assert os.environ["BDH_ATTN_AUTO"] == "0"
                assert os.environ["BDH_ATTN_AUTO_THRESHOLD"] == "512"
                assert os.environ["BDH_ATTN_AUTO_COLD_THRESHOLD"] == "128"
                raise RuntimeError("stop nested auto")

        assert os.environ["BDH_ATTN_AUTO"] == "1"
        assert os.environ["BDH_ATTN_AUTO_THRESHOLD"] == "256"
        assert os.environ["BDH_ATTN_AUTO_COLD_THRESHOLD"] == "128"

    assert os.environ["BDH_ATTN_AUTO"] == "old-auto"
    assert os.environ["BDH_ATTN_AUTO_THRESHOLD"] == "old-decode"
    assert os.environ["BDH_ATTN_AUTO_COLD_THRESHOLD"] == "old-cold"


def test_attn_auto_nested_cold_exception_restores_outer_gates(monkeypatch):
    """An interrupted cold-only nested probe restores the outer decode gate."""
    monkeypatch.setenv("BDH_ATTN_AUTO", "old-auto")
    monkeypatch.setenv("BDH_ATTN_AUTO_THRESHOLD", "old-decode")
    monkeypatch.setenv("BDH_ATTN_AUTO_COLD_THRESHOLD", "old-cold")

    with bench_generate._attn_auto(True, threshold=256, cold_threshold=128):
        with pytest.raises(RuntimeError, match="stop nested cold auto"):
            with bench_generate._attn_auto(True, cold_threshold=512):
                assert os.environ["BDH_ATTN_AUTO"] == "1"
                assert os.environ["BDH_ATTN_AUTO_THRESHOLD"] == "256"
                assert os.environ["BDH_ATTN_AUTO_COLD_THRESHOLD"] == "512"
                raise RuntimeError("stop nested cold auto")

        assert os.environ["BDH_ATTN_AUTO"] == "1"
        assert os.environ["BDH_ATTN_AUTO_THRESHOLD"] == "256"
        assert os.environ["BDH_ATTN_AUTO_COLD_THRESHOLD"] == "128"

    assert os.environ["BDH_ATTN_AUTO"] == "old-auto"
    assert os.environ["BDH_ATTN_AUTO_THRESHOLD"] == "old-decode"
    assert os.environ["BDH_ATTN_AUTO_COLD_THRESHOLD"] == "old-cold"


def test_attn_auto_applies_zero_thresholds_and_restores_them(monkeypatch):
    """Zero is a valid strict gate and must not be treated as omitted."""
    monkeypatch.setenv("BDH_ATTN_AUTO", "0")
    monkeypatch.setenv("BDH_ATTN_AUTO_THRESHOLD", "old-decode")
    monkeypatch.setenv("BDH_ATTN_AUTO_COLD_THRESHOLD", "old-cold")

    with bench_generate._attn_auto(True, threshold=0, cold_threshold=0):
        assert os.environ["BDH_ATTN_AUTO"] == "1"
        assert os.environ["BDH_ATTN_AUTO_THRESHOLD"] == "0"
        assert os.environ["BDH_ATTN_AUTO_COLD_THRESHOLD"] == "0"

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


def test_attn_auto_only_overrides_requested_cold_threshold(monkeypatch):
    """An omitted decode gate stays independent during a cold-only override."""
    monkeypatch.setenv("BDH_ATTN_AUTO", "0")
    monkeypatch.setenv("BDH_ATTN_AUTO_THRESHOLD", "old-decode")
    monkeypatch.setenv("BDH_ATTN_AUTO_COLD_THRESHOLD", "old-cold")

    with bench_generate._attn_auto(True, cold_threshold=128):
        assert os.environ["BDH_ATTN_AUTO"] == "1"
        assert os.environ["BDH_ATTN_AUTO_THRESHOLD"] == "old-decode"
        assert os.environ["BDH_ATTN_AUTO_COLD_THRESHOLD"] == "128"

    assert os.environ["BDH_ATTN_AUTO"] == "0"
    assert os.environ["BDH_ATTN_AUTO_THRESHOLD"] == "old-decode"
    assert os.environ["BDH_ATTN_AUTO_COLD_THRESHOLD"] == "old-cold"


def test_attn_auto_does_not_create_omitted_decode_threshold(monkeypatch):
    """A cold-only override must not invent an absent decode gate."""
    monkeypatch.delenv("BDH_ATTN_AUTO", raising=False)
    monkeypatch.delenv("BDH_ATTN_AUTO_THRESHOLD", raising=False)
    monkeypatch.delenv("BDH_ATTN_AUTO_COLD_THRESHOLD", raising=False)

    with bench_generate._attn_auto(True, cold_threshold=128):
        assert os.environ["BDH_ATTN_AUTO"] == "1"
        assert "BDH_ATTN_AUTO_THRESHOLD" not in os.environ
        assert os.environ["BDH_ATTN_AUTO_COLD_THRESHOLD"] == "128"

    assert "BDH_ATTN_AUTO" not in os.environ
    assert "BDH_ATTN_AUTO_THRESHOLD" not in os.environ
    assert "BDH_ATTN_AUTO_COLD_THRESHOLD" not in os.environ


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


def test_attn_auto_restores_omitted_threshold_after_exception(monkeypatch):
    """An interrupted single-gate override must not leak the omitted gate."""
    monkeypatch.setenv("BDH_ATTN_AUTO", "0")
    monkeypatch.setenv("BDH_ATTN_AUTO_THRESHOLD", "old-decode")
    monkeypatch.delenv("BDH_ATTN_AUTO_COLD_THRESHOLD", raising=False)

    with pytest.raises(RuntimeError, match="stop auto"):
        with bench_generate._attn_auto(True, cold_threshold=128):
            assert os.environ["BDH_ATTN_AUTO"] == "1"
            assert os.environ["BDH_ATTN_AUTO_THRESHOLD"] == "old-decode"
            assert os.environ["BDH_ATTN_AUTO_COLD_THRESHOLD"] == "128"
            raise RuntimeError("stop auto")

    assert os.environ["BDH_ATTN_AUTO"] == "0"
    assert os.environ["BDH_ATTN_AUTO_THRESHOLD"] == "old-decode"
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
