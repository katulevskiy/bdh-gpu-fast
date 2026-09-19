"""CPU-safe smoke coverage for the AUTO threshold sweep harness."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "bdh_bench_generate", ROOT / "benchmarks" / "bench_generate.py"
)
assert _SPEC is not None and _SPEC.loader is not None
bench_generate = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(bench_generate)


def test_threshold_sweep_smoke_dedupes_and_preserves_cold_gate(monkeypatch, capsys):
    """Sweep order is stable and an explicit cold gate survives each run."""
    seen: list[tuple[int, int | None, str | None]] = []

    def fake_run_auto_ab(args, device):
        assert device.type == "cpu"
        seen.append(
            (args.auto_threshold, args.auto_cold_threshold, args.auto_threshold_sweep)
        )
        return 0

    monkeypatch.setattr(bench_generate, "run_auto_ab", fake_run_auto_ab)
    args = argparse.Namespace(
        auto_threshold_sweep="512, 256,512, 1024",
        auto_cold_threshold=256,
    )

    assert bench_generate.run_auto_ab_sweep(args, torch.device("cpu")) == 0
    assert seen == [
        (512, 256, None),
        (256, 256, None),
        (1024, 256, None),
    ]
    output = capsys.readouterr().out
    assert "decode thresholds=[512, 256, 1024]" in output
    assert "cold_thr=256" in output


@pytest.mark.parametrize("raw", ["", "1,,2", "-1,2", "nope,2"])
def test_threshold_sweep_rejects_malformed_values(raw):
    with pytest.raises(ValueError):
        bench_generate._parse_threshold_sweep(raw)
