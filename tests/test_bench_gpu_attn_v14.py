"""CPU-safe v14 contracts for truthful GPU skip stdout diagnostics."""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "benchmarks" / "bench_gpu_attn.py"


def test_runtime_skip_stdout_preserves_cuda_availability(
    monkeypatch, tmp_path, capsys
):
    """Runtime skips must not contradict JSON CUDA availability diagnostics."""
    namespace: dict[str, object] = {
        "__name__": "bench_gpu_attn_v14_test",
        "__file__": str(SCRIPT),
    }
    exec(compile(SCRIPT.read_text(), str(SCRIPT), "exec"), namespace)
    torch = namespace["torch"]
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.backends.cuda, "is_built", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)
    summary_path = tmp_path / "runtime-skip-summary.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [str(SCRIPT), "--json-out", str(summary_path)],
    )

    assert namespace["main"]() == 0
    summary = json.loads(summary_path.read_text())
    output = capsys.readouterr().out

    assert summary["status"] == "skip"
    assert summary["cuda_available"] is True
    assert summary["cuda_runtime_state"] == "no_visible_device"
    assert summary["timing_scope"] == "none"
    assert "cuda_available=true" in output
    assert "cuda_available=false" not in output
    assert "median ms" not in output
