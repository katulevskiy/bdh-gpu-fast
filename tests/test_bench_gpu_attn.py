"""CPU contracts for the GPU measurement harness."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "benchmarks" / "bench_gpu_attn.py"


def test_no_cuda_skip_is_actionable_and_clean(tmp_path):
    """The CPU box reports a structured handoff instead of raising or timing."""
    summary_path = tmp_path / "gpu-summary.json"
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--json-out", str(summary_path)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "GPU_ATTN_SKIP status=skip reason=cuda_unavailable" in result.stdout
    assert "GPU_ATTN_COMMANDS (from OPT_BACKLOG.md):" in result.stdout
    assert "python benchmarks/bench_gpu_attn.py --mode decode --T 512" in result.stdout
    assert "python benchmarks/bench_gpu_attn.py --dtype bfloat16" in result.stdout
    assert "python benchmarks/bench_gpu_attn.py --dtype float16" in result.stdout
    assert "median ms" not in result.stdout

    summary = json.loads(summary_path.read_text())
    assert summary["status"] == "skip"
    assert summary["reason"] == "cuda_unavailable"
    assert summary["cuda_available"] is False
    assert summary["commands"]["cold"]
    assert summary["commands"]["decode"]
    assert summary["commands"]["dtype"] == [
        "python benchmarks/bench_gpu_attn.py --dtype bfloat16",
        "python benchmarks/bench_gpu_attn.py --dtype float16",
    ]


def test_backend_matrix_includes_online_decode():
    namespace: dict[str, object] = {"__name__": "bench_gpu_attn_test", "__file__": str(SCRIPT)}
    exec(compile(SCRIPT.read_text(), str(SCRIPT), "exec"), namespace)
    assert list(namespace["BACKENDS_COLD"]) == [
        "eager",
        "blocked",
        "online",
        "triton",
        "cuda",
    ]
    assert list(namespace["BACKENDS_DECODE"]) == [
        "eager",
        "blocked",
        "online",
        "triton",
        "cuda",
    ]
