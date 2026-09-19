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
    assert summary["schema_version"] == 4
    assert summary["status"] == "skip"
    assert summary["reason"] == "cuda_unavailable"
    assert summary["skips"] == [
        {
            "scope": "run",
            "status": "skip",
            "reason": "cuda_unavailable",
            "detail": "torch.cuda.is_available() is false",
        }
    ]
    assert summary["timing_scope"] == "none"
    assert summary["cuda_available"] is False
    assert summary["commands"]["cold"]
    assert summary["commands"]["decode"]
    assert summary["commands"]["dtype"] == [
        "python benchmarks/bench_gpu_attn.py --dtype bfloat16",
        "python benchmarks/bench_gpu_attn.py --dtype float16",
    ]


def test_force_cpu_summary_does_not_claim_gpu_timings(tmp_path):
    """Forced smoke timings are explicitly marked as CPU-only."""
    summary_path = tmp_path / "cpu-summary.json"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--force-cpu",
            "--warmup",
            "0",
            "--iters",
            "1",
            "--B",
            "1",
            "--H",
            "1",
            "--T",
            "2",
            "--N",
            "2",
            "--D",
            "2",
            "--json-out",
            str(summary_path),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    summary = json.loads(summary_path.read_text())
    assert summary["schema_version"] == 4
    assert summary["status"] == "cpu_smoke"
    assert summary["reason"] == "force_cpu"
    assert summary["device"] == "cpu"
    assert summary["timing_scope"] == "cpu"
    assert summary["gpu_name"] is None
    assert summary["skips"] == []
    assert all(item["status"] == "ok" for item in summary["results"])
    assert all(item["reason"] is None for item in summary["results"])
    assert "do not claim GPU wins" in result.stdout


def test_backend_matrix_includes_online_decode():
    namespace: dict[str, object] = {
        "__name__": "bench_gpu_attn_test",
        "__file__": str(SCRIPT),
    }
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


def test_backend_skip_result_has_no_timing_claim():
    namespace: dict[str, object] = {"__name__": "bench_gpu_attn_test", "__file__": str(SCRIPT)}
    exec(compile(SCRIPT.read_text(), str(SCRIPT), "exec"), namespace)
    skip = namespace["_backend_skip"]("triton", phase="correctness", exc=RuntimeError("not available"))
    assert skip == {
        "backend": "triton",
        "status": "skip",
        "reason": "backend_unavailable",
        "phase": "correctness",
        "detail": "RuntimeError: not available",
        "bit_identical": None,
        "allclose_at_1e-4": None,
        "max_abs_delta": None,
        "max_rel_delta": None,
        "median_ms": None,
    }


def test_force_cpu_overrides_available_cuda():
    """CPU smoke remains CPU-only even when CUDA is present."""
    namespace: dict[str, object] = {
        "__name__": "bench_gpu_attn_test",
        "__file__": str(SCRIPT),
    }
    exec(compile(SCRIPT.read_text(), str(SCRIPT), "exec"), namespace)
    select_device = namespace["_select_device"]
    assert select_device(cuda_available=True, force_cpu=True).type == "cpu"
    assert select_device(cuda_available=True, force_cpu=False).type == "cuda"
    assert select_device(cuda_available=False, force_cpu=True).type == "cpu"


def test_force_cpu_bench_does_not_synchronize_cuda(monkeypatch):
    """CPU smoke must not synchronize CUDA merely because CUDA is present."""
    namespace: dict[str, object] = {
        "__name__": "bench_gpu_attn_test",
        "__file__": str(SCRIPT),
    }
    exec(compile(SCRIPT.read_text(), str(SCRIPT), "exec"), namespace)
    torch = namespace["torch"]
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    def fail_if_called():
        raise AssertionError("CPU smoke synchronized CUDA")

    monkeypatch.setattr(torch.cuda, "synchronize", fail_if_called)
    device = torch.device("cpu")
    namespace["_bench"](
        lambda value: value,
        (torch.ones(1),),
        device=device,
        warmup=0,
        iters=1,
    )


def test_no_cuda_skip_returns_before_measurement(monkeypatch):
    """Unavailable CUDA must not enter device selection or timing code."""
    namespace: dict[str, object] = {
        "__name__": "bench_gpu_attn_test",
        "__file__": str(SCRIPT),
    }
    exec(compile(SCRIPT.read_text(), str(SCRIPT), "exec"), namespace)
    torch = namespace["torch"]
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    def fail_if_called(*args, **kwargs):
        raise AssertionError("CUDA-unavailable skip entered measurement")

    monkeypatch.setitem(namespace, "_select_device", fail_if_called)
    monkeypatch.setitem(namespace, "_bench", fail_if_called)
    monkeypatch.setattr(sys, "argv", [str(SCRIPT)])

    assert namespace["main"]() == 0
