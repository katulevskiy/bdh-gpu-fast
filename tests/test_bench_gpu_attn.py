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
    assert summary["schema_version"] == 5
    assert summary["status"] == "skip"
    assert summary["reason"] == "cuda_unavailable"
    assert summary["mode"] == "cold"
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
    assert isinstance(summary["cuda_built"], bool)
    assert isinstance(summary["cuda_device_count"], int)
    assert summary["cuda_device_count"] >= 0
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
    assert summary["schema_version"] == 5
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


def test_force_cpu_main_skips_gpu_metadata_when_cuda_is_available(monkeypatch, tmp_path):
    """Forced CPU smoke must not initialize GPU metadata or claim GPU timing."""
    namespace: dict[str, object] = {
        "__name__": "bench_gpu_attn_test",
        "__file__": str(SCRIPT),
    }
    exec(compile(SCRIPT.read_text(), str(SCRIPT), "exec"), namespace)
    torch = namespace["torch"]
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    def fail_if_called(*args, **kwargs):
        raise AssertionError("CPU smoke initialized GPU metadata")

    monkeypatch.setattr(torch.cuda, "get_device_properties", fail_if_called)
    summary_path = tmp_path / "forced-cpu-summary.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
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
    )

    assert namespace["main"]() == 0
    summary = json.loads(summary_path.read_text())
    assert summary["cuda_available"] is True
    assert summary["device"] == "cpu"
    assert summary["timing_scope"] == "cpu"
    assert summary["gpu_name"] is None


def test_timing_backend_skip_preserves_no_timing_claim(monkeypatch, tmp_path):
    """A backend that fails only during timing stays an explicit skip."""
    namespace: dict[str, object] = {
        "__name__": "bench_gpu_attn_test",
        "__file__": str(SCRIPT),
    }
    exec(compile(SCRIPT.read_text(), str(SCRIPT), "exec"), namespace)
    torch = namespace["torch"]
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    calls = 0

    def flaky_backend(q, k, v):
        nonlocal calls
        calls += 1
        if calls > 1:
            raise RuntimeError("timing backend disappeared")
        return q

    def eager_backend(q, k, v):
        return q

    monkeypatch.setitem(namespace, "eager_tril_attn", eager_backend)
    monkeypatch.setitem(
        namespace, "BACKENDS_COLD", {"eager": eager_backend, "flaky": flaky_backend}
    )
    summary_path = tmp_path / "timing-skip-summary.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
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
    )

    assert namespace["main"]() == 0
    summary = json.loads(summary_path.read_text())
    assert summary["status"] == "cpu_smoke"
    assert summary["timing_scope"] == "cpu"
    assert summary["skips"] == [
        {
            "scope": "backend",
            "backend": "flaky",
            "status": "skip",
            "reason": "backend_unavailable",
            "phase": "timing",
            "detail": "RuntimeError: timing backend disappeared",
            "bit_identical": None,
            "allclose_at_1e-4": None,
            "max_abs_delta": None,
            "max_rel_delta": None,
            "median_ms": None,
        }
    ]
    result_by_backend = {item["backend"]: item for item in summary["results"]}
    assert result_by_backend["flaky"]["median_ms"] is None
    assert result_by_backend["eager"]["median_ms"] is not None


def test_cold_reference_uses_raw_strict_lower_triangle():
    """The benchmark reference stays raw score × strict tril, without scaling."""
    namespace: dict[str, object] = {
        "__name__": "bench_gpu_attn_test",
        "__file__": str(SCRIPT),
    }
    exec(compile(SCRIPT.read_text(), str(SCRIPT), "exec"), namespace)
    torch = namespace["torch"]
    q = torch.tensor([[[[1.0, 2.0], [3.0, -1.0], [2.0, 1.0]]]])
    k = torch.tensor([[[[2.0, 1.0], [-1.0, 3.0], [1.0, -2.0]]]])
    v = torch.tensor([[[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]]])

    expected = (q @ k.transpose(-2, -1)).tril(diagonal=-1) @ v
    actual = namespace["BACKENDS_COLD"]["eager"](q, k, v)

    assert torch.equal(actual, expected)


def test_decode_reference_uses_raw_past_scores_without_scaling():
    """Packed past decode uses raw score×V, with every past key valid."""
    namespace: dict[str, object] = {
        "__name__": "bench_gpu_attn_test",
        "__file__": str(SCRIPT),
    }
    exec(compile(SCRIPT.read_text(), str(SCRIPT), "exec"), namespace)
    torch = namespace["torch"]
    q = torch.tensor(
        [[[[2.0, 1.0]], [[-1.0, 3.0]]]],
    )
    k = torch.tensor(
        [
            [
                [[1.0, 0.0], [0.0, 3.0], [-1.0, 2.0]],
                [[2.0, 1.0], [1.0, -1.0], [0.0, 2.0]],
            ]
        ]
    )
    v = torch.tensor([[[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]]])

    expected = (q @ k.transpose(-2, -1)) @ v
    actual = namespace["BACKENDS_DECODE"]["eager"](q, k, v)

    assert torch.equal(actual, expected)


def test_no_cuda_skip_includes_native_cuda_handoff(tmp_path):
    """A cold-path skip keeps the optional native CUDA handoff actionable."""
    summary_path = tmp_path / "gpu-summary.json"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--mode",
            "cold",
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
    assert summary["status"] == "skip"
    assert summary["timing_scope"] == "none"
    assert summary["commands"]["native_cuda_optional"] == [
        "BDH_BUILD_EXT=1 BDH_BUILD_CUDA=1 pip install -e . --no-build-isolation",
        "python benchmarks/bench_gpu_attn.py",
    ]
    assert (
        "BDH_BUILD_EXT=1 BDH_BUILD_CUDA=1 pip install -e . --no-build-isolation"
        in result.stdout
    )
    assert "median ms" not in result.stdout


def test_no_cuda_decode_skip_preserves_requested_mode(tmp_path):
    """Decode handoff diagnostics must not silently report the cold default."""
    summary_path = tmp_path / "decode-summary.json"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--mode",
            "decode",
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
    assert summary["status"] == "skip"
    assert summary["mode"] == "decode"
    assert summary["timing_scope"] == "none"
    assert "median ms" not in result.stdout


def test_no_cuda_skip_prints_structured_cold_diagnostics(tmp_path):
    """Cold CUDA skips expose diagnostics without implying a CPU measurement."""
    summary_path = tmp_path / "gpu-summary.json"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--mode",
            "cold",
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
    summary_lines = [
        line.removeprefix("GPU_ATTN_SUMMARY ")
        for line in result.stdout.splitlines()
        if line.startswith("GPU_ATTN_SUMMARY ")
    ]

    assert len(summary_lines) == 1
    assert json.loads(summary_lines[0]) == summary
    assert summary["device"] == "cpu"
    assert summary["mode"] == "cold"
    assert summary["timing_scope"] == "none"
    assert summary["cuda_available"] is False
    assert summary["gpu_name"] is None
    assert isinstance(summary["cuda_built"], bool)
    assert isinstance(summary["cuda_device_count"], int)
    assert summary["cuda_device_count"] >= 0
    assert summary["torch_version"]
    assert summary["cuda_version"] is None or isinstance(summary["cuda_version"], str)
    assert summary["backend_info"]
    assert "torch=" in result.stdout
    assert "cuda_available=false" in result.stdout
    assert f"cuda_built={summary['cuda_built']}" in result.stdout
    assert f"cuda_device_count={summary['cuda_device_count']}" in result.stdout
    assert "backend_info=" in result.stdout
    assert "median ms" not in result.stdout
