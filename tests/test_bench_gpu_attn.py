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
    assert summary["schema_version"] == 12
    assert summary["status"] == "skip"
    assert summary["reason"] == "cuda_unavailable"
    assert summary["mode"] == "cold"
    assert summary["skips"] == [
        {
            "scope": "run",
            "status": "skip",
            "reason": "cuda_unavailable",
            "detail": "torch.cuda.is_available() is false",
            "timing_scope": "none",
            "cuda_available": False,
            "cuda_runtime_state": summary["cuda_runtime_state"],
            "cuda_built": summary["cuda_built"],
            "cuda_built_probe_error": summary["cuda_built_probe_error"],
            "cuda_device_count": summary["cuda_device_count"],
            "cuda_device_probe_error": summary["cuda_device_probe_error"],
            "cuda_available_probe_error": summary["cuda_available_probe_error"],
        }
    ]
    assert summary["timing_scope"] == "none"
    assert "results" not in summary
    assert summary["cuda_available"] is False
    assert isinstance(summary["cuda_built"], bool)
    assert isinstance(summary["cuda_device_count"], int)
    assert summary["cuda_device_count"] >= 0
    expected_state = (
        "not_built"
        if not summary["cuda_built"]
        else (
            "no_visible_device"
            if summary["cuda_device_count"] == 0
            else "runtime_unavailable"
        )
    )
    assert summary["cuda_runtime_state"] == expected_state
    assert f"cuda_runtime_state={expected_state}" in result.stdout
    assert "timing_scope=none" in result.stdout
    assert summary["commands"]["cold"]
    assert summary["commands"]["decode"]
    assert summary["commands"]["dtype"] == [
        "python benchmarks/bench_gpu_attn.py --dtype bfloat16",
        "python benchmarks/bench_gpu_attn.py --dtype float16",
    ]


def test_cuda_runtime_diagnostics_classify_states(monkeypatch):
    """Skip diagnostics distinguish build, visibility, and runtime gaps."""
    namespace: dict[str, object] = {
        "__name__": "bench_gpu_attn_test",
        "__file__": str(SCRIPT),
    }
    exec(compile(SCRIPT.read_text(), str(SCRIPT), "exec"), namespace)
    torch = namespace["torch"]
    diagnostics = namespace["_cuda_runtime_diagnostics"]

    monkeypatch.setattr(torch.backends.cuda, "is_built", lambda: False)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)
    assert diagnostics(cuda_available=False)["cuda_runtime_state"] == "not_built"

    monkeypatch.setattr(torch.backends.cuda, "is_built", lambda: True)
    assert diagnostics(cuda_available=False)["cuda_runtime_state"] == "no_visible_device"

    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    assert diagnostics(cuda_available=False)["cuda_runtime_state"] == "runtime_unavailable"
    assert diagnostics(cuda_available=True)["cuda_runtime_state"] == "available"


def test_cuda_runtime_diagnostics_survives_device_probe_failure(monkeypatch):
    """A failing CUDA device probe remains an honest runtime skip."""
    namespace: dict[str, object] = {
        "__name__": "bench_gpu_attn_test",
        "__file__": str(SCRIPT),
    }
    exec(compile(SCRIPT.read_text(), str(SCRIPT), "exec"), namespace)
    torch = namespace["torch"]
    diagnostics = namespace["_cuda_runtime_diagnostics"]

    monkeypatch.setattr(torch.backends.cuda, "is_built", lambda: True)

    def fail_device_probe():
        raise RuntimeError("driver query failed")

    monkeypatch.setattr(torch.cuda, "device_count", fail_device_probe)
    result = diagnostics(cuda_available=False)

    assert result["cuda_built"] is True
    assert result["cuda_device_count"] == 0
    assert result["cuda_device_probe_error"] == "RuntimeError: driver query failed"
    assert result["cuda_runtime_state"] == "runtime_unavailable"


def test_cuda_runtime_diagnostics_survives_build_probe_failure(monkeypatch):
    """A failing CUDA build probe remains an honest runtime skip."""
    namespace: dict[str, object] = {
        "__name__": "bench_gpu_attn_test",
        "__file__": str(SCRIPT),
    }
    exec(compile(SCRIPT.read_text(), str(SCRIPT), "exec"), namespace)
    torch = namespace["torch"]
    diagnostics = namespace["_cuda_runtime_diagnostics"]

    def fail_build_probe():
        raise RuntimeError("CUDA build query failed")

    monkeypatch.setattr(torch.backends.cuda, "is_built", fail_build_probe)
    result = diagnostics(cuda_available=False)

    assert result["cuda_built"] is False
    assert result["cuda_built_probe_error"] == (
        "RuntimeError: CUDA build query failed"
    )
    assert result["cuda_device_count"] == 0
    assert result["cuda_device_probe_error"] is None
    assert result["cuda_runtime_state"] == "build_probe_failed"


def test_no_cuda_skip_stays_structured_when_availability_probe_fails(
    monkeypatch, tmp_path, capsys
):
    """A failing availability probe remains an honest CPU-safe skip."""
    namespace: dict[str, object] = {
        "__name__": "bench_gpu_attn_test",
        "__file__": str(SCRIPT),
    }
    exec(compile(SCRIPT.read_text(), str(SCRIPT), "exec"), namespace)
    torch = namespace["torch"]
    monkeypatch.setattr(torch.backends.cuda, "is_built", lambda: True)

    def fail_availability_probe():
        raise RuntimeError("driver availability failed")

    monkeypatch.setattr(torch.cuda, "is_available", fail_availability_probe)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    summary_path = tmp_path / "availability-failure-summary.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [str(SCRIPT), "--json-out", str(summary_path)],
    )

    assert namespace["main"]() == 0
    summary = json.loads(summary_path.read_text())
    output = capsys.readouterr().out

    assert summary["status"] == "skip"
    assert summary["reason"] == "cuda_unavailable"
    assert summary["timing_scope"] == "none"
    assert summary["cuda_available"] is False
    assert summary["cuda_available_probe_error"] == (
        "RuntimeError: driver availability failed"
    )
    assert summary["cuda_runtime_state"] == "runtime_unavailable"
    assert summary["skips"][0]["detail"] == (
        "torch.cuda.is_available() raised RuntimeError: driver availability failed"
    )
    assert summary["skips"][0]["cuda_available_probe_error"] == (
        "RuntimeError: driver availability failed"
    )
    assert summary["backend_info"] == {
        "status": "unavailable",
        "detail": "RuntimeError: driver availability failed",
    }
    assert "results" not in summary
    assert "cuda_available_probe_error=RuntimeError: driver availability failed" in output
    assert "median ms" not in output


def test_no_cuda_skip_stays_structured_when_device_probe_fails(
    monkeypatch, tmp_path, capsys
):
    """The run-level skip keeps the failed driver probe diagnostics-only."""
    namespace: dict[str, object] = {
        "__name__": "bench_gpu_attn_test",
        "__file__": str(SCRIPT),
    }
    exec(compile(SCRIPT.read_text(), str(SCRIPT), "exec"), namespace)
    torch = namespace["torch"]
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.cuda, "is_built", lambda: True)

    def fail_device_probe():
        raise RuntimeError("driver query failed")

    monkeypatch.setattr(torch.cuda, "device_count", fail_device_probe)
    summary_path = tmp_path / "runtime-failure-summary.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [str(SCRIPT), "--json-out", str(summary_path)],
    )

    assert namespace["main"]() == 0
    summary = json.loads(summary_path.read_text())
    output = capsys.readouterr().out

    assert summary["status"] == "skip"
    assert summary["reason"] == "cuda_unavailable"
    assert summary["timing_scope"] == "none"
    assert summary["cuda_built"] is True
    assert summary["cuda_device_count"] == 0
    assert summary["cuda_device_probe_error"] == "RuntimeError: driver query failed"
    assert summary["cuda_runtime_state"] == "runtime_unavailable"
    assert summary["skips"][0]["cuda_runtime_state"] == "runtime_unavailable"
    assert summary["skips"][0]["cuda_device_probe_error"] == "RuntimeError: driver query failed"
    assert "results" not in summary
    assert "GPU_ATTN_SKIP status=skip reason=cuda_unavailable" in output
    assert "timing_scope=none" in output
    assert "cuda_device_probe_error=RuntimeError: driver query failed" in output
    assert "median ms" not in output


def test_device_probe_failure_skips_before_cuda_measurement(
    monkeypatch, tmp_path, capsys
):
    """A true availability probe cannot bypass a failed device probe."""
    namespace: dict[str, object] = {
        "__name__": "bench_gpu_attn_test",
        "__file__": str(SCRIPT),
    }
    exec(compile(SCRIPT.read_text(), str(SCRIPT), "exec"), namespace)
    torch = namespace["torch"]
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.backends.cuda, "is_built", lambda: True)

    def fail_device_probe():
        raise RuntimeError("driver query failed")

    monkeypatch.setattr(torch.cuda, "device_count", fail_device_probe)
    monkeypatch.setitem(
        namespace,
        "_select_device",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("device-probe failure entered measurement")
        ),
    )
    summary_path = tmp_path / "device-probe-failure-summary.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [str(SCRIPT), "--json-out", str(summary_path)],
    )

    assert namespace["main"]() == 0
    summary = json.loads(summary_path.read_text())
    output = capsys.readouterr().out

    assert summary["status"] == "skip"
    assert summary["reason"] == "cuda_unavailable"
    assert summary["timing_scope"] == "none"
    assert summary["cuda_available"] is True
    assert summary["cuda_runtime_state"] == "runtime_unavailable"
    assert summary["cuda_device_count"] == 0
    assert summary["cuda_device_probe_error"] == (
        "RuntimeError: driver query failed"
    )
    assert summary["skips"] == [
        {
            "scope": "run",
            "status": "skip",
            "reason": "cuda_unavailable",
            "detail": "CUDA device probe raised RuntimeError: driver query failed",
            "timing_scope": "none",
            "cuda_available": True,
            "cuda_runtime_state": "runtime_unavailable",
            "cuda_built": True,
            "cuda_built_probe_error": None,
            "cuda_device_count": 0,
            "cuda_device_probe_error": "RuntimeError: driver query failed",
            "cuda_available_probe_error": None,
        }
    ]
    assert "GPU_ATTN_SKIP status=skip reason=cuda_unavailable" in output
    assert "CUDA device probe raised RuntimeError: driver query failed" in output
    assert "median ms" not in output


def test_no_visible_device_skips_before_cuda_measurement(
    monkeypatch, tmp_path, capsys
):
    """A true availability probe with no visible device stays diagnostics-only."""
    namespace: dict[str, object] = {
        "__name__": "bench_gpu_attn_test",
        "__file__": str(SCRIPT),
    }
    exec(compile(SCRIPT.read_text(), str(SCRIPT), "exec"), namespace)
    torch = namespace["torch"]
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.backends.cuda, "is_built", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)

    def fail_if_called(*args, **kwargs):
        raise AssertionError("no-visible-device skip entered measurement")

    monkeypatch.setitem(namespace, "_select_device", fail_if_called)
    monkeypatch.setitem(namespace, "_bench", fail_if_called)
    summary_path = tmp_path / "no-visible-device-summary.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [str(SCRIPT), "--json-out", str(summary_path)],
    )

    assert namespace["main"]() == 0
    summary = json.loads(summary_path.read_text())
    output = capsys.readouterr().out

    assert summary["status"] == "skip"
    assert summary["reason"] == "cuda_unavailable"
    assert summary["timing_scope"] == "none"
    assert summary["cuda_available"] is True
    assert summary["cuda_built"] is True
    assert summary["cuda_device_count"] == 0
    assert summary["cuda_device_probe_error"] is None
    assert summary["cuda_runtime_state"] == "no_visible_device"
    assert summary["skips"] == [
        {
            "scope": "run",
            "status": "skip",
            "reason": "cuda_unavailable",
            "detail": "CUDA runtime state is no_visible_device",
            "timing_scope": "none",
            "cuda_available": True,
            "cuda_runtime_state": "no_visible_device",
            "cuda_built": True,
            "cuda_built_probe_error": None,
            "cuda_device_count": 0,
            "cuda_device_probe_error": None,
            "cuda_available_probe_error": None,
        }
    ]
    assert "CUDA runtime state is no_visible_device" in output
    assert "timing_scope=none" in output
    assert "median ms" not in output


def test_build_probe_failure_skips_before_cuda_measurement(
    monkeypatch, tmp_path, capsys
):
    """A failing CUDA build probe remains a structured pre-measurement skip."""
    namespace: dict[str, object] = {
        "__name__": "bench_gpu_attn_test",
        "__file__": str(SCRIPT),
    }
    exec(compile(SCRIPT.read_text(), str(SCRIPT), "exec"), namespace)
    torch = namespace["torch"]
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    def fail_build_probe():
        raise RuntimeError("CUDA build query failed")

    monkeypatch.setattr(torch.backends.cuda, "is_built", fail_build_probe)
    monkeypatch.setitem(
        namespace,
        "backend_info",
        lambda: {"status": "unavailable", "detail": "build probe failed"},
    )

    def fail_if_called(*args, **kwargs):
        raise AssertionError("build-probe failure entered measurement")

    monkeypatch.setattr(torch.cuda, "device_count", fail_if_called)
    monkeypatch.setitem(namespace, "_select_device", fail_if_called)
    summary_path = tmp_path / "build-failure-summary.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [str(SCRIPT), "--json-out", str(summary_path)],
    )

    assert namespace["main"]() == 0
    summary = json.loads(summary_path.read_text())
    output = capsys.readouterr().out

    assert summary["status"] == "skip"
    assert summary["reason"] == "cuda_unavailable"
    assert summary["timing_scope"] == "none"
    assert summary["cuda_available"] is True
    assert summary["cuda_built"] is False
    assert summary["cuda_built_probe_error"] == (
        "RuntimeError: CUDA build query failed"
    )
    assert summary["cuda_device_count"] == 0
    assert summary["cuda_device_probe_error"] is None
    assert summary["cuda_runtime_state"] == "build_probe_failed"
    assert summary["skips"] == [
        {
            "scope": "run",
            "status": "skip",
            "reason": "cuda_unavailable",
            "detail": "CUDA runtime state is build_probe_failed",
            "timing_scope": "none",
            "cuda_available": True,
            "cuda_runtime_state": "build_probe_failed",
            "cuda_built": False,
            "cuda_built_probe_error": "RuntimeError: CUDA build query failed",
            "cuda_device_count": 0,
            "cuda_device_probe_error": None,
            "cuda_available_probe_error": None,
        }
    ]
    assert "CUDA runtime state is build_probe_failed" in output
    assert "median ms" not in output


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
    assert summary["schema_version"] == 12
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


def test_no_cuda_decode_skip_stdout_matches_json_without_timings(tmp_path):
    """Decode skips keep the structured stdout handoff diagnostics-only."""
    summary_path = tmp_path / "decode-summary.json"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--mode",
            "decode",
            "--B",
            "3",
            "--H",
            "5",
            "--T",
            "17",
            "--N",
            "7",
            "--D",
            "11",
            "--dtype",
            "bfloat16",
            "--warmup",
            "4",
            "--iters",
            "9",
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
    assert summary["request"] == {
        "B": 3,
        "H": 5,
        "T": 17,
        "N": 7,
        "D": 11,
        "dtype": "bfloat16",
        "warmup": 4,
        "iters": 9,
    }
    assert summary["timing_scope"] == "none"
    assert "results" not in summary
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
    assert summary["cuda_runtime_state"] in {
        "not_built",
        "no_visible_device",
        "runtime_unavailable",
        "build_probe_failed",
    }
    assert summary["torch_version"]
    assert summary["cuda_version"] is None or isinstance(summary["cuda_version"], str)
    assert summary["backend_info"]
    assert "torch=" in result.stdout
    assert "cuda_available=false" in result.stdout
    assert f"cuda_built={summary['cuda_built']}" in result.stdout
    assert f"cuda_device_count={summary['cuda_device_count']}" in result.stdout
    assert "backend_info=" in result.stdout
    assert "median ms" not in result.stdout


def test_no_cuda_skip_records_requested_measurement_config(tmp_path):
    """Skip diagnostics retain the exact CPU-side request for later GPU handoff."""
    summary_path = tmp_path / "requested-summary.json"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--mode",
            "decode",
            "--B",
            "3",
            "--H",
            "5",
            "--T",
            "17",
            "--N",
            "7",
            "--D",
            "11",
            "--dtype",
            "bfloat16",
            "--warmup",
            "4",
            "--iters",
            "9",
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
    assert summary["skips"][0]["timing_scope"] == "none"
    assert summary["skips"][0]["cuda_available"] is False
    assert summary["skips"][0]["cuda_runtime_state"] == summary["cuda_runtime_state"]
    assert summary["request"] == {
        "B": 3,
        "H": 5,
        "T": 17,
        "N": 7,
        "D": 11,
        "dtype": "bfloat16",
        "warmup": 4,
        "iters": 9,
    }
    assert "median ms" not in result.stdout
