"""CPU-safe v15 contracts for null parity fields on GPU run-level skips."""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "benchmarks" / "bench_gpu_attn.py"

PARITY_NULL_FIELDS = (
    "bit_identical",
    "allclose_at_1e-4",
    "max_abs_delta",
    "max_rel_delta",
    "median_ms",
)


def test_run_level_cuda_skip_exposes_null_parity_delta_fields(
    monkeypatch, tmp_path, capsys
):
    """Run-level CUDA skips share backend-skip parity keys as null, never timing."""
    namespace: dict[str, object] = {
        "__name__": "bench_gpu_attn_v15_test",
        "__file__": str(SCRIPT),
    }
    exec(compile(SCRIPT.read_text(), str(SCRIPT), "exec"), namespace)
    torch = namespace["torch"]
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.backends.cuda, "is_built", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)
    summary_path = tmp_path / "parity-skip-summary.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
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
    )

    assert namespace["main"]() == 0
    summary = json.loads(summary_path.read_text())
    output = capsys.readouterr().out
    skip = summary["skips"][0]

    assert namespace["SUMMARY_SCHEMA_VERSION"] == 13
    assert summary["schema_version"] == 13
    assert summary["status"] == "skip"
    assert summary["reason"] == "cuda_unavailable"
    assert summary["mode"] == "decode"
    assert summary["timing_scope"] == "none"
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
    assert "results" not in summary
    assert skip["scope"] == "run"
    assert skip["timing_scope"] == "none"
    assert skip["cuda_runtime_state"] == "no_visible_device"
    for field in PARITY_NULL_FIELDS:
        assert field in skip
        assert skip[field] is None
    assert "median ms" not in output
    assert "× vs eager" not in output


def test_skip_summary_helper_emits_null_parity_fields():
    """_skip_summary keeps parity keys present-but-null without inventing timings."""
    namespace: dict[str, object] = {
        "__name__": "bench_gpu_attn_v15_helper_test",
        "__file__": str(SCRIPT),
    }
    exec(compile(SCRIPT.read_text(), str(SCRIPT), "exec"), namespace)
    summary = namespace["_skip_summary"](
        mode="cold",
        B=2,
        H=4,
        T=128,
        N=64,
        D=128,
        dtype="float32",
        warmup=10,
        iters=50,
        cuda_available=False,
        cuda_runtime={
            "cuda_built": False,
            "cuda_built_probe_error": None,
            "cuda_device_count": 0,
            "cuda_device_probe_error": None,
            "cuda_available_probe_error": None,
            "cuda_runtime_state": "not_built",
        },
    )
    skip = summary["skips"][0]
    assert summary["schema_version"] == 13
    assert summary["timing_scope"] == "none"
    for field in PARITY_NULL_FIELDS:
        assert skip[field] is None
