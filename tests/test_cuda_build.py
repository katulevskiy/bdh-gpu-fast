"""CPU-safe configuration smoke tests for the optional native extension."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _setup_name(env: dict[str, str]) -> str:
    result = subprocess.run(
        [sys.executable, "setup.py", "--name"],
        cwd=ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout + result.stderr


def test_default_setup_is_pure_python_noop():
    """The default install path must not require a compiler or CUDA toolkit."""
    env = os.environ.copy()
    for name in ("BDH_BUILD_EXT", "BDH_BUILD_CUDA", "BDH_FORCE_CPU_EXT"):
        env.pop(name, None)

    output = _setup_name(env)

    assert "pure-Python install" in output
    assert any(line.strip() == "bdh-gpu-opt" for line in output.splitlines())


def test_forced_cuda_without_nvcc_is_clear_noop():
    """A CUDA torch wheel without nvcc must skip before CUDAExtension setup."""
    if shutil.which("nvcc"):
        pytest.skip("nvcc is available; exercise the real CUDA build on that host")

    env = os.environ.copy()
    env.update(
        {
            "BDH_BUILD_EXT": "1",
            "BDH_BUILD_CUDA": "1",
            # Do not let a stale CUDA_HOME make this CPU-box test ambiguous.
            "CUDA_HOME": str(ROOT / ".missing-cuda-home"),
            "CUDA_PATH": str(ROOT / ".missing-cuda-path"),
        }
    )

    output = _setup_name(env)

    assert "skipping CUDA extension build" in output
    assert "nvcc not found" in output
    assert "CPU refs remain available" in output


def test_forced_cpu_ext_smoke_is_explicit():
    """BDH_FORCE_CPU_EXT=1 selects the CPU-only extension setup branch."""
    env = os.environ.copy()
    env.update({"BDH_BUILD_EXT": "1", "BDH_FORCE_CPU_EXT": "1"})
    env.pop("BDH_BUILD_CUDA", None)

    output = _setup_name(env)

    assert "Building bdh_cuda_ext CPU-only" in output
    assert "skipping CUDA extension build" not in output
    assert "pure-Python install" not in output
