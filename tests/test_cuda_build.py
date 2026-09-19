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


def test_cuda_flag_without_extension_opt_in_is_pure_python_noop():
    """BDH_BUILD_CUDA alone must not opt into native extension setup."""
    env = os.environ.copy()
    env.pop("BDH_BUILD_EXT", None)
    env.update({"BDH_BUILD_CUDA": "1"})
    env.pop("BDH_FORCE_CPU_EXT", None)

    output = _setup_name(env)

    assert "pure-Python install" in output
    assert "skipping CUDA extension build" not in output
    assert any(line.strip() == "bdh-gpu-opt" for line in output.splitlines())


def test_force_cpu_flag_without_extension_opt_in_is_pure_python_noop():
    """BDH_FORCE_CPU_EXT alone must not opt into native extension setup."""
    env = os.environ.copy()
    env.pop("BDH_BUILD_EXT", None)
    env.update({"BDH_FORCE_CPU_EXT": "1"})
    env.pop("BDH_BUILD_CUDA", None)

    output = _setup_name(env)

    assert "pure-Python install" in output
    assert "Building bdh_cuda_ext" not in output
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
    assert "nvcc not found in CUDA_HOME/CUDA_PATH or PATH" in output
    assert "CPU refs remain available" in output


def test_non_executable_nvcc_stub_is_clear_noop(tmp_path):
    """A stale non-executable nvcc path must not enter CUDAExtension setup."""
    nvcc = tmp_path / "cuda-home" / "bin" / "nvcc"
    nvcc.parent.mkdir(parents=True)
    nvcc.write_text("#!/bin/sh\n", encoding="utf-8")
    nvcc.chmod(0o644)
    empty_path = tmp_path / "empty-path"
    empty_path.mkdir()

    env = os.environ.copy()
    env.update(
        {
            "BDH_BUILD_EXT": "1",
            "BDH_BUILD_CUDA": "1",
            "CUDA_HOME": str(tmp_path / "cuda-home"),
            "CUDA_PATH": str(tmp_path / "missing-cuda-path"),
            "PATH": str(empty_path),
        }
    )

    output = _setup_name(env)

    assert "skipping CUDA extension build" in output
    assert "nvcc not found in CUDA_HOME/CUDA_PATH or PATH" in output
    assert "CPU refs remain available" in output


def test_cuda_path_non_executable_nvcc_stub_is_clear_noop(tmp_path):
    """A stale CUDA_PATH nvcc path must not enter CUDAExtension setup."""
    nvcc = tmp_path / "cuda-path" / "bin" / "nvcc"
    nvcc.parent.mkdir(parents=True)
    nvcc.write_text("#!/bin/sh\n", encoding="utf-8")
    nvcc.chmod(0o644)
    empty_path = tmp_path / "empty-path"
    empty_path.mkdir()

    env = os.environ.copy()
    env.update(
        {
            "BDH_BUILD_EXT": "1",
            "BDH_BUILD_CUDA": "1",
            "CUDA_HOME": str(tmp_path / "missing-cuda-home"),
            "CUDA_PATH": str(tmp_path / "cuda-path"),
            "PATH": str(empty_path),
        }
    )

    output = _setup_name(env)

    assert "skipping CUDA extension build" in output
    assert "nvcc not found in CUDA_HOME/CUDA_PATH or PATH" in output
    assert "CPU refs remain available" in output


def test_path_non_executable_nvcc_stub_is_clear_noop(tmp_path):
    """A stale PATH nvcc entry must not enter CUDAExtension setup."""
    nvcc = tmp_path / "bin" / "nvcc"
    nvcc.parent.mkdir()
    nvcc.write_text("#!/bin/sh\n", encoding="utf-8")
    nvcc.chmod(0o644)
    empty_cuda_home = tmp_path / "missing-cuda-home"
    empty_cuda_path = tmp_path / "missing-cuda-path"
    empty_cuda_home.mkdir()
    empty_cuda_path.mkdir()

    env = os.environ.copy()
    env.update(
        {
            "BDH_BUILD_EXT": "1",
            "BDH_BUILD_CUDA": "1",
            "CUDA_HOME": str(empty_cuda_home),
            "CUDA_PATH": str(empty_cuda_path),
            "PATH": str(nvcc.parent),
        }
    )

    output = _setup_name(env)

    assert "skipping CUDA extension build" in output
    assert "nvcc not found in CUDA_HOME/CUDA_PATH or PATH" in output
    assert "CPU refs remain available" in output


def test_nvcc_directory_is_clear_noop(tmp_path):
    """A directory named nvcc must not enter CUDAExtension setup."""
    nvcc = tmp_path / "cuda-home" / "bin" / "nvcc"
    nvcc.mkdir(parents=True)
    empty_path = tmp_path / "empty-path"
    empty_path.mkdir()

    env = os.environ.copy()
    env.update(
        {
            "BDH_BUILD_EXT": "1",
            "BDH_BUILD_CUDA": "1",
            "CUDA_HOME": str(tmp_path / "cuda-home"),
            "CUDA_PATH": str(tmp_path / "missing-cuda-path"),
            "PATH": str(empty_path),
        }
    )

    output = _setup_name(env)

    assert "skipping CUDA extension build" in output
    assert "nvcc not found in CUDA_HOME/CUDA_PATH or PATH" in output
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


def test_forced_cpu_ext_takes_precedence_over_cuda_flag():
    """The explicit CPU-only request wins when both native flags are set."""
    env = os.environ.copy()
    env.update(
        {
            "BDH_BUILD_EXT": "1",
            "BDH_BUILD_CUDA": "1",
            "BDH_FORCE_CPU_EXT": "1",
            "CUDA_HOME": str(ROOT / ".missing-cuda-home"),
            "CUDA_PATH": str(ROOT / ".missing-cuda-path"),
        }
    )

    output = _setup_name(env)

    assert "Building bdh_cuda_ext CPU-only" in output
    assert "skipping CUDA extension build" not in output
    assert "pure-Python install" not in output


@pytest.mark.parametrize("value", ["0", "true", "yes"])
def test_non_one_extension_flag_remains_pure_python_noop(value):
    """Only the exact opt-in value may enter native extension setup."""
    env = os.environ.copy()
    env.update(
        {
            "BDH_BUILD_EXT": value,
            "BDH_BUILD_CUDA": "1",
            "BDH_FORCE_CPU_EXT": "1",
        }
    )

    output = _setup_name(env)

    assert "pure-Python install" in output
    assert "Building bdh_cuda_ext" not in output
    assert any(line.strip() == "bdh-gpu-opt" for line in output.splitlines())
