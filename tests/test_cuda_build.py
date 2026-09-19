"""CPU-safe configuration smoke tests for the optional native extension."""

from __future__ import annotations

import ast
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


def test_missing_torch_keeps_native_opt_in_cpu_safe(tmp_path):
    """A missing torch import must skip native setup without breaking metadata."""
    (tmp_path / "torch.py").write_text(
        "raise ImportError('torch unavailable for setup smoke test')\n",
        encoding="utf-8",
    )

    env = os.environ.copy()
    env.update({"BDH_BUILD_EXT": "1"})
    pythonpath = os.pathsep.join(filter(None, [str(tmp_path), env.get("PYTHONPATH")]))
    env["PYTHONPATH"] = pythonpath

    output = _setup_name(env)

    assert "torch not importable — skipping native extension build" in output
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


def test_cuda_path_nvcc_directory_is_clear_noop(tmp_path):
    """A directory named nvcc in CUDA_PATH must not enter CUDAExtension setup."""
    nvcc = tmp_path / "cuda-path" / "bin" / "nvcc"
    nvcc.mkdir(parents=True)
    empty_path = tmp_path / "empty-path"
    empty_path.mkdir()
    empty_cuda_home = tmp_path / "missing-cuda-home"
    empty_cuda_home.mkdir()

    env = os.environ.copy()
    env.update(
        {
            "BDH_BUILD_EXT": "1",
            "BDH_BUILD_CUDA": "1",
            "CUDA_HOME": str(empty_cuda_home),
            "CUDA_PATH": str(tmp_path / "cuda-path"),
            "PATH": str(empty_path),
        }
    )

    output = _setup_name(env)

    assert "skipping CUDA extension build" in output
    assert "nvcc not found in CUDA_HOME/CUDA_PATH or PATH" in output
    assert "CPU refs remain available" in output


def test_cuda_home_executable_nvcc_selects_cuda_setup(tmp_path):
    """An executable nvcc under CUDA_HOME must select CUDA extension setup."""
    nvcc = tmp_path / "cuda-home" / "bin" / "nvcc"
    nvcc.parent.mkdir(parents=True)
    nvcc.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    nvcc.chmod(0o755)
    empty_cuda_path = tmp_path / "missing-cuda-path"
    empty_cuda_path.mkdir()
    empty_path = tmp_path / "empty-path"
    empty_path.mkdir()

    env = os.environ.copy()
    env.update(
        {
            "BDH_BUILD_EXT": "1",
            "BDH_BUILD_CUDA": "1",
            "CUDA_HOME": str(tmp_path / "cuda-home"),
            "CUDA_PATH": str(empty_cuda_path),
            "PATH": str(empty_path),
        }
    )

    output = _setup_name(env)

    assert "Building bdh_cuda_ext WITH CUDA" in output
    assert "skipping CUDA extension build" not in output
    assert "pure-Python install" not in output


def test_cuda_path_executable_nvcc_falls_back_from_stale_cuda_home(tmp_path):
    """A stale CUDA_HOME entry must not mask a usable CUDA_PATH nvcc."""
    stale_nvcc = tmp_path / "cuda-home" / "bin" / "nvcc"
    stale_nvcc.parent.mkdir(parents=True)
    stale_nvcc.write_text("#!/bin/sh\n", encoding="utf-8")
    stale_nvcc.chmod(0o644)

    nvcc = tmp_path / "cuda-path" / "bin" / "nvcc"
    nvcc.parent.mkdir(parents=True)
    nvcc.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    nvcc.chmod(0o755)
    empty_path = tmp_path / "empty-path"
    empty_path.mkdir()

    env = os.environ.copy()
    env.update(
        {
            "BDH_BUILD_EXT": "1",
            "BDH_BUILD_CUDA": "1",
            "CUDA_HOME": str(tmp_path / "cuda-home"),
            "CUDA_PATH": str(tmp_path / "cuda-path"),
            "PATH": str(empty_path),
        }
    )

    output = _setup_name(env)

    assert "Building bdh_cuda_ext WITH CUDA" in output
    assert "skipping CUDA extension build" not in output
    assert "pure-Python install" not in output


def test_cuda_path_executable_nvcc_selects_cuda_setup(tmp_path):
    """An executable nvcc under CUDA_PATH must select CUDA extension setup."""
    nvcc = tmp_path / "cuda-path" / "bin" / "nvcc"
    nvcc.parent.mkdir(parents=True)
    nvcc.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    nvcc.chmod(0o755)
    empty_path = tmp_path / "empty-path"
    empty_path.mkdir()
    empty_cuda_home = tmp_path / "cuda-home"
    empty_cuda_home.mkdir()

    env = os.environ.copy()
    env.update(
        {
            "BDH_BUILD_EXT": "1",
            "BDH_BUILD_CUDA": "1",
            "CUDA_HOME": str(empty_cuda_home),
            "CUDA_PATH": str(tmp_path / "cuda-path"),
            "PATH": str(empty_path),
        }
    )

    output = _setup_name(env)

    assert "Building bdh_cuda_ext WITH CUDA" in output
    assert "skipping CUDA extension build" not in output
    assert "pure-Python install" not in output


def test_path_nvcc_falls_back_from_stale_cuda_home(tmp_path):
    """A stale CUDA_HOME entry must not mask a usable PATH nvcc."""
    stale_nvcc = tmp_path / "cuda-home" / "bin" / "nvcc"
    stale_nvcc.parent.mkdir(parents=True)
    stale_nvcc.write_text("#!/bin/sh\n", encoding="utf-8")
    stale_nvcc.chmod(0o644)

    nvcc = tmp_path / "bin" / "nvcc"
    nvcc.parent.mkdir(parents=True)
    nvcc.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    nvcc.chmod(0o755)
    empty_cuda_path = tmp_path / "missing-cuda-path"
    empty_cuda_path.mkdir()

    env = os.environ.copy()
    env.update(
        {
            "BDH_BUILD_EXT": "1",
            "BDH_BUILD_CUDA": "1",
            "CUDA_HOME": str(tmp_path / "cuda-home"),
            "CUDA_PATH": str(empty_cuda_path),
            "PATH": str(nvcc.parent),
        }
    )

    output = _setup_name(env)

    assert "Building bdh_cuda_ext WITH CUDA" in output
    assert "skipping CUDA extension build" not in output
    assert "pure-Python install" not in output


def test_path_executable_nvcc_selects_cuda_setup(tmp_path):
    """An executable nvcc on PATH must select CUDA extension setup."""
    nvcc = tmp_path / "bin" / "nvcc"
    nvcc.parent.mkdir(parents=True)
    nvcc.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    nvcc.chmod(0o755)
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

    assert "Building bdh_cuda_ext WITH CUDA" in output
    assert "skipping CUDA extension build" not in output
    assert "pure-Python install" not in output


def test_path_non_executable_nvcc_stub_is_clear_noop(tmp_path):
    """A stale PATH nvcc entry must not enter CUDAExtension setup."""
    nvcc = tmp_path / "bin" / "nvcc"
    nvcc.parent.mkdir(parents=True)
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


def test_path_nvcc_directory_is_clear_noop(tmp_path):
    """A directory named nvcc on PATH must not enter CUDAExtension setup."""
    nvcc = tmp_path / "bin" / "nvcc"
    nvcc.mkdir(parents=True)
    nvcc.chmod(0o755)
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


def test_native_opt_in_stays_cpu_only_without_cuda(tmp_path):
    """The default native opt-in must select C++ when torch has no CUDA."""
    torch = tmp_path / "torch"
    cpp_extension = torch / "utils" / "cpp_extension.py"
    cpp_extension.parent.mkdir(parents=True)
    (torch / "__init__.py").write_text(
        "class _Cuda:\n"
        "    @staticmethod\n"
        "    def is_available():\n"
        "        return False\n"
        "cuda = _Cuda()\n",
        encoding="utf-8",
    )
    (torch / "utils" / "__init__.py").write_text("", encoding="utf-8")
    cpp_extension.write_text(
        "import os\n"
        "from pathlib import Path\n"
        "from setuptools import Extension\n"
        "class BuildExtension: ...\n"
        "def CppExtension(*args, **kwargs):\n"
        "    Path(os.environ['BDH_EXT_KWARGS']).write_text(\n"
        "        repr(kwargs), encoding='utf-8'\n"
        "    )\n"
        "    return Extension(*args, **kwargs)\n"
        "def CUDAExtension(*args, **kwargs):\n"
        "    raise AssertionError('CUDAExtension should not be selected')\n",
        encoding="utf-8",
    )

    nvcc = tmp_path / "cuda-home" / "bin" / "nvcc"
    nvcc.parent.mkdir(parents=True)
    nvcc.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    nvcc.chmod(0o755)
    kwargs_trace = tmp_path / "kwargs.txt"
    env = os.environ.copy()
    pythonpath = os.pathsep.join(filter(None, [str(tmp_path), env.get("PYTHONPATH")]))
    env.update(
        {
            "BDH_BUILD_EXT": "1",
            "BDH_EXT_KWARGS": str(kwargs_trace),
            "CUDA_HOME": str(nvcc.parents[1]),
            "CUDA_PATH": str(tmp_path / "missing-cuda-path"),
            "PYTHONPATH": pythonpath,
        }
    )
    env.pop("BDH_BUILD_CUDA", None)
    env.pop("BDH_FORCE_CPU_EXT", None)

    output = _setup_name(env)
    kwargs = ast.literal_eval(kwargs_trace.read_text(encoding="utf-8"))

    assert set(kwargs) == {"name", "sources", "include_dirs", "extra_compile_args"}
    assert kwargs["name"] == "bdh_cuda_ext"
    assert kwargs["sources"] == [
        str(ROOT / "csrc" / "tril_attn_cpu.cpp"),
        str(ROOT / "csrc" / "tril_attn_bind.cpp"),
    ]
    assert kwargs["include_dirs"] == [str(ROOT / "csrc")]
    assert kwargs["extra_compile_args"] == {"cxx": ["-O3", "-std=c++20"]}
    assert "Building bdh_cuda_ext CPU-only" in output
    assert "Building bdh_cuda_ext WITH CUDA" not in output
    assert "skipping CUDA extension build" not in output


def test_forced_cpu_ext_overrides_detected_cuda(tmp_path):
    """CPU-only mode must win even when torch reports CUDA availability."""
    torch = tmp_path / "torch"
    cpp_extension = torch / "utils" / "cpp_extension.py"
    cpp_extension.parent.mkdir(parents=True)
    (torch / "__init__.py").write_text(
        "class _Cuda:\n"
        "    @staticmethod\n"
        "    def is_available():\n"
        "        return True\n"
        "cuda = _Cuda()\n",
        encoding="utf-8",
    )
    (torch / "utils" / "__init__.py").write_text("", encoding="utf-8")
    cpp_extension.write_text(
        "from setuptools import Extension\n"
        "class BuildExtension: ...\n"
        "def CppExtension(*args, **kwargs):\n"
        "    return Extension(*args, **kwargs)\n"
        "def CUDAExtension(*args, **kwargs):\n"
        "    raise AssertionError('CUDAExtension should not be selected')\n",
        encoding="utf-8",
    )

    env = os.environ.copy()
    pythonpath = os.pathsep.join(filter(None, [str(tmp_path), env.get("PYTHONPATH")]))
    env.update(
        {
            "BDH_BUILD_EXT": "1",
            "BDH_FORCE_CPU_EXT": "1",
            "PYTHONPATH": pythonpath,
        }
    )
    env.pop("BDH_BUILD_CUDA", None)

    output = _setup_name(env)

    assert "Building bdh_cuda_ext CPU-only" in output
    assert "Building bdh_cuda_ext WITH CUDA" not in output
    assert "pure-Python install" not in output


def test_forced_cpu_ext_precedes_detected_cuda_and_valid_nvcc(tmp_path):
    """CPU-only mode must win over both CUDA detection and a valid nvcc."""
    torch = tmp_path / "torch"
    cpp_extension = torch / "utils" / "cpp_extension.py"
    cpp_extension.parent.mkdir(parents=True)
    (torch / "__init__.py").write_text(
        "class _Cuda:\n"
        "    @staticmethod\n"
        "    def is_available():\n"
        "        return True\n"
        "cuda = _Cuda()\n",
        encoding="utf-8",
    )
    (torch / "utils" / "__init__.py").write_text("", encoding="utf-8")
    cpp_extension.write_text(
        "from setuptools import Extension\n"
        "class BuildExtension: ...\n"
        "def CppExtension(*args, **kwargs):\n"
        "    return Extension(*args, **kwargs)\n"
        "def CUDAExtension(*args, **kwargs):\n"
        "    raise AssertionError('CUDAExtension should not be selected')\n",
        encoding="utf-8",
    )

    nvcc = tmp_path / "cuda-home" / "bin" / "nvcc"
    nvcc.parent.mkdir(parents=True)
    nvcc.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    nvcc.chmod(0o755)
    empty_cuda_path = tmp_path / "missing-cuda-path"
    empty_cuda_path.mkdir()

    env = os.environ.copy()
    pythonpath = os.pathsep.join(filter(None, [str(tmp_path), env.get("PYTHONPATH")]))
    env.update(
        {
            "BDH_BUILD_EXT": "1",
            "BDH_BUILD_CUDA": "1",
            "BDH_FORCE_CPU_EXT": "1",
            "CUDA_HOME": str(nvcc.parents[1]),
            "CUDA_PATH": str(empty_cuda_path),
            "PATH": str(nvcc.parent),
            "PYTHONPATH": pythonpath,
        }
    )

    output = _setup_name(env)

    assert "Building bdh_cuda_ext CPU-only" in output
    assert "Building bdh_cuda_ext WITH CUDA" not in output
    assert "skipping CUDA extension build" not in output


def test_forced_cpu_ext_does_not_require_nvcc_when_cuda_is_detected(tmp_path):
    """CPU-only mode must not require a toolkit on a CUDA-enabled torch wheel."""
    torch = tmp_path / "torch"
    cpp_extension = torch / "utils" / "cpp_extension.py"
    cpp_extension.parent.mkdir(parents=True)
    (torch / "__init__.py").write_text(
        "class _Cuda:\n"
        "    @staticmethod\n"
        "    def is_available():\n"
        "        return True\n"
        "cuda = _Cuda()\n",
        encoding="utf-8",
    )
    (torch / "utils" / "__init__.py").write_text("", encoding="utf-8")
    cpp_extension.write_text(
        "from setuptools import Extension\n"
        "class BuildExtension: ...\n"
        "def CppExtension(*args, **kwargs):\n"
        "    return Extension(*args, **kwargs)\n"
        "def CUDAExtension(*args, **kwargs):\n"
        "    raise AssertionError('CUDAExtension should not be selected')\n",
        encoding="utf-8",
    )

    empty_path = tmp_path / "empty-path"
    empty_path.mkdir()
    env = os.environ.copy()
    pythonpath = os.pathsep.join(filter(None, [str(tmp_path), env.get("PYTHONPATH")]))
    env.update(
        {
            "BDH_BUILD_EXT": "1",
            "BDH_BUILD_CUDA": "1",
            "BDH_FORCE_CPU_EXT": "1",
            "CUDA_HOME": str(tmp_path / "missing-cuda-home"),
            "CUDA_PATH": str(tmp_path / "missing-cuda-path"),
            "PATH": str(empty_path),
            "PYTHONPATH": pythonpath,
        }
    )

    output = _setup_name(env)

    assert "Building bdh_cuda_ext CPU-only" in output
    assert "Building bdh_cuda_ext WITH CUDA" not in output
    assert "skipping CUDA extension build" not in output


def test_forced_cpu_ext_does_not_probe_cuda_availability(tmp_path):
    """CPU-only mode must not query CUDA availability on a CPU-safe path."""
    torch = tmp_path / "torch"
    cpp_extension = torch / "utils" / "cpp_extension.py"
    cpp_extension.parent.mkdir(parents=True)
    (torch / "__init__.py").write_text(
        "class _Cuda:\n"
        "    @staticmethod\n"
        "    def is_available():\n"
        "        raise AssertionError('CPU-only mode must not probe CUDA')\n"
        "cuda = _Cuda()\n",
        encoding="utf-8",
    )
    (torch / "utils" / "__init__.py").write_text("", encoding="utf-8")
    cpp_extension.write_text(
        "from setuptools import Extension\n"
        "class BuildExtension: ...\n"
        "def CppExtension(*args, **kwargs):\n"
        "    return Extension(*args, **kwargs)\n"
        "def CUDAExtension(*args, **kwargs):\n"
        "    raise AssertionError('CUDAExtension should not be selected')\n",
        encoding="utf-8",
    )

    env = os.environ.copy()
    pythonpath = os.pathsep.join(filter(None, [str(tmp_path), env.get("PYTHONPATH")]))
    env.update(
        {
            "BDH_BUILD_EXT": "1",
            "BDH_BUILD_CUDA": "1",
            "BDH_FORCE_CPU_EXT": "1",
            "PYTHONPATH": pythonpath,
        }
    )

    output = _setup_name(env)

    assert "Building bdh_cuda_ext CPU-only" in output
    assert "Building bdh_cuda_ext WITH CUDA" not in output
    assert "skipping CUDA extension build" not in output


def test_forced_cpu_ext_registers_only_cpp_sources(tmp_path):
    """CPU-only mode must not register the CUDA source even with both flags."""
    torch = tmp_path / "torch"
    cpp_extension = torch / "utils" / "cpp_extension.py"
    cpp_extension.parent.mkdir(parents=True)
    (torch / "__init__.py").write_text(
        "class _Cuda:\n"
        "    @staticmethod\n"
        "    def is_available():\n"
        "        raise AssertionError('CPU-only mode must not probe CUDA')\n"
        "cuda = _Cuda()\n",
        encoding="utf-8",
    )
    (torch / "utils" / "__init__.py").write_text("", encoding="utf-8")
    cpp_extension.write_text(
        "import os\n"
        "from pathlib import Path\n"
        "from setuptools import Extension\n"
        "class BuildExtension: ...\n"
        "def CppExtension(*args, **kwargs):\n"
        "    Path(os.environ['BDH_EXT_SOURCES']).write_text(\n"
        "        repr(kwargs['sources']), encoding='utf-8'\n"
        "    )\n"
        "    return Extension(*args, **kwargs)\n"
        "def CUDAExtension(*args, **kwargs):\n"
        "    raise AssertionError('CUDAExtension should not be selected')\n",
        encoding="utf-8",
    )

    trace = tmp_path / "sources.txt"
    env = os.environ.copy()
    pythonpath = os.pathsep.join(filter(None, [str(tmp_path), env.get("PYTHONPATH")]))
    env.update(
        {
            "BDH_BUILD_EXT": "1",
            "BDH_BUILD_CUDA": "1",
            "BDH_FORCE_CPU_EXT": "1",
            "BDH_EXT_SOURCES": str(trace),
            "PYTHONPATH": pythonpath,
        }
    )

    output = _setup_name(env)
    sources = trace.read_text(encoding="utf-8")

    assert sources.count(".cpp") == 2
    assert "tril_attn_cpu.cpp" in sources
    assert "tril_attn_bind.cpp" in sources
    assert ".cu" not in sources
    assert "Building bdh_cuda_ext CPU-only" in output
    assert "Building bdh_cuda_ext WITH CUDA" not in output
    assert "skipping CUDA extension build" not in output


def test_forced_cpu_ext_has_no_cuda_compile_metadata(tmp_path):
    """CPU-only mode must not pass CUDA metadata to the C++ extension."""
    torch = tmp_path / "torch"
    cpp_extension = torch / "utils" / "cpp_extension.py"
    cpp_extension.parent.mkdir(parents=True)
    (torch / "__init__.py").write_text(
        "class _Cuda:\n"
        "    @staticmethod\n"
        "    def is_available():\n"
        "        return True\n"
        "cuda = _Cuda()\n",
        encoding="utf-8",
    )
    (torch / "utils" / "__init__.py").write_text("", encoding="utf-8")
    cpp_extension.write_text(
        "import os\n"
        "from pathlib import Path\n"
        "from setuptools import Extension\n"
        "class BuildExtension: ...\n"
        "def CppExtension(*args, **kwargs):\n"
        "    Path(os.environ['BDH_EXT_KWARGS']).write_text(\n"
        "        repr(kwargs), encoding='utf-8'\n"
        "    )\n"
        "    return Extension(*args, **kwargs)\n"
        "def CUDAExtension(*args, **kwargs):\n"
        "    raise AssertionError('CUDAExtension should not be selected')\n",
        encoding="utf-8",
    )

    nvcc = tmp_path / "cuda-home" / "bin" / "nvcc"
    nvcc.parent.mkdir(parents=True)
    nvcc.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    nvcc.chmod(0o755)
    kwargs_trace = tmp_path / "kwargs.txt"
    env = os.environ.copy()
    pythonpath = os.pathsep.join(filter(None, [str(tmp_path), env.get("PYTHONPATH")]))
    env.update(
        {
            "BDH_BUILD_EXT": "1",
            "BDH_BUILD_CUDA": "1",
            "BDH_FORCE_CPU_EXT": "1",
            "BDH_EXT_KWARGS": str(kwargs_trace),
            "CUDA_HOME": str(nvcc.parents[1]),
            "CUDA_PATH": str(tmp_path / "missing-cuda-path"),
            "PATH": str(nvcc.parent),
            "PYTHONPATH": pythonpath,
        }
    )

    output = _setup_name(env)
    kwargs = ast.literal_eval(kwargs_trace.read_text(encoding="utf-8"))

    assert set(kwargs) == {"name", "sources", "include_dirs", "extra_compile_args"}
    assert kwargs["name"] == "bdh_cuda_ext"
    assert kwargs["sources"] == [
        str(ROOT / "csrc" / "tril_attn_cpu.cpp"),
        str(ROOT / "csrc" / "tril_attn_bind.cpp"),
    ]
    assert kwargs["include_dirs"] == [str(ROOT / "csrc")]
    assert kwargs["extra_compile_args"] == {"cxx": ["-O3", "-std=c++20"]}
    assert "Building bdh_cuda_ext CPU-only" in output
    assert "Building bdh_cuda_ext WITH CUDA" not in output
    assert "skipping CUDA extension build" not in output



def test_forced_cuda_ext_registers_cuda_sources_and_metadata(tmp_path):
    """CUDA mode must register the CUDA source and only its CUDA metadata."""
    torch = tmp_path / "torch"
    cpp_extension = torch / "utils" / "cpp_extension.py"
    cpp_extension.parent.mkdir(parents=True)
    (torch / "__init__.py").write_text(
        "class _Cuda:\n"
        "    @staticmethod\n"
        "    def is_available():\n"
        "        return False\n"
        "cuda = _Cuda()\n",
        encoding="utf-8",
    )
    (torch / "utils" / "__init__.py").write_text("", encoding="utf-8")
    cpp_extension.write_text(
        "import os\n"
        "from pathlib import Path\n"
        "from setuptools import Extension\n"
        "class BuildExtension: ...\n"
        "def CppExtension(*args, **kwargs):\n"
        "    raise AssertionError('CppExtension should not be selected')\n"
        "def CUDAExtension(*args, **kwargs):\n"
        "    Path(os.environ['BDH_EXT_KWARGS']).write_text(\n"
        "        repr(kwargs), encoding='utf-8'\n"
        "    )\n"
        "    return Extension(*args, **kwargs)\n",
        encoding="utf-8",
    )

    nvcc = tmp_path / "cuda-home" / "bin" / "nvcc"
    nvcc.parent.mkdir(parents=True)
    nvcc.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    nvcc.chmod(0o755)
    kwargs_trace = tmp_path / "kwargs.txt"
    env = os.environ.copy()
    pythonpath = os.pathsep.join(filter(None, [str(tmp_path), env.get("PYTHONPATH")]))
    env.update(
        {
            "BDH_BUILD_EXT": "1",
            "BDH_BUILD_CUDA": "1",
            "BDH_EXT_KWARGS": str(kwargs_trace),
            "CUDA_HOME": str(nvcc.parents[1]),
            "CUDA_PATH": str(tmp_path / "missing-cuda-path"),
            "PATH": str(nvcc.parent),
            "PYTHONPATH": pythonpath,
        }
    )

    output = _setup_name(env)
    kwargs = ast.literal_eval(kwargs_trace.read_text(encoding="utf-8"))

    assert set(kwargs) == {
        "name",
        "sources",
        "include_dirs",
        "extra_compile_args",
        "define_macros",
    }
    assert kwargs["name"] == "bdh_cuda_ext"
    assert kwargs["sources"] == [
        str(ROOT / "csrc" / "tril_attn_cpu.cpp"),
        str(ROOT / "csrc" / "tril_attn_bind.cpp"),
        str(ROOT / "csrc" / "tril_attn_cuda.cu"),
    ]
    assert kwargs["include_dirs"] == [str(ROOT / "csrc")]
    assert kwargs["extra_compile_args"] == {
        "cxx": ["-O3", "-std=c++20"],
        "nvcc": ["-O3", "--use_fast_math", "-DWITH_CUDA"],
    }
    assert kwargs["define_macros"] == [("WITH_CUDA", None)]
    assert "Building bdh_cuda_ext WITH CUDA" in output
    assert "Building bdh_cuda_ext CPU-only" not in output
    assert "skipping CUDA extension build" not in output


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


def test_forced_cpu_ext_precedes_valid_nvcc(tmp_path):
    """CPU-only mode must win even when an executable nvcc is discoverable."""
    cuda_home = tmp_path / "cuda-home"
    nvcc = cuda_home / "bin" / "nvcc"
    nvcc.parent.mkdir(parents=True)
    nvcc.write_text("#!/bin/sh\n", encoding="utf-8")
    nvcc.chmod(0o755)

    env = os.environ.copy()
    env.update(
        {
            "BDH_BUILD_EXT": "1",
            "BDH_BUILD_CUDA": "1",
            "BDH_FORCE_CPU_EXT": "1",
            "CUDA_HOME": str(cuda_home),
            "CUDA_PATH": str(tmp_path / "missing-cuda-path"),
            "PATH": str(nvcc.parent),
        }
    )

    output = _setup_name(env)

    assert "Building bdh_cuda_ext CPU-only" in output
    assert "Building bdh_cuda_ext WITH CUDA" not in output
    assert "skipping CUDA extension build" not in output


@pytest.mark.parametrize("value", ["0", "true", "yes", "01", " 1"])
def test_non_exact_force_cpu_flag_does_not_override_cuda_opt_in(tmp_path, value):
    """Only the exact BDH_FORCE_CPU_EXT=1 value may override CUDA setup."""
    nvcc = tmp_path / "bin" / "nvcc"
    nvcc.parent.mkdir(parents=True)
    nvcc.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    nvcc.chmod(0o755)

    env = os.environ.copy()
    env.update(
        {
            "BDH_BUILD_EXT": "1",
            "BDH_BUILD_CUDA": "1",
            "BDH_FORCE_CPU_EXT": value,
            "CUDA_HOME": str(tmp_path / "missing-cuda-home"),
            "CUDA_PATH": str(tmp_path / "missing-cuda-path"),
            "PATH": str(nvcc.parent),
        }
    )

    output = _setup_name(env)

    assert "Building bdh_cuda_ext WITH CUDA" in output
    assert "Building bdh_cuda_ext CPU-only" not in output
    assert "skipping CUDA extension build" not in output


@pytest.mark.parametrize("value", ["0", "true", "yes"])
def test_non_one_cuda_flag_does_not_select_cuda_setup(value):
    """Only the exact BDH_BUILD_CUDA=1 value may select CUDA setup."""
    env = os.environ.copy()
    env.update(
        {
            "BDH_BUILD_EXT": "1",
            "BDH_BUILD_CUDA": value,
            "CUDA_HOME": str(ROOT / ".missing-cuda-home"),
            "CUDA_PATH": str(ROOT / ".missing-cuda-path"),
            # Keep this contract deterministic on hosts with a visible GPU.
            "CUDA_VISIBLE_DEVICES": "",
        }
    )
    env.pop("BDH_FORCE_CPU_EXT", None)

    output = _setup_name(env)

    assert "Building bdh_cuda_ext CPU-only" in output
    assert "Building bdh_cuda_ext WITH CUDA" not in output
    assert "skipping CUDA extension build" not in output


@pytest.mark.parametrize("value", ["", "01", " 1"])
def test_non_exact_cuda_flag_stays_cpu_only_with_nvcc(tmp_path, value):
    """CUDA toolkit presence must not widen the exact CUDA opt-in."""
    nvcc = tmp_path / "bin" / "nvcc"
    nvcc.parent.mkdir(parents=True)
    nvcc.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    nvcc.chmod(0o755)

    env = os.environ.copy()
    env.update(
        {
            "BDH_BUILD_EXT": "1",
            "BDH_BUILD_CUDA": value,
            "CUDA_HOME": str(tmp_path / "missing-cuda-home"),
            "CUDA_PATH": str(tmp_path / "missing-cuda-path"),
            "CUDA_VISIBLE_DEVICES": "",
            "PATH": str(nvcc.parent),
        }
    )
    env.pop("BDH_FORCE_CPU_EXT", None)

    output = _setup_name(env)

    assert "Building bdh_cuda_ext CPU-only" in output
    assert "Building bdh_cuda_ext WITH CUDA" not in output
    assert "skipping CUDA extension build" not in output


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
