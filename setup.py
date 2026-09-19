"""Optional build for bdh_cuda_ext (strict tril score×V).

Pure-Python path (always)::

    # no install required — kernels/cuda_attn.py CPU ref works as-is
    python -m pytest tests/test_cuda_attn.py -v

Optional native scaffold (needs working torch C++ ABI + compiler; CUDA for .cu)::

    BDH_BUILD_EXT=1 pip install -e . --no-build-isolation
    # or with CUDA objects even without a device:
    BDH_BUILD_EXT=1 BDH_BUILD_CUDA=1 pip install -e . --no-build-isolation

If the native build fails (common on mismatched g++/torch), training and tests
still use kernels.cuda_attn.tril_score_v_ref — the CUDA .cu scaffold remains
ready for a GPU box with a matching toolchain.
"""

from __future__ import annotations

import os
from pathlib import Path

from setuptools import setup

ROOT = Path(__file__).parent


def _extensions():
    """Build C++/CUDA only when explicitly requested."""
    if os.environ.get("BDH_BUILD_EXT", "") != "1":
        print(
            "bdh-gpu-opt: pure-Python install "
            "(set BDH_BUILD_EXT=1 to compile csrc/ native extension)"
        )
        return [], {}

    try:
        import torch
        from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension
    except ImportError:
        print("torch not importable — skipping native extension build")
        return [], {}

    csrc = ROOT / "csrc"
    sources = [
        str(csrc / "tril_attn_cpu.cpp"),
        str(csrc / "tril_attn_bind.cpp"),
    ]
    include_dirs = [str(csrc)]
    extra_compile_args = {"cxx": ["-O3", "-std=c++17"]}

    use_cuda = torch.cuda.is_available() and os.environ.get("BDH_FORCE_CPU_EXT", "") != "1"
    force_cuda = os.environ.get("BDH_BUILD_CUDA", "") == "1"

    if use_cuda or force_cuda:
        sources.append(str(csrc / "tril_attn_cuda.cu"))
        ext = CUDAExtension(
            name="bdh_cuda_ext",
            sources=sources,
            include_dirs=include_dirs,
            extra_compile_args={
                **extra_compile_args,
                "nvcc": ["-O3", "--use_fast_math", "-DWITH_CUDA"],
            },
            define_macros=[("WITH_CUDA", None)],
        )
        print("Building bdh_cuda_ext WITH CUDA")
    else:
        ext = CppExtension(
            name="bdh_cuda_ext",
            sources=sources,
            include_dirs=include_dirs,
            extra_compile_args=extra_compile_args,
        )
        print("Building bdh_cuda_ext CPU-only (no CUDA device / BDH_BUILD_CUDA unset)")

    return [ext], {"build_ext": BuildExtension}


ext_modules, cmdclass = _extensions()

setup(
    name="bdh-gpu-opt",
    version="0.1.0",
    description="Private BDH GPU optimization sandbox (optional CUDA tril attn)",
    packages=["kernels"],
    package_dir={"kernels": "kernels"},
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    python_requires=">=3.10",
    install_requires=["torch"],
    zip_safe=False,
)
