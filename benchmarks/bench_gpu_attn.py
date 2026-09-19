#!/usr/bin/env python3
"""GPU microbench: eager | blocked | online | triton | cuda strict-tril attn.

Compares cold-path ``tril(diagonal=-1)`` score×V **and** T=1 decode vs
packed past KR/V backends on CUDA.
Skips cleanly when CUDA is unavailable (this sandbox has no GPU).

Usage (A100 / H100)::

    python benchmarks/bench_gpu_attn.py
    python benchmarks/bench_gpu_attn.py --mode decode --T 512
    python benchmarks/bench_gpu_attn.py --B 4 --H 8 --T 512 --N 64 --D 128
    python benchmarks/bench_gpu_attn.py --warmup 10 --iters 50

Optional native CUDA ext (otherwise ``cuda`` = pure-PyTorch ref)::

    BDH_BUILD_EXT=1 BDH_BUILD_CUDA=1 pip install -e . --no-build-isolation

Private ``katulevskiy/bdh-gpu-opt`` only — do not target pathwaycom.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kernels.attention import (  # noqa: E402
    blocked_decode_attn,
    blocked_tril_attn,
    eager_decode_attn,
    eager_tril_attn,
    online_decode_attn,
    online_tril_attn,
    triton_decode_attn,
    triton_tril_attn,
)
from kernels.attention_dispatch import backend_info  # noqa: E402
from kernels.cuda_attn import tril_decode, tril_score_v  # noqa: E402


BACKENDS_COLD: dict[str, Callable[..., torch.Tensor]] = {
    "eager": eager_tril_attn,
    "blocked": blocked_tril_attn,
    "online": online_tril_attn,
    "triton": triton_tril_attn,
    "cuda": tril_score_v,
}

BACKENDS_DECODE: dict[str, Callable[..., torch.Tensor]] = {
    "eager": eager_decode_attn,
    "blocked": blocked_decode_attn,
    "online": online_decode_attn,
    "triton": triton_decode_attn,
    "cuda": tril_decode,
}


# Keep these commands in sync with the GPU microbench runbook in
# OPT_BACKLOG.md. They are printed on CPU so a skipped run is still an
# actionable handoff to a CUDA box; no CPU timings are substituted.
BACKLOG_COMMANDS: dict[str, tuple[str, ...]] = {
    "cold": (
        "python benchmarks/bench_gpu_attn.py",
        "python benchmarks/bench_gpu_attn.py --B 4 --H 8 --T 512 --N 64 --D 128 --warmup 20 --iters 100",
    ),
    "decode": ("python benchmarks/bench_gpu_attn.py --mode decode --T 512",),
    "dtype": (
        "python benchmarks/bench_gpu_attn.py --dtype bfloat16",
        "python benchmarks/bench_gpu_attn.py --dtype float16",
    ),
    "native_cuda_optional": (
        "BDH_BUILD_EXT=1 BDH_BUILD_CUDA=1 pip install -e . --no-build-isolation",
        "python benchmarks/bench_gpu_attn.py",
    ),
}


def _summary_path(path: str | None, summary: dict[str, Any]) -> None:
    if path is None:
        return
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


def _skip_summary() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "skip",
        "reason": "cuda_unavailable",
        "device": "cpu",
        "cuda_available": False,
        "gpu_name": None,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "backend_info": backend_info(),
        "commands": {key: list(value) for key, value in BACKLOG_COMMANDS.items()},
    }


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _bench(
    fn: Callable[..., torch.Tensor],
    args: tuple[torch.Tensor, ...],
    *,
    warmup: int,
    iters: int,
) -> float:
    for _ in range(warmup):
        fn(*args)
    _sync()
    times: list[float] = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn(*args)
        _sync()
        times.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(times)


def _bit_identical_report(
    name: str, got: torch.Tensor, ref: torch.Tensor
) -> dict[str, Any]:
    """Print and return exact equality + closeness vs eager reference."""
    identical = bool(torch.equal(got, ref))
    diff = (got - ref).abs()
    max_abs = float(diff.max().item()) if diff.numel() else 0.0
    ref_abs = ref.abs().clamp_min(1e-12)
    max_rel = float((diff / ref_abs).max().item()) if diff.numel() else 0.0
    close = bool(torch.allclose(got, ref, rtol=1e-4, atol=1e-4))
    print(
        f"  vs eager [{name:7s}] bit_identical={identical}  "
        f"allclose@1e-4={close}  max|Δ|={max_abs:.3e}  max_rel={max_rel:.3e}"
    )
    return {
        "backend": name,
        "bit_identical": identical,
        "allclose_at_1e-4": close,
        "max_abs_delta": max_abs,
        "max_rel_delta": max_rel,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--B", type=int, default=2)
    p.add_argument("--H", type=int, default=4)
    p.add_argument("--T", type=int, default=128, help="Cold seq len, or past length S in decode mode")
    p.add_argument("--N", type=int, default=64)
    p.add_argument("--D", type=int, default=128)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--dtype", choices=("float32", "bfloat16", "float16"), default="float32")
    p.add_argument(
        "--mode",
        choices=("cold", "decode"),
        default="cold",
        help="cold = full tril score×V; decode = Tq=1 vs packed past of length T",
    )
    p.add_argument(
        "--force-cpu",
        action="store_true",
        help="Run on CPU anyway (not the intended path; for harness smoke only).",
    )
    p.add_argument(
        "--json-out",
        type=str,
        default=None,
        metavar="PATH",
        help="Write the structured skip or CUDA comparison summary to PATH.",
    )
    args = p.parse_args()

    cuda_ok = torch.cuda.is_available()
    if not cuda_ok and not args.force_cpu:
        summary = _skip_summary()
        print(
            "GPU_ATTN_SKIP status=skip reason=cuda_unavailable device=cpu\n"
            "CUDA is required for Triton/CUDA timing; no CPU timings were substituted."
        )
        print("GPU_ATTN_COMMANDS (from OPT_BACKLOG.md):")
        for group, commands in summary["commands"].items():
            print(f"  {group}:")
            for command in commands:
                print(f"    {command}")
        print(
            f"  torch={summary['torch_version']}  cuda_available=false "
            f"cuda_version={summary['cuda_version']}  backend_info={summary['backend_info']}"
        )
        print("GPU_ATTN_SUMMARY " + json.dumps(summary, sort_keys=True))
        _summary_path(args.json_out, summary)
        return 0

    device = torch.device("cuda" if cuda_ok else "cpu")
    dtype = getattr(torch, args.dtype)
    B, H, T, N, D = args.B, args.H, args.T, args.N, args.D
    mode = args.mode

    torch.manual_seed(0)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(0)

    backends = BACKENDS_DECODE if mode == "decode" else BACKENDS_COLD
    if mode == "decode":
        # Tq=1 query vs packed past length T (=S)
        Q = torch.randn(B, H, 1, N, device=device, dtype=dtype)
        K = torch.randn(B, H, T, N, device=device, dtype=dtype)
        V = torch.randn(B, 1, T, D, device=device, dtype=dtype)
        shape_note = f"decode Tq=1 S={T} N={N} D={D}"
        ref_fn = eager_decode_attn
    else:
        Q = torch.randn(B, H, T, N, device=device, dtype=dtype)
        K = Q.clone()
        V = torch.randn(B, 1, T, D, device=device, dtype=dtype)
        shape_note = f"cold T={T} N={N} D={D}"
        ref_fn = eager_tril_attn

    info = backend_info()
    props = torch.cuda.get_device_properties(0) if device.type == "cuda" else None
    gpu_name = props.name if props else "cpu"
    print(f"device={device}  gpu={gpu_name!r}  dtype={dtype}  mode={mode}")
    print(f"shape B={B} H={H} {shape_note}")
    print(f"torch={torch.__version__}  cuda={cuda_ok}  backend_info={info}")
    print(f"warmup={args.warmup}  iters={args.iters}")
    print("--- bit-identical / closeness vs eager ---")

    comparisons: dict[str, dict[str, Any]] = {}
    with torch.no_grad():
        ref = ref_fn(Q, K, V)
        outs: dict[str, torch.Tensor] = {"eager": ref}
        comparisons["eager"] = {
            "backend": "eager",
            "bit_identical": True,
            "allclose_at_1e-4": True,
            "max_abs_delta": 0.0,
            "max_rel_delta": 0.0,
        }
        for name, fn in backends.items():
            if name == "eager":
                continue
            outs[name] = fn(Q, K, V)
            comparisons[name] = _bit_identical_report(name, outs[name], ref)

    print("--- median wall time (ms) ---")
    timings: dict[str, float] = {}
    t_eager = _bench(ref_fn, (Q, K, V), warmup=args.warmup, iters=args.iters)
    timings["eager"] = t_eager
    print(f"  eager   {t_eager:8.3f} ms  (baseline)")
    for name, fn in backends.items():
        if name == "eager":
            continue
        t = _bench(fn, (Q, K, V), warmup=args.warmup, iters=args.iters)
        timings[name] = t
        speedup = (t_eager / t) if t > 0 else float("inf")
        print(f"  {name:7s} {t:8.3f} ms  ({speedup:.2f}× vs eager)")

    summary = {
        "schema_version": 1,
        "status": "ok",
        "mode": mode,
        "device": "cuda",
        "gpu_name": gpu_name,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cuda_available": True,
        "backend_info": info,
        "shape": {"B": B, "H": H, "T": T, "N": N, "D": D},
        "dtype": args.dtype,
        "warmup": args.warmup,
        "iters": args.iters,
        "results": [
            {**comparisons[name], "median_ms": timings[name]}
            for name in backends
        ],
    }
    print("GPU_ATTN_SUMMARY " + json.dumps(summary, sort_keys=True))
    _summary_path(args.json_out, summary)

    if device.type != "cuda":
        print(
            "NOTE: --force-cpu used; Triton/CUDA kernels not executed; "
            "do not claim GPU wins from these medians."
        )
    else:
        print(
            "NOTE: record median ms + bit_identical flags; "
            "prefer bit_identical or allclose@1e-4 before claiming a win. "
            "Default BDH_ATTN_IMPL remains eager until GPU data lands."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
