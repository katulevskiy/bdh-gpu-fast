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
import statistics
import sys
import time
from pathlib import Path
from typing import Callable

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kernels.attention import (  # noqa: E402
    blocked_decode_attn,
    blocked_tril_attn,
    eager_decode_attn,
    eager_tril_attn,
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
    "triton": triton_decode_attn,
    "cuda": tril_decode,
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


def _bit_identical_report(name: str, got: torch.Tensor, ref: torch.Tensor) -> None:
    """Print exact equality + max abs/rel diffs vs eager reference."""
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
    args = p.parse_args()

    cuda_ok = torch.cuda.is_available()
    if not cuda_ok and not args.force_cpu:
        print(
            "SKIP: CUDA not available — GPU attn microbench requires a CUDA device.\n"
            "  On A100/H100:\n"
            "    python benchmarks/bench_gpu_attn.py\n"
            "    python benchmarks/bench_gpu_attn.py --B 4 --H 8 --T 512 --N 64 --D 128\n"
            "    python benchmarks/bench_gpu_attn.py --mode decode --T 512\n"
            "  Optional native ext:\n"
            "    BDH_BUILD_EXT=1 BDH_BUILD_CUDA=1 pip install -e . --no-build-isolation\n"
            f"  torch={torch.__version__}  cuda={cuda_ok}  backend_info={backend_info()}"
        )
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

    with torch.no_grad():
        ref = ref_fn(Q, K, V)
        outs: dict[str, torch.Tensor] = {"eager": ref}
        for name, fn in backends.items():
            if name == "eager":
                continue
            outs[name] = fn(Q, K, V)
            _bit_identical_report(name, outs[name], ref)

    print("--- median wall time (ms) ---")
    t_eager = _bench(ref_fn, (Q, K, V), warmup=args.warmup, iters=args.iters)
    print(f"  eager   {t_eager:8.3f} ms  (baseline)")
    for name, fn in backends.items():
        if name == "eager":
            continue
        t = _bench(fn, (Q, K, V), warmup=args.warmup, iters=args.iters)
        speedup = (t_eager / t) if t > 0 else float("inf")
        print(f"  {name:7s} {t:8.3f} ms  ({speedup:.2f}× vs eager)")

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
