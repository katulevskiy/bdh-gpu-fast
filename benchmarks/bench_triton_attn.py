#!/usr/bin/env python3
"""Microbench: eager vs blocked vs triton BDH strict-tril attention."""

from __future__ import annotations

import os
import statistics
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kernels.attention import (
    blocked_tril_attn,
    eager_tril_attn,
    max_score_tile_elems,
    online_tril_attn,
    triton_tril_attn,
)
from kernels.attention_dispatch import backend_info


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def bench(fn, args, warmup=5, iters=20, device=torch.device("cpu")):
    for _ in range(warmup):
        fn(*args)
    _sync(device)
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn(*args)
        _sync(device)
        times.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(times)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B, H, T, N, D = 2, 4, 128, 64, 128
    torch.manual_seed(0)
    Q = torch.randn(B, H, T, N, device=device)
    K = Q.clone()
    V = torch.randn(B, 1, T, D, device=device)

    info = backend_info()
    print(f"device={device}  B={B} H={H} T={T} N={N} D={D}")
    print(f"backend_info={info}")
    print(f"torch={torch.__version__} cuda={torch.cuda.is_available()}")

    # Correctness spot-check
    ref = eager_tril_attn(Q, K, V)
    blk = blocked_tril_attn(Q, K, V)
    onl = online_tril_attn(Q, K, V)
    tri = triton_tril_attn(Q, K, V)
    d_blk = (blk - ref).abs().max().item()
    d_onl = (onl - ref).abs().max().item()
    d_tri = (tri - ref).abs().max().item()
    print(
        f"correctness max|blocked-eager|={d_blk:.3e}  "
        f"max|online-eager|={d_onl:.3e}  "
        f"max|triton_path-eager|={d_tri:.3e}"
    )
    print(
        f"score peak elems: eager T*T={T*T}  "
        f"blocked/online bound={max_score_tile_elems(T, 64)}  "
        f"(no full T×T materialization)"
    )

    t_eager = bench(eager_tril_attn, (Q, K, V), device=device)
    t_blocked = bench(blocked_tril_attn, (Q, K, V), device=device)
    t_online = bench(online_tril_attn, (Q, K, V), device=device)
    t_triton = bench(triton_tril_attn, (Q, K, V), device=device)

    print(f"eager   median: {t_eager:.3f} ms")
    print(f"blocked median: {t_blocked:.3f} ms  ({t_eager / t_blocked:.2f}× vs eager)" if t_blocked else "")
    print(f"online  median: {t_online:.3f} ms  ({t_eager / t_online:.2f}× vs eager)" if t_online else "")
    print(f"triton  median: {t_triton:.3f} ms  ({t_eager / t_triton:.2f}× vs eager)" if t_triton else "")

    if device.type != "cuda":
        print(
            "NOTE: CPU-only box — Triton kernel not executed; "
            "BDH_ATTN_IMPL=triton uses blocked/online pure-PyTorch fallback. "
            "Online/blocked is vectorized (opt/blocked-vec) and much faster than "
            "the old Python row loop, but on CPU still typically slower than eager "
            "for modest T (peak score memory still ≪ T×T). "
            "Do not claim GPU wins from these CPU medians. "
            "See also benchmarks/bench_blocked_vec.py."
        )


if __name__ == "__main__":
    main()
