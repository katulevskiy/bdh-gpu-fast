#!/usr/bin/env python3
"""CPU microbench: eager vs vectorized blocked/online strict-tril attention.

Records median ms for mid-T shapes used in OPT notes. Does **not** claim GPU
wins. Default BDH_ATTN_IMPL remains eager — this measures the blocked path.
"""

from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kernels.attention import (  # noqa: E402
    DEFAULT_BLOCK_COLD,
    blocked_tril_attn,
    eager_tril_attn,
    max_score_tile_elems,
    online_tril_attn,
    pick_cold_block_size,
)


def bench(fn, args, warmup=8, iters=40):
    for _ in range(warmup):
        fn(*args)
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn(*args)
        times.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(times)


def main():
    device = torch.device("cpu")
    B, H, N, D = 1, 2, 32, 64  # tiny B / n_head as in the opt brief
    print(f"device={device}  B={B} H={H} N={N} D={D}  torch={torch.__version__}")
    print(f"DEFAULT_BLOCK_COLD={DEFAULT_BLOCK_COLD}  pick_cold_block_size(256)={pick_cold_block_size(256)}")
    print(
        f"{'T':>4} {'eager_ms':>10} {'blocked_ms':>12} {'online_ms':>10} "
        f"{'eag/blk':>8} {'peak_elems':>12} {'eager_TxT':>10}"
    )

    # Warm MKL / oneDNN
    Qw = torch.randn(B, H, 64, N, device=device)
    Vw = torch.randn(B, 1, 64, D, device=device)
    for _ in range(15):
        eager_tril_attn(Qw, Qw, Vw)
        blocked_tril_attn(Qw, Qw, Vw)

    rows = []
    for T in (32, 64, 128, 256, 512, 1024):
        torch.manual_seed(0)
        Q = torch.randn(B, H, T, N, device=device)
        K = Q.clone()
        V = torch.randn(B, 1, T, D, device=device)

        ref = eager_tril_attn(Q, K, V)
        blk = blocked_tril_attn(Q, K, V)
        onl = online_tril_attn(Q, K, V)
        d_blk = (blk - ref).abs().max().item()
        # Long-T tile reorder vs one eager GEMM — allow mild fp drift.
        tol = 1e-3 if T >= 512 else 1e-4
        assert torch.allclose(blk, ref, rtol=tol, atol=tol), d_blk
        assert torch.equal(blk, onl)

        te = bench(eager_tril_attn, (Q, K, V))
        tb = bench(blocked_tril_attn, (Q, K, V))
        to = bench(online_tril_attn, (Q, K, V))
        peak = max_score_tile_elems(T)  # adaptive BS at T>=256
        ratio = te / tb if tb > 0 else float("inf")
        print(
            f"{T:4d} {te:10.3f} {tb:12.3f} {to:10.3f} "
            f"{ratio:8.3f}x {peak:12d} {T*T:10d}"
        )
        rows.append((T, te, tb, to, ratio, peak, d_blk))

    print()
    print("Notes:")
    print("  - blocked == online (alias).")
    print("  - ratio = eager/blocked (>1 means blocked faster than eager).")
    print("  - Vectorized blocked aims to beat the *old* Python-row blocked wall;")
    print("    beating eager on CPU is not required and usually fails for mid T.")
    print("  - peak_elems is the documented score-tile bound (still < T×T for mid T).")
    if device.type != "cuda":
        print("  - CPU-only: do not claim GPU speedups from these medians.")


if __name__ == "__main__":
    main()
