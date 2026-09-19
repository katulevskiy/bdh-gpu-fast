#!/usr/bin/env python3
"""Honest CPU peak-memory probe: eager vs blocked vs online strict-tril attn.

Measures, for growing T:

  * theoretical score-element bound (eager T×T vs max_score_tile_elems)
  * measured peak score-tile elems via matmul spy (no full T×T on blocked)
  * tracemalloc peak Python bytes during one forward
  * wall-time median (to show peak-mem win can coexist with slower wall)

DEFAULTS UNCHANGED — does not flip BDH_ATTN_IMPL (stays eager). CPU-only;
do not cite these numbers as GPU wins. Attention math remains tril(diagonal=-1).

Usage:
  .venv/bin/python benchmarks/bench_attn_mem.py
  .venv/bin/python benchmarks/bench_attn_mem.py --T 64,128,256,512 --smoke
"""

from __future__ import annotations

import argparse
import gc
import statistics
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Callable, List, Sequence, Tuple

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kernels.attention import (  # noqa: E402
    DEFAULT_BLOCK_COLD,
    blocked_tril_attn,
    eager_tril_attn,
    max_score_tile_elems,
    online_tril_attn,
)


ImplFn = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]


def _make_qkv(
    B: int, H: int, T: int, N: int, D: int, seed: int, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    g = torch.Generator(device="cpu").manual_seed(seed)
    Q = torch.randn(B, H, T, N, generator=g, device="cpu").to(device)
    K = Q.clone()
    V = torch.randn(B, 1, T, D, generator=g, device="cpu").to(device)
    return Q, K, V


def peak_score_elems(fn: ImplFn, Q, K, V) -> int:
    """Max elems among Q@K.T-style score tiles (not scores@V outputs).

    Identifies the right-hand operand via storage: any view whose base
    storage matches ``K`` (including ``K.transpose`` / slices) is a score
    GEMM; ``V`` storage is ignored. Avoids T==N / T==D shape collisions.
    """
    B, H = Q.shape[0], Q.shape[1]
    k_storages = {K.untyped_storage().data_ptr()}
    # Contiguous clones inside blocked widen path still derive from Kf —
    # fall back to shape: right operand looks like K.T (…, N, S).
    N = Q.shape[-1]
    D = V.shape[-1]
    nonlocal_peak = [0]
    orig = torch.Tensor.__matmul__

    def _is_k_rhs(other: torch.Tensor) -> bool:
        try:
            if other.untyped_storage().data_ptr() in k_storages:
                return True
        except Exception:
            pass
        # K.T / contiguous copy of a K slice: (..., N, S) with S != D
        # or S == D but -2 == N (head) distinguishing from V (..., S, D).
        if other.dim() >= 2 and other.shape[-2] == N:
            # V is (..., S, D); K.T is (..., N, S). When S==D both end in D,
            # but V has feature last — still (S,D) with S possibly ==N.
            # Prefer: treat as K.T when the contracting dim on the left is N
            # AND other is not clearly V broadcast (size 1 head already expanded).
            return True
        return False

    def spy(self, other):
        out = orig(self, other)
        if (
            out.dim() == 4
            and out.shape[0] == B
            and out.shape[1] == H
            and self.dim() >= 1
            and other.dim() >= 2
            and self.shape[-1] == N
            and _is_k_rhs(other)
            # Exclude scores@V: left last-dim is sequence (not head N) OR
            # right last-dim is D while right[-2] != N. When T==N the left
            # of scores@V also ends in N — then require right ends in D
            # AND right comes from V storage.
            and not (
                other.shape[-1] == D
                and other.shape[-2] != N
            )
            and not (
                other.shape[-1] == D
                and other.shape[-2] == N
                and self.shape[-2] == self.shape[-1]  # square scores @ V
            )
        ):
            elems = int(out.shape[-2]) * int(out.shape[-1])
            nonlocal_peak[0] = max(nonlocal_peak[0], elems)
        return out

    torch.Tensor.__matmul__ = spy  # type: ignore[method-assign]
    try:
        _ = fn(Q, K, V)
    finally:
        torch.Tensor.__matmul__ = orig  # type: ignore[method-assign]
    return nonlocal_peak[0]


def peak_tracemalloc_bytes(fn: ImplFn, Q, K, V) -> int:
    """Peak Python allocator bytes during one forward (after GC)."""
    gc.collect()
    tracemalloc.start()
    try:
        tracemalloc.reset_peak()
        _ = fn(Q, K, V)
        _, peak = tracemalloc.get_traced_memory()
        return int(peak)
    finally:
        tracemalloc.stop()
        gc.collect()


def wall_ms(fn: ImplFn, Q, K, V, *, warmup: int = 3, reps: int = 11) -> float:
    for _ in range(warmup):
        fn(Q, K, V)
    xs: List[float] = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn(Q, K, V)
        xs.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(xs)


def probe_T(
    T: int,
    *,
    B: int,
    H: int,
    N: int,
    D: int,
    block_size: int,
    device: torch.device,
    warmup: int,
    reps: int,
) -> dict:
    Q, K, V = _make_qkv(B, H, T, N, D, seed=T * 17 + 3, device=device)

    def eager(q, k, v):
        return eager_tril_attn(q, k, v)

    def blocked(q, k, v):
        return blocked_tril_attn(q, k, v, block_size=block_size)

    def online(q, k, v):
        return online_tril_attn(q, k, v, block_size=block_size)

    ref = eager(Q, K, V)
    blk = blocked(Q, K, V)
    onl = online(Q, K, V)
    maxdiff_b = (blk - ref).abs().max().item()
    maxdiff_o = (onl - ref).abs().max().item()
    assert torch.allclose(blk, ref, rtol=1e-4, atol=1e-4), maxdiff_b
    assert torch.allclose(onl, ref, rtol=1e-4, atol=1e-4), maxdiff_o
    assert torch.equal(blk, onl)
    # Strict tril(-1): position 0 must be exact zero.
    assert torch.equal(ref[:, :, 0, :], torch.zeros_like(ref[:, :, 0, :]))

    eager_bound = T * T
    blocked_bound = max_score_tile_elems(T, block_size)

    peak_e = peak_score_elems(eager, Q, K, V)
    peak_b = peak_score_elems(blocked, Q, K, V)
    peak_o = peak_score_elems(online, Q, K, V)

    tm_e = peak_tracemalloc_bytes(eager, Q, K, V)
    tm_b = peak_tracemalloc_bytes(blocked, Q, K, V)
    tm_o = peak_tracemalloc_bytes(online, Q, K, V)

    te = wall_ms(eager, Q, K, V, warmup=warmup, reps=reps)
    tb = wall_ms(blocked, Q, K, V, warmup=warmup, reps=reps)
    to = wall_ms(online, Q, K, V, warmup=warmup, reps=reps)

    return {
        "T": T,
        "eager_bound": eager_bound,
        "blocked_bound": blocked_bound,
        "peak_score_eager": peak_e,
        "peak_score_blocked": peak_b,
        "peak_score_online": peak_o,
        "tm_eager": tm_e,
        "tm_blocked": tm_b,
        "tm_online": tm_o,
        "eager_ms": te,
        "blocked_ms": tb,
        "online_ms": to,
        "maxdiff_blocked": maxdiff_b,
        "maxdiff_online": maxdiff_o,
        # Peak-mem win = measured score tiles strictly below eager T×T
        # (bound may be looser than measured for small T).
        "peak_mem_blocked_wins": peak_b < peak_e,
        "wall_blocked_slower": tb > te,
    }


def format_table(rows: Sequence[dict], *, B: int, H: int) -> str:
    """Primary peak metric = score elems × B × H × 4B (fp32).

    tracemalloc is reported only in paste summary — blocked's many small
    tiles can show *higher* Python allocator peaks while score tiles are
    still ≪ T×T (the OOM-relevant figure).
    """
    bytes_per = B * H * 4  # fp32 elems → bytes for full BH score tensor
    hdr = (
        f"{'T':>5} {'eag_bnd':>9} {'blk_bnd':>9} "
        f"{'pk_e':>8} {'pk_b':>8} "
        f"{'eag_MiB':>8} {'blk_MiB':>8} "
        f"{'eag_ms':>8} {'blk_ms':>8} {'e/b':>6} {'peak_win':>8}"
    )
    lines = [hdr]
    for r in rows:
        ratio = r["eager_ms"] / r["blocked_ms"] if r["blocked_ms"] > 0 else float("inf")
        peak_win = "YES" if r["peak_mem_blocked_wins"] else "no"
        e_mib = r["peak_score_eager"] * bytes_per / (1024 * 1024)
        b_mib = r["peak_score_blocked"] * bytes_per / (1024 * 1024)
        lines.append(
            f"{r['T']:5d} {r['eager_bound']:9d} {r['blocked_bound']:9d} "
            f"{r['peak_score_eager']:8d} {r['peak_score_blocked']:8d} "
            f"{e_mib:8.3f} {b_mib:8.3f} "
            f"{r['eager_ms']:8.3f} {r['blocked_ms']:8.3f} {ratio:6.2f}x "
            f"{peak_win:>8}"
        )
    return "\n".join(lines)


def verdict_lines(rows: Sequence[dict]) -> List[str]:
    lines: List[str] = []
    peak_wins = [r for r in rows if r["peak_mem_blocked_wins"]]
    both = [r for r in peak_wins if r["wall_blocked_slower"]]
    wall_wins = [r for r in rows if r["blocked_ms"] < r["eager_ms"]]

    lines.append(
        f"  blocked/online peak-score win vs eager at T∈"
        f"{[r['T'] for r in peak_wins] or 'none'} "
        f"(measured Q@K.T score elems; online==blocked)."
    )
    if both:
        lines.append(
            f"  At T∈{[r['T'] for r in both]} blocked wins **peak mem** "
            f"even though wall is slower than eager — use blocked for "
            f"peak-score budget, not CPU wall."
        )
    else:
        lines.append(
            "  No T in this sweep where blocked peak-wins while wall-slower "
            "(unusual on mid-T CPU; still keep default eager for wall)."
        )
    if wall_wins:
        lines.append(
            f"  Wall: blocked beat eager at T∈{[r['T'] for r in wall_wins]} "
            f"(rare on mid-T; long-T decode is a different path)."
        )
    else:
        lines.append(
            "  Wall: blocked/online never beat eager in this cold-prefill sweep "
            "(expected on CPU mid-T)."
        )
    lines.append(
        "  online == blocked (alias). Default BDH_ATTN_IMPL remains eager. "
        "No GPU claims."
    )
    return lines


def parse_T_list(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--T",
        type=str,
        default="32,64,128,256,512,1024",
        help="Comma-separated sequence lengths",
    )
    ap.add_argument("--B", type=int, default=1)
    ap.add_argument("--H", type=int, default=2)
    ap.add_argument("--N", type=int, default=32)
    ap.add_argument("--D", type=int, default=64)
    ap.add_argument("--block-size", type=int, default=DEFAULT_BLOCK_COLD)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--reps", type=int, default=11)
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="Tiny T list + fewer reps for CI / pytest smoke",
    )
    args = ap.parse_args()

    if args.smoke:
        Ts = [32, 64, 128]
        warmup, reps = 1, 3
    else:
        Ts = parse_T_list(args.T)
        warmup, reps = args.warmup, args.reps

    device = torch.device("cpu")
    print(
        f"device={device} torch={torch.__version__} "
        f"cuda={torch.cuda.is_available()} "
        f"B={args.B} H={args.H} N={args.N} D={args.D} "
        f"BS={args.block_size}"
    )
    print(
        "Honest peak-mem probe (CPU). Defaults unchanged; "
        "tril(diagonal=-1); no GPU claims."
    )
    print()

    # Warm MKL / oneDNN a bit
    Qw, Kw, Vw = _make_qkv(args.B, args.H, 64, args.N, args.D, seed=0, device=device)
    for _ in range(8):
        eager_tril_attn(Qw, Kw, Vw)
        blocked_tril_attn(Qw, Kw, Vw, block_size=args.block_size)

    rows: List[dict] = []
    for T in Ts:
        r = probe_T(
            T,
            B=args.B,
            H=args.H,
            N=args.N,
            D=args.D,
            block_size=args.block_size,
            device=device,
            warmup=warmup,
            reps=reps,
        )
        rows.append(r)

    print(format_table(rows, B=args.B, H=args.H))
    print()
    print("--- verdict ---")
    for line in verdict_lines(rows):
        print(line)

    # Machine-readable paste line for OPT_NOTES
    print("--- paste summary ---")
    for r in rows:
        print(
            f"T={r['T']} bound_e={r['eager_bound']} bound_b={r['blocked_bound']} "
            f"pk_e={r['peak_score_eager']} pk_b={r['peak_score_blocked']} "
            f"tm_e={r['tm_eager']} tm_b={r['tm_blocked']} "
            f"ms_e={r['eager_ms']:.3f} ms_b={r['blocked_ms']:.3f} "
            f"peak_win={int(r['peak_mem_blocked_wins'])} "
            f"wall_slower={int(r['wall_blocked_slower'])}"
        )


if __name__ == "__main__":
    main()
