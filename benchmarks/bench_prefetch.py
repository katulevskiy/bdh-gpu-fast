"""CPU microbench: BatchPrefetcher sync vs async + synthetic overlap proof.

Honest CPU-only numbers. Real get_batch is ~0.05–0.1 ms — smaller than typical
thread handoff — so end-to-end train_step deltas are often ~noise here.

The synthetic probe injects a known host delay inside the producer to prove
the double-buffer hides that delay behind a busy step (architectural win).
Do **not** treat CPU medians as GPU H2D-overlap claims.
"""

from __future__ import annotations

import os
import statistics
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ.setdefault("BDH_COMPILE", "0")
import train as tr  # noqa: E402
import bdh  # noqa: E402


def timed(fn, warmup=5, reps=40):
    for _ in range(warmup):
        fn()
    xs = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        xs.append(time.perf_counter() - t0)
    return statistics.median(xs)


def _busy_ms(ms: float):
    end = time.perf_counter() + ms / 1000.0
    n = 0
    while time.perf_counter() < end:
        n += 1
    return n


def bench_gather():
    tr.BLOCK_SIZE = 512
    tr.BATCH_SIZE = 32
    tr.device = torch.device("cpu")
    tr._train_data = tr._val_data = tr._offsets = None
    tr.fetch_data()
    th = timed(lambda: tr._gather_batch_host("train"), warmup=10, reps=80)
    tg = timed(lambda: tr.get_batch("train"), warmup=10, reps=80)
    return th, tg


def bench_synthetic_overlap(host_ms=2.0, step_ms=3.0):
    """Prove queue double-buffer hides host_ms behind step_ms.

    Patches ``_gather_batch_host`` with a busy-wait so the producer cost is
    known; compares serial (step then gather) vs BatchPrefetcher async.
    """
    tr.BLOCK_SIZE = 64
    tr.BATCH_SIZE = 4
    tr.device = torch.device("cpu")
    tr._train_data = tr._val_data = tr._offsets = None
    tr.fetch_data()
    # Patch the producer path (numpy) — that is what async BatchPrefetcher uses.
    real_np = tr._gather_batch_host_numpy

    def slow_gather(split):
        # sleep (not busy-wait): releases the core so train_step can run —
        # models non-compute host prep / H2D wait without OMP fight.
        time.sleep(host_ms / 1000.0)
        return real_np(split)

    tr._gather_batch_host_numpy = slow_gather  # type: ignore[assignment]
    try:
        def serial():
            _busy_ms(step_ms)  # step stays CPU-bound
            slow_gather("train")  # host delay is sleep inside

        t_serial = timed(serial, warmup=3, reps=25)

        loader = tr.BatchPrefetcher("train", async_host=True)
        # Prime: one batch ready, producer filling next (slow numpy gather).
        _ = loader.next()

        def overlapped():
            _busy_ms(step_ms)
            loader.next()

        t_async = timed(overlapped, warmup=3, reps=25)
        loader.close()
    finally:
        tr._gather_batch_host_numpy = real_np  # type: ignore[assignment]
    return t_serial, t_async, host_ms, step_ms


def bench_train_loop():
    tr.BLOCK_SIZE = 64
    tr.BATCH_SIZE = 4
    tr.device = torch.device("cpu")
    tr._train_data = tr._val_data = tr._offsets = None
    tr.fetch_data()
    cfg = bdh.BDHConfig(
        n_layer=1, n_embd=32, n_head=1, mlp_internal_dim_multiplier=8, dropout=0.0
    )

    def run(async_host: bool, steps: int = 30):
        model = bdh.BDH(cfg).to(tr.device)
        opt = tr.make_optimizer(model)
        loader = tr.BatchPrefetcher("train", async_host=async_host)
        try:
            x, y = loader.next()

            def one():
                nonlocal x, y
                tr.train_step(model, opt, x, y)
                x, y = loader.next()

            return timed(one, warmup=5, reps=steps)
        finally:
            loader.close()

    # Warm process with a throwaway sync run, then measure both.
    _ = run(False, steps=5)
    t_sync = run(False)
    t_async = run(True)
    return t_sync, t_async


def main():
    print(f"device=cpu torch={torch.__version__} cuda={torch.cuda.is_available()}")
    th, tg = bench_gather()
    print(f"host gather median:                 {th*1000:.3f} ms")
    print(f"get_batch median:                   {tg*1000:.3f} ms")

    t_serial, t_async, host_ms, step_ms = bench_synthetic_overlap()
    print(
        f"synthetic overlap (host {host_ms} ms + step {step_ms} ms): "
        f"serial {t_serial*1000:.3f} ms | async {t_async*1000:.3f} ms | "
        f"speedup {t_serial/max(t_async,1e-9):.2f}×"
    )

    t_sync, t_as = bench_train_loop()
    print(
        f"tiny train_step+prefetch loop:      sync {t_sync*1000:.3f} ms | "
        f"async {t_as*1000:.3f} ms | ratio sync/async {t_sync/max(t_as,1e-9):.2f}×"
    )
    print(
        "Honesty: real gather ≪ step on this box → e2e delta is ~noise; "
        "synthetic probe is the overlap proof. GPU H2D overlap unmeasured."
    )


if __name__ == "__main__":
    main()
