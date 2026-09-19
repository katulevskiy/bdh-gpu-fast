"""Microbenchmark: baseline vs optimized forward + generate (CPU-friendly)."""

from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import bdh
import bdh_baseline as baseline


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def timed(fn, warmup=3, reps=10):
    for _ in range(warmup):
        fn()
    _sync()
    xs = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        _sync()
        xs.append(time.perf_counter() - t0)
    return statistics.median(xs)


def main():
    device = torch.device("cpu")
    cfg = bdh.BDHConfig(
        n_layer=4,
        n_embd=128,
        n_head=4,
        mlp_internal_dim_multiplier=32,
        dropout=0.0,
    )
    torch.manual_seed(0)
    m_base = baseline.BDH(cfg).to(device).eval()
    m_opt = bdh.BDH(cfg).to(device).eval()
    m_opt.load_state_dict(m_base.state_dict())

    B, T = 4, 128
    x = torch.randint(0, cfg.vocab_size, (B, T), device=device)
    prompt = torch.randint(0, cfg.vocab_size, (1, 16), device=device)

    def fwd_base():
        with torch.no_grad():
            m_base(x)

    def fwd_opt():
        with torch.no_grad():
            m_opt(x)

    def gen_base():
        with torch.no_grad():
            torch.manual_seed(0)
            m_base.generate(prompt.clone(), max_new_tokens=32)

    def gen_opt():
        with torch.no_grad():
            torch.manual_seed(0)
            m_opt.generate(prompt.clone(), max_new_tokens=32)

    tb = timed(fwd_base)
    to = timed(fwd_opt)
    gb = timed(gen_base, warmup=1, reps=5)
    go = timed(gen_opt, warmup=1, reps=5)

    print(f"device={device} cfg layers={cfg.n_layer} d={cfg.n_embd} B={B} T={T}")
    print(f"forward baseline median: {tb*1000:.2f} ms")
    print(f"forward optimized median: {to*1000:.2f} ms  ({tb/to:.2f}x)")
    print(f"generate(32) baseline median: {gb*1000:.2f} ms")
    print(f"generate(32) optimized median: {go*1000:.2f} ms  ({gb/go:.2f}x)")
    print(
        "Note: no CUDA on this runner; forward wins are mostly layout/RoPE."
        " generate speedup comes from KV-style cache."
    )


if __name__ == "__main__":
    main()
