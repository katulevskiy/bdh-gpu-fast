"""Honest CPU microbench: layout-v2 eval encoder cache vs train einsum.

No GPU claims. Medians under OMP_NUM_THREADS (default env).
"""

from __future__ import annotations

import os
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import bdh


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def timed(fn, warmup=8, reps=31):
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
    omp = os.environ.get("OMP_NUM_THREADS", "unset")
    device = torch.device("cpu")
    print(f"device={device} torch={torch.__version__} cuda={torch.cuda.is_available()} OMP={omp}")

    B, D, nh, N = 4, 128, 4, 1024
    torch.manual_seed(0)
    w = torch.randn(nh, D, N, device=device)
    wl = w.transpose(1, 2).reshape(nh * N, D).contiguous()

    print("encoder micro (ReLU) einsum vs cached-linear")
    for T in (1, 16, 32, 64, 128):
        x = torch.randn(B, T, D, device=device)

        def einsum():
            return F.relu(torch.einsum("btd,hdn->bthn", x, w), inplace=True)

        def linear_cached():
            return F.relu(F.linear(x, wl).view(B, T, nh, N), inplace=True)

        te = timed(einsum) * 1e3
        tl = timed(linear_cached) * 1e3
        print(f"  T={T:<3}  einsum {te:.3f} ms  cached-linear {tl:.3f} ms  ratio {te/tl:.3f}x")

    cfg = bdh.BDHConfig(
        n_layer=4,
        n_embd=128,
        n_head=4,
        mlp_internal_dim_multiplier=32,
        dropout=0.0,
        vocab_size=256,
    )
    m = bdh.BDH(cfg).to(device)
    print("full forward train(einsum) vs eval(cached-linear)")
    for T in (1, 32, 128):
        idx = torch.randint(0, cfg.vocab_size, (B, T), device=device)
        m.train()
        with torch.no_grad():
            t_train = timed(lambda: m(idx)) * 1e3
        m.eval()
        with torch.no_grad():
            m(idx)  # warm cache
            t_eval = timed(lambda: m(idx)) * 1e3
        print(f"  T={T:<3}  train {t_train:.3f} ms  eval {t_eval:.3f} ms  ratio {t_train/t_eval:.3f}x")

    prompt = torch.randint(0, cfg.vocab_size, (1, 32), device=device)

    def gen(seed=0):
        with torch.no_grad():
            torch.manual_seed(seed)
            return m.generate(prompt.clone(), max_new_tokens=32)

    m.eval()
    with torch.no_grad():
        m(prompt)
    t_gen = timed(gen, warmup=2, reps=7) * 1e3
    print(f"generate(prompt=32,new=32) eval  {t_gen:.3f} ms  (cached encoder path)")

    m.train()
    idx = torch.randint(0, cfg.vocab_size, (2, 16), device=device)
    with torch.no_grad():
        a, _ = m(idx)
    m.eval()
    with torch.no_grad():
        b, _ = m(idx)
    print(f"train==eval logits bit-identical: {torch.equal(a, b)}")


if __name__ == "__main__":
    main()
