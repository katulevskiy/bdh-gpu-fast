"""Microbench: train-step variants (fused AdamW, set_to_none, compile path).

Also used by opt/train-fuse to document CPU honesty after sync-light logging.

CPU-honest: no CUDA on the default runner. Measures median step time for a
tiny BDH config. Compile is opt-in (BDH_BENCH_COMPILE=1) because inductor
warmup is long on CPU.
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

import bdh

# Import train helpers without forcing compile during import benches.
os.environ.setdefault("BDH_COMPILE", "0")
import train as tr  # noqa: E402


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def timed(fn, warmup=3, reps=15):
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


def _cfg():
    return bdh.BDHConfig(
        n_layer=2,
        n_embd=64,
        n_head=2,
        mlp_internal_dim_multiplier=16,
        dropout=0.0,
    )


def _batch(device, B=4, T=64):
    x = torch.randint(0, 256, (B, T), device=device)
    y = torch.randint(0, 256, (B, T), device=device)
    return x, y


def make_legacy_step(model, device):
    """Upstream-ish: plain AdamW, zero_grad() (zeros), no set_to_none."""
    opt = torch.optim.AdamW(
        model.parameters(), lr=tr.LEARNING_RATE, weight_decay=tr.WEIGHT_DECAY
    )
    x, y = _batch(device)

    def step():
        nonlocal x, y
        with tr.ctx:
            _, loss = model(x, y)
        loss.backward()
        opt.step()
        opt.zero_grad()  # fills zeros — more allocator traffic
        x, y = _batch(device)
        return loss

    return step


def make_opt_step(model, device, fused=True):
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=tr.LEARNING_RATE,
        weight_decay=tr.WEIGHT_DECAY,
        fused=fused,
    )
    x, y = _batch(device)

    def step():
        nonlocal x, y
        loss = tr.train_step(model, opt, x, y)
        x, y = _batch(device)
        return loss

    return step


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = _cfg()
    print(f"device={device} layers={cfg.n_layer} d={cfg.n_embd} B=4 T=64")

    torch.manual_seed(0)
    m_legacy = bdh.BDH(cfg).to(device)
    torch.manual_seed(0)
    m_opt = bdh.BDH(cfg).to(device)
    m_opt.load_state_dict(m_legacy.state_dict())

    t_legacy = timed(make_legacy_step(m_legacy, device))
    try:
        t_fused = timed(make_opt_step(m_opt, device, fused=True))
        fused_ok = True
    except Exception as e:
        print(f"fused AdamW step failed ({e}); measuring foreach/default path")
        torch.manual_seed(0)
        m_opt = bdh.BDH(cfg).to(device)
        m_opt.load_state_dict(m_legacy.state_dict())
        t_fused = timed(make_opt_step(m_opt, device, fused=False))
        fused_ok = False

    print(f"train_step legacy (zero_grad fill, no fused) median: {t_legacy*1000:.2f} ms")
    label = "fused+set_to_none" if fused_ok else "set_to_none (fused unavailable)"
    print(
        f"train_step {label} median:                     {t_fused*1000:.2f} ms  "
        f"({t_legacy/t_fused:.2f}x)"
    )

    torch.manual_seed(0)
    m_a = bdh.BDH(cfg).to(device)
    torch.manual_seed(0)
    m_b = bdh.BDH(cfg).to(device)
    m_b.load_state_dict(m_a.state_dict())
    opt_a = torch.optim.AdamW(m_a.parameters(), lr=1e-3, fused=fused_ok)
    opt_b = torch.optim.AdamW(m_b.parameters(), lr=1e-3, fused=fused_ok)
    x, y = _batch(device)

    def step_fill():
        with tr.ctx:
            _, loss = m_a(x, y)
        loss.backward()
        opt_a.step()
        opt_a.zero_grad(set_to_none=False)

    def step_none():
        with tr.ctx:
            _, loss = m_b(x, y)
        loss.backward()
        opt_b.step()
        opt_b.zero_grad(set_to_none=True)

    t_fill = timed(step_fill)
    t_none = timed(step_none)
    print(f"zero_grad fill median:      {t_fill*1000:.2f} ms")
    print(f"zero_grad set_to_none median: {t_none*1000:.2f} ms  ({t_fill/t_none:.2f}x)")

    if os.environ.get("BDH_BENCH_COMPILE", "0") in ("1", "true", "True"):
        print("BDH_BENCH_COMPILE=1 — warming torch.compile (can take minutes on CPU)...")
        torch.manual_seed(0)
        m_c = torch.compile(
            bdh.BDH(cfg).to(device),
            mode=os.environ.get("BDH_COMPILE_MODE", "default"),
        )
        opt_c = torch.optim.AdamW(
            m_c.parameters(), lr=1e-3, weight_decay=0.1, fused=fused_ok
        )
        x, y = _batch(device)

        def step_c():
            return tr.train_step(m_c, opt_c, x, y)

        for _ in range(2):
            step_c()
        t_c = timed(step_c, warmup=2, reps=10)
        print(
            f"train_step compiled median: {t_c*1000:.2f} ms  "
            f"({t_fused/t_c:.2f}x vs fused eager)"
        )
    else:
        print("skip compile bench (set BDH_BENCH_COMPILE=1 to enable)")

    print(
        "Note: CUDA graphs / reduce-overhead need a GPU; see train_fast.py comments."
    )


if __name__ == "__main__":
    main()
