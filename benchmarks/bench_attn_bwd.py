"""Microbench: train_step AUTOGRAD 0/1 × IMPL eager|blocked (analytic StrictTrilAttnFn).

Compares a tiny BDH train step under:
  * eager forward + PyTorch autograd (AUTOGRAD=0)
  * eager forward + dense analytic bwd (AUTOGRAD=1, IMPL=eager)
  * blocked/online forward + tiled analytic bwd (AUTOGRAD=1, IMPL=blocked)

CPU-honest: this sandbox has no CUDA. Absolute ms are CPU-only — do not claim
GPU wins. On CPU, blocked forward is often slower than eager (see
opt/blocked-vec); the win for blocked+AUTOGRAD is peak score memory (no full
T×T in fwd or bwd), not wall time on this box.
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

# Keep compile off for a clean AUTOGRAD A/B.
os.environ.setdefault("BDH_COMPILE", "0")

import bdh  # noqa: E402


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def timed(fn, warmup=3, reps=12):
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


def make_train_step(model, device, fused: bool):
    try:
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, fused=fused)
    except Exception:
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        fused = False
    x, y = _batch(device)

    def step():
        opt.zero_grad(set_to_none=True)
        _, loss = model(x, y)
        loss.backward()
        opt.step()
        return loss

    return step, fused


def bench_combo(cfg, device, *, autograd: str, impl: str, state_dict, fused: bool):
    os.environ["BDH_ATTN_AUTOGRAD"] = autograd
    os.environ["BDH_ATTN_IMPL"] = impl
    torch.manual_seed(0)
    model = bdh.BDH(cfg).to(device)
    model.load_state_dict(state_dict)
    model.train()
    step, fused_ok = make_train_step(model, device, fused=fused)
    step()  # dry run
    t = timed(step, warmup=2, reps=8)
    return t, fused_ok


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = _cfg()
    print(
        f"device={device} layers={cfg.n_layer} d={cfg.n_embd} nh={cfg.n_head} "
        f"B=4 T=64 dropout=0"
    )
    print(
        f"torch={torch.__version__} cuda={torch.cuda.is_available()} "
        f"(CPU-only numbers are not GPU claims)"
    )

    torch.manual_seed(0)
    ref = bdh.BDH(cfg).to(device)
    state = {k: v.detach().clone() for k, v in ref.state_dict().items()}

    fused = device.type == "cuda"
    combos = [
        ("0", "eager", "eager PyTorch bwd"),
        ("1", "eager", "eager fwd + dense analytic bwd"),
        ("1", "blocked", "blocked fwd + tiled analytic bwd"),
        ("1", "online", "online(=blocked) fwd + tiled analytic bwd"),
    ]
    results = []
    fused_ok = False
    for flag, impl, label in combos:
        t, fused_ok = bench_combo(
            cfg, device, autograd=flag, impl=impl, state_dict=state, fused=fused
        )
        results.append((flag, impl, label, t))
        print(
            f"AUTOGRAD={flag} IMPL={impl:7s}  median {t * 1000:.2f} ms  ({label})"
        )

    t_ref = results[0][3]
    print(f"optimizer fused={fused_ok}")
    if t_ref > 0:
        for flag, impl, label, t in results[1:]:
            print(f"  vs AUTOGRAD=0/eager: {t_ref / t:.2f}x  ({label})")

    if device.type != "cuda":
        print(
            "honest: CPU medians only. Blocked+AUTOGRAD avoids full T×T in fwd "
            "and bwd (peak score tiles) but is typically slower wall-time than "
            "eager on CPU. GPU unmeasured — re-run on A100/H100 before claiming "
            "train wins."
        )
    else:
        print(
            "GPU box: blocked|online + AUTOGRAD=1 is the intended fused-fwd + "
            "tiled-analytic-bwd train path."
        )


if __name__ == "__main__":
    main()
