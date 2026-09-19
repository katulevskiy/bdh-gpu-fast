"""Microbench: train_step with BDH_ATTN_AUTOGRAD=0 vs 1 (analytic StrictTrilAttnFn).

Compares a tiny BDH train step under the default eager path (PyTorch autograd
through GEMMs) vs the opt-in analytic tril attention Function.

CPU-honest: this sandbox has no CUDA. Absolute ms are CPU-only — do not claim
GPU wins. Profile of AUTOGRAD=1 may still be dominated by eager T×T forward
when BDH_ATTN_IMPL=eager (analytic bwd recomputes M; forward still materializes
scores unless IMPL=blocked|triton|cuda).
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
os.environ.setdefault("BDH_ATTN_IMPL", "eager")

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


def bench_autograd_flag(cfg, device, *, flag: str, state_dict, fused: bool):
    os.environ["BDH_ATTN_AUTOGRAD"] = flag
    # Clear any sticky IMPL from prior runs in-process.
    os.environ.setdefault("BDH_ATTN_IMPL", "eager")
    torch.manual_seed(0)
    model = bdh.BDH(cfg).to(device)
    model.load_state_dict(state_dict)
    model.train()
    step, fused_ok = make_train_step(model, device, fused=fused)
    # One dry run to catch wiring errors before timing.
    step()
    t = timed(step, warmup=2, reps=10)
    return t, fused_ok


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = _cfg()
    print(
        f"device={device} layers={cfg.n_layer} d={cfg.n_embd} nh={cfg.n_head} "
        f"B=4 T=64 dropout=0 IMPL={os.environ.get('BDH_ATTN_IMPL', 'eager')}"
    )
    print(
        f"torch={torch.__version__} cuda={torch.cuda.is_available()} "
        f"(CPU-only numbers are not GPU claims)"
    )

    torch.manual_seed(0)
    ref = bdh.BDH(cfg).to(device)
    state = {k: v.detach().clone() for k, v in ref.state_dict().items()}

    fused = device.type == "cuda"
    t0, fused_ok = bench_autograd_flag(
        cfg, device, flag="0", state_dict=state, fused=fused
    )
    t1, _ = bench_autograd_flag(
        cfg, device, flag="1", state_dict=state, fused=fused
    )

    ratio = t0 / t1 if t1 > 0 else float("inf")
    print(
        f"BDH_ATTN_AUTOGRAD=0 (eager PyTorch bwd) train_step median: "
        f"{t0 * 1000:.2f} ms"
    )
    print(
        f"BDH_ATTN_AUTOGRAD=1 (StrictTrilAttnFn analytic) train_step median: "
        f"{t1 * 1000:.2f} ms  ({ratio:.2f}x vs off; >1 means analytic faster)"
    )
    print(f"optimizer fused={fused_ok}")
    if device.type != "cuda":
        print(
            "honest: CPU medians only. With IMPL=eager, AUTOGRAD=1 still pays "
            "full T×T forward + analytic M recompute — expect similar or "
            "slightly slower than autograd-through-eager. GPU unmeasured."
        )
    else:
        print(
            "GPU box: also try BDH_ATTN_IMPL=blocked|triton|cuda with "
            "AUTOGRAD=1 (analytic bwd unlocks non-differentiable forwards)."
        )


if __name__ == "__main__":
    main()
