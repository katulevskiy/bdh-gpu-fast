"""Microbench: residual LN temporaries (functional add vs in-place reuse).

Honest CPU numbers for opt/ln-deepen. Not a GPU claim.
Torch has no eager affine-free fused add+LayerNorm; this compares the #30
out-of-place ``x + y`` form vs reusing the inner LN output via ``y.add_(x)``.
"""

from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.profiler import ProfilerActivity, profile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import bdh


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def timed(fn, warmup=20, reps=100):
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


def residual_oop(x, y_mlp, shape, eps):
    y = F.layer_norm(y_mlp, shape, weight=None, bias=None, eps=eps)
    return F.layer_norm(x + y, shape, weight=None, bias=None, eps=eps)


def residual_inplace(x, y_mlp, shape, eps):
    y = F.layer_norm(y_mlp, shape, weight=None, bias=None, eps=eps)
    y.add_(x)
    return F.layer_norm(y, shape, weight=None, bias=None, eps=eps)


def main():
    device = torch.device("cpu")
    B, T, D = 4, 64, 128
    shape = (D,)
    eps = 1e-5
    torch.manual_seed(0)
    x = torch.randn(B, T, D, device=device)
    y_mlp = torch.randn(B, T, D, device=device)

    # Bit-identical check
    a = residual_oop(x, y_mlp, shape, eps)
    b = residual_inplace(x.clone(), y_mlp.clone(), shape, eps)
    print(f"bitexact residual: {torch.equal(a, b)}")

    t_oop = timed(lambda: residual_oop(x, y_mlp, shape, eps))
    t_ip = timed(lambda: residual_inplace(x.clone(), y_mlp.clone(), shape, eps))
    print(f"residual-only median oop:     {t_oop*1e3:.3f} ms")
    print(f"residual-only median inplace: {t_ip*1e3:.3f} ms")
    print(f"residual-only ratio oop/ip:   {t_oop/t_ip:.3f}x")

    # Op mix (one call each, after warmup)
    residual_oop(x, y_mlp, shape, eps)
    residual_inplace(x.clone(), y_mlp.clone(), shape, eps)
    with profile(activities=[ProfilerActivity.CPU]) as p:
        for _ in range(40):
            residual_oop(x, y_mlp, shape, eps)
    oop_ops = {e.key: e.count for e in p.key_averages()
               if e.key in ("aten::add", "aten::add_", "aten::native_layer_norm")}
    with profile(activities=[ProfilerActivity.CPU]) as p:
        for _ in range(40):
            residual_inplace(x.clone(), y_mlp.clone(), shape, eps)
    ip_ops = {e.key: e.count for e in p.key_averages()
              if e.key in ("aten::add", "aten::add_", "aten::native_layer_norm")}
    print(f"oop ops (40x): {oop_ops}")
    print(f"inplace ops (40x): {ip_ops}")

    # Full forward tip-style (defaults unchanged — both use model._residual_ln)
    cfg = bdh.BDHConfig(
        n_layer=4, n_embd=128, n_head=4, mlp_internal_dim_multiplier=32, dropout=0.0
    )

    def fwd_with(res_fn):
        m = bdh.BDH(cfg).eval()
        m._residual_ln = res_fn.__get__(m, bdh.BDH)
        xb = torch.randint(0, cfg.vocab_size, (4, 64), device=device)

        def run():
            with torch.no_grad():
                m(xb)

        return timed(run, warmup=3, reps=12)

    def _model_oop(self, x_t, y):
        return residual_oop(x_t, y, self._ln_shape, self._ln_eps)

    def _model_ip(self, x_t, y):
        return residual_inplace(x_t, y, self._ln_shape, self._ln_eps)

    t_fo = fwd_with(_model_oop)
    t_fi = fwd_with(_model_ip)
    print(f"forward median oop:     {t_fo*1e3:.2f} ms")
    print(f"forward median inplace: {t_fi*1e3:.2f} ms")
    print(f"forward ratio oop/ip:   {t_fo/t_fi:.3f}x")
    print("Note: e2e forward on CPU is often ~noise; residual-only + op mix are the honest signal.")


if __name__ == "__main__":
    main()
