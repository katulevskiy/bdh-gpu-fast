"""Microbench: encoder einsum vs F.linear (+ optional fused bias).

Honest CPU numbers for opt/encoder-fuse. Not a GPU claim.
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


def timed(fn, warmup=10, reps=50):
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
    B, T, D, nh, N = 4, 64, 128, 4, 256
    torch.manual_seed(0)
    x = torch.randn(B, T, D, device=device)
    w = torch.randn(nh, D, N, device=device)
    bias = torch.randn(nh, N, device=device)
    y = torch.randn(B, nh, T, D, device=device)

    def einsum_enc():
        return F.relu(torch.einsum("btd,hdn->bthn", x, w), inplace=True)

    def linear_enc():
        wl = w.transpose(1, 2).reshape(nh * N, D)
        return F.relu(F.linear(x, wl).view(B, T, nh, N), inplace=True)

    def einsum_bias():
        out = torch.einsum("btd,hdn->bthn", x, w)
        out.add_(bias)
        return F.relu(out, inplace=True)

    def linear_bias():
        wl = w.transpose(1, 2).reshape(nh * N, D)
        return F.relu(
            F.linear(x, wl, bias.reshape(nh * N)).view(B, T, nh, N), inplace=True
        )

    def v_einsum():
        return F.relu(torch.einsum("bhtd,hdn->bthn", y, w), inplace=True)

    def v_matmul_contig():
        return F.relu(
            torch.matmul(y, w).permute(0, 2, 1, 3).contiguous(), inplace=True
        )

    cfg = bdh.BDHConfig(
        n_layer=4,
        n_embd=128,
        n_head=4,
        mlp_internal_dim_multiplier=8,
        dropout=0.0,
        vocab_size=256,
    )
    torch.manual_seed(0)
    m = bdh.BDH(cfg).to(device).eval()
    idx = torch.randint(0, cfg.vocab_size, (4, 64), device=device)

    def fwd():
        with torch.no_grad():
            m(idx)

    rows = [
        ("encoder einsum (default)", einsum_enc),
        ("encoder F.linear + weight copy", linear_enc),
        ("encoder einsum + add_ bias", einsum_bias),
        ("encoder F.linear fused bias", linear_bias),
        ("encoder_v einsum", v_einsum),
        ("encoder_v matmul+contig", v_matmul_contig),
        ("full forward (landed path)", fwd),
    ]
    print(
        f"device={device} torch={torch.__version__} cuda={torch.cuda.is_available()} "
        f"OMP={os.environ.get('OMP_NUM_THREADS', '?')}"
    )
    print(f"micro B={B} T={T} D={D} nh={nh} N={N}")
    for name, fn in rows:
        ms = timed(fn) * 1000
        print(f"  {name:36s} {ms:8.3f} ms median")

    assert torch.equal(einsum_enc(), linear_enc())
    assert torch.equal(einsum_bias(), linear_bias())
    assert torch.equal(v_einsum(), v_matmul_contig())
    print("bit-identical: enc / enc+bias / encoder_v  OK")


if __name__ == "__main__":
    main()
