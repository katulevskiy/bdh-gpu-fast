"""Microbench: dense vs sparse / masked ReLU GEMMs (experimental, CPU-friendly).

Reports activation density on a tiny BDH forward (random init ≈ 50%) and
synthetic paper-like ~5% density timings. Sparse wins are density-dependent;
on this CPU box expect dense to win at 50% and sparse COO/CSR to help only at
very low density (conversion overhead dominates small shapes).
"""

from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import bdh
import bdh_sparse as sp


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def timed(fn, warmup=2, reps=8):
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


def measure_bdh_densities(cfg: bdh.BDHConfig, B: int = 2, T: int = 64):
    torch.manual_seed(0)
    m = bdh.BDH(cfg).eval()
    idx = torch.randint(0, cfg.vocab_size, (B, T))
    D = cfg.n_embd
    nh = cfg.n_head
    N = D * cfg.mlp_internal_dim_multiplier // nh
    x = m.embed(idx).unsqueeze(1)
    x = m.ln(x)
    stats = []
    with torch.no_grad():
        for level in range(cfg.n_layer):
            x_latent = x @ m.encoder
            x_s = F.relu(x_latent)
            yKV, _, _ = m.attn(Q=x_s, K=x_s, V=x)
            yKV = m.ln(yKV)
            y_s = F.relu(yKV @ m.encoder_v)
            xy = x_s * y_s
            stats.append(
                {
                    "layer": level,
                    "x": sp.relu_density(x_s),
                    "y": sp.relu_density(y_s),
                    "xy": sp.relu_density(xy),
                }
            )
            xy_flat = xy.permute(0, 2, 1, 3).contiguous().view(B, 1, T, N * nh)
            yMLP = xy_flat @ m.decoder
            x = m.ln(x + m.ln(yMLP))
    return stats


def bench_matmul(M: int, K: int, N: int, density: float, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    raw = torch.randn(M, K, generator=g).abs()
    act = sp.force_sparsity(
        raw, density=density, generator=torch.Generator().manual_seed(seed + 1)
    )
    W = torch.randn(K, N, generator=torch.Generator().manual_seed(seed + 2))
    measured = sp.relu_density(act)

    def dense():
        return act @ W

    def coo():
        return sp.sparse_relu_matmul(act, W, layout="coo")

    def csr():
        return sp.sparse_relu_matmul(act, W, layout="csr")

    def row_m():
        return sp.masked_densify_matmul(act, W)

    def col_m():
        return sp.masked_densify_matmul_gather(act, W)

    # Correctness smoke
    ref = dense()
    # Large K: sparse.mm vs dense GEMM can differ at ~1e-4 abs (FP32 accum order).
    for name, fn in [("coo", coo), ("csr", csr), ("row", row_m), ("col", col_m)]:
        assert torch.allclose(fn(), ref, rtol=1e-3, atol=1e-4), name

    return {
        "density_target": density,
        "density_measured": measured,
        "dense_ms": timed(dense) * 1000,
        "coo_ms": timed(coo) * 1000,
        "csr_ms": timed(csr) * 1000,
        "row_ms": timed(row_m) * 1000,
        "col_ms": timed(col_m) * 1000,
    }


def main():
    device = torch.device("cpu")
    print(f"device={device} torch={torch.__version__} cuda={torch.cuda.is_available()}")
    print("--- BDH random-init ReLU densities (paper cites ~5% when trained) ---")
    cfg = bdh.BDHConfig(
        n_layer=4,
        n_embd=128,
        n_head=4,
        mlp_internal_dim_multiplier=32,
        dropout=0.0,
    )
    stats = measure_bdh_densities(cfg)
    for s in stats:
        print(
            f"  L{s['layer']}: x_sparse={s['x']:.4f}  "
            f"y_sparse={s['y']:.4f}  xy_sparse={s['xy']:.4f}"
        )
    mx = sum(s["x"] for s in stats) / len(stats)
    mxy = sum(s["xy"] for s in stats) / len(stats)
    print(f"  mean x_sparse={mx:.4f}  mean xy_sparse={mxy:.4f}")

    print("--- decoder-shaped matmul (M=B*T, K=nh*N, N=D) ---")
    # B=4, T=128, nh=4, N=1024 → K=4096, D=128  (heavy); use smaller for CPU
    M, K, N = 256, 1024, 128
    for density in (0.50, 0.25, 0.05):
        r = bench_matmul(M, K, N, density=density)
        print(
            f"  dens~{r['density_measured']:.3f} (target {r['density_target']}): "
            f"dense={r['dense_ms']:.2f}ms  coo={r['coo_ms']:.2f}ms  "
            f"csr={r['csr_ms']:.2f}ms  row={r['row_ms']:.2f}ms  "
            f"col={r['col_ms']:.2f}ms"
        )

    print(
        "Note: experimental module only; default bdh.py path unchanged. "
        "Sparse speedups need low density + GPU sparse kernels; "
        "CPU conversion overhead often dominates."
    )


if __name__ == "__main__":
    main()
