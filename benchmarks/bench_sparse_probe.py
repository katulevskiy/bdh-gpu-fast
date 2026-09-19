"""P2 sparse probe: ReLU density on short train runs + CPU sparse-vs-dense crossover.

DEFAULT OFF — does not wire sparse into BDH.forward. Uses bdh_sparse helpers only.
Run with ``BDH_SPARSE_PROBE=1``; an unset/false gate exits before training or
benchmark work. This is a measurement harness, not a runtime sparse switch.

Reports:
  1. x / y / xy post-ReLU density at init and during a short CPU train
  2. decoder-shaped dense vs COO/CSR/row/col timings across densities
  3. whether sparse ever beats dense on this CPU box (honest)

Usage:
  .venv/bin/python benchmarks/bench_sparse_probe.py
  .venv/bin/python benchmarks/bench_sparse_probe.py --steps 100 --log-every 25

Exit status:
  0  probe completed, or the default-off gate intentionally did no work
  2  an enforced density guardrail observed a failing sample
  3  guardrail enforcement was requested without density samples
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Keep compile / dataloader off for a cheap CPU probe.
os.environ.setdefault("BDH_COMPILE", "0")
os.environ.setdefault("BDH_DATALOADER", "0")

import bdh
import bdh_sparse as sp
import train as tr

EXIT_OK = 0
EXIT_DENSITY_GUARDRAIL = 2
EXIT_GUARDRAIL_NO_SAMPLES = 3


def density_guardrail(
    history: list[dict],
    *,
    min_final_x: float,
    min_final_xy: float,
) -> tuple[bool, str]:
    """Check that the re-smoke still observes non-paper density on CPU.

    The thresholds are deliberately below the prior ~27% x / ~12% xy result,
    so this catches a surprising density collapse without pretending that a
    short CPU run reproduces the paper's trained ~5% claim.
    """
    if not history:
        return False, "no density samples collected"
    final = history[-1]
    x = float(final["x"])
    xy = float(final["xy"])
    ok = x >= min_final_x and xy >= min_final_xy
    detail = (
        f"final_x={x:.4f} final_xy={xy:.4f} "
        f"required_x>={min_final_x:.4f} required_xy>={min_final_xy:.4f}"
    )
    return ok, detail


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def timed(fn, warmup: int = 2, reps: int = 8) -> float:
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


def measure_layer_densities(
    model: bdh.BDH, idx: torch.Tensor
) -> list[dict[str, float | int]]:
    """Mirror BDH.forward ReLU sites without mutating training state.

    Matches tip layout: ``x`` is ``(B,T,D)``, encoder ``(nh,D,N)`` → ``(B,T,nh,N)``,
    ``xy = x_bthn * y_bthn`` before decoder view. Density is post-ReLU support.
    """
    cfg = model.config
    B, T = idx.shape
    nh = cfg.n_head
    N = cfg.n_embd * cfg.mlp_internal_dim_multiplier // nh
    was_training = model.training
    model.eval()
    stats: list[dict[str, float | int]] = []
    with torch.no_grad():
        x = model._ln(model.embed(idx))  # (B, T, D)
        cos_sin = model.attn.rope_cos_sin(T, 0, x.device)
        for _level in range(cfg.n_layer):
            x_bthn = model._encoder_relu(x, model.encoder, model.encoder_bias)
            x_sparse = x_bthn.permute(0, 2, 1, 3)  # (B, nh, T, N)
            v_tok = x.unsqueeze(1)
            yKV, _, _ = model.attn(
                Q=x_sparse, K=x_sparse, V=v_tok, rope_start=0, cos_sin=cos_sin
            )
            yKV = model._ln(yKV)
            y_bthn = model._encoder_v_relu(yKV, model.encoder_v, model.encoder_v_bias)
            xy_bthn = x_bthn * y_bthn
            stats.append(
                {
                    "layer": _level,
                    "x": sp.relu_density(x_bthn),
                    "y": sp.relu_density(y_bthn),
                    "xy": sp.relu_density(xy_bthn),
                }
            )
            yMLP = model._linear(
                xy_bthn.view(B, T, nh * N), model.decoder, model.decoder_bias
            )
            x = model._residual_ln(x, yMLP)
    if was_training:
        model.train()
    return stats


def mean_densities(stats: list[dict[str, float | int]]) -> dict[str, float]:
    n = max(len(stats), 1)
    return {
        "x": sum(float(s["x"]) for s in stats) / n,
        "y": sum(float(s["y"]) for s in stats) / n,
        "xy": sum(float(s["xy"]) for s in stats) / n,
    }


def ensure_corpus() -> None:
    if not (ROOT / "input.txt").exists():
        # Prefer sibling worktree corpus to avoid network.
        sibling = Path("/workspace/bdh-gpu-opt/input.txt")
        if sibling.exists():
            (ROOT / "input.txt").write_bytes(sibling.read_bytes())
        else:
            tr.fetch_data()


def short_train_density(
    *,
    steps: int,
    log_every: int,
    B: int,
    T: int,
    seed: int,
) -> list[dict]:
    """Train a tiny BDH for ``steps`` and snapshot ReLU densities periodically."""
    torch.manual_seed(seed)
    ensure_corpus()
    tr._train_data = None
    tr._val_data = None
    tr._offsets = None
    tr._load_splits()

    cfg = bdh.BDHConfig(
        n_layer=2,
        n_embd=64,
        n_head=2,
        mlp_internal_dim_multiplier=16,
        dropout=0.0,
    )
    # train.get_batch uses BLOCK_SIZE; shrink for the probe.
    old_block, old_batch = tr.BLOCK_SIZE, tr.BATCH_SIZE
    tr.BLOCK_SIZE = T
    tr.BATCH_SIZE = B
    try:
        device = tr.device
        model = bdh.BDH(cfg).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.1)
        model.train()
        history: list[dict] = []

        def snapshot(step: int, loss_val: float | None) -> None:
            x_probe, _ = tr.get_batch("train")
            # Clamp to T if get_batch used BLOCK_SIZE
            x_probe = x_probe[:, :T]
            dens = mean_densities(measure_layer_densities(model, x_probe))
            row = {
                "step": step,
                "loss": loss_val,
                **dens,
            }
            history.append(row)
            loss_s = f"{loss_val:.4f}" if loss_val is not None else "n/a"
            print(
                f"  step={step:4d}  loss={loss_s}  "
                f"mean x={dens['x']:.4f}  y={dens['y']:.4f}  xy={dens['xy']:.4f}"
            )

        snapshot(0, None)
        for step in range(1, steps + 1):
            x, y = tr.get_batch("train")
            x, y = x[:, :T], y[:, :T]
            _, loss = model(x, y)
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            if step % log_every == 0 or step == steps:
                snapshot(step, float(loss.detach().item()))
        return history
    finally:
        tr.BLOCK_SIZE = old_block
        tr.BATCH_SIZE = old_batch


def bench_matmul(M: int, K: int, N: int, density: float, seed: int = 0) -> dict:
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

    ref = dense()
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


def crossover_report(rows: list[dict]) -> list[str]:
    """Return human lines describing when each sparse path beat dense (if ever)."""
    lines = []
    for path in ("coo", "csr", "row", "col"):
        winners = [
            r
            for r in rows
            if r[f"{path}_ms"] < r["dense_ms"]
        ]
        if not winners:
            lines.append(
                f"  {path}: never beat dense on this CPU sweep "
                f"(densities {[r['density_measured'] for r in rows]})"
            )
        else:
            best = min(winners, key=lambda r: r[f"{path}_ms"] / r["dense_ms"])
            dens = ", ".join(f"{r['density_measured']:.3f}" for r in winners)
            speed = best["dense_ms"] / best[f"{path}_ms"]
            lines.append(
                f"  {path}: beat dense at density in {{{dens}}} "
                f"(best {speed:.2f}× @ dens~{best['density_measured']:.3f})"
            )
    return lines


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--block", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--skip-crossover", action="store_true")
    ap.add_argument(
        "--density-only",
        action="store_true",
        help="run only the short-train density re-smoke (implies --skip-crossover)",
    )
    ap.add_argument(
        "--enforce-density-guardrail",
        action="store_true",
        help="fail if final density falls below the conservative re-smoke floor",
    )
    ap.add_argument("--min-final-x", type=float, default=0.20)
    ap.add_argument("--min-final-xy", type=float, default=0.08)
    args = ap.parse_args(argv)

    if not sp.sparse_probe_enabled():
        print(
            "SPARSE PROBE DISABLED (default): no training or benchmark work. "
            "Set BDH_SPARSE_PROBE=1 for the optional probe. "
            f"exit_code={EXIT_OK} reason=probe_disabled"
        )
        return EXIT_OK

    device = torch.device("cpu")
    print(
        f"device={device} torch={torch.__version__} "
        f"cuda={torch.cuda.is_available()}"
    )
    print("DEFAULT OFF: sparse not wired into BDH.forward")

    history: list[dict] = []
    if not args.skip_train:
        print(
            f"--- short train density "
            f"(steps={args.steps}, B={args.batch}, T={args.block}) ---"
        )
        history = short_train_density(
            steps=args.steps,
            log_every=args.log_every,
            B=args.batch,
            T=args.block,
            seed=args.seed,
        )
        init = history[0]
        final = history[-1]
        print(
            f"  Δ density init→final: "
            f"x {init['x']:.4f}→{final['x']:.4f}  "
            f"y {init['y']:.4f}→{final['y']:.4f}  "
            f"xy {init['xy']:.4f}→{final['xy']:.4f}"
        )
        print(
            "  note: paper cites ~5% after full training; short CPU probes "
            "show x/xy density falling but still >>5% at this scale."
        )

    if history:
        ok, detail = density_guardrail(
            history,
            min_final_x=args.min_final_x,
            min_final_xy=args.min_final_xy,
        )
        print(f"density_guardrail={'pass' if ok else 'fail'} {detail}")
        if args.enforce_density_guardrail and not ok:
            print("density re-smoke guardrail failed", file=sys.stderr)
            print(
                f"exit_code={EXIT_DENSITY_GUARDRAIL} reason=guardrail_failed",
                file=sys.stderr,
            )
            return EXIT_DENSITY_GUARDRAIL
    elif args.enforce_density_guardrail:
        print(
            "density_guardrail=unavailable no density samples collected; "
            "remove --skip-train or disable enforcement",
            file=sys.stderr,
        )
        print(
            f"exit_code={EXIT_GUARDRAIL_NO_SAMPLES} "
            "reason=guardrail_no_samples",
            file=sys.stderr,
        )
        return EXIT_GUARDRAIL_NO_SAMPLES

    rows: list[dict] = []
    if not args.skip_crossover and not args.density_only:
        print("--- CPU sparse vs dense crossover (decoder-shaped GEMM) ---")
        # B=4 T=128 → M=512; nh*N with tiny cfg is small — use mid shapes that
        # still finish quickly on CPU while stressing conversion overhead.
        # Decoder-ish GEMMs. Large shape included to hunt for a CPU crossover;
        # on this box sparse still loses (conversion / gather dominates).
        shapes = [
            (256, 1024, 128),
            (512, 2048, 128),
            (1024, 4096, 256),
        ]
        densities = (0.50, 0.25, 0.10, 0.05, 0.02, 0.01, 0.005, 0.001)
        for M, K, N in shapes:
            print(f"  shape M={M} K={K} N={N}")
            for density in densities:
                r = bench_matmul(M, K, N, density=density)
                r["M"], r["K"], r["N"] = M, K, N
                rows.append(r)
                best_sp = min(r["coo_ms"], r["csr_ms"], r["row_ms"], r["col_ms"])
                flag = "WIN" if best_sp < r["dense_ms"] else "lose"
                print(
                    f"    dens~{r['density_measured']:.3f}: "
                    f"dense={r['dense_ms']:.2f}ms  coo={r['coo_ms']:.2f}ms  "
                    f"csr={r['csr_ms']:.2f}ms  row={r['row_ms']:.2f}ms  "
                    f"col={r['col_ms']:.2f}ms  [{flag}]"
                )

        print("--- crossover summary (this CPU box) ---")
        for line in crossover_report(rows):
            print(line)
        any_win = any(
            min(r["coo_ms"], r["csr_ms"], r["row_ms"], r["col_ms"]) < r["dense_ms"]
            for r in rows
        )
        if not any_win:
            print(
                "  VERDICT: sparse never beat dense on CPU in this sweep. "
                "Keep default OFF; revisit on GPU / trained ~5% density."
            )
        else:
            print(
                "  VERDICT: sparse beat dense only at the densities listed above; "
                "still DEFAULT OFF in bdh.py."
            )

    # Machine-readable one-liner for OPT_NOTES paste.
    print("--- paste summary ---")
    if history:
        init, final = history[0], history[-1]
        print(
            f"train_density steps={args.steps} "
            f"init_x={init['x']:.4f} init_xy={init['xy']:.4f} "
            f"final_x={final['x']:.4f} final_xy={final['xy']:.4f}"
        )
    if rows:
        wins = [
            f"{r['density_measured']:.3f}@{r['M']}x{r['K']}"
            for r in rows
            if min(r["coo_ms"], r["csr_ms"], r["row_ms"], r["col_ms"]) < r["dense_ms"]
        ]
        print(f"cpu_sparse_wins={wins if wins else 'none'}")

    print(f"exit_code={EXIT_OK} reason=probe_complete")
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
