"""Microbench: train-step variants (fused AdamW, set_to_none, compile path).

Also used by opt/train-fuse / opt/compile-bench / opt/compile-blocked to
document CPU honesty.

CPU-honest: no CUDA on the default runner. Measures median step time for a
tiny BDH config.

BDH_COMPILE=0 vs 1 (opt/compile-bench):
  Always attempted unless BDH_BENCH_COMPILE=0. Uses train.maybe_compile so
  inductor / CXX / probe failures soft-skip (print + return) instead of
  crashing. Same weights, same AdamW path, same fixed batch — honest A/B.
  Absolute ms are CPU-only; do not claim GPU / CUDA-graph wins here.

COMPILE × ATTN_IMPL × AUTOGRAD matrix (opt/compile-blocked):
  Set BDH_BENCH_COMPILE_BLOCKED=1 (default on when this section is wanted;
  also runs if BDH_BENCH_COMPILE=1 and BDH_BENCH_COMPILE_BLOCKED unset → on).
  Tiny cfg, soft-skip per cell on compile/probe failure. CPU medians only.
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


def _is_dynamo_compiled(model: torch.nn.Module) -> bool:
    """True when torch.compile wrapped the module (OptimizedModule._orig_mod)."""
    return getattr(model, "_orig_mod", None) is not None


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


def bench_compile_vs_eager(cfg, device, fused_ok: bool) -> None:
    """Honest BDH_COMPILE=0 vs 1 train-step medians via maybe_compile.

    Soft-skips (prints reason, no exception) if inductor/CXX/probe unavailable.
    CPU box: report wall medians only — GPU / reduce-overhead is OPT_BACKLOG next.
    """
    print("--- BDH_COMPILE=0 vs 1 (honest train-step) ---")
    print(
        f"device={device} mode={os.environ.get('BDH_COMPILE_MODE', tr.COMPILE_MODE)} "
        f"probe={os.environ.get('BDH_COMPILE_PROBE', tr.COMPILE_PROBE)} "
        f"cuda={torch.cuda.is_available()}"
    )

    torch.manual_seed(0)
    m_eager = bdh.BDH(cfg).to(device)
    m_eager.train()
    x, y = _batch(device)
    assert not _is_dynamo_compiled(m_eager)

    # --- eager (BDH_COMPILE=0): plain module, no maybe_compile ---
    opt0 = torch.optim.AdamW(
        m_eager.parameters(),
        lr=tr.LEARNING_RATE,
        weight_decay=tr.WEIGHT_DECAY,
        fused=fused_ok,
    )

    def step0():
        return tr.train_step(m_eager, opt0, x, y)

    t0 = timed(step0, warmup=3, reps=12)
    print(f"BDH_COMPILE=0 (eager) train_step median: {t0*1000:.2f} ms")

    # --- compiled (BDH_COMPILE=1): maybe_compile soft-fallback ---
    torch.manual_seed(0)
    m1_src = bdh.BDH(cfg).to(device)
    m1_src.load_state_dict(m_eager.state_dict())
    m1_src.train()

    was = tr.USE_COMPILE
    tr.USE_COMPILE = True
    try:
        m1 = tr.maybe_compile(m1_src, example_x=x, example_y=y)
    except Exception as e:
        tr.USE_COMPILE = was
        print(
            f"skip compile bench: maybe_compile raised "
            f"{type(e).__name__}: {e}"
        )
        return
    tr.USE_COMPILE = was

    if not _is_dynamo_compiled(m1):
        print(
            "skip compile bench: torch.compile unavailable or probe fell back "
            "to eager (see maybe_compile log above). Soft-skip — not a hard fail."
        )
        return

    opt1 = torch.optim.AdamW(
        m1.parameters(),
        lr=tr.LEARNING_RATE,
        weight_decay=tr.WEIGHT_DECAY,
        fused=fused_ok,
    )

    def step1():
        return tr.train_step(m1, opt1, x, y)

    # Inductor first steps already probed inside maybe_compile; short extra warm.
    try:
        t1 = timed(step1, warmup=2, reps=10)
    except Exception as e:
        print(
            f"skip compile bench: compiled train_step failed "
            f"({type(e).__name__}: {e}). Soft-skip."
        )
        return

    ratio = t0 / t1 if t1 > 0 else float("inf")
    print(f"BDH_COMPILE=1 (compiled) train_step median: {t1*1000:.2f} ms")
    print(
        f"ratio eager/compiled: {ratio:.2f}x  "
        f"(>1 means compile faster on this device)"
    )
    if device.type != "cuda":
        print(
            "honest: CPU inductor medians only — no CUDA-graph / GPU claim. "
            "Next: GPU BDH_COMPILE=0 vs 1 (see OPT_BACKLOG)."
        )
    else:
        print(
            "GPU box: also try BDH_COMPILE_MODE=reduce-overhead for CUDA graphs "
            "(static B×T; see train_fast.py)."
        )


def bench_compile_blocked_matrix(cfg, device, fused_ok: bool) -> None:
    """Honest COMPILE × ATTN_IMPL × AUTOGRAD train-step matrix (CPU-first).

    Cells: COMPILE in {0,1} × IMPL in {eager, blocked} × AUTOGRAD in {0,1}.
    Soft-skips a cell (prints reason) if compile/probe unavailable or step
    raises — never hard-fails the harness. Restores env after each cell.
    Absolute ms are CPU-only; do not claim GPU wins.
    """
    print("--- COMPILE × ATTN_IMPL × AUTOGRAD train-step matrix ---")
    print(
        f"device={device} mode={os.environ.get('BDH_COMPILE_MODE', tr.COMPILE_MODE)} "
        f"probe={os.environ.get('BDH_COMPILE_PROBE', tr.COMPILE_PROBE)} "
        f"cuda={torch.cuda.is_available()} cfg=layers={cfg.n_layer} d={cfg.n_embd} "
        f"B=4 T=64 dropout={cfg.dropout}"
    )

    impls = ("eager", "blocked")
    compiles = (0, 1)
    autograds = (0, 1)

    saved = {
        "BDH_COMPILE": os.environ.get("BDH_COMPILE"),
        "BDH_ATTN_IMPL": os.environ.get("BDH_ATTN_IMPL"),
        "BDH_ATTN_AUTOGRAD": os.environ.get("BDH_ATTN_AUTOGRAD"),
    }
    was_use = tr.USE_COMPILE
    x, y = _batch(device)
    rows = []

    def _restore_env():
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        tr.USE_COMPILE = was_use

    try:
        for compile_on in compiles:
            for impl in impls:
                for autograd in autograds:
                    tag = f"COMPILE={compile_on} IMPL={impl} AUTOGRAD={autograd}"
                    os.environ["BDH_ATTN_IMPL"] = impl
                    if autograd:
                        os.environ["BDH_ATTN_AUTOGRAD"] = "1"
                    else:
                        os.environ.pop("BDH_ATTN_AUTOGRAD", None)

                    torch.manual_seed(0)
                    m = bdh.BDH(cfg).to(device)
                    m.train()

                    if compile_on:
                        os.environ["BDH_COMPILE"] = "1"
                        tr.USE_COMPILE = True
                        try:
                            m = tr.maybe_compile(m, example_x=x, example_y=y)
                        except Exception as e:
                            print(
                                f"{tag}: soft-skip maybe_compile "
                                f"{type(e).__name__}: {e}"
                            )
                            rows.append((tag, None, "soft-skip compile"))
                            continue
                        if not _is_dynamo_compiled(m):
                            print(
                                f"{tag}: soft-skip (fell back to eager — "
                                "inductor/probe unavailable)"
                            )
                            rows.append((tag, None, "soft-skip eager-fallback"))
                            continue
                    else:
                        os.environ["BDH_COMPILE"] = "0"
                        tr.USE_COMPILE = False

                    opt = torch.optim.AdamW(
                        m.parameters(),
                        lr=tr.LEARNING_RATE,
                        weight_decay=tr.WEIGHT_DECAY,
                        fused=fused_ok,
                    )

                    def step():
                        return tr.train_step(m, opt, x, y)

                    try:
                        # Shorter reps for matrix (8 cells); warm inductor once.
                        warm = 2 if compile_on else 2
                        reps = 8 if compile_on else 10
                        med = timed(step, warmup=warm, reps=reps)
                    except Exception as e:
                        print(
                            f"{tag}: soft-skip train_step "
                            f"{type(e).__name__}: {e}"
                        )
                        rows.append((tag, None, f"soft-skip step:{type(e).__name__}"))
                        continue

                    ms = med * 1000.0
                    print(f"{tag}: median {ms:.2f} ms")
                    rows.append((tag, ms, "ok"))
    finally:
        _restore_env()

    # Compact table for OPT_NOTES copy-paste
    print("--- matrix summary (median ms; soft-skip = —) ---")
    print(f"{'COMPILE':>8} {'IMPL':>8} {'AUTOGRAD':>8} {'median_ms':>12} {'status':>18}")
    for tag, ms, status in rows:
        # tag = "COMPILE=X IMPL=Y AUTOGRAD=Z"
        parts = dict(p.split("=", 1) for p in tag.split())
        med_s = f"{ms:.2f}" if ms is not None else "—"
        print(
            f"{parts['COMPILE']:>8} {parts['IMPL']:>8} {parts['AUTOGRAD']:>8} "
            f"{med_s:>12} {status:>18}"
        )
    if device.type != "cuda":
        print(
            "honest: CPU inductor medians only — no CUDA-graph / GPU claim. "
            "tril(diagonal=-1) preserved; defaults still COMPILE=0 / IMPL=eager / "
            "AUTOGRAD off."
        )



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

    # Default on: honest BDH_COMPILE=0 vs 1. Opt out with BDH_BENCH_COMPILE=0
    # (legacy fused/zero_grad-only runs). Soft-skip inside if inductor missing.
    if os.environ.get("BDH_BENCH_COMPILE", "1") in ("1", "true", "True"):
        bench_compile_vs_eager(cfg, device, fused_ok)
    else:
        print("skip compile bench (set BDH_BENCH_COMPILE=1 to enable; default is on)")

    # COMPILE × blocked × AUTOGRAD matrix (opt/compile-blocked). Default on;
    # opt out with BDH_BENCH_COMPILE_BLOCKED=0.
    if os.environ.get("BDH_BENCH_COMPILE_BLOCKED", "1") in ("1", "true", "True"):
        bench_compile_blocked_matrix(cfg, device, fused_ok)
    else:
        print(
            "skip compile×blocked matrix "
            "(set BDH_BENCH_COMPILE_BLOCKED=1 to enable; default is on)"
        )

    print(
        "Note: CUDA graphs / reduce-overhead need a GPU; see train_fast.py + OPT_BACKLOG."
    )


if __name__ == "__main__":
    main()
