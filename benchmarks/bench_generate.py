#!/usr/bin/env python3
"""Microbench: BDH.generate (CacheManager) — attn impls + long-S AUTO A/B.

Modes
-----
* ``impls`` (default): end-to-end autoregressive ``generate`` under
  **eager | blocked | triton | cuda** when each backend is available.
  Soft-skips / labels fallbacks that cannot run on this device.
* ``auto-ab``: validate ``#55``/``#56`` wins **outside** score×V microbench —
  prompt lengths ``S ∈ {256, 1024, 2048}`` (configurable) with
  ``BDH_ATTN_AUTO=0`` vs ``1`` (default threshold 512). Cold/prefill stays
  ``BDH_ATTN_IMPL=eager``; only T=1 decode switches when ``past_len > thr``.

Honest CPU numbers: this sandbox is often ``cuda=False``. Report wall medians
and tokens-match flags — **do not** claim GPU / kernel wins from CPU medians.
Default ``BDH_ATTN_IMPL`` / ``BDH_ATTN_AUTO`` remain eager / off.

Usage::

    python benchmarks/bench_generate.py
    python benchmarks/bench_generate.py --prompt 32 --new 64 --warmup 2 --iters 5
    python benchmarks/bench_generate.py --mode auto-ab
    python benchmarks/bench_generate.py --mode auto-ab --prompts 256,1024,2048 \\
        --new 8 --warmup 1 --iters 3
    python benchmarks/bench_generate.py --device cuda   # A100/H100

Private ``katulevskiy/bdh-gpu-opt`` only — never pathwaycom.
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import bdh  # noqa: E402
from kernels.attention_dispatch import (  # noqa: E402
    DEFAULT_ATTN_AUTO_THRESHOLD,
    attn_auto_enabled,
    attn_auto_threshold,
    backend_info,
    resolve_attn_impl,
    resolve_decode_impl,
)


IMPLS = ("eager", "blocked", "triton", "cuda")
DEFAULT_AUTO_PROMPTS = (256, 1024, 2048)


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@contextmanager
def _attn_impl(name: str):
    """Temporarily set BDH_ATTN_IMPL; restore prior value on exit."""
    prev = os.environ.get("BDH_ATTN_IMPL")
    os.environ["BDH_ATTN_IMPL"] = name
    try:
        yield
    finally:
        if prev is None:
            os.environ.pop("BDH_ATTN_IMPL", None)
        else:
            os.environ["BDH_ATTN_IMPL"] = prev


@contextmanager
def _attn_auto(enabled: bool, threshold: int | None = None):
    """Temporarily set BDH_ATTN_AUTO (+ optional THRESHOLD); restore on exit.

    Env-string caches in ``attention_dispatch`` re-resolve when the raw env
    string changes, so toggling here is enough for A/B without process restart.
    """
    prev_auto = os.environ.get("BDH_ATTN_AUTO")
    prev_thr = os.environ.get("BDH_ATTN_AUTO_THRESHOLD")
    os.environ["BDH_ATTN_AUTO"] = "1" if enabled else "0"
    if threshold is not None:
        os.environ["BDH_ATTN_AUTO_THRESHOLD"] = str(int(threshold))
    try:
        yield
    finally:
        if prev_auto is None:
            os.environ.pop("BDH_ATTN_AUTO", None)
        else:
            os.environ["BDH_ATTN_AUTO"] = prev_auto
        if threshold is not None:
            if prev_thr is None:
                os.environ.pop("BDH_ATTN_AUTO_THRESHOLD", None)
            else:
                os.environ["BDH_ATTN_AUTO_THRESHOLD"] = prev_thr


def timed(fn, *, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    _sync()
    xs: list[float] = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        _sync()
        xs.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(xs)


def count_torch_cat(fn) -> int:
    cats = {"n": 0}
    orig = torch.cat

    def hooked(*a, **k):
        cats["n"] += 1
        return orig(*a, **k)

    torch.cat = hooked  # type: ignore[assignment]
    try:
        fn()
    finally:
        torch.cat = orig  # type: ignore[assignment]
    return cats["n"]


def _should_run(impl: str, device: torch.device) -> tuple[bool, str]:
    """Return (run?, reason). Skip only when the named backend cannot execute."""
    if impl == "eager":
        return True, "reference"
    if impl == "blocked":
        return True, "online fused tiles (CPU/CUDA)"
    if impl == "triton":
        info_probe = None
        with _attn_impl("triton"):
            info_probe = backend_info()
        eff = info_probe["effective"]
        if device.type != "cuda" or not info_probe["has_triton"]:
            return True, f"fallback effective={eff} (no CUDA Triton kernel)"
        return True, f"effective={eff}"
    if impl == "cuda":
        with _attn_impl("cuda"):
            info_probe = backend_info()
        eff = info_probe["effective"]
        if not info_probe["has_cuda_ext"]:
            return True, f"effective={eff} (pure-PyTorch ref; no native ext)"
        return True, f"effective={eff}"
    return False, f"unknown impl {impl!r}"


def _cfg(args) -> bdh.BDHConfig:
    return bdh.BDHConfig(
        n_layer=args.layers,
        n_embd=args.d,
        n_head=args.heads,
        mlp_internal_dim_multiplier=args.mlp_mult,
        dropout=0.0,
    )


def _resolve_device(args) -> torch.device | None:
    if args.device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("SKIP: --device cuda but torch.cuda.is_available() is False")
        return None
    return device


def _print_header(device: torch.device, cfg: bdh.BDHConfig, args, extra: str) -> None:
    props = torch.cuda.get_device_properties(0) if device.type == "cuda" else None
    gpu_name = props.name if props else "cpu"
    print(
        f"device={device} gpu={gpu_name!r} torch={torch.__version__} "
        f"cuda={torch.cuda.is_available()}"
    )
    print(
        f"cfg layers={cfg.n_layer} d={cfg.n_embd} nh={cfg.n_head} "
        f"B={args.batch} {extra} temp={args.temperature}"
    )
    print(f"warmup={args.warmup} iters={args.iters}")


def run_impls(args, device: torch.device) -> int:
    requested = [s.strip().lower() for s in args.impls.split(",") if s.strip()]
    for name in requested:
        if name not in IMPLS:
            print(f"ERROR: unknown impl {name!r}; choose from {IMPLS}")
            return 2

    cfg = _cfg(args)
    torch.manual_seed(0)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(0)
    model = bdh.BDH(cfg).to(device).eval()
    prompt = torch.randint(
        0, cfg.vocab_size, (args.batch, args.prompt), device=device
    )

    _print_header(
        device,
        cfg,
        args,
        f"prompt={args.prompt} new={args.new}",
    )
    print("--- BDH.generate + CacheManager across BDH_ATTN_IMPL ---")

    results: dict[str, dict] = {}
    ref_tokens: torch.Tensor | None = None

    for impl in requested:
        ok, reason = _should_run(impl, device)
        if not ok:
            print(f"  {impl:7s} SKIP  ({reason})")
            continue

        with _attn_impl(impl):
            info = backend_info()
            resolve_attn_impl(impl)

            def one_generate(seed: int = 0) -> torch.Tensor:
                torch.manual_seed(seed)
                if device.type == "cuda":
                    torch.cuda.manual_seed_all(seed)
                with torch.no_grad():
                    return model.generate(
                        prompt.clone(),
                        max_new_tokens=args.new,
                        temperature=args.temperature,
                    )

            out = one_generate(seed=0)
            if impl == "eager":
                ref_tokens = out.detach().clone()
                match = True
            else:
                assert ref_tokens is not None
                match = bool(torch.equal(out, ref_tokens))

            med = timed(
                lambda: one_generate(seed=1),
                warmup=args.warmup,
                iters=args.iters,
            )
            cats = count_torch_cat(lambda: one_generate(seed=2))

        tok_s = (args.batch * args.new) / (med / 1000.0) if med > 0 else float("inf")
        results[impl] = {
            "median_ms": med,
            "tokens_match_eager": match,
            "aten_cat": cats,
            "effective": info["effective"],
            "reason": reason,
            "tok_s": tok_s,
        }
        match_s = "yes" if match else "NO"
        print(
            f"  {impl:7s} median={med:8.2f} ms  "
            f"~{tok_s:7.1f} tok/s  "
            f"match_eager={match_s:3s}  "
            f"aten::cat={cats}  "
            f"effective={info['effective']}  ({reason})"
        )

    if "eager" in results and len(results) > 1:
        t0 = results["eager"]["median_ms"]
        print("--- vs eager ---")
        for impl, r in results.items():
            if impl == "eager":
                continue
            ratio = (t0 / r["median_ms"]) if r["median_ms"] > 0 else float("inf")
            print(
                f"  {impl:7s} {ratio:.2f}× vs eager  "
                f"(CPU-honest: not a GPU claim unless device=cuda)"
            )

    print(
        "NOTE: generate uses CacheManager (packed KR/V, cat-free). "
        "Absolute ms on CPU are honest wall times only — re-run on A100/H100 "
        "before claiming Triton/CUDA wins. Default BDH_ATTN_IMPL stays eager."
    )
    if device.type != "cuda":
        print(
            "NOTE: cuda=False on this box — triton → blocked fallback; "
            "cuda → pure-PyTorch ref. Use --device cuda on a GPU box."
        )
    return 0


def run_auto_ab(args, device: torch.device) -> int:
    """E2E generate A/B: BDH_ATTN_AUTO=0 vs 1 at long past lengths (#55/#56)."""
    prompts = [int(x) for x in args.prompts.split(",") if x.strip()]
    if not prompts:
        print("ERROR: --prompts must list at least one positive length")
        return 2
    if any(p <= 0 for p in prompts):
        print(f"ERROR: prompt lengths must be > 0, got {prompts}")
        return 2

    thr = int(args.auto_threshold)
    cfg = _cfg(args)
    torch.manual_seed(0)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(0)
    model = bdh.BDH(cfg).to(device).eval()

    _print_header(
        device,
        cfg,
        args,
        f"prompts={prompts} new={args.new} AUTO_THRESHOLD={thr}",
    )
    print(
        "--- BDH.generate long-S AUTO A/B "
        f"(IMPL=eager; AUTO 0 vs 1; thr={thr}) ---"
    )
    print(
        "Semantics: cold/prefill always eager; T=1 decode → blocked "
        f"(or triton on CUDA) only when AUTO=1 and past_len > {thr}."
    )

    # Force eager IMPL for the whole A/B (AUTO only applies when base=eager).
    rows: list[dict] = []
    with _attn_impl("eager"):
        for S in prompts:
            prompt = torch.randint(
                0, cfg.vocab_size, (args.batch, S), device=device
            )
            # past_len at first decode step == S; report whether AUTO fires.
            fires = S > thr
            # Also note mid-generate crossover when S <= thr < S+new-1
            crossover = (not fires) and (S + args.new - 1 > thr)

            cell: dict = {"prompt": S, "fires_at_start": fires, "crossover": crossover}

            ref_tokens: torch.Tensor | None = None
            for auto_on in (False, True):
                label = "AUTO=1" if auto_on else "AUTO=0"
                with _attn_auto(auto_on, threshold=thr):
                    assert attn_auto_enabled() is auto_on
                    assert attn_auto_threshold() == thr
                    # Sanity: resolve_decode_impl at past_len=S
                    dec = resolve_decode_impl(S)
                    if auto_on and fires:
                        # CUDA+Triton → triton; else #55 blocked (CPU / no Triton).
                        ok_dec = dec in ("blocked", "triton")
                        expected = dec  # whatever resolve picked is the contract
                    else:
                        ok_dec = dec == "eager"
                        expected = "eager"

                    def one_generate(seed: int = 0) -> torch.Tensor:
                        torch.manual_seed(seed)
                        if device.type == "cuda":
                            torch.cuda.manual_seed_all(seed)
                        with torch.no_grad():
                            return model.generate(
                                prompt.clone(),
                                max_new_tokens=args.new,
                                temperature=args.temperature,
                            )

                    out = one_generate(seed=0)
                    if not auto_on:
                        ref_tokens = out.detach().clone()
                        match = True
                    else:
                        assert ref_tokens is not None
                        match = bool(torch.equal(out, ref_tokens))

                    med = timed(
                        lambda: one_generate(seed=1),
                        warmup=args.warmup,
                        iters=args.iters,
                    )
                    cats = count_torch_cat(lambda: one_generate(seed=2))

                key = "auto1" if auto_on else "auto0"
                cell[key] = {
                    "median_ms": med,
                    "match": match,
                    "aten_cat": cats,
                    "decode_at_S": dec,
                    "decode_ok": ok_dec,
                    "expected_decode": expected,
                }
                tok_s = (
                    (args.batch * args.new) / (med / 1000.0) if med > 0 else float("inf")
                )
                fire_note = (
                    f"decode@{S}={dec}"
                    + (f" (expect {expected})" if not ok_dec else "")
                )
                if crossover and auto_on:
                    fire_note += f"; mid-gen may cross thr={thr}"
                match_s = "yes" if match else "NO"
                print(
                    f"  S={S:<5d} {label}  median={med:8.2f} ms  "
                    f"~{tok_s:7.1f} tok/s  match_AUTO0={match_s:3s}  "
                    f"aten::cat={cats}  {fire_note}"
                )

            a0 = cell["auto0"]["median_ms"]
            a1 = cell["auto1"]["median_ms"]
            spd = (a0 / a1) if a1 > 0 else float("inf")
            cell["speedup"] = spd
            rows.append(cell)
            print(
                f"  S={S:<5d} AUTO1/AUTO0 = {spd:.2f}×  "
                f"(>1 means AUTO faster; CPU-honest, not a GPU claim)"
            )

    print("--- summary table ---")
    print(
        f"{'prompt':>6}  {'new':>4}  {'AUTO=0 ms':>10}  {'AUTO=1 ms':>10}  "
        f"{'spd':>6}  {'match':>5}  {'cats':>4}  {'decode@S AUTO1':>14}  note"
    )
    for r in rows:
        a0 = r["auto0"]
        a1 = r["auto1"]
        match = "yes" if a1["match"] else "NO"
        cats = a1["aten_cat"]
        note = []
        if r["fires_at_start"]:
            note.append(f"AUTO fires (S>{thr})")
        elif r["crossover"]:
            note.append(f"crossover during new tokens (thr={thr})")
        else:
            note.append(f"S+new-1≤{thr}: AUTO never fires")
        print(
            f"{r['prompt']:6d}  {args.new:4d}  {a0['median_ms']:10.2f}  "
            f"{a1['median_ms']:10.2f}  {r['speedup']:5.2f}×  {match:>5}  "
            f"{cats:4d}  {a1['decode_at_S']:>14}  {'; '.join(note)}"
        )

    print(
        "NOTE: AUTO A/B keeps cold/prefill on eager IMPL — unlike IMPL=blocked "
        "A/B (#55), which also tiles prefill. Long-S win here is decode-only "
        f"after past_len > {thr}. Default BDH_ATTN_AUTO stays off."
    )
    if device.type != "cuda":
        print(
            "NOTE: cuda=False on this box — AUTO→blocked (#55 CPU path). "
            "On CUDA+Triton, AUTO prefers triton decode (triton-decode-v3). "
            "Re-tune threshold on A100/H100 before claiming GPU wins."
        )
    # Fail the process if any parity / cat / decode-resolve check broke.
    bad = [
        r
        for r in rows
        if (not r["auto1"]["match"])
        or r["auto0"]["aten_cat"] != 0
        or r["auto1"]["aten_cat"] != 0
        or (not r["auto0"]["decode_ok"])
        or (not r["auto1"]["decode_ok"])
    ]
    if bad:
        print(f"ERROR: {len(bad)} prompt(s) failed match/cat/decode checks")
        return 1
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--mode",
        choices=("impls", "auto-ab"),
        default="impls",
        help="impls = BDH_ATTN_IMPL sweep (default); auto-ab = long-S AUTO 0/1",
    )
    p.add_argument("--prompt", type=int, default=16, help="prompt length (impls mode)")
    p.add_argument(
        "--prompts",
        default=",".join(str(x) for x in DEFAULT_AUTO_PROMPTS),
        help="comma prompt lengths for --mode auto-ab (default 256,1024,2048)",
    )
    p.add_argument("--new", type=int, default=32, help="max_new_tokens")
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--d", type=int, default=128)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--mlp-mult", type=int, default=32)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--iters", type=int, default=5)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument(
        "--auto-threshold",
        type=int,
        default=DEFAULT_ATTN_AUTO_THRESHOLD,
        help=f"BDH_ATTN_AUTO_THRESHOLD for auto-ab (default {DEFAULT_ATTN_AUTO_THRESHOLD})",
    )
    p.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="auto = cuda if available else cpu",
    )
    p.add_argument(
        "--impls",
        default="eager,blocked,triton,cuda",
        help="comma list from eager|blocked|triton|cuda (impls mode)",
    )
    args = p.parse_args()

    device = _resolve_device(args)
    if device is None:
        return 0

    # auto-ab defaults: fewer new tokens at long S (CPU feasible; matches #55
    # generate table spirit). Caller can still override --new.
    if args.mode == "auto-ab" and args.new == 32 and "--new" not in sys.argv:
        # Heuristic default when user did not pass --new: keep wall time
        # feasible on CPU for S up to 2048 (cold T×T still eager).
        args.new = 8
        print(f"(auto-ab default) --new={args.new} for CPU-feasible long-S wall")

    if args.mode == "auto-ab":
        return run_auto_ab(args, device)
    return run_impls(args, device)


if __name__ == "__main__":
    raise SystemExit(main())
