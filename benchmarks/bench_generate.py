#!/usr/bin/env python3
"""Microbench: BDH.generate (CacheManager) across BDH_ATTN_IMPL backends.

Compares end-to-end autoregressive ``generate`` (packed KR/V CacheManager,
cat-free) under **eager | blocked | triton | cuda** when each backend is
available. Soft-skips backends that cannot run on this device (e.g. true
Triton kernel needs CUDA).

Honest CPU numbers: this sandbox is often ``cuda=False``. Report wall medians
and tokens-match-eager flags — **do not** claim GPU / kernel wins from CPU
medians. Default ``BDH_ATTN_IMPL`` remains eager until GPU data lands.

Usage::

    python benchmarks/bench_generate.py
    python benchmarks/bench_generate.py --prompt 32 --new 64 --warmup 2 --iters 5
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
from kernels.attention_dispatch import backend_info, resolve_attn_impl  # noqa: E402


IMPLS = ("eager", "blocked", "triton", "cuda")


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
        # Triton kernel needs CUDA + triton package; otherwise dispatch falls
        # back to blocked — still useful to time the effective path, but label it.
        info_probe = None
        with _attn_impl("triton"):
            info_probe = backend_info()
        eff = info_probe["effective"]
        if device.type != "cuda" or not info_probe["has_triton"]:
            # Still run: effective=blocked on CPU — honest labeling below.
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


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prompt", type=int, default=16, help="prompt length")
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
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="auto = cuda if available else cpu",
    )
    p.add_argument(
        "--impls",
        default="eager,blocked,triton,cuda",
        help="comma list from eager|blocked|triton|cuda",
    )
    args = p.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            print("SKIP: --device cuda but torch.cuda.is_available() is False")
            return 0

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

    props = torch.cuda.get_device_properties(0) if device.type == "cuda" else None
    gpu_name = props.name if props else "cpu"
    print(
        f"device={device} gpu={gpu_name!r} torch={torch.__version__} "
        f"cuda={torch.cuda.is_available()}"
    )
    print(
        f"cfg layers={cfg.n_layer} d={cfg.n_embd} nh={cfg.n_head} "
        f"B={args.batch} prompt={args.prompt} new={args.new} "
        f"temp={args.temperature}"
    )
    print(f"warmup={args.warmup} iters={args.iters}")
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
            # Resolve once so a bad env fails early.
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

            # Correctness vs eager (same seed → same multinomial draws if logits match)
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


if __name__ == "__main__":
    raise SystemExit(main())
