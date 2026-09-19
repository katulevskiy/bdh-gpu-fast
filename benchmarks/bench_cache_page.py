#!/usr/bin/env python3
"""Microbench: CacheManager geometric vs linear page growth (long generate).

Measures ``n_grows`` and ``bytes_copied_on_grow`` for geometric (tip
``_next_capacity``: ≥2×, page-aligned) vs linear ``capacity + page_size``
across page sizes on a long tokenwise fill to ``max_seq``.

Also smoke-checks opt-in ``generate(..., cache_page_size=)``: ``aten::cat=0``,
tokens match fixed-capacity generate, and reports grow stats.

Honest CPU numbers only — realloc/copy accounting, **not** GPU GEMM wall.
Defaults unchanged (``page_size=None`` / ``cache_page_size=None`` → full
prealloc, zero grows). Private ``katulevskiy/bdh-gpu-opt`` only — never
pathwaycom.

Usage::

    python benchmarks/bench_cache_page.py
    python benchmarks/bench_cache_page.py --max-seq 2048 --pages 8,16,32,64,128
    python benchmarks/bench_cache_page.py --smoke

See OPT_NOTES.md § opt/cache-page-bench.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import bdh  # noqa: E402
from bdh_cache import CacheManager  # noqa: E402


def _elem_stride_bytes(cm: CacheManager) -> int:
    """Bytes per sequence slot of live KR+V prefix (matches ``_grow`` accounting)."""
    elem = torch.empty((), dtype=cm.storage_dtype).element_size()
    return (
        cm.n_layer
        * cm.batch_size
        * (cm.n_head * cm.n_latent + cm.n_embd)
        * elem
    )


def _linear_next_capacity(self: CacheManager, need: int) -> int:
    """Linear ``+page_size`` growth (historical; for A/B only)."""
    ps = self.page_size
    assert ps is not None
    target = max(int(need), self.capacity + ps)
    target = ((target + ps - 1) // ps) * ps
    return min(self.max_seq, target)


def _tokenwise_fill(cm: CacheManager, n_tokens: int) -> None:
    """Append+commit one token at a time across all layers (forces mid-stream grows)."""
    nh, N, D = cm.n_head, cm.n_latent, cm.n_embd
    B = cm.batch_size
    z_kr = torch.zeros(B, nh, 1, N, dtype=cm.storage_dtype, device=cm.device)
    z_v = torch.zeros(B, 1, 1, D, dtype=cm.storage_dtype, device=cm.device)
    for _ in range(n_tokens):
        for level in range(cm.n_layer):
            cm.append(level, z_kr, z_v)
        cm.commit()


def _run_policy(
    cfg: bdh.BDHConfig,
    max_seq: int,
    page: int,
    device: torch.device,
    next_cap: Optional[Callable] = None,
) -> Dict[str, int]:
    cm = CacheManager.from_config(
        cfg,
        batch_size=1,
        max_seq=max_seq,
        device=device,
        page_size=page,
    )
    if next_cap is not None:
        # Bind linear policy without mutating the class for other instances.
        cm._next_capacity = next_cap.__get__(cm, CacheManager)  # type: ignore[method-assign]
    t0 = time.perf_counter()
    _tokenwise_fill(cm, max_seq)
    wall_ms = (time.perf_counter() - t0) * 1000.0
    assert cm.seq_len == max_seq
    assert cm.capacity == max_seq
    return {
        "page": page,
        "max_seq": max_seq,
        "n_grows": cm.n_grows,
        "bytes_copied": cm.bytes_copied_on_grow,
        "elem_stride": _elem_stride_bytes(cm),
        "wall_ms": wall_ms,
    }


def _analytic_linear_bytes(elem_stride: int, page: int, max_seq: int) -> Tuple[int, int]:
    """Closed form: grows = max_seq/page - 1; bytes = stride * page * n(n+1)/2."""
    if max_seq % page != 0:
        # Round up pages to cover max_seq (matches page-aligned linear).
        n_pages = (max_seq + page - 1) // page
    else:
        n_pages = max_seq // page
    n_grows = max(0, n_pages - 1)
    bytes_ = elem_stride * page * (n_grows * (n_grows + 1) // 2)
    return n_grows, bytes_


def _count_aten_cat(fn: Callable[[], None]) -> int:
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


def _generate_smoke(
    cfg: bdh.BDHConfig,
    device: torch.device,
    prompt_len: int,
    n_new: int,
    page: int,
) -> Dict[str, object]:
    torch.manual_seed(0)
    m = bdh.BDH(cfg).to(device).eval()
    prompt = torch.randint(0, cfg.vocab_size, (1, prompt_len), device=device)

    with torch.no_grad():
        torch.manual_seed(1)
        fixed = m.generate(prompt.clone(), max_new_tokens=n_new, temperature=1.0)

        # Capture grow stats via a manual packed decode with paging (generate
        # rebuilds CacheManager internally; mirror its max_seq / page_size).
        max_seq = prompt_len + n_new
        cm = CacheManager.from_config(
            cfg, 1, max_seq, device, page_size=page
        )
        logits, _ = m(prompt.clone(), cache=cm)
        for _ in range(n_new):
            step = logits[:, -1:, :].argmax(dim=-1)
            logits, _ = m(step, cache=cm)

        torch.manual_seed(1)

        def _paged_gen():
            return m.generate(
                prompt.clone(),
                max_new_tokens=n_new,
                temperature=1.0,
                cache_page_size=page,
            )

        cats = _count_aten_cat(lambda: _paged_gen())
        torch.manual_seed(1)
        paged = _paged_gen()

    return {
        "prompt": prompt_len,
        "new": n_new,
        "page": page,
        "match_fixed": bool(torch.equal(fixed, paged)),
        "aten_cat": cats,
        "n_grows": cm.n_grows,
        "bytes_copied": cm.bytes_copied_on_grow,
        "capacity_final": cm.capacity,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-seq", type=int, default=2048)
    ap.add_argument(
        "--pages",
        type=str,
        default="8,16,32,64,128,256",
        help="Comma-separated page sizes to sweep",
    )
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--mlp-mult", type=int, default=32)
    ap.add_argument(
        "--gen-prompt",
        type=int,
        default=32,
        help="generate smoke prompt length",
    )
    ap.add_argument("--gen-new", type=int, default=256)
    ap.add_argument("--gen-page", type=int, default=16)
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="Tiny sweep for CI (max_seq=128, pages=8,16)",
    )
    args = ap.parse_args()

    if args.smoke:
        args.max_seq = 128
        args.pages = "8,16"
        args.gen_prompt = 8
        args.gen_new = 32
        args.gen_page = 8

    pages = [int(x) for x in args.pages.split(",") if x.strip()]
    device = torch.device("cpu")
    cfg = bdh.BDHConfig(
        n_layer=args.layers,
        n_embd=args.d,
        n_head=args.heads,
        mlp_internal_dim_multiplier=args.mlp_mult,
        dropout=0.0,
    )

    print(
        f"device={device} layers={cfg.n_layer} d={cfg.n_embd} nh={cfg.n_head} "
        f"max_seq={args.max_seq} dtype=fp32"
    )
    print(
        "policy: geometric = tip CacheManager._next_capacity (≥2×, page-aligned); "
        "linear = capacity+page_size (A/B only, not default)"
    )
    print()
    hdr = (
        f"{'page':>6} {'geo_grows':>9} {'lin_grows':>9} {'geo_bytes':>12} "
        f"{'lin_bytes':>12} {'bytes_x':>8} {'grows_x':>8} {'geo_ms':>8}"
    )
    print(hdr)
    print("-" * len(hdr))

    rows: List[Dict[str, object]] = []
    for page in pages:
        if page > args.max_seq:
            continue
        geo = _run_policy(cfg, args.max_seq, page, device, next_cap=None)
        lin = _run_policy(
            cfg, args.max_seq, page, device, next_cap=_linear_next_capacity
        )
        # Sanity vs closed form.
        pred_g, pred_b = _analytic_linear_bytes(
            int(lin["elem_stride"]), page, args.max_seq
        )
        assert lin["n_grows"] == pred_g, (lin["n_grows"], pred_g)
        assert lin["bytes_copied"] == pred_b, (lin["bytes_copied"], pred_b)

        bx = (
            lin["bytes_copied"] / geo["bytes_copied"]
            if geo["bytes_copied"]
            else float("inf")
        )
        gx = (
            lin["n_grows"] / geo["n_grows"] if geo["n_grows"] else float("inf")
        )
        print(
            f"{page:6d} {geo['n_grows']:9d} {lin['n_grows']:9d} "
            f"{geo['bytes_copied']:12d} {lin['bytes_copied']:12d} "
            f"{bx:7.1f}× {gx:7.1f}× {geo['wall_ms']:7.1f}"
        )
        rows.append(
            {
                "page": page,
                "geo_grows": geo["n_grows"],
                "lin_grows": lin["n_grows"],
                "geo_bytes": geo["bytes_copied"],
                "lin_bytes": lin["bytes_copied"],
                "bytes_x": bx,
                "grows_x": gx,
            }
        )

    print()
    print("### generate smoke (opt-in cache_page_size; defaults unchanged)")
    smoke = _generate_smoke(
        cfg, device, args.gen_prompt, args.gen_new, args.gen_page
    )
    print(
        f"generate prompt={smoke['prompt']} +new={smoke['new']} "
        f"cache_page_size={smoke['page']}: "
        f"match_fixed={smoke['match_fixed']} aten::cat={smoke['aten_cat']} "
        f"mirror_decode n_grows={smoke['n_grows']} "
        f"bytes_copied={smoke['bytes_copied']}"
    )
    print()
    print(
        "Note: wall_ms is tokenwise fill overhead on CPU (not GPU). "
        "Geometric cuts realloc copy bytes / grow count (O(S) vs O(S²)); "
        "default generate still preallocates full max_seq when "
        "cache_page_size=None (zero grows)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
