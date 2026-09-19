"""Benchmark packed CacheManager vs legacy torch.cat cache: allocs + speed + cat counts."""

from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import bdh
from bdh_cache import CacheManager


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


def count_torch_cat(fn):
    """Hook torch.cat and return call count for one invocation of fn."""
    cats = {"n": 0}
    orig = torch.cat

    def hooked(*a, **k):
        cats["n"] += 1
        return orig(*a, **k)

    torch.cat = hooked
    try:
        fn()
    finally:
        torch.cat = orig
    return cats["n"]


def legacy_decode(model, prompt, n_new):
    cache = [None] * model.config.n_layer
    logits, _ = model(prompt, cache=cache)
    for _ in range(n_new):
        step = logits[:, -1:, :].argmax(dim=-1)  # greedy, deterministic
        logits, _ = model(step, cache=cache)
    return cache


def packed_decode(model, prompt, n_new, storage_dtype=None, page_size=None):
    max_seq = prompt.size(1) + n_new
    cache = CacheManager.from_config(
        model.config,
        batch_size=prompt.size(0),
        max_seq=max_seq,
        device=prompt.device,
        storage_dtype=storage_dtype,
        page_size=page_size,
    )
    logits, _ = model(prompt, cache=cache)
    for _ in range(n_new):
        step = logits[:, -1:, :].argmax(dim=-1)
        logits, _ = model(step, cache=cache)
    return cache


def main():
    device = torch.device("cpu")
    cfg = bdh.BDHConfig(
        n_layer=4,
        n_embd=128,
        n_head=4,
        mlp_internal_dim_multiplier=32,
        dropout=0.0,
    )
    torch.manual_seed(0)
    m = bdh.BDH(cfg).to(device).eval()
    prompt_len, n_new = 64, 128
    prompt = torch.randint(0, cfg.vocab_size, (1, prompt_len), device=device)

    def run_legacy():
        with torch.no_grad():
            legacy_decode(m, prompt.clone(), n_new)

    def run_packed():
        with torch.no_grad():
            packed_decode(m, prompt.clone(), n_new)

    def run_packed_fp16():
        with torch.no_grad():
            packed_decode(m, prompt.clone(), n_new, storage_dtype=torch.float16)

    def run_packed_paged():
        with torch.no_grad():
            packed_decode(m, prompt.clone(), n_new, page_size=64)

    def run_generate():
        with torch.no_grad():
            # greedy-ish via temperature → still hits generate path
            torch.manual_seed(0)
            m.generate(prompt.clone(), max_new_tokens=32, temperature=1.0)

    t_leg = timed(run_legacy)
    t_pack = timed(run_packed)
    t_fp16 = timed(run_packed_fp16)
    t_page = timed(run_packed_paged)

    # Preallocated footprint
    max_seq = prompt_len + n_new
    bytes_fp32 = CacheManager.from_config(
        cfg, 1, max_seq, device, storage_dtype=torch.float32
    ).bytes_allocated
    bytes_fp16 = CacheManager.from_config(
        cfg, 1, max_seq, device, storage_dtype=torch.float16
    ).bytes_allocated
    bytes_page0 = CacheManager.from_config(
        cfg, 1, max_seq, device, page_size=64
    ).bytes_allocated

    # Legacy peak: after full decode, sum numel of live tensors
    with torch.no_grad():
        leg_cache = legacy_decode(m, prompt.clone(), n_new)
    leg_bytes = sum(
        e["kr"].numel() * e["kr"].element_size()
        + e["v"].numel() * e["v"].element_size()
        for e in leg_cache
    )

    cat_legacy = count_torch_cat(run_legacy)
    cat_packed = count_torch_cat(run_packed)
    cat_generate = count_torch_cat(run_generate)

    print(f"device={device} layers={cfg.n_layer} d={cfg.n_embd}")
    print(f"decode prompt={prompt_len} + new={n_new} (greedy)")
    print(f"legacy cat-cache median:  {t_leg*1000:.2f} ms")
    print(f"packed fp32 median:       {t_pack*1000:.2f} ms  ({t_leg/t_pack:.2f}x vs legacy)")
    print(f"packed fp16 storage:      {t_fp16*1000:.2f} ms  ({t_leg/t_fp16:.2f}x vs legacy)")
    print(f"packed page=64 median:    {t_page*1000:.2f} ms  ({t_leg/t_page:.2f}x vs legacy)")
    print(f"legacy final cache bytes: {leg_bytes}")
    print(f"packed fp32 prealloc:     {bytes_fp32}")
    print(f"packed fp16 prealloc:     {bytes_fp16}  ({bytes_fp32/bytes_fp16:.2f}x smaller)")
    print(f"packed page=64 initial:   {bytes_page0}  (grows toward max_seq)")
    print(f"torch.cat calls legacy decode:   {cat_legacy}")
    print(f"torch.cat calls packed decode:   {cat_packed}")
    print(f"torch.cat calls generate(+32):   {cat_generate}")
    print(
        "Note: cache-v2 layer-contiguous + generate prealloc → near-zero aten::cat "
        "on generate; packed decode path already cat-free for KR/V (T=1)."
    )


if __name__ == "__main__":
    main()
