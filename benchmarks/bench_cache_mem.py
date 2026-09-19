"""Benchmark packed CacheManager vs legacy torch.cat cache: allocs + speed."""

from __future__ import annotations

import gc
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


def count_alloc_events(fn, reps=3):
    """CPU: use torch allocator stats if CUDA; else estimate via tensor count proxy.

    On CPU we measure peak allocated via a simple RSS-less approach: run under
    ``torch.cuda`` stats when available; otherwise count how many times
    ``torch.cat`` would dominate by instrumenting nothing and reporting
    preallocated bytes + wall time.
    """
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        for _ in range(reps):
            fn()
        return torch.cuda.max_memory_allocated()
    # CPU: no reliable per-op alloc counter without hooks; return None
    return None


def legacy_decode(model, prompt, n_new):
    cache = [None] * model.config.n_layer
    logits, _ = model(prompt, cache=cache)
    for _ in range(n_new):
        step = logits[:, -1:, :].argmax(dim=-1)  # greedy, deterministic
        logits, _ = model(step, cache=cache)
    return cache


def packed_decode(model, prompt, n_new, storage_dtype=None):
    max_seq = prompt.size(1) + n_new
    cache = CacheManager.from_config(
        model.config,
        batch_size=prompt.size(0),
        max_seq=max_seq,
        device=prompt.device,
        storage_dtype=storage_dtype,
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

    t_leg = timed(run_legacy)
    t_pack = timed(run_packed)
    t_fp16 = timed(run_packed_fp16)

    # Preallocated footprint
    nh, D = cfg.n_head, cfg.n_embd
    N = cfg.mlp_internal_dim_multiplier * D // nh
    max_seq = prompt_len + n_new
    bytes_fp32 = CacheManager.from_config(
        cfg, 1, max_seq, device, storage_dtype=torch.float32
    ).bytes_allocated
    bytes_fp16 = CacheManager.from_config(
        cfg, 1, max_seq, device, storage_dtype=torch.float16
    ).bytes_allocated

    # Legacy peak: after full decode, sum numel of live tensors
    with torch.no_grad():
        leg_cache = legacy_decode(m, prompt.clone(), n_new)
    leg_bytes = sum(
        e["kr"].numel() * e["kr"].element_size()
        + e["v"].numel() * e["v"].element_size()
        for e in leg_cache
    )

    print(f"device={device} layers={cfg.n_layer} d={cfg.n_embd}")
    print(f"decode prompt={prompt_len} + new={n_new} (greedy)")
    print(f"legacy cat-cache median:  {t_leg*1000:.2f} ms")
    print(f"packed fp32 median:       {t_pack*1000:.2f} ms  ({t_leg/t_pack:.2f}x vs legacy)")
    print(f"packed fp16 storage:      {t_fp16*1000:.2f} ms  ({t_leg/t_fp16:.2f}x vs legacy)")
    print(f"legacy final cache bytes: {leg_bytes}")
    print(f"packed fp32 prealloc:     {bytes_fp32}")
    print(f"packed fp16 prealloc:     {bytes_fp16}  ({bytes_fp32/bytes_fp16:.2f}x smaller)")
    print(
        "Note: packed avoids per-step torch.cat realloc/copy of growing KR/V; "
        "CPU medians vary; GPU would amplify bandwidth locality wins."
    )


if __name__ == "__main__":
    main()
