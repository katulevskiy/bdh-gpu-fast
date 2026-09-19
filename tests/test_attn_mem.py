"""Smoke tests for the CPU peak-memory attn probe (eager vs blocked vs online)."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kernels.attention import (  # noqa: E402
    DEFAULT_BLOCK_COLD,
    blocked_tril_attn,
    eager_tril_attn,
    max_score_tile_elems,
    online_tril_attn,
)
from kernels.attention_dispatch import resolve_attn_impl  # noqa: E402


def _qkv(T: int, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    B, H, N, D = 1, 2, 16, 32
    Q = torch.randn(B, H, T, N, generator=g)
    K = Q.clone()
    V = torch.randn(B, 1, T, D, generator=g)
    return Q, K, V


def test_default_attn_impl_remains_eager(monkeypatch):
    monkeypatch.delenv("BDH_ATTN_IMPL", raising=False)
    assert resolve_attn_impl() == "eager"


def test_tril_neg1_pos0_zero_all_impls():
    Q, K, V = _qkv(T=24, seed=7)
    for fn in (
        lambda q, k, v: eager_tril_attn(q, k, v),
        lambda q, k, v: blocked_tril_attn(q, k, v, block_size=8),
        lambda q, k, v: online_tril_attn(q, k, v, block_size=8),
    ):
        out = fn(Q, K, V)
        assert torch.equal(out[:, :, 0, :], torch.zeros_like(out[:, :, 0, :]))


def test_blocked_online_match_eager_and_alias():
    Q, K, V = _qkv(T=48, seed=11)
    ref = eager_tril_attn(Q, K, V)
    blk = blocked_tril_attn(Q, K, V, block_size=16)
    onl = online_tril_attn(Q, K, V, block_size=16)
    assert torch.allclose(blk, ref, rtol=1e-4, atol=1e-4)
    assert torch.equal(blk, onl)


def test_peak_score_bound_blocked_below_eager_growing_T():
    """Documented bound: blocked ≪ eager T×T for mid/long T (peak-mem win)."""
    BS = DEFAULT_BLOCK_COLD
    # At T<=BS bound can equal BS*BS == T*T (T=64); peak win starts mid-T.
    for T in (128, 256, 512, 1024):
        eager_b = T * T
        blk_b = max_score_tile_elems(T, BS)
        assert blk_b < eager_b, (T, blk_b, eager_b)
    # Bound stays budget-capped (≪ T²) for long T.
    assert max_score_tile_elems(1024, BS) <= max(BS * BS, 256 * 256) * 2


def test_bench_attn_mem_smoke():
    """Run the probe script in --smoke mode (CPU, tiny T)."""
    script = ROOT / "benchmarks" / "bench_attn_mem.py"
    env = os.environ.copy()
    env.setdefault("BDH_ATTN_IMPL", "eager")
    env.setdefault("OMP_NUM_THREADS", "2")
    proc = subprocess.run(
        [sys.executable, str(script), "--smoke"],
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stdout + "\n" + proc.stderr
    out = proc.stdout
    assert "peak_win" in out or "peak-mem" in out.lower()
    assert "Default BDH_ATTN_IMPL remains eager" in out or "defaults unchanged" in out.lower()
    assert "No GPU claims" in out or "no GPU" in out.lower()


def test_probe_helpers_importable():
    """Load bench helpers via file path (benchmarks is not a package)."""
    import importlib.util

    path = ROOT / "benchmarks" / "bench_attn_mem.py"
    spec = importlib.util.spec_from_file_location("bench_attn_mem", path)
    assert spec and spec.loader
    bam = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bam)

    Q, K, V = _qkv(T=32, seed=3)
    peak = bam.peak_score_elems(eager_tril_attn, Q, K, V)
    assert peak == 32 * 32
    peak_b = bam.peak_score_elems(
        lambda q, k, v: blocked_tril_attn(q, k, v, block_size=16), Q, K, V
    )
    assert peak_b < peak
