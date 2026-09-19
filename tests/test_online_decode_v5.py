"""Focused CPU coverage for the post-#187 online decode path."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import bdh
from bdh_cache import CacheManager
from kernels.attention import eager_decode_attn
from kernels.attention_dispatch import bdh_attn_decode


def test_b3_long_s_packed_shared_v_decode_parity():
    """B>1 long-S packed KR/V views stay exact across CPU decode backends."""
    B, H, S, N, D = 3, 4, 2049, 16, 32
    cfg = bdh.BDHConfig(
        n_layer=1,
        n_embd=D,
        n_head=H,
        mlp_internal_dim_multiplier=8,
        dropout=0.0,
        vocab_size=128,
    )
    cm = CacheManager(
        n_layer=1,
        max_seq=S + 31,
        batch_size=B,
        n_head=H,
        n_latent=N,
        n_embd=D,
        device="cpu",
    )
    torch.manual_seed(905)
    cm._kr_buf[0].normal_()
    cm._v_buf[0].normal_()
    cm.seq_len = S
    K, V = cm.get_past(0)
    assert K is not None and V is not None
    assert K.stride() == (H * cm.capacity * N, cm.capacity * N, N, 1)
    assert V.stride() == (cm.capacity * D, cm.capacity * D, D, 1)

    Q = torch.randn(B, H, 1, N)
    ref = eager_decode_attn(Q, K, V)
    for impl in ("blocked", "online", "triton", "cuda"):
        got = bdh_attn_decode(Q, K, V, impl=impl)
        assert torch.allclose(got, ref, rtol=1e-4, atol=1e-5), (
            f"impl={impl} maxdiff={(got - ref).abs().max().item()}"
        )
