"""Layout-v35: CPU contract for preallocated decode logits views."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import bdh


def _cfg() -> bdh.BDHConfig:
    return bdh.BDHConfig(
        n_layer=2,
        n_embd=64,
        n_head=4,
        mlp_internal_dim_multiplier=8,
        dropout=0.0,
        vocab_size=32,
    )


def test_logits_out_preserves_strided_storage_for_single_and_multi_batch():
    """Decode logits may target a non-contiguous view without clobbering neighbors."""
    torch.manual_seed(0)
    cfg = _cfg()
    model = bdh.BDH(cfg).eval()

    for batch in (1, 2):
        idx = torch.randint(0, cfg.vocab_size, (batch, 1))
        with torch.inference_mode():
            reference = model(idx)[0][:, -1, :]

            storage = torch.full((batch, cfg.vocab_size, 2), -777.0)
            logits_out = storage[..., 0]
            assert logits_out.shape == (batch, cfg.vocab_size)
            assert logits_out.stride() == (cfg.vocab_size * 2, 2)

            got, loss = model(idx, logits_out=logits_out)

        assert loss is None
        assert got is logits_out
        torch.testing.assert_close(got, reference)
        assert torch.equal(storage[..., 1], torch.full_like(storage[..., 1], -777.0))
