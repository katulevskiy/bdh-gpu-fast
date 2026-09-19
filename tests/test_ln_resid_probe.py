"""Probe-only coverage for the residual LayerNorm audit.

This test intentionally does not claim a new optimization: it locks the
remaining eager operator shape on the CPU torch 2.14 environment.
"""

from __future__ import annotations

import torch
from torch.profiler import ProfilerActivity, profile

import bdh


def _model() -> bdh.BDH:
    return bdh.BDH(
        bdh.BDHConfig(
            n_layer=1,
            n_embd=32,
            n_head=4,
            mlp_internal_dim_multiplier=8,
            dropout=0.0,
        )
    ).eval()


def test_residual_ln_has_no_eager_add_or_dtype_copy():
    """The landed reuse path is already the safe eager deepen."""
    model = _model()
    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        torch.manual_seed(0)
        x = torch.randn(2, 8, 32, dtype=dtype)
        y = torch.randn_like(x)
        with profile(activities=[ProfilerActivity.CPU]) as prof:
            got = model._residual_ln(x, y)

        counts = {event.key: event.count for event in prof.key_averages()}
        assert got.dtype == dtype
        assert counts.get("aten::native_layer_norm", 0) == 2
        assert counts.get("aten::add_", 0) == 1
        assert counts.get("aten::add", 0) == 0
        assert counts.get("aten::copy_", 0) == 0
        assert counts.get("aten::to", 0) == 0
        assert counts.get("aten::_to_copy", 0) == 0


def test_residual_ln_inputs_are_not_mutated():
    model = _model()
    torch.manual_seed(1)
    x = torch.randn(2, 8, 32)
    y = torch.randn_like(x)
    x0, y0 = x.clone(), y.clone()
    model._residual_ln(x, y)
    assert torch.equal(x, x0)
    assert torch.equal(y, y0)
