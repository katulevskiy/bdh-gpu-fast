"""Focused v16 contract coverage for strict raw score×V dispatch."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kernels.attention_dispatch import bdh_attn  # noqa: E402


@pytest.mark.parametrize("impl", ["blocked", "online"])
def test_dispatch_preserves_strict_raw_score_contract(impl):
    """Dispatch must keep raw scores and exclude the diagonal on CPU paths."""
    Q = torch.tensor([[[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]]], dtype=torch.float64)
    K = torch.tensor([[[[2.0, -1.0], [1.0, 3.0], [-2.0, 4.0]]]], dtype=torch.float64)
    V = torch.tensor([[[[7.0], [11.0], [13.0]]]], dtype=torch.float64)

    scores = (Q @ K.transpose(-2, -1)).tril(diagonal=-1)
    expected = scores @ V
    got = bdh_attn(Q, K, V, impl=impl)

    assert torch.equal(got, expected)
    assert torch.equal(got[:, :, 0, :], torch.zeros_like(got[:, :, 0, :]))
