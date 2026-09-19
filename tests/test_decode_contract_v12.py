"""CPU-safe contracts for incremental decode score-times-V semantics."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kernels.attention_dispatch import bdh_attn_decode


@pytest.mark.parametrize("impl", ["eager", "blocked", "online", "triton", "cuda"])
def test_decode_is_raw_score_times_v_without_scale_or_softmax(impl):
    """Every decode backend keeps raw QK-transpose-times-V semantics on CPU."""
    # QK-transpose = [2, 3], so raw score-times-V is 2*5 + 3*7 = 31.
    q = torch.tensor([[[[1.0, 1.0]]]])
    k_past = torch.tensor([[[[1.0, 1.0], [2.0, 1.0]]]])
    v_past = torch.tensor([[[[5.0], [7.0]]]])
    expected = torch.tensor([[[[31.0]]]])

    got = bdh_attn_decode(q, k_past, v_past, impl=impl)

    assert torch.equal(got, expected)
