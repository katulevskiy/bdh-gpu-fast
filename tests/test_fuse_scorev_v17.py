"""Focused v17 contract coverage for distinct per-head score×V dispatch.

This stays CPU-safe: it exercises only the blocked/online PyTorch paths.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kernels.attention_dispatch import bdh_attn  # noqa: E402


@pytest.mark.parametrize("impl", ["blocked", "online"])
def test_dispatch_preserves_raw_score_contract_for_distinct_qk_and_per_head_v(impl):
    """Dispatch must preserve distinct Q/K and per-head V without softmax."""
    Q = torch.tensor(
        [
            [
                [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
                [[-1.0, 0.0], [2.0, -3.0], [4.0, 1.0]],
            ],
            [
                [[2.0, -2.0], [1.0, 3.0], [-4.0, 5.0]],
                [[0.0, 1.0], [-2.0, 4.0], [3.0, -1.0]],
            ],
        ],
        dtype=torch.float64,
    )
    K = torch.tensor(
        [
            [
                [[2.0, -1.0], [1.0, 3.0], [-2.0, 4.0]],
                [[3.0, 1.0], [-1.0, 2.0], [2.0, -4.0]],
            ],
            [
                [[-1.0, 2.0], [4.0, 1.0], [3.0, -2.0]],
                [[2.0, 3.0], [1.0, -2.0], [-3.0, 4.0]],
            ],
        ],
        dtype=torch.float64,
    )
    V = torch.tensor(
        [
            [
                [[7.0, 11.0], [13.0, 17.0], [19.0, 23.0]],
                [[29.0, 31.0], [37.0, 41.0], [43.0, 47.0]],
            ],
            [
                [[53.0, 59.0], [61.0, 67.0], [71.0, 73.0]],
                [[79.0, 83.0], [89.0, 97.0], [101.0, 103.0]],
            ],
        ],
        dtype=torch.float64,
    )

    scores = (Q @ K.transpose(-2, -1)).tril(diagonal=-1)
    expected = scores @ V
    got = bdh_attn(Q, K, V, impl=impl)

    assert torch.equal(got, expected)
    assert torch.equal(got[:, :, 0, :], torch.zeros_like(got[:, :, 0, :]))
