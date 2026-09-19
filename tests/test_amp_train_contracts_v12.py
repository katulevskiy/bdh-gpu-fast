"""AMP train v12: enforce the forward-only CE input contract."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ.setdefault("BDH_AMP_DTYPE", "float32")
os.environ.setdefault("BDH_COMPILE", "0")

import train as tr


@pytest.fixture(autouse=True)
def _restore_fp32_amp():
    yield
    tr.configure_amp("float32")


def test_forward_only_promotes_and_flattens_logits_before_ce(monkeypatch):
    """Forward-only CE receives flattened fp32 logits outside autocast."""
    if tr.device.type == "cpu" and not tr.cpu_bf16_available():
        pytest.skip("CPU bf16 autocast unavailable")
    tr.configure_amp("bfloat16", forward_only=True)

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = torch.nn.Parameter(torch.eye(4))
            self.received_targets = "unset"
            self.logits_dtype = None
            self.forward_autocast = False

        def forward(self, x, y=None):
            self.received_targets = y
            self.forward_autocast = torch.is_autocast_enabled(
                device_type=tr.device.type
            )
            features = torch.ones((*x.shape[:2], 4), device=x.device)
            logits = features @ self.proj
            self.logits_dtype = logits.dtype
            return logits, None

    model = Tiny().to(tr.device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    x = torch.zeros((2, 3, 4), device=tr.device)
    y = torch.tensor([[0, 1, 2], [3, 0, 1]], device=tr.device)
    observed = {}
    real_cross_entropy = torch.nn.functional.cross_entropy

    def record_cross_entropy(input, target, *args, **kwargs):
        observed["input_dtype"] = input.dtype
        observed["input_shape"] = tuple(input.shape)
        observed["target_dtype"] = target.dtype
        observed["target_shape"] = tuple(target.shape)
        observed["autocast"] = torch.is_autocast_enabled(device_type=tr.device.type)
        return real_cross_entropy(input, target, *args, **kwargs)

    monkeypatch.setattr(
        torch.nn.functional, "cross_entropy", record_cross_entropy
    )
    loss = tr.train_step(model, optimizer, x, y)

    assert model.received_targets is None
    assert model.forward_autocast is True
    assert model.logits_dtype is torch.bfloat16
    assert observed == {
        "input_dtype": torch.float32,
        "input_shape": (6, 4),
        "target_dtype": torch.int64,
        "target_shape": (6,),
        "autocast": False,
    }
    assert loss.dtype is torch.float32
    assert torch.isfinite(loss)
    assert all(parameter.grad is None for parameter in model.parameters())
