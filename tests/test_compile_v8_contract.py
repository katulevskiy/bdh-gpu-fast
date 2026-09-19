"""CPU-safe compile-probe fallback contract coverage.

No GPU or performance claims: this test only checks probe cleanup and mode
restoration around a synthetic backward failure.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import bdh


def _small_cfg() -> bdh.BDHConfig:
    return bdh.BDHConfig(
        n_layer=2,
        n_embd=64,
        n_head=4,
        mlp_internal_dim_multiplier=8,
        dropout=0.0,
        vocab_size=256,
    )


def test_failed_backward_probe_restores_eval_mode_and_clears_grads(
    monkeypatch, capsys
):
    """Failed train_bwd probes leave an eval caller clean and in eval mode."""
    import train as tr

    monkeypatch.setenv("BDH_COMPILE", "1")
    monkeypatch.setenv("BDH_COMPILE_PROBE", "train_bwd")
    monkeypatch.setenv("BDH_COMPILE_MODE", "default")
    monkeypatch.setenv("BDH_COMPILE_FULLGRAPH", "0")
    importlib.reload(tr)

    class FailingBackward(torch.autograd.Function):
        @staticmethod
        def forward(ctx, value):
            return value.detach()

        @staticmethod
        def backward(ctx, grad_output):
            raise RuntimeError("synthetic backward probe failure")

    class ProbeFailure(torch.nn.Module):
        def __init__(self, target):
            super().__init__()
            self.target = target

        def forward(self, *args, **kwargs):
            param = next(self.target.parameters())
            # Simulate a partial gradient from work completed before failure.
            param.grad = torch.ones_like(param)
            return torch.zeros(1), FailingBackward.apply(param.sum())

    monkeypatch.setattr(
        tr.torch, "compile", lambda model, **kwargs: ProbeFailure(model)
    )
    model = bdh.BDH(_small_cfg()).eval()
    x = torch.randint(0, 256, (2, 8))
    y = torch.randint(0, 256, (2, 8))
    try:
        out = tr.maybe_compile(model, example_x=x, example_y=y)
    finally:
        monkeypatch.setenv("BDH_COMPILE", "0")
        importlib.reload(tr)

    captured = capsys.readouterr().out
    assert out is model
    assert not model.training
    assert all(param.grad is None for param in model.parameters())
    assert "first probe failed" in captured
    assert "original eager module is retained" in captured
