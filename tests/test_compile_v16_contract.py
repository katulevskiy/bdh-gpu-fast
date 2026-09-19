"""CPU-safe compile-construction fallback contract coverage.

No GPU or performance claims: this test checks that a ``torch.compile``
construction failure returns the original caller state without attempting a
probe or clearing caller-owned gradients.
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


def test_compile_construction_failure_preserves_caller_state(monkeypatch, capsys):
    """Constructor failure falls back without mutating mode or existing grads."""
    import train as tr

    monkeypatch.setenv("BDH_COMPILE", "1")
    monkeypatch.setenv("BDH_COMPILE_PROBE", "train_bwd")
    monkeypatch.setenv("BDH_COMPILE_MODE", "default")
    monkeypatch.setenv("BDH_COMPILE_FULLGRAPH", "0")
    importlib.reload(tr)

    def fail_compile(model, **kwargs):
        assert kwargs == {"mode": "default"}
        raise RuntimeError("synthetic compile construction failure")

    monkeypatch.setattr(tr.torch, "compile", fail_compile)

    model = bdh.BDH(_small_cfg()).eval()
    param = next(model.parameters())
    param.grad = torch.ones_like(param)
    x = torch.randint(0, 256, (2, 8))
    y = torch.randint(0, 256, (2, 8))
    try:
        out = tr.maybe_compile(model, example_x=x, example_y=y)
    finally:
        monkeypatch.setenv("BDH_COMPILE", "0")
        monkeypatch.setenv("BDH_COMPILE_PROBE", "train_bwd")
        importlib.reload(tr)

    captured = capsys.readouterr().out
    assert out is model
    assert not out.training
    assert param.grad is not None
    assert torch.equal(param.grad, torch.ones_like(param))
    assert "torch.compile failed" in captured
    assert "soft-fallback to original eager module" in captured
    assert "probe=train_bwd" in captured
    assert "first probe failed" not in captured
