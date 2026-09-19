"""CPU-safe compile probe normalization contract coverage.

No GPU or performance claims: this test checks that the compile probe
environment accepts the documented case-insensitive, whitespace-tolerant names
while retaining the train_bwd cleanup contract.
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


def test_probe_name_normalizes_case_and_whitespace(monkeypatch, capsys):
    """Normalized train_bwd names still run the backward probe cleanly."""
    import train as tr

    monkeypatch.setenv("BDH_COMPILE", "1")
    monkeypatch.setenv("BDH_COMPILE_PROBE", "  TRAIN_BWD  ")
    monkeypatch.setenv("BDH_COMPILE_MODE", "default")
    monkeypatch.setenv("BDH_COMPILE_FULLGRAPH", "0")
    importlib.reload(tr)

    compile_kwargs = {}

    def compile_spy(model, **kwargs):
        compile_kwargs.update(kwargs)
        return model

    monkeypatch.setattr(tr.torch, "compile", compile_spy)

    model = bdh.BDH(_small_cfg()).train()
    x = torch.randint(0, 256, (2, 8))
    y = torch.randint(0, 256, (2, 8))
    try:
        out = tr.maybe_compile(model, example_x=x, example_y=y)
    finally:
        monkeypatch.setenv("BDH_COMPILE", "0")
        monkeypatch.setenv("BDH_COMPILE_PROBE", "train_bwd")
        importlib.reload(tr)

    captured = capsys.readouterr().out
    assert tr.COMPILE_PROBE == "train_bwd"
    assert compile_kwargs == {"mode": "default"}
    assert out is model
    assert out.training
    assert all(param.grad is None for param in model.parameters())
    assert "torch.compile enabled" in captured
    assert "probe=train_bwd" in captured
    assert "no backward probe was attempted" not in captured
