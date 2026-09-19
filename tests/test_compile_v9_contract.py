"""CPU-safe successful compile-probe contract coverage.

No GPU or performance claims: this test checks that a successful backward
probe restores a training caller and clears probe gradients before return.
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


def test_successful_backward_probe_restores_train_mode_and_clears_grads(
    monkeypatch, capsys
):
    """Successful train_bwd probes leave a training caller clean and training."""
    import train as tr

    monkeypatch.setenv("BDH_COMPILE", "1")
    monkeypatch.setenv("BDH_COMPILE_PROBE", "train_bwd")
    monkeypatch.setenv("BDH_COMPILE_MODE", "default")
    monkeypatch.setenv("BDH_COMPILE_FULLGRAPH", "0")
    importlib.reload(tr)
    # Keep the success-path contract independent of inductor/C++ availability.
    monkeypatch.setattr(tr.torch, "compile", lambda model, **kwargs: model)

    model = bdh.BDH(_small_cfg()).train()
    x = torch.randint(0, 256, (2, 8))
    y = torch.randint(0, 256, (2, 8))
    try:
        out = tr.maybe_compile(model, example_x=x, example_y=y)
    finally:
        monkeypatch.setenv("BDH_COMPILE", "0")
        importlib.reload(tr)

    captured = capsys.readouterr().out
    assert out is model
    assert out.training
    assert all(param.grad is None for param in model.parameters())
    assert "torch.compile enabled" in captured
    assert "probe=train_bwd" in captured
