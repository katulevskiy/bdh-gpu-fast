"""CPU-safe compile probe input contract coverage.

No GPU or performance claims: this test checks that the default backward
probe refuses to return a compiled wrapper when no target batch is supplied.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest
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


def test_backward_probe_without_targets_falls_back_to_eager(
    monkeypatch, capsys
):
    """Missing targets skip the backward probe and preserve eager caller state."""
    import train as tr

    monkeypatch.setenv("BDH_COMPILE", "1")
    monkeypatch.setenv("BDH_COMPILE_PROBE", "train_bwd")
    monkeypatch.setenv("BDH_COMPILE_MODE", "default")
    monkeypatch.setenv("BDH_COMPILE_FULLGRAPH", "0")
    importlib.reload(tr)

    compile_kwargs = {}

    class UnexpectedProbe(torch.nn.Module):
        def forward(self, *args, **kwargs):
            raise AssertionError("backward probe should not run without targets")

    def compile_spy(model, **kwargs):
        compile_kwargs.update(kwargs)
        return UnexpectedProbe()

    monkeypatch.setattr(tr.torch, "compile", compile_spy)

    model = bdh.BDH(_small_cfg()).eval()
    param = next(model.parameters())
    param.grad = torch.ones_like(param)
    x = torch.randint(0, 256, (2, 8))
    try:
        out = tr.maybe_compile(model, example_x=x)
    finally:
        monkeypatch.setenv("BDH_COMPILE", "0")
        monkeypatch.setenv("BDH_COMPILE_PROBE", "train_bwd")
        importlib.reload(tr)

    captured = capsys.readouterr().out
    assert compile_kwargs == {"mode": "default"}
    assert out is model
    assert not out.training
    assert param.grad is not None
    assert torch.equal(param.grad, torch.ones_like(param))
    assert "probe=train_bwd requires example_y" in captured
    assert "no backward probe was attempted" in captured
    assert "first probe failed" not in captured


def test_backward_probe_without_targets_preserves_training_grads(
    monkeypatch, capsys
):
    """Missing targets preserve a training caller and every existing gradient."""
    import train as tr

    monkeypatch.setenv("BDH_COMPILE", "1")
    monkeypatch.setenv("BDH_COMPILE_PROBE", "train_bwd")
    monkeypatch.setenv("BDH_COMPILE_MODE", "default")
    monkeypatch.setenv("BDH_COMPILE_FULLGRAPH", "0")
    importlib.reload(tr)

    compile_kwargs = {}
    probe_calls = []

    class UnexpectedProbe(torch.nn.Module):
        def forward(self, *args, **kwargs):
            probe_calls.append(True)
            raise AssertionError("backward probe should not run without targets")

    def compile_spy(model, **kwargs):
        compile_kwargs.update(kwargs)
        return UnexpectedProbe()

    monkeypatch.setattr(tr.torch, "compile", compile_spy)

    model = bdh.BDH(_small_cfg()).train()
    expected_grads = []
    for index, param in enumerate(model.parameters(), start=1):
        grad = torch.full_like(param, float(index))
        param.grad = grad
        expected_grads.append(grad.clone())
    x = torch.randint(0, 256, (2, 8))
    try:
        out = tr.maybe_compile(model, example_x=x)
    finally:
        monkeypatch.setenv("BDH_COMPILE", "0")
        monkeypatch.setenv("BDH_COMPILE_PROBE", "train_bwd")
        importlib.reload(tr)

    captured = capsys.readouterr().out
    assert compile_kwargs == {"mode": "default"}
    assert probe_calls == []
    assert out is model
    assert out.training
    assert all(
        param.grad is not None and torch.equal(param.grad, expected)
        for param, expected in zip(model.parameters(), expected_grads)
    )
    assert "probe=train_bwd requires example_y" in captured
    assert "no backward probe was attempted" in captured


@pytest.mark.parametrize("fullgraph", [False, True])
@pytest.mark.parametrize("caller_training", [False, True])
def test_compile_without_probe_returns_wrapper_and_preserves_caller_state(
    monkeypatch, capsys, fullgraph, caller_training
):
    """An omitted example batch returns an unprobed wrapper without mutation."""
    import train as tr

    monkeypatch.setenv("BDH_COMPILE", "1")
    monkeypatch.setenv("BDH_COMPILE_PROBE", "train_bwd")
    monkeypatch.setenv("BDH_COMPILE_MODE", "default")
    monkeypatch.setenv("BDH_COMPILE_FULLGRAPH", "1" if fullgraph else "0")
    importlib.reload(tr)

    compile_kwargs = {}

    class UnprobedWrapper(torch.nn.Module):
        def forward(self, *args, **kwargs):
            raise AssertionError("the wrapper must not run without example_x")

    wrapper = UnprobedWrapper()

    def compile_spy(model, **kwargs):
        compile_kwargs.update(kwargs)
        return wrapper

    monkeypatch.setattr(tr.torch, "compile", compile_spy)

    model = bdh.BDH(_small_cfg()).train(caller_training)
    expected_grads = []
    for index, param in enumerate(model.parameters(), start=1):
        grad = torch.full_like(param, float(index))
        param.grad = grad
        expected_grads.append(grad.clone())
    try:
        out = tr.maybe_compile(model)
    finally:
        monkeypatch.setenv("BDH_COMPILE", "0")
        monkeypatch.setenv("BDH_COMPILE_PROBE", "train_bwd")
        importlib.reload(tr)

    captured = capsys.readouterr().out
    expected_compile_kwargs = {"mode": "default"}
    if fullgraph:
        expected_compile_kwargs["fullgraph"] = True
    assert compile_kwargs == expected_compile_kwargs
    assert out is wrapper
    assert model.training is caller_training
    assert all(
        param.grad is not None and torch.equal(param.grad, expected)
        for param, expected in zip(model.parameters(), expected_grads)
    )
    assert "without a first probe" in captured
    assert "probe=train_bwd" in captured
    assert f"fullgraph={fullgraph}" in captured
    assert "first probe failed" not in captured


def test_compile_failure_without_probe_preserves_caller_state(
    monkeypatch, capsys
):
    """No-probe construction failure keeps the original module untouched."""
    import train as tr

    monkeypatch.setenv("BDH_COMPILE", "1")
    monkeypatch.setenv("BDH_COMPILE_PROBE", "train_bwd")
    monkeypatch.setenv("BDH_COMPILE_MODE", "default")
    monkeypatch.setenv("BDH_COMPILE_FULLGRAPH", "0")
    importlib.reload(tr)

    compile_kwargs = {}

    def fail_compile(model, **kwargs):
        compile_kwargs.update(kwargs)
        raise RuntimeError("synthetic compile construction failure")

    monkeypatch.setattr(tr.torch, "compile", fail_compile)

    model = bdh.BDH(_small_cfg()).train()
    expected_grads = []
    for index, param in enumerate(model.parameters(), start=1):
        grad = torch.full_like(param, float(index))
        param.grad = grad
        expected_grads.append(grad.clone())
    try:
        out = tr.maybe_compile(model)
    finally:
        monkeypatch.setenv("BDH_COMPILE", "0")
        monkeypatch.setenv("BDH_COMPILE_PROBE", "train_bwd")
        importlib.reload(tr)

    captured = capsys.readouterr().out
    assert compile_kwargs == {"mode": "default"}
    assert out is model
    assert out.training
    assert all(
        param.grad is not None and torch.equal(param.grad, expected)
        for param, expected in zip(model.parameters(), expected_grads)
    )
    assert "torch.compile failed" in captured
    assert "soft-fallback to original eager module" in captured
    assert "without a first probe" not in captured
    assert "first probe failed" not in captured
