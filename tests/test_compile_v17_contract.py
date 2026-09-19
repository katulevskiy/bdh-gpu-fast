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


def test_unknown_probe_name_defaults_to_backward_contract(monkeypatch, capsys):
    """An unknown probe setting keeps the safe backward-probe contract."""
    import train as tr

    monkeypatch.setenv("BDH_COMPILE", "1")
    monkeypatch.setenv("BDH_COMPILE_PROBE", "unexpected")
    monkeypatch.setenv("BDH_COMPILE_MODE", "default")
    monkeypatch.setenv("BDH_COMPILE_FULLGRAPH", "0")
    importlib.reload(tr)

    compile_kwargs = {}

    class UnexpectedProbe(torch.nn.Module):
        def forward(self, *args, **kwargs):
            raise AssertionError("the backward probe should require targets")

    def compile_spy(model, **kwargs):
        compile_kwargs.update(kwargs)
        return UnexpectedProbe()

    monkeypatch.setattr(tr.torch, "compile", compile_spy)

    model = bdh.BDH(_small_cfg()).train()
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
    assert out.training
    assert "probe=train_bwd requires example_y" in captured
    assert "probe=train_bwd" in captured
    assert "probe=unexpected" not in captured
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


@pytest.mark.parametrize("caller_training", [False, True])
def test_forward_probe_without_targets_returns_wrapper_and_restores_state(
    monkeypatch, capsys, caller_training
):
    """The forward-only train probe accepts only inputs and restores mode."""
    import train as tr

    monkeypatch.setenv("BDH_COMPILE", "1")
    monkeypatch.setenv("BDH_COMPILE_PROBE", "train")
    monkeypatch.setenv("BDH_COMPILE_MODE", "default")
    monkeypatch.setenv("BDH_COMPILE_FULLGRAPH", "0")
    importlib.reload(tr)

    compile_kwargs = {}

    class ProbedWrapper(torch.nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
            self.calls = 0

        def forward(self, *args, **kwargs):
            self.calls += 1
            return self.module(*args, **kwargs)

    model = bdh.BDH(_small_cfg()).train(caller_training)
    wrapper = ProbedWrapper(model)

    def compile_spy(compiled_model, **kwargs):
        assert compiled_model is model
        compile_kwargs.update(kwargs)
        return wrapper

    monkeypatch.setattr(tr.torch, "compile", compile_spy)

    x = torch.randint(0, 256, (2, 8))
    try:
        out = tr.maybe_compile(model, example_x=x)
    finally:
        monkeypatch.setenv("BDH_COMPILE", "0")
        monkeypatch.setenv("BDH_COMPILE_PROBE", "train_bwd")
        importlib.reload(tr)

    captured = capsys.readouterr().out
    assert compile_kwargs == {"mode": "default"}
    assert out is wrapper
    assert wrapper.calls == 1
    assert out.training is caller_training
    assert model.training is caller_training
    assert "torch.compile enabled (mode=default, probe=train" in captured
    assert "first probe failed" not in captured


@pytest.mark.parametrize("fullgraph", [False, True])
@pytest.mark.parametrize("caller_training", [False, True])
def test_failed_backward_probe_falls_back_cleanly(
    monkeypatch, capsys, fullgraph, caller_training
):
    """A failed backward probe discards the wrapper and clears probe grads."""
    import train as tr

    monkeypatch.setenv("BDH_COMPILE", "1")
    monkeypatch.setenv("BDH_COMPILE_PROBE", "train_bwd")
    monkeypatch.setenv("BDH_COMPILE_MODE", "default")
    monkeypatch.setenv("BDH_COMPILE_FULLGRAPH", "1" if fullgraph else "0")
    importlib.reload(tr)

    compile_kwargs = {}

    class FailingProbe(torch.nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
            self.calls = 0

        def forward(self, x, y):
            self.calls += 1
            logits, loss = self.module(x, y)

            def fail_backward(_grad):
                raise RuntimeError("synthetic backward probe failure")

            assert loss is not None
            loss.register_hook(fail_backward)
            return logits, loss

    model = bdh.BDH(_small_cfg()).train(caller_training)
    for index, param in enumerate(model.parameters(), start=1):
        param.grad = torch.full_like(param, float(index))
    wrapper = FailingProbe(model)

    def compile_spy(compiled_model, **kwargs):
        assert compiled_model is model
        compile_kwargs.update(kwargs)
        return wrapper

    monkeypatch.setattr(tr.torch, "compile", compile_spy)

    x = torch.randint(0, 256, (2, 8))
    y = torch.randint(0, 256, (2, 8))
    try:
        out = tr.maybe_compile(model, example_x=x, example_y=y)
    finally:
        monkeypatch.setenv("BDH_COMPILE", "0")
        monkeypatch.setenv("BDH_COMPILE_PROBE", "train_bwd")
        importlib.reload(tr)

    captured = capsys.readouterr().out
    expected_compile_kwargs = {"mode": "default"}
    if fullgraph:
        expected_compile_kwargs["fullgraph"] = True
    assert compile_kwargs == expected_compile_kwargs
    assert wrapper.calls == 1
    assert out is model
    assert out.training is caller_training
    assert all(param.grad is None for param in model.parameters())
    assert "torch.compile first probe failed" in captured
    assert "synthetic backward probe failure" in captured
    assert "the compiled wrapper is discarded" in captured
    assert f"fullgraph={fullgraph}" in captured


@pytest.mark.parametrize("caller_training", [False, True])
def test_compile_failure_without_probe_preserves_caller_state(
    monkeypatch, capsys, caller_training
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
    assert compile_kwargs == {"mode": "default"}
    assert out is model
    assert out.training is caller_training
    assert all(
        param.grad is not None and torch.equal(param.grad, expected)
        for param, expected in zip(model.parameters(), expected_grads)
    )
    assert "torch.compile failed" in captured
    assert "soft-fallback to original eager module" in captured
    assert "without a first probe" not in captured
    assert "first probe failed" not in captured


@pytest.mark.parametrize("fullgraph", [False, True])
@pytest.mark.parametrize("caller_training", [False, True])
@pytest.mark.parametrize("preexisting_grads", [False, True])
def test_successful_backward_probe_returns_clean_wrapper(
    monkeypatch, capsys, fullgraph, caller_training, preexisting_grads
):
    """A successful backward probe clears all grads and restores caller mode."""
    import train as tr

    monkeypatch.setenv("BDH_COMPILE", "1")
    monkeypatch.setenv("BDH_COMPILE_PROBE", "train_bwd")
    monkeypatch.setenv("BDH_COMPILE_MODE", "default")
    monkeypatch.setenv("BDH_COMPILE_FULLGRAPH", "1" if fullgraph else "0")
    importlib.reload(tr)

    compile_kwargs = {}

    model = bdh.BDH(_small_cfg()).train(caller_training)
    if preexisting_grads:
        for index, param in enumerate(model.parameters(), start=1):
            param.grad = torch.full_like(param, float(index))

    class ProbedWrapper(torch.nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
            self.calls = 0

        def forward(self, *args, **kwargs):
            self.calls += 1
            return self.module(*args, **kwargs)

    wrapper = ProbedWrapper(model)

    def compile_spy(compiled_model, **kwargs):
        assert compiled_model is model
        compile_kwargs.update(kwargs)
        return wrapper

    monkeypatch.setattr(tr.torch, "compile", compile_spy)

    x = torch.randint(0, 256, (2, 8))
    y = torch.randint(0, 256, (2, 8))
    try:
        out = tr.maybe_compile(model, example_x=x, example_y=y)
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
    assert wrapper.calls == 1
    assert out.training is caller_training
    assert model.training is caller_training
    assert all(param.grad is None for param in model.parameters())
    assert "torch.compile enabled (mode=default, probe=train_bwd" in captured
    assert "first probe failed" not in captured


@pytest.mark.parametrize("caller_training", [False, True])
@pytest.mark.parametrize("with_targets", [False, True])
@pytest.mark.parametrize("preexisting_grads", [False, True])
def test_eval_probe_is_no_grad_and_restores_caller_state(
    monkeypatch, capsys, caller_training, with_targets, preexisting_grads
):
    """The eval probe preserves grads, handles targets, and restores mode."""
    import train as tr

    monkeypatch.setenv("BDH_COMPILE", "1")
    monkeypatch.setenv("BDH_COMPILE_PROBE", "eval")
    monkeypatch.setenv("BDH_COMPILE_MODE", "default")
    monkeypatch.setenv("BDH_COMPILE_FULLGRAPH", "0")
    importlib.reload(tr)

    compile_kwargs = {}

    class EvalProbe(torch.nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
            self.calls = 0
            self.training_at_call = []
            self.grad_enabled_at_call = []
            self.targets_at_call = []

        def forward(self, x, y=None):
            self.calls += 1
            self.training_at_call.append(self.training)
            self.grad_enabled_at_call.append(torch.is_grad_enabled())
            self.targets_at_call.append(y)
            return self.module(x, y) if y is not None else self.module(x)

    model = bdh.BDH(_small_cfg()).train(caller_training)
    expected_grads = []
    if preexisting_grads:
        for index, param in enumerate(model.parameters(), start=1):
            grad = torch.full_like(param, float(index))
            param.grad = grad
            expected_grads.append(grad.clone())
    wrapper = EvalProbe(model)

    def compile_spy(compiled_model, **kwargs):
        assert compiled_model is model
        compile_kwargs.update(kwargs)
        return wrapper

    monkeypatch.setattr(tr.torch, "compile", compile_spy)

    x = torch.randint(0, 256, (2, 8))
    y = torch.randint(0, 256, (2, 8)) if with_targets else None
    try:
        out = tr.maybe_compile(model, example_x=x, example_y=y)
    finally:
        monkeypatch.setenv("BDH_COMPILE", "0")
        monkeypatch.setenv("BDH_COMPILE_PROBE", "train_bwd")
        importlib.reload(tr)

    captured = capsys.readouterr().out
    assert compile_kwargs == {"mode": "default"}
    assert wrapper.calls == 1
    assert wrapper.training_at_call == [False]
    assert wrapper.grad_enabled_at_call == [False]
    assert (wrapper.targets_at_call[0] is not None) is with_targets
    assert out is wrapper
    assert out.training is caller_training
    assert model.training is caller_training
    if preexisting_grads:
        assert all(
            param.grad is not None and torch.equal(param.grad, expected)
            for param, expected in zip(model.parameters(), expected_grads)
        )
    assert "torch.compile enabled (mode=default, probe=eval" in captured
    assert "first probe failed" not in captured


@pytest.mark.parametrize("caller_training", [False, True])
def test_failed_eval_probe_discards_wrapper_and_restores_state(
    monkeypatch, capsys, caller_training
):
    """A failed eval probe keeps the eager module and its caller state."""
    import train as tr

    monkeypatch.setenv("BDH_COMPILE", "1")
    monkeypatch.setenv("BDH_COMPILE_PROBE", "eval")
    monkeypatch.setenv("BDH_COMPILE_MODE", "default")
    monkeypatch.setenv("BDH_COMPILE_FULLGRAPH", "0")
    importlib.reload(tr)

    compile_kwargs = {}

    class FailingEvalProbe(torch.nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
            self.calls = 0

        def forward(self, x, y=None):
            self.calls += 1
            raise RuntimeError("synthetic eval probe failure")

    model = bdh.BDH(_small_cfg()).train(caller_training)
    expected_grads = []
    for index, param in enumerate(model.parameters(), start=1):
        grad = torch.full_like(param, float(index))
        param.grad = grad
        expected_grads.append(grad.clone())
    wrapper = FailingEvalProbe(model)

    def compile_spy(compiled_model, **kwargs):
        assert compiled_model is model
        compile_kwargs.update(kwargs)
        return wrapper

    monkeypatch.setattr(tr.torch, "compile", compile_spy)

    x = torch.randint(0, 256, (2, 8))
    try:
        out = tr.maybe_compile(model, example_x=x)
    finally:
        monkeypatch.setenv("BDH_COMPILE", "0")
        monkeypatch.setenv("BDH_COMPILE_PROBE", "train_bwd")
        importlib.reload(tr)

    captured = capsys.readouterr().out
    assert compile_kwargs == {"mode": "default"}
    assert wrapper.calls == 1
    assert out is model
    assert out.training is caller_training
    assert model.training is caller_training
    assert all(
        param.grad is not None and torch.equal(param.grad, expected)
        for param, expected in zip(model.parameters(), expected_grads)
    )
    assert "torch.compile first probe failed" in captured
    assert "synthetic eval probe failure" in captured
    assert "the compiled wrapper is discarded" in captured
    assert "probe=eval" in captured
    assert "torch.compile enabled" not in captured
