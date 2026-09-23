"""CPU-safe compile probe cleanup and forward-failure contract coverage.

No GPU or performance claims: this test deepens the failed forward-only train
probe path and the train_bwd cleanup path when ``clear_grads`` itself raises.
Defaults remain ``BDH_COMPILE=0`` / eager outside these opt-in probes.
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


@pytest.mark.parametrize("caller_training", [False, True])
@pytest.mark.parametrize("preexisting_grads", [False, True])
def test_failed_forward_train_probe_discards_wrapper_and_preserves_grads(
    monkeypatch, capsys, caller_training, preexisting_grads
):
    """A failed forward-only train probe keeps caller grads and restores mode.

    Unlike ``train_bwd``, the forward-only path must not clear pre-existing
    gradients when the compiled wrapper is discarded.
    """
    import train as tr

    monkeypatch.setenv("BDH_COMPILE", "1")
    monkeypatch.setenv("BDH_COMPILE_PROBE", "train")
    monkeypatch.setenv("BDH_COMPILE_MODE", "default")
    monkeypatch.setenv("BDH_COMPILE_FULLGRAPH", "0")
    importlib.reload(tr)

    compile_kwargs = {}
    clear_calls = []

    class FailingForwardProbe(torch.nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
            self.calls = 0

        def forward(self, *args, **kwargs):
            self.calls += 1
            raise RuntimeError("synthetic forward train probe failure")

    model = bdh.BDH(_small_cfg()).train(caller_training)
    expected_grads = []
    if preexisting_grads:
        for index, param in enumerate(model.parameters(), start=1):
            grad = torch.full_like(param, float(index))
            param.grad = grad
            expected_grads.append(grad.clone())
    wrapper = FailingForwardProbe(model)

    def compile_spy(compiled_model, **kwargs):
        assert compiled_model is model
        compile_kwargs.update(kwargs)
        return wrapper

    monkeypatch.setattr(tr.torch, "compile", compile_spy)
    # Track whether the failed forward path invokes clear_grads at all.
    original_clear = tr.clear_grads

    def counting_clear(module_or_optimizer):
        clear_calls.append(module_or_optimizer)
        return original_clear(module_or_optimizer)

    monkeypatch.setattr(tr, "clear_grads", counting_clear)

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
    assert clear_calls == []
    assert out is model
    assert out.training is caller_training
    assert model.training is caller_training
    if preexisting_grads:
        assert all(
            param.grad is not None and torch.equal(param.grad, expected)
            for param, expected in zip(model.parameters(), expected_grads)
        )
    else:
        assert all(param.grad is None for param in model.parameters())
    assert "torch.compile first probe failed" in captured
    assert "synthetic forward train probe failure" in captured
    assert "the compiled wrapper is discarded" in captured
    assert "probe=train" in captured
    assert "probe=train_bwd" not in captured
    assert "probe gradient cleanup failed" not in captured
    assert "torch.compile enabled" not in captured


@pytest.mark.parametrize("caller_training", [False, True])
def test_failed_backward_probe_survives_clear_grads_failure(
    monkeypatch, capsys, caller_training
):
    """If clear_grads raises, train_bwd still discards the wrapper and cleans."""
    import train as tr

    monkeypatch.setenv("BDH_COMPILE", "1")
    monkeypatch.setenv("BDH_COMPILE_PROBE", "train_bwd")
    monkeypatch.setenv("BDH_COMPILE_MODE", "default")
    monkeypatch.setenv("BDH_COMPILE_FULLGRAPH", "0")
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

    def boom_clear(_module_or_optimizer):
        raise RuntimeError("synthetic clear_grads failure")

    monkeypatch.setattr(tr.torch, "compile", compile_spy)
    monkeypatch.setattr(tr, "clear_grads", boom_clear)

    x = torch.randint(0, 256, (2, 8))
    y = torch.randint(0, 256, (2, 8))
    try:
        out = tr.maybe_compile(model, example_x=x, example_y=y)
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
    assert all(param.grad is None for param in model.parameters())
    assert "torch.compile first probe failed" in captured
    assert "synthetic backward probe failure" in captured
    assert "the compiled wrapper is discarded" in captured
    assert "probe gradient cleanup failed" in captured
    assert "synthetic clear_grads failure" in captured
    assert "direct parameter grad discard" in captured
    assert "probe=train_bwd" in captured
    assert "torch.compile enabled" not in captured


@pytest.mark.parametrize(
    "probe_raw",
    ["  TRAIN  ", "Train", "\ttrain\n"],
)
@pytest.mark.parametrize("caller_training", [False, True])
def test_normalized_forward_probe_name_preserves_caller_grads(
    monkeypatch, capsys, probe_raw, caller_training
):
    """Whitespace/case variants of ``train`` keep the forward-only contract."""
    import train as tr

    monkeypatch.setenv("BDH_COMPILE", "1")
    monkeypatch.setenv("BDH_COMPILE_PROBE", probe_raw)
    monkeypatch.setenv("BDH_COMPILE_MODE", "default")
    monkeypatch.setenv("BDH_COMPILE_FULLGRAPH", "0")
    importlib.reload(tr)

    compile_kwargs = {}
    clear_calls = []

    class ProbedWrapper(torch.nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
            self.calls = 0

        def forward(self, *args, **kwargs):
            self.calls += 1
            return self.module(*args, **kwargs)

    model = bdh.BDH(_small_cfg()).train(caller_training)
    expected_grads = []
    for index, param in enumerate(model.parameters(), start=1):
        grad = torch.full_like(param, float(index))
        param.grad = grad
        expected_grads.append(grad.clone())
    wrapper = ProbedWrapper(model)

    def compile_spy(compiled_model, **kwargs):
        assert compiled_model is model
        compile_kwargs.update(kwargs)
        return wrapper

    original_clear = tr.clear_grads

    def counting_clear(module_or_optimizer):
        clear_calls.append(module_or_optimizer)
        return original_clear(module_or_optimizer)

    monkeypatch.setattr(tr.torch, "compile", compile_spy)
    monkeypatch.setattr(tr, "clear_grads", counting_clear)

    x = torch.randint(0, 256, (2, 8))
    try:
        assert tr.COMPILE_PROBE == "train"
        out = tr.maybe_compile(model, example_x=x)
        captured = capsys.readouterr().out
        assert compile_kwargs == {"mode": "default"}
        assert wrapper.calls == 1
        assert clear_calls == []
        assert out is wrapper
        assert out.training is caller_training
        assert model.training is caller_training
        assert all(
            param.grad is not None and torch.equal(param.grad, expected)
            for param, expected in zip(model.parameters(), expected_grads)
        )
        assert "torch.compile enabled (mode=default, probe=train" in captured
        assert "probe=train_bwd" not in captured
        assert "first probe failed" not in captured
        assert "no backward probe was attempted" not in captured
    finally:
        monkeypatch.setenv("BDH_COMPILE", "0")
        monkeypatch.setenv("BDH_COMPILE_PROBE", "train_bwd")
        importlib.reload(tr)
