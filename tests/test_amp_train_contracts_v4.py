"""AMP train v4: CPU-safe dtype skips and the CUDA-only scaler gate."""

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


def _configure_or_cpu_skip(amp_name: str) -> None:
    """Configure one dtype, turning unsupported CPU backends into a clear skip."""
    try:
        tr.configure_amp(amp_name, forward_only=True)
    except RuntimeError as exc:
        if tr.device.type == "cpu" and "autocast is unavailable" in str(exc):
            pytest.skip(f"CPU-safe skip for {amp_name}: {exc}")
        raise


@pytest.mark.parametrize("amp_name", ["bfloat16", "float16"])
def test_cpu_amp_skip_reason_keeps_dtype_and_backend_context(monkeypatch, amp_name):
    """CPU skips retain the requested dtype and actionable backend context."""
    if tr.device.type != "cpu":
        pytest.skip("CPU-only backend skip contract")
    probe_name = {
        "bfloat16": "cpu_bf16_available",
        "float16": "cpu_fp16_available",
    }[amp_name]
    monkeypatch.setattr(tr, probe_name, lambda: False)
    with pytest.raises(pytest.skip.Exception) as caught:
        _configure_or_cpu_skip(amp_name)
    message = str(caught.value)
    assert f"CPU-safe skip for {amp_name}" in message
    assert f"BDH_AMP_DTYPE={amp_name}" in message
    assert f"{amp_name} requested on CPU" in message
    assert "autocast is unavailable" in message


@pytest.mark.parametrize(
    ("amp_name", "expected_ptdtype"),
    [
        ("float32", torch.float32),
        ("bfloat16", torch.bfloat16),
        ("float16", torch.float16),
    ],
)
def test_train_dtype_matrix_is_cpu_safe(amp_name, expected_ptdtype):
    """All supported dtypes have an explicit, backend-safe config outcome."""
    _configure_or_cpu_skip(amp_name)
    assert tr.dtype == amp_name
    assert tr.ptdtype is expected_ptdtype
    assert tr._amp_forward_only is (amp_name != "float32")
    assert tr._use_scaler is (
        amp_name == "float16"
        and tr.device.type == "cuda"
        and torch.cuda.is_available()
    )
    assert tr.scaler is not None
    assert tr.scaler.is_enabled() is tr._use_scaler


@pytest.mark.parametrize("amp_name", ["bfloat16", "float16"])
def test_cpu_amp_failure_message_is_actionable(monkeypatch, amp_name):
    """Unsupported CPU AMP reports the requested dtype before state changes."""
    if tr.device.type != "cpu":
        pytest.skip("CPU-only backend skip contract")
    probe_name = {
        "bfloat16": "cpu_bf16_available",
        "float16": "cpu_fp16_available",
    }[amp_name]
    monkeypatch.setattr(tr, probe_name, lambda: False)
    tr.configure_amp("float32")
    with pytest.raises(RuntimeError) as caught:
        tr.configure_amp(amp_name, forward_only=True)
    message = str(caught.value)
    assert f"BDH_AMP_DTYPE={amp_name}" in message
    assert "requested on CPU" in message
    assert "autocast is unavailable" in message
    assert tr.dtype == "float32"
    assert tr._use_scaler is False


def test_cpu_amp_failure_preserves_full_previous_state(monkeypatch):
    """An unavailable CPU request leaves every prior AMP state field intact."""
    if tr.device.type != "cpu":
        pytest.skip("CPU-only backend skip contract")
    tr.configure_amp("float32")
    previous = {
        "dtype": tr.dtype,
        "ptdtype": tr.ptdtype,
        "ctx": tr.ctx,
        "scaler": tr.scaler,
        "use_scaler": tr._use_scaler,
        "forward_only": tr._amp_forward_only,
    }
    monkeypatch.setattr(tr, "cpu_fp16_available", lambda: False)
    with pytest.raises(RuntimeError, match="autocast is unavailable"):
        tr.configure_amp("float16", forward_only=True)
    assert tr.dtype == previous["dtype"]
    assert tr.ptdtype is previous["ptdtype"]
    assert tr.ctx is previous["ctx"]
    assert tr.scaler is previous["scaler"]
    assert tr._use_scaler is previous["use_scaler"]
    assert tr._amp_forward_only is previous["forward_only"]


def test_cpu_amp_failure_preserves_active_amp_state(monkeypatch):
    """A failed CPU switch cannot discard an already active AMP context."""
    if tr.device.type != "cpu":
        pytest.skip("CPU-only backend failure contract")
    _configure_or_cpu_skip("bfloat16")
    previous = {
        "dtype": tr.dtype,
        "ptdtype": tr.ptdtype,
        "ctx": tr.ctx,
        "scaler": tr.scaler,
        "use_scaler": tr._use_scaler,
        "forward_only": tr._amp_forward_only,
    }
    monkeypatch.setattr(tr, "cpu_fp16_available", lambda: False)
    with pytest.raises(RuntimeError, match="autocast is unavailable"):
        tr.configure_amp("float16", forward_only=False)
    assert tr.dtype == previous["dtype"]
    assert tr.ptdtype is previous["ptdtype"]
    assert tr.ctx is previous["ctx"]
    assert tr.scaler is previous["scaler"]
    assert tr._use_scaler is previous["use_scaler"]
    assert tr._amp_forward_only is previous["forward_only"]


def test_amp_construction_failure_preserves_full_previous_state(monkeypatch):
    """A backend construction failure cannot partially replace AMP state."""
    if tr.device.type != "cpu":
        pytest.skip("CPU-only AMP construction contract")
    tr.configure_amp("float32")
    previous = {
        "dtype": tr.dtype,
        "ptdtype": tr.ptdtype,
        "ctx": tr.ctx,
        "scaler": tr.scaler,
        "use_scaler": tr._use_scaler,
        "forward_only": tr._amp_forward_only,
    }
    monkeypatch.setattr(tr, "cpu_bf16_available", lambda: True)

    def _raise_grad_scaler(*args, **kwargs):
        raise RuntimeError("synthetic GradScaler construction failure")

    monkeypatch.setattr(tr.torch.amp, "GradScaler", _raise_grad_scaler)
    with pytest.raises(RuntimeError, match="synthetic GradScaler construction failure"):
        tr.configure_amp("bfloat16", forward_only=True)
    assert tr.dtype == previous["dtype"]
    assert tr.ptdtype is previous["ptdtype"]
    assert tr.ctx is previous["ctx"]
    assert tr.scaler is previous["scaler"]
    assert tr._use_scaler is previous["use_scaler"]
    assert tr._amp_forward_only is previous["forward_only"]


def test_amp_context_failure_preserves_full_previous_state(monkeypatch):
    """An autocast-context failure cannot partially replace AMP state."""
    if tr.device.type != "cpu":
        pytest.skip("CPU-only AMP construction contract")
    tr.configure_amp("float32")
    previous = {
        "dtype": tr.dtype,
        "ptdtype": tr.ptdtype,
        "ctx": tr.ctx,
        "scaler": tr.scaler,
        "use_scaler": tr._use_scaler,
        "forward_only": tr._amp_forward_only,
    }
    monkeypatch.setattr(tr, "cpu_bf16_available", lambda: True)

    def _raise_autocast_context(*args, **kwargs):
        raise RuntimeError("synthetic autocast context construction failure")

    monkeypatch.setattr(tr.bdh, "_autocast_context", _raise_autocast_context)
    with pytest.raises(RuntimeError, match="synthetic autocast context construction failure"):
        tr.configure_amp("bfloat16", forward_only=True)
    assert tr.dtype == previous["dtype"]
    assert tr.ptdtype is previous["ptdtype"]
    assert tr.ctx is previous["ctx"]
    assert tr.scaler is previous["scaler"]
    assert tr._use_scaler is previous["use_scaler"]
    assert tr._amp_forward_only is previous["forward_only"]


def test_invalid_amp_request_preserves_full_previous_state():
    """An invalid dtype request cannot partially replace AMP configuration."""
    tr.configure_amp("float32")
    previous = {
        "dtype": tr.dtype,
        "ptdtype": tr.ptdtype,
        "ctx": tr.ctx,
        "scaler": tr.scaler,
        "use_scaler": tr._use_scaler,
        "forward_only": tr._amp_forward_only,
    }
    with pytest.raises(ValueError, match="BDH_AMP_DTYPE must be"):
        tr.configure_amp("float64")
    assert tr.dtype == previous["dtype"]
    assert tr.ptdtype is previous["ptdtype"]
    assert tr.ctx is previous["ctx"]
    assert tr.scaler is previous["scaler"]
    assert tr._use_scaler is previous["use_scaler"]
    assert tr._amp_forward_only is previous["forward_only"]


@pytest.mark.parametrize("amp_name", ["bfloat16", "float16"])
def test_amp_forward_only_switch_is_explicit(monkeypatch, amp_name):
    """AMP keeps the full-forward default unless logits-only mode is requested."""
    probe_name = {
        "bfloat16": "cpu_bf16_available",
        "float16": "cpu_fp16_available",
    }[amp_name]
    with monkeypatch.context() as mp:
        mp.setattr(tr, "device", torch.device("cpu"))
        mp.setattr(tr, probe_name, lambda: True)
        tr.configure_amp(amp_name, forward_only=False)
        assert tr.dtype == amp_name
        assert tr._amp_forward_only is False
        tr.configure_amp(amp_name, forward_only=True)
        assert tr._amp_forward_only is True


@pytest.mark.parametrize(
    ("env_value", "expected"),
    [(None, False), ("0", False), ("false", False), ("1", True), ("true", True), ("True", True)],
)
def test_amp_forward_only_env_gate_is_explicit(monkeypatch, env_value, expected):
    """The env opt-in selects logits-only AMP without changing CPU safety."""
    with monkeypatch.context() as mp:
        mp.setattr(tr, "device", torch.device("cpu"))
        mp.setattr(tr, "cpu_bf16_available", lambda: True)
        if env_value is None:
            mp.delenv("BDH_AMP_FORWARD_ONLY", raising=False)
        else:
            mp.setenv("BDH_AMP_FORWARD_ONLY", env_value)
        tr.configure_amp("bfloat16", forward_only=None)
        assert tr.dtype == "bfloat16"
        assert tr._amp_forward_only is expected


@pytest.mark.parametrize("env_value", ["1", "true", "True"])
def test_amp_forward_only_env_cannot_enable_mode_for_float32(monkeypatch, env_value):
    """The logits-only opt-in cannot change the explicit fp32 contract."""
    with monkeypatch.context() as mp:
        mp.setattr(tr, "device", torch.device("cpu"))
        mp.setenv("BDH_AMP_FORWARD_ONLY", env_value)
        tr.configure_amp("float32", forward_only=None)
        assert tr.dtype == "float32"
        assert tr._amp_forward_only is False
        assert tr._use_scaler is False


def test_float32_switch_clears_active_amp_forward_only_state(monkeypatch):
    """Switching back to fp32 clears a previously active AMP mode."""
    with monkeypatch.context() as mp:
        mp.setattr(tr, "device", torch.device("cpu"))
        mp.setattr(tr, "cpu_bf16_available", lambda: True)
        tr.configure_amp("bfloat16", forward_only=True)
        assert tr._amp_forward_only is True

        mp.setenv("BDH_AMP_FORWARD_ONLY", "1")
        tr.configure_amp("float32", forward_only=None)
        assert tr.dtype == "float32"
        assert tr.ptdtype is torch.float32
        assert tr._amp_forward_only is False
        assert tr._use_scaler is False
        assert tr.scaler.is_enabled() is False


@pytest.mark.parametrize(
    ("amp_name", "device_type", "cuda_available", "expected"),
    [
        ("float32", "cpu", False, False),
        ("bfloat16", "cpu", False, False),
        ("float16", "cpu", False, False),
        ("float16", "cuda", False, False),
        ("bfloat16", "cuda", True, False),
        ("float16", "cuda", True, True),
    ],
)
def test_gradscaler_gate_is_cuda_float16_only(
    monkeypatch, amp_name, device_type, cuda_available, expected
):
    """The scaler gate requires float16, a CUDA device, and live CUDA support."""
    tr.configure_amp("float32")
    with monkeypatch.context() as mp:
        mp.setattr(tr, "device", torch.device(device_type))
        mp.setattr(torch.cuda, "is_available", lambda: cuda_available)
        assert tr._grad_scaler_allowed(amp_name) is expected
    tr.configure_amp("float32")


def test_cpu_amp_throughput_claim_is_none(monkeypatch):
    """CPU AMP remains correctness-only and cannot claim GPU throughput."""
    with monkeypatch.context() as mp:
        mp.setattr(tr, "device", torch.device("cpu"))
        assert tr.amp_throughput_claim_device() == "none"


@pytest.mark.parametrize("amp_name", ["bfloat16", "float16"])
def test_cpu_amp_throughput_claim_stays_none_with_active_amp(monkeypatch, amp_name):
    """Active CPU AMP keeps the throughput claim disabled for every AMP dtype."""
    probe_name = {
        "bfloat16": "cpu_bf16_available",
        "float16": "cpu_fp16_available",
    }[amp_name]
    with monkeypatch.context() as mp:
        mp.setattr(tr, "device", torch.device("cpu"))
        mp.setattr(tr, probe_name, lambda: True)
        tr.configure_amp(amp_name, forward_only=True)
        assert tr.amp_throughput_claim_device() == "none"


def test_cpu_amp_throughput_claim_ignores_cuda_runtime(monkeypatch):
    """A CPU AMP run cannot claim GPU throughput even if CUDA reports available."""
    with monkeypatch.context() as mp:
        mp.setattr(tr, "device", torch.device("cpu"))
        mp.setattr(torch.cuda, "is_available", lambda: True)
        mp.setattr(tr, "cpu_bf16_available", lambda: True)
        tr.configure_amp("bfloat16", forward_only=True)
        assert tr._use_scaler is False
        assert tr.amp_throughput_claim_device() == "none"


def test_cuda_amp_throughput_claim_requires_live_runtime(monkeypatch):
    """A stale CUDA device selection cannot claim throughput without CUDA."""
    with monkeypatch.context() as mp:
        mp.setattr(tr, "device", torch.device("cuda"))
        mp.setattr(torch.cuda, "is_available", lambda: False)
        assert tr.amp_throughput_claim_device() == "none"


def test_forward_only_runs_logits_under_amp_and_ce_in_fp32(monkeypatch):
    """Forward-only AMP calls logits-only forward and keeps CE outside autocast."""
    events = []
    ce_dtypes = []
    active = False

    class _ProbeContext:
        def __enter__(self):
            nonlocal active
            active = True
            events.append("enter")
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            nonlocal active
            active = False
            events.append("exit")

    class _TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.eye(3))

        def forward(self, x, y=None):
            assert active
            events.append(("forward", y))
            return (x @ self.weight).to(torch.float16), None

    original_cross_entropy = torch.nn.functional.cross_entropy

    def _record_cross_entropy(*args, **kwargs):
        assert not active
        ce_dtypes.append(args[0].dtype)
        return original_cross_entropy(*args, **kwargs)

    monkeypatch.setattr(tr, "ctx", _ProbeContext())
    monkeypatch.setattr(tr, "_amp_forward_only", True)
    monkeypatch.setattr(tr, "_use_scaler", False)
    monkeypatch.setattr(tr, "scaler", None)
    monkeypatch.setattr(torch.nn.functional, "cross_entropy", _record_cross_entropy)

    model = _TinyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    x = torch.tensor([[[2.0, 0.0, 0.0], [0.0, 2.0, 0.0]]])
    y = torch.tensor([[0, 1]])
    loss = tr.train_step(model, optimizer, x, y)

    assert loss.dtype is torch.float32
    assert events == ["enter", ("forward", None), "exit"]
    assert ce_dtypes == [torch.float32]


def test_default_amp_keeps_full_forward_under_context(monkeypatch):
    """Default AMP passes targets to the full model forward inside autocast."""
    events = []
    forward_targets = []
    forward_losses = []
    active = False

    class _ProbeContext:
        def __enter__(self):
            nonlocal active
            active = True
            events.append("enter")
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            nonlocal active
            active = False
            events.append("exit")

    class _TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.eye(3))

        def forward(self, x, y=None):
            assert active
            forward_targets.append(y)
            events.append(("forward", y is not None))
            logits = x @ self.weight
            model_loss = logits.square().mean()
            forward_losses.append(model_loss)
            return logits, model_loss

    monkeypatch.setattr(tr, "ctx", _ProbeContext())
    monkeypatch.setattr(tr, "_amp_forward_only", False)
    monkeypatch.setattr(tr, "_use_scaler", False)
    monkeypatch.setattr(tr, "scaler", None)

    model = _TinyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    x = torch.tensor([[[2.0, 0.0, 0.0], [0.0, 2.0, 0.0]]])
    y = torch.tensor([[0, 1]])
    loss = tr.train_step(model, optimizer, x, y)

    assert loss.ndim == 0
    assert events == ["enter", ("forward", True), "exit"]
    assert forward_targets[0] is y
    assert loss is forward_losses[0]


def test_unscaled_amp_train_step_updates_and_clears_grads(monkeypatch):
    """The CPU-safe unscaled AMP path updates parameters then clears grads."""
    monkeypatch.setattr(tr, "ctx", tr.nullcontext())
    monkeypatch.setattr(tr, "_amp_forward_only", False)
    monkeypatch.setattr(tr, "_use_scaler", False)
    monkeypatch.setattr(tr, "scaler", None)

    class _TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1, 1))

        def forward(self, x, y=None):
            logits = x @ self.weight
            return logits, logits.square().mean()

    model = _TinyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    initial_weight = model.weight.detach().clone()
    x = torch.tensor([[2.0]])
    y = torch.tensor([[0]])

    loss = tr.train_step(model, optimizer, x, y)

    assert loss.ndim == 0
    assert not torch.equal(model.weight.detach(), initial_weight)
    assert model.weight.grad is None


def test_failed_unscaled_optimizer_step_still_clears_grads(monkeypatch):
    """The unscaled AMP path clears grads even when optimizer.step fails."""
    monkeypatch.setattr(tr, "ctx", tr.nullcontext())
    monkeypatch.setattr(tr, "_amp_forward_only", False)
    monkeypatch.setattr(tr, "_use_scaler", False)
    monkeypatch.setattr(tr, "scaler", None)

    class _TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1, 1))

        def forward(self, x, y=None):
            logits = x @ self.weight
            return logits, logits.square().mean()

    model = _TinyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    step_calls = []

    def _raise_step():
        assert model.weight.grad is not None
        step_calls.append(True)
        raise RuntimeError("synthetic optimizer step failure")

    monkeypatch.setattr(optimizer, "step", _raise_step)
    with pytest.raises(RuntimeError, match="synthetic optimizer step failure"):
        tr.train_step(model, optimizer, torch.tensor([[2.0]]), torch.tensor([[0]]))

    assert step_calls == [True]
    assert model.weight.grad is None


def test_injected_scaler_path_orders_hooks_and_clears_grads(monkeypatch):
    """A CPU-safe scaler probe preserves scale/backward/step/update ordering."""
    events = []

    class _ScaledLoss:
        def __init__(self, loss):
            self.loss = loss

        def backward(self):
            events.append("backward")
            self.loss.backward()

    class _ProbeScaler:
        def is_enabled(self):
            return True

        def scale(self, loss):
            events.append("scale")
            return _ScaledLoss(loss)

        def step(self, optimizer):
            events.append("step")
            optimizer.step()

        def update(self):
            events.append("update")

    monkeypatch.setattr(tr, "ctx", tr.nullcontext())
    monkeypatch.setattr(tr, "_amp_forward_only", False)
    monkeypatch.setattr(tr, "_use_scaler", True)
    monkeypatch.setattr(tr, "scaler", _ProbeScaler())

    class _TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1, 1))

        def forward(self, x, y=None):
            logits = x @ self.weight
            return logits, logits.square().mean()

    model = _TinyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    initial_weight = model.weight.detach().clone()

    loss = tr.train_step(model, optimizer, torch.tensor([[2.0]]), torch.tensor([[0]]))

    assert loss.ndim == 0
    assert events == ["scale", "backward", "step", "update"]
    assert not torch.equal(model.weight.detach(), initial_weight)
    assert model.weight.grad is None


def test_scaler_update_failure_still_clears_grads(monkeypatch):
    """Scaler cleanup remains guaranteed when update fails after the step."""
    events = []

    class _ScaledLoss:
        def __init__(self, loss):
            self.loss = loss

        def backward(self):
            events.append("backward")
            self.loss.backward()

    class _FailingScaler:
        def is_enabled(self):
            return True

        def scale(self, loss):
            events.append("scale")
            return _ScaledLoss(loss)

        def step(self, optimizer):
            events.append("step")
            optimizer.step()

        def update(self):
            events.append("update")
            raise RuntimeError("synthetic scaler update failure")

    monkeypatch.setattr(tr, "ctx", tr.nullcontext())
    monkeypatch.setattr(tr, "_amp_forward_only", False)
    monkeypatch.setattr(tr, "_use_scaler", True)
    monkeypatch.setattr(tr, "scaler", _FailingScaler())

    class _TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1, 1))

        def forward(self, x, y=None):
            logits = x @ self.weight
            return logits, logits.square().mean()

    model = _TinyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    with pytest.raises(RuntimeError, match="synthetic scaler update failure"):
        tr.train_step(model, optimizer, torch.tensor([[2.0]]), torch.tensor([[0]]))

    assert events == ["scale", "backward", "step", "update"]
    assert model.weight.grad is None


def test_scaler_step_failure_still_clears_grads(monkeypatch):
    """Scaler cleanup remains guaranteed when the scaled step fails."""
    events = []

    class _ScaledLoss:
        def __init__(self, loss):
            self.loss = loss

        def backward(self):
            events.append("backward")
            self.loss.backward()

    class _FailingScaler:
        def is_enabled(self):
            return True

        def scale(self, loss):
            events.append("scale")
            return _ScaledLoss(loss)

        def step(self, optimizer):
            events.append("step")
            assert optimizer is not None
            raise RuntimeError("synthetic scaler step failure")

        def update(self):
            events.append("update")

    monkeypatch.setattr(tr, "ctx", tr.nullcontext())
    monkeypatch.setattr(tr, "_amp_forward_only", False)
    monkeypatch.setattr(tr, "_use_scaler", True)
    monkeypatch.setattr(tr, "scaler", _FailingScaler())

    class _TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1, 1))

        def forward(self, x, y=None):
            logits = x @ self.weight
            return logits, logits.square().mean()

    model = _TinyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    with pytest.raises(RuntimeError, match="synthetic scaler step failure"):
        tr.train_step(model, optimizer, torch.tensor([[2.0]]), torch.tensor([[0]]))

    assert events == ["scale", "backward", "step"]
    assert model.weight.grad is None


def test_disabled_scaler_falls_back_to_unscaled_step_and_clears_grads(monkeypatch):
    """A disabled scaler cannot intercept the CPU-safe unscaled optimizer path."""
    events = []

    class _DisabledScaler:
        def is_enabled(self):
            return False

        def scale(self, loss):
            events.append("scale")
            raise AssertionError("disabled scaler must not scale")

        def step(self, optimizer):
            events.append("step")
            raise AssertionError("disabled scaler must not step")

        def update(self):
            events.append("update")
            raise AssertionError("disabled scaler must not update")

    monkeypatch.setattr(tr, "ctx", tr.nullcontext())
    monkeypatch.setattr(tr, "_amp_forward_only", False)
    monkeypatch.setattr(tr, "_use_scaler", True)
    monkeypatch.setattr(tr, "scaler", _DisabledScaler())

    class _TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1, 1))

        def forward(self, x, y=None):
            logits = x @ self.weight
            return logits, logits.square().mean()

    model = _TinyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    initial_weight = model.weight.detach().clone()

    loss = tr.train_step(model, optimizer, torch.tensor([[2.0]]), torch.tensor([[0]]))

    assert loss.ndim == 0
    assert events == []
    assert not torch.equal(model.weight.detach(), initial_weight)
    assert model.weight.grad is None


def test_forward_only_scaler_path_keeps_ce_in_fp32_and_clears_grads(monkeypatch):
    """Forward-only AMP keeps CE in fp32 before the injected scaler path."""
    events = []
    ce_dtypes = []

    class _ScaledLoss:
        def __init__(self, loss):
            self.loss = loss

        def backward(self):
            events.append("backward")
            self.loss.backward()

    class _ProbeScaler:
        def is_enabled(self):
            return True

        def scale(self, loss):
            events.append("scale")
            return _ScaledLoss(loss)

        def step(self, optimizer):
            events.append("step")
            optimizer.step()

        def update(self):
            events.append("update")

    original_cross_entropy = torch.nn.functional.cross_entropy

    def _record_cross_entropy(*args, **kwargs):
        ce_dtypes.append(args[0].dtype)
        return original_cross_entropy(*args, **kwargs)

    class _TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.eye(3))

        def forward(self, x, y=None):
            logits = (x @ self.weight).to(torch.float16)
            return logits, None

    monkeypatch.setattr(tr, "ctx", tr.nullcontext())
    monkeypatch.setattr(tr, "_amp_forward_only", True)
    monkeypatch.setattr(tr, "_use_scaler", True)
    monkeypatch.setattr(tr, "scaler", _ProbeScaler())
    monkeypatch.setattr(torch.nn.functional, "cross_entropy", _record_cross_entropy)

    model = _TinyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    initial_weight = model.weight.detach().clone()

    loss = tr.train_step(
        model,
        optimizer,
        torch.tensor([[[2.0, 0.0, 0.0], [0.0, 2.0, 0.0]]]),
        torch.tensor([[0, 1]]),
    )

    assert loss.dtype is torch.float32
    assert ce_dtypes == [torch.float32]
    assert events == ["scale", "backward", "step", "update"]
    assert not torch.equal(model.weight.detach(), initial_weight)
    assert model.weight.grad is None


def test_cuda_amp_throughput_claim_reports_live_runtime(monkeypatch):
    """A live CUDA runtime is the only positive throughput claim surface."""
    with monkeypatch.context() as mp:
        mp.setattr(tr, "device", torch.device("cuda"))
        mp.setattr(torch.cuda, "is_available", lambda: True)
        assert tr.amp_throughput_claim_device() == "cuda"
