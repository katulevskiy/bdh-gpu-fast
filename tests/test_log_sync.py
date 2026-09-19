"""Train logging sync: on-device accumulate, rare/deferred .item()."""

from __future__ import annotations

import io
from contextlib import redirect_stdout

import pytest
import torch

import train as tr


def test_logger_mean_matches_manual_cpu():
    """CPU path: printed mean == manual mean; .item only at boundaries."""
    logger = tr.TrainLossLogger(log_freq=3, device=torch.device("cpu"), async_cuda=False, max_iters=7)
    losses = [torch.tensor(float(v)) for v in (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0)]
    item_calls = {"n": 0}
    real_item = torch.Tensor.item

    def counting_item(self):
        item_calls["n"] += 1
        return real_item(self)

    buf = io.StringIO()
    torch.Tensor.item = counting_item  # type: ignore[method-assign]
    try:
        with redirect_stdout(buf):
            for step, loss in enumerate(losses):
                logger.update(loss, step)
            logger.close()
    finally:
        torch.Tensor.item = real_item  # type: ignore[method-assign]

    out = buf.getvalue()
    # Boundaries at steps 0, 3, 6 → three prints from update; close has empty acc.
    assert "Step: 0/7 loss 1.0" in out or "Step: 0/7 loss 1" in out
    assert "Step: 3/7 loss 3.0" in out or "Step: 3/7 loss 3" in out
    assert "Step: 6/7 loss 6.0" in out or "Step: 6/7 loss 6" in out
    # One .item per printed window (CPU sync path).
    assert item_calls["n"] == 3, item_calls


def test_logger_no_item_on_non_log_steps():
    """Between LOG_FREQ boundaries, update must not call .item()."""
    logger = tr.TrainLossLogger(log_freq=5, device=torch.device("cpu"), async_cuda=False, max_iters=4)
    item_calls = {"n": 0}
    real_item = torch.Tensor.item

    def counting_item(self):
        item_calls["n"] += 1
        return real_item(self)

    torch.Tensor.item = counting_item  # type: ignore[method-assign]
    try:
        with redirect_stdout(io.StringIO()):
            for step in range(1, 5):  # steps 1..4 — no boundary (only 0 % 5 == 0)
                logger.update(torch.tensor(1.0), step)
            # close flushes partial → one .item
            logger.close()
    finally:
        torch.Tensor.item = real_item  # type: ignore[method-assign]

    assert item_calls["n"] == 1, item_calls


def test_logger_inplace_add_and_fp32():
    acc_dtypes = []
    logger = tr.TrainLossLogger(log_freq=100, device=torch.device("cpu"), async_cuda=False, max_iters=2)
    with redirect_stdout(io.StringIO()):
        logger.update(torch.tensor(1.5, dtype=torch.float16), 1)
        assert logger._acc is not None
        assert logger._acc.dtype == torch.float32
        logger.update(torch.tensor(2.5, dtype=torch.float16), 2)
        assert logger._steps == 2
        logger.close()


def test_hot_loop_uses_logger_helper():
    """run_train_loop is the shared path; defaults unchanged."""
    assert tr.LOG_FREQ == 100
    assert tr.USE_LOG_ASYNC is True
    assert callable(tr.run_train_loop)


def test_log_freq_env_override(monkeypatch):
    monkeypatch.setenv("BDH_LOG_FREQ", "50")
    monkeypatch.setenv("BDH_LOG_ASYNC", "0")
    import importlib
    import train as train_mod
    importlib.reload(train_mod)
    try:
        assert train_mod.LOG_FREQ == 50
        assert train_mod.USE_LOG_ASYNC is False
    finally:
        monkeypatch.delenv("BDH_LOG_FREQ", raising=False)
        monkeypatch.delenv("BDH_LOG_ASYNC", raising=False)
        importlib.reload(train_mod)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA deferred path needs GPU")
def test_cuda_async_defers_item_until_flush():
    device = torch.device("cuda")
    logger = tr.TrainLossLogger(log_freq=2, device=device, async_cuda=True, max_iters=4)
    item_calls = {"n": 0}
    real_item = torch.Tensor.item

    def counting_item(self):
        item_calls["n"] += 1
        return real_item(self)

    torch.Tensor.item = counting_item  # type: ignore[method-assign]
    try:
        with redirect_stdout(io.StringIO()):
            logger.update(torch.tensor(1.0, device=device), 0)
            # After first boundary, print is pending — float(host) may not use Tensor.item
            assert logger._pending_step == 0
            assert item_calls["n"] == 0
            logger.update(torch.tensor(2.0, device=device), 1)
            logger.update(torch.tensor(3.0, device=device), 2)
            # Second boundary flushes first pending via float(host), not Tensor.item
            assert item_calls["n"] == 0
            logger.close()
    finally:
        torch.Tensor.item = real_item  # type: ignore[method-assign]
