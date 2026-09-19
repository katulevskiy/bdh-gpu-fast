"""Follow-up contracts for the #159 train logger lifecycle."""

from __future__ import annotations

import io
from contextlib import redirect_stdout

import pytest
import torch

import train as tr


def test_logger_close_flushes_partial_window_once():
    """close() emits a partial window once and remains idempotent."""
    logger = tr.TrainLossLogger(
        log_freq=3,
        device=torch.device("cpu"),
        async_cuda=True,
        max_iters=5,
    )
    buf = io.StringIO()
    with redirect_stdout(buf):
        logger.update(torch.tensor(1.0), step=0)
        logger.update(torch.tensor(2.0), step=1)
        logger.update(torch.tensor(3.0), step=2)
        logger.close()
        logger.close()

    lines = [line for line in buf.getvalue().splitlines() if line.startswith("Step:")]
    assert len(lines) == 2
    assert "Step: 0/5 loss 1" in lines[0]
    assert "Step: 4/5 loss 2.5" in lines[1]


def test_logger_rejects_updates_after_close():
    """A closed logger cannot silently accept more train observations."""
    logger = tr.TrainLossLogger(
        log_freq=1,
        device=torch.device("cpu"),
        async_cuda=True,
        max_iters=1,
    )
    logger.close()
    with pytest.raises(RuntimeError, match="update after close"):
        logger.update(torch.tensor(1.0), step=0)
