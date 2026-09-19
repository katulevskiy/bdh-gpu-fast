"""Smoke: train_step clear_grads (set_to_none) + fused AdamW path.

Defaults unchanged (BDH_FUSED_ADAMW=1, COMPILE=0). Asserts grads are None
after train_step and that make_optimizer / clear_grads honor the harden path.
"""

from __future__ import annotations

import inspect
import os
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]

# Keep import-time defaults: no compile, fused on (env default).
os.environ.setdefault("BDH_COMPILE", "0")
os.environ.setdefault("BDH_AMP_DTYPE", "float32")

import bdh
import train as tr


def _tiny_model(device=None):
    device = device or tr.device
    cfg = bdh.BDHConfig(
        n_layer=1,
        n_embd=32,
        n_head=2,
        mlp_internal_dim_multiplier=4,
        dropout=0.0,
        vocab_size=256,
    )
    return bdh.BDH(cfg).to(device)


def _batch(B=2, T=16, device=None):
    device = device or tr.device
    x = torch.randint(0, 256, (B, T), device=device)
    y = torch.randint(0, 256, (B, T), device=device)
    return x, y


def test_clear_grads_sets_param_grad_none():
    model = _tiny_model()
    x, y = _batch()
    model.train()
    _, loss = model(x, y)
    loss.backward()
    assert any(p.grad is not None for p in model.parameters())
    tr.clear_grads(model)
    for p in model.parameters():
        assert p.grad is None, "clear_grads must set_to_none, not fill zeros"


def test_clear_grads_sets_optimizer_owned_grads_none():
    model = _tiny_model()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x, y = _batch()
    _, loss = model(x, y)
    loss.backward()
    assert any(p.grad is not None for p in model.parameters())
    tr.clear_grads(opt)
    for p in model.parameters():
        assert p.grad is None


def test_train_step_leaves_grads_none():
    model = _tiny_model()
    opt = tr.make_optimizer(model)
    x, y = _batch()
    loss = tr.train_step(model, opt, x, y)
    assert torch.isfinite(loss.detach())
    for p in model.parameters():
        assert p.grad is None, "train_step must clear via set_to_none"


def test_train_step_smoke_three_iters():
    model = _tiny_model()
    opt = tr.make_optimizer(model)
    x, y = _batch()
    losses = []
    for _ in range(3):
        losses.append(float(tr.train_step(model, opt, x, y).detach()))
        for p in model.parameters():
            assert p.grad is None
    assert all(v == v and abs(v) < 1e6 for v in losses)  # finite, sane


def test_train_step_source_uses_clear_grads_not_fill_zero():
    src = inspect.getsource(tr.train_step)
    assert "clear_grads(optimizer)" in src
    assert "zero_grad(set_to_none=False)" not in src
    # No bare optimizer.zero_grad() without going through clear_grads.
    assert "optimizer.zero_grad(" not in src


def test_clear_grads_source_requires_set_to_none():
    src = inspect.getsource(tr.clear_grads)
    assert "set_to_none=True" in src


def test_make_optimizer_fused_default_when_available():
    """BDH_FUSED_ADAMW default is on; fused AdamW works on this torch build."""
    assert tr.USE_FUSED_ADAMW is True
    model = _tiny_model()
    opt = tr.make_optimizer(model)
    # torch stores the flag in defaults when fused=True succeeded.
    assert opt.defaults.get("fused") is True


def test_make_optimizer_respects_fused_off(monkeypatch):
    model = _tiny_model()
    monkeypatch.setattr(tr, "USE_FUSED_ADAMW", False)
    opt = tr.make_optimizer(model)
    assert opt.defaults.get("fused") in (None, False)
