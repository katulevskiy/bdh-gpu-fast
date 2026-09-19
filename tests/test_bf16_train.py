"""Optional train AMP (BDH_AMP_DTYPE) parity vs fp32.

Default training stays fp32. Set BDH_AMP_DTYPE=bfloat16|float16 for autocast.
GradScaler is only for float16+CUDA (asserted here; this box is CPU-only).
Attention math (tril(-1), no softmax, no scale) is unchanged.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Ensure default fp32 before importing train (reads env at import).
os.environ.setdefault("BDH_AMP_DTYPE", "float32")
os.environ.setdefault("BDH_COMPILE", "0")

import bdh
import train as tr

# Documented tolerances for train AMP vs fp32 (CPU autocast). Not bit-identical.
# Observed on this box (tiny cfg): bf16 logits max ~4e-3, grads ~5e-3; fp16 lower.
# Headroom matches decode-amp forward table style.
TRAIN_AMP_ATOL = {
    torch.float16: 5e-2,
    torch.bfloat16: 5e-2,
}
TRAIN_AMP_RTOL = {
    torch.float16: 5e-2,
    torch.bfloat16: 5e-2,
}
TRAIN_AMP_GRAD_ATOL = {
    torch.float16: 1e-1,
    torch.bfloat16: 1e-1,
}
TRAIN_AMP_GRAD_RTOL = {
    torch.float16: 1e-1,
    torch.bfloat16: 1e-1,
}


def _small_cfg(**kwargs) -> bdh.BDHConfig:
    defaults = dict(
        n_layer=2,
        n_embd=64,
        n_head=4,
        mlp_internal_dim_multiplier=8,
        dropout=0.0,
        vocab_size=256,
    )
    defaults.update(kwargs)
    return bdh.BDHConfig(**defaults)


def _amp_or_skip(name: str) -> torch.dtype:
    """Resolve dtype; skip bf16 on CPU when unavailable."""
    resolved = tr.parse_amp_dtype(name)
    pt = {"float16": torch.float16, "bfloat16": torch.bfloat16}[resolved]
    if resolved == "bfloat16" and tr.device.type == "cpu" and not tr.cpu_bf16_available():
        pytest.skip("CPU bf16 autocast unavailable")
    return pt


def test_parse_amp_dtype_aliases():
    assert tr.parse_amp_dtype(None) == "float32"
    assert tr.parse_amp_dtype("") == "float32"
    assert tr.parse_amp_dtype("fp32") == "float32"
    assert tr.parse_amp_dtype("bf16") == "bfloat16"
    assert tr.parse_amp_dtype("FP16") == "float16"
    with pytest.raises(ValueError):
        tr.parse_amp_dtype("float64")


def test_default_amp_is_fp32_no_scaler():
    tr.configure_amp("float32")
    assert tr.dtype == "float32"
    assert tr._use_scaler is False
    assert tr.scaler is not None and tr.scaler.is_enabled() is False


def test_gradscaler_only_fp16_cuda():
    """Scaler enabled iff float16 AND CUDA; bf16 never scales."""
    tr.configure_amp("bfloat16")
    assert tr._use_scaler is False
    assert tr.scaler.is_enabled() is False

    tr.configure_amp("float16")
    # This sandbox is CPU-only → still disabled; document CUDA gate.
    expect = tr.device.type == "cuda"
    assert tr._use_scaler is expect
    assert tr.scaler.is_enabled() is expect
    tr.configure_amp("float32")  # restore


def test_tril_neg1_unchanged_under_amp_config():
    """Configuring AMP must not alter attention mask semantics."""
    tr.configure_amp("bfloat16")
    cfg = _small_cfg()
    attn = bdh.Attention(cfg)
    N = cfg.mlp_internal_dim_multiplier * cfg.n_embd // cfg.n_head
    Q = torch.randn(2, cfg.n_head, 8, N)
    V = torch.randn(2, 1, 8, cfg.n_embd)
    out, _, _ = attn(Q, Q, V)
    # Position 0 has no past → zeros (tril diagonal=-1 excludes self).
    assert torch.count_nonzero(out[:, :, 0, :]) == 0
    tr.configure_amp("float32")


@pytest.mark.parametrize("amp_name", ["float16", "bfloat16"])
def test_train_amp_logits_loss_close_to_fp32(amp_name):
    amp_dtype = _amp_or_skip(amp_name)
    cfg = _small_cfg()
    torch.manual_seed(42)
    model = bdh.BDH(cfg)
    sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
    x = torch.randint(0, cfg.vocab_size, (4, 32))
    y = torch.randint(0, cfg.vocab_size, (4, 32))

    # fp32 reference
    m_ref = bdh.BDH(cfg)
    m_ref.load_state_dict(sd)
    m_ref.train()
    with torch.enable_grad():
        ref_logits, ref_loss = m_ref(x, y)

    # AMP path via train.configure_amp + ctx (same as train_step)
    tr.configure_amp(amp_name)
    m_amp = bdh.BDH(cfg)
    m_amp.load_state_dict(sd)
    m_amp.train()
    with tr.ctx:
        amp_logits, amp_loss = m_amp(x, y)

    atol, rtol = TRAIN_AMP_ATOL[amp_dtype], TRAIN_AMP_RTOL[amp_dtype]
    max_logits = (ref_logits - amp_logits.float()).abs().max().item()
    max_loss = (ref_loss - amp_loss.float()).abs().item()
    assert torch.allclose(ref_logits, amp_logits.float(), rtol=rtol, atol=atol), (
        f"{amp_name} logits vs fp32 max={max_logits} (atol={atol}, rtol={rtol})"
    )
    assert torch.allclose(ref_loss, amp_loss.float(), rtol=rtol, atol=atol), (
        f"{amp_name} loss vs fp32 abs={max_loss} (atol={atol}, rtol={rtol})"
    )
    tr.configure_amp("float32")


@pytest.mark.parametrize("amp_name", ["float16", "bfloat16"])
def test_train_amp_grads_close_to_fp32(amp_name):
    """Forward+backward under train.ctx vs fp32; capture grads before zero_grad.

    train_step() clears grads; this mirrors its autocast/scaler gate without step.
    """
    amp_dtype = _amp_or_skip(amp_name)
    cfg = _small_cfg()
    torch.manual_seed(43)
    base = bdh.BDH(cfg)
    sd = {k: v.detach().clone() for k, v in base.state_dict().items()}
    x = torch.randint(0, cfg.vocab_size, (4, 32), device=tr.device)
    y = torch.randint(0, cfg.vocab_size, (4, 32), device=tr.device)

    def fwd_bwd(amp: str):
        tr.configure_amp(amp)
        m = bdh.BDH(cfg).to(tr.device)
        m.load_state_dict(sd)
        m.train()
        m.zero_grad(set_to_none=True)
        opt = torch.optim.SGD(m.parameters(), lr=0.0)
        with tr.ctx:
            _logits, loss = m(x, y)
        # Same gate as train_step: scale only when fp16+CUDA scaler enabled.
        if tr._use_scaler:
            tr.scaler.scale(loss).backward()
            tr.scaler.unscale_(opt)  # compare unscaled grads to fp32
        else:
            loss.backward()
        grads = {
            n: p.grad.detach().float().clone()
            for n, p in m.named_parameters()
            if p.grad is not None
        }
        return loss.detach().float(), grads

    ref_loss, ref_g = fwd_bwd("float32")
    amp_loss, amp_g = fwd_bwd(amp_name)

    atol, rtol = TRAIN_AMP_ATOL[amp_dtype], TRAIN_AMP_RTOL[amp_dtype]
    g_atol, g_rtol = TRAIN_AMP_GRAD_ATOL[amp_dtype], TRAIN_AMP_GRAD_RTOL[amp_dtype]
    assert torch.allclose(ref_loss, amp_loss, rtol=rtol, atol=atol), (
        f"{amp_name} loss abs={(ref_loss - amp_loss).abs().item()}"
    )
    assert ref_g.keys() == amp_g.keys()
    for n in ref_g:
        d = (ref_g[n] - amp_g[n]).abs().max().item()
        assert torch.allclose(ref_g[n], amp_g[n], rtol=g_rtol, atol=g_atol), (
            f"{amp_name} grad {n} max_diff={d} (atol={g_atol})"
        )
    tr.configure_amp("float32")


def test_cpu_bf16_smoke_train_step():
    """If CPU bf16 works, one train_step under BDH_AMP_DTYPE=bfloat16 succeeds."""
    if tr.device.type != "cpu":
        pytest.skip("CPU-only smoke")
    if not tr.cpu_bf16_available():
        pytest.skip("CPU bf16 unavailable")
    tr.configure_amp("bfloat16")
    assert tr.dtype == "bfloat16"
    assert tr._use_scaler is False
    cfg = _small_cfg()
    torch.manual_seed(7)
    model = bdh.BDH(cfg).to(tr.device)
    opt = tr.make_optimizer(model)
    x = torch.randint(0, cfg.vocab_size, (2, 16), device=tr.device)
    y = torch.randint(0, cfg.vocab_size, (2, 16), device=tr.device)
    loss = tr.train_step(model, opt, x, y)
    assert torch.isfinite(loss.float())
    tr.configure_amp("float32")


def test_train_step_fp32_still_default_path():
    tr.configure_amp("float32")
    cfg = _small_cfg()
    torch.manual_seed(9)
    model = bdh.BDH(cfg).to(tr.device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x = torch.randint(0, cfg.vocab_size, (2, 16), device=tr.device)
    y = torch.randint(0, cfg.vocab_size, (2, 16), device=tr.device)
    loss = tr.train_step(model, opt, x, y)
    assert loss.dtype == torch.float32
    assert torch.isfinite(loss)
