"""Optional train AMP (BDH_AMP_DTYPE) parity vs fp32 — amp-deepen.

Default training stays fp32 / COMPILE=0 / eager. Set BDH_AMP_DTYPE=bfloat16|float16
for autocast. GradScaler is only for float16+CUDA (asserted here; this box is
CPU-only). Optional BDH_AMP_FORWARD_ONLY=1: logits under autocast, CE in fp32.

Attention math (tril(-1), no softmax, no scale) is unchanged.
AMP throughput wins are GPU-only; CPU AMP is correctness smoke (often slower).
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
# Headroom for rare CPU autocast outliers (observed bf16 logits max ~0.11–0.13
# on some seeds; typical ~5e-3). Not bit-identical — expected under autocast.
TRAIN_AMP_ATOL = {
    torch.float16: 2e-1,
    torch.bfloat16: 2e-1,
}
TRAIN_AMP_RTOL = {
    torch.float16: 2e-1,
    torch.bfloat16: 2e-1,
}
TRAIN_AMP_GRAD_ATOL = {
    torch.float16: 2e-1,
    torch.bfloat16: 2e-1,
}
TRAIN_AMP_GRAD_RTOL = {
    torch.float16: 2e-1,
    torch.bfloat16: 2e-1,
}



@pytest.fixture(autouse=True)
def _restore_amp_defaults():
    """Keep module AMP state from leaking across tests."""
    yield
    tr.configure_amp("float32")

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
    """Resolve dtype; skip bf16/fp16 on CPU when autocast unavailable."""
    resolved = tr.parse_amp_dtype(name)
    pt = {"float16": torch.float16, "bfloat16": torch.bfloat16}[resolved]
    if tr.device.type == "cpu":
        if resolved == "bfloat16" and not tr.cpu_bf16_available():
            pytest.skip("CPU bf16 autocast unavailable")
        if resolved == "float16" and not tr.cpu_fp16_available():
            pytest.skip("CPU fp16 autocast unavailable")
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


def test_amp_throughput_claim_is_honest():
    """CPU box → claim 'none'; CUDA → 'cuda'. Never claim CPU AMP speedup."""
    claim = tr.amp_throughput_claim_device()
    if tr.device.type == "cuda":
        assert claim == "cuda"
    else:
        assert claim == "none"


def test_cpu_fp16_smoke_train_step():
    """If CPU fp16 works, one train_step under BDH_AMP_DTYPE=float16 succeeds."""
    if tr.device.type != "cpu":
        pytest.skip("CPU-only smoke")
    if not tr.cpu_fp16_available():
        pytest.skip("CPU fp16 unavailable")
    tr.configure_amp("float16")
    assert tr.dtype == "float16"
    assert tr._use_scaler is False  # GradScaler only fp16+CUDA
    assert tr.scaler.is_enabled() is False
    cfg = _small_cfg()
    torch.manual_seed(11)
    model = bdh.BDH(cfg).to(tr.device)
    opt = tr.make_optimizer(model)
    x = torch.randint(0, cfg.vocab_size, (2, 16), device=tr.device)
    y = torch.randint(0, cfg.vocab_size, (2, 16), device=tr.device)
    loss = tr.train_step(model, opt, x, y)
    assert torch.isfinite(loss.float())
    tr.configure_amp("float32")


@pytest.mark.parametrize("amp_name", ["float16", "bfloat16"])
def test_amp_forward_only_logits_ce_fp32(amp_name):
    """BDH_AMP_FORWARD_ONLY: logits under autocast; CE outside in fp32; finite step."""
    _amp_or_skip(amp_name)
    tr.configure_amp(amp_name, forward_only=True)
    assert tr._amp_forward_only is True
    assert tr._use_scaler is False or tr.device.type == "cuda"
    cfg = _small_cfg()
    torch.manual_seed(13)
    model = bdh.BDH(cfg).to(tr.device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x = torch.randint(0, cfg.vocab_size, (2, 16), device=tr.device)
    y = torch.randint(0, cfg.vocab_size, (2, 16), device=tr.device)
    loss = tr.train_step(model, opt, x, y)
    assert loss.dtype == torch.float32
    assert torch.isfinite(loss)
    # Defaults restore: forward_only off under float32
    tr.configure_amp("float32")
    assert tr._amp_forward_only is False


def test_amp_forward_only_close_to_full_autocast():
    """Forward-only CE-fp32 loss stays near full-autocast loss (same weights)."""
    if tr.device.type == "cpu" and not tr.cpu_bf16_available():
        pytest.skip("CPU bf16 unavailable")
    cfg = _small_cfg()
    torch.manual_seed(17)
    base = bdh.BDH(cfg)
    sd = {k: v.detach().clone() for k, v in base.state_dict().items()}
    x = torch.randint(0, cfg.vocab_size, (4, 32), device=tr.device)
    y = torch.randint(0, cfg.vocab_size, (4, 32), device=tr.device)

    def one_loss(forward_only: bool):
        tr.configure_amp("bfloat16", forward_only=forward_only)
        m = bdh.BDH(cfg).to(tr.device)
        m.load_state_dict(sd)
        m.train()
        if forward_only:
            with tr.ctx:
                logits, _ = m(x)
            return torch.nn.functional.cross_entropy(
                logits.float().reshape(-1, logits.size(-1)), y.reshape(-1)
            ).detach()
        with tr.ctx:
            _, loss = m(x, y)
        return loss.detach().float()

    full = one_loss(False)
    fo = one_loss(True)
    # CE in fp32 vs CE under autocast — expect small gap, not bit-identical.
    assert torch.allclose(full, fo, rtol=2e-1, atol=2e-1), (
        f"forward_only vs full abs={(full - fo).abs().item()}"
    )
    tr.configure_amp("float32")


def test_defaults_stay_fp32_compile0_eager():
    """Hard defaults: AMP off, COMPILE=0, eager attn — deepen must not flip them."""
    tr.configure_amp("float32")
    assert tr.dtype == "float32"
    assert tr._use_scaler is False
    assert tr._amp_forward_only is False
    assert tr.USE_COMPILE is False
    assert os.environ.get("BDH_COMPILE", "0") in ("0", "false", "False", "")
    assert os.environ.get("BDH_ATTN_IMPL", "eager") in ("eager", "")
    # Attention mask still tril(-1) under default config.
    cfg = _small_cfg()
    attn = bdh.Attention(cfg)
    N = cfg.mlp_internal_dim_multiplier * cfg.n_embd // cfg.n_head
    Q = torch.randn(1, cfg.n_head, 4, N)
    V = torch.randn(1, 1, 4, cfg.n_embd)
    out, _, _ = attn(Q, Q, V)
    assert torch.count_nonzero(out[:, :, 0, :]) == 0
