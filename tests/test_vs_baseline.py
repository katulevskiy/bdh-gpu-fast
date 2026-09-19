"""Compare optimized bdh.py vs frozen bdh_baseline.py for training forward."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import bdh
import bdh_baseline as baseline


def _cfg(**kwargs) -> bdh.BDHConfig:
    defaults = dict(
        n_layer=3,
        n_embd=64,
        n_head=4,
        mlp_internal_dim_multiplier=8,
        dropout=0.0,
        vocab_size=256,
    )
    defaults.update(kwargs)
    return bdh.BDHConfig(**defaults)


def _twin(cfg: bdh.BDHConfig, seed: int = 0):
    """Identical weights: baseline init, then copy into optimized model."""
    torch.manual_seed(seed)
    m_base = baseline.BDH(cfg)
    m_opt = bdh.BDH(cfg)
    missing, unexpected = m_opt.load_state_dict(m_base.state_dict(), strict=True)
    assert not missing and not unexpected
    return m_base, m_opt


def test_training_forward_logits_match_dropout_zero():
    """With dropout=0, training-mode forward must match baseline exactly."""
    cfg = _cfg(dropout=0.0)
    m_base, m_opt = _twin(cfg, seed=101)
    m_base.train()
    m_opt.train()

    torch.manual_seed(202)
    x = torch.randint(0, cfg.vocab_size, (4, 32))
    y = torch.randint(0, cfg.vocab_size, (4, 32))

    lb, lossb = m_base(x, y)
    lo, losso = m_opt(x, y)

    assert lb.shape == lo.shape == (4, 32, cfg.vocab_size)
    assert torch.equal(lb, lo), (
        f"train logits differ max abs={(lb - lo).abs().max().item()}"
    )
    assert torch.equal(lossb, losso), f"loss {lossb.item()} vs {losso.item()}"


def test_eval_forward_logits_and_loss_match():
    cfg = _cfg(dropout=0.0)
    m_base, m_opt = _twin(cfg, seed=7)
    m_base.eval()
    m_opt.eval()

    torch.manual_seed(8)
    x = torch.randint(0, cfg.vocab_size, (3, 24))
    y = torch.randint(0, cfg.vocab_size, (3, 24))
    lb, lossb = m_base(x, y)
    lo, losso = m_opt(x, y)
    assert torch.equal(lb, lo)
    assert torch.equal(lossb, losso)


def test_backward_grads_match_dropout_zero():
    """Same train forward+backward with dropout=0 → matching parameter grads."""
    cfg = _cfg(dropout=0.0, n_layer=2)
    m_base, m_opt = _twin(cfg, seed=33)
    m_base.train()
    m_opt.train()

    torch.manual_seed(34)
    x = torch.randint(0, cfg.vocab_size, (2, 16))
    y = torch.randint(0, cfg.vocab_size, (2, 16))

    _, lossb = m_base(x, y)
    _, losso = m_opt(x, y)
    lossb.backward()
    losso.backward()

    for (nb, pb), (no, po) in zip(m_base.named_parameters(), m_opt.named_parameters()):
        assert nb == no
        assert pb.grad is not None and po.grad is not None
        assert torch.allclose(pb.grad, po.grad, rtol=1e-5, atol=1e-5), (
            f"grad mismatch on {nb}: max={(pb.grad - po.grad).abs().max().item()}"
        )


def test_attention_module_matches_baseline_cold_path():
    """Cold (no cache) Attention output must match baseline Attention."""
    cfg = _cfg()
    torch.manual_seed(5)
    a_opt = bdh.Attention(cfg)
    a_base = baseline.Attention(cfg)
    # freqs buffer is deterministic from config; copy anyway for safety
    a_opt.load_state_dict(a_base.state_dict())

    N = cfg.mlp_internal_dim_multiplier * cfg.n_embd // cfg.n_head
    Q = torch.randn(2, cfg.n_head, 12, N)
    V = torch.randn(2, 1, 12, cfg.n_embd)

    out_base = a_base(Q, Q, V)
    out_opt, _, _ = a_opt(Q, Q, V)
    assert torch.allclose(out_base, out_opt, rtol=0, atol=0), (
        f"attn max diff={(out_base - out_opt).abs().max().item()}"
    )


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception as e:
            failed += 1
            print(f"FAIL {fn.__name__}: {e}")
    if failed:
        raise SystemExit(1)
    print(f"All {len(tests)} tests passed.")
