"""Compare optimized bdh.py against frozen bdh_baseline.py."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import bdh
import bdh_baseline as baseline


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


def _twin_models(cfg, seed: int = 0):
    torch.manual_seed(seed)
    m_base = baseline.BDH(cfg)
    m_opt = bdh.BDH(cfg)
    m_opt.load_state_dict(m_base.state_dict())
    m_base.eval()
    m_opt.eval()
    return m_base, m_opt


def test_forward_logits_and_loss_match():
    cfg = _small_cfg()
    m_base, m_opt = _twin_models(cfg, seed=42)
    torch.manual_seed(123)
    x = torch.randint(0, cfg.vocab_size, (3, 24))
    y = torch.randint(0, cfg.vocab_size, (3, 24))
    lb, lossb = m_base(x, y)
    lo, losso = m_opt(x, y)
    assert torch.equal(lb, lo), f"logits differ max={(lb - lo).abs().max()}"
    assert torch.equal(lossb, losso), f"loss differ {lossb} vs {losso}"


def test_attention_excludes_diagonal():
    """Original uses tril(diagonal=-1); position 0 output must be exactly 0."""
    cfg = _small_cfg()
    attn = bdh.Attention(cfg)
    N = cfg.mlp_internal_dim_multiplier * cfg.n_embd // cfg.n_head
    Q = torch.randn(2, cfg.n_head, 8, N)
    V = torch.randn(2, 1, 8, cfg.n_embd)
    out, _, _ = attn(Q, Q, V)
    assert torch.count_nonzero(out[:, :, 0, :]) == 0


def test_rope_matches_baseline():
    cfg = _small_cfg()
    N = cfg.mlp_internal_dim_multiplier * cfg.n_embd // cfg.n_head
    phases = torch.rand(1, 1, 5, N, dtype=torch.float32)
    v = torch.randn(2, cfg.n_head, 5, N)
    a = bdh.Attention.rope(phases, v)
    b = baseline.Attention.rope(phases, v)
    assert torch.allclose(a, b, rtol=0, atol=0)


def test_kv_cache_prefill_matches_full():
    cfg = _small_cfg()
    _, m_opt = _twin_models(cfg, seed=7)
    x = torch.randint(0, cfg.vocab_size, (2, 12))
    full, _ = m_opt(x)
    cache = [None] * cfg.n_layer
    cached, _ = m_opt(x, cache=cache)
    assert torch.equal(full, cached)


def test_kv_cache_tokenwise_matches_full():
    cfg = _small_cfg()
    _, m_opt = _twin_models(cfg, seed=9)
    x = torch.randint(0, cfg.vocab_size, (1, 10))
    full, _ = m_opt(x)
    cache = [None] * cfg.n_layer
    parts = []
    for t in range(x.size(1)):
        logits, _ = m_opt(x[:, t : t + 1], cache=cache)
        parts.append(logits)
    tok = torch.cat(parts, dim=1)
    assert torch.allclose(full, tok, rtol=1e-5, atol=1e-5)


def test_generate_matches_baseline_greedyish():
    """Same weights + same RNG → same sampled tokens."""
    cfg = _small_cfg(n_layer=2)
    m_base, m_opt = _twin_models(cfg, seed=11)
    prompt = torch.randint(0, cfg.vocab_size, (1, 4))
    torch.manual_seed(99)
    g1 = m_base.generate(prompt.clone(), max_new_tokens=8, temperature=1.0)
    torch.manual_seed(99)
    g2 = m_opt.generate(prompt.clone(), max_new_tokens=8, temperature=1.0)
    assert torch.equal(g1, g2)


def test_train_mode_dropout_still_runs():
    cfg = _small_cfg(dropout=0.5)
    m = bdh.BDH(cfg)
    m.train()
    x = torch.randint(0, cfg.vocab_size, (2, 8))
    logits, loss = m(x, x)
    assert logits.shape == (2, 8, cfg.vocab_size)
    assert loss.ndim == 0
    loss.backward()


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
