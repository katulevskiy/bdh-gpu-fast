"""opt/gen-sample: fused lm_head+sample for T=1 decode (post gen-host)."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import bdh
import bdh_baseline as baseline
from bdh_cache import CacheManager


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


def _model(cfg, seed=0):
    torch.manual_seed(seed)
    return bdh.BDH(cfg).eval()


def test_lm_head_last_into_matches_vocab_logits():
    cfg = _small_cfg()
    m = _model(cfg)
    for T in (1, 5):
        x = torch.randn(2, T, cfg.n_embd)
        ref = m._vocab_logits(x)[:, -1, :].float()
        buf = torch.empty(2, cfg.vocab_size, dtype=torch.float32)
        m._lm_head_last_into(x, buf)
        assert torch.equal(ref, buf), f"T={T} maxdiff={(ref - buf).abs().max()}"




def test_lm_head_last_into_b1_mv_matches_mm():
    """B=1 uses mv/addmv; must match B>1 mm path / vocab logits."""
    cfg = _small_cfg()
    m = _model(cfg, seed=5)
    x = torch.randn(1, 1, cfg.n_embd)
    ref = m._vocab_logits(x)[:, -1, :].float()
    buf = torch.empty(1, cfg.vocab_size, dtype=torch.float32)
    m._lm_head_last_into(x, buf)
    assert torch.allclose(ref, buf, rtol=1e-5, atol=1e-5)
    # Tied path
    cfg2 = _small_cfg()
    cfg2.tie_weights = True
    m2 = _model(cfg2, seed=6)
    x2 = torch.randn(1, 3, cfg2.n_embd)
    ref2 = m2._vocab_logits(x2)[:, -1, :].float()
    buf2 = torch.empty(1, cfg2.vocab_size, dtype=torch.float32)
    m2._lm_head_last_into(x2, buf2)
    assert torch.allclose(ref2, buf2, rtol=1e-5, atol=1e-5)


def test_forward_logits_out_t1_view():
    cfg = _small_cfg()
    m = _model(cfg)
    prompt = torch.randint(0, cfg.vocab_size, (1, 4))
    cache = CacheManager.from_config(
        cfg,
        batch_size=1,
        max_seq=8,
        device=prompt.device,
        compute_dtype=torch.float32,
    )
    with torch.inference_mode():
        m(prompt, cache=cache)
        buf = torch.empty(1, cfg.vocab_size, dtype=torch.float32)
        idx = prompt[:, -1:]
        logits, loss = m(idx, cache=cache, logits_out=buf)
        assert loss is None
        assert logits.shape == (1, cfg.vocab_size)
        assert logits.data_ptr() == buf.data_ptr()
        assert torch.equal(logits, buf)


def test_generate_tokens_match_tip_style_sampling():
    """Default temp=1 / no top_k: same multinomial draws as gen-host tip loop."""
    cfg = _small_cfg()
    m = _model(cfg, seed=0)
    prompt = torch.randint(0, cfg.vocab_size, (1, 4))

    def tip_style(idx, max_new_tokens):
        B, prompt_len = idx.size()
        max_seq = prompt_len + max_new_tokens
        device = idx.device
        cache = CacheManager.from_config(
            cfg, batch_size=B, max_seq=max_seq, device=device, compute_dtype=torch.float32
        )
        m.attn.ensure_rope_table(max_seq, device)
        out = torch.empty(B, max_seq, dtype=idx.dtype, device=device)
        out[:, :prompt_len].copy_(idx)
        with torch.inference_mode():
            logits, _ = m(idx, cache=cache)
            for t in range(max_new_tokens):
                step = logits[:, -1, :]
                if step.dtype != torch.float32:
                    step = step.float()
                probs = F.softmax(step, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
                out[:, prompt_len + t] = idx_next[:, 0]
                logits, _ = m(idx_next, cache=cache)
        return out

    for seed in (99, 7):
        torch.manual_seed(seed)
        a = tip_style(prompt.clone(), 8)
        torch.manual_seed(seed)
        b = m.generate(prompt.clone(), max_new_tokens=8, temperature=1.0)
        assert torch.equal(a, b)


def test_generate_matches_baseline_twin():
    cfg = _small_cfg()
    torch.manual_seed(11)
    m_base = baseline.BDH(cfg).eval()
    m_opt = bdh.BDH(cfg).eval()
    m_opt.load_state_dict(m_base.state_dict())
    prompt = torch.randint(0, cfg.vocab_size, (1, 4))
    torch.manual_seed(99)
    g1 = m_base.generate(prompt.clone(), max_new_tokens=8, temperature=1.0)
    torch.manual_seed(99)
    g2 = m_opt.generate(prompt.clone(), max_new_tokens=8, temperature=1.0)
    assert torch.equal(g1, g2)


def test_generate_zero_torch_cat():
    cfg = _small_cfg()
    m = _model(cfg)
    prompt = torch.randint(0, cfg.vocab_size, (1, 4))
    cats = {"n": 0}
    orig = torch.cat

    def hooked(*a, **k):
        cats["n"] += 1
        return orig(*a, **k)

    torch.cat = hooked  # type: ignore[assignment]
    try:
        with torch.inference_mode():
            out = m.generate(prompt.clone(), max_new_tokens=6, temperature=1.0)
    finally:
        torch.cat = orig  # type: ignore[assignment]
    assert out.shape == (1, 10)
    assert cats["n"] == 0


def test_generate_topk_runs():
    cfg = _small_cfg()
    m = _model(cfg)
    prompt = torch.randint(0, cfg.vocab_size, (1, 3))
    out = m.generate(prompt.clone(), max_new_tokens=4, temperature=0.8, top_k=8)
    assert out.shape == (1, 7)
