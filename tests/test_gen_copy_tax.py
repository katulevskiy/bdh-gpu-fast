"""gen-copy-tax-v1: RoPE complex-pair store + sample-into-out — parity + copy_ gate."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import bdh
import bdh_baseline as baseline
from bdh_cache import CacheManager
from kernels.rope import (
    _alloc_rope_out,
    _store_pairs,
    eager_rope_rotate,
    rope_rotate_paired,
    rope_rotate_t1,
)


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
    torch.manual_seed(seed)
    m_opt = bdh.BDH(cfg).eval()
    torch.manual_seed(seed)
    m_base = baseline.BDH(cfg).eval()
    m_base.load_state_dict(m_opt.state_dict())
    return m_base, m_opt


def _setitem_store(y0, y1, v, out):
    """Historical empty+pair setitem reference."""
    if out is None:
        out = _alloc_rope_out(v)
    op = out.reshape(*v.shape[:-1], -1, 2)
    op[..., 0] = y0
    op[..., 1] = y1
    return out


def test_store_pairs_fp32_matches_setitem_contig_and_narrow():
    torch.manual_seed(0)
    v = torch.randn(2, 4, 1, 32)
    y0 = torch.randn(2, 4, 1, 16)
    y1 = torch.randn(2, 4, 1, 16)
    # contiguous out
    a = torch.empty_like(v)
    b = torch.empty_like(v)
    _setitem_store(y0, y1, v, a)
    _store_pairs(y0, y1, v, b)
    assert torch.equal(a, b)
    # out=None
    assert torch.equal(_store_pairs(y0, y1, v, None), _setitem_store(y0, y1, v, None))
    # non-contig cache narrow
    buf_a = torch.zeros(2, 4, 16, 32)
    buf_b = torch.zeros(2, 4, 16, 32)
    oa = buf_a.narrow(2, 3, 1)
    ob = buf_b.narrow(2, 3, 1)
    _setitem_store(y0, y1, v, oa)
    _store_pairs(y0, y1, v, ob)
    assert torch.equal(oa, ob)


def test_store_pairs_fp32_single_aten_copy():
    """fp32 out= path should issue one aten::copy_ (complex pack), not two."""
    torch.manual_seed(1)
    v = torch.randn(1, 4, 1, 64)
    y0 = torch.randn(1, 4, 1, 32)
    y1 = torch.randn(1, 4, 1, 32)
    buf = torch.zeros(1, 4, 8, 64)
    out = buf.narrow(2, 2, 1)
    with profile(activities=[ProfilerActivity.CPU]) as prof:
        for _ in range(32):
            _store_pairs(y0, y1, v, out)
    n = next(e.count for e in prof.key_averages() if e.key == "aten::copy_")
    cat = next((e.count for e in prof.key_averages() if e.key == "aten::cat"), 0)
    assert n == 32, n  # one per call
    assert cat == 0


def test_rope_t1_paired_match_eager():
    cfg = _cfg()
    attn = bdh.Attention(cfg)
    N = cfg.mlp_internal_dim_multiplier * cfg.n_embd // cfg.n_head
    device = torch.device("cpu")
    attn.ensure_rope_table(16, device)
    cos, sin = attn.rope_cos_sin(1, 4, device)
    torch.manual_seed(3)
    v = torch.randn(2, cfg.n_head, 1, N)
    out = torch.empty_like(v)
    a = rope_rotate_t1(v, cos, sin)
    b = rope_rotate_t1(v, cos, sin, out=out)
    c = eager_rope_rotate(v.contiguous(), cos, sin)
    assert torch.equal(a, b)
    assert torch.equal(a, c)
    paired = attn.t1_cis_pairs(4, device)
    assert paired is not None
    assert torch.equal(a, rope_rotate_paired(v, paired[0], paired[1]))


def test_sample_idx_out_writes_and_rng_parity():
    """idx_out= multinomial path matches return-value path under same seed."""
    torch.manual_seed(0)
    logits = torch.randn(2, 50)
    probs = torch.empty(2, 50)
    # return path
    torch.manual_seed(11)
    a = bdh.BDH._sample_from_logits(
        logits.clone(),
        scale=None,
        do_topk=False,
        top_k_n=0,
        probs_buf=probs.clone(),
        softmax=torch.nn.functional.softmax,
        multinomial=torch.multinomial,
    )
    # out= path
    out = torch.empty(2, 1, dtype=torch.long)
    torch.manual_seed(11)
    b = bdh.BDH._sample_from_logits(
        logits.clone(),
        scale=None,
        do_topk=False,
        top_k_n=0,
        probs_buf=probs.clone(),
        softmax=torch.nn.functional.softmax,
        multinomial=torch.multinomial,
        idx_out=out,
    )
    assert b is out
    assert torch.equal(a, out)


def test_generate_parity_vs_baseline_seeded():
    cfg = _cfg(n_layer=2, n_embd=64)
    m_base, m_opt = _twin(cfg, seed=42)
    prompt = torch.randint(
        0, cfg.vocab_size, (1, 5), generator=torch.Generator().manual_seed(9)
    )
    with torch.inference_mode():
        torch.manual_seed(0)
        out_o = m_opt.generate(prompt.clone(), max_new_tokens=6, temperature=1.0)
        torch.manual_seed(0)
        out_b = m_base.generate(prompt.clone(), max_new_tokens=6, temperature=1.0)
    assert torch.equal(out_o, out_b)


def test_generate_cache_continuity_across_steps():
    cfg = _cfg(n_layer=2)
    torch.manual_seed(0)
    m = bdh.BDH(cfg).eval()
    idx = torch.randint(0, cfg.vocab_size, (1, 6))
    max_new = 8
    cache = CacheManager.from_config(
        cfg, batch_size=1, max_seq=idx.size(1) + max_new, device=idx.device
    )
    with torch.inference_mode():
        m(idx, cache=cache)
        assert cache.seq_len == idx.size(1)
        kr0, v0 = cache.get_past(0)
        assert kr0 is not None and kr0.size(2) == idx.size(1)
        for t in range(max_new):
            tok = torch.randint(0, cfg.vocab_size, (1, 1))
            m(tok, cache=cache)
            assert cache.seq_len == idx.size(1) + t + 1
        kr1, v1 = cache.get_past(0)
        assert kr1.size(2) == idx.size(1) + max_new
        assert torch.equal(kr1[:, :, : idx.size(1)], kr0)
        assert torch.equal(v1[:, :, : idx.size(1)], v0)


def test_generate_aten_copy_below_profile_v9_and_cat_zero():
    """CPU profiler: copy_/gen well below profile-v9 ~558; aten::cat stays 0."""
    cfg = _cfg(n_layer=4, n_embd=128, n_head=4)
    torch.manual_seed(0)
    m = bdh.BDH(cfg).eval()
    idx = torch.randint(0, cfg.vocab_size, (1, 16))
    with torch.inference_mode():
        m.generate(idx, max_new_tokens=4, temperature=1.0)
        with profile(activities=[ProfilerActivity.CPU]) as prof:
            for _ in range(3):
                m.generate(idx, max_new_tokens=32, temperature=1.0)
    n = next(e.count for e in prof.key_averages() if e.key == "aten::copy_")
    per = n / 3
    # profile-v9 ~558; complex RoPE store + sample-into-out → ~398
    assert per < 480, f"expected copy_/gen < 480, got {per}"
    cat_n = next((e.count for e in prof.key_averages() if e.key == "aten::cat"), 0)
    assert cat_n == 0
