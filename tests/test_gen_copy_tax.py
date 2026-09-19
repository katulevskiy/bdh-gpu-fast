"""gen-copy-tax + gen-vcopy: RoPE pair store, sample out=, V/sampler probe."""

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
    """CPU profiler: copy_/gen below profile-v10 ~398; aten::cat stays 0."""
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
    # profile-v10 ~398; gen-vcopy T>1 pair store → ~394
    assert per < 420, f"expected copy_/gen < 420, got {per}"
    cat_n = next((e.count for e in prof.key_averages() if e.key == "aten::cat"), 0)
    assert cat_n == 0


def test_eager_rope_t_gt1_single_aten_copy_matches_strided_ref():
    """T>1 eager RoPE: one fp32 copy_ via _store_pairs; bit-identical to strided."""
    torch.manual_seed(2)
    v = torch.randn(2, 4, 16, 32)
    cos = torch.randn(1, 1, 16, 32)
    sin = torch.randn(1, 1, 16, 32)

    # Historical strided reference (kept inline so we do not depend on old eager).
    def _strided(v, cos, sin, out):
        ve, vo = v[..., 0::2], v[..., 1::2]
        out[..., 0::2] = ve * cos[..., 0::2] - vo * sin[..., 0::2]
        out[..., 1::2] = vo * cos[..., 1::2] + ve * sin[..., 1::2]
        return out

    buf_a = torch.zeros(2, 4, 32, 32)
    buf_b = torch.zeros(2, 4, 32, 32)
    oa = buf_a.narrow(2, 2, 16)
    ob = buf_b.narrow(2, 2, 16)
    _strided(v, cos, sin, oa)
    eager_rope_rotate(v, cos, sin, out=ob)
    assert torch.equal(oa, ob)

    out = buf_b.narrow(2, 2, 16)
    with profile(activities=[ProfilerActivity.CPU]) as prof:
        for _ in range(8):
            eager_rope_rotate(v, cos, sin, out=out)
    n = next(e.count for e in prof.key_averages() if e.key == "aten::copy_")
    cat = next((e.count for e in prof.key_averages() if e.key == "aten::cat"), 0)
    assert n == 8, n
    assert cat == 0


def test_topk_gather_out_parity_and_one_fewer_copy():
    """top_k idx_out= uses gather(out=); token parity + isolated copy_ cut."""
    torch.manual_seed(0)
    logits = torch.randn(2, 64)
    probs = torch.empty(2, 64)
    out_a = torch.empty(2, 1, dtype=torch.long)
    out_b = torch.empty(2, 1, dtype=torch.long)

    torch.manual_seed(21)
    bdh.BDH._sample_from_logits(
        logits.clone(),
        scale=None,
        do_topk=True,
        top_k_n=8,
        probs_buf=probs.clone(),
        softmax=torch.nn.functional.softmax,
        multinomial=torch.multinomial,
        idx_out=out_a,
    )
    # Manual gather+copy reference under same seed
    torch.manual_seed(21)
    values, indices = torch.topk(logits.clone(), 8, dim=-1)
    probs_k = torch.nn.functional.softmax(values, dim=-1)
    idx_k = torch.multinomial(probs_k, 1)
    out_b.copy_(indices.gather(1, idx_k))
    assert torch.equal(out_a, out_b)

    # Isolated: gather(out=) vs gather+copy_ (no multinomial noise).
    indices = torch.randint(0, 50, (4, 16))
    idx_k = torch.randint(0, 16, (4, 1))
    buf = torch.empty(4, 1, dtype=torch.long)
    with profile(activities=[ProfilerActivity.CPU]) as p_out:
        for _ in range(64):
            torch.gather(indices, 1, idx_k, out=buf)
    with profile(activities=[ProfilerActivity.CPU]) as p_copy:
        for _ in range(64):
            buf.copy_(indices.gather(1, idx_k))
    n_out = next((e.count for e in p_out.key_averages() if e.key == "aten::copy_"), 0)
    n_copy = next((e.count for e in p_copy.key_averages() if e.key == "aten::copy_"), 0)
    assert n_out < n_copy, f"gather out= {n_out} should be < gather+copy_ {n_copy}"


def test_remaining_generate_copy_ceiling_probe():
    """Probe the owned V snapshot and ATen sampler copies left after #110."""
    cfg = _cfg(n_embd=16, n_head=4, mlp_internal_dim_multiplier=4)
    cache = CacheManager.from_config(
        cfg, batch_size=1, max_seq=1, device=torch.device("cpu")
    )
    source = torch.randn(1, 1, 1, cfg.n_embd)
    _, dst_v = cache.reserve(0, 1)
    with profile(activities=[ProfilerActivity.CPU]) as p_v:
        for _ in range(8):
            dst_v.copy_(source)
    v_copy = next(e.count for e in p_v.key_averages() if e.key == "aten::copy_")
    source.zero_()
    # The packed cache must retain its own snapshot after residual x is reused.
    assert not torch.equal(dst_v, source)
    assert v_copy == 8

    probs = torch.softmax(torch.randn(1, 64), dim=-1)
    idx_out = torch.empty(1, 1, dtype=torch.long)
    with profile(activities=[ProfilerActivity.CPU]) as p_multi:
        for _ in range(8):
            torch.multinomial(probs, num_samples=1, out=idx_out)
    multi_copy = next(
        e.count for e in p_multi.key_averages() if e.key == "aten::copy_"
    )
    # Even with out=, ATen's default multinomial implementation retains
    # internal copy_ work; replacing it would change the default RNG contract.
    assert multi_copy >= 8


def test_vcopy_attribution_buckets_sum_near_total():
    """Probe: V + RoPE + multinomial explain ~all generate aten::copy_."""
    cfg = _cfg(n_layer=4, n_embd=128, n_head=4)
    torch.manual_seed(0)
    m = bdh.BDH(cfg).eval()
    idx = torch.randint(0, cfg.vocab_size, (1, 16))
    n_layer, n_new = 4, 32
    with torch.inference_mode():
        m.generate(idx, max_new_tokens=2, temperature=1.0)
        with profile(activities=[ProfilerActivity.CPU]) as prof:
            m.generate(idx, max_new_tokens=n_new, temperature=1.0)
    total = next(e.count for e in prof.key_averages() if e.key == "aten::copy_")
    cat = next((e.count for e in prof.key_averages() if e.key == "aten::cat"), 0)
    # V: one/layer/forward; RoPE: one/layer/forward after gen-vcopy; multi: 4/step
    v_writes = n_layer * (1 + n_new)
    rope = n_layer * (1 + n_new)
    multi = 4 * n_new
    prompt = 1
    accounted = v_writes + rope + multi + prompt
    # Allow small profiler noise; must stay cat-free
    assert cat == 0
    assert abs(total - accounted) <= 12, (
        f"copy_={total} accounted={accounted} "
        f"(V={v_writes} RoPE={rope} multi={multi} prompt={prompt})"
    )
