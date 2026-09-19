"""Layout-v3: CPU probe for the remaining eager layout materializations."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import bdh


def _cfg(**kwargs) -> bdh.BDHConfig:
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


def _profile_call(fn):
    with profile(
        activities=[ProfilerActivity.CPU],
        record_shapes=True,
    ) as prof:
        fn()
    return prof


def _count(prof, key: str) -> int:
    return next((event.count for event in prof.key_averages() if event.key == key), 0)


def _event_shapes(prof, key: str) -> list[tuple[int, ...]]:
    shapes = []
    for event in prof.events():
        if event.key != key or not event.input_shapes:
            continue
        shape = event.input_shapes[0]
        if isinstance(shape, list) and all(isinstance(dim, int) for dim in shape):
            shapes.append(tuple(shape))
    return shapes


def _clone_shapes(prof) -> set[tuple[int, ...]]:
    return set(_event_shapes(prof, "aten::clone"))


def test_cpu_probe_documents_remaining_layout_materializations():
    """Keep the safe view/layout boundary explicit without changing defaults.

    The eval encoder cache intentionally pays one ``(nh*N,D)`` materialization
    when it is warmed.  Once warm, default eager forward/generate has no explicit
    ``aten::contiguous`` or ``aten::cat``; CPU eager attention still reports two
    clone-backed layout materializations per layer.  Those are backend/BLAS
    behavior, not a license to flip the activation layout globally.
    """
    cfg = _cfg()
    model = bdh.BDH(cfg)

    # Layout-v2's cached F.linear weight is the one intentional contiguous
    # materialization.  Keep it outside the profiled forward hot path.
    warm = _profile_call(model.eval)
    nh = cfg.n_head
    D = cfg.n_embd
    N = cfg.mlp_internal_dim_multiplier * D // nh
    assert model._encoder_w_lin is not None
    assert model._encoder_w_lin.is_contiguous()
    assert _count(warm, "aten::clone") == 1
    assert _count(warm, "aten::copy_") == 1
    assert (nh, N, D) in _clone_shapes(warm)

    idx = torch.randint(0, cfg.vocab_size, (2, 8))
    with torch.inference_mode():
        forward = _profile_call(lambda: model(idx))
        generate = _profile_call(
            lambda: model.generate(idx[:, :5], max_new_tokens=2, temperature=1.0)
        )

    # Default eager remains cat-free and does not call explicit contiguous in
    # either forward or packed generate.  The two clone shapes are the current
    # CPU probe signature for eager score×V / broadcast handling, not a GPU claim.
    assert _count(forward, "aten::contiguous") == 0
    assert _count(forward, "aten::cat") == 0
    assert _count(forward, "aten::clone") == 2 * cfg.n_layer
    assert (2, nh, 8, D) in _clone_shapes(forward)
    assert (nh, 2, 8, D, 1) in _clone_shapes(forward)
    assert _count(generate, "aten::cat") == 0
    # Remaining default-generate contiguous calls are inside ATen sampling:
    # softmax materializes the reused probability buffer once, and multinomial's
    # index_select materializes one (B,) index result per sampled token.  They
    # are not model activation layout flips and need a sampler/RNG change to cut.
    generate_contig_shapes = _event_shapes(generate, "aten::contiguous")
    assert generate_contig_shapes.count((2, cfg.vocab_size)) == 1
    assert generate_contig_shapes.count((2,)) == 2


def test_cpu_probe_covers_scaled_and_topk_sampler_signatures():
    """Cover sampler-owned layout branches without changing model defaults."""
    cfg = _cfg()
    model = bdh.BDH(cfg).eval()
    idx = torch.randint(0, cfg.vocab_size, (2, 8))

    # Non-default sampler options own the first-step logits buffer.  Probe the
    # top-k boundary, narrow path, k==V fallback, and overflow clamp, not just
    # scaled sampling.
    cases = (
        ("scaled", dict(temperature=0.7)),
        ("topk-one", dict(temperature=0.7, top_k=1)),
        ("topk-narrow", dict(temperature=0.7, top_k=8)),
        ("topk-full", dict(temperature=0.7, top_k=cfg.vocab_size)),
        ("topk-overflow", dict(temperature=0.7, top_k=cfg.vocab_size + 1)),
    )
    with torch.inference_mode():
        for name, kwargs in cases:
            prof = _profile_call(
                lambda kwargs=kwargs: model.generate(
                    idx[:, :5], max_new_tokens=2, **kwargs
                )
            )
            shapes = _event_shapes(prof, "aten::contiguous")
            assert _count(prof, "aten::cat") == 0, name
            # Both the prefill sample and the reused decode logits buffer stay
            # on the sampler-owned path: one (B,) result per generated token.
            assert shapes.count((2,)) == 2, (name, shapes)
            assert (2, cfg.vocab_size) not in shapes, (name, shapes)


def test_sampler_idx_out_accepts_noncontiguous_decode_narrow():
    """Sampler outputs must support the strided one-column decode view."""
    torch.manual_seed(0)
    logits = torch.randn(2, 32)
    cases = (
        ("multinomial", dict(scale=None, do_topk=False, top_k_n=0)),
        ("topk", dict(scale=0.7, do_topk=True, top_k_n=8)),
    )

    for name, kwargs in cases:
        destination = torch.empty(2, 3, dtype=torch.long)
        idx_out = destination[:, 1:2]
        assert not idx_out.is_contiguous(), name

        torch.manual_seed(17)
        got = bdh.BDH._sample_from_logits(
            logits.clone(),
            **kwargs,
            probs_buf=torch.empty_like(logits),
            softmax=torch.nn.functional.softmax,
            multinomial=torch.multinomial,
            idx_out=idx_out,
        )
        assert got is idx_out

        torch.manual_seed(17)
        ref = bdh.BDH._sample_from_logits(
            logits.clone(),
            **kwargs,
            probs_buf=torch.empty_like(logits),
            softmax=torch.nn.functional.softmax,
            multinomial=torch.multinomial,
        )
        assert torch.equal(got, ref), name
        assert torch.equal(destination[:, 1:2], ref), name


def test_sampler_idx_out_accepts_noncontiguous_full_vocab_topk_fallback():
    """The full-vocab top-k fallback must preserve the strided output contract."""
    torch.manual_seed(0)
    logits = torch.randn(2, 32)

    # k == V takes the fallback branch; overflow exercises its clamp to V.
    for name, top_k_n in (("full", 32), ("overflow", 40)):
        destination = torch.empty(2, 3, dtype=torch.long)
        idx_out = destination[:, 1:2]
        assert idx_out.stride() == (3, 1), name

        torch.manual_seed(17)
        got = bdh.BDH._sample_from_logits(
            logits.clone(),
            scale=0.7,
            do_topk=True,
            top_k_n=top_k_n,
            probs_buf=torch.empty_like(logits),
            softmax=torch.nn.functional.softmax,
            multinomial=torch.multinomial,
            idx_out=idx_out,
        )
        assert got is idx_out

        torch.manual_seed(17)
        ref = bdh.BDH._sample_from_logits(
            logits.clone(),
            scale=0.7,
            do_topk=True,
            top_k_n=top_k_n,
            probs_buf=torch.empty_like(logits),
            softmax=torch.nn.functional.softmax,
            multinomial=torch.multinomial,
        )
        assert torch.equal(got, ref), name
        assert torch.equal(destination[:, 1:2], ref), name


def test_sampler_idx_out_accepts_noncontiguous_column_stride():
    """Sampler output may be a (B, 1) view with a strided last dimension."""
    torch.manual_seed(0)
    logits = torch.randn(2, 32)
    cases = (
        ("multinomial", dict(scale=None, do_topk=False, top_k_n=0)),
        ("topk", dict(scale=0.7, do_topk=True, top_k_n=8)),
        ("topk-full", dict(scale=0.7, do_topk=True, top_k_n=32)),
    )

    for name, kwargs in cases:
        destination = torch.empty(2, 3, 2, dtype=torch.long)
        idx_out = destination[:, 1:2, 0]
        assert idx_out.shape == (2, 1), name
        assert idx_out.stride() == (6, 2), name

        torch.manual_seed(17)
        got = bdh.BDH._sample_from_logits(
            logits.clone(),
            **kwargs,
            probs_buf=torch.empty_like(logits),
            softmax=torch.nn.functional.softmax,
            multinomial=torch.multinomial,
            idx_out=idx_out,
        )
        assert got is idx_out

        torch.manual_seed(17)
        ref = bdh.BDH._sample_from_logits(
            logits.clone(),
            **kwargs,
            probs_buf=torch.empty_like(logits),
            softmax=torch.nn.functional.softmax,
            multinomial=torch.multinomial,
        )
        assert torch.equal(got, ref), name
        assert torch.equal(destination[:, 1:2, 0], ref), name


def test_sampler_idx_out_does_not_clobber_strided_neighbors():
    """Sampler writes must stay within the requested non-contiguous view."""
    torch.manual_seed(0)
    logits = torch.randn(2, 32)
    cases = (
        ("multinomial", dict(scale=None, do_topk=False, top_k_n=0)),
        ("topk-one", dict(scale=0.7, do_topk=True, top_k_n=1)),
        ("topk-narrow", dict(scale=0.7, do_topk=True, top_k_n=8)),
        ("topk-full", dict(scale=0.7, do_topk=True, top_k_n=32)),
        ("topk-overflow", dict(scale=0.7, do_topk=True, top_k_n=40)),
    )

    for name, kwargs in cases:
        destination = torch.full((2, 3, 3), -123, dtype=torch.long)
        before = destination.clone()
        idx_out = destination[:, 1:2, 1]
        assert idx_out.shape == (2, 1), name
        assert idx_out.stride() == (9, 3), name

        torch.manual_seed(17)
        got = bdh.BDH._sample_from_logits(
            logits.clone(),
            **kwargs,
            probs_buf=torch.empty_like(logits),
            softmax=torch.nn.functional.softmax,
            multinomial=torch.multinomial,
            idx_out=idx_out,
        )
        assert got is idx_out

        target = torch.zeros_like(destination, dtype=torch.bool)
        target[:, 1:2, 1] = True
        assert torch.equal(
            destination.masked_select(~target), before.masked_select(~target)
        ), name

        torch.manual_seed(17)
        ref = bdh.BDH._sample_from_logits(
            logits.clone(),
            **kwargs,
            probs_buf=torch.empty_like(logits),
            softmax=torch.nn.functional.softmax,
            multinomial=torch.multinomial,
        )
        assert torch.equal(got, ref), name
        assert torch.equal(destination[:, 1:2, 1], ref), name


def test_sampler_idx_out_accepts_unit_batch_stride_view():
    """Sampler output may have unit batch stride and a strided singleton axis."""
    torch.manual_seed(0)
    logits = torch.randn(2, 32)
    cases = (
        ("multinomial", dict(scale=None, do_topk=False, top_k_n=0)),
        ("topk-narrow", dict(scale=0.7, do_topk=True, top_k_n=8)),
        ("topk-full", dict(scale=0.7, do_topk=True, top_k_n=32)),
    )

    for name, kwargs in cases:
        destination = torch.full((1, 3, 2), -123, dtype=torch.long)
        before = destination.clone()
        # Transpose a middle-row view so the (B, 1) output has batch stride 1
        # and singleton-axis stride 6, unlike the existing narrow contracts.
        idx_out = destination[:, 1, :].transpose(0, 1)
        assert idx_out.shape == (2, 1), name
        assert idx_out.stride() == (1, 6), name

        torch.manual_seed(17)
        got = bdh.BDH._sample_from_logits(
            logits.clone(),
            **kwargs,
            probs_buf=torch.empty_like(logits),
            softmax=torch.nn.functional.softmax,
            multinomial=torch.multinomial,
            idx_out=idx_out,
        )
        assert got is idx_out

        target = torch.zeros_like(destination, dtype=torch.bool)
        target[:, 1, :] = True
        assert torch.equal(
            destination.masked_select(~target), before.masked_select(~target)
        ), name

        torch.manual_seed(17)
        ref = bdh.BDH._sample_from_logits(
            logits.clone(),
            **kwargs,
            probs_buf=torch.empty_like(logits),
            softmax=torch.nn.functional.softmax,
            multinomial=torch.multinomial,
        )
        assert torch.equal(got, ref), name
        assert torch.equal(destination[:, 1, :].transpose(0, 1), ref), name


def test_sampler_idx_out_accepts_zero_singleton_stride_view():
    """A singleton output axis may have zero stride without aliasing batches."""
    torch.manual_seed(0)
    logits = torch.randn(2, 32)
    cases = (
        ("multinomial", dict(scale=None, do_topk=False, top_k_n=0)),
        ("topk-one", dict(scale=0.7, do_topk=True, top_k_n=1)),
        ("topk-narrow", dict(scale=0.7, do_topk=True, top_k_n=8)),
        ("topk-full", dict(scale=0.7, do_topk=True, top_k_n=32)),
        ("topk-overflow", dict(scale=0.7, do_topk=True, top_k_n=40)),
    )

    for name, kwargs in cases:
        destination = torch.full((2, 3, 2), -123, dtype=torch.long)
        before = destination.clone()
        # Expand a singleton axis before narrowing it back to (B, 1).  The
        # zero stride is harmless for the size-one axis but catches kernels
        # that assume a conventional positive output stride.
        idx_out = destination[:, 1:2, :1].expand(-1, 2, -1)[:, :1, 0]
        assert idx_out.shape == (2, 1), name
        assert idx_out.stride() == (6, 0), name

        torch.manual_seed(17)
        got = bdh.BDH._sample_from_logits(
            logits.clone(),
            **kwargs,
            probs_buf=torch.empty_like(logits),
            softmax=torch.nn.functional.softmax,
            multinomial=torch.multinomial,
            idx_out=idx_out,
        )
        assert got is idx_out

        target = torch.zeros_like(destination, dtype=torch.bool)
        target[:, 1, 0] = True
        assert torch.equal(
            destination.masked_select(~target), before.masked_select(~target)
        ), name

        torch.manual_seed(17)
        ref = bdh.BDH._sample_from_logits(
            logits.clone(),
            **kwargs,
            probs_buf=torch.empty_like(logits),
            softmax=torch.nn.functional.softmax,
            multinomial=torch.multinomial,
        )
        assert torch.equal(got, ref), name
        assert torch.equal(destination[:, 1, 0].reshape(2, 1), ref), name


def test_sampler_accepts_strided_logits_with_strided_output():
    """Decode sampler views may be strided on both input and output."""
    torch.manual_seed(0)
    logits_storage = torch.randn(2, 3, 32)
    cases = (
        ("multinomial", dict(scale=None, do_topk=False, top_k_n=0)),
        ("topk-narrow", dict(scale=0.7, do_topk=True, top_k_n=8)),
        ("topk-full", dict(scale=0.7, do_topk=True, top_k_n=32)),
        ("topk-overflow", dict(scale=0.7, do_topk=True, top_k_n=40)),
    )

    for name, kwargs in cases:
        # This is the layout of logits[:, -1, :] from a (B, T, V) prefill.
        logits = logits_storage.clone()[:, 1, :]
        assert logits.shape == (2, 32), name
        assert logits.stride() == (96, 1), name

        destination = torch.full((1, 3, 2), -123, dtype=torch.long)
        before = destination.clone()
        idx_out = destination[:, 1, :].transpose(0, 1)
        assert idx_out.shape == (2, 1), name
        assert idx_out.stride() == (1, 6), name

        torch.manual_seed(17)
        got = bdh.BDH._sample_from_logits(
            logits,
            **kwargs,
            probs_buf=torch.empty(2, 32),
            softmax=torch.nn.functional.softmax,
            multinomial=torch.multinomial,
            idx_out=idx_out,
        )
        assert got is idx_out

        target = torch.zeros_like(destination, dtype=torch.bool)
        target[:, 1, :] = True
        assert torch.equal(
            destination.masked_select(~target), before.masked_select(~target)
        ), name

        torch.manual_seed(17)
        ref = bdh.BDH._sample_from_logits(
            logits.clone(),
            **kwargs,
            probs_buf=torch.empty(2, 32),
            softmax=torch.nn.functional.softmax,
            multinomial=torch.multinomial,
        )
        assert torch.equal(got, ref), name
        assert torch.equal(destination[:, 1, :].transpose(0, 1), ref), name


def test_sampler_accepts_nonunit_vocab_stride_with_strided_output():
    """Sampler views may stride between adjacent vocabulary entries."""
    torch.manual_seed(0)
    logits_storage = torch.randn(2, 32, 2)
    logits = logits_storage[..., 0]
    assert logits.shape == (2, 32)
    assert logits.stride() == (64, 2)

    cases = (
        ("multinomial", dict(scale=None, do_topk=False, top_k_n=0)),
        ("topk-narrow", dict(scale=0.7, do_topk=True, top_k_n=8)),
        ("topk-full", dict(scale=0.7, do_topk=True, top_k_n=32)),
        ("topk-overflow", dict(scale=0.7, do_topk=True, top_k_n=40)),
    )

    for name, kwargs in cases:
        destination = torch.full((2, 3), -123, dtype=torch.long)
        before = destination.clone()
        idx_out = destination[:, 1:2]
        assert idx_out.stride() == (3, 1), name

        torch.manual_seed(17)
        got = bdh.BDH._sample_from_logits(
            logits.clone(),
            **kwargs,
            probs_buf=torch.empty(2, 32),
            softmax=torch.nn.functional.softmax,
            multinomial=torch.multinomial,
            idx_out=idx_out,
        )
        assert got is idx_out

        target = torch.zeros_like(destination, dtype=torch.bool)
        target[:, 1:2] = True
        assert torch.equal(
            destination.masked_select(~target), before.masked_select(~target)
        ), name

        torch.manual_seed(17)
        ref = bdh.BDH._sample_from_logits(
            logits.contiguous(),
            **kwargs,
            probs_buf=torch.empty(2, 32),
            softmax=torch.nn.functional.softmax,
            multinomial=torch.multinomial,
        )
        assert torch.equal(got, ref), name
        assert torch.equal(destination[:, 1:2], ref), name


def test_sampler_accepts_nonunit_vocab_stride_with_zero_stride_output():
    """Non-unit vocab views must preserve the zero-stride output contract."""
    torch.manual_seed(0)
    logits_storage = torch.randn(2, 32, 2)
    logits = logits_storage[..., 0]
    assert logits.shape == (2, 32)
    assert logits.stride() == (64, 2)

    cases = (
        ("multinomial", dict(scale=None, do_topk=False, top_k_n=0)),
        ("topk-one", dict(scale=0.7, do_topk=True, top_k_n=1)),
        ("topk-narrow", dict(scale=0.7, do_topk=True, top_k_n=8)),
        ("topk-overflow", dict(scale=0.7, do_topk=True, top_k_n=40)),
    )

    for name, kwargs in cases:
        destination = torch.full((2, 3, 2), -123, dtype=torch.long)
        before = destination.clone()
        idx_out = destination[:, 1:2, :1].expand(-1, 2, -1)[:, :1, 0]
        assert idx_out.shape == (2, 1), name
        assert idx_out.stride() == (6, 0), name

        torch.manual_seed(17)
        got = bdh.BDH._sample_from_logits(
            logits.clone(),
            **kwargs,
            probs_buf=torch.empty(2, 32),
            softmax=torch.nn.functional.softmax,
            multinomial=torch.multinomial,
            idx_out=idx_out,
        )
        assert got is idx_out

        target = torch.zeros_like(destination, dtype=torch.bool)
        target[:, 1, 0] = True
        assert torch.equal(
            destination.masked_select(~target), before.masked_select(~target)
        ), name

        torch.manual_seed(17)
        ref = bdh.BDH._sample_from_logits(
            logits.contiguous(),
            **kwargs,
            probs_buf=torch.empty(2, 32),
            softmax=torch.nn.functional.softmax,
            multinomial=torch.multinomial,
        )
        assert torch.equal(got, ref), name
        assert torch.equal(destination[:, 1, 0].reshape(2, 1), ref), name


def test_sampler_accepts_nonunit_vocab_stride_with_unit_batch_stride_output():
    """Non-unit vocab views must preserve unit-batch-stride output layout."""
    torch.manual_seed(0)
    logits_storage = torch.randn(2, 32, 2)
    logits = logits_storage[..., 0]
    assert logits.shape == (2, 32)
    assert logits.stride() == (64, 2)

    cases = (
        ("multinomial", dict(scale=None, do_topk=False, top_k_n=0)),
        ("topk-one", dict(scale=0.7, do_topk=True, top_k_n=1)),
        ("topk-narrow", dict(scale=0.7, do_topk=True, top_k_n=8)),
        ("topk-full", dict(scale=0.7, do_topk=True, top_k_n=32)),
        ("topk-overflow", dict(scale=0.7, do_topk=True, top_k_n=40)),
    )

    for name, kwargs in cases:
        destination = torch.full((1, 3, 2), -123, dtype=torch.long)
        before = destination.clone()
        idx_out = destination[:, 1, :].transpose(0, 1)
        assert idx_out.shape == (2, 1), name
        assert idx_out.stride() == (1, 6), name

        torch.manual_seed(17)
        got = bdh.BDH._sample_from_logits(
            logits.clone(),
            **kwargs,
            probs_buf=torch.empty(2, 32),
            softmax=torch.nn.functional.softmax,
            multinomial=torch.multinomial,
            idx_out=idx_out,
        )
        assert got is idx_out

        target = torch.zeros_like(destination, dtype=torch.bool)
        target[:, 1, :] = True
        assert torch.equal(
            destination.masked_select(~target), before.masked_select(~target)
        ), name

        torch.manual_seed(17)
        ref = bdh.BDH._sample_from_logits(
            logits.contiguous(),
            **kwargs,
            probs_buf=torch.empty(2, 32),
            softmax=torch.nn.functional.softmax,
            multinomial=torch.multinomial,
        )
        assert torch.equal(got, ref), name
        assert torch.equal(destination[:, 1, :].transpose(0, 1), ref), name

def test_sampler_accepts_strided_logits_and_probability_buffer():
    """Sampler scratch layout may be strided alongside decode views."""
    torch.manual_seed(0)
    logits_storage = torch.randn(2, 32, 2)
    logits = logits_storage[..., 0]
    probs_storage = torch.empty(2, 32, 2)
    probs_buf = probs_storage[..., 0]
    assert logits.stride() == (64, 2)
    assert probs_buf.stride() == (64, 2)

    cases = (
        ("multinomial", dict(scale=None, do_topk=False, top_k_n=0)),
        ("topk-narrow", dict(scale=0.7, do_topk=True, top_k_n=8)),
        ("topk-full", dict(scale=0.7, do_topk=True, top_k_n=32)),
        ("topk-overflow", dict(scale=0.7, do_topk=True, top_k_n=40)),
    )

    for name, kwargs in cases:
        destination = torch.full((1, 3, 2), -123, dtype=torch.long)
        before = destination.clone()
        idx_out = destination[:, 1, :].transpose(0, 1)
        assert idx_out.shape == (2, 1), name
        assert idx_out.stride() == (1, 6), name

        torch.manual_seed(17)
        got = bdh.BDH._sample_from_logits(
            logits.clone(),
            **kwargs,
            probs_buf=probs_buf,
            softmax=torch.nn.functional.softmax,
            multinomial=torch.multinomial,
            idx_out=idx_out,
        )
        assert got is idx_out

        target = torch.zeros_like(destination, dtype=torch.bool)
        target[:, 1, :] = True
        assert torch.equal(
            destination.masked_select(~target), before.masked_select(~target)
        ), name

        torch.manual_seed(17)
        ref = bdh.BDH._sample_from_logits(
            logits.contiguous(),
            **kwargs,
            probs_buf=torch.empty(2, 32),
            softmax=torch.nn.functional.softmax,
            multinomial=torch.multinomial,
        )
        assert torch.equal(got, ref), name
        assert torch.equal(destination[:, 1, :].transpose(0, 1), ref), name


def test_sampler_probability_buffer_preserves_strided_neighbors():
    """Full-vocab sampler writes must stay inside a strided scratch view."""
    torch.manual_seed(0)
    logits_storage = torch.randn(2, 32, 2)
    logits = logits_storage[..., 0]
    probs_storage = torch.full((2, 32, 2), -777.0)
    probs_buf = probs_storage[..., 0]
    before = probs_storage.clone()
    assert probs_buf.stride() == (64, 2)

    # These branches write the full-vocab softmax into probs_buf.  The narrow
    # top-k branch intentionally allocates k-space probabilities instead.
    cases = (
        ("multinomial", dict(scale=None, do_topk=False, top_k_n=0)),
        ("topk-full", dict(scale=0.7, do_topk=True, top_k_n=32)),
        ("topk-overflow", dict(scale=0.7, do_topk=True, top_k_n=40)),
    )

    for name, kwargs in cases:
        probs_storage.copy_(before)
        destination = torch.empty(2, 1, dtype=torch.long)
        torch.manual_seed(17)
        got = bdh.BDH._sample_from_logits(
            logits.clone(),
            **kwargs,
            probs_buf=probs_buf,
            softmax=torch.nn.functional.softmax,
            multinomial=torch.multinomial,
            idx_out=destination,
        )
        assert got is destination, name

        # Channel 1 is the neighboring storage excluded by the scratch view.
        assert torch.equal(probs_storage[..., 1], before[..., 1]), name


def test_sampler_probability_buffer_preserves_padded_neighbors():
    """Full-vocab sampling must honor an offset, padded scratch view."""
    torch.manual_seed(0)
    logits = torch.randn(2, 32)
    probs_storage = torch.full((2, 34), -777.0)
    probs_buf = probs_storage[:, 1:-1]
    before = probs_storage.clone()
    assert probs_buf.shape == (2, 32)
    assert probs_buf.stride() == (34, 1)
    assert probs_buf.storage_offset() == 1

    cases = (
        ("multinomial", dict(scale=None, do_topk=False, top_k_n=0)),
        ("topk-full", dict(scale=0.7, do_topk=True, top_k_n=32)),
        ("topk-overflow", dict(scale=0.7, do_topk=True, top_k_n=40)),
    )

    for name, kwargs in cases:
        probs_storage.copy_(before)
        destination = torch.empty(2, 1, dtype=torch.long)
        torch.manual_seed(17)
        got = bdh.BDH._sample_from_logits(
            logits.clone(),
            **kwargs,
            probs_buf=probs_buf,
            softmax=torch.nn.functional.softmax,
            multinomial=torch.multinomial,
            idx_out=destination,
        )
        assert got is destination, name

        assert torch.equal(probs_storage[:, 0], before[:, 0]), name
        assert torch.equal(probs_storage[:, -1], before[:, -1]), name

        torch.manual_seed(17)
        ref = bdh.BDH._sample_from_logits(
            logits.clone(),
            **kwargs,
            probs_buf=torch.empty(2, 32),
            softmax=torch.nn.functional.softmax,
            multinomial=torch.multinomial,
        )
        assert torch.equal(got, ref), name


def test_sampler_probability_buffer_preserves_padded_nonunit_vocab_neighbors():
    """Full-vocab sampling must honor padding with non-unit vocab stride."""
    torch.manual_seed(0)
    logits_storage = torch.randn(2, 32, 2)
    logits = logits_storage[..., 0]
    probs_storage = torch.full((2, 34, 2), -777.0)
    probs_buf = probs_storage[:, 1:-1, 0]
    before = probs_storage.clone()
    assert logits.stride() == (64, 2)
    assert probs_buf.shape == (2, 32)
    assert probs_buf.stride() == (68, 2)
    assert probs_buf.storage_offset() == 2

    cases = (
        ("multinomial", dict(scale=None, do_topk=False, top_k_n=0)),
        ("topk-full", dict(scale=0.7, do_topk=True, top_k_n=32)),
        ("topk-overflow", dict(scale=0.7, do_topk=True, top_k_n=40)),
    )

    for name, kwargs in cases:
        probs_storage.copy_(before)
        destination = torch.empty(2, 1, dtype=torch.long)
        torch.manual_seed(17)
        got = bdh.BDH._sample_from_logits(
            logits.clone(),
            **kwargs,
            probs_buf=probs_buf,
            softmax=torch.nn.functional.softmax,
            multinomial=torch.multinomial,
            idx_out=destination,
        )
        assert got is destination, name

        assert torch.equal(probs_storage[..., 1], before[..., 1]), name
        assert torch.equal(probs_storage[:, 0], before[:, 0]), name
        assert torch.equal(probs_storage[:, -1], before[:, -1]), name

        torch.manual_seed(17)
        ref = bdh.BDH._sample_from_logits(
            logits.contiguous(),
            **kwargs,
            probs_buf=torch.empty(2, 32),
            softmax=torch.nn.functional.softmax,
            multinomial=torch.multinomial,
        )
        assert torch.equal(got, ref), name

def test_sampler_padded_nonunit_probability_buffer_preserves_strided_output():
    """Padded scratch and strided decode output may be combined safely."""
    torch.manual_seed(0)
    logits_storage = torch.randn(2, 32, 2)
    logits = logits_storage[..., 0]
    probs_storage = torch.full((2, 34, 2), -777.0)
    probs_buf = probs_storage[:, 1:-1, 0]
    probs_before = probs_storage.clone()
    assert logits.stride() == (64, 2)
    assert probs_buf.stride() == (68, 2)
    assert probs_buf.storage_offset() == 2

    cases = (
        ("multinomial", dict(scale=None, do_topk=False, top_k_n=0)),
        ("topk-full", dict(scale=0.7, do_topk=True, top_k_n=32)),
        ("topk-overflow", dict(scale=0.7, do_topk=True, top_k_n=40)),
    )

    for name, kwargs in cases:
        probs_storage.copy_(probs_before)
        destination = torch.full((1, 3, 2), -123, dtype=torch.long)
        destination_before = destination.clone()
        idx_out = destination[:, 1, :].transpose(0, 1)
        assert idx_out.shape == (2, 1), name
        assert idx_out.stride() == (1, 6), name

        torch.manual_seed(17)
        got = bdh.BDH._sample_from_logits(
            logits.clone(),
            **kwargs,
            probs_buf=probs_buf,
            softmax=torch.nn.functional.softmax,
            multinomial=torch.multinomial,
            idx_out=idx_out,
        )
        assert got is idx_out, name

        target = torch.zeros_like(destination, dtype=torch.bool)
        target[:, 1, :] = True
        assert torch.equal(
            destination.masked_select(~target),
            destination_before.masked_select(~target),
        ), name
        assert torch.equal(probs_storage[..., 1], probs_before[..., 1]), name
        assert torch.equal(probs_storage[:, 0], probs_before[:, 0]), name
        assert torch.equal(probs_storage[:, -1], probs_before[:, -1]), name

        torch.manual_seed(17)
        ref = bdh.BDH._sample_from_logits(
            logits.contiguous(),
            **kwargs,
            probs_buf=torch.empty(2, 32),
            softmax=torch.nn.functional.softmax,
            multinomial=torch.multinomial,
        )
        assert torch.equal(got, ref), name
        assert torch.equal(destination[:, 1, :].transpose(0, 1), ref), name


def test_sampler_narrow_topk_preserves_reused_probability_buffer():
    """Narrow top-k sampling must leave the full-vocab scratch view untouched."""
    torch.manual_seed(0)
    logits_storage = torch.randn(2, 32, 2)
    logits = logits_storage[..., 0]
    probs_storage = torch.full((2, 34, 2), -777.0)
    probs_buf = probs_storage[:, 1:-1, 0]
    probs_before = probs_storage.clone()
    assert logits.stride() == (64, 2)
    assert probs_buf.stride() == (68, 2)
    assert probs_buf.storage_offset() == 2

    for name, top_k_n in (("topk-one", 1), ("topk-narrow", 8)):
        probs_storage.copy_(probs_before)
        destination = torch.full((1, 3, 2), -123, dtype=torch.long)
        destination_before = destination.clone()
        idx_out = destination[:, 1, :].transpose(0, 1)
        assert idx_out.shape == (2, 1), name
        assert idx_out.stride() == (1, 6), name

        torch.manual_seed(17)
        got = bdh.BDH._sample_from_logits(
            logits.clone(),
            scale=0.7,
            do_topk=True,
            top_k_n=top_k_n,
            probs_buf=probs_buf,
            softmax=torch.nn.functional.softmax,
            multinomial=torch.multinomial,
            idx_out=idx_out,
        )
        assert got is idx_out, name

        target = torch.zeros_like(destination, dtype=torch.bool)
        target[:, 1, :] = True
        assert torch.equal(
            destination.masked_select(~target),
            destination_before.masked_select(~target),
        ), name
        assert torch.equal(probs_storage, probs_before), name

        torch.manual_seed(17)
        ref = bdh.BDH._sample_from_logits(
            logits.contiguous(),
            scale=0.7,
            do_topk=True,
            top_k_n=top_k_n,
            probs_buf=torch.empty(2, 32),
            softmax=torch.nn.functional.softmax,
            multinomial=torch.multinomial,
        )
        assert torch.equal(got, ref), name
        assert torch.equal(destination[:, 1, :].transpose(0, 1), ref), name
