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

    # Non-default sampler options own the first-step logits buffer.  Probe both
    # the narrow top-k path and the k==V fallback, not just default multinomial.
    cases = (
        ("scaled", dict(temperature=0.7)),
        ("topk-narrow", dict(temperature=0.7, top_k=8)),
        ("topk-full", dict(temperature=0.7, top_k=cfg.vocab_size)),
    )
    with torch.inference_mode():
        for name, kwargs in cases:
            prof = _profile_call(
                lambda kwargs=kwargs: model.generate(
                    idx[:, :5], max_new_tokens=1, **kwargs
                )
            )
            shapes = _event_shapes(prof, "aten::contiguous")
            assert _count(prof, "aten::cat") == 0, name
            assert shapes.count((2,)) == 1, (name, shapes)
            assert (2, cfg.vocab_size) not in shapes, (name, shapes)
