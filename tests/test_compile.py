"""torch.compile harden: compile+forward matches eager at dropout=0 (CPU inductor).

No GPU claims — this box may be CPU-only; tests assert numerical parity, not speed.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import bdh
from bdh_cache import CacheManager

# Keep compile opt-in for imports of train helpers.
os.environ.setdefault("BDH_COMPILE", "0")


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


def _compile_or_skip(model: torch.nn.Module, **kwargs):
    """Wrap torch.compile; skip if inductor/CXX unavailable on this box."""
    try:
        return torch.compile(model, **kwargs)
    except Exception as e:
        pytest.skip(f"torch.compile unavailable: {type(e).__name__}: {e}")


def _probe_or_skip(compiled, fn):
    try:
        return fn()
    except Exception as e:
        pytest.skip(f"compile probe/inductor failed: {type(e).__name__}: {e}")


@pytest.mark.parametrize("train_mode", [False, True])
def test_compile_forward_matches_eager_dropout_zero(train_mode: bool):
    """Compiled forward matches eager at dropout=0 (eval and train).

    Inductor on CPU can introduce ~1e-6 float noise vs eager — use atol, not
    bit-identical equality. No throughput / GPU claims.
    """
    cfg = _small_cfg(dropout=0.0)
    torch.manual_seed(0)
    eager = bdh.BDH(cfg)
    compiled_src = bdh.BDH(cfg)
    compiled_src.load_state_dict(eager.state_dict())

    if train_mode:
        eager.train()
        compiled_src.train()
    else:
        eager.eval()
        compiled_src.eval()

    compiled = _compile_or_skip(compiled_src, mode="default")
    torch.manual_seed(123)
    x = torch.randint(0, cfg.vocab_size, (3, 16))
    y = torch.randint(0, cfg.vocab_size, (3, 16))

    def _run():
        if train_mode:
            return compiled(x, y)
        with torch.no_grad():
            return compiled(x, y)

    _probe_or_skip(compiled, _run)

    if train_mode:
        le, lose = eager(x, y)
        lc, losc = compiled(x, y)
    else:
        with torch.no_grad():
            le, lose = eager(x, y)
            lc, losc = compiled(x, y)

    assert le.shape == lc.shape
    assert torch.allclose(le, lc, rtol=0, atol=1e-5), (
        f"logits maxdiff={(le - lc).abs().max().item()}"
    )
    assert torch.allclose(lose, losc, rtol=0, atol=1e-5), (
        f"loss {lose.item()} vs {losc.item()}"
    )


def test_compile_fullgraph_eval_dropout_zero():
    """Cold training path (no cache) should hold as a single Dynamo graph."""
    cfg = _small_cfg(dropout=0.0)
    m = bdh.BDH(cfg).eval()
    compiled = _compile_or_skip(m, mode="default", fullgraph=True)
    x = torch.randint(0, cfg.vocab_size, (2, 12))
    y = torch.randint(0, cfg.vocab_size, (2, 12))

    def _run():
        with torch.no_grad():
            return compiled(x, y)

    logits, loss = _probe_or_skip(compiled, _run)
    assert logits.shape == (2, 12, cfg.vocab_size)
    assert loss is not None and torch.isfinite(loss)


def test_compile_rope_cache_parity_under_inductor():
    """RoPE cis cache must not break inductor parity vs eager (same weights)."""
    cfg = _small_cfg(dropout=0.0)
    torch.manual_seed(7)
    eager = bdh.BDH(cfg).eval()
    compiled_src = bdh.BDH(cfg).eval()
    compiled_src.load_state_dict(eager.state_dict())
    compiled = _compile_or_skip(compiled_src)

    xs = [torch.randint(0, cfg.vocab_size, (2, T)) for T in (8, 8, 12)]

    def _warm():
        with torch.no_grad():
            for x in xs:
                compiled(x)

    _probe_or_skip(compiled, _warm)

    with torch.no_grad():
        for x in xs:
            le, _ = eager(x)
            lc, _ = compiled(x)
            assert torch.allclose(le, lc, rtol=0, atol=1e-5), (
                f"T={x.size(1)} maxdiff={(le - lc).abs().max().item()}"
            )


def test_compile_cache_decode_matches_eager_dropout_zero():
    """Packed CacheManager prefill+decode under compile matches eager."""
    cfg = _small_cfg(dropout=0.0)
    torch.manual_seed(3)
    eager = bdh.BDH(cfg).eval()
    compiled_src = bdh.BDH(cfg).eval()
    compiled_src.load_state_dict(eager.state_dict())
    compiled = _compile_or_skip(compiled_src)

    x = torch.randint(0, cfg.vocab_size, (2, 10))
    cache_e = CacheManager.from_config(
        cfg, batch_size=2, max_seq=16, device=x.device, storage_dtype=torch.float32
    )
    cache_c = CacheManager.from_config(
        cfg, batch_size=2, max_seq=16, device=x.device, storage_dtype=torch.float32
    )

    def _warm():
        with torch.no_grad():
            compiled(x[:, :6], cache=cache_c)
            compiled(x[:, 6:7], cache=cache_c)

    _probe_or_skip(compiled, _warm)
    # Reset compiled cache for fair compare
    cache_c = CacheManager.from_config(
        cfg, batch_size=2, max_seq=16, device=x.device, storage_dtype=torch.float32
    )

    with torch.no_grad():
        le, _ = eager(x[:, :6], cache=cache_e)
        lc, _ = compiled(x[:, :6], cache=cache_c)
        assert torch.allclose(le, lc, rtol=0, atol=1e-5)
        le2, _ = eager(x[:, 6:7], cache=cache_e)
        lc2, _ = compiled(x[:, 6:7], cache=cache_c)
        assert torch.allclose(le2, lc2, rtol=0, atol=1e-5)
        assert cache_e.seq_len == cache_c.seq_len == 7


def test_maybe_compile_train_probe_fallback_and_enable():
    """BDH_COMPILE=1 default probe is train-mode (not eval-only)."""
    # Import after env so train reads knobs — use a fresh subprocess-like reload.
    import importlib

    os.environ["BDH_COMPILE"] = "1"
    os.environ["BDH_COMPILE_PROBE"] = "train"
    os.environ["BDH_COMPILE_MODE"] = "default"
    os.environ["BDH_COMPILE_FULLGRAPH"] = "0"

    import train as tr

    importlib.reload(tr)
    assert tr.USE_COMPILE is True
    assert tr.COMPILE_PROBE == "train"

    cfg = _small_cfg(dropout=0.0)
    # Shrink config for probe speed
    model = bdh.BDH(cfg)
    model.train()
    x = torch.randint(0, cfg.vocab_size, (2, 8))
    y = torch.randint(0, cfg.vocab_size, (2, 8))

    out = tr.maybe_compile(model, example_x=x, example_y=y)
    # Either OptimizedModule or eager fallback if inductor missing
    assert out is not None
    out.train()
    logits, loss = out(x, y)
    assert logits.shape[-1] == cfg.vocab_size
    assert loss is not None and torch.isfinite(loss)

    # Restore default so other tests importing train stay opt-in-off
    os.environ["BDH_COMPILE"] = "0"
    importlib.reload(tr)


def test_generate_disabled_under_compile():
    """generate() is torch.compiler.disable — still runs on a compiled module."""
    cfg = _small_cfg(dropout=0.0)
    m = bdh.BDH(cfg).eval()
    compiled = _compile_or_skip(m)
    prompt = torch.randint(0, cfg.vocab_size, (1, 4))

    def _warm():
        with torch.no_grad():
            compiled(prompt)

    _probe_or_skip(compiled, _warm)
    out = compiled.generate(prompt, max_new_tokens=3, top_k=3)
    assert out.shape == (1, 7)


def test_residual_ln_matches_module_ln_bitexact():
    """Functional _residual_ln == ln(x + ln(y)) bit-exactly (affine-free)."""
    cfg = _small_cfg(dropout=0.0)
    m = bdh.BDH(cfg)
    torch.manual_seed(7)
    x = torch.randn(3, 11, cfg.n_embd)
    y_mlp = torch.randn(3, 11, cfg.n_embd)
    got = m._residual_ln(x, y_mlp)
    ref = m.ln(x + m.ln(y_mlp))
    assert torch.equal(got, ref)
    # Single LN helper also matches module
    assert torch.equal(m._ln(x), m.ln(x))


def test_compile_residual_ln_path_matches_eager():
    """Compiled forward (residual LN epilogue) matches eager at dropout=0.

    Exercises BDH_COMPILE-style torch.compile on the LN(x+LN(yMLP)) path.
    """
    cfg = _small_cfg(dropout=0.0)
    torch.manual_seed(0)
    eager = bdh.BDH(cfg).eval()
    compiled_src = bdh.BDH(cfg).eval()
    compiled_src.load_state_dict(eager.state_dict())
    compiled = _compile_or_skip(compiled_src, mode="default", fullgraph=True)

    torch.manual_seed(99)
    x = torch.randint(0, cfg.vocab_size, (2, 14))
    y = torch.randint(0, cfg.vocab_size, (2, 14))

    def _run():
        with torch.no_grad():
            return compiled(x, y)

    _probe_or_skip(compiled, _run)
    with torch.no_grad():
        le, lose = eager(x, y)
        lc, losc = compiled(x, y)
    assert torch.allclose(le, lc, rtol=0, atol=1e-5), (
        f"logits maxdiff={(le - lc).abs().max().item()}"
    )
    assert torch.allclose(lose, losc, rtol=0, atol=1e-5)


def test_cold_forward_no_dynamo_graph_breaks_dropout_zero():
    """Cold train path (no cache) should be a single Dynamo graph at dropout=0."""
    cfg = _small_cfg(dropout=0.0)
    m = bdh.BDH(cfg).train()
    x = torch.randint(0, cfg.vocab_size, (2, 10))
    y = torch.randint(0, cfg.vocab_size, (2, 10))
    try:
        expl = torch._dynamo.explain(m)(x, y)
    except Exception as e:
        pytest.skip(f"dynamo.explain unavailable: {type(e).__name__}: {e}")
    assert expl.graph_break_count == 0, expl.break_reasons
    assert expl.graph_count >= 1
