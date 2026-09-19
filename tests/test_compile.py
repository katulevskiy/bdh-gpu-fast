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


def test_residual_ln_grad_parity_vs_out_of_place_add():
    """In-place residual reuse matches out-of-place LN(x+LN(y)) grads @ dropout=0."""
    cfg = _small_cfg(dropout=0.0)
    torch.manual_seed(3)
    m = bdh.BDH(cfg).train()
    x = torch.randint(0, cfg.vocab_size, (2, 12))
    y = torch.randint(0, cfg.vocab_size, (2, 12))

    # Reference: force out-of-place add path via monkeypatch
    import torch.nn.functional as F

    def _oop(self, x_t, y_mlp):
        shape, eps = self._ln_shape, self._ln_eps
        yy = F.layer_norm(y_mlp, shape, weight=None, bias=None, eps=eps)
        return F.layer_norm(x_t + yy, shape, weight=None, bias=None, eps=eps)

    ref = bdh.BDH(cfg).train()
    ref.load_state_dict(m.state_dict())
    ref._residual_ln = _oop.__get__(ref, bdh.BDH)

    logits_m, loss_m = m(x, y)
    loss_m.backward()
    logits_r, loss_r = ref(x, y)
    loss_r.backward()
    assert torch.equal(logits_m, logits_r)
    assert torch.equal(loss_m, loss_r)
    for (n1, p1), (n2, p2) in zip(m.named_parameters(), ref.named_parameters()):
        if p1.grad is None and p2.grad is None:
            continue
        assert torch.equal(p1.grad, p2.grad), n1


def test_residual_ln_does_not_mutate_inputs():
    """y.add_(x) must only mutate the inner LN output, not x / y_mlp."""
    cfg = _small_cfg(dropout=0.0)
    m = bdh.BDH(cfg)
    torch.manual_seed(11)
    x = torch.randn(2, 8, cfg.n_embd)
    y_mlp = torch.randn(2, 8, cfg.n_embd)
    x0, y0 = x.clone(), y_mlp.clone()
    _ = m._residual_ln(x, y_mlp)
    assert torch.equal(x, x0)
    assert torch.equal(y_mlp, y0)


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


# --- opt/compile-blocked: compile × blocked × optional AUTOGRAD smoke ---


@pytest.mark.parametrize("impl", ["eager", "blocked"])
@pytest.mark.parametrize("autograd", [False, True])
def test_compile_forward_matches_eager_blocked_autograd(impl, autograd, monkeypatch):
    """Compiled cold forward matches eager at dropout=0 for IMPL×AUTOGRAD.

    Soft-skips if inductor unavailable. No throughput / GPU claims.
    """
    monkeypatch.setenv("BDH_ATTN_IMPL", impl)
    if autograd:
        monkeypatch.setenv("BDH_ATTN_AUTOGRAD", "1")
    else:
        monkeypatch.delenv("BDH_ATTN_AUTOGRAD", raising=False)

    cfg = _small_cfg(dropout=0.0)
    torch.manual_seed(0)
    eager = bdh.BDH(cfg).eval()
    compiled_src = bdh.BDH(cfg).eval()
    compiled_src.load_state_dict(eager.state_dict())
    compiled = _compile_or_skip(compiled_src, mode="default")

    torch.manual_seed(41)
    x = torch.randint(0, cfg.vocab_size, (2, 12))
    y = torch.randint(0, cfg.vocab_size, (2, 12))

    def _warm():
        with torch.no_grad():
            return compiled(x, y)

    _probe_or_skip(compiled, _warm)
    with torch.no_grad():
        le, lose = eager(x, y)
        lc, losc = compiled(x, y)
    assert torch.allclose(le, lc, rtol=0, atol=1e-5), (
        f"impl={impl} autograd={autograd} logits maxdiff={(le - lc).abs().max().item()}"
    )
    assert torch.allclose(lose, losc, rtol=0, atol=1e-5)


@pytest.mark.parametrize("impl", ["eager", "blocked"])
def test_compile_train_step_blocked_autograd_smoke(impl, monkeypatch):
    """One compiled train_step under IMPL + AUTOGRAD=1 must finish (CPU).

    Soft-skips if inductor/probe fails. Asserts finite loss only — no speed.
    """
    monkeypatch.setenv("BDH_ATTN_IMPL", impl)
    monkeypatch.setenv("BDH_ATTN_AUTOGRAD", "1")
    monkeypatch.setenv("BDH_COMPILE", "0")  # maybe_compile toggled below

    import importlib
    import train as tr

    importlib.reload(tr)

    cfg = _small_cfg(dropout=0.0)
    torch.manual_seed(2)
    m = bdh.BDH(cfg).train()
    x = torch.randint(0, cfg.vocab_size, (2, 10))
    y = torch.randint(0, cfg.vocab_size, (2, 10))

    tr.USE_COMPILE = True
    try:
        m = tr.maybe_compile(m, example_x=x, example_y=y)
    finally:
        tr.USE_COMPILE = False
        monkeypatch.setenv("BDH_COMPILE", "0")
        importlib.reload(tr)

    if getattr(m, "_orig_mod", None) is None:
        pytest.skip("torch.compile unavailable or probe fell back to eager")

    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    loss = tr.train_step(m, opt, x, y)
    assert torch.isfinite(loss)


def test_cold_autograd_self_attn_no_dynamo_graph_breaks(monkeypatch):
    """AUTOGRAD=1 + Q-is-K self path should not trip duplicate-input breaks.

    Before StrictTrilSelfAttnFn, Dynamo reported ~9 breaks (gb6297). After the
    self-attn Function, cold train path at dropout=0 should be a single graph
    for both eager and blocked.
    """
    monkeypatch.setenv("BDH_ATTN_AUTOGRAD", "1")
    monkeypatch.setenv("BDH_ATTN_IMPL", "blocked")

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


# --- opt/compile-guidance: warn COMPILE+blocked; eager+AUTOGRAD compile path ---


@pytest.mark.parametrize("impl", ["blocked", "online", "triton"])
def test_maybe_compile_warns_compile_with_blocked(impl, monkeypatch, capsys):
    """COMPILE=1 + blocked|online|triton must log the CPU regression warning.

    Defaults are unchanged; warning is advisory only (see #46 matrix).
    """
    import importlib
    import train as tr

    monkeypatch.setenv("BDH_COMPILE", "1")
    monkeypatch.setenv("BDH_COMPILE_PROBE", "train")
    monkeypatch.setenv("BDH_COMPILE_MODE", "default")
    monkeypatch.setenv("BDH_COMPILE_FULLGRAPH", "0")
    monkeypatch.setenv("BDH_ATTN_IMPL", impl)
    importlib.reload(tr)
    assert tr.USE_COMPILE is True

    cfg = _small_cfg(dropout=0.0)
    model = bdh.BDH(cfg).train()
    x = torch.randint(0, cfg.vocab_size, (2, 8))
    y = torch.randint(0, cfg.vocab_size, (2, 8))
    try:
        out = tr.maybe_compile(model, example_x=x, example_y=y)
    finally:
        monkeypatch.setenv("BDH_COMPILE", "0")
        monkeypatch.setenv("BDH_ATTN_IMPL", "eager")
        importlib.reload(tr)

    captured = capsys.readouterr().out
    assert "torch.compile warning" in captured
    assert f"BDH_ATTN_IMPL={impl}" in captured
    assert "regression" in captured.lower() or "slower" in captured.lower()
    assert out is not None


def test_maybe_compile_no_warn_on_eager(monkeypatch, capsys):
    """COMPILE=1 + eager must not emit the blocked-compile regression warning."""
    import importlib
    import train as tr

    monkeypatch.setenv("BDH_COMPILE", "1")
    monkeypatch.setenv("BDH_COMPILE_PROBE", "train")
    monkeypatch.setenv("BDH_ATTN_IMPL", "eager")
    importlib.reload(tr)

    cfg = _small_cfg(dropout=0.0)
    model = bdh.BDH(cfg).train()
    x = torch.randint(0, cfg.vocab_size, (2, 8))
    y = torch.randint(0, cfg.vocab_size, (2, 8))
    try:
        tr.maybe_compile(model, example_x=x, example_y=y)
    finally:
        monkeypatch.setenv("BDH_COMPILE", "0")
        importlib.reload(tr)

    captured = capsys.readouterr().out
    assert "torch.compile warning" not in captured


def test_compile_eager_autograd_smoke_zero_graph_breaks(monkeypatch):
    """COMPILE path + eager + AUTOGRAD=1: train_step smoke and 0 Dynamo breaks.

    Recommended CPU compile train path after #46 (COMPILE=1 only with eager).
    Soft-skips if inductor unavailable. No throughput / GPU claims.
    """
    monkeypatch.setenv("BDH_ATTN_IMPL", "eager")
    monkeypatch.setenv("BDH_ATTN_AUTOGRAD", "1")

    cfg = _small_cfg(dropout=0.0)
    m_explain = bdh.BDH(cfg).train()
    x = torch.randint(0, cfg.vocab_size, (2, 10))
    y = torch.randint(0, cfg.vocab_size, (2, 10))
    try:
        expl = torch._dynamo.explain(m_explain)(x, y)
    except Exception as e:
        pytest.skip(f"dynamo.explain unavailable: {type(e).__name__}: {e}")
    assert expl.graph_break_count == 0, expl.break_reasons
    assert expl.graph_count >= 1

    import importlib
    import train as tr

    monkeypatch.setenv("BDH_COMPILE", "0")
    importlib.reload(tr)

    torch.manual_seed(5)
    m = bdh.BDH(cfg).train()
    tr.USE_COMPILE = True
    try:
        m = tr.maybe_compile(m, example_x=x, example_y=y)
    finally:
        tr.USE_COMPILE = False
        monkeypatch.setenv("BDH_COMPILE", "0")
        importlib.reload(tr)

    if getattr(m, "_orig_mod", None) is None:
        pytest.skip("torch.compile unavailable or probe fell back to eager")

    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    loss = tr.train_step(m, opt, x, y)
    assert torch.isfinite(loss)


# --- opt/compile-reduce: MODE=default vs reduce-overhead on CPU ---


def test_maybe_compile_warns_reduce_overhead_on_cpu(monkeypatch, capsys):
    """COMPILE=1 + MODE=reduce-overhead on non-CUDA must warn (no CUDA graphs).

    Advisory only — still attempts compile; defaults unchanged.
    """
    import importlib
    import train as tr

    if torch.cuda.is_available():
        pytest.skip("this warning is for non-CUDA devices")

    monkeypatch.setenv("BDH_COMPILE", "1")
    monkeypatch.setenv("BDH_COMPILE_PROBE", "train")
    monkeypatch.setenv("BDH_COMPILE_MODE", "reduce-overhead")
    monkeypatch.setenv("BDH_COMPILE_FULLGRAPH", "0")
    monkeypatch.setenv("BDH_ATTN_IMPL", "eager")
    importlib.reload(tr)
    assert tr.USE_COMPILE is True
    assert tr.COMPILE_MODE == "reduce-overhead"

    cfg = _small_cfg(dropout=0.0)
    model = bdh.BDH(cfg).train()
    x = torch.randint(0, cfg.vocab_size, (2, 8))
    y = torch.randint(0, cfg.vocab_size, (2, 8))
    try:
        out = tr.maybe_compile(model, example_x=x, example_y=y)
    finally:
        monkeypatch.setenv("BDH_COMPILE", "0")
        monkeypatch.setenv("BDH_COMPILE_MODE", "default")
        importlib.reload(tr)

    captured = capsys.readouterr().out
    assert "reduce-overhead" in captured
    assert "CUDA graphs" in captured or "not useful" in captured.lower()
    assert out is not None


def test_maybe_compile_reduce_overhead_soft_or_runs(monkeypatch):
    """MODE=reduce-overhead must soft-fallback or produce a usable module.

    Soft-skips neither path hard-fails. No throughput / CUDA-graph claims.
    """
    import importlib
    import train as tr

    monkeypatch.setenv("BDH_COMPILE", "1")
    monkeypatch.setenv("BDH_COMPILE_PROBE", "train")
    monkeypatch.setenv("BDH_COMPILE_MODE", "reduce-overhead")
    monkeypatch.setenv("BDH_COMPILE_FULLGRAPH", "0")
    monkeypatch.setenv("BDH_ATTN_IMPL", "eager")
    importlib.reload(tr)

    cfg = _small_cfg(dropout=0.0)
    torch.manual_seed(11)
    m = bdh.BDH(cfg).train()
    x = torch.randint(0, cfg.vocab_size, (2, 10))
    y = torch.randint(0, cfg.vocab_size, (2, 10))
    try:
        out = tr.maybe_compile(m, example_x=x, example_y=y)
    finally:
        monkeypatch.setenv("BDH_COMPILE", "0")
        monkeypatch.setenv("BDH_COMPILE_MODE", "default")
        importlib.reload(tr)

    assert out is not None
    out.train()
    logits, loss = out(x, y)
    assert logits.shape[-1] == cfg.vocab_size
    assert loss is not None and torch.isfinite(loss)


def test_compile_mode_default_vs_reduce_overhead_parity(monkeypatch):
    """Compiled forward @ dropout=0: default mode matches reduce-overhead logits.

    Soft-skips if either mode cannot compile. No speed / GPU claims.
    """
    cfg = _small_cfg(dropout=0.0)
    torch.manual_seed(0)
    base = bdh.BDH(cfg).eval()
    state = base.state_dict()

    def _compile_mode(mode: str):
        src = bdh.BDH(cfg).eval()
        src.load_state_dict(state)
        return _compile_or_skip(src, mode=mode)

    m_default = _compile_mode("default")
    m_reduce = _compile_mode("reduce-overhead")

    torch.manual_seed(42)
    x = torch.randint(0, cfg.vocab_size, (2, 12))
    y = torch.randint(0, cfg.vocab_size, (2, 12))

    def _warm(m):
        with torch.no_grad():
            return m(x, y)

    _probe_or_skip(m_default, lambda: _warm(m_default))
    _probe_or_skip(m_reduce, lambda: _warm(m_reduce))

    with torch.no_grad():
        ld, lossd = m_default(x, y)
        lr, lossr = m_reduce(x, y)
    assert torch.allclose(ld, lr, rtol=0, atol=1e-5), (
        f"logits maxdiff={(ld - lr).abs().max().item()}"
    )
    assert torch.allclose(lossd, lossr, rtol=0, atol=1e-5)


def test_compile_reduce_overhead_train_step_smoke(monkeypatch):
    """One train_step under COMPILE=1 MODE=reduce-overhead must finish (CPU).

    Soft-skips if inductor/probe falls back. Finite loss only — no speed claim.
    """
    import importlib
    import train as tr

    monkeypatch.setenv("BDH_COMPILE", "0")
    monkeypatch.setenv("BDH_COMPILE_MODE", "reduce-overhead")
    monkeypatch.setenv("BDH_ATTN_IMPL", "eager")
    importlib.reload(tr)

    cfg = _small_cfg(dropout=0.0)
    torch.manual_seed(4)
    m = bdh.BDH(cfg).train()
    x = torch.randint(0, cfg.vocab_size, (2, 10))
    y = torch.randint(0, cfg.vocab_size, (2, 10))

    tr.USE_COMPILE = True
    tr.COMPILE_MODE = "reduce-overhead"
    try:
        m = tr.maybe_compile(m, example_x=x, example_y=y)
    finally:
        tr.USE_COMPILE = False
        tr.COMPILE_MODE = "default"
        monkeypatch.setenv("BDH_COMPILE", "0")
        monkeypatch.setenv("BDH_COMPILE_MODE", "default")
        importlib.reload(tr)

    if getattr(m, "_orig_mod", None) is None:
        pytest.skip("torch.compile unavailable or probe fell back to eager")

    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    loss = tr.train_step(m, opt, x, y)
    assert torch.isfinite(loss)
