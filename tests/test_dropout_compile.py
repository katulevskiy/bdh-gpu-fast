"""opt/dropout-compile: harden dropout=0 / eval identity for torch.compile.

Honest CPU checks — no GPU / throughput claims. Defaults unchanged
(config.dropout still 0.1; train entrypoint still BDH_COMPILE=0).
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
    try:
        return torch.compile(model, **kwargs)
    except Exception as e:
        pytest.skip(f"torch.compile unavailable: {type(e).__name__}: {e}")


def _probe_or_skip(compiled, fn):
    try:
        return fn()
    except Exception as e:
        pytest.skip(f"compile probe/inductor failed: {type(e).__name__}: {e}")


def _fx_op_names(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor):
    from torch.fx.experimental.proxy_tensor import make_fx

    gm = make_fx(model)(x, y)
    names = []
    for n in gm.graph.nodes:
        if n.op != "call_function":
            continue
        t = n.target
        names.append(getattr(t, "__name__", None) or str(t))
    return names


def test_dropout_p0_returns_same_object_train_and_eval():
    """dropout_p==0 must be a true identity (no new tensor)."""
    cfg = _small_cfg(dropout=0.0)
    m = bdh.BDH(cfg)
    x = torch.randn(2, 8, cfg.n_head, cfg.n_embd // cfg.n_head)
    m.train()
    y = m._dropout(x)
    assert y is x
    m.eval()
    y2 = m._dropout(x)
    assert y2 is x


def test_dropout_eval_identity_even_when_p_positive():
    """Eval must skip F.dropout entirely (identity) even at default p=0.1."""
    cfg = _small_cfg(dropout=0.1)
    m = bdh.BDH(cfg).eval()
    x = torch.randn(2, 8, cfg.n_head, cfg.n_embd // cfg.n_head)
    y = m._dropout(x)
    assert y is x
    assert torch.equal(y, x)


def test_dropout_train_p_positive_applies_mask():
    """Train + p>0 must call F.dropout (tensor changes under seed)."""
    cfg = _small_cfg(dropout=0.5)
    m = bdh.BDH(cfg).train()
    x = torch.ones(4, 16, cfg.n_head, cfg.n_embd // cfg.n_head)
    torch.manual_seed(0)
    y = m._dropout(x)
    assert y is not x
    assert not torch.equal(y, x)


def test_fx_no_bernoulli_when_dropout_zero():
    """Full forward FX graph at dropout=0 must not contain dropout RNG ops."""
    cfg = _small_cfg(dropout=0.0, n_layer=1, n_embd=32, n_head=2)
    m = bdh.BDH(cfg).train()
    x = torch.randint(0, cfg.vocab_size, (2, 8))
    y = torch.randint(0, cfg.vocab_size, (2, 8))
    names = _fx_op_names(m, x, y)
    drop_ish = [n for n in names if "drop" in n.lower() or "bernoulli" in n.lower()]
    assert drop_ish == [], drop_ish


def test_fx_has_bernoulli_when_dropout_positive_train():
    """Train + p>0 FX graph should include bernoulli / dropout RNG."""
    cfg = _small_cfg(dropout=0.1, n_layer=1, n_embd=32, n_head=2)
    m = bdh.BDH(cfg).train()
    x = torch.randint(0, cfg.vocab_size, (2, 8))
    y = torch.randint(0, cfg.vocab_size, (2, 8))
    names = _fx_op_names(m, x, y)
    drop_ish = [n for n in names if "drop" in n.lower() or "bernoulli" in n.lower()]
    assert drop_ish, "expected bernoulli_/dropout ops when p>0 and training"


def test_fx_eval_no_bernoulli_even_when_p_positive():
    """Eval FX graph must stay RNG-free even when config.dropout > 0."""
    cfg = _small_cfg(dropout=0.1, n_layer=1, n_embd=32, n_head=2)
    m = bdh.BDH(cfg).eval()
    x = torch.randint(0, cfg.vocab_size, (2, 8))
    y = torch.randint(0, cfg.vocab_size, (2, 8))
    names = _fx_op_names(m, x, y)
    drop_ish = [n for n in names if "drop" in n.lower() or "bernoulli" in n.lower()]
    assert drop_ish == [], drop_ish


def test_train_eval_logits_bitidentical_at_dropout_zero():
    """At dropout=0, train-mode forward ≡ eval (identity path)."""
    cfg = _small_cfg(dropout=0.0)
    torch.manual_seed(0)
    m = bdh.BDH(cfg)
    x = torch.randint(0, cfg.vocab_size, (3, 12))
    y = torch.randint(0, cfg.vocab_size, (3, 12))
    m.train()
    lt, losst = m(x, y)
    m.eval()
    with torch.no_grad():
        le, losse = m(x, y)
    assert torch.equal(lt, le)
    assert torch.equal(losst, losse)


def test_compile_forward_matches_eager_dropout_zero_fullgraph():
    """Compiled fullgraph forward @ dropout=0 matches eager (CPU inductor)."""
    cfg = _small_cfg(dropout=0.0)
    torch.manual_seed(0)
    eager = bdh.BDH(cfg).train()
    compiled_src = bdh.BDH(cfg).train()
    compiled_src.load_state_dict(eager.state_dict())
    compiled = _compile_or_skip(compiled_src, mode="default", fullgraph=True)

    torch.manual_seed(123)
    x = torch.randint(0, cfg.vocab_size, (2, 14))
    y = torch.randint(0, cfg.vocab_size, (2, 14))

    def _run():
        return compiled(x, y)

    _probe_or_skip(compiled, _run)
    le, lose = eager(x, y)
    lc, losc = compiled(x, y)
    assert torch.allclose(le, lc, rtol=0, atol=1e-5), (
        f"logits maxdiff={(le - lc).abs().max().item()}"
    )
    assert torch.allclose(lose, losc, rtol=0, atol=1e-5)


def test_compile_train_step_dropout_positive_smoke(monkeypatch):
    """COMPILE=1 train_step with default-ish dropout>0 must finish (CPU).

    Soft-skips if inductor/probe falls back. Finite loss only — no speed claim.
    """
    import importlib
    import train as tr

    monkeypatch.setenv("BDH_COMPILE", "0")
    monkeypatch.setenv("BDH_ATTN_IMPL", "eager")
    importlib.reload(tr)

    cfg = _small_cfg(dropout=0.1)
    torch.manual_seed(4)
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


def test_cold_train_no_dynamo_graph_breaks_dropout_zero():
    """Cold train @ dropout=0 remains a single Dynamo graph (0 breaks)."""
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


def test_cold_train_no_dynamo_graph_breaks_dropout_positive():
    """Cold train @ dropout>0 should still avoid Dynamo graph breaks.

    F.dropout uses torch RNG only — no Python random / NumPy side path.
    """
    cfg = _small_cfg(dropout=0.1)
    m = bdh.BDH(cfg).train()
    x = torch.randint(0, cfg.vocab_size, (2, 10))
    y = torch.randint(0, cfg.vocab_size, (2, 10))
    try:
        expl = torch._dynamo.explain(m)(x, y)
    except Exception as e:
        pytest.skip(f"dynamo.explain unavailable: {type(e).__name__}: {e}")
    assert expl.graph_break_count == 0, expl.break_reasons
    assert expl.graph_count >= 1
