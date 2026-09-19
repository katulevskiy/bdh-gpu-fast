"""Prove tril(diagonal=-1) mask, KV-cache decode parity, and RoPE continuity."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import bdh


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


def test_tril_diagonal_minus_one_position_zero_is_exactly_zero():
    """Mask is tril(diagonal=-1): position i attends only to j < i.

    Position 0 has no valid keys, so its attention output must be exactly 0
    (not approximately — zeros from empty sum).
    """
    cfg = _small_cfg()
    attn = bdh.Attention(cfg)
    N = cfg.mlp_internal_dim_multiplier * cfg.n_embd // cfg.n_head
    B, T = 2, 8
    torch.manual_seed(0)
    Q = torch.randn(B, cfg.n_head, T, N)
    V = torch.randn(B, 1, T, cfg.n_embd)

    out, _, _ = attn(Q, Q, V)

    assert out.shape == (B, cfg.n_head, T, cfg.n_embd)
    # Exact zero at t=0 (strict lower-triangular excludes diagonal and above)
    assert torch.equal(out[:, :, 0, :], torch.zeros_like(out[:, :, 0, :])), (
        f"pos0 max abs={out[:, :, 0, :].abs().max().item()}"
    )
    assert torch.count_nonzero(out[:, :, 0, :]) == 0

    # Sanity: later positions generally non-zero with random inputs
    assert out[:, :, 1:, :].abs().sum() > 0


def test_cached_incremental_decode_matches_full_forward():
    """Token-by-token decode with cache must match a single full forward (atol 1e-5)."""
    cfg = _small_cfg()
    torch.manual_seed(21)
    model = bdh.BDH(cfg)
    model.eval()

    torch.manual_seed(22)
    tokens = torch.randint(0, cfg.vocab_size, (2, 16))

    with torch.no_grad():
        full_logits, _ = model(tokens)

        cache = [None] * cfg.n_layer
        parts = []
        for t in range(tokens.size(1)):
            step_logits, _ = model(tokens[:, t : t + 1], cache=cache)
            parts.append(step_logits)
        cached_logits = torch.cat(parts, dim=1)

    assert cached_logits.shape == full_logits.shape
    max_diff = (full_logits - cached_logits).abs().max().item()
    assert torch.allclose(full_logits, cached_logits, rtol=1e-5, atol=1e-5), (
        f"incremental vs full max abs diff={max_diff}"
    )


def test_rope_phases_continue_across_cache_steps():
    """RoPE at absolute position S+t during decode must match full-sequence RoPE.

    Prefill length S, then decode one token: phases for the new token must use
    rope_start=S (positions S..S+T-1), not restart at 0.
    """
    cfg = _small_cfg()
    attn = bdh.Attention(cfg)
    N = cfg.mlp_internal_dim_multiplier * cfg.n_embd // cfg.n_head
    B, S, T_new = 1, 5, 3
    torch.manual_seed(3)
    Q_full = torch.randn(B, cfg.n_head, S + T_new, N)
    V_full = torch.randn(B, 1, S + T_new, cfg.n_embd)

    # Full cold path
    out_full, kr_full, _ = attn(Q_full, Q_full, V_full)

    # Prefill then incremental chunk with past
    Q_pre, V_pre = Q_full[:, :, :S], V_full[:, :, :S]
    out_pre, kr_pre, v_pre = attn(Q_pre, Q_pre, V_pre)

    Q_new, V_new = Q_full[:, :, S:], V_full[:, :, S:]
    out_inc, kr_new, _ = attn(
        Q_new, Q_new, V_new, rope_start=S, past_kr=kr_pre, past_v=v_pre
    )

    # Stored RoPE'd keys for the new chunk must equal full KR at positions S:
    assert torch.allclose(kr_new, kr_full[:, :, S:], rtol=0, atol=1e-6), (
        f"kr phase mismatch max={(kr_new - kr_full[:, :, S:]).abs().max().item()}"
    )

    # Incremental attention outs for the new positions match full outs
    assert torch.allclose(out_inc, out_full[:, :, S:], rtol=1e-5, atol=1e-5), (
        f"out mismatch max={(out_inc - out_full[:, :, S:]).abs().max().item()}"
    )

    # Explicit phase tensor check: _rope_phases(T, rope_start=S) == full[S:]
    phases_full = attn._rope_phases(S + T_new, 0, Q_full.device)
    phases_cont = attn._rope_phases(T_new, S, Q_full.device)
    assert torch.equal(phases_cont, phases_full[:, :, S:]), (
        "RoPE phase continuation must equal sliced full phases"
    )

    # Model-level: cache after prefill has length S; next step uses rope_start=S
    torch.manual_seed(4)
    model = bdh.BDH(cfg)
    model.eval()
    tokens = torch.randint(0, cfg.vocab_size, (1, S + T_new))
    with torch.no_grad():
        full_logits, _ = model(tokens)
        cache = [None] * cfg.n_layer
        pre_logits, _ = model(tokens[:, :S], cache=cache)
        assert cache[0]["kr"].size(2) == S
        rest = []
        for t in range(S, S + T_new):
            lg, _ = model(tokens[:, t : t + 1], cache=cache)
            rest.append(lg)
        rest_logits = torch.cat(rest, dim=1)
        assert torch.allclose(
            torch.cat([pre_logits, rest_logits], dim=1),
            full_logits,
            rtol=1e-5,
            atol=1e-5,
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
