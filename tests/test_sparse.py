"""Numerical equivalence: sparse / masked densification vs dense ReLU GEMMs.

Default BDH path (bdh.py) is untouched — these tests only exercise bdh_sparse.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import bdh
import bdh_sparse as sp
from benchmarks import bench_sparse_probe as probe


def _rand_latent(*shape, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=g)


def test_sparse_probe_gate_defaults_off(monkeypatch):
    """Optional sparse materialization requires an explicit env gate."""
    monkeypatch.delenv(sp.SPARSE_PROBE_ENV, raising=False)
    assert not sp.sparse_probe_enabled()
    x = _rand_latent(2, 3, seed=0)
    weight = _rand_latent(3, 4, seed=1)
    dense = sp.encoder_relu_matmul(x, weight, use_sparse=False)
    with pytest.raises(RuntimeError, match="BDH_SPARSE_PROBE=1"):
        sp.encoder_relu_matmul(x, weight, use_sparse=True)

    monkeypatch.setenv(sp.SPARSE_PROBE_ENV, "true")
    assert sp.sparse_probe_enabled()
    probed = sp.encoder_relu_matmul(x, weight, use_sparse=True)
    assert torch.equal(probed, dense)


def test_sparse_probe_exit_codes(monkeypatch, capsys):
    """The default-off no-op and enforced guardrail are scriptable outcomes."""
    monkeypatch.delenv(sp.SPARSE_PROBE_ENV, raising=False)
    assert probe.main(["--skip-train", "--skip-crossover"]) == probe.EXIT_OK
    assert "DISABLED" in capsys.readouterr().out

    monkeypatch.setenv(sp.SPARSE_PROBE_ENV, "1")
    monkeypatch.setattr(
        probe,
        "short_train_density",
        lambda **_: [{"x": 0.01, "y": 0.01, "xy": 0.01}],
    )
    assert (
        probe.main(
            [
                "--enforce-density-guardrail",
                "--skip-crossover",
            ]
        )
        == probe.EXIT_DENSITY_GUARDRAIL
    )
    captured = capsys.readouterr()
    assert "density_guardrail=fail" in captured.out
    assert "density re-smoke guardrail failed" in captured.err


def test_relu_density_random_approx_half():
    x = _rand_latent(4, 8, 64, seed=1)
    d = sp.relu_density(F.relu(x))
    assert 0.35 < d < 0.65


def test_sparse_coo_matmul_matches_dense():
    act = F.relu(_rand_latent(32, 64, seed=2))
    W = _rand_latent(64, 48, seed=3)
    dense = act @ W
    out = sp.sparse_relu_matmul(act, W, layout="coo")
    assert torch.allclose(out, dense, rtol=1e-5, atol=1e-6)


def test_sparse_csr_matmul_matches_dense():
    act = F.relu(_rand_latent(16, 128, seed=4))
    W = _rand_latent(128, 32, seed=5)
    dense = act @ W
    out = sp.sparse_relu_matmul(act, W, layout="csr")
    assert torch.allclose(out, dense, rtol=1e-5, atol=1e-6)


def test_sparse_matches_dense_on_higher_rank():
    """BDH-like (B, nh, T, N) @ (N, D) via flatten."""
    B, nh, T, N, D = 2, 4, 8, 32, 16
    latent = _rand_latent(B, nh, T, N, seed=6)
    W = _rand_latent(N, D, seed=7)
    ref = sp.dense_relu_matmul(latent, W)
    for layout in ("coo", "csr"):
        out = sp.sparse_relu_matmul(latent, W, layout=layout)
        assert torch.allclose(out, ref, rtol=1e-5, atol=1e-6), layout


def test_masked_row_densify_matches_dense():
    act = F.relu(_rand_latent(40, 64, seed=8))
    # Force some all-zero rows
    act[0] = 0
    act[5] = 0
    W = _rand_latent(64, 24, seed=9)
    ref = act @ W
    out = sp.masked_densify_matmul(act, W)
    assert torch.allclose(out, ref, rtol=1e-5, atol=1e-6)


def test_masked_col_gather_matches_dense():
    act = F.relu(_rand_latent(20, 96, seed=10))
    # Zero some columns globally
    act[:, 3:10] = 0
    act[:, 50:60] = 0
    W = _rand_latent(96, 40, seed=11)
    ref = act @ W
    out = sp.masked_densify_matmul_gather(act, W)
    assert torch.allclose(out, ref, rtol=1e-5, atol=1e-6)


def test_forced_low_density_sparse_matches():
    """Paper-like ~5% density: sparse path must still match dense."""
    g = torch.Generator().manual_seed(12)
    raw = torch.randn(64, 256, generator=g).abs()  # positive
    act = sp.force_sparsity(raw, density=0.05, generator=torch.Generator().manual_seed(13))
    assert 0.03 < sp.relu_density(act) < 0.08
    W = _rand_latent(256, 128, seed=14)
    ref = act @ W
    out_coo = sp.sparse_relu_matmul(act, W, layout="coo")
    out_csr = sp.sparse_relu_matmul(act, W, layout="csr")
    out_row = sp.masked_densify_matmul(act, W)
    out_col = sp.masked_densify_matmul_gather(act, W)
    for name, out in [
        ("coo", out_coo),
        ("csr", out_csr),
        ("row", out_row),
        ("col", out_col),
    ]:
        assert torch.allclose(out, ref, rtol=1e-5, atol=1e-5), name


def test_product_relu_sparser_than_factors():
    x = _rand_latent(2, 4, 16, 64, seed=15)
    y = _rand_latent(2, 4, 16, 64, seed=16)
    xy, dx, dy, dxy = sp.product_relu_sparse(x, y)
    assert dxy <= dx + 1e-6
    assert dxy <= dy + 1e-6
    # Independence ≈ dx*dy for random signs post-ReLU
    assert abs(dxy - dx * dy) < 0.05


def test_decoder_path_sparse_matches_bdh_layout():
    """xy (B,nh,T,N) → permute/view → @ decoder  vs sparse / masked."""
    cfg = bdh.BDHConfig(
        n_layer=2,
        n_embd=64,
        n_head=4,
        mlp_internal_dim_multiplier=8,
        dropout=0.0,
    )
    nh = cfg.n_head
    D = cfg.n_embd
    N = cfg.mlp_internal_dim_multiplier * D // nh
    B, T = 3, 12
    x_lat = _rand_latent(B, nh, T, N, seed=20)
    y_lat = _rand_latent(B, nh, T, N, seed=21)
    xy = F.relu(x_lat) * F.relu(y_lat)
    decoder = _rand_latent(nh * N, D, seed=22)

    ref = sp.decoder_matmul_dense(xy, decoder, n_head=nh)
    out_coo = sp.decoder_matmul_sparse(xy, decoder, n_head=nh, layout="coo")
    out_csr = sp.decoder_matmul_sparse(xy, decoder, n_head=nh, layout="csr")
    out_row = sp.decoder_matmul_masked(xy, decoder, n_head=nh, mode="row")
    out_col = sp.decoder_matmul_masked(xy, decoder, n_head=nh, mode="col")

    for name, out in [
        ("coo", out_coo),
        ("csr", out_csr),
        ("row", out_row),
        ("col", out_col),
    ]:
        assert out.shape == ref.shape, name
        assert torch.allclose(out, ref, rtol=1e-5, atol=1e-5), (
            name,
            (out - ref).abs().max().item(),
        )


def test_sparse_roundtrip_q_matches_relu(monkeypatch):
    """Gated encoder probe densifies back to plain ReLU."""
    monkeypatch.setenv(sp.SPARSE_PROBE_ENV, "1")
    B, T, D, nh, N = 2, 8, 64, 4, 32
    x = _rand_latent(B, 1, T, D, seed=30)
    # encoder: (nh, D, N) as in BDH — use einsum-style via broadcast matmul
    encoder = _rand_latent(nh, D, N, seed=31)
    dense = sp.encoder_relu_matmul(x, encoder, use_sparse=False)
    for layout in ("coo", "csr"):
        rt = sp.encoder_relu_matmul(x, encoder, use_sparse=True, layout=layout)
        assert torch.allclose(rt, dense, rtol=0, atol=0), layout


def test_zeros_contribute_nothing_explicit_mask():
    """Applying an explicit ReLU support mask to dense GEMM == sparse GEMM."""
    latent = _rand_latent(24, 80, seed=40)
    act = F.relu(latent)
    mask = act > 0
    W = _rand_latent(80, 16, seed=41)
    # Masked dense: zero the inactive entries then multiply
    masked = act * mask.to(act.dtype)
    assert torch.equal(masked, act)
    ref = masked @ W
    out = sp.sparse_relu_matmul(act, W, layout="coo")
    assert torch.allclose(out, ref, rtol=1e-5, atol=1e-6)


def test_default_bdh_unaffected_import(monkeypatch):
    """BDH.forward stays dense even when the optional probe gate is on."""
    monkeypatch.setenv(sp.SPARSE_PROBE_ENV, "1")
    cfg = bdh.BDHConfig(
        n_layer=1,
        n_embd=32,
        n_head=2,
        mlp_internal_dim_multiplier=4,
        dropout=0.0,
    )
    torch.manual_seed(99)
    m = bdh.BDH(cfg).eval()
    x = torch.randint(0, 256, (2, 6))
    logits_a, _ = m(x)
    # If the production path were wired to sparse helpers this would fail.
    def unexpected_sparse_call(*args, **kwargs):
        raise AssertionError("BDH.forward must not call sparse helpers")

    monkeypatch.setattr(sp, "sparse_relu_matmul", unexpected_sparse_call)
    _ = sp.relu_density(torch.relu(torch.randn(4, 4)))
    logits_b, _ = m(x)
    assert torch.equal(logits_a, logits_b)
