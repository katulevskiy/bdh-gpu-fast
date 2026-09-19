# Copyright 2025 Pathway Technology, Inc. / private opt fork
"""Experimental ReLU-sparsity helpers for BDH (DEFAULT OFF).

Paper claim: post-ReLU activations are highly sparse (~5% density when trained).
This module explores optional sparse matmul and masked-densification paths that
are mathematically equivalent to the dense ReLU GEMMs used in ``bdh.py``:

  x_sparse = relu(x @ encoder)          # Q/K latent
  y_sparse = relu(yKV @ encoder_v)
  xy       = x_sparse * y_sparse        # even sparser
  yMLP     = xy.view(...) @ decoder     # sparse @ dense

Nothing here is wired into ``BDH.forward``. Import and opt-in explicitly.

Short-train density + CPU crossover: ``benchmarks/bench_sparse_probe.py``.
Keep this module DEFAULT OFF until a measured GPU win exists.

Strategies
----------
1. **torch.sparse** (COO / CSR): ``sparse.mm(A_sp, W)`` ≡ ``A_dense @ W``
2. **masked densification**: gather nonzero rows → dense GEMM on the subset →
   scatter back into a zero-initialized output (exact when the mask is the
   ReLU support).

Both must match dense math on random inputs when the sparsity pattern is the
true ReLU support (zeros contribute nothing).
"""

from __future__ import annotations

import os
from typing import Literal, Optional, Tuple

import torch
import torch.nn.functional as F

SparseLayout = Literal["coo", "csr"]
SPARSE_PROBE_ENV = "BDH_SPARSE_PROBE"
_TRUTHY = frozenset(("1", "true", "yes", "on"))


def sparse_probe_enabled() -> bool:
    """Return whether the explicitly opt-in sparse probe gate is enabled.

    The production ``bdh.py`` path never consults this flag. It only protects
    the optional model-hook round trip and the benchmark harness, so an
    accidental ``use_sparse=True`` cannot silently become a default change.
    """
    return os.environ.get(SPARSE_PROBE_ENV, "").strip().lower() in _TRUTHY


def require_sparse_probe_enabled() -> None:
    """Raise an actionable error when an optional sparse hook is not gated on."""
    if not sparse_probe_enabled():
        raise RuntimeError(
            f"sparse probe is gated off; set {SPARSE_PROBE_ENV}=1 "
            "for the optional, non-production path"
        )


def relu_density(x: torch.Tensor, eps: float = 0.0) -> float:
    """Fraction of elements strictly greater than ``eps`` (post-ReLU support)."""
    if x.numel() == 0:
        return 0.0
    return float((x > eps).sum().item()) / float(x.numel())


def apply_relu_sparse_stats(
    latent: torch.Tensor,
) -> Tuple[torch.Tensor, float]:
    """ReLU + density. Returns (relu(latent), density in [0, 1])."""
    out = F.relu(latent)
    return out, relu_density(out)


def to_sparse_activation(
    dense: torch.Tensor,
    layout: SparseLayout = "coo",
) -> torch.Tensor:
    """Convert a (already non-negative / ReLU'd) dense tensor to sparse 2D.

    Higher-rank inputs are flattened to ``(-1, last_dim)`` so they can feed
    ``torch.sparse.mm`` against a 2D weight ``(last_dim, out_features)``.
    """
    if dense.ndim < 2:
        raise ValueError(f"expected rank >= 2, got {tuple(dense.shape)}")
    flat = dense.reshape(-1, dense.size(-1))
    # Drop explicit zeros so the sparse tensor encodes the ReLU support.
    if layout == "coo":
        return flat.to_sparse_coo().coalesce()
    if layout == "csr":
        return flat.to_sparse_csr()
    raise ValueError(f"unknown layout {layout!r}")


def sparse_dense_matmul(
    sparse_act: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    """``sparse_act @ weight`` for 2D sparse × 2D dense.

    Equivalent to ``sparse_act.to_dense() @ weight`` when ``sparse_act`` stores
    the exact nonzero pattern of a dense ReLU activation.
    """
    if not sparse_act.is_sparse and not sparse_act.is_sparse_csr:
        raise TypeError("sparse_act must be a sparse tensor (COO or CSR)")
    if weight.ndim != 2:
        raise ValueError(f"weight must be 2D, got {tuple(weight.shape)}")
    if sparse_act.size(-1) != weight.size(0):
        raise ValueError(
            f"shape mismatch: sparse ...x{sparse_act.size(-1)} @ "
            f"{tuple(weight.shape)}"
        )
    return torch.sparse.mm(sparse_act, weight)


def dense_relu_matmul(latent_or_act: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Reference dense path: treat last dim as features, ``relu(...) @ W`` if
    needed. If ``latent_or_act`` is already post-ReLU (non-negative support),
    callers may pass it directly — we still apply ReLU for safety (idempotent
    on true ReLU outputs).
    """
    act = F.relu(latent_or_act)
    flat = act.reshape(-1, act.size(-1))
    out = flat @ weight
    return out.view(*act.shape[:-1], weight.size(-1))


def sparse_relu_matmul(
    latent_or_act: torch.Tensor,
    weight: torch.Tensor,
    layout: SparseLayout = "coo",
) -> torch.Tensor:
    """Sparse path: ReLU → sparse 2D → ``sparse.mm`` → reshape to dense output.

    Numerically matches ``dense_relu_matmul`` when the sparse pattern is the
    ReLU support (zeros omitted).
    """
    act = F.relu(latent_or_act)
    shape_prefix = act.shape[:-1]
    sp = to_sparse_activation(act, layout=layout)
    out2d = sparse_dense_matmul(sp, weight)
    return out2d.view(*shape_prefix, weight.size(-1))


def masked_densify_matmul(
    act: torch.Tensor,
    weight: torch.Tensor,
    threshold: float = 0.0,
) -> torch.Tensor:
    """Masked densification: GEMM only on rows with any activation > threshold.

    For each flattened row of ``relu(act)``:
      - if the row is all ≤ threshold → output row stays zero
      - else → compute ``row @ weight`` (dense)

    Exact match to full dense ``relu(act) @ weight`` because zero rows
    contribute nothing and dropped entries within a kept row are true zeros
    under ReLU (we still multiply the full kept row, which may contain zeros —
    that is intentional densification of the *row support*, not element support).

    Element-exact variant: see ``masked_densify_matmul_gather``.
    """
    act = F.relu(act)
    flat = act.reshape(-1, act.size(-1))
    row_active = (flat > threshold).any(dim=-1)
    out = flat.new_zeros(flat.size(0), weight.size(-1))
    if row_active.any():
        out[row_active] = flat[row_active] @ weight
    return out.view(*act.shape[:-1], weight.size(-1))


def masked_densify_matmul_gather(
    act: torch.Tensor,
    weight: torch.Tensor,
    threshold: float = 0.0,
) -> torch.Tensor:
    """Element-support masked densification (gather nonzero columns per batch).

    Flattens to ``(M, K)``, builds a single shared column mask
    ``col_active[k] = any_row act[m,k] > threshold``, gathers those columns of
    ``act`` and rows of ``weight``, runs a smaller dense GEMM, and scatters
    into a zero output. Exact when inactive columns are all-zero (true for a
    global column mask derived from the same ReLU tensor).

    Note: this uses a *global* column mask across rows (cheap, GPU-friendly).
    For unstructured per-row sparsity, prefer ``sparse_relu_matmul``.
    """
    act = F.relu(act)
    flat = act.reshape(-1, act.size(-1))  # (M, K)
    col_active = (flat > threshold).any(dim=0)  # (K,)
    out = flat.new_zeros(flat.size(0), weight.size(-1))
    if not col_active.any():
        return out.view(*act.shape[:-1], weight.size(-1))
    # Gather active columns of act and matching rows of weight.
    a_sub = flat[:, col_active]  # (M, K_active)
    w_sub = weight[col_active, :]  # (K_active, N)
    out = a_sub @ w_sub
    return out.view(*act.shape[:-1], weight.size(-1))


def product_relu_sparse(
    x_latent: torch.Tensor,
    y_latent: torch.Tensor,
) -> Tuple[torch.Tensor, float, float, float]:
    """BDH-style ``relu(x) * relu(y)`` with density stats.

    Returns ``(xy, dens_x, dens_y, dens_xy)``.
    """
    x_s = F.relu(x_latent)
    y_s = F.relu(y_latent)
    xy = x_s * y_s
    return xy, relu_density(x_s), relu_density(y_s), relu_density(xy)


def decoder_matmul_dense(
    xy_sparse: torch.Tensor,
    decoder: torch.Tensor,
    n_head: int,
) -> torch.Tensor:
    """Match ``bdh.py`` layout: ``(B, nh, T, N) → (B, 1, T, nh*N) @ decoder``.

    ``xy_sparse`` is ``(B, nh, T, N)`` (post elementwise product of ReLUs).
    ``decoder`` is ``(nh*N, D)``.
    """
    B, nh, T, N = xy_sparse.shape
    assert nh == n_head
    xy = xy_sparse.permute(0, 2, 1, 3).contiguous().view(B, 1, T, N * nh)
    return xy @ decoder


def decoder_matmul_sparse(
    xy_sparse: torch.Tensor,
    decoder: torch.Tensor,
    n_head: int,
    layout: SparseLayout = "coo",
) -> torch.Tensor:
    """Sparse equivalent of ``decoder_matmul_dense`` (same permute/view)."""
    B, nh, T, N = xy_sparse.shape
    assert nh == n_head
    xy = xy_sparse.permute(0, 2, 1, 3).contiguous().view(B, 1, T, N * nh)
    # xy is already non-negative (product of ReLUs); still safe under ReLU.
    return sparse_relu_matmul(xy, decoder, layout=layout)


def decoder_matmul_masked(
    xy_sparse: torch.Tensor,
    decoder: torch.Tensor,
    n_head: int,
    mode: Literal["row", "col"] = "col",
) -> torch.Tensor:
    """Masked-densification equivalent of ``decoder_matmul_dense``."""
    B, nh, T, N = xy_sparse.shape
    assert nh == n_head
    xy = xy_sparse.permute(0, 2, 1, 3).contiguous().view(B, 1, T, N * nh)
    if mode == "row":
        return masked_densify_matmul(xy, decoder)
    if mode == "col":
        return masked_densify_matmul_gather(xy, decoder)
    raise ValueError(f"unknown mode {mode!r}")


def force_sparsity(
    dense: torch.Tensor,
    density: float,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Randomly zero elements so that approx ``density`` fraction remain.

    Useful for synthetic low-density benches (paper ~5%). Keeps magnitude of
    surviving entries; not a training op.
    """
    if not 0.0 <= density <= 1.0:
        raise ValueError("density must be in [0, 1]")
    mask = torch.rand(dense.shape, generator=generator, device=dense.device) < density
    return dense * mask.to(dense.dtype)


# ---------------------------------------------------------------------------
# Optional model hook (still DEFAULT OFF — caller must pass use_sparse=True)
# ---------------------------------------------------------------------------

def encoder_relu_matmul(
    x: torch.Tensor,
    encoder: torch.Tensor,
    *,
    use_sparse: bool = False,
    layout: SparseLayout = "coo",
) -> torch.Tensor:
    """``relu(x @ encoder)`` with optional sparse materialization check path.

    The encoder multiply itself is dense (``x`` is LayerNorm'd embeddings, not
    sparse). Sparsity appears *after* ReLU. This helper exists so callers can
    round-trip through sparse storage and densify for the attention Q/K path
    while proving ``to_dense()`` matches the plain ReLU tensor.
    """
    latent = x @ encoder
    act = F.relu(latent)
    if not use_sparse:
        return act
    require_sparse_probe_enabled()
    # Round-trip: dense → sparse → dense must be exact for the ReLU support.
    sp = to_sparse_activation(act, layout=layout)
    restored = sp.to_dense().view_as(act)
    return restored
