"""Tests for strict tril(diagonal=-1) score×V kernel API.

CPU reference always runs. Native CUDA tests use an explicit, actionable
skip reason when the extension or CUDA device is unavailable — expected on CPU
boxes; no GPU claim is implied by the CPU mirrors.

Import of ``kernels.cuda_attn`` must never raise when ``bdh_cuda_ext`` is
missing (build smoke / soft-fail).
"""

from __future__ import annotations

import pytest
import torch

from kernels.cuda_attn import (
    CUDA_TILE_M,
    CUDA_TILE_M_CPU_MAX,
    CUDA_TILE_M_MAX,
    CUDA_TILE_N,
    CUDA_TILE_N_CPU_MAX,
    CUDA_TILE_N_MAX,
    ext_status,
    has_cuda_ext,
    has_cuda_kernel,
    pick_cuda_cold_tiles,
    tril_score_v,
    tril_score_v_ref,
    tril_score_v_tiled_ref,
)


def _naive(q, k, v):
    """Inline golden: identical math to bdh.Attention cold path."""
    scores = (q @ k.transpose(-2, -1)).tril(diagonal=-1)
    return scores @ v


def _cuda_skip_reason() -> str | None:
    """Return an actionable reason for skipping native CUDA tests."""
    if not torch.cuda.is_available():
        return "CUDA unavailable: torch.cuda.is_available() is false (CPU-safe skip)"
    if not has_cuda_ext():
        return "CUDA extension unavailable: bdh_cuda_ext is not built (CPU-safe skip)"
    if not has_cuda_kernel():
        return "CUDA kernel unavailable: bdh_cuda_ext has_cuda=False (CPU-safe skip)"
    return None


_CUDA_SKIP_REASON = _cuda_skip_reason()


@pytest.fixture(params=["cpu"])
def device(request):
    return torch.device(request.param)


def test_cuda_skip_diagnostic_is_actionable():
    """CPU runs expose why native CUDA coverage is skipped."""
    if _CUDA_SKIP_REASON is None:
        pytest.skip("native CUDA available; diagnostic contract is not needed")
    assert "CPU-safe skip" in _CUDA_SKIP_REASON
    print(f"native CUDA tests: SKIP: {_CUDA_SKIP_REASON}")


def test_import_fails_soft():
    """Build smoke: module imports even when native ext is absent."""
    import kernels.cuda_attn as ca

    # Soft import — never raises; status string always usable.
    s = ca.ext_status()
    assert isinstance(s, str) and len(s) > 0
    if not ca.has_cuda_ext():
        assert "unavailable" in s or "native=" in s
        # Dispatch still works via CPU ref.
        q = torch.randn(1, 2, 4, 8)
        k = torch.randn(1, 2, 4, 8)
        v = torch.randn(1, 2, 4, 4)
        out = ca.tril_score_v(q, k, v)
        assert out.shape == (1, 2, 4, 4)


def test_ext_status_prints():
    # Smoke: status string is non-empty either way.
    s = ext_status()
    assert isinstance(s, str) and len(s) > 0
    print("cuda_attn:", s)


def test_cpu_ref_matches_naive(device):
    torch.manual_seed(0)
    B, H, T, Dk, Dv = 2, 4, 8, 16, 32
    q = torch.randn(B, H, T, Dk, device=device)
    k = torch.randn(B, H, T, Dk, device=device)
    v = torch.randn(B, H, T, Dv, device=device)
    out = tril_score_v_ref(q, k, v)
    gold = _naive(q, k, v)
    assert out.shape == (B, H, T, Dv)
    assert torch.allclose(out, gold, rtol=0, atol=0)  # bit-identical eager


def test_cpu_ref_broadcast_v_heads(device):
    """Match bdh.Attention: V is (B, 1, T, D) and broadcasts over heads."""
    torch.manual_seed(1)
    B, H, T, Dk, Dv = 2, 4, 7, 8, 16
    q = torch.randn(B, H, T, Dk, device=device)
    k = q.clone()
    v = torch.randn(B, 1, T, Dv, device=device)
    out = tril_score_v_ref(q, k, v)
    gold = _naive(q, k, v)
    assert out.shape == (B, H, T, Dv)
    assert torch.allclose(out, gold, rtol=0, atol=0)


def test_diagonal_excluded_position0_zero(device):
    torch.manual_seed(2)
    B, H, T, Dk, Dv = 1, 2, 5, 8, 4
    q = torch.randn(B, H, T, Dk, device=device)
    k = torch.randn(B, H, T, Dk, device=device)
    v = torch.randn(B, H, T, Dv, device=device)
    out = tril_score_v_ref(q, k, v)
    assert torch.allclose(out[:, :, 0, :], torch.zeros_like(out[:, :, 0, :]))


def test_tril_score_v_dispatch_cpu(device):
    """Public dispatcher must match ref on CPU (ext or fallback)."""
    torch.manual_seed(3)
    q = torch.randn(1, 2, 6, 8, device=device)
    k = torch.randn(1, 2, 6, 8, device=device)
    v = torch.randn(1, 2, 6, 4, device=device)
    assert torch.allclose(tril_score_v(q, k, v), tril_score_v_ref(q, k, v), rtol=1e-5, atol=1e-5)


def test_t1_all_zeros(device):
    q = torch.randn(2, 3, 1, 8, device=device)
    k = torch.randn(2, 3, 1, 8, device=device)
    v = torch.randn(2, 3, 1, 5, device=device)
    out = tril_score_v_ref(q, k, v)
    assert torch.allclose(out, torch.zeros_like(out))


def test_tiled_ref_matches_eager(device):
    """CUDA-mirror tiled online CPU ref ≡ golden eager (bit-close)."""
    torch.manual_seed(10)
    B, H, T, Dk, Dv = 2, 3, 20, 8, 12
    q = torch.randn(B, H, T, Dk, device=device)
    k = torch.randn(B, H, T, Dk, device=device)
    v = torch.randn(B, H, T, Dv, device=device)
    tiled = tril_score_v_tiled_ref(q, k, v)
    gold = tril_score_v_ref(q, k, v)
    assert tiled.shape == gold.shape
    assert torch.allclose(tiled, gold, rtol=1e-5, atol=1e-5)
    assert torch.allclose(tiled[:, :, 0, :], torch.zeros_like(tiled[:, :, 0, :]))


def test_tiled_ref_broadcast_v_and_multi_tile(device):
    """T spanning >1 CUDA TILE_M + V=(B,1,...) broadcast."""
    torch.manual_seed(11)
    assert CUDA_TILE_M == 16 and CUDA_TILE_N == 16  # base constants
    B, H, T, Dk, Dv = 1, 4, 40, 8, 16  # T > 2 * TILE_M
    q = torch.randn(B, H, T, Dk, device=device)
    k = torch.randn(B, H, T, Dk, device=device)
    v = torch.randn(B, 1, T, Dv, device=device)
    tiled = tril_score_v_tiled_ref(q, k, v)
    gold = tril_score_v_ref(q, k, v)
    assert torch.allclose(tiled, gold, rtol=1e-5, atol=1e-5)
    # Custom smaller tiles still match
    tiled4 = tril_score_v_tiled_ref(q, k, v, tile_m=4, tile_n=4)
    assert torch.allclose(tiled4, gold, rtol=1e-5, atol=1e-5)


def test_tiled_ref_t0_and_t1(device):
    q = torch.randn(1, 2, 0, 8, device=device)
    k = torch.randn(1, 2, 0, 8, device=device)
    v = torch.randn(1, 2, 0, 4, device=device)
    assert tril_score_v_tiled_ref(q, k, v).shape == (1, 2, 0, 4)

    q1 = torch.randn(1, 2, 1, 8, device=device)
    k1 = torch.randn(1, 2, 1, 8, device=device)
    v1 = torch.randn(1, 1, 1, 4, device=device)
    out = tril_score_v_tiled_ref(q1, k1, v1)
    assert torch.allclose(out, torch.zeros_like(out))


@pytest.mark.skipif(not has_cuda_ext(), reason="bdh_cuda_ext not built (pip install -e .)")
def test_native_cpu_matches_ref():
    torch.manual_seed(4)
    q = torch.randn(2, 2, 5, 8)
    k = torch.randn(2, 2, 5, 8)
    v = torch.randn(2, 2, 5, 4)
    import bdh_cuda_ext

    out = bdh_cuda_ext.tril_score_v_cpu(q, k, v)
    assert torch.allclose(out, tril_score_v_ref(q, k, v), rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(
    _CUDA_SKIP_REASON is not None,
    reason=_CUDA_SKIP_REASON or "native CUDA unavailable",
)
def test_cuda_kernel_matches_ref():
    torch.manual_seed(5)
    device = torch.device("cuda")
    q = torch.randn(2, 4, 16, 32, device=device)
    k = torch.randn(2, 4, 16, 32, device=device)
    v = torch.randn(2, 4, 16, 64, device=device)
    out = tril_score_v(q, k, v)
    gold = tril_score_v_ref(q, k, v)
    assert torch.allclose(out, gold, rtol=1e-3, atol=1e-3)


@pytest.mark.skipif(
    _CUDA_SKIP_REASON is not None,
    reason=_CUDA_SKIP_REASON or "native CUDA unavailable",
)
def test_cuda_broadcast_v():
    torch.manual_seed(6)
    device = torch.device("cuda")
    q = torch.randn(1, 4, 12, 16, device=device)
    k = q.clone()
    v = torch.randn(1, 1, 12, 32, device=device)
    out = tril_score_v(q, k, v)
    gold = tril_score_v_ref(q, k, v)
    assert out.shape == gold.shape
    assert torch.allclose(out, gold, rtol=1e-3, atol=1e-3)


def test_cpu_ref_larger_t_no_diag(device):
    """CPU golden still holds for T that would span multiple CUDA tiles (16)."""
    torch.manual_seed(7)
    B, H, T, Dk, Dv = 1, 2, 33, 8, 16
    q = torch.randn(B, H, T, Dk, device=device)
    k = torch.randn(B, H, T, Dk, device=device)
    v = torch.randn(B, H, T, Dv, device=device)
    out = tril_score_v_ref(q, k, v)
    gold = _naive(q, k, v)
    assert torch.allclose(out, gold, rtol=0, atol=0)
    assert torch.allclose(out[:, :, 0, :], torch.zeros_like(out[:, :, 0, :]))
    # Tiled path also matches across multi-tile span
    assert torch.allclose(
        tril_score_v_tiled_ref(q, k, v), gold, rtol=1e-5, atol=1e-5
    )


@pytest.mark.skipif(
    _CUDA_SKIP_REASON is not None,
    reason=_CUDA_SKIP_REASON or "native CUDA unavailable",
)
def test_cuda_tiled_cold_matches_ref_multi_tile():
    """Tiled cold kernel (TILE_M=16) vs ref across >1 query tile + V broadcast."""
    torch.manual_seed(8)
    device = torch.device("cuda")
    B, H, T, Dk, Dv = 2, 4, 40, 32, 48
    q = torch.randn(B, H, T, Dk, device=device)
    k = torch.randn(B, H, T, Dk, device=device)
    v = torch.randn(B, 1, T, Dv, device=device)
    out = tril_score_v(q, k, v)
    gold = tril_score_v_ref(q, k, v)
    assert out.shape == gold.shape
    assert torch.allclose(out, gold, rtol=1e-3, atol=1e-3)
    assert torch.allclose(out[:, :, 0, :], torch.zeros_like(out[:, :, 0, :]))


def test_pick_cuda_cold_tiles_long_t():
    """cuda-cold-v2: adaptive tiles grow with T (pair #75 BS@T≥256)."""
    assert CUDA_TILE_M == 16 and CUDA_TILE_N == 16
    tm, tn = pick_cuda_cold_tiles(32)
    assert tm == 16 and tn == 16
    tm, tn = pick_cuda_cold_tiles(128)
    assert tm == 32 and tn == 32
    tm, tn = pick_cuda_cold_tiles(512)
    assert tm == 128 and tn == 128  # CPU refs up to 128
    assert tm <= CUDA_TILE_M_CPU_MAX and tn <= CUDA_TILE_N_CPU_MAX
    # Smem-aware caps at MAX (32/64)
    tm_s, tn_s = pick_cuda_cold_tiles(512, Dk=64, for_smem=True)
    assert tm_s <= CUDA_TILE_M_MAX and tn_s <= CUDA_TILE_N_MAX
    assert tm_s >= CUDA_TILE_M and tn_s >= CUDA_TILE_N
    # Explicit override
    tm_e, tn_e = pick_cuda_cold_tiles(512, tile_m=16, tile_n=16)
    assert tm_e == 16 and tn_e == 16


def test_pick_cuda_cold_tiles_wide_head_bound():
    """cuda-cold-v3 mirrors Triton #108's bounded wide-head policy."""
    # CPU mirror keeps long wide-head tiles at the Triton 64x64 bound rather
    # than applying the normal 128x128 long-T preference.
    tm, tn = pick_cuda_cold_tiles(512, Dk=128, Dv=256)
    assert (tm, tn) == (64, 64)
    # CUDA smem mirror keeps the mid-size 32x32 policy and remains bounded.
    tm_s, tn_s = pick_cuda_cold_tiles(512, Dk=128, Dv=256, for_smem=True)
    assert (tm_s, tn_s) == (32, 32)
    # Explicit overrides remain authoritative for wide heads.
    assert pick_cuda_cold_tiles(
        512, Dk=128, Dv=256, tile_m=16, tile_n=16
    ) == (16, 16)


def test_tiled_ref_wide_head_matches_eager(device):
    """Wide-head CPU tile bound preserves strict tril score×V parity."""
    torch.manual_seed(24)
    B, H, T, Dk, Dv = 1, 1, 260, 128, 256
    q = torch.randn(B, H, T, Dk, device=device)
    k = torch.randn(B, H, T, Dk, device=device)
    v = torch.randn(B, 1, T, Dv, device=device)
    tm, tn = pick_cuda_cold_tiles(T, Dk, Dv=Dv)
    assert (tm, tn) == (64, 64)
    tiled = tril_score_v_tiled_ref(q, k, v)
    gold = tril_score_v_ref(q, k, v)
    assert torch.allclose(tiled, gold, rtol=1e-4, atol=1e-4)
    assert torch.count_nonzero(tiled[:, :, 0, :]) == 0


def test_tiled_ref_long_t_adaptive_matches_eager(device):
    """Long-T adaptive tiled CPU ref ≡ eager; diagonal excluded; no full T×T claim."""
    torch.manual_seed(20)
    B, H, T, Dk, Dv = 1, 2, 96, 8, 12  # T>64 → TM/TN=32
    q = torch.randn(B, H, T, Dk, device=device)
    k = torch.randn(B, H, T, Dk, device=device)
    v = torch.randn(B, H, T, Dv, device=device)
    tm, tn = pick_cuda_cold_tiles(T, Dk)
    assert tm == 32 and tn == 32
    tiled = tril_score_v_tiled_ref(q, k, v)
    gold = tril_score_v_ref(q, k, v)
    assert tiled.shape == gold.shape
    assert torch.allclose(tiled, gold, rtol=1e-5, atol=1e-5)
    assert torch.allclose(tiled[:, :, 0, :], torch.zeros_like(tiled[:, :, 0, :]))
    # Fixed small tiles still match
    tiled16 = tril_score_v_tiled_ref(q, k, v, tile_m=16, tile_n=16)
    assert torch.allclose(tiled16, gold, rtol=1e-5, atol=1e-5)


def test_tiled_ref_long_t_broadcast_v_matches_blocked(device):
    """T≥256 adaptive tiles + V broadcast ≡ blocked (#75) and eager."""
    from kernels.attention import blocked_tril_attn

    torch.manual_seed(21)
    B, H, T, Dk, Dv = 1, 2, 260, 8, 8
    q = torch.randn(B, H, T, Dk, device=device)
    k = torch.randn(B, H, T, Dk, device=device)
    v = torch.randn(B, 1, T, Dv, device=device)
    tm, tn = pick_cuda_cold_tiles(T, Dk)
    assert tm == 128 and tn == 128
    tiled = tril_score_v_tiled_ref(q, k, v)
    gold = tril_score_v_ref(q, k, v)
    blocked = blocked_tril_attn(q, k, v)
    assert torch.allclose(tiled, gold, rtol=1e-4, atol=1e-4)
    assert torch.allclose(tiled, blocked, rtol=1e-4, atol=1e-4)
    assert torch.allclose(tiled[:, :, 0, :], torch.zeros_like(tiled[:, :, 0, :]))


def test_tril_score_v_dispatch_long_t_uses_tiled(device):
    """Large T soft-fallback prefers tiled online (no ext) vs eager golden."""
    torch.manual_seed(22)
    # T*T > 256*256 → tiled path in tril_score_v
    T = 288
    q = torch.randn(1, 1, T, 8, device=device)
    k = torch.randn(1, 1, T, 8, device=device)
    v = torch.randn(1, 1, T, 4, device=device)
    out = tril_score_v(q, k, v)
    gold = tril_score_v_ref(q, k, v)
    assert torch.allclose(out, gold, rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(
    _CUDA_SKIP_REASON is not None,
    reason=_CUDA_SKIP_REASON or "native CUDA unavailable",
)
def test_cuda_adaptive_cold_long_t_matches_ref():
    """Soft-skip without GPU: adaptive cold CUDA vs eager @ long T + V broadcast."""
    torch.manual_seed(23)
    device = torch.device("cuda")
    B, H, T, Dk, Dv = 1, 2, 96, 32, 48
    q = torch.randn(B, H, T, Dk, device=device)
    k = torch.randn(B, H, T, Dk, device=device)
    v = torch.randn(B, 1, T, Dv, device=device)
    out = tril_score_v(q, k, v)
    gold = tril_score_v_ref(q, k, v)
    assert out.shape == gold.shape
    assert torch.allclose(out, gold, rtol=1e-3, atol=1e-3)
    assert torch.allclose(out[:, :, 0, :], torch.zeros_like(out[:, :, 0, :]))
