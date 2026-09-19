"""CPU-safe packed multi-query decode GEMM contract coverage."""

import pytest
import torch

from kernels.attention import _DECODE_ONESHOT_ELEMS, eager_decode_attn
from kernels.attention_dispatch import bdh_attn_decode


@pytest.mark.parametrize("v_heads", [1, 4])
def test_packed_multi_query_decode_gemm_parity(v_heads):
    """Decode GEMMs preserve packed strides for a multi-query past scan."""
    B, H, Tq, N, D = 2, 4, 3, 8, 16
    S = _DECODE_ONESHOT_ELEMS + 17
    capacity = S + 11
    g = torch.Generator().manual_seed(804 + v_heads)

    q = torch.randn(B, H, Tq, N, generator=g)
    k_buf = torch.randn(B, H, capacity, N, generator=g)
    v_buf = torch.randn(B, v_heads, capacity, D, generator=g)
    k = k_buf.narrow(2, 0, S)
    v = v_buf.narrow(2, 0, S)

    assert k.stride() == (H * capacity * N, capacity * N, N, 1)
    assert v.stride() == (
        v_heads * capacity * D,
        capacity * D,
        D,
        1,
    )

    ref = eager_decode_attn(q, k, v)
    for impl in ("blocked", "online", "triton"):
        got = bdh_attn_decode(q, k, v, impl=impl, block_size=64)
        assert got.shape == (B, H, Tq, D)
        assert torch.allclose(got, ref, rtol=1e-4, atol=1e-5), (
            f"impl={impl} v_heads={v_heads} "
            f"maxdiff={(got - ref).abs().max().item()}"
        )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("v_heads", [1, 4])
def test_packed_multi_query_decode_gemm_dtype_parity(dtype, v_heads):
    """Packed multi-query decode preserves dtype while matching eager GEMMs."""
    B, H, Tq, N, D = 2, 4, 3, 8, 16
    S = 33
    capacity = S + 11
    g = torch.Generator().manual_seed(814 + v_heads)

    q = torch.randn(B, H, Tq, N, generator=g, dtype=dtype)
    k_buf = torch.randn(B, H, capacity, N, generator=g, dtype=dtype)
    v_buf = torch.randn(B, v_heads, capacity, D, generator=g, dtype=dtype)
    k = k_buf.narrow(2, 0, S)
    v = v_buf.narrow(2, 0, S)

    ref = eager_decode_attn(q, k, v)
    for impl in ("blocked", "online", "triton"):
        got = bdh_attn_decode(q, k, v, impl=impl, block_size=64)
        assert got.shape == (B, H, Tq, D)
        assert got.dtype == dtype
        assert torch.allclose(got, ref, rtol=1e-3, atol=1e-2), (
            f"impl={impl} dtype={dtype} v_heads={v_heads} "
            f"maxdiff={(got - ref).abs().max().item()}"
        )
