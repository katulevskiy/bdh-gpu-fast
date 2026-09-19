"""CPU-safe packed T=1 decode dtype contracts."""

import pytest
import torch

from kernels.attention import eager_decode_attn
from kernels.attention_dispatch import bdh_attn_decode


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("v_heads", [1, 4])
def test_packed_t1_decode_dtype_parity(dtype, v_heads):
    """T=1 cache-shaped views preserve dtype across decode dispatch."""
    B, H, N, D = 2, 4, 8, 16
    S = 33
    capacity = S + 17
    g = torch.Generator().manual_seed(914 + v_heads)

    q = torch.randn(B, H, 1, N, generator=g, dtype=dtype)
    k_buf = torch.randn(B, H, capacity, N, generator=g, dtype=dtype)
    v_buf = torch.randn(B, v_heads, capacity, D, generator=g, dtype=dtype)
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
    for impl in ("eager", "blocked", "online", "triton"):
        got = bdh_attn_decode(q, k, v, impl=impl, block_size=64)
        assert got.shape == (B, H, 1, D)
        assert got.dtype == dtype
        assert torch.allclose(got, ref, rtol=1e-3, atol=1e-2), (
            f"impl={impl} dtype={dtype} v_heads={v_heads} "
            f"maxdiff={(got - ref).abs().max().item()}"
        )
