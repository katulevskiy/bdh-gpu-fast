"""CPU-safe v13 contracts for packed GPU benchmark decode inputs."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "benchmarks" / "bench_gpu_attn.py"


def test_decode_reference_accepts_packed_shared_and_per_head_v():
    """Decode measurement keeps raw score×V for capacity-strided V layouts."""
    namespace: dict[str, object] = {
        "__name__": "bench_gpu_attn_v13_test",
        "__file__": str(SCRIPT),
    }
    exec(compile(SCRIPT.read_text(), str(SCRIPT), "exec"), namespace)
    torch = namespace["torch"]

    B, H, S, N, D, capacity = 2, 3, 4, 2, 2, 7
    q = torch.tensor(
        [
            [[[1.0, 2.0]], [[2.0, -1.0]], [[-1.0, 1.0]]],
            [[[3.0, 1.0]], [[-2.0, 2.0]], [[1.0, -3.0]]],
        ]
    )
    k_buf = torch.arange(B * H * capacity * N, dtype=q.dtype).reshape(
        B, H, capacity, N
    )
    k = k_buf.narrow(2, 1, S)
    assert k.stride(1) == capacity * N

    for v_heads in (1, H):
        v_buf = torch.arange(B * v_heads * capacity * D, dtype=q.dtype).reshape(
            B, v_heads, capacity, D
        )
        v = v_buf.narrow(2, 1, S)
        assert v.stride(1) == capacity * D

        expected = (q @ k.transpose(-2, -1)) @ v
        actual = namespace["BACKENDS_DECODE"]["eager"](q, k, v)

        assert torch.equal(actual, expected)
