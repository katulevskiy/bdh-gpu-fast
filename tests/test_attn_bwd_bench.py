"""CPU-safe contract tests for the CUDA-only analytic attention harness."""

from __future__ import annotations

from benchmarks import bench_attn_bwd


def test_bench_matrix_covers_all_impl_autograd_pairs():
    assert bench_attn_bwd.IMPLS == ("eager", "blocked", "online", "triton", "cuda")
    assert bench_attn_bwd.AUTOGRAD_FLAGS == ("0", "1")


def test_bench_dtype_aliases():
    assert bench_attn_bwd._parse_dtype("fp32")[0] == "float32"
    assert bench_attn_bwd._parse_dtype("BF16")[0] == "bfloat16"
    assert bench_attn_bwd._parse_dtype("half")[0] == "float16"


def test_bench_skips_cleanly_without_cuda(monkeypatch, capsys):
    monkeypatch.setattr(bench_attn_bwd.torch.cuda, "is_available", lambda: False)
    assert bench_attn_bwd.main([]) == 0
    assert "SKIP: CUDA unavailable" in capsys.readouterr().out


def test_bench_nograd_skip_reason_is_explicit():
    exc = RuntimeError("element 0 of tensors does not require grad")
    reason = bench_attn_bwd._nograd_skip_reason("0", "cuda", exc)
    assert reason == (
        "cuda forward exposes no autograd graph with AUTOGRAD=0: "
        "element 0 of tensors does not require grad"
    )
    assert bench_attn_bwd._nograd_skip_reason("1", "cuda", exc) is None
    assert bench_attn_bwd._nograd_skip_reason("0", "blocked", exc) is None
