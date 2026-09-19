# Private BDH-GPU optimization sandbox

Forked from pathwaycom/bdh for local kernel/perf work.
**Do not open PRs against pathwaycom/bdh.**

Push only to `https://github.com/katulevskiy/bdh-gpu-opt` (private).

## Environment

- Machine: CPU-only for this run (`torch 2.14.0+cu130`, `cuda=False`)
- Venv: `/workspace/bdh-gpu-opt/.venv`
- Baseline frozen as `bdh_baseline.py` (byte-identical to upstream `bdh.py` at mirror time)

## Attention semantics (critical)

Original attention is **not** Transformer SDPA:

```python
scores = (QR @ KR.mT).tril(diagonal=-1)
return scores @ V
```

| Property | Original BDH | Standard SDPA `is_causal=True` |
|----------|--------------|--------------------------------|
| Softmax | **No** | Yes |
| Scale `1/sqrt(d)` | **No** | Yes (default) |
| Diagonal | **Excluded** (`diagonal=-1`) | Included |

**Decision:** preserve exact `tril(diagonal=-1)` math. Do **not** call
`F.scaled_dot_product_attention`. Skipping the discarded upper triangle inside
a dense GEMM still requires a custom kernel (CUDA/Triton); pure PyTorch still
forms the full `T×T` product then masks. Documented as a remaining GPU
opportunity, not an intentional approx.

Cold path uses in-place `scores.tril_(diagonal=-1)` to avoid an extra alloc.

## Optimizations implemented

1. **RoPE cleanup** (`Attention.rope`)
   - Build rotated vector via indexed writes instead of `stack`→`view`
   - Skip redundant `.to(v.dtype)` when phases already match `v` (typical fp32)
   - Math verified bit-identical vs baseline in tests

2. **Inference KV-style cache** (`forward(..., cache=)`, `generate`)
   - Per layer stores RoPE'd keys `kr` and values `v`
   - Prefill: one full forward populates cache
   - Decode: single-token forward attends only to past (`j < i`), matching
     `tril(diagonal=-1)` (no self-attention on the new token)
   - `generate()` no longer recomputes the full prefix every step

3. **Layout / compile-friendly MLP merge**
   - `permute → contiguous → view` before `@ decoder` instead of
     `transpose(1,2).reshape`

4. **`train.py` batch construction**
   - Vectorized numpy window gather instead of Python list-of-slices loop

## Correctness

```text
.venv/bin/python tests/test_correctness.py
# All 7 tests passed (logits/loss exact match; cache ≈ 1e-5; generate RNG match)
```

No intentional numerical approximations.

## Benchmarks (CPU microbench)

```text
.venv/bin/python benchmarks/bench_forward.py
# device=cpu  layers=4 d=128 B=4 T=128
# forward baseline ~31 ms → optimized ~24 ms  (~1.3×)
# generate(32) baseline ~95 ms → optimized ~36 ms  (~2.6×)
```

### Honest limits

- **No GPU on this box** — GEMM/attention kernel wins not measured; expect the
  big remaining win on GPU to be a custom strict-lower-triangular score×V kernel
  (or flash-style) that never materializes the upper triangle, still without
  softmax/scale.
- Forward ~1.3× on CPU is mostly RoPE/layout; not a kernel rewrite.
- Generate ~2.6× is the real algorithmic win (cache); grows with
  `prompt_len * max_new_tokens` vs baseline O(T²) recompute.
- Cache stores full past `kr`/`v` (memory ∝ sequence length × layers).

## Non-goals / out of scope here

- PRs or pushes to `pathwaycom/bdh` or public forks
- Changing attention to softmax / including diagonal
- CUDA `.cu` kernels (upstream is pure PyTorch)

## How to re-run

```bash
cd /workspace/bdh-gpu-opt
source .venv/bin/activate
python tests/test_correctness.py
python benchmarks/bench_forward.py
```
