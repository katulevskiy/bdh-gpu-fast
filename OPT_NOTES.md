# Private BDH-GPU optimization sandbox

Forked from pathwaycom/bdh for local kernel/perf work.
**Do not open PRs against pathwaycom/bdh.**

Push only to `https://github.com/katulevskiy/bdh-gpu-opt` (private).

## Environment

- Machine: CPU-only for this run (`torch 2.14.0+cu130`, `cuda=False`)
- Venv: `/workspace/bdh-gpu-opt/.venv`
- Baseline frozen as `bdh_baseline.py` (byte-identical to upstream `bdh.py` at mirror time)

## Inefficiencies found (upstream / baseline)

1. **Attention materializes full T×T scores then masks** — `(QR @ KR.mT).tril(diagonal=-1)` still computes the discarded upper triangle; no fused lower-triangular kernel.
2. **`generate()` was O(prompt × steps) full recomputes** — each new token re-ran the entire prefix with no KV cache.
3. **RoPE allocated via `stack`→`view`** — extra temporaries; redundant `.to(v.dtype)` when phases already match.
4. **MLP path used `transpose(1,2).reshape`** — less contiguous / compile-friendly than `permute → contiguous → view`.
5. **`train.py` batching** — Python list-of-`from_numpy` per sample; re-opened memmap every batch.
6. **Not SDPA-compatible** — original attention has **no softmax**, **no 1/√d scale**, and **excludes the diagonal** (`tril(-1)`). Dropping in `F.scaled_dot_product_attention` would change semantics.

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

## What changed (optimized `bdh.py` + `train.py`)

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
   - Pre-tensorize train/val splits once from the uint8 memmap
   - Vectorized advanced-index window gather (no Python list of `from_numpy`)
   - Semantics unchanged: train=first 90% / val=last 10%

## Correctness

```text
.venv/bin/python -m pytest tests/ -v
# 14 passed in ~1–11s (first run slower)
#   test_correctness.py     (7)  — logits/loss, diagonal, RoPE, cache, generate, dropout
#   test_attention_mask.py  (3)  — tril(-1) pos0==0, incremental decode, RoPE phase continuity
#   test_vs_baseline.py     (4)  — train/eval logits, backward grads, cold Attention vs baseline
```

No intentional numerical approximations.

## Benchmarks (CPU microbench — honest numbers)

```text
.venv/bin/python benchmarks/bench_forward.py
# device=cpu  layers=4 d=128 B=4 T=128  (torch 2.14.0+cu130, cuda=False)
# forward baseline median: 34.46 ms → optimized 26.69 ms  (1.29×)
# generate(32) baseline median: 88.93 ms → optimized 38.70 ms  (2.30×)

.venv/bin/python benchmarks/bench_batch.py
# device=cpu  BLOCK_SIZE=512 BATCH_SIZE=32
# get_batch baseline (list/from_numpy): 0.572 ms → vectorized+pretensor 0.129 ms  (4.42×)
```

### Honest limits

- **No GPU on this box** — GEMM/attention kernel wins not measured; expect the
  big remaining win on GPU to be a custom strict-lower-triangular score×V kernel
  (or flash-style) that never materializes the upper triangle, still without
  softmax/scale.
- Forward ~1.3× on CPU is mostly RoPE/layout; not a kernel rewrite.
- Generate ~2.3× is the real algorithmic win (cache); grows with
  `prompt_len * max_new_tokens` vs baseline O(T²) recompute.
- Batch ~4.4× is dataloader micro-optimization; dwarfed by model forward on GPU.
- Cache stores full past `kr`/`v` (memory ∝ sequence length × layers).

## Non-goals / out of scope here

- PRs or pushes to `pathwaycom/bdh` or public forks
- Changing attention to softmax / including diagonal
- CUDA `.cu` kernels (upstream is pure PyTorch)

## How to re-run

```bash
cd /workspace/bdh-gpu-opt
source .venv/bin/activate
pip install -r requirements.txt
python -m pytest tests/ -v
python benchmarks/bench_forward.py
python benchmarks/bench_batch.py
```
