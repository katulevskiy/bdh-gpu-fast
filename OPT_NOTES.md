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
# 14 passed
#   test_correctness.py     (7)  — logits/loss, diagonal, RoPE, cache, generate, dropout
#   test_attention_mask.py  (3)  — tril(-1) pos0==0, incremental decode, RoPE phase continuity
#   test_vs_baseline.py     (4)  — train/eval logits, backward grads, cold Attention vs baseline
```

No intentional numerical approximations.

## Benchmarks (CPU microbench — honest numbers, re-measured)

```text
.venv/bin/python benchmarks/bench_forward.py
# device=cpu  layers=4 d=128 B=4 T=128  (torch 2.14.0+cu130, cuda=False)
# forward baseline median: 35.50 ms → optimized 27.44 ms  (1.29×)
# generate(32) baseline median: 87.62 ms → optimized 34.41 ms  (2.55×)

.venv/bin/python benchmarks/bench_batch.py
# device=cpu  BLOCK_SIZE=512 BATCH_SIZE=32
# get_batch baseline (list/from_numpy): 0.598 ms → vectorized+pretensor 0.056 ms  (10.7×)
```

CPU medians vary run-to-run (~1.3× forward, ~2.2–2.6× generate, ~4–11× batch).

### Honest limits

- **No GPU on this box** — GEMM/attention kernel wins not measured; expect the
  big remaining win on GPU to be a custom strict-lower-triangular score×V kernel
  (or flash-style) that never materializes the upper triangle, still without
  softmax/scale.
- Forward ~1.3× on CPU is mostly RoPE/layout; not a kernel rewrite.
- Generate ~2.5× is the real algorithmic win (cache); grows with
  `prompt_len * max_new_tokens` vs baseline O(T²) recompute.
- Batch ~10× is dataloader micro-optimization; dwarfed by model forward on GPU.
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

## opt/sparse-relu (experimental, default OFF)

**Owned files:** `bdh_sparse.py`, `tests/test_sparse.py`, `benchmarks/bench_sparse.py`.
**Default `bdh.py` path unchanged** — sparse helpers are opt-in imports only.

### Motivation

Paper claims post-ReLU activations ≈ **5% density** when trained. BDH already uses
`relu` on encoder / encoder_v latents and the product `xy = x_sparse * y_sparse`
before `@ decoder`. Zeros contribute nothing to those GEMMs, so sparse matmul or
masked densification can replace dense `A @ W` if the sparsity pattern is the
true ReLU support.

### What was implemented

1. **`sparse_relu_matmul`** — ReLU → COO/CSR → `torch.sparse.mm` → dense out
2. **`masked_densify_matmul`** — skip all-zero *rows*
3. **`masked_densify_matmul_gather`** — drop globally inactive *columns*, smaller dense GEMM
4. **BDH decoder layout helpers** — same `permute → view → @ decoder` as `bdh.py`
5. **`force_sparsity`** — synthetic paper-like density for benches

### Correctness (positive)

```text
.venv/bin/python -m pytest tests/test_sparse.py -v
# 12 passed — COO/CSR/masked/row/col match dense ReLU GEMMs on random inputs;
# decoder-layout path matches; sparse round-trip Q matches plain ReLU;
# importing bdh_sparse does not alter BDH.forward
```

Large-K FP32: `sparse.mm` vs dense can differ at ~1e-4 abs (accumulation order);
tests use shapes where atol/rtol 1e-5 holds; bench uses 1e-3/1e-4.

### Density (this box, random init — not trained)

```text
mean x_sparse ≈ 0.50   (ReLU of ~symmetric Gaussian)
mean xy_sparse ≈ 0.25  (product of two independent ReLUs)
```

Trained ~5% density is **not** reproduced at init; synthetic `force_sparsity(0.05)`
used for low-density timing.

### Perf (CPU microbench — honest, not a win)

```text
.venv/bin/python benchmarks/bench_sparse.py
# dens~0.50: dense << coo/csr/row/col  (conversion dominates)
# dens~0.05: dense still faster on CPU for decoder-shaped GEMMs
```

**Verdict:** numerical equivalence **proven**; keep as **experimental module,
default off**. Real speedups need trained sparsity + GPU sparse kernels (or a
custom sparse×dense decoder). Do not wire into `BDH.forward` yet.

### How to re-run

```bash
python -m pytest tests/test_sparse.py -v
python benchmarks/bench_sparse.py
```

## opt/triton-attn — strict-tril score@V kernel (Triton + PyTorch)

**Branch:** `opt/triton-attn` (private `katulevskiy/bdh-gpu-opt` only).

### Semantics (unchanged)

```text
out = tril(Q @ K.T, diagonal=-1) @ V
```

- **No** softmax, **no** `1/sqrt(d)`, diagonal **excluded**.
- Not interchangeable with `F.scaled_dot_product_attention`.

### What landed

| Path | Role |
|------|------|
| `kernels/attention.py` | `eager_tril_attn` (reference), `blocked_tril_attn` (tiled, no full upper Δ), `triton_tril_attn` (CUDA Triton fused; CPU→blocked) |
| `kernels/attention_dispatch.py` | `BDH_ATTN_IMPL=eager\|triton\|blocked` + `bdh_attn()` |
| `bdh.py` | Thin cold-path hook; default **eager** (zero behavior change) |
| `tests/test_triton_attn.py` | Correctness vs eager tril(-1); CUDA kernel test skipped without GPU |
| `benchmarks/bench_triton_attn.py` | Microbench + honest CPU notes |

### Wire-up

```bash
export BDH_ATTN_IMPL=eager    # default — full T×T then tril_
export BDH_ATTN_IMPL=blocked  # tiled pure PyTorch (lower peak score memory)
export BDH_ATTN_IMPL=triton   # Triton on CUDA; blocked fallback on CPU
```

If `bdh.py` is contended: keep calling `kernels.attention_dispatch.bdh_attn` from a one-line cold-path swap (documented in that module).

### Correctness

```text
.venv/bin/python -m pytest tests/test_triton_attn.py -v
# 22 passed, 1 skipped (CUDA Triton) on CPU-only box
# blocked / triton-fallback match eager within rtol=1e-5 / atol=1e-5
# pos0 output exactly zero; V (B,1,T,D) head broadcast preserved
```

### Benchmarks (CPU — honest)

```text
.venv/bin/python benchmarks/bench_triton_attn.py
# device=cpu  B=2 H=4 T=128 N=64 D=128
# correctness max|blocked-eager|≈1e-4  (fp accumulation over tiles)
# eager   median: ~74 ms
# blocked median: ~425 ms  (0.17× — Python tile loop slower on CPU)
# triton  median: ~same as blocked (no CUDA → blocked fallback)
```

**No GPU on this box** — Triton kernel compiled in-tree but not executed; CUDA pytest is `skipif`. Expect the fused kernel to win on GPU by never writing the upper triangle and fusing score×V. On CPU, prefer `eager` (default); `blocked` is for lower peak score memory / parity testing.

### Non-goals

- No Cursor cloud agents
- No PRs to `pathwaycom/bdh`
- No softmax / diagonal inclusion

## opt/cuda-ext — native tril score×V scaffold (2026-09-19)

### Goal
CUDA/C++ extension scaffold for BDH’s strict lower-triangular attention
`(Q@K.T).tril(diagonal=-1) @ V` — **no softmax, no scale, diagonal excluded**.

### What landed
- `csrc/` — `tril_attn_cuda.cu` (naive fused kernel), CPU C++ path, pybind
- `kernels/cuda_attn.py` — public API: `tril_score_v_ref` (always) + `tril_score_v` (dispatch)
- `setup.py` / `pyproject.toml` — optional package; native compile **opt-in**
- `tests/test_cuda_attn.py` — CPU ref always asserted; CUDA/native skipped if absent
- `kernels/README.md` — build docs

### Build
```bash
pip install -e .                                 # pure Python (default)
BDH_BUILD_EXT=1 pip install -e . --no-build-isolation
BDH_BUILD_EXT=1 BDH_BUILD_CUDA=1 pip install -e . --no-build-isolation
```

### Correctness (this box)
```text
.venv/bin/python -m pytest tests/ -q
# 54 passed, 4 skipped (post-rebase onto main w/ triton+sparse)
#   test_cuda_attn: 6 passed (CPU ref), 3 skipped (no bdh_cuda_ext / no CUDA)
```

### Honest status
- **Positive:** CPU reference is correct and ready; CUDA `.cu` scaffold is in-tree
  for a real GPU + matching nvcc/torch toolchain.
- **This machine:** CPU-only (`cuda=False`). Default `pip install -e .` does not
  compile. `BDH_BUILD_EXT=1` fails here (torch 2.14 headers vs g++ 14
  `at::symint::sizes` error) — documented, not blocking.
- **Not wired into `bdh.py` cold path yet** — call
  `from kernels.cuda_attn import tril_score_v` when integrating; keeps default
  training path unchanged until GPU validation.
- Still **do not** use `F.scaled_dot_product_attention`.

## Profiler (operator-level)

Harness: `benchmarks/profile_forward.py`

```bash
cd /workspace/bdh-gpu-opt
source .venv/bin/activate
python benchmarks/profile_forward.py              # attn + forward + generate
python benchmarks/profile_forward.py --mode attn
python benchmarks/profile_forward.py --mode forward --T 256
python benchmarks/profile_forward.py --mode generate --new-tokens 32
```

- Activities: CPU always; CUDA when `torch.cuda.is_available()`.
- Chrome traces: `benchmarks/traces/bdh_{attn,forward,generate}_{cpu|cuda}.json`
  (gitignored; dir kept via `.gitkeep`). Open in `chrome://tracing` or Perfetto.
- Prints top ops by **CPU total** and **self CPU**.
- Ranked follow-ups: see `OPT_BACKLOG.md`.

### Snapshot (CPU, 2026-09-19, profiler overhead included)

Same cfg as `bench_forward.py` (4 layers, d=128, B=4, T=128; gen 16→+32).
Use **percentages**, not absolute ms (profiler inflates wall time heavily).

| Mode | Top self-CPU ops (approx) | Takeaway |
|------|---------------------------|----------|
| Attention | `mul` ~28%, `copy_` ~20%, `bmm` ~12%, RoPE trig ~20%, `tril_` ~6% | Full TxT `bmm` then mask; RoPE + copies expensive on CPU |
| Forward | `copy_` ~23%, `mul` ~16%, `bmm` ~13%, LN ~9%, `mm` ~7%, ReLU ~7% | Matmul + memory movement; compile/fuse candidates |
| Generate | `bmm` ~42%, `mm` ~28%, LN ~11%, `cat` ~10% | Incremental GEMM dominates; **cache `cat` is the memory tax** |
