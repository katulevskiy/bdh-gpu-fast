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
| `kernels/attention_dispatch.py` | `BDH_ATTN_IMPL` + `bdh_attn()` (see opt/attn-unify for `cuda`) |
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
# 74 passed, 4 skipped (post-rebase onto main w/ triton+sparse)
#   test_cuda_attn: 6 passed (CPU ref), 3 skipped (no bdh_cuda_ext / no CUDA)
```

### Honest status
- **Positive:** CPU reference is correct and ready; CUDA `.cu` scaffold is in-tree
  for a real GPU + matching nvcc/torch toolchain.
- **This machine:** CPU-only (`cuda=False`). Default `pip install -e .` does not
  compile. `BDH_BUILD_EXT=1` fails here (torch 2.14 headers vs g++ 14
  `at::symint::sizes` error) — documented, not blocking.
- **Wired via `BDH_ATTN_IMPL=cuda`** in `opt/attn-unify` (cold path only; see that section).
  Default training path remains `eager` until GPU validation.
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

## opt/cache-pack — packed KR/V cache (2026-09-19)

### Change
- New `bdh_cache.CacheManager`: preallocate `(B, nh, max_seq, N)` / `(B, 1, max_seq, D)`
  per layer; `append` writes slices; `commit` advances `seq_len` once after all layers.
- `bdh.BDH.forward` accepts `CacheManager` **or** legacy `list` of `{'kr','v'}` (cat path kept).
- `generate()` uses packed cache with `max_seq = prompt + max_new_tokens`.
  Optional `cache_dtype=torch.float16` stores half; `get_past` casts to fp32 for RoPE score GEMMs.

### Correctness
```text
.venv/bin/python -m pytest tests/test_cache_pack.py tests/test_correctness.py \
  tests/test_attention_mask.py tests/test_vs_baseline.py -q
# cache_pack: 9 passed; core suite green
```
Packed prefill/tokenwise matches full forward and legacy cat cache (atol 1e-5 / exact where applicable).

### fp16 storage numerics
- Prefill + short decode logits: **within 1e-4** vs fp32 storage (fp32 compute).
- 16-step tokenwise accumulation measured **~1.3e-4** max abs logit diff → allow 2e-4 in test; still ~1e-4 order.
- Default `generate()` keeps fp32 storage (exact). Pass `cache_dtype=torch.float16` for half.

### Benchmark (`benchmarks/bench_cache_mem.py`, CPU)
```text
decode prompt=64 + new=128, layers=4 d=128
legacy cat-cache median:  179.27 ms
packed fp32 median:       225.29 ms  (0.80× — slower on CPU this size)
packed fp16 storage:      204.15 ms  (0.88×)
legacy final cache bytes: 12_976_128
packed fp32 prealloc:     12_976_128  (same final footprint; no per-step realloc growth)
packed fp16 prealloc:      6_488_064  (2.00× smaller)
```
**Merge rationale:** prealloc correctness landed; eliminates O(steps) `torch.cat` realloc/copy
of growing KR/V; fp16 option halves cache RAM. CPU wall time not improved here (copy_ +
Python overhead vs amortized cat); expect better locality/bandwidth behavior on GPU.

### Non-goals
- Still no softmax / no diagonal / no SDPA substitution.

## opt/qkv-fuse — Q/K/V proj + RoPE + attn prep allocs

**Branch:** `opt/qkv-fuse` (private `katulevskiy/bdh-gpu-opt` only).

### Goal

Cut temporary tensors on the path from encoder projection → ReLU → RoPE →
attention prep, without changing `tril(diagonal=-1)` math or `BDH_ATTN_IMPL`
dispatch.

### What changed (`bdh.py` only)

1. **Shared RoPE cis across layers** — `Attention.rope_cos_sin(T, rope_start, device)`
   once per `BDH.forward`; each layer gets the same `(cos, sin)` via
   `Attention.forward(..., cos_sin=...)`. Removes per-layer `arange` /
   `remainder` / `cos` / `sin` (≈`n_layer×` before).

2. **Fused RoPE into `out=`** — pairwise even/odd rotate writes straight into
   one `empty_like` result; no full-size `v_rot` buffer and no final
   `v*cos + v_rot*sin` extra tensor. `out=` optional; `rope(phases, v)` API
   unchanged for tests. fp32 path bit-identical to baseline; mixed-dtype keeps
   baseline cast-then-add rounding.

3. **In-place ReLU on Q and encoder_v projections** —
   `F.relu(x @ encoder, inplace=True)` / same for `encoder_v` — one buffer
   instead of GEMM out + separate ReLU out.

4. **Attention cold path** — `BDH_ATTN_IMPL=eager|blocked|triton|cuda` via
   unified dispatch (`opt/attn-unify`); eager still
   `tril(diagonal=-1)` then `@ V`.

### Profiler delta (CPU, 4 layers, B=4 T=128 d=128, 5× forward)

| Op (approx calls) | Before | After |
|-------------------|--------|-------|
| `cos` / `sin` / `remainder` | 20 each | **5** each (once/forward) |
| `neg` (v_rot half) | 20 | **0** |
| `copy_` | 180 | **150** |
| `arange` (RoPE positions) | 40 | **10** |

Wall medians on this CPU box are noisy; the win is fewer RoPE/trig/copy
allocs, not a new GEMM kernel. Full TxT score materialization remains the
big GPU opportunity (`opt/triton-attn`).

## opt/attn-unify — single dispatch for eager|blocked|triton|cuda (2026-09-19)

**Branch:** `opt/attn-unify` (private `katulevskiy/bdh-gpu-opt` only).

### Goal

One clean cold-path dispatch for all attention backends behind `BDH_ATTN_IMPL`,
including the CUDA scaffold from `opt/cuda-ext`. Preserve exact
`tril(diagonal=-1)` math (no softmax, no `1/sqrt(d)`). Default remains **eager**.
Compatible with `opt/qkv-fuse` (shared `cos_sin`, inplace ReLU) and
`opt/cache-pack` (`CacheManager`).

### What landed

| Path | Change |
|------|--------|
| `kernels/attention_dispatch.py` | `resolve_attn_impl` / `bdh_attn` accept `eager|blocked|triton|cuda`; `backend_info()` reports effective backend |
| `bdh.py` `Attention.forward` | Cold path (`past_kr is None`) always calls `bdh_attn`; docstring documents cache behavior |
| `kernels/__init__.py`, `kernels/README.md` | Export `backend_info`; document four impls + generate caveat |
| `tests/test_attn_unify.py` | Env switching, cuda→ref parity, cold-path hook, cache-path ignores impl |

### Wire-up

```bash
export BDH_ATTN_IMPL=eager     # default — full T×T then tril_
export BDH_ATTN_IMPL=blocked   # tiled pure PyTorch
export BDH_ATTN_IMPL=triton    # Triton on CUDA; blocked on CPU
export BDH_ATTN_IMPL=cuda      # kernels.cuda_attn (ext if built, else ref)
```

### generate / KV-cache honesty

- **Cold prefill** (empty cache / training): respects `BDH_ATTN_IMPL`.
- **Incremental decode** (`past_kr` set, including every `generate()` step after
  prefill): **always eager PyTorch**. Blocked / Triton / CUDA kernels do not
  yet implement incremental score×V over `concat(past, new)`. Documented in
  `Attention.forward` and `kernels/attention_dispatch.py` module docstring —
  not a silent fallback.

### Correctness

```text
.venv/bin/python -m pytest tests/ -q
# 81 passed, 4 skipped (CUDA/native/Triton GPU) on CPU-only box
# test_attn_unify: env resolve, all 4 impls match eager, cold hook, cache ignores impl
```

### Non-goals

- No Cursor cloud agents
- No PRs to `pathwaycom/bdh`
- No merge of this PR from the agent
- No softmax / diagonal inclusion / SDPA

## opt/compile-train (train loop)

**Owns:** `train.py`, optional `train_fast.py`, `benchmarks/bench_train_step.py`.
Did **not** change `bdh.py` core.

### Goals

torch.compile-friendly training loop, `zero_grad(set_to_none=True)`, fused AdamW
when available, fewer graph-break footguns, better dataloader prefetch, CUDA-graph
notes for when a GPU exists.

### Changes

1. **`optimizer.zero_grad(set_to_none=True)`** via `train_step()` — frees grad
   storage instead of filling zeros (friendlier to CUDA graphs / allocator).
2. **Fused AdamW** — `make_optimizer()` tries `fused=True`, falls back on error.
   Env: `BDH_FUSED_ADAMW=0` to disable.
3. **Compile path** — `maybe_compile()` wraps `torch.compile(mode=BDH_COMPILE_MODE)`
   and **probes** with the first batch (inductor errors often appear only then).
   Falls back to eager. Env: `BDH_COMPILE=0`, `BDH_COMPILE_MODE=default|reduce-overhead|max-autotune`.
4. **Graph-break hygiene** — logging detaches loss; GradScaler only when
   fp16+CUDA; batch fetch / print stay outside the compiled module.
   (`opt/train-fuse` further defers `.item()` to `LOG_FREQ`.)
5. **`BatchPrefetcher`** — one-slot prefetch; on CUDA uses a side stream for
   pin+`non_blocking` H2D overlapped with the previous step.
6. **Reusable `_offsets`** arange for window gather (reset-safe if `BLOCK_SIZE` changes).
7. **`train_fast.py`** — optional entry: defaults to `reduce-overhead` on CUDA and
   documents CUDA-graph requirements (static B×T, no sync in step, fused AdamW,
   set_to_none, side-stream prefetch; keep `generate()` out of the hot loop).

### Benchmarks (CPU — honest)

```text
BDH_COMPILE=0 .venv/bin/python benchmarks/bench_train_step.py
# device=cpu  layers=2 d=64 B=4 T=64  (torch 2.14.0+cu130, cuda=False)
# train_step legacy (zero_grad fill, no fused) median: 12.75 ms
# train_step fused+set_to_none median:                 10.06 ms  (1.27×)
# zero_grad fill vs set_to_none:                       9.18 → 8.89 ms  (1.03×)

BDH_COMPILE=0 .venv/bin/python benchmarks/bench_batch.py
# get_batch baseline 0.563 ms → vectorized+pretensor 0.072 ms  (7.84×)
# (also: x/y now .contiguous() so CE targets.view(-1) works)
```

Compile microbench is opt-in (`BDH_BENCH_COMPILE=1`): inductor warmup is minutes on
CPU and needs `g++` + `python3-dev` (Python.h). No CUDA on this runner — CUDA-graph
/`reduce-overhead` wins not measured. Under heavy multi-agent CPU load medians
inflate wildly; prefer a quiet box for re-measure.

### Correctness

`pytest tests/` → **54 passed, 4 skipped** on this branch (train-only diff).
3-step smoke with fused AdamW + BatchPrefetcher OK. get_batch split semantics
unchanged; contiguous fix is value-identical.

### How to run

```bash
BDH_COMPILE=0 python train.py          # eager + fused AdamW + prefetch
BDH_COMPILE=1 python train.py          # compile with probe/fallback
BDH_MAX_ITERS=50 python train_fast.py  # aggressive defaults
python benchmarks/bench_train_step.py
BDH_BENCH_COMPILE=1 python benchmarks/bench_train_step.py
```

## opt/attn-bwd — analytic autograd for strict-tril attention (2026-09-19)

**Branch:** `opt/attn-bwd` (private `katulevskiy/bdh-gpu-opt` only).

### Goal
Give custom forward kernels (blocked / Triton / CUDA) a correct training path:
`O = tril(Q @ K.T, diagonal=-1) @ V` with **analytic** `dQ, dK, dV` — still
**no softmax / no scale / diagonal excluded**.

### What landed
| Path | Role |
|------|------|
| `kernels/attention_bwd.py` | `StrictTrilAttnFn`, `analytic_tril_attn_backward`, `strict_tril_attn` |
| `kernels/attention_dispatch.py` | `BDH_ATTN_AUTOGRAD=1` / `use_autograd_fn=` opt-in |
| `bdh.py` | Cold path: default eager unchanged; env enables Function |
| `tests/test_attn_bwd.py` | gradcheck, finite-diff, vs eager + baseline Attention |

### Math (backward)
```text
M  = tril(Q @ K.T, -1)
O  = M @ V                    # V may be (B,1,T,D) broadcast over H
dM = dO @ V_eff.T
dS = tril(dM, -1)
dQ = dS @ K
dK = dS.T @ Q
dV_eff = M.T @ dO
dV = sum_H(dV_eff) if V was head-broadcast else dV_eff
```

### Wire-up (optional)
```bash
# default — identical to previous eager cold path (PyTorch autograd through GEMMs)
unset BDH_ATTN_AUTOGRAD

# opt-in Function + analytic bwd (works with BDH_ATTN_IMPL=eager|blocked|triton)
export BDH_ATTN_AUTOGRAD=1
export BDH_ATTN_IMPL=blocked   # example
```

Or call directly:
```python
from kernels.attention_bwd import strict_tril_attn
out = strict_tril_attn(Q, K, V, impl="eager", use_fn=True)
```

### Correctness (this box, CPU-only)
```text
.venv/bin/python -m pytest tests/test_attn_bwd.py -v
# 11 passed
#   analytic == eager autograd (fp64, exact within 1e-8)
#   StrictTrilAttnFn grads == eager for impl=eager and impl=blocked
#   torch.autograd.gradcheck passed (eager + blocked, small shapes)
#   finite-diff spot-check on Q coordinate
#   Attention + BDH_ATTN_AUTOGRAD=1 grads match bdh_baseline.Attention
#   default path (env unset) still bit-identical eager

.venv/bin/python -m pytest tests/ -q
# 65 passed, 4 skipped
```

### Honest status
- **Positive:** analytic bwd proven; Function wraps eager/blocked (and Triton when
  CUDA available) so training no longer depends on differentiating the forward
  kernel graph.
- **Default unchanged:** `BDH_ATTN_AUTOGRAD` unset → bdh.py still uses inlined
  `scores.tril_ @ V` with native PyTorch autograd.
- **This machine:** CPU-only; Triton CUDA bwd not executed here (forward falls
  back to blocked under the Function).
- Still **do not** use `F.scaled_dot_product_attention`.

### Non-goals
- No PRs to `pathwaycom/bdh`
- No fused CUDA/Triton backward kernel yet (analytic PyTorch recompute of M)

## opt/train-fuse — follow-up on compile-train (2026-09-19)

**Branch:** `opt/train-fuse` (private only). Builds on `#10` / `opt/compile-train`.
Still **no** `bdh.py` attention changes; loss math and 90/10 split unchanged.

### Deltas vs main (`opt/compile-train`)

1. **`BDH_COMPILE` default `0` (opt-in)** — set `BDH_COMPILE=1` to enable
   `torch.compile`. Matches “optional compile” lane goal; CPU boxes without
   inductor stay eager without an env override. `train_fast.py` still
   `setdefault("BDH_COMPILE", "1")`.
2. **Fewer host syncs in the train loop** — accumulate `loss.detach()` on-device
   and call `.item()` only every `LOG_FREQ` (was `float(loss.detach())` every
   step, which syncs on CUDA). Same printed loss semantics.
3. **Re-measured CPU microbench** (quiet box) documented below.

### Benchmarks (CPU — honest, re-measured)

```text
BDH_COMPILE=0 .venv/bin/python benchmarks/bench_train_step.py
# device=cpu  layers=2 d=64 B=4 T=64  (torch 2.14.0+cu130, cuda=False)
# train_step legacy (zero_grad fill, no fused) median: 7.76 ms
# train_step fused+set_to_none median:                 8.21 ms  (0.95×)
# zero_grad fill vs set_to_none:                       7.69 → 7.43 ms  (1.03×)
# (quiet box; contended multi-agent runs inflate to seconds — ignore those)
```

On CPU, end-to-end step time is still ~noise vs compile-train (forward+backward
dominates; `.item()` is cheap without a device). The sync reduction matters on
**CUDA**. Do not claim GPU train speedups from these medians.

### Correctness

```text
.venv/bin/python -m pytest tests/ -q
# 74 passed, 4 skipped
```

## opt/decode-amp — decode path + optional AMP (on CacheManager)

**Branch:** `opt/decode-amp` (private `katulevskiy/bdh-gpu-opt` only).
**Base:** `5242ad6` (cache-pack CacheManager on main).

### Goals

1. Incremental decode: Attention + blocked/triton stay correct for **single-token**
   steps on top of packed `CacheManager` (no duplicate growth buffers).
2. Optional AMP / autocast (fp16/bf16) for `forward` + `generate`; default **fp32**.
3. Preserve `tril(diagonal=-1)` / no-softmax / no-scale semantics.

### What landed (coherent with cache-pack)

| Piece | Change |
|-------|--------|
| Cache growth | **Uses `bdh_cache.CacheManager`** (prealloc `max_seq`, slice writes). Dropped earlier doubling-dict helpers — overlapping with CacheManager. Legacy list path still cats. |
| `Attention.forward` incremental | Hot path `T==1, S>0`: `QR @ past_kr.mT @ past_v`. Multi-token + past + non-eager: concat + `bdh_attn` so blocked/triton match cold path. |
| AMP | `generate(..., amp_dtype=float16\|bfloat16)` wraps autocast; keeps `cache_dtype=` from cache-pack. Sampling logits cast to fp32. RoPE phases stay **float32**. Default `amp_dtype=None` → fp32. |
| `tests/test_decode_amp.py` | Decode parity under eager/blocked/triton with CacheManager; AMP vs fp32 atol/rtol; generate smoke. |

### Semantics (unchanged)

```text
out = tril(Q @ K.T, diagonal=-1) @ V   # no softmax, no 1/sqrt(d)
```

### AMP correctness (CPU autocast)

| path | dtype | atol | rtol |
|------|-------|------|------|
| forward (full) | float16 / bfloat16 | 5e-2 | 5e-2 |
| cached tokenwise decode | float16 / bfloat16 | 5e-1 | 5e-1 |

Decode accumulates step error under CPU autocast (observed bf16 max up to ~0.3).
## opt/ln-fuse — LayerNorm + residual + encoder projection temps

**Branch:** `opt/ln-fuse` (private `katulevskiy/bdh-gpu-opt` only).

### Goals

Cut temporaries on the block epilogue `LN(yMLP)` → residual add → `LN(x+y)` and
keep encoder / encoder_v `GEMM+ReLU` fused. Preserve bit-identical logits/grads
vs baseline (`torch.equal` / existing pytest). Leave `tril(-1)`, `CacheManager`,
qkv-fuse RoPE sharing, and `BDH_ATTN_IMPL` untouched.

### Changes (`bdh.py`)

1. **`_ln` / cached shape** — `F.layer_norm(x, self._ln_shape, eps=1e-5)` with
   `_ln_shape=(D,)` matching affine-free `self.ln` (module kept for API /
   state_dict compatibility; no affine params).

2. **`_residual_ln(x, y_mlp)`** — `y = LN(y_mlp); y.add_(x); return LN(y)`.
   Reuses the inner LN output buffer for the residual sum so the block no longer
   allocates a separate `x + y` tensor. Autograd-safe (in-place into LN *output*;
   backward still has `y_mlp`). Bit-identical to `LN(x + LN(y_mlp))`.

3. **`_proj_relu`** — shared helper for `F.relu(x @ weight, inplace=True)` on
   encoder and encoder_v (same fuse as `opt/qkv-fuse`).

4. **Inference `x_sparse.mul_(y_sparse)`** — when `not torch.is_grad_enabled()`,
   product reuses the Q ReLU buffer (training still uses out-of-place `*` so the
   ReLU mask stays valid for backward).

### Correctness

```text
.venv/bin/python -m pytest tests/ -q
# 92 passed, 4 skipped on this CPU-only box
```

### Benchmarks (CPU — honest)

AMP bf16 `generate` on this CPU is typically **slower** than fp32 (casts dominate).
Cache no-realloc win is owned by cache-pack `CacheManager` (see section above).
No GPU on this box — AMP is for CUDA throughput.

### Non-goals

- No Cursor cloud agents
- No PRs to `pathwaycom/bdh`
- No softmax / diagonal inclusion / scale
- No duplicate doubling cache alongside CacheManager
.venv/bin/python -m pytest tests/ -v
# 74 passed, 4 skipped
# includes test_vs_baseline (logits/grads/attn), tril(-1), CacheManager,
# BDH_ATTN_IMPL hooks, attn-bwd
```

No intentional numerical approximations (fp32 bit-identical vs prior optimized path
and vs `bdh_baseline.py` on dropout=0).

### CPU timing (honest, noisy box)

```text
.venv/bin/python benchmarks/bench_forward.py
# device=cpu  layers=4 d=128 B=4 T=128
# forward baseline median:  2269.02 ms
# forward optimized median: 2021.95 ms  (1.12× cumulative vs frozen baseline)
# generate(32) baseline:      93.04 ms
# generate(32) optimized:     31.63 ms  (2.94× — mostly KV cache from earlier opts)

# Residual LN microbench only (4× LN(y)+LN(x+y) vs LN(y);y.add_(x);LN(y), fresh tensors):
# old ~287 ms → new ~204 ms median (~1.4× on that slice)
```

Absolute ms are load-inflated on this multi-agent CPU box; treat ratios as rough.
No CUDA here — LN+residual fuse is an alloc/traffic win that should also help GPU
eager / compile, not a new kernel.

### Non-goals

- No PRs to `pathwaycom/bdh`
- No change to attention math / `BDH_ATTN_IMPL`
- No affine LN / RMSNorm swap
## opt/inc-decode — efficient single-token decode (2026-09-19)

**Branch:** `opt/inc-decode` (private `katulevskiy/bdh-gpu-opt` only).
**Base:** `5242ad6` (CacheManager + Triton/blocked + CUDA scaffold).

### Goal

Extend blocked (and triton CPU fallback) attention so **T=1 decode** against
packed KR/V cache slices never materializes a full `(S+1)×(S+1)` score matrix,
while preserving `tril(diagonal=-1)` (new token does **not** attend to itself).

### What landed

| Piece | Change |
|-------|--------|
| `kernels/attention.py` | `blocked_decode_attn`, `eager_decode_attn`, `triton_decode_attn` (CPU → blocked) |
| `kernels/attention_dispatch.py` | `bdh_attn_decode(...)` gated by `BDH_ATTN_IMPL` |
| `bdh.Attention.forward` | Hot path `T==1, S>0`: eager two-GEMM; blocked/triton → `bdh_attn_decode` on past slices. Multi-token + past + non-eager: concat + `bdh_attn` then slice new positions. |
| `tests/test_inc_decode.py` | Kernel parity, no-self-attend, incremental vs full, CacheManager path, default eager |

### Semantics

```text
# decode at absolute index S (past length S):
out = (Q @ K_past.mT) @ V_past     # all keys j < S; no self
# ≡ last row of tril(Q_all @ K_all.T, diagonal=-1) @ V_all
```

Default `BDH_ATTN_IMPL` remains **eager** (unchanged training / cold path).

### Correctness (this box, CPU)

```text
.venv/bin/python -m pytest tests/test_inc_decode.py tests/test_cache_pack.py \
  tests/test_attention_mask.py tests/test_triton_attn.py -q
# 60 passed, 1 skipped (CUDA Triton) — also ran cache_pack + attention_mask + triton_attn + correctness
```

### Honest limits

- No GPU here — Triton decode is the blocked fallback; a fused CUDA decode
  kernel is future work.
- On CPU, tiled decode is for **memory shape** / API parity, not a wall-time win
  vs the eager two-GEMM for modest S.
- Still no softmax / no scale / no SDPA.

## opt/weight-layout — contiguous encoder/decoder layouts (2026-09-19)

**Branch:** `opt/weight-layout` (private `katulevskiy/bdh-gpu-opt` only).
**Base tip:** `efb6b80` (attn-unify on main).

### Goal

Cut encoder / decoder / `mm` overhead from bad activation layouts and hot-path
`transpose` / `contiguous` copies, without changing `tril(diagonal=-1)` math,
`BDH_ATTN_IMPL`, `CacheManager`, or attn-bwd hooks. Parameter **shapes** stay
baseline-compatible (`encoder`/`encoder_v` `(nh,D,N)`, `decoder` `(nh*N,D)`,
`lm_head` `(D,V)`) so `load_state_dict` from `bdh_baseline` remains strict.

### What changed (`bdh.py`)

1. **Activations stay `(B, T, D)`** — drop the permanent `unsqueeze(1)` that
   forced broadcast GEMMs against `(nh,D,N)` and a later `permute+contiguous`
   before the decoder.
2. **Encoder / encoder_v via einsum** — `_encoder_relu` /
   `_encoder_v_relu` produce contiguous `(B, T, nh, N)`. Attention still sees
   `(B, nh, T, N)` as a **permute view** (no copy). `CacheManager.append` still
   `copy_`s into packed buffers.
3. **Decoder merge is a free view** — `xy_bthn.reshape(B, T, nh*N)` then
   `_linear` (`F.linear` on `weight.T` **view**, never `.contiguous()` on the
   transpose). Removes the per-layer `permute→contiguous→view` copy.
4. **Bias fusion hooks** — `encoder_bias` / `encoder_v_bias` / `decoder_bias` /
   `lm_head_bias` registered as `None` (not in checkpoints). When set,
   `F.linear` / add fuses them.
5. **Decode scores** — `past_kr.mT` / `QR.mT` (view) instead of
   `.transpose(-2, -1)`.

### Correctness

```text
.venv/bin/python -m pytest tests/ -q
# 110 passed, 4 skipped (CUDA/native/Triton GPU) on CPU-only box
# vs baseline: train/eval logits bit-identical; grads atol 1e-5
# tril(-1), CacheManager, BDH_ATTN_IMPL, attn-bwd hooks unchanged
```

### Benchmarks (CPU — honest)

```text
.venv/bin/python benchmarks/bench_forward.py
# device=cpu  layers=4 d=128 B=4 T=128  (torch 2.14.0+cu130, cuda=False)
# forward baseline median: ~32 ms → optimized ~32 ms  (~1.0× vs frozen baseline;
#   cumulative RoPE/cache already in tip — layout delta is alloc-path, not a new kernel)
# generate(32) baseline median: ~91 ms → optimized ~47 ms  (~1.9×; mostly prior KV cache)

# Quiet A/B vs main tip `bdh.py` (same weights): logits bit-identical; wall ~noise
# Profiler (1× forward, 4 layers): no `aten::contiguous` on decoder path;
#   `aten::einsum`×8 + `aten::linear`×5 (4× decoder + lm_head)
```

**No GPU on this box** — expect layout friendliness to matter more under
`torch.compile` / CUDA GEMM than in CPU wall medians.

### Non-goals

- No PRs to `pathwaycom/bdh`
- Do not default `BDH_ATTN_IMPL=blocked` on CPU
- No Parameter shape migration / checkpoint break
- No softmax / diagonal / SDPA

## opt/dataloader — train data path (2026-09-19)

**Branch:** `opt/dataloader` (private `katulevskiy/bdh-gpu-opt` only).
**Base:** `73d6002` (main tip). **Does not** change `bdh.py`, loss math, or 90/10 split.

### Goals

1. Keep **vectorized** `get_batch` (pre-tensorized splits + advanced-index windows).
2. **`pin_memory` + `non_blocking`** H2D when `device.type == "cuda"`.
3. Optional **`DataLoader`** with **`persistent_workers`** when `num_workers > 0`.
4. Avoid host sync in the hot loop (no `.item()` / `.cpu()` on device tensors in
   gather; logging still defers `.item()` to `LOG_FREQ` from train-fuse).

### Changes (`train.py`)

| Piece | Change |
|-------|--------|
| `_gather_batch_host` | Vectorized window gather → contiguous CPU `x,y` (no device sync) |
| `_to_train_device` | CUDA: pin if needed + `.to(..., non_blocking=True)`; CPU: plain `.to` |
| `get_batch` | gather + transfer (same public API / semantics) |
| Corpus pin | On CUDA, `_train_data` / `_val_data` are `pin_memory()`'d once at load |
| `make_torch_dataloader` | `IterableDataset` yields **full** batches (`batch_size=None`) so gather stays vectorized; `pin_memory` on CUDA; `persistent_workers=(N>0)`; `prefetch_factor=2` |
| `DataLoaderBatchSource` | Same `.next()` API as `BatchPrefetcher` |
| `make_batch_source` | Default `BatchPrefetcher`; `BDH_DATALOADER=1` → DataLoader path |
| Env | `BDH_DATALOADER=0|1`, `BDH_NUM_WORKERS` (default 2 when DataLoader on, else 0) |

`train_fast.py` uses `make_batch_source`. Hot loop still: prefetch/next during step,
accumulate `loss.detach()`, `.item()` only at `LOG_FREQ`.

### How to run

```bash
python train.py                       # BatchPrefetcher (default)
BDH_DATALOADER=1 python train.py      # DataLoader + persistent workers (N=2)
BDH_DATALOADER=1 BDH_NUM_WORKERS=4 python train.py
```
## opt/embed-tie — embedding + LM head path (2026-09-19)

**Branch:** `opt/embed-tie` (private `katulevskiy/bdh-gpu-opt` only).
**Base:** `73d6002` (ln-fuse tip on main).

### Goal

Tighten the token-embed → final vocab projection path: fewer cast/copy opportunities,
contiguous `(B,T,V)` logits, and **document** weight-tying status. Do **not** change
published attention semantics (`tril(-1)`, CacheManager, `BDH_ATTN_IMPL`, attn-bwd).

### Baseline tying status (documented)

Published / `bdh_baseline.py` keeps **separate** Parameters:

| Tensor | Shape | Role |
|--------|-------|------|
| `embed.weight` | `(V, D)` | token lookup |
| `lm_head` | `(D, V)` | `hidden @ lm_head` |

They are **not** GPT-tied (distinct storages, independent init). Enabling tying by
default would change trainable param count and break strict `load_state_dict` from
baseline checkpoints — so default remains **untied**.

### What landed (`bdh.py`)

1. **`BDHConfig.tie_weights: bool = False`** — opt-in only. When `True`,
   `lm_head` is registered `None` and logits use `F.linear(h, embed.weight)`.
2. **`_embed_tokens(idx)`** — `LN(embed(idx))` on contiguous `(B,T,D)`, then
   `unsqueeze(1)` for the residual body (bit-identical to LN-after-unsqueeze).
3. **`_vocab_logits(x)`** — squeeze to `(B,T,D)`, ensure contiguous hidden,
   then:
   - untied: `F.linear(h, lm_head.T)` (transpose **view**, no `(V,D)` clone);
     bit-identical to `h @ lm_head`
   - tied: `F.linear(h, embed.weight)`
4. **`lm_head_bias`** registered `None` (not in checkpoints) for optional
   `F.linear` epilogue later.
5. CE uses `reshape(-1, V)` on the contiguous logits buffer.

### Preserved

- `tril(diagonal=-1)` attention math
- `CacheManager` packed decode path
- `BDH_ATTN_IMPL` dispatch + attn-bwd hooks
- Default untied `state_dict` strict-load vs `bdh_baseline`

### Correctness

```text
.venv/bin/python -m pytest tests/ -q
# 139 passed, 4 skipped
# includes tests/test_dataloader.py (split 90/10, vectorized parity, DataLoader
# workers=0 / persistent_workers=2, make_batch_source, deferred .item())
```

### Benchmarks (CPU — honest)

```text
.venv/bin/python benchmarks/bench_batch.py
# device=cpu  BLOCK_SIZE=512 BATCH_SIZE=32
# get_batch baseline (list/from_numpy) median: 0.624 ms
# get_batch vectorized+pretensor median:       0.071 ms  (8.80×)
# host gather only median:                     0.068 ms
# DataLoader (workers=0) median:               0.086 ms
# DataLoader workers=2 persistent:             ~0.57 ms  (IPC overhead; not a CPU win)
```

On this CPU-only box, **default BatchPrefetcher + vectorized gather** remains the
fast path. DataLoader+workers is for overlapping host prep with **GPU** compute;
do not claim a CPU train speedup from workers. No CUDA here — pin/non_blocking
overlap not measured.
## opt/cuda-decode — CUDA scaffold T=1 decode vs packed KR/V (2026-09-19)

**Branch:** `opt/cuda-decode` (private `katulevskiy/bdh-gpu-opt` only).
**Base:** `73d6002` (ln-fuse + inc-decode + attn-unify + cuda-ext on main).

### Goal

Extend the `csrc/` CUDA/C++ scaffold with a **single-token / Tq decode**
kernel against packed past KR/V (no full `(S+1)×(S+1)` scores), preserving
`tril(diagonal=-1)` (new token does **not** attend to itself). Wire under
`BDH_ATTN_IMPL=cuda`; default stays **eager**.

### What landed

| Piece | Change |
|-------|--------|
| `csrc/tril_attn.h` | `tril_decode` / `_cpu` / `_cuda` decls |
| `csrc/tril_attn_cpu.cpp` | CPU C++ `(Q @ K_past.mT) @ V_past` |
| `csrc/tril_attn_cuda.cu` | Naive fused decode kernel (scaffold) |
| `csrc/tril_attn_bind.cpp` | Bind `tril_decode` (+ cuda symbols when `WITH_CUDA`) |
| `kernels/cuda_attn.py` | `tril_decode_ref` (always) + `tril_decode` (ext or ref) |
| `kernels/attention_dispatch.py` | `bdh_attn_decode(..., impl=cuda)` → `tril_decode` |
| `bdh.Attention.forward` | Doc/comment: T=1 + `BDH_ATTN_IMPL=cuda` uses decode path |
| `tests/test_cuda_decode.py` | CPU ref always; native CUDA skipped if no GPU/ext |

### Semantics

```text
# decode at absolute index S (past length S):
out = (Q @ K_past.mT) @ V_past     # all keys j < S; no self
# ≡ last row of tril(Q_all @ K_all.T, diagonal=-1) @ V_all
```

No softmax, no `1/√d`, no SDPA.

### Usage

```bash
export BDH_ATTN_IMPL=cuda   # cold + T=1 decode via kernels.cuda_attn
# default eager unchanged
BDH_BUILD_EXT=1 pip install -e . --no-build-isolation          # optional native
BDH_BUILD_EXT=1 BDH_BUILD_CUDA=1 pip install -e . --no-build-isolation
```

### Correctness (this box, CPU)

```text
.venv/bin/python -m pytest tests/ -q
# 141 passed, 7 skipped on this CPU-only box
# (CUDA/native decode + Triton CUDA paths skipped — no GPU / no ext)
```

### Honest limits

- **No GPU on this box** — CUDA decode kernel unmeasured; CPU ref + skip tests.
- Native ext may be absent; `tril_decode` falls back to pure PyTorch ref.
- Naive CUDA kernel is a scaffold (one thread per `(b,h,i,d)`); tiled/shared-mem later.
- Still no softmax / no scale / no SDPA.
/workspace/bdh-gpu-opt/.venv/bin/python -m pytest tests/ -q
# 135 passed, 4 skipped
# includes test_embed_tie, test_vs_baseline, tril(-1), CacheManager,
# BDH_ATTN_IMPL, attn-bwd
```

### CPU timings (honest, noisy multi-agent box)

```text
/workspace/bdh-gpu-opt/.venv/bin/python benchmarks/bench_forward.py
# device=cpu  layers=4 d=128 B=4 T=128
# forward baseline median:  36.60 ms
# forward optimized median: 27.70 ms  (1.32× cumulative vs frozen baseline)
# generate(32) baseline:      85.30 ms
# generate(32) optimized:     31.09 ms  (2.74× — mostly KV cache from earlier opts)

# Vocab-proj microbench (B=8 T=128 D=256 V=256): mm vs F.linear view ~1.00×
# (layout/API win; no new kernel). Embed+LN LN-then-unsqueeze ~parity.
```

Absolute ms vary with box load; ratios are rough. No CUDA here — contiguous
vocab GEMM + optional tie matter more under GPU / compile.

### Non-goals

- No PRs to `pathwaycom/bdh`
- No change to CE loss / tril(-1) / train=first 90% val=last 10%
- Do not default `BDH_DATALOADER=1` on CPU
- Do not default `BDH_ATTN_IMPL=cuda`

## opt/rope-cache — cached RoPE cos/sin tables (2026-09-19)

**Branch:** `opt/rope-cache` (private `katulevskiy/bdh-gpu-opt` only).
**Base:** `ea0957a` (main tip after cuda-decode #15). Preserves `tril(-1)`, `CacheManager`, `BDH_ATTN_IMPL`, weight-layout.

### Goal

Avoid regenerating RoPE phase → cos/sin every forward when sequence length `T`
is unchanged (training / full prefill). Tables are keyed by
`(T, head_dim, device, dtype)` and reused across layers (existing caller share)
and across batches.

### What changed (`bdh.py` `Attention`)

| Piece | Change |
|-------|--------|
| `_rope_cis_key` / `_rope_cis` | Single-slot cache for `rope_start=0` |
| `_rope_cis_cache_key` | `(T, head_dim, device.type, device.index, dtype)` |
| `rope_cos_sin` | Hit cache when key matches; else build, `detach()`, store |
| Decode | `rope_start!=0` still computes fresh (absolute positions) |

Unchanged: `rope()` math, baseline parity, shared `cos_sin` across layers in
`BDH.forward`, KV `CacheManager`, attention dispatch.

### Correctness

```text
.venv/bin/python -m pytest tests/ -q
# 155 passed, 7 skipped (CPU box; CUDA/native/Triton skips)
# includes tests/test_rope_cache.py — cache hit identity, vs fresh phases,
# baseline rope bit-identical, T key change, rope_start!=0 not aliased to 0,
# multi-batch forward reuses tables
```

### Benchmarks (CPU — honest)

```text
rope_cos_sin T=128 uncached median: 0.108 ms
rope_cos_sin T=128 cached   median: 0.001 ms  (~79× on the table build alone)
forward B=4 T=128 layers=4 median:  ~10.8 ms  (full step; RoPE table is a
  small slice of wall time once shared across layers — win is alloc/trig
  elimination across batches, not a new GEMM kernel)
```

No GPU on this box. Do not claim end-to-end train speedup from table cache alone.

### Non-goals

- No PRs to `pathwaycom/bdh`
- No softmax / diagonal / SDPA
- No change to weight-layout or `BDH_ATTN_IMPL` defaults

- No default weight tying (would change published param semantics)
- No attention / cache / compile changes in this branch
