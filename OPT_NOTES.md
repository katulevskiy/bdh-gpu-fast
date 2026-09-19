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
