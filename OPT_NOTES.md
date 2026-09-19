
# Private BDH-GPU optimization sandbox

Forked from pathwaycom/bdh for local kernel/perf work.
**Do not open PRs against pathwaycom/bdh.**

Push only to `https://github.com/katulevskiy/bdh-gpu-opt` (private).

## Environment

- Machine: CPU-only for this run (`torch 2.14.0+cu130`, `cuda=False`)
- Venv: `/workspace/bdh-gpu-opt/.venv`
- Baseline frozen as `bdh_baseline.py` (byte-identical to upstream `bdh.py` at mirror time)

## GPU measurement harness deepen (2026-09-19)

`benchmarks/bench_gpu_attn.py` now emits a structured `GPU_ATTN_SKIP` /
`GPU_ATTN_SUMMARY` record and exits 0 when CUDA is unavailable. The skip record
prints the cold, T=1 decode, dtype, and optional native-extension commands from
`OPT_BACKLOG.md`; `--json-out PATH` saves the same schema for a CUDA-box
collection pass. On CUDA, cold and decode both compare eager, blocked, online,
Triton, and CUDA against eager with bit-identical / `allclose@1e-4`, max-abs
delta, and median-ms fields.

This CPU box still has `cuda=False`: no GPU timings or wins are claimed, and
P0 remains environment-blocked pending real CUDA/Triton cold validation.
Defaults remain eager and AUTO-off; attention math remains raw
`(Q @ K.T).tril(diagonal=-1) @ V` with no softmax or scale.

## P3 profiler CI scaffold (2026-09-19)

- `benchmarks/profile_smoke.py` runs one tiny, CPU-only `torch.profiler` forward pass and writes a Chrome trace under `benchmarks/traces/`.
- The smoke is suitable for a manual/nightly CPU job: install the CPU PyTorch wheel, run `python benchmarks/profile_smoke.py`, then optionally upload `benchmarks/traces/*.json` with `actions/upload-artifact@v4` (seven-day retention is sufficient). Keep the job off pull requests; it makes no GPU performance claims.
- Traces remain gitignored; local check: `python benchmarks/profile_smoke.py`.

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

**Superseded for ranking:** early-day snapshot below still useful historically;
**post-#19–#22** numbers (cats gone, RoPE cached, fuse-scorev landed under
`blocked`) live in **§ opt/profile-v2** at tip `3d3ed2b`.

| Mode | Top self-CPU ops (approx) | Takeaway |
|------|---------------------------|----------|
| Attention | `mul` ~28%, `copy_` ~20%, `bmm` ~12%, RoPE trig ~20%, `tril_` ~6% | Full TxT `bmm` then mask; RoPE + copies expensive on CPU |
| Forward | `copy_` ~23%, `mul` ~16%, `bmm` ~13%, LN ~9%, `mm` ~7%, ReLU ~7% | Matmul + memory movement; compile/fuse candidates |
| Generate | `bmm` ~42%, `mm` ~28%, LN ~11%, `cat` ~10% | Incremental GEMM dominates; **cache `cat` was the memory tax** (gone after #20) |

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

## opt/cache-v2 — deepen CacheManager (2026-09-19)

**Branch:** `opt/cache-v2` (private `katulevskiy/bdh-gpu-opt` only).
**Base tip:** `c953c79` (main).

### Goal
Eliminate remaining `aten::cat` in `generate`, pack KR/V layer-contiguously,
optional page/block growth — keep `tril(diagonal=-1)` incremental decode.

### What landed
| Piece | Detail |
|-------|--------|
| Layer-contiguous packing | `_kr_buf` `(n_layer,B,nh,capacity,N)`, `_v_buf` `(n_layer,B,1,capacity,D)`; `_kr`/`_v` are views |
| Page growth | `page_size=` grows capacity in pages up to `max_seq`; `generate(..., cache_page_size=)` |
| `stage()` | Write new block + return contiguous past+new view (no KR/V `cat`) |
| Generate out buffer | Preallocate `(B, prompt+new)`; slice-write tokens — **0× `torch.cat`** |
| Attention multi-token+past | `empty`+`copy_` instead of `torch.cat` (eager + non-eager) |

### Correctness
```text
.venv/bin/python -m pytest tests/test_cache_pack.py tests/test_correctness.py \
  tests/test_attention_mask.py tests/test_vs_baseline.py \
  tests/test_decode_amp.py tests/test_inc_decode.py -q
# 66 passed
```

### Cat-call reduction (`benchmarks/bench_cache_mem.py`, CPU)
```text
torch.cat calls legacy decode (64+128):  1024
torch.cat calls packed decode:              0
torch.cat calls generate(16→+32):           0   (was 32 on tip after cache-pack;
                                                 was ~864 in pre-cache-pack profile)
```
Profiler generate mode: **no `aten::cat` events**.

### Benchmark (same harness)
```text
legacy cat-cache median:  219.50 ms
packed fp32 median:       194.01 ms  (1.13× vs legacy)
packed fp16 storage:      209.15 ms
packed page=64 median:    194.96 ms
packed page=64 initial:   4_325_376 bytes (grows toward max_seq)
```

### Non-goals
- Softmax / diagonal / SDPA still forbidden.
- Legacy list cache path still uses `torch.cat` (compat only).

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

## opt/dropout-fuse — compile-friendly dropout + residual path (2026-09-19)

**Branch:** `opt/dropout-fuse` (private `katulevskiy/bdh-gpu-opt` only).
**Base tip:** `28b5c45` (main after embed-tie #16 + rope-cache #18).

### Goal

Make the sparse-product dropout → decoder → residual-LN path friendlier to
`torch.compile` / `BDH_COMPILE`, without changing attention math or layouts.

### What changed (`bdh.py`)

1. **`nn.Dropout` → float `dropout_p` + `_dropout()`** — uses `F.dropout`
   (ATen / **torch RNG only**). No Python `random` / NumPy RNG side paths that
   graph-break Dynamo.
2. **`dropout_p == 0` is a true identity** — early return, **no** dropout RNG
   op in the compiled graph. Bit-identical to baseline `nn.Dropout(0)` (train
   and eval), matching `tests/test_vs_baseline.py`.
3. **Decoder merge uses `.view`** — after identity dropout the activation stays
   contiguous `(B,T,nh,N)`; active `F.dropout` also returns contiguous. Avoids
   defensive `.reshape` that could hide a layout copy.
4. **Residual path unchanged** — still `_residual_ln` (inner LN buffer reuse
   from `opt/ln-fuse`).

Preserved: `tril(diagonal=-1)`, `CacheManager`, `BDH_ATTN_IMPL`, weight-layout
encoder `(B,T,nh,N)` / decoder view + `F.linear`, embed-tie vocab path, rope-cache.

### `BDH_COMPILE` interactions

| Setting | Dropout behavior under compile |
|---------|--------------------------------|
| `BDH_COMPILE=1`, `dropout=0` (benches / parity tests) | Identity; **no** `aten::dropout` / RNG in graph. Preferred for bit-exact compare. |
| `BDH_COMPILE=1`, `dropout>0`, `model.train()` | `F.dropout` → torch generator RNG (not Python). Dynamo-friendly single op. |
| `BDH_COMPILE=1`, `model.eval()` | Dropout identity regardless of `p` (`training=False`). |
| `BDH_COMPILE=0` | Same eager semantics; default train entrypoint. |

`train.py` still compiles only the module (`maybe_compile`); batch fetch /
logging / `.item()` stay outside. `train_fast.py` defaults `BDH_COMPILE=1`.

### Correctness (this box, CPU)

```text
.venv/bin/python -m pytest tests/ -q
# (see CI / local run after rebase onto embed-tie + rope-cache)
# dropout=0: train/eval logits + grads match bdh_baseline (bit-identical / atol 1e-5)
```

### Honest limits

- No GPU here — compile graph cleanliness matters more on CUDA inductor than in
  CPU wall time; dropout=0 path is mainly for parity + smaller FX graph.
- Does not fuse dropout into a custom CUDA kernel; still ATen `native_dropout`
  when `p>0`.
- `torch.is_grad_enabled()` train/infer product branching unchanged (autograd
  vs inplace ReLU-buffer reuse).
- Still no softmax / no scale / no SDPA.

### Non-goals

- No PRs to `pathwaycom/bdh`
- No change to CE loss / tril(-1) / train=first 90% val=last 10%
- No change to `BDH_ATTN_IMPL` / `CacheManager` / Parameter layout migration
- No default weight tying (embed-tie remains opt-in)

## opt/triton-decode2 — polish blocked/Triton decode vs packed KR/V (2026-09-19)

**Branch:** `opt/triton-decode2` (private `katulevskiy/bdh-gpu-opt` only).
**Base tip:** `35564ad` (main after dropout-fuse #17).

### Goal

Polish the **single-token / Tq decode** path against packed past KR/V that
landed in `opt/inc-decode`: fewer Python tile trips, better default tile sizes,
share the past score×V helper with cold `blocked_tril_attn`, and add a dedicated
Triton decode kernel (CUDA) that skips causal masking. Preserve
`tril(diagonal=-1)` and keep **default `BDH_ATTN_IMPL=eager`**.

### What landed

| Piece | Change |
|-------|--------|
| `kernels/attention.py` | `_expand_v_heads`, `_pick_tile_size`, `_tiled_score_v` shared by decode + cold past region; `DEFAULT_BLOCK_DECODE=256` (cold stays 64); `blocked_tril_attn` past = `_tiled_score_v`; Triton `_bdh_decode_fwd_kernel` + `triton_decode_attn` (CUDA fused, else blocked) |
| `kernels/attention_dispatch.py` | `bdh_attn_decode` default tile = `DEFAULT_BLOCK_DECODE`; triton always goes through `triton_decode_attn` |
| `tests/test_inc_decode.py` | Tile picker, shared past vs cold blocked, decode ≡ last row of eager `tril(-1)`, default eager |

### Semantics (unchanged)

```text
# decode at absolute index S (past length S):
out = (Q @ K_past.mT) @ V_past     # all keys j < S; no self
# ≡ last row of tril(Q_all @ K_all.T, diagonal=-1) @ V_all
```

### Wire-up

```bash
export BDH_ATTN_IMPL=eager     # default — two-GEMM decode, full TxT cold
export BDH_ATTN_IMPL=blocked   # tiled decode + cold (shared _tiled_score_v)
export BDH_ATTN_IMPL=triton    # CUDA decode kernel; CPU → blocked decode
export BDH_ATTN_IMPL=cuda      # kernels.cuda_attn.tril_decode
```

### Correctness (this box, CPU)

```text
.venv/bin/python -m pytest tests/test_inc_decode.py tests/test_triton_attn.py \
  tests/test_attn_unify.py tests/test_attention_mask.py -q
# 81 passed, 4 skipped (CUDA paths) across inc_decode+triton_attn+attn_unify+attention_mask+cuda_decode
# blocked/triton decode match eager tril(-1) last row (rtol/atol 1e-5)
# cold blocked still matches eager; default impl remains eager
```

### Honest limits (CPU)

- **No GPU on this box** — Triton decode kernel is in-tree but unmeasured;
  `triton_decode_attn` → `blocked_decode_attn` here.
- On CPU, larger decode tiles cut **Python loop count**, not wall time vs the
  eager two-GEMM for modest S. Tiled path is for peak score-memory shape /
  API parity with the GPU path.
- Adaptive `_pick_tile_size` still caps score elems (~256²) so a pathological
  long-S decode does not allocate a full `(1×S)` if callers force tiny
  `block_size` — but the default one-shots when `Tq*S` fits the budget.
- Still no softmax / no scale / no SDPA. No PR to `pathwaycom/*`.

## opt/fuse-scorev — online fused strict-tril score×V (2026-09-19)

**Branch:** `opt/fuse-scorev` (private `katulevskiy/bdh-gpu-opt` only).
**Base tip:** `35564ad` (main after dropout-fuse #17).

### Goal

Deepen the existing blocked tiles so strict-tril attention accumulates
`out[i] = sum_{j<i} (Q_i·K_j) V_j` **without materializing a full T×T score
matrix**. Preserve `tril(diagonal=-1)`, **no softmax**, **no `1/√d`**. Default
`BDH_ATTN_IMPL` remains **eager**.

### What changed (`kernels/attention.py`)

1. **`_accumulate_qk_v`** — past tiles fuse score×V in one expression; when
   `Bi*Bj` is large, **stream query rows** so peak score storage is `O(Bj)`
   not `O(Bi·Bj)`.
2. **`_accumulate_diag_online`** — diagonal block is **row-wise online**
   (keys `j ∈ [i0, i0+r)` for local row `r`). No `Bi×Bi` scores + `tril_`.
3. **`blocked_tril_attn`** — uses the helpers above (same public API /
   `BDH_ATTN_IMPL=blocked|triton` CPU fallback).
4. **`online_tril_attn`** — explicit alias for benches / OPT notes.
5. **`max_score_tile_elems(T, BS)`** — documents peak score-element bound
   vs eager `T*T`.

Dispatch / default unchanged: `BDH_ATTN_IMPL` default **eager**; blocked and
triton-on-CPU pick up the deepened fusion automatically.

### Correctness (this box, CPU)

```text
.venv/bin/python -m pytest tests/ -q
# 195 passed, 7 skipped (CUDA/native paths skip on this box)
```

Assertions: blocked/online match eager (`atol/rtol 1e-5`); position 0 is exact
zero; matmul spy shows eager allocates `(B,H,T,T)` scores while blocked does
**not**; default impl stays `eager`.

### Honest CPU microbench (no GPU wins claimed)

```bash
.venv/bin/python benchmarks/bench_triton_attn.py
# device=cpu  B=2 H=4 T=128 N=64 D=128  torch=2.14.0+cu130 cuda=False
# correctness max|blocked-eager|≈1.2e-4  (tile/row reorder vs one eager GEMM)
# score peak elems: eager T*T=16384  blocked/online bound=4096
# eager   median: ~0.4 ms
# blocked median: ~6.1 ms  (0.07× vs eager — slower on CPU)
# online  median: ~6.3 ms  (alias of blocked)
```

On this CPU-only box the Python tile/row loop is **slower** than eager for
modest `T` (interpreter overhead). The win here is **peak score memory**
(bound ≪ `T*T`) and a correct online algorithm for GPU kernels to mirror.
**Do not claim GPU speedups from these CPU medians.** Re-measure on CUDA.

### Non-goals

- No PRs to `pathwaycom/*`
- No softmax / diagonal inclusion / SDPA
- No change to default `BDH_ATTN_IMPL=eager`
- No fake GPU speedups from CPU profiler/bench absolute times

## opt/compile-harden — BDH_COMPILE probe + inductor CPU harden (2026-09-19)

**Branch:** `opt/compile-harden` (private `katulevskiy/bdh-gpu-opt` only).
**Base tip:** `75ce4a2` (current main after cache-v2 + triton-decode2 + fuse-scorev).

### Goal

Harden the optional `torch.compile` path (`BDH_COMPILE=1`): probe the graph
that training actually runs, document Dynamo graph-break boundaries, keep
dropout / RoPE / cache inductor-safe on **CPU**, and lock compile↔eager
parity at `dropout=0`. **No fake GPU claims** (this box: `cuda=False`).

### Probe improvements (`train.maybe_compile`)

| Knob | Default | Meaning |
|------|---------|---------|
| `BDH_COMPILE` | `0` | Opt-in compile |
| `BDH_COMPILE_MODE` | `default` | passed to `torch.compile` |
| `BDH_COMPILE_PROBE` | `train` | `eval` \| `train` \| `train_bwd` |
| `BDH_COMPILE_FULLGRAPH` | `0` | `fullgraph=True` when set |

Previously the probe ran **`eval()` + `no_grad` only**. Dynamo treats train vs
eval as **separate graphs** (dropout / `torch.is_grad_enabled()` guards) — an
eval-only probe can “succeed” then pay a second compile (or fail) on the first
`train_step`. Default probe is now **`train`** forward with targets; `train_bwd`
also runs `loss.backward()` + `zero_grad(set_to_none=True)`.

On non-CUDA devices, `mode=reduce-overhead` prints an honest note: **no CUDA
graphs** here (inductor may still run). Failures log exception type + device.

### Model harden (`bdh.py`)

1. **Hoist** `bdh_attn` / `bdh_attn_decode` imports — no lazy import inside
   `Attention.forward` (avoids a Dynamo graph break / recompile edge).
2. **RoPE cis cache** — skip `_rope_cis` Python attribute writes while
   `torch.compiler.is_compiling()` so tracing does not bake a mid-graph module
   mutation; eager / post-compile execution still warms the table.
3. **`generate()`** — `@torch.compiler.disable` (dynamic length, multinomial,
   `CacheManager` mutation). Compiled `forward` remains usable for train.

Dropout path unchanged from #17: `dropout_p==0` identity; `F.dropout` when
`p>0` (torch RNG only).

### Documented graph-break / compile boundaries

| Region | Compiled? | Notes |
|--------|-----------|-------|
| `BDH.forward` cold train (no cache, fixed B×T) | Yes (target) | `fullgraph=True` OK at dropout=0 on CPU inductor |
| `get_batch` / DataLoader / prefetch | **Outside** | Keep host gather + H2D out of the module |
| Logging `.item()` / print | **Outside** | Only every `LOG_FREQ` |
| `generate()` / variable-T sample | **Disabled** | Decorator; call after train |
| `CacheManager` decode under compile | Works on CPU | Prefill+step parity tested; still prefer eager generate |
| `BDH_ATTN_IMPL` env | Specialized at compile | Change env → need recompile |
| CUDA graphs (`reduce-overhead`) | CUDA only | Not claimed on this CPU box |

### Correctness (this box, CPU)

```text
.venv/bin/python -m pytest tests/test_compile.py tests/ -q
# compile+forward matches eager at dropout=0 (atol 1e-5; inductor float noise)
# fullgraph eval smoke; rope cache + packed cache decode parity; maybe_compile probe
```

Inductor CPU can differ from eager at ~1e-7–1e-6; tests use `atol=1e-5`, not
`torch.equal`. **No end-to-end train speedup claimed** on CPU from compile alone.

### Honest limits

- No GPU on this box — do not claim CUDA-graph or GPU inductor wins.
- `train_bwd` probe doubles compile time on first step (intentional).
- Changing `T` may recompile (Dynamo dynamic shapes); fixed `BLOCK_SIZE` train
  loop avoids that.
- Still no softmax / no scale / no SDPA / no PRs to `pathwaycom/*`.

### Non-goals

- No PRs to `pathwaycom/bdh`
- No attention math / Parameter layout changes
- No default `BDH_COMPILE=1` on `train.py` (stays opt-in; `train_fast` still
  setdefaults to 1)

## opt/profile-v2 — re-profile after #19–#22 (2026-09-19)

**Branch:** `opt/profile-v2` (private `katulevskiy/bdh-gpu-opt` only).
**Base tip:** `3d3ed2b` (main after triton-decode2 #19, cache-v2 #20, fuse-scorev #21, compile-harden #22).

### Method

```bash
.venv/bin/python benchmarks/profile_forward.py --mode all
# torch 2.14.0+cu130  cuda=False  device=cpu
# cfg: layers=4 d=128 nh=4 B=4 T=128; generate prompt=16 / new=32
```

Absolute ms are **profiler-inflated**. Rank by **% self CPU** (and call counts).
Chrome traces under `benchmarks/traces/` (gitignored).

### New % breakdown (self CPU)

| Mode | Top self-CPU ops | Takeaway vs pre-#19 snapshot |
|------|------------------|------------------------------|
| Attention | `bmm` ~36%, `mul` ~19%, `copy_` ~11%, `tril` ~7%, `sub` ~5% | Still full T×T eager score×V; RoPE trig no longer a top line (table cache #18) |
| Forward | `bmm` ~29%, `copy_` ~20%, `mm` ~12%, `mul`/`mul_` ~12%, `clamp_min_` ~4%, LN ~3.5%, `tril` ~3% | GEMM + copies dominate; compile (#22) / GPU fuse still the lever |
| Generate | `BDH.generate` ~26%, `bmm` ~20%, `copy_` ~8%, `mm` ~4%, `einsum` ~4%, `slice` ~3% | **`aten::cat` = 0** (was ~10% self / hundreds of calls). Remaining tax is decode GEMM + host copies |

### Confirmed landed (profile-visible)

- **#20 cache-v2:** generate mode has **no `aten::cat` events** in the full profiler table.
- **#21 fuse-scorev:** online/blocked path in-tree; **default `BDH_ATTN_IMPL=eager`**, so this re-profile still shows eager `bmm`+`tril` (expected).
- **#18 rope-cache / #22 compile-harden:** docs + CPU parity; not claimed as GPU wins.

### Ranked follow-ups

See updated `OPT_BACKLOG.md`: P0 = GPU measure of fused score×V; P1 = compile GPU + decode GEMM/copy; cache-cat / fuse-scorev / compile-CPU rows struck as done.

### Non-goals

- No PRs to `pathwaycom/*`
- No softmax / diagonal / SDPA
- No GPU speedup claims from these CPU % figures
## opt/bf16-train — optional train AMP via BDH_AMP_DTYPE (2026-09-19)

**Branch:** `opt/bf16-train` (private `katulevskiy/bdh-gpu-opt` only).
**Base tip:** `3d3ed2b` (compile-harden on main).

### Goal

Make mixed-precision **training** opt-in (default remains **fp32**), matching the
decode-amp philosophy. Env `BDH_AMP_DTYPE` selects autocast dtype; GradScaler
only for **fp16 + CUDA**. Preserve `tril(diagonal=-1)` / no-softmax / no-scale.

### Knobs (`train.py`)

| Env / API | Default | Meaning |
|-----------|---------|---------|
| `BDH_AMP_DTYPE` | unset → `float32` | `float32`/`fp32`/`off`, `bfloat16`/`bf16`, `float16`/`fp16`/`half` |
| `configure_amp(name)` | from env | Reconfigure module `ctx` / `scaler` (tests) |
| `cpu_bf16_available()` | — | CPU bf16 autocast smoke / gate |
| GradScaler | **off** unless `float16` **and** CUDA | bf16 never loss-scales |

Previously `train.py` auto-picked bf16/fp16 whenever CUDA was present and left
CPU on a nullcontext. Now AMP is **explicit** via env; CPU autocast runs when
requested (bf16/fp16) so smoke/parity work on this box.

### Semantics (unchanged)

```text
out = tril(Q @ K.T, diagonal=-1) @ V   # no softmax, no 1/sqrt(d)
```

Attention / Parameter layout untouched. `train_step` still wraps forward in
`ctx` and branches scaler vs plain `backward`+`step`.

### AMP correctness (CPU autocast — documented atol)

| path | dtype | atol | rtol |
|------|-------|------|------|
| train forward logits + loss | float16 / bfloat16 | 5e-2 | 5e-2 |
| train grads (1× fwd+bwd) | float16 / bfloat16 | 1e-1 | 1e-1 |

Observed on this box (tiny cfg, torch 2.14 CPU): bf16 logits max ~4e-3, grads
~5e-3; fp16 lower. Table uses headroom (same order as decode-amp forward).
**Not bit-identical** — expected under autocast.

```text
.venv/bin/python -m pytest tests/test_bf16_train.py -v
# CPU bf16 smoke + parity vs fp32; GradScaler gate; tril(-1) smoke
```

### Honest limits

- **No GPU on this box** (`cuda=False`) — no train throughput claim for AMP.
  GradScaler path is unit-gated (`_use_scaler = dtype==float16 and cuda`) but
  not timed here.
- CPU bf16/fp16 autocast often **slower** than fp32 (cast overhead); useful for
  correctness smoke, not a speed win.
- If `BDH_AMP_DTYPE=bfloat16` on a CPU without bf16, `configure_amp` raises.
- Still no softmax / diagonal inclusion / SDPA / PRs to `pathwaycom/*`.

### Non-goals

- No default AMP on (stays float32 until env set)
- No attention math changes
- No pathwaycom PRs

## opt/triton-cold — cold-path Triton fused tril score×V (2026-09-19)

**Branch:** `opt/triton-cold` (private `katulevskiy/bdh-gpu-opt` only).
**Base tip:** `3d3ed2b` (main after compile-harden).

### Goal

Improve the **cold / full-T** Triton fused `tril(diagonal=-1)` score×V path
(not only decode): better tiles, less host staging, share helpers with the
online blocked path from `opt/fuse-scorev`. Keep **default `BDH_ATTN_IMPL=eager`**.

### What changed (`kernels/attention.py`)

| Piece | Change |
|-------|--------|
| `_as_contiguous` | Contiguous copy only when needed (shared cold + decode host) |
| `_pick_triton_cold_tiles` | Power-of-2 `BLOCK_M/N/D/K`; defaults track `DEFAULT_BLOCK_COLD=64` (was fixed 32×32) |
| `_bdh_attn_fwd_kernel` | `V_BROADCAST` — heads share values via `bh // H` (host stages `B×T×D`, not `B×H×T×D`) |
| `triton_tril_attn` | Adaptive tiles; conditional contig; V squeeze+broadcast; CPU → `blocked_tril_attn(..., DEFAULT_BLOCK_COLD)` |
| `blocked_tril_attn` / `online_tril_attn` | Default BS = `DEFAULT_BLOCK_COLD`; use shared `_expand_v_heads` |
| Decode host | Uses `_as_contiguous` / `_expand_v_heads` (same helpers; no behavior change on CPU) |

Dispatch / default unchanged: unset `BDH_ATTN_IMPL` → **eager**.

### Semantics (unchanged)

```text
out = (Q @ K.T).tril(diagonal=-1) @ V   # no softmax, no 1/√d, diagonal excluded
```

### Correctness (this box, CPU)

```text
.venv/bin/python -m pytest tests/ -q
# 223 passed, 7 skipped (CUDA/native/Triton GPU paths)
# tests/test_triton_attn.py — tile picker, V broadcast view, triton→online
#   blocked fallback ≡ eager tril(-1), default remains eager
```

### Honest CPU microbench (no GPU wins claimed)

```bash
.venv/bin/python benchmarks/bench_triton_attn.py
# device=cpu  B=2 H=4 T=128 N=64 D=128  torch=2.14.0+cu130 cuda=False
# correctness max|triton_path-eager|≈1.2e-4
# score peak elems: eager T*T=16384  blocked/online bound=4096
# eager   median: ~0.36 ms
# blocked median: ~6.0 ms  (0.06× — Python tile/row loop)
# triton  median: ~6.1 ms  (CPU → online blocked fallback; kernel not run)
```

**No GPU on this box** — cold Triton kernel is in-tree but unexecuted; measure
on CUDA before claiming speedups. On CPU prefer **eager** (default). Staging
win (no `B×H×T×D` V expand copy) and larger tiles matter on GPU.

### Non-goals

- No PRs to `pathwaycom/*`
- No softmax / diagonal inclusion / SDPA
- No change to default `BDH_ATTN_IMPL=eager`
- No fake GPU speedups from CPU medians

## opt/cuda-cold — tiled online cold CUDA tril score×V (2026-09-19)

**Branch:** `opt/cuda-cold` (private `katulevskiy/bdh-gpu-opt` only).
**Base tip:** `e5f8031` (main after triton-cold).

### Goal

Improve the `csrc/` **cold** CUDA `tril(diagonal=-1)` score×V scaffold from a
naive per-element loop toward **tiled / online accumulation** — no global
T×T score buffer. Keep CPU reference always tested; skip CUDA without a GPU.
Wire remains `BDH_ATTN_IMPL=cuda` (already via `opt/attn-unify`); default
**eager**.

### What landed

| Piece | Change |
|-------|--------|
| `csrc/tril_attn_cuda.cu` | Cold `tril_score_v_tiled_kernel`: grid `(⌈T/16⌉, B·H, ⌈Dv/32⌉)`, block `(32,16)`; shared Q/K/V tiles; register `score×V` accumulate; smem>48KiB → fused naive fallback (still no T×T) |
| `kernels/cuda_attn.py` | Docs: tiled cold vs decode scaffold; `BDH_ATTN_IMPL=cuda` |
| `kernels/README.md` | Cold kernel shape note |
| `tests/test_cuda_attn.py` | Extra CPU ref @ T=33 (multi-tile span); CUDA multi-tile+broadcast skipped w/o GPU |

Decode CUDA path unchanged (`opt/cuda-decode`). Cold dispatch already:
`bdh_attn(..., impl=cuda)` → `kernels.cuda_attn.tril_score_v`.

### Semantics (unchanged)

```text
out = (Q @ K.T).tril(diagonal=-1) @ V   # no softmax, no 1/√d, diagonal excluded
```

### Usage

```bash
export BDH_ATTN_IMPL=cuda   # cold + decode via kernels.cuda_attn
# default eager unchanged
BDH_BUILD_EXT=1 BDH_BUILD_CUDA=1 pip install -e . --no-build-isolation
```

### Correctness (this box, CPU)

```text
.venv/bin/python -m pytest tests/ -q
# 234 passed, 8 skipped (CUDA/native/Triton GPU paths)
# test_cuda_attn: CPU ref always; multi-tile T=33; CUDA tiled skipped (no GPU)
```

### Honest limits

- **No GPU on this box** — tiled CUDA kernel unexecuted; CPU ref + skip tests.
- Native ext may be absent; `tril_score_v` falls back to pure PyTorch ref
  (ref may materialize T×T; that is the golden path, not the GPU design).
- Tile sizes fixed (16×16×32); further shared-mem / occupancy tuning = GPU work.
- Still no softmax / no scale / no SDPA / no pathwaycom PRs.
- Do **not** default `BDH_ATTN_IMPL=cuda`.

### Non-goals

- No PRs to `pathwaycom/*`
- No change to default `BDH_ATTN_IMPL=eager`
- No fake GPU speedups from CPU medians

## opt/decode-copy — cut generate copy_/slice tax (2026-09-19)

**Branch:** `opt/decode-copy` (private `katulevskiy/bdh-gpu-opt` only).
**Base tip:** `d64157a` (main after cuda-cold; rebased from `e5f8031`).

### Goal

Reduce host `copy_` / view tax on `generate` after cache-v2 removed `aten::cat`.
Keep **cat-free** generate and `tril(diagonal=-1)` incremental decode.

### What landed

| Piece | Detail |
|-------|--------|
| `CacheManager.reserve(level, T)` | Writable `narrow` views at `[seq_len : seq_len+T]` (no copy) |
| In-place RoPE into KR | `Attention.forward(..., out_kr=)` writes RoPE into reserved KR when inference + storage==compute |
| V slot | Still one `copy_` into packed V (activation must move) |
| `get_past` / `stage` | `narrow` views; **never** `.contiguous()` |
| `_write_block` | `dst.copy_(src)` casts without intermediate `.to()` alloc |
| Multi-token+past | `_try_extend_seq`: zero-copy past||new when QR/V already adjacent in packed buffer; else empty+`copy_` (still cat-free) |

### Profile note (CPU hook, cfg layers=4 d=128 nh=4, prompt=16 / new=32)

| Metric | Before | After |
|--------|--------|-------|
| `Tensor.copy_` in `generate` | **265** | **133** |
| Breakdown after | 1× prompt `out.copy_` + 4×(1 prefill V + 32 decode V) = 133 | KR `copy_` eliminated via `reserve`+`out_kr` |
| `torch.cat` in `generate` | 0 | **0** (unchanged) |

Profiler `aten::copy_` still includes GEMM/epilogue internals; the hooked
`Tensor.copy_` count is the CacheManager/host write tax this branch targets.

### Correctness

```bash
.venv/bin/python -m pytest tests/test_cache_pack.py tests/test_correctness.py \
  tests/test_attention_mask.py tests/test_vs_baseline.py \
  tests/test_decode_amp.py tests/test_inc_decode.py -q
# 76 passed (incl. reserve / copy-halving / no-contiguous tests)
```

### Non-goals

- Softmax / diagonal / SDPA
- PRs to `pathwaycom/*`
- Claiming GPU wall-time wins from CPU copy counts

## opt/rope-fuse — fused RoPE rotate (2026-09-19)

**Branch:** `opt/rope-fuse` (private `katulevskiy/bdh-gpu-opt` only).
**Base tip:** `5af4f50` (main after decode-copy).

### Goal

Optional fused RoPE **rotate** that applies cached cos/sin without the strided
even/odd mul/copy storm. Keep ``rope_cos_sin`` cache by
``(T, head_dim, device, dtype)``. Default remains **eager**.

### What changed

| Piece | Change |
|-------|--------|
| `kernels/rope.py` | `eager_rope_rotate` (historical strided path), `fused_rope_rotate_pytorch` (pair-contiguous), optional Triton + analytic bwd |
| `kernels/rope_dispatch.py` | `BDH_ROPE_IMPL=eager\|fused` + `bdh_rope_rotate()` |
| `bdh.py` `Attention.rope` | Thin dispatch; still accepts `cos_sin=` from cache |
| `tests/test_rope_fuse.py` | Bit-identical fused vs eager/baseline; cache reuse; CUDA Triton skipped without GPU |

```bash
export BDH_ROPE_IMPL=eager   # default — strided even/odd (zero behavior change)
export BDH_ROPE_IMPL=fused   # pair-contiguous PyTorch; Triton on CUDA when usable
```

``rope_cos_sin`` / ``_rope_cis`` cache **unchanged**.

### Semantics (unchanged)

```text
y0 = x0 * c0 - x1 * s0
y1 = x1 * c1 + x0 * s1
```

Per-element phases (full ``N``), not classic half-dim shared-(cos,sin).

### Correctness (this box, CPU)

```text
.venv/bin/python -m pytest tests/ -q
# tests/test_rope_fuse.py — fused ≡ eager ≡ baseline (atol=0 on CPU fp32/fp16 cast path)
```

### Honest CPU microbench (no GPU wins claimed)

```text
device=cpu  B=4 H=4 T=128 N=256  torch=2.14.0+cu130 cuda=False
correctness max|fused-eager|=0
eager_rotate  median: ~67 ms
fused_pytorch median: ~75 ms  (≈0.9× — expand+stack overhead on CPU)
```

**No GPU on this box** — Triton rotate kernel is in-tree but unexecuted; CUDA
pytest is `skipif`. On CPU prefer **eager** (default). Expect fused/Triton to
matter on GPU by writing each pair once without strided `0::2`/`1::2` stores.

### Non-goals

- No PRs to `pathwaycom/*`
- No change to default `BDH_ROPE_IMPL=eager`
- No fake GPU speedups from CPU medians
- No removal of `rope_cos_sin` cache
## opt/ln-compile — compile-friendly LayerNorm + residual (2026-09-19)

**Branch:** `opt/ln-compile` (private `katulevskiy/bdh-gpu-opt` only).
**Base tip:** `d64157a` (`opt/cuda-cold` on main).

### Goal

Make the block epilogue **LayerNorm + residual** friendlier to `torch.compile` /
`BDH_COMPILE`: prefer `F.layer_norm`, avoid Python control flow / in-place
aliasing on the residual sum, keep **bit-identical** numerics vs
`nn.LayerNorm` (affine-free). No attention math changes.

### What changed (`bdh.py`)

1. **`_ln` / `_residual_ln` — `F.layer_norm` only** — explicit
   `weight=None, bias=None, eps=...`. Never call `self.ln(...)` on the hot path
   (`nn.Module.__call__` / hooks stay out of the Dynamo graph). `self.ln` remains
   registered for baseline API parity (affine-free → empty state_dict).
2. **`_residual_ln` is pure functional** — `LN(x + LN(y_mlp))` with out-of-place
   `x + y` instead of in-place `y.add_(x)`. Same numerics as ln-fuse (#12);
   cleaner for AOTAutograd functionalization / compile.
3. **Drop `torch.is_grad_enabled()` sparse-product branch** — always
   `x_bthn * y_bthn` (out-of-place). Removes a Python branch that specialized
   train vs `no_grad` into **two** Dynamo graphs next to the residual LN
   epilogue. (`grad_enabled` is still read for packed-cache in-place RoPE
   safety from `opt/decode-copy` — not for the residual product.)
4. **`_ln_eps`** sourced once from `float(self.ln.eps)` at init.

Preserved: `tril(diagonal=-1)`, `CacheManager`, `BDH_ATTN_IMPL`, dropout
identity at `p=0`, encoder `(B,T,nh,N)` / decoder `F.linear`, embed-tie.

### `BDH_COMPILE` / tests

| Check | Expectation |
|-------|-------------|
| `_residual_ln` vs `ln(x+ln(y))` | **bit-identical** (`torch.equal`) |
| `torch.compile(..., fullgraph=True)` cold forward @ dropout=0 | matches eager @ atol 1e-5 |
| `torch._dynamo.explain` cold train @ dropout=0 | **0** graph breaks |

```text
.venv/bin/python -m pytest tests/test_compile.py tests/test_vs_baseline.py -q
```

### Honest limits

- No GPU on this box — no CUDA-graph / GPU inductor speedup claims. Win is
  graph cleanliness (fewer breaks / one less train↔no_grad specialization).
- Functional residual may allocate one extra add buffer vs ln-fuse in-place
  reuse; trade accepted for compile friendliness.
- Still no softmax / no scale / no SDPA / no PRs to `pathwaycom/*`.

### Non-goals

- No PRs to `pathwaycom/bdh`
- No change to CE loss / tril(-1) / Parameter layouts
- No default `BDH_COMPILE=1` on `train.py`

## opt/compile-bench — CPU BDH_COMPILE=0 vs 1 train-step (2026-09-19)

**Branch:** `opt/compile-bench` (private `katulevskiy/bdh-gpu-opt` only).
**Base tip:** `410656f` (rope-fuse); merged `origin/main` through `opt/ln-compile`.

### Goal

Land an **honest** CPU train-step microbench comparing `BDH_COMPILE=0` (eager)
vs `BDH_COMPILE=1` (`torch.compile` via `train.maybe_compile`). Soft-skip when
inductor / CXX / probe is unavailable — never hard-fail the harness. Document
**GPU** as the real next measurement step in `OPT_BACKLOG.md`.

### Harness (`benchmarks/bench_train_step.py`)

| Knob | Default | Meaning |
|------|---------|---------|
| `BDH_BENCH_COMPILE` | `1` | Run 0-vs-1 section (`0` = fused/zero_grad-only) |
| `BDH_COMPILE_MODE` | `default` | Passed through `maybe_compile` |
| `BDH_COMPILE_PROBE` | `train` | Matches train loop graph |

Procedure:

1. Build identical tiny BDH (`layers=2 d=64 nh=2 B=4 T=64`, `dropout=0`).
2. **Eager arm:** plain `BDH` module (no compile); median `train_step`.
3. **Compile arm:** `USE_COMPILE=True` → `maybe_compile` + train probe; require
   `OptimizedModule._orig_mod` or soft-skip with printed reason.
4. Same AdamW (`fused` when available), same fixed batch, CUDA sync when present.

### Measured (this box, 2026-09-19 Europe/Podgorica)

Cold inductor (first process, no disk cache) can inflate compile medians into
hundreds of ms — **not** used as the steady-state claim. Warm re-run:

```text
BDH_BENCH_COMPILE=1 .venv/bin/python benchmarks/bench_train_step.py
# device=cpu layers=2 d=64 B=4 T=64  (torch 2.14.0+cu130, cuda=False)
# train_step legacy:              8.47 ms
# train_step fused+set_to_none:   8.62 ms
# zero_grad fill → set_to_none:   8.24 → 8.37 ms (~noise)
# BDH_COMPILE=0 (eager) median:     9.97 ms
# BDH_COMPILE=1 (compiled) median:  6.72 ms  (ratio eager/compiled 1.48×)
# honest: CPU-only; no GPU / CUDA-graph claim
```

Small CPU win after warm inductor cache; still **not** a GPU result. Harness
soft-skips if compile/probe falls back. GPU A/B (`reduce-overhead`) = backlog P1.

### Honest limits

- This box is CPU-only (`cuda=False`) — absolute ms are **not** GPU claims.
- Soft-skip if compile/probe falls back to eager (missing g++/Python.h/inductor).
- `reduce-overhead` / CUDA graphs remain a **GPU** follow-up (backlog P1).
- No softmax / scale / SDPA; no PRs to `pathwaycom/*`.

### Non-goals

- No default `BDH_COMPILE=1` on `train.py`
- No attention math changes
- No fake speedups from CPU inductor noise under multi-agent load
## opt/decode-gemm — incremental T=1 decode score×V vs packed KR/V (2026-09-19)

**Branch:** `opt/decode-gemm` (private `katulevskiy/bdh-gpu-opt` only).
**Base tip:** `b7979f1` (main after compile-bench; includes rope-fuse `410656f` + decode-copy + ln-compile).

### Goal

Cut generate **bmm** tax further on the blocked / Triton / CUDA decode path
against packed past KR/V (`CacheManager` slices). Keep **cat-free** generate,
`tril(diagonal=-1)` (no self-attend on the new token), and **default eager**.

### What changed

| Piece | Change |
|-------|--------|
| `kernels/attention.py` | `blocked_decode_attn` keeps broadcast `V=(B,1,S,D)` (no expand); `_tiled_score_v` docs; Triton decode **`V_BROADCAST`** + `_pick_triton_decode_tiles`; `eager_decode` uses matmul broadcast |
| `csrc/tril_attn_cuda.cu` | Decode **tiled online** (shared Q/K/V tiles, register score×V) + naive smem fallback — no `Tq×S` global scores |
| `kernels/cuda_attn.py` | `tril_decode_ref` broadcast-V without expand |
| `benchmarks/bench_gpu_attn.py` | `--mode decode` (Tq=1 vs past length `T`) |
| `tests/test_inc_decode.py` | Broadcast-V, long-S tiles, cuda dispatch ≡ eager tril(-1) last row |

```bash
export BDH_ATTN_IMPL=eager     # default — two-GEMM decode
export BDH_ATTN_IMPL=blocked   # tiled decode, broadcast V
export BDH_ATTN_IMPL=triton    # CUDA fused + V_BROADCAST; CPU → blocked
export BDH_ATTN_IMPL=cuda      # tiled CUDA decode ext / ref
```

### Semantics (unchanged)

```text
# decode at absolute index S (past length S):
out = (Q @ K_past.mT) @ V_past     # all keys j < S; no self
# ≡ last row of tril(Q_all @ K_all.T, diagonal=-1) @ V_all
```

### Correctness (this box, CPU)

```text
.venv/bin/python -m pytest tests/test_inc_decode.py tests/test_cuda_decode.py \
  tests/test_attention_mask.py tests/test_cache_pack.py -q
# 257 passed, 9 skipped (CUDA paths) — blocked/triton/cuda decode ≡ eager last row
# CacheManager still cat-free; default BDH_ATTN_IMPL=eager
```

### Honest CPU microbench (no GPU wins claimed)

```text
device=cpu  B=4 H=4 S=128 N=64 D=128  torch=2.14.0+cu130 cuda=False
correctness max|blocked-eager|=0  max|triton-eager|=0
eager_decode   median: ~48 ms
blocked_decode median: ~49 ms  (≈1.0× — tile/broadcast overhead ≈ noise)
triton_decode  median: ~49 ms  (CPU → blocked)

long S=2048 (B=1 H=2 N=32 D=64, block_size=64):
max|blocked-eager|=0
eager ~36 ms / blocked ~34 ms  (tile path; not a claimed win)
```

**No GPU on this box** — Triton `V_BROADCAST` + CUDA tiled decode are in-tree
but unexecuted; `bench_gpu_attn.py --mode decode` skips cleanly. On GPU expect
wins from (1) no `B·H·S·D` V expand staging, (2) fused score×V without
materializing `Tq×S`, (3) tiled smem CUDA vs naive O(S·Dk) per thread.

## opt/sparse-probe — ReLU density on short train + CPU crossover (2026-09-19)

**Branch:** `opt/sparse-probe` (private `katulevskiy/bdh-gpu-opt` only).
**Base tip:** `410656f` (main after rope-fuse).
**DEFAULT OFF** — `bdh.py` forward unchanged; probe uses `bdh_sparse` helpers only.

### Goal

P2 follow-through on experimental sparse ReLU path:

1. Measure post-ReLU **x / y / xy** density on a **short CPU train** (not just init).
2. Document when sparse matmul / masked densify **beats dense** on CPU (if ever).
3. Keep sparse **default off** unless a clear win appears.

### What landed

| Piece | Role |
|-------|------|
| `benchmarks/bench_sparse_probe.py` | Short train density snapshots + dense vs COO/CSR/row/col crossover sweep |
| `bdh_sparse.py` | Doc pointer only (API unchanged; still opt-in) |
| `OPT_BACKLOG.md` | P2 sparsity row → density measured on CPU |

```bash
.venv/bin/python benchmarks/bench_sparse_probe.py
.venv/bin/python benchmarks/bench_sparse_probe.py --steps 150 --log-every 25
.venv/bin/python benchmarks/bench_sparse_probe.py --skip-train   # crossover only
.venv/bin/python benchmarks/bench_sparse_probe.py --skip-crossover
```

### Short-train density (this box, CPU)

Config: `n_layer=2 n_embd=64 n_head=2 mlp_mult=16`, `B=8 T=64`, AdamW 1e-3,
tiny Shakespeare via `train.get_batch`, `torch 2.14.0+cu130`, `cuda=False`.

```text
step    loss     mean x    mean y    mean xy
0       n/a      0.5018    0.4856    0.2427
25      3.81     0.4553    0.4770    0.2417
50      3.15     0.4019    0.4846    0.2095
75      2.88     0.3202    0.4844    0.1807
100     2.65     0.2902    0.4652    0.1531
125     2.44     0.2791    0.4473    0.1305
150     2.45     0.2682    0.4292    0.1162
```

**Takeaway:** training **does** drive sparsity (x ~50%→27%, xy ~24%→12% in 150
steps), but paper-cited **~5%** is **not** reached on this short CPU probe.
`y` (encoder_v ReLU) stays ~43–49%. Do not assume 5% density at init or after
a few dozen steps.

### CPU sparse vs dense crossover

Decoder-shaped `act @ W` with synthetic `force_sparsity`, shapes
`(M,K,N) ∈ {(256,1024,128), (512,2048,128), (1024,4096,256)}`, densities
`0.50 … 0.001`. Paths: dense, COO `sparse.mm`, CSR, row-mask, col-gather.

```text
# Representative (M=256 K=1024 N=128):
dens~0.50: dense≪sparse (coo/csr/row/col all slower; conversion dominates)
dens~0.05: dense still fastest
dens~0.001: dense still fastest on mid shapes

# Large (M=1024 K=4096 N=256), dens~0.001:
# one-off flaky micro-wins for COO were NOT reproducible under the timed
# warmup/reps harness — treat as noise, not a win.
```

**Verdict (CPU):** sparse **never reliably beat dense** across the sweep, even
at 0.1% density. Row/col masked densify lose to dense GEMM + indexing overhead.
**Keep DEFAULT OFF.** Real opportunity (if any) needs trained ≪10% density **and**
GPU sparse kernels — re-measure on CUDA before wiring into `BDH.forward`.


### Non-goals

- No PRs to `pathwaycom/*`
### Non-goals

- No PRs to `pathwaycom/*`
- No change to default `BDH_ATTN_IMPL=eager`
- No re-introducing `aten::cat` in generate / CacheManager
- No change to default `bdh.py` path / no `use_sparse=True` default
- No fake GPU speedups from CPU medians


## opt/gen-bench — generate × CacheManager × attn impls (2026-09-19)

**Branch:** `opt/gen-bench` (private `katulevskiy/bdh-gpu-opt` only).
**Base tip:** `f005b3f` (`opt/sparse-probe`).

### Goal

Land an **honest** end-to-end `BDH.generate` microbench comparing
`BDH_ATTN_IMPL=eager|blocked|triton|cuda` when available. Generate always uses
packed `CacheManager` (cat-free). Soft-label Triton/CUDA fallbacks on CPU
(effective=blocked / cuda_ref). Document **GPU** as the real next measurement
step in `OPT_BACKLOG.md`.

### Harness (`benchmarks/bench_generate.py`)

| Knob | Default | Meaning |
|------|---------|---------|
| `--prompt` / `--new` | 16 / 32 | prompt length + max_new_tokens |
| `--impls` | eager,blocked,triton,cuda | comma list of backends |
| `--device` | auto | cuda if available else cpu |
| `--warmup` / `--iters` | 2 / 5 | timing reps |

Procedure:

1. Tiny-ish BDH (`layers=4 d=128 nh=4`, `dropout=0`), fixed prompt.
2. For each impl: set `BDH_ATTN_IMPL`, run `generate` with fixed seeds.
3. Report median wall ms, tok/s, tokens-match-eager, `aten::cat` count,
   and `backend_info()["effective"]`.
4. CUDA sync when present.

### Measured (this box, 2026-09-19 Europe/Podgorica)

```text
python benchmarks/bench_generate.py --warmup 2 --iters 5
# device=cpu gpu='cpu' torch=2.14.0+cu130 cuda=False
# cfg layers=4 d=128 nh=4 B=1 prompt=16 new=32
# eager   median=  51.58 ms  match_eager=yes  aten::cat=0  effective=eager
# blocked median=  51.76 ms  match_eager=yes  aten::cat=0  effective=blocked
# triton  median=  50.51 ms  match_eager=yes  aten::cat=0  effective=blocked (CPU fallback)
# cuda    median=  54.43 ms  match_eager=yes  aten::cat=0  effective=cuda_ref
# vs eager: ~1.00× / 1.02× / 0.95×  (noise — not a GPU claim)
```

### Honest limits

- This box is CPU-only (`cuda=False`) — absolute ms are **not** GPU claims.
- Triton without CUDA → blocked; cuda without ext → pure-PyTorch ref.
- Default `BDH_ATTN_IMPL` remains **eager** until A100/H100 e2e data lands.
- No softmax / scale / SDPA; no PRs to `pathwaycom/*`.

### Non-goals

- No default `BDH_ATTN_IMPL` change
- No re-introducing `aten::cat` in generate / CacheManager
- No fake GPU speedups from CPU medians
## opt/mlp-fuse — MLP temps + optional fused bias+ReLU (2026-09-19)

**Branch:** `opt/mlp-fuse` (private `katulevskiy/bdh-gpu-opt` only).
**Base tip:** `f005b3f` (main after sparse-probe).

### Goal

Tighten the MLP `permute→view→linear` path further: fewer contiguous copies on
the decoder merge, and optional fused bias+ReLU on encoder / encoder_v when
biases are set. Keep default (bias=`None`) numerics bit-identical vs tip /
`bdh_baseline`, and preserve `tril(diagonal=-1)`.

### What changed (`bdh.py`)

1. **`_bias_relu_`** — optional `out.add_(bias)` into a fresh einsum buffer, then
   in-place ReLU. Avoids the `out + bias` temporary. Bit-identical to
   `F.relu(out + bias, inplace=True)`. Default `bias=None` is plain in-place ReLU.
2. **`_encoder_relu` / `_encoder_v_relu`** — call `_bias_relu_` after einsum
   (still produce contiguous `(B,T,nh,N)` for a free decoder view).
3. **`_mlp_merge`** — consolidates `x*y` → `_dropout` → `.view(B,T,nh*N)` →
   `_linear` (`F.linear` + optional `decoder_bias` epilogue). Never calls
   unconditional `.contiguous()`; hot path stays a zero-copy view.
4. **Removed unused `_proj_relu`** — old broadcast `@` helper superseded by
   weight-layout einsum path.

Preserved: `tril(diagonal=-1)`, `CacheManager`, `BDH_ATTN_IMPL`, RoPE /
dropout / LN-compile paths, baseline Parameter shapes / `state_dict`.

### Correctness (this box, CPU)

```text
.venv/bin/python -m pytest tests/ -q
# 265 passed, 9 skipped (CUDA/native/Triton GPU) on CPU-only box
# tests/test_mlp_fuse.py — 8 passed (bias+ReLU, view/no-contiguous,
#   decoder bias fuse, vs baseline, attn pos0==0)
```

No intentional numerical approximations (fp32 bit-identical vs baseline when
biases unset / dropout=0).

### Honest limits

- No GPU on this box — bias+ReLU / view wins are alloc/epilogue cleanliness for
  eager + `torch.compile`; not a new CUDA/Triton MLP kernel.
- Optional biases remain **absent** in baseline checkpoints (`None`); fuse only
  engages when a caller registers them.
- Product `x*y` stays out-of-place (no `is_grad_enabled` branch) so Dynamo keeps
  one train/eval graph.

### Non-goals

- No PRs to `pathwaycom/*`
- No change to default `BDH_ATTN_IMPL=eager`
- No softmax / diagonal inclusion / scale
- No re-introducing `aten::cat` in generate / CacheManager

## opt/cuda-ref-v2 — deepen CUDA attn CPU refs + build smoke (2026-09-19)

**Branch:** `opt/cuda-ref-v2` (private `katulevskiy/bdh-gpu-opt` only).
**Base tip:** `f50837d` (main after mlp-fuse).

### Goal

Deepen CUDA/C++ attention scaffolds (`csrc/` + `kernels/cuda_attn.py`) for
**correctness and CPU-ref quality** so a GPU `BDH_BUILD_EXT` build is drop-in
ready: better tiled online cold+decode refs, clearer `BDH_BUILD_EXT` docs, and
build smoke that soft-fails when the native ext is missing. **No GPU speedups
claimed** (this box has no CUDA device).

### What landed

| Piece | Change |
|-------|--------|
| `kernels/cuda_attn.py` | `tril_score_v_ref` golden eager (bit-identical); **`tril_score_v_tiled_ref`** CUDA-mirror TILE_M/N=16 online cold (no full T×T); **`tril_decode_tiled_ref`** + `tril_decode_ref` tiles large past; dispatch prefers tiled for large T when native missing (less T×T staging); soft import unchanged |
| `csrc/tril_attn_cpu.cpp` | Eager vectorized for small T/S; tiled online (TILE_M/N=16) for large; `maybe_acc` — no float cast when already f32/f64; **no V expand** (matmul broadcast) |
| `csrc/tril_attn.h` | Docs for CPU tiled / staging policy |
| `setup.py` / `pyproject.toml` | Explicit `BDH_BUILD_EXT` / `BDH_BUILD_CUDA` / `BDH_FORCE_CPU_EXT` docs |
| `kernels/README.md` | Honesty: CPU refs ≠ GPU wins; tiled ref API table |
| `tests/test_cuda_attn.py` | Soft-import smoke; tiled≡eager; multi-tile + broadcast |
| `tests/test_cuda_decode.py` | Soft-import smoke; tiled decode≡eager; long-past tile branch |

### Semantics (unchanged)

```text
out = (Q @ K.T).tril(diagonal=-1) @ V   # cold — no softmax, no 1/√d
out = (Q @ K_past.mT) @ V_past          # decode — past-only, no self
```

### Build

```bash
pip install -e .                                          # pure Python (default)
BDH_BUILD_EXT=1 pip install -e . --no-build-isolation     # optional native
BDH_BUILD_EXT=1 BDH_BUILD_CUDA=1 pip install -e . --no-build-isolation
export BDH_ATTN_IMPL=cuda   # cold + decode via kernels.cuda_attn
# default BDH_ATTN_IMPL=eager unchanged
```

### Correctness (this box, CPU)

```text
.venv/bin/python -m pytest tests/test_cuda_attn.py tests/test_cuda_decode.py -q
# CPU refs + tiled mirrors always; CUDA/native skipped without GPU/ext
```

### Honest limits

- **No GPU on this box** — tiled `.cu` kernels unexecuted; CPU tiled refs validate
  the online structure that the GPU build will run.
- Native ext may be absent; `import kernels.cuda_attn` **never** raises
  (`has_cuda_ext()==False`, `ext_status()` reports soft failure).
- Tiled refs are **bit-close** to eager (same math; tile accumulation order may
  differ in ulps on long T). Golden `tril_score_v_ref` stays bit-identical.
- Still no softmax / no scale / no SDPA / no pathwaycom PRs.
- Do **not** default `BDH_ATTN_IMPL=cuda`. Do **not** claim GPU speedups.

### Non-goals

- No PRs to `pathwaycom/*`
- No change to default `BDH_ATTN_IMPL=eager`
- No fake GPU speedups from CPU medians

## opt/blocked-vec — vectorized CPU blocked/online tril attn (2026-09-19)

**Branch:** `opt/blocked-vec` (private `katulevskiy/bdh-gpu-opt` only).
**Base tip:** `c953c79` (main).

### Goal

Speed up CPU `blocked` / `online` strict-tril attention by cutting Python
tile/row loops — vectorize with torch ops while still avoiding a full `T×T`
score matrix when possible. Preserve `tril(diagonal=-1)`, no softmax / scale /
SDPA. Default `BDH_ATTN_IMPL=eager` unchanged.

### What changed

| Path | Change |
|------|--------|
| `kernels/attention.py` | `blocked_tril_attn`: past = budgeted one-shot / chunked `(Qi@K[:i0].mT)@V[:i0]`; diagonal = `Bi×Bi` + `tril(diagonal=-1)` (no Python row loop). `online_tril_attn` still an alias. `_accumulate_*` online helpers retained for low-peak reference. `max_score_tile_elems` documents new peak (past `Bi·i0` budget-capped + `Bi×Bi` diag), still ≪ `T×T` for mid T. |
| `benchmarks/bench_blocked_vec.py` | CPU microbench eager vs blocked/online for T∈{32,64,128,256}, tiny B/H. |
| `tests/test_fuse_scorev.py` | Bound assertion updated for vectorized peak (still `< T×T`). |

### Correctness (this box, CPU)

```text
.venv/bin/python -m pytest tests/test_fuse_scorev.py tests/test_triton_attn.py -q
# 65 passed, 1 skipped
```

Bit-close vs eager at existing atol/rtol (`1e-4`). Position 0 stays exact zero.
Matmul spy still: eager allocates `(B,H,T,T)` scores; blocked does not.

### Measured CPU medians (B=1 H=2 N=32 D=64, CPU, OMP≈2)

**Before** (old online-row blocked, same shapes, clean run before change):

| T | eager ms | blocked ms (old) | eager/old_blk |
|---|----------|------------------|---------------|
| 32 | 0.024 | 0.941 | 0.03× |
| 64 | 0.032 | 1.873 | 0.02× |
| 128 | 0.052 | 3.716 | 0.01× |
| 256 | 0.098 | 9.192 | 0.01× |

**After** (vectorized blocked):

| T | eager ms | blocked ms (new) | eager/new_blk | vs old blocked |
|---|----------|------------------|---------------|----------------|
| 32 | 0.021 | 0.034 | 0.61× | **~28×** faster |
| 64 | 0.027 | 0.052 | 0.51× | **~36×** faster |
| 128 | 0.066 | 0.205 | 0.32× | **~18×** faster |
| 256 | 0.128 | 0.359 | 0.36× | **~26×** faster |

```bash
OMP_NUM_THREADS=2 .venv/bin/python benchmarks/bench_blocked_vec.py
```

### Honest claim

- **Win vs prior blocked/online CPU wall:** yes (~18–36× on these mid-T shapes).
- **Win vs eager on CPU:** **no** — still ~0.3–0.6× eager (eager’s single big GEMM +
  `tril` wins for mid T when `T×T` fits). Do **not** default `BDH_ATTN_IMPL=blocked`
  on CPU.
- Peak score elems still bound ≪ `T×T` for mid T (e.g. T=256 → bound 16320 vs 65536).
- No GPU on this box — do not claim GPU speedups from these CPU medians.

### Non-goals

- No PRs to `pathwaycom/*`
- No change to default `BDH_ATTN_IMPL=eager`
- No fake “beats eager” claims on CPU

## opt/attn-bwd-train — analytic StrictTrilAttnFn as first-class train path (2026-09-19)

**Branch:** `opt/attn-bwd-train` (private `katulevskiy/bdh-gpu-opt` only).
**Base tip:** `62c226b` (main after blocked-vec #38).

### Goal

Make `BDH_ATTN_AUTOGRAD=1` a clean, documented train path: cold (+ multi-token
with past) routes through `StrictTrilAttnFn` / `analytic_tril_attn_backward`,
CacheManager T=1 decode / `generate` stay intact, default remains OFF.

### What changed

| Path | Role |
|------|------|
| `kernels/attention_bwd.py` | `_forward_impl` supports `cuda` (+ `online`→blocked); docs |
| `kernels/attention_dispatch.py` | Document cold / multi-token AUTOGRAD vs T=1 decode |
| `bdh.py` `Attention.forward` | AUTOGRAD=1: cold + multi-token-with-past → `bdh_attn` → StrictTrilAttnFn; T=1 decode unchanged |
| `benchmarks/bench_attn_bwd.py` | Tiny-cfg train_step A/B: AUTOGRAD 0 vs 1 |
| `tests/test_attn_bwd.py` | Full-model grad parity @ dropout=0; pos0==0; generate+flag; cuda+AUTOGRAD |

### When to use the flag

```bash
# default — PyTorch autograd through eager GEMMs (unchanged)
unset BDH_ATTN_AUTOGRAD

# opt-in analytic train path (needed when BDH_ATTN_IMPL=blocked|triton|cuda
# forwards are not safely differentiable via autograd-through-eager)
export BDH_ATTN_AUTOGRAD=1
export BDH_ATTN_IMPL=blocked   # example
```

Use AUTOGRAD=1 when training with a non-eager attention forward. With
`IMPL=eager`, analytic bwd is still correct (parity @ dropout=0) but the
profile may still be dominated by the eager T×T forward + M recompute.

### Correctness (this box, CPU-only)

```text
.venv/bin/python -m pytest tests/test_attn_bwd.py -q
# 15 passed
#   analytic == eager autograd; StrictTrilAttnFn grads; gradcheck
#   full BDH grad parity AUTOGRAD 0 vs 1 @ dropout=0
#   pos0 == 0; generate works with flag; cuda+AUTOGRAD forwards

.venv/bin/python -m pytest tests/ -q
# (see commit / PR body for full count)
```

### Train-step microbench (CPU, Europe/Podgorica)

Tiny cfg `layers=2 d=64 nh=2 B=4 T=64 dropout=0`, `IMPL=eager`, AdamW:

```text
BDH_ATTN_AUTOGRAD=0  median ~7548 ms
BDH_ATTN_AUTOGRAD=1  median ~5533 ms  (~1.36× vs off on this run)
```

**Honest:** CPU-only; absolute ms are box-noise / load-sensitive. Do **not**
claim GPU speedups. With `IMPL=eager`, AUTOGRAD=1 still materializes T×T in
forward. GPU unmeasured — re-run `benchmarks/bench_attn_bwd.py` on A100/H100
(and try `IMPL=blocked|triton|cuda`) before claiming train wins.

### Non-goals

- Default stays AUTOGRAD **off**
- No softmax / scale / SDPA
- No PRs to `pathwaycom/*`
- No fused CUDA/Triton backward kernel (analytic PyTorch M recompute)
## opt/profile-v3 — re-profile after mlp-fuse #36 (2026-09-19)

**Branch:** `opt/profile-v3` (private `katulevskiy/bdh-gpu-opt` only).
**Base tip:** `b160469` (main after docs-matrix #34, gen-bench #35, mlp-fuse #36,
cuda-ref-v2 #37, blocked-vec #38, attn-bwd-train #39). Profile windows also captured at `f50837d`
(post-#36) and `c953c79` (post-#37); default **eager** path unchanged by #37/#38/#39 (autograd opt-in only).

### Method

```bash
.venv/bin/python benchmarks/profile_forward.py --mode all
# torch 2.14.0+cu130  cuda=False  device=cpu
# cfg: layers=4 d=128 nh=4 B=4 T=128; generate prompt=16 / new=32
```

Absolute ms are **profiler-inflated** (and noisy across short active windows).
Rank by **% self CPU** and call counts. Chrome traces under `benchmarks/traces/`
(gitignored). Compare to § opt/profile-v2 (tip `3d3ed2b`).

### New % breakdown (self CPU)

Representative midpoints across runs on this box (range when spread was large).
Default `BDH_ATTN_IMPL=eager` throughout:

| Mode | Top self-CPU ops | vs profile-v2 / post-#36 note |
|------|------------------|-------------------------------|
| Attention | `bmm` ~16–32%, `mul` ~23–31%, `copy_` ~14–27%, `sub`/`add` ~7–9%, `tril` ~1–9% | Still full T×T eager score×V. RoPE `mul`/`copy_` vs GEMM mix wobbles with short windows; no structural change since v2. |
| Forward | `bmm` ~15–28%, `copy_` ~18–25%, `mm` ~10–18%, `mul` ~11–17%, `clamp_min_`/`relu_` ~1–8%, LN ~1–11%, `tril` ~0.4–3% | Same shape as v2 (GEMM + copies). **`aten::contiguous` = 0** (mlp-fuse #36 / weight-layout view path). CPU wall ~noise; not a % reshuffle. |
| Generate | `bmm` ~14–44%, `copy_` ~6–11%, `mm` ~3–30%, LN ~2–12%, `einsum`/`mul`/`slice` smaller; Python `BDH.generate` often ~26% when attributed | **`aten::cat` = 0** still (cache-v2 #20). Decode GEMM + host `copy_` remain the tax; gen-bench #35 harness ready for GPU. |

### Confirmed landed (profile-visible / structural)

- **#36 mlp-fuse:** hot MLP path shows **no `aten::contiguous`**; `relu_` /
  `clamp_min_` present; decoder merge stays view-based. Not a top-line % win on CPU.
- **#38 blocked-vec:** vectorizes `blocked`/`online` tiles only — **default eager
  profile unchanged** (still T×T `bmm`+`tril`). CPU blocked still < eager.
- **#35 gen-bench / #34 docs-matrix / #37 cuda-ref-v2:** harness/docs/refs — no
  eager-path profile delta.
- **#20 cache-v2:** generate still **zero `aten::cat`**.
- **#21 fuse-scorev:** default still `eager` → profile still shows `bmm`+`tril`.

### Ranked follow-ups

Unchanged priority vs backlog: **P0** = GPU measure fused score×V; **P1** = GPU
compile train-step + generate/decode GEMM via `bench_generate.py` /
`bench_gpu_attn.py --mode decode`. Strike mlp-fuse / gen-bench / docs-matrix /
blocked-vec from “next” — already on main.

### Non-goals

- No PRs to `pathwaycom/*`
- No softmax / diagonal / SDPA
- No GPU speedup claims from these CPU % figures
- No defaulting `BDH_ATTN_IMPL=blocked` on CPU

## opt/blocked-autograd — blocked/online + tiled analytic bwd train (2026-09-19)

**Branch:** `opt/blocked-autograd` (private `katulevskiy/bdh-gpu-opt` only).
**Base tip:** `d3ff475` (main after profile-v3 #40).

### Goal

Train with `BDH_ATTN_IMPL=blocked|online` + `BDH_ATTN_AUTOGRAD=1` using fused /
vectorized blocked forward **and** an analytic backward that does **not** force
a full T×T score matrix (unlike #39 dense M-recompute).

### Math / design

`O = tril(Q@K.T, -1) @ V` is linear (no softmax). Gradients:

- `dS = tril(dO @ V.T, -1)` → `dQ = dS @ K`, `dK = dS.T @ Q` (no need for M)
- `dV = M.T @ dO` with `M = tril(Q@K.T, -1)` (recompute M tile-wise)

Saving full M from forward would erase the blocked peak-memory win. Tile-wise
recompute (same past / diagonal tiling as `blocked_tril_attn`) is exact for
this op — no FlashAttention-style softmax stats required. Dense
`analytic_tril_attn_backward` remains the eager+AUTOGRAD reference.

### What changed

| Path | Role |
|------|------|
| `kernels/attention_bwd.py` | `analytic_tril_attn_backward_blocked`; StrictTrilAttnFn uses tiled bwd for blocked/online/triton/cuda |
| `kernels/attention_dispatch.py` | `online` → `blocked` alias; docs for tiled AUTOGRAD path |
| `benchmarks/bench_attn_bwd.py` | A/B: AUTOGRAD 0/1 × IMPL eager|blocked|online |
| `tests/test_attn_bwd.py` | blocked≡dense analytic; blocked/online FN grads; full-model parity; online env |
| `kernels/README.md` | online alias + AUTOGRAD train snippet |

### When to use

```bash
# default — unchanged (eager, AUTOGRAD off)
unset BDH_ATTN_AUTOGRAD
unset BDH_ATTN_IMPL   # or =eager

# blocked/online train without full T×T in fwd or bwd
export BDH_ATTN_AUTOGRAD=1
export BDH_ATTN_IMPL=blocked   # or online
```

### Correctness (this box, CPU-only)

```text
.venv/bin/python -m pytest tests/test_attn_bwd.py -q
# 23 passed — tiled≡dense; blocked|online FN grads; full BDH parity @ dropout=0;
# pos0==0; online alias

.venv/bin/python -m pytest tests/test_attn_unify.py tests/test_fuse_scorev.py \
  tests/test_attention_mask.py -q
# 55 passed
```

### Train-step microbench (CPU, Europe/Podgorica)

Tiny cfg `layers=2 d=64 nh=2 B=4 T=64 dropout=0`, AdamW:

```text
AUTOGRAD=0 IMPL=eager    median ~7265 ms
AUTOGRAD=1 IMPL=eager    median ~4872 ms  (~1.49× vs off)
AUTOGRAD=1 IMPL=blocked  median ~3716 ms  (~1.95× vs off)
AUTOGRAD=1 IMPL=online   median ~3838 ms  (~1.89× vs off)
```

**Honest:** CPU-only; absolute ms are box-noise / load-sensitive. Do **not**
claim GPU speedups. Blocked+AUTOGRAD avoids full T×T in fwd **and** bwd (peak
score tiles). On this CPU box wall-time happened to beat eager+AUTOGRAD for
this tiny cfg — still typically expect blocked forward slower than eager for
larger mid-T microbenches (see opt/blocked-vec). GPU unmeasured.

### Non-goals

- Default stays AUTOGRAD **off**, IMPL **eager**
- No softmax / scale / SDPA
- No PRs to `pathwaycom/*`
- No fused CUDA/Triton backward kernel (tiled analytic is still PyTorch tiles)

## opt/prefetch-v2 — deepen BatchPrefetcher host overlap (2026-09-19)

**Branch:** `opt/prefetch-v2` (private `katulevskiy/bdh-gpu-opt` only).
**Base tip:** `d3ff475` (main, after profile-v3 #40). **Does not** touch `bdh.py`,
attention, or `tril(diagonal=-1)`.

### Goal

Reduce host stall between `train_step` calls by overlapping the **next**
`get_batch` host gather with the current step. Keep the existing
`.next()` API and 90/10 split / LM window semantics.

### What changed (`train.py`)

| Piece | Change |
|-------|--------|
| `BatchPrefetcher` | Daemon host thread + `queue.Queue(maxsize=1)` double-buffer (default). Producer refills while caller runs `train_step`. |
| `_gather_batch_host_numpy` | Producer gather via numpy indexing — avoids PyTorch OpenMP steal from the step. |
| CUDA path | Producer: gather + `pin_memory`. Caller: side-stream `non_blocking` H2D + wait-event. |
| Sync A/B | `BDH_PREFETCH_ASYNC=0` or `async_host=False` — one-slot preload on the caller thread. |
| Public `get_batch` | Unchanged torch vectorized path (seed-stable for tests). |

Hot loop unchanged::

    loss = train_step(model, opt, x, y)
    x, y = loader.next()  # wait for ready slot; producer already refilling

### Correctness

```text
.venv/bin/python -m pytest tests/test_dataloader.py -q
# 13 passed
# full: 288 passed, 9 skipped
```

LM windows still contiguous; `x[:,1:]==y[:,:-1]`. Async/sync both covered.

### Measured CPU (honest, this box, `cuda=False`, OMP≈2)

```text
OMP_NUM_THREADS=2 .venv/bin/python benchmarks/bench_prefetch.py
# host gather / get_batch median:           ~0.07 ms
# synthetic overlap (host sleep 2 ms + busy step 3 ms):
#   serial ~5.2 ms | async ~3.1 ms | **~1.7×**
# tiny train_step+prefetch loop:            sync/async often ~noise–modest
#   (load-sensitive on this shared CPU box)
```

**Claim:** measurable **overlap win** when host prep is non-trivial (synthetic
probe with sleep-modeled host delay). Real vectorized gather is already
~0.07 ms, so e2e train_step wins on this CPU box are **small / noisy** — do
not claim a large wall-clock train speedup here. GPU pin/H2D overlap still
unmeasured (no CUDA on this runner).

### Non-goals

- No PRs to `pathwaycom/*`
- No change to default `BDH_ATTN_IMPL` / attention math
- No fake GPU wins from CPU medians


## opt/gen-host — cut generate Python/host overhead (2026-09-19)

**Branch:** `opt/gen-host` (private `katulevskiy/bdh-gpu-opt` only).
**Base tip:** `bf93a11` (main after #42+#43 docs/prefetch).

### Goal

Profile showed `BDH.generate` ~26% self CPU after cats=0 / decode-copy. Cut
remaining **Python/host tax** (loop, sampling, repeated getattr/dispatch,
RoPE trig per decode step) without changing default attn math. Keep
`aten::cat=0` and `tril(diagonal=-1)`.

### What changed

1. **`Attention.ensure_rope_table(max_T)`** — one cos/sin table for
   `[0, max_seq)`; `rope_cos_sin` `narrow`s for prefill + decode (no per-step
   `arange`+trig). `generate` warms it once.
2. **Eager-import `bdh_rope_rotate` / `resolve_attn_impl`** — no lazy import
   inside `Attention.rope` every layer/step.
3. **Env resolve caches** in `attention_dispatch` / `rope_dispatch` (skip
   strip/lower/alias when env string unchanged). `online`→`blocked` alias kept.
4. **`generate` hoists `resolve_attn_impl()`** into `Attention._attn_impl_override`
   for the decode loop (cleared in `finally`); warms rope resolve once.
5. **`@torch.inference_mode()`** (was `@torch.no_grad()`); tighter sampling
   (skip temp scale when `temperature==1.0`; hoist softmax/multinomial;
   `masked_fill_` for top-k; write `out[:, pos]` without slice cat).
6. **`CacheManager.storage_matches_compute`** — hoisted dtype equality for the
   packed in-place RoPE path.

Preserved: default `BDH_ATTN_IMPL=eager`, `tril(-1)`, no softmax/scale,
`aten::cat=0`, tokens-match-eager / vs-baseline generate tests.

### Measured (this box, 2026-09-19 Europe/Podgorica, CPU-honest)

Isolated subprocess tip `8e2f5a0` vs this branch — cfg `layers=4 d=128 nh=4`,
`prompt=16 new=32`, warmup=3 iters=7, `temperature=1.0`, `cuda=False`:

```text
BEFORE  median=52.34 ms  ~611 tok/s  arange=33  environ.get=268  aten::cat=0
AFTER   median=39.84 ms  ~803 tok/s  arange=1   environ.get=142  aten::cat=0
DELTA   wall −12.5 ms (~1.31×)  arange 33→1  env.get 268→142  tokens match tip
```

Official harness:

```text
python benchmarks/bench_generate.py --warmup 2 --iters 5 --impls eager
# eager median≈40–50 ms  match_eager=yes  aten::cat=0  (CPU; not a GPU claim)
```

### Correctness

```text
.venv/bin/python -m pytest tests/test_gen_host.py tests/ -q
# test_gen_host: RoPE table slices, arange=1, cat=0, tokens vs baseline,
#   hoist clears override, environ.get reduced
# Full suite: 293 passed, 9 skipped (CUDA/native) on CPU-only box
```

### Honest limits

- CPU-only box — absolute ms are wall times, not GPU kernel wins.
- `environ.get` still hits on the rope path each layer (cached resolve); attn
  override removes the decode-loop attn gets.
- Default attn math unchanged; no claim of Triton/CUDA speedup.

### Non-goals

- No PRs to `pathwaycom/*`
- No default `BDH_ATTN_IMPL` change
- No re-introducing `aten::cat` in generate / CacheManager
- No softmax / diagonal inclusion / scale

## opt/compile-blocked — compile × blocked × autograd train matrix (2026-09-19)

**Branch:** `opt/compile-blocked` (private `katulevskiy/bdh-gpu-opt` only).
Prove or harden the **CPU-honest** train path for
`torch.compile` × `BDH_ATTN_IMPL=blocked` × optional `BDH_ATTN_AUTOGRAD=1`.
Extend the train-step harness with a full matrix; fix cheap Dynamo graph
breaks on AUTOGRAD self-attn; smoke-test compile+blocked @ dropout=0.
**Do not claim GPU.** Defaults stay `COMPILE=0` / `IMPL=eager` / AUTOGRAD off.

### Dynamo fix (cheap)

`bdh_attn(QR, QR, V)` passed the **same** tensor twice into
`StrictTrilAttnFn.apply`, which Dynamo rejects (`gb6297` duplicate tensor
input) → ~9 graph breaks per cold train forward under `AUTOGRAD=1`.

| Piece | Change |
|-------|--------|
| `kernels/attention_bwd.py` | `StrictTrilSelfAttnFn(Q, V, impl)` when `Q is K`; bwd returns `dQ+dK` |
| `strict_tril_attn` | Routes `Q is K` → SelfAttnFn; distinct Q/K keep three-arg Fn |
| `kernels/__init__.py` | Export `StrictTrilSelfAttnFn` |

After fix: `torch._dynamo.explain` cold train @ dropout=0 → **0 breaks** for
eager|blocked × AUTOGRAD 0|1.

### Harness

`benchmarks/bench_train_step.py` → `bench_compile_blocked_matrix`:

| Knob | Default | Meaning |
|------|---------|---------|
| `BDH_BENCH_COMPILE_BLOCKED` | `1` | Run COMPILE×IMPL×AUTOGRAD matrix (`0` = skip) |
| Cells | 8 | `COMPILE∈{0,1}` × `IMPL∈{eager,blocked}` × `AUTOGRAD∈{0,1}` |

Soft-skip per cell if inductor/probe falls back. Tiny cfg
`layers=2 d=64 nh=2 B=4 T=64 dropout=0`. Restores env after the matrix.

### Measured (this box, 2026-09-19 Europe/Podgorica, CPU-only)

`torch 2.14.0+cu130`, `cuda=False`, AdamW fused, sequential cells (same process):

```text
 COMPILE     IMPL AUTOGRAD    median_ms
       0    eager        0         7.76
       0    eager        1         9.50
       0  blocked        0         8.50
       0  blocked        1        10.19
       1    eager        0         5.25   (~1.48× vs eager/0)
       1    eager        1         6.42
       1  blocked        0       576.90   << slower — not a win
       1  blocked        1       870.08   << slower — not a win
```

**Honest read:** On this CPU box, `COMPILE=1` + default **eager** still shows
a small warm-ish win vs eager (same ballpark as `opt/compile-bench`).
`COMPILE=1` + **blocked** is **~70–100× slower** than eager compile — inductor
does not rescue the tiled Python block loop; keep blocked for peak-score
memory / parity, not default train. Absolute ms are load-sensitive (a later
warm re-run under multi-agent load inflated all compile cells into hundreds
of ms); use **ratios**, not absolute ms, and never as GPU claims.

### Correctness

```text
.venv/bin/python -m pytest tests/test_attn_bwd.py tests/test_compile.py -q
# SelfAttnFn grad parity; compile×blocked forward parity @ dropout=0;
# train_step smoke; 0 Dynamo breaks under AUTOGRAD=1
```

### Non-goals

- No default `BDH_COMPILE=1` / `BDH_ATTN_IMPL=blocked` / `BDH_ATTN_AUTOGRAD=1`
- No softmax / scale / SDPA; `tril(diagonal=-1)` preserved
- No PRs to `pathwaycom/*`
- No GPU / CUDA-graph claims from this CPU matrix

## opt/encoder-fuse — fewer encoder einsum temporaries (2026-09-19)

**Branch:** `opt/encoder-fuse` (private `katulevskiy/bdh-gpu-opt` only).
**Base tip:** `8d56385` (main after #46 compile-blocked).

### Goal

Cut encoder / encoder_v `einsum` + bias+ReLU temporaries on the hot forward
path. Try a fused linear-style path while staying bit-identical at `dropout=0`.
Hard constraints: `tril(diagonal=-1)` unchanged; Parameter shapes / `state_dict`
unchanged; defaults unchanged.

### Attempt

| Candidate | Result (CPU, `OMP=2`) |
|-----------|------------------------|
| Always-on `F.linear` for `_encoder_relu` via `(nh,D,N)→(nh*N,D)` | **Loses** — transpose+reshape copies weight each call; full forward ~9.6 ms vs einsum ~9.0 ms |
| `matmul` + `permute→contiguous` for `_encoder_v_relu` | **Loses** — extra activation contig copy (~0.32 vs ~0.21 ms) |
| Cached / pre-transposed `(nh*N,D)` Parameter | **Rejected** — would change `state_dict` shapes vs baseline |

### What landed (`bdh.py`)

1. **`_hdn_as_linear_weight`** — documented `(nh,D,N)→(nh*N,D)` helper for the
   optional bias path / benches.
2. **`_encoder_relu` hybrid**
   - **Default (`bias is None`)** — keep `einsum("btd,hdn->bthn")` + in-place
     ReLU (writes contiguous `(B,T,nh,N)` without a weight transpose-copy).
   - **Optional bias** — `F.linear` with bias fused in the GEMM epilogue, then
     `view(B,T,nh,N)` + in-place ReLU. Bit-identical to `F.relu(einsum+bias)`.
3. **`_encoder_v_relu`** — stays on einsum + `_bias_relu_` (matmul+contig lost).

Preserved: `tril(diagonal=-1)`, `CacheManager`, `BDH_ATTN_IMPL`, RoPE / dropout /
LN paths, baseline Parameter shapes / `state_dict` (biases remain `None`).

### Correctness (this box, CPU)

```text
.venv/bin/python -m pytest tests/test_encoder_fuse.py tests/test_mlp_fuse.py \
  tests/test_vs_baseline.py tests/test_correctness.py -q
# encoder-fuse: linear helper, default/bias paths, encoder_v, vs baseline,
#   state_dict shapes, attn pos0==0
```

No intentional numerical approximations at `dropout=0`.

### Honest CPU numbers (`OMP_NUM_THREADS=2`, Europe/Podgorica)

```text
OMP_NUM_THREADS=2 .venv/bin/python benchmarks/bench_encoder_fuse.py
# device=cpu torch=2.14.0+cu130 cuda=False OMP=2
# micro B=4 T=64 D=128 nh=4 N=256
#   encoder einsum (default)             0.266 ms
#   encoder F.linear + weight copy       0.328 ms  (always-on loses)
#   encoder einsum + add_ bias           0.291 ms
#   encoder F.linear fused bias          0.387 ms  (epilogue cleaner; wall no win)
#   encoder_v einsum                     0.270 ms
#   encoder_v matmul+contig              0.337 ms  (loses)
#   full forward (landed / hybrid)       9.076 ms
# always-on F.linear full forward       ~9.6 ms  (earlier A/B; regression — not landed)
# pytest tests/ — 313 passed, 9 skipped
```

**Verdict:** no reliable default-path CPU wall win (einsum already best for
`bias=None`). Land hybrid for optional fused-bias epilogue + docs of the
always-on linear attempt. Expect GPU `F.linear` epilogue to matter more when
biases are registered; unmeasured here.

### Non-goals

- No PRs to `pathwaycom/*`
- No Parameter shape / checkpoint migration
- No default `BDH_ATTN_IMPL` change
- No softmax / diagonal inclusion / scale
- No fake GPU speedups from CPU medians

## opt/gen-sample — fuse lm_head+sample for T=1 decode (2026-09-19)

**Branch:** `opt/gen-sample` (private `katulevskiy/bdh-gpu-opt` only).
**Base tip:** `6c36760` (main after #44 gen-host + #46 compile-blocked + #47 encoder-fuse).

### Overlap with #44 `opt/gen-host`

`opt/gen-host` already landed the Python/host tax cuts (RoPE table,
`resolve_*` hoist, `inference_mode`, sampling constant hoist). This branch
**does not** redo that work. Pivot: **fuse vocab proj + sample** on the T=1
decode step only.

### What changed

1. **`BDH._lm_head_last_into(x, out)`** — last-token `mm`/`addmm` into a
   preallocated fp32 `(B, V)` buffer (bit-identical to
   `_vocab_logits(x)[:, -1, :].float()`).
2. **`forward(..., logits_out=)`** — when `T==1`, writes via
   `_lm_head_last_into` and returns the owned `(B, V)` buffer (no per-step
   `(B, 1, V)` `F.linear` alloc; no unused `unsqueeze`).
3. **`BDH._sample_from_logits`** — reused `probs_buf` (`softmax(..., out=)`);
   optional **fused top-k**: `topk` → softmax/multinomial over **k** →
   `gather` (distribution-equivalent; RNG differs vs mask/-inf when
   `top_k` set). Default `top_k=None` keeps full-vocab multinomial →
   **tokens-match** tip / baseline.
4. **`generate`** — preallocates `logits_buf`/`probs_buf`; first sample may
   use a view when fp32 + no scale/top-k (no extra `copy_`); decode steps
   pass `logits_out=logits_buf`. Preserves `aten::cat=0`,
   `tril(diagonal=-1)`, defaults unchanged.

### Measured (this box, Europe/Podgorica, CPU-honest, `cuda=False`)

Isolated subprocess tip `origin/main` `bdh.py` vs this branch — same weights,
tokens match on default path:

```text
cfg layers=4 d=128 nh=4 prompt=16 new=32 temp=1.0 top_k=None  (default V=256)
  TIP_MAIN median≈57.8 ms
  FUSED    median≈59.9 ms
  DELTA    ~noise / slight CPU regression on tiny V (allocator reuse)

cfg layers=2 d=64 V=8192 prompt=16 new=24
  top_k=None: TIP≈13.8 ms  FUSED≈14.2 ms  (~noise)
  top_k=64:   TIP≈16.9 ms  FUSED≈11.0 ms  (~1.53×)  ← fused top-k win

Harness: benchmarks/bench_generate.py --warmup 2 --iters 5 --impls eager
  eager median≈60 ms  match_eager=yes  aten::cat=0
```

Profiler (default V=256, +32 decode steps): `aten::linear` **165→133** (−32
decode vocab linears); `aten::cat=0` unchanged.

**Claim carefully:** structural fuse is real (no per-step vocab `linear`
alloc on decode; fused top-k helps large-V + `top_k`). Default small-V wall
clock on this CPU box is **noise** — do not claim a big e2e generate speedup
vs #44 without GPU / larger vocab.

### Correctness

```text
.venv/bin/python -m pytest tests/ -q
# 311 passed, 9 skipped
# tests/test_gen_sample.py — lm_head parity, logits_out, tokens vs tip-style,
#   twin baseline, cat=0, top_k smoke
```

### Non-goals

- No PRs to `pathwaycom/*`
- No redoing gen-host hoist / RoPE table
- No default `BDH_ATTN_IMPL` change / no tril math change
- No re-introducing `aten::cat`

## opt/compile-guidance — warn COMPILE+blocked; document eager path (2026-09-19)

**Branch:** `opt/compile-guidance` (private `katulevskiy/bdh-gpu-opt` only).
**Base tip:** `c7a7471` (main after #48 gen-sample; post #47 encoder-fuse / #46 compile-blocked).

### Goal

Harden the **recommended CPU train path** after #46 showed
`COMPILE=1` + **eager** is the only compile win on this box (~5.25 ms vs
~7.76 eager baseline; `COMPILE=1`+blocked ~70–100× slower). Document
operator guidance; optionally warn in `maybe_compile` when `COMPILE=1`
with tiled backends. **Do not change defaults.** GPU compile still open.

### Code

| Piece | Change |
|-------|--------|
| `train.maybe_compile` | If `BDH_ATTN_IMPL∈{blocked,online,triton}` and `BDH_COMPILE=1`, log a clear CPU-regression warning (advisory; still compiles) |
| `tests/test_compile.py` | Warning coverage; COMPILE+eager+AUTOGRAD smoke + **0** Dynamo graph breaks |

### Operator recommendation (CPU)

```bash
# Recommended compile train path on CPU:
BDH_COMPILE=1 BDH_ATTN_IMPL=eager python train.py
# Optional analytic bwd (still 0 graph breaks after SelfAttnFn #46):
BDH_COMPILE=1 BDH_ATTN_IMPL=eager BDH_ATTN_AUTOGRAD=1 python train.py
```

Do **not** pair `BDH_COMPILE=1` with `blocked` / `online` / `triton` on CPU
for speed — inductor does not rescue the tiled Python loop. Keep those IMPLs
for peak-score memory / parity experiments without compile, or for future
**GPU** measurement.

### Non-goals

- No default flip of `BDH_COMPILE` / `BDH_ATTN_IMPL` / `BDH_ATTN_AUTOGRAD`
- No softmax / scale / SDPA; `tril(diagonal=-1)` preserved
- No PRs to `pathwaycom/*`
- No GPU / CUDA-graph claims

## opt/profile-v4 — re-profile tip after gen-sample #48 (2026-09-19)

**Branch:** `opt/profile-v4` (private `katulevskiy/bdh-gpu-opt` only).
**Base tip:** `f16111b` (main after compile-guidance #49; post gen-sample #48,
encoder-fuse #47, compile-blocked #46, gen-host #44). Profile windows captured at
`c7a7471` (post-#48); default **eager** path unchanged by #49 (guidance/warn only).

### Method

```bash
.venv/bin/python benchmarks/profile_forward.py --mode all
# torch 2.14.0+cu130  cuda=False  device=cpu
# cfg: layers=4 d=128 nh=4 B=4 T=128; generate prompt=16 / new=32
```

Absolute ms are **profiler-inflated** (and noisy across short active windows).
Rank by **% self CPU** and call counts. Chrome traces under `benchmarks/traces/`
(gitignored). Compare to § opt/profile-v3 (tip `b160469` / docs `d3ff475`).

### New % breakdown (self CPU)

Representative midpoints on this box. Default `BDH_ATTN_IMPL=eager` throughout:

| Mode | Top self-CPU ops | vs profile-v3 / post-#44–#48 note |
|------|------------------|-----------------------------------|
| Attention | `mul` ~31%, `copy_` ~26%, `bmm` ~16%, `sub`/`add` ~8–10%, `tril` ~7% | Same eager T×T shape as v3 (RoPE `mul`/`copy_` vs GEMM mix wobbles). No structural change from #44–#49. |
| Forward | `copy_` ~24%, `mul` ~18%, `bmm` ~14%, LN ~11%, `mm` ~10%, `clamp_min_` ~7%, `tril` ~3% | Same GEMM+copies shape as v3. **`aten::contiguous` = 0** still (mlp-fuse #36). #47 encoder-fuse keeps default einsum — no % reshuffle. |
| Generate | `bmm` ~43%, `mm` ~30%, LN ~12%, `copy_` ~11%; `BDH.generate` self ~1.4% | **`aten::cat` = 0** still (#20). **gen-host #44 visible:** Python `BDH.generate` self dropped vs v3’s often-~26% attributed host tax. **gen-sample #48:** structural T=1 lm_head+sample fuse; default-V wall % still decode GEMM-dominated (tiny V; no big % reshuffle). |

### Confirmed landed (profile-visible / structural)

- **#44 gen-host:** generate host self-CPU attribution much lower (~1.4% vs ~26% in v3 windows); remaining tax is decode `bmm`/`mm`/`copy_`.
- **#48 gen-sample:** fused T=1 vocab+sample path on tip; profiler % on default V=256 still GEMM-led (honest: wall ~noise on tiny V per § opt/gen-sample).
- **#20 cache-v2:** generate still **zero `aten::cat`** (0 calls in traces).
- **#36 mlp-fuse:** forward still **zero `aten::contiguous`**.
- **#49 compile-guidance / #46 / #47:** no default-eager profile delta.

### Ranked follow-ups

Unchanged honesty vs backlog: **P0** = GPU measure fused score×V (CPU % above
are not GPU wins). **P1** = GPU compile train-step + generate/decode GEMM
(`bench_generate.py` / `bench_gpu_attn.py --mode decode`). gen-host / gen-sample /
compile-guidance already on main — strike from “next.”
### Non-goals

- No PRs to `pathwaycom/*`
- No softmax / diagonal / SDPA
- No GPU speedup claims from these CPU % figures
- No defaulting `BDH_ATTN_IMPL=blocked` on CPU

## opt/ln-deepen — cut residual LN temporaries (2026-09-19)

**Branch:** `opt/ln-deepen` (private `katulevskiy/bdh-gpu-opt` only).
**Base tip:** `527ead2` (main after #50 profile-v4; started at `c7a7471`, rebased through #49/#50).

### Goal

Further cut **LayerNorm + residual** copy/temp tax on forward (profile still
shows LN + `copy_` significant). Keep the #30 compile-friendly **`F.layer_norm`**
path (no `self.ln` module calls, no `is_grad_enabled` product branch). Preserve
`tril(-1)`, defaults, bit-identical @ dropout=0.

### What we tried

| Approach | Dynamo 0 breaks? | Bit-identical? | Result |
|----------|------------------|----------------|--------|
| Eager fused add+LN (aten / F) | — | — | **No API** for affine-free fused add+`layer_norm` on this torch |
| Nested `F.layer_norm(x + F.layer_norm(...))` | yes | yes | Same temps as #30; inductor may fuse on GPU compile |
| **`y = LN(y_mlp); y.add_(x); LN(y)`** (ln-fuse reuse) | **yes** | **yes** (fwd+grad) | **Landed** — drops out-of-place `x+y` temp; #30 `F.layer_norm` kept |

Re-measure vs #30's caution: in-place into the **inner LN output** (not into `x`
or `y_mlp`) stays AOTAutograd/Dynamo-clean (`torch._dynamo.explain` → **0**
graph breaks on cold train @ dropout=0) and **grad bit-exact** vs out-of-place add.

### Code

`_residual_ln`: `F.layer_norm` twice + `y.add_(x)` buffer reuse (same pattern as
#12 ln-fuse, but still never call `self.ln` on the hot path — #30 preserved).

### Honest CPU numbers (`torch 2.14.0+cu130`, this box, `cuda=False`)

```text
.venv/bin/python benchmarks/bench_residual_ln.py
# residual-only (B=4 T=64 D=128): oop 15.95 ms / inplace 16.00 ms  (ratio ~1.00× — parity)
# op mix 40×: oop {native_layer_norm:80, add:40}; inplace {native_layer_norm:80, add_:40}
# forward tip-style (layers=4 d=128 nh=4 B=4 T=64): e2e wall **~noise** (do not claim ratio)
```

**Claim carefully:** structural temp cut is real (`aten::add` → `aten::add_` into
LN out — no separate residual-sum tensor). Residual-only wall is **parity** on
this CPU; e2e forward still GEMM/`copy_`-dominated (profile-v4: LN ~11%,
`copy_` ~24%). No GPU fused-kernel claim.

### Correctness

```text
.venv/bin/python -m pytest tests/test_compile.py tests/test_vs_baseline.py -q
# residual bitexact vs module LN; grad parity vs oop add; compile path; 0 graph breaks
```

### Non-goals

- No PRs to `pathwaycom/*`
- No RMSNorm / affine LN / softmax / SDPA
- No default `BDH_COMPILE` / attn impl change
- No custom CUDA/Triton LN kernel on this branch

## opt/decode-mm — cut generate decode GEMM tax (2026-09-19)

**Branch:** `opt/decode-mm` (private `katulevskiy/bdh-gpu-opt` only).
**Base tip:** `d25ec33` (main after docs #52 / ln-deepen #51; profile-v4 generate still `bmm`~43% + `mm`~30%).

### Goal

Attack generate decode **bmm** + **mm** tax from profile-v4 by deepening existing
blocked / Triton / CUDA T=1 decode vs packed KR/V, plus a B=1 lm_head `mv` path.
Keep **cat-free** CacheManager, `tril(diagonal=-1)`, and **default eager**.

### What changed

| Piece | Change |
|-------|--------|
| `kernels/attention.py` | `_two_gemm_decode` (Tq=1 BH-`bmm` when V per-head; else 4D @ broadcast-V); tiled decode uses `out.add_`; Triton decode tiles up to 256 |
| `bdh.py` | T=1 Attention always → `bdh_attn_decode`; `_lm_head_last_into` uses `mv`/`addmv` when `B==1` |
| `kernels/cuda_attn.py` | `CUDA_DECODE_TILE_N=32`; tiled ref via `_two_gemm_decode` + `add_` |
| `csrc/tril_attn_cuda.cu` | `DECODE_TILE_N=32`; dedicated **`tril_decode_tq1_kernel`** (thin grid) |
| `csrc/tril_attn_cpu.cpp` | Decode tiled path uses `DECODE_TILE_N=32` |
| `tests/test_inc_decode.py` | BH-bmm parity, broadcast-V, tile `add_`, cuda tiled ref, unified T=1 dispatch |
| `tests/test_gen_sample.py` | B=1 lm_head mv ≡ vocab logits (untied + tied) |

```bash
export BDH_ATTN_IMPL=eager     # default — _two_gemm_decode
export BDH_ATTN_IMPL=blocked   # tiled decode, out.add_
export BDH_ATTN_IMPL=triton    # CUDA fused; CPU → blocked
export BDH_ATTN_IMPL=cuda      # Tq=1 CUDA kernel / DECODE_TILE_N ref
```

### Semantics (unchanged)

```text
# decode at absolute index S (past length S):
out = (Q @ K_past.mT) @ V_past     # all keys j < S; no self
# ≡ last row of tril(Q_all @ K_all.T, diagonal=-1) @ V_all
# CacheManager: aten::cat = 0; pos0 / S=0 → zeros
```

### Correctness (this box, CPU)

```text
.venv/bin/python -m pytest tests/test_inc_decode.py tests/test_gen_sample.py \
  tests/test_attention_mask.py tests/test_cuda_decode.py tests/test_cache_pack.py -q
# blocked/triton/cuda decode ≡ eager last row; B=1 lm_head mv ≡ mm; pos0==0; cats=0
```

### Honest CPU microbench (no GPU wins claimed)

```text
# tip d25ec33 vs opt/decode-mm  (OMP_NUM_THREADS=2, Europe/Podgorica)
# bench_generate.py --warmup 2 --iters 5  layers=4 d=128 prompt=16/+32
tip    eager 56.84 ms | blocked 59.29 ms | match_eager=yes | aten::cat=0
branch eager 59.22 ms | blocked 61.81 ms | match_eager=yes | aten::cat=0
# delta ~noise / slight regress (~4%) — NOT a wall win

# decode score×V micro (B=4 H=4 S=128 N=64 D=128):
correctness max|blocked-eager|=0
eager_decode   median: ~0.043 ms
blocked_decode median: ~0.037 ms
triton_decode  median: ~0.034 ms  (CPU → blocked)
cuda_ref       median: ~0.035 ms
two_gemm BH-V  median: ~0.023 ms  (per-head V bmm path)

# long S=2048 (B=1 H=2 N=32 D=64, block_size=64): maxdiff=0; ~0.03 ms both
```

**Verdict:** no reliable default-path CPU wall win vs tip. Land **structural**
deepen (shared `_two_gemm_decode`, CUDA Tq=1 + `DECODE_TILE_N`, B=1 lm_head `mv`)
+ honest docs. **No GPU on this box** — `tril_decode_tq1_kernel` unexecuted;
`bench_gpu_attn.py --mode decode` skips cleanly. Expect GPU wins from thin Tq=1
grid + larger past tiles + fused score×V. Default remains eager.

### Non-goals

- No PRs to `pathwaycom/*`
- No change to default `BDH_ATTN_IMPL=eager`
- No re-introducing `aten::cat` in generate / CacheManager
- No softmax / scale / SDPA
- No fake GPU speedups from CPU medians

## opt/amp-deepen — harden opt-in AMP train path (2026-09-19)

**Branch:** `opt/amp-deepen` (private `katulevskiy/bdh-gpu-opt` only).
**Base tip:** `6e23cfe` (decode-mm #53 on main).

### Goal

Deepen the opt-in train AMP path from #24 for **correctness + CPU honesty**:
extend CPU bf16/fp16 smoke, optional hot-forward-only autocast, honest tiny
`train_step` AMP vs fp32 bench, and document that **AMP throughput wins are
GPU-only**. Defaults stay **fp32 / COMPILE=0 / eager**; GradScaler **only**
`float16+CUDA`; attention still `tril(diagonal=-1)`.

### Knobs (`train.py`)

| Env / API | Default | Meaning |
|-----------|---------|---------|
| `BDH_AMP_DTYPE` | unset → `float32` | `float32`/`fp32`/`off`, `bfloat16`/`bf16`, `float16`/`fp16`/`half` |
| `BDH_AMP_FORWARD_ONLY` | `0` | `1` → autocast logits only; CE in fp32 outside |
| `configure_amp(name, forward_only=…)` | from env | Reconfigure `ctx` / `scaler` / `_amp_forward_only` |
| `cpu_bf16_available()` / `cpu_fp16_available()` | — | CPU autocast gates |
| `amp_throughput_claim_device()` | — | `"cuda"` or `"none"` (honest) |
| GradScaler | **off** unless `float16` **and** CUDA | bf16 never loss-scales; CPU never scales |

### Semantics (unchanged)

```text
out = tril(Q @ K.T, diagonal=-1) @ V   # no softmax, no 1/sqrt(d)
```

Default `train_step` still wraps `model(x, y)` in `ctx`. With
`BDH_AMP_FORWARD_ONLY=1`, only `model(x)` logits are under autocast; CE uses
`logits.float()` outside.

### AMP correctness (CPU — documented atol; #54 headroom)

| path | dtype | atol | rtol |
|------|-------|------|------|
| train forward logits + loss | float16 / bfloat16 | 2e-1 | 2e-1 |
| train grads (1× fwd+bwd) | float16 / bfloat16 | 2e-1 | 2e-1 |
| forward_only vs full autocast loss | bfloat16 | 2e-1 | 2e-1 |

Typical bf16 logits max ~5e-3; rare seeds ~0.11–0.13 → table uses headroom.

```text
.venv/bin/python -m pytest tests/test_bf16_train.py -v
# 16 tests: aliases, GradScaler gate, tril(-1), parity, bf16+fp16 smoke,
# forward_only, defaults fp32/COMPILE=0/eager, amp_claim honesty
```

### Honest CPU train_step bench (this box, tiny cfg)

```text
BDH_BENCH_COMPILE=0 BDH_BENCH_COMPILE_BLOCKED=0 BDH_BENCH_AMP=1 \
  python benchmarks/bench_train_step.py

device=cpu cuda=False amp_claim=none  layers=2 d=64 B=4 T=64
float32:            median  8.25 ms   1.00x
bfloat16:           median  7.12 ms   0.86x   # noise / cast mix; not a win claim
float16:            median  8.68 ms   1.05x   # slightly slower
bfloat16+fwd_only:  median 11.49 ms   1.39x   # CE outside often slower on CPU
```

**Verdict:** CPU AMP is for **correctness smoke**. Medians wander around fp32
(sometimes a hair faster, often slower — especially forward_only). **No
Tensor Core / GPU throughput claim** from these numbers. Use AMP on CUDA train
boxes (`amp_claim=cuda`); GradScaler only for `float16`.

### Non-goals

- No default AMP on (stays float32 until env set)
- No COMPILE / attn-impl default change
- No attention math changes / softmax / SDPA
- No PRs to `pathwaycom/*`
- No fake GPU speedups from CPU medians

## opt/decode-online-v2 — deepen blocked T=1 decode (2026-09-19)

**Branch:** `opt/decode-online-v2` (private `katulevskiy/bdh-gpu-opt` only).
**Base tip:** `dbf2c21` (main after `#54` amp-deepen / `#53` decode-mm).

### Goal

After decode-mm, deepen **blocked/online** T=1 decode vs packed KR/V for
**lower peak score mem** and **better CPU wall vs eager when S is long**
(CacheManager broadcast `V=(B,1,S,D)`). Keep `tril(diagonal=-1)`, **cat=0**,
**default eager** unchanged.

### What changed

| Piece | Change |
|-------|--------|
| `kernels/attention.py` | `_DECODE_ONESHOT_ELEMS=2048`; `_tiled_score_v(..., oneshot_elems=)`; broadcast-V decode uses tight oneshot (tiles long S); per-head V keeps large budget (BH-bmm); Tq=1 per-head tile loop reuses flattened Q; `max_decode_score_elems`; `online_decode_attn` alias |
| `kernels/attention_dispatch.py` | Docs: blocked/online decode peak ~Tq×tile on long S |
| `kernels/README.md` | Decode-online-v2 oneshot / peak note |
| `tests/test_inc_decode.py` | Long-S broadcast parity S∈{64,256,1024,4096}; peak bound; tile-when-over-budget; online alias |

```bash
export BDH_ATTN_IMPL=eager     # default — unchanged
export BDH_ATTN_IMPL=blocked   # online tiled decode (tight oneshot)
export BDH_ATTN_IMPL=online    # alias of blocked
```

### Semantics (unchanged)

```text
# decode at absolute index S (past length S):
out = (Q @ K_past.mT) @ V_past     # all keys j < S; no self
# ≡ last row of tril(Q_all @ K_all.T, diagonal=-1) @ V_all
# CacheManager: aten::cat = 0; pos0 / S=0 → zeros
```

### Correctness (this box, CPU)

```text
.venv/bin/python -m pytest tests/test_inc_decode.py tests/test_cuda_decode.py \
  tests/test_attention_mask.py tests/test_cache_pack.py tests/test_gen_sample.py -q
# 88 passed, 3 skipped — blocked/online/triton/cuda decode ≡ eager last row
# peak_decode(S>2048) < S; cats=0; default BDH_ATTN_IMPL=eager
```

### Honest CPU microbench (no GPU wins claimed)

`OMP_NUM_THREADS=2`, Europe/Podgorica, `torch 2.14.0+cu130`, `cuda=False`.

**Decode score×V** (B=4 H=4 N=64 D=128, broadcast V):

| S | eager ms | blocked ms | spd | peak elems | eager peak |
|---|----------|------------|-----|------------|------------|
| 64 | ~0.025 | ~0.026 | ~0.96× | 64 | 64 |
| 256 | ~0.082 | ~0.076 | ~1.07× | 256 | 256 |
| 1024 | ~0.44 | ~0.45 | ~0.98× | 1024 | 1024 |
| 4096 | ~9.0 | ~2.1 | **~4.4×** | **1024** | 4096 |

Long-S cliff: eager broadcast-V oneshot blows up; blocked tiles under
`_DECODE_ONESHOT_ELEMS` → lower peak **and** better wall.

**Generate** (CacheManager, layers=4 d=128, tokens match eager, `aten::cat=0`):

| prompt | new | eager ms | blocked ms | blk/eager | match | cats |
|--------|-----|----------|------------|-----------|-------|------|
| 64 | 16 | ~15.4 | ~17.0 | ~0.91× | yes | 0 |
| 256 | 16 | ~25.6 | ~30.7 | ~0.83× | yes | 0 |
| 1024 | 8 | ~96.6 | ~62.0 | **~1.56×** | yes | 0 |

Note: generate wall at prompt=1024 includes **cold** blocked prefill (no full
T×T) as well as decode tiles — still a fair IMPL=blocked vs eager A/B; decode
micro above isolates score×V.

**Verdict:** structural deepen lands (tight broadcast oneshot, peak helper,
online alias). Mid-S decode ~parity; **long S** blocked/online wins wall + peak
on CPU for broadcast-V. Default remains eager. **No GPU** — measure with
`bench_gpu_attn.py --mode decode` / `bench_generate.py` on A100/H100.

### Non-goals

- No PRs to `pathwaycom/*`
- No change to default `BDH_ATTN_IMPL=eager`
- No re-introducing `aten::cat` in generate / CacheManager
- No softmax / scale / SDPA
- No fake GPU speedups from CPU medians

## opt/attn-auto — opt-in long-S decode → blocked (2026-09-19)

**Branch:** `opt/attn-auto` (private `katulevskiy/bdh-gpu-opt` only).
**Base tip:** `33eb300` (main after `#55` decode-online-v2).

### Goal

After `#55` showed **blocked** T=1 decode wins at long `S` (CPU-honest), add an
**opt-in** auto backend: keep **default eager** everywhere, but when
`BDH_ATTN_AUTO=1`, switch **decode only** to blocked once `past_len` exceeds a
threshold. Cold / prefill stays on `BDH_ATTN_IMPL`. Hard constraints unchanged:
`tril(diagonal=-1)`, `aten::cat=0`, default behavior unchanged unless the new
env is set.

### What changed

| Piece | Change |
|-------|--------|
| `kernels/attention_dispatch.py` | `BDH_ATTN_AUTO` / `BDH_ATTN_AUTO_THRESHOLD`; `resolve_decode_impl(past_len)`; `bdh_attn_decode` applies AUTO; cold `bdh_attn` unchanged |
| `kernels/__init__.py` / `kernels/README.md` | Export + document AUTO knobs |
| `tests/test_attn_auto.py` | Default-off, threshold switch, non-eager override, parity, Attention module |

```bash
# default — unchanged (eager cold + eager decode)
unset BDH_ATTN_AUTO

# opt-in long-S decode → blocked; cold/prefill still BDH_ATTN_IMPL (eager)
export BDH_ATTN_AUTO=1
export BDH_ATTN_AUTO_THRESHOLD=512   # optional; default 512; switch when past_len > thr
```

### Threshold rationale (#55 CPU benches)

`OMP_NUM_THREADS=2`, Europe/Podgorica, `torch 2.14.0+cu130`, `cuda=False`
(from `#55` / `opt/decode-online-v2`):

**Decode score×V** (B=4 H=4 N=64 D=128, broadcast V):

| S | eager ms | blocked ms | spd | note |
|---|----------|------------|-----|------|
| 64 | ~0.025 | ~0.026 | ~0.96× | stay eager |
| 256 | ~0.082 | ~0.076 | ~1.07× | ~parity |
| 1024 | ~0.44 | ~0.45 | ~0.98× | ~parity |
| 4096 | ~9.0 | ~2.1 | **~4.4×** | prefer blocked |

**Generate** (CacheManager; tokens match; `aten::cat=0`):

| prompt | blocked/eager | note |
|--------|---------------|------|
| 64 | ~0.91× | stay eager |
| 256 | ~0.83× | stay eager |
| 1024 | **~1.56×** | prefer blocked (cold+decode) |

**Default threshold = 512** sits between the mid-S “stay eager” regime and the
long-S wins (`S≥1024` generate / `S=4096` decode). Switch is strict
`past_len > threshold`. Re-tune via `BDH_ATTN_AUTO_THRESHOLD` after GPU benches.

### Semantics

```text
cold / prefill:  always BDH_ATTN_IMPL   (AUTO ignored)
T=1 decode:
  if AUTO off or IMPL != eager:  IMPL
  elif past_len > THRESHOLD:     blocked
  else:                          eager
# tril(diagonal=-1); past-only; cat=0; default AUTO off
```

### Correctness (this box, CPU)

```text
.venv/bin/python -m pytest tests/test_attn_auto.py tests/test_inc_decode.py \
  tests/test_attn_unify.py tests/test_gen_sample.py -q
# AUTO off → eager at any S; AUTO on → blocked when S>512; parity vs eager;
# explicit IMPL=blocked|triton|cuda not overridden; cold path ignores AUTO
```

### Non-goals

- No change to default `BDH_ATTN_IMPL=eager` or decode path when AUTO unset
- No AUTO on cold / prefill / train
- No PRs to `pathwaycom/*`
- No softmax / scale / SDPA
- No fake GPU speedups from CPU medians

## opt/cache-page — deepen CacheManager paging / growth (2026-09-19)

**Branch:** `opt/cache-page` (private `katulevskiy/bdh-gpu-opt` only).
**Base tip:** `a96acaf` (main after attn-auto #56).

### Goal

Deepen `CacheManager` page growth for **long generate**: fewer realloc
copies on long S, keep **`aten::cat=0`** and **`tril(diagonal=-1)`**.
Defaults unchanged (`page_size=None` → full prealloc; `generate` default
unchanged).

### What landed

| Piece | Detail |
|-------|--------|
| Geometric growth | `_next_capacity`: `max(need, 2×capacity)`, page-aligned, ≤`max_seq` |
| Cheap realloc | `_grow` uses `torch.empty` + live-prefix `copy_` (no zero-fill free slots) |
| Stats | `n_grows`, `bytes_copied_on_grow` |
| `ensure_capacity(need)` | Public one-shot grow hint (optional; not wired into default generate) |
| `reserve` / `stage` | Grow-before-write; docs note prior views invalidate on grow |

Linear `+page_size` recopied the prefix every page (≈ O(S²) bytes). Doubling
keeps total grow-copy bytes ≈ O(S). Example: `page=16`, `max_seq=512` → **5**
grows (16→32→…→512) vs **31** linear steps.

### Correctness (this box, CPU)

```text
.venv/bin/python -m pytest tests/test_cache_pack.py tests/test_correctness.py \
  tests/test_attention_mask.py tests/test_vs_baseline.py \
  tests/test_decode_amp.py tests/test_inc_decode.py -q
# 99 passed (cache_pack 21 incl. geometric / ensure_capacity /
#   reserve+stage across grow / long paged generate cat=0 + match)
```

### Honest CPU notes (this box)

```text
page=16 → max_seq=512 tokenwise fill (2 layers, d=64):
  geometric n_grows=5   (16→32→64→128→256→512)
  linear would be 31    (+16 each step)
  bytes_copied_on_grow ≈ 2.3 MB vs ~36.6 MB linear estimate (~16× less)
generate(prompt=8, +64, cache_page_size=8): aten::cat = 0
```

- **CPU-only** box. Geometric paging cuts **realloc copy bytes / grow count**,
  not attention GEMM wall time. Do **not** claim GPU wins from these numbers.
- Default `generate` (`cache_page_size=None`) still preallocates
  `prompt+max_new_tokens` — **zero grows**, same as tip. Opt-in
  `cache_page_size=` trades peak RSS for O(log S) grows.
- `aten::cat` in `generate` remains **0** (paged and fixed).
- `tril(diagonal=-1)` incremental decode unchanged.

### Non-goals

- No change to default `page_size` / `cache_page_size` (still `None`)
- No softmax / scale / SDPA; no pathwaycom PRs
- No fake GPU speedups from CPU grow-count deltas

## opt/layout-v2 — deepen weight/activation layout (2026-09-19)

**Branch:** `opt/layout-v2` (private `katulevskiy/bdh-gpu-opt` only).
**Base tip:** `0b80d0b` (#57 cache-page on main; after #56 attn-auto).

### Goal

Further cut forward `copy_` / `mm` tax from layout after #13/#16/#36/#47.
Preserve Parameter shapes / `state_dict`; `tril(-1)`; defaults.

### Attempt vs land

| Candidate | CPU (`OMP=2`) |
|-----------|---------------|
| channels_last on `(B,nh,T,N)` / activations | **Loses** — extra contig copy; not bit-free |
| Always-on uncached `F.linear` for encoder (`(nh,D,N)→(nh*N,D)` each call) | **Loses** — same as #47 |
| matmul + permute→contiguous for `encoder_v` | **Loses** — extra activation copy |
| **Eval-only versioned contiguous `(nh*N,D)` cache + `F.linear`** | **Wins** on T≤32 / generate; T=128 ~noise |

### What landed (`bdh.py`)

1. **`_encoder_w_lin` cache** — contiguous `(nh*N, D)` for `self.encoder`; not a Parameter / not in `state_dict`.
2. **Warm in `train(False)` / `eval()`**; clear in `train(True)`; **`_load_from_state_dict` force-refresh** so `load_state_dict` cannot leave a stale buffer.
3. **`_encoder_relu_fwd`** — eager eval uses cached `F.linear` when cache coherent; **train keeps einsum**; under **`torch.compiler.is_compiling()`** keep einsum so Dynamo does not bake the non-Parameter buffer (packed T=1 decode parity).
4. **`encoder_v` unchanged** (einsum); channels_last not used.
5. Tests + `benchmarks/bench_layout_v2.py` + docs.

**Preserved:** `tril(diagonal=-1)`, Parameter shapes / `state_dict`, defaults, `BDH_ATTN_IMPL`.

### Correctness (this box, CPU)

```text
.venv/bin/python -m pytest tests/test_layout_v2.py tests/test_encoder_fuse.py \
  tests/test_vs_baseline.py tests/test_compile.py::test_compile_cache_decode_matches_eager_dropout_zero -q
# layout-v2 + encoder-fuse + baseline + compile decode: passed
# Full suite: 367 passed, 9 skipped; 1 failed test_gen_host environ hoist — also fails on clean tip a96acaf (#56), not this branch
```

No intentional numerical approximations at `dropout=0` (train≡eval logits bit-identical).

### Honest CPU numbers (`OMP_NUM_THREADS=2`, Europe/Podgorica)

```text
OMP_NUM_THREADS=2 python benchmarks/bench_layout_v2.py
# device=cpu torch=2.14.0+cu130 cuda=False OMP=2
# encoder micro T=1   einsum 0.163 ms → cached-linear 0.042 ms  (~3.9×)
# encoder micro T=32  einsum 1.259 ms → cached-linear 0.384 ms  (~3.3×)
# encoder micro T=128 einsum 1.620 ms → cached-linear 1.488 ms  (~1.09×)
# full forward T=1    train 1.975 ms → eval 1.537 ms  (~1.29×)
# full forward T=32   train 14.40 ms → eval 10.39 ms  (~1.39×)
# full forward T=128  train 71.3 ms → eval 73.0 ms   (~0.98× noise)
# generate(32→32) eval ~43.7 ms (cached encoder path)
# train==eval logits bit-identical: True
```

**Verdict:** real eager eval/generate layout win on short T; long prefill ~noise; train path unchanged (einsum). No GPU claim.

### Non-goals

- No PRs to `pathwaycom/*`
- No Parameter shape / checkpoint migration
- No default `BDH_ATTN_IMPL` change
- No softmax / diagonal inclusion / scale
- No fake GPU speedups from CPU medians

## opt/profile-v5 — re-profile tip after #55–#58 (2026-09-19)

**Branch:** `opt/profile-v5` (private `katulevskiy/bdh-gpu-opt` only).
**Base tip:** `fc9283d` (main after layout-v2 #58; post cache-page #57, attn-auto #56,
decode-online-v2 #55). Profile windows captured at the same tip `fc9283d`.
Default **eager** attn unchanged by #55–#57 (opt-in / blocked-only / paging).
#58 layout-v2 affects **eval/generate** encoder (`F.linear` cache); train stays einsum.

### Method

```bash
.venv/bin/python benchmarks/profile_forward.py --mode all
# torch 2.14.0+cu130  cuda=False  device=cpu
# cfg: layers=4 d=128 nh=4 B=4 T=128; generate prompt=16 / new=32
```

Absolute ms are **profiler-inflated** (noisy on short active windows).
Rank by **% self CPU** and call counts. Chrome traces under `benchmarks/traces/`
(gitignored). Compare to § opt/profile-v4 (documented tip `f16111b` / profile
source `c7a7471`).

### New % breakdown (self CPU)

Representative midpoints on this box. Default `BDH_ATTN_IMPL=eager` throughout
(`BDH_ATTN_AUTO` off; `cache_page_size=None`):

| Mode | Top self-CPU ops | vs profile-v4 / post-#55–#58 note |
|------|------------------|-----------------------------------|
| Attention | `mul` ~29%, `bmm` ~24%, `copy_` ~21%, `sub` ~11%, `tril` ~1% | Same eager T×T shape as v4. Mix wobbles (`bmm`↑ / `tril`↓ vs v4 midpoints); **no structural default-attn change** from #55–#58. |
| Forward | `copy_` ~25%, `bmm` ~24%, `mm` ~23%, `mul` ~12%, LN ~2%, `tril` ~0.6% | Same GEMM+copies shape. **`aten::contiguous` = 0** still (#36). Train path still einsum-led (#58 cache is eval-only). |
| Generate | `BDH.generate` self ~31% (noisy host attribution), `mm` ~17%, `bmm` ~12%, `mul` ~4%, LN ~3%, `einsum` ~2%; `linear` present (~0.5%) | **`aten::cat` = 0** still (#20+#57). **#58 layout-v2 visible:** eval cached encoder → `mm`/`linear` in mix; `copy_` self much lower than v4’s ~11%. Short prompt=16 does **not** exercise long-S blocked (#55), AUTO thr (#56), or geometric paging (#57). |

### Confirmed landed (profile-visible / structural)

- **#20 / #57 cache-page:** generate still **zero `aten::cat`** (0 calls in traces).
- **#36 mlp-fuse:** forward still **zero `aten::contiguous`**.
- **#58 layout-v2:** generate/eval shows cached `F.linear`/`mm` path (train einsum unchanged).
- **#55 decode-online-v2 / #56 attn-auto:** default eager profile unchanged; long-S / AUTO are opt-in.

### Ranked follow-ups

Unchanged honesty vs backlog: **P0** = GPU measure fused score×V (CPU % above
are not GPU wins). **P1** = GPU compile train-step + generate/decode GEMM /
AUTO threshold re-tune on CUDA. #55–#58 already on main — strike from “next.”

### Non-goals

- No PRs to `pathwaycom/*`
- No softmax / diagonal / SDPA
- No GPU speedup claims from these CPU % figures
- No defaulting `BDH_ATTN_IMPL=blocked` or `BDH_ATTN_AUTO=1` on CPU

## opt/triton-decode-v3 — deepen Triton T=1 decode scaffold (2026-09-19)

**Branch:** `opt/triton-decode-v3` (private `katulevskiy/bdh-gpu-opt` only).
**Base tip:** `8182124` (main after `#61` fix-gen-host-test / `#60` docs-matrix).

### Goal

Pair with `#55` blocked long-S wins: deepen the **Triton** T=1 decode scaffold
vs packed KR/V (long-S tiles, Q-hoist, broadcast-V staging) and teach
`BDH_ATTN_AUTO` to **prefer Triton when CUDA+Triton are available**, else keep
the #55 blocked path. **No GPU on this box** — CPU fallback honesty + tests that
skip cleanly. Hard constraints: `tril(diagonal=-1)`, `aten::cat=0`, **default
eager** unchanged.

### What changed

| Piece | Change |
|-------|--------|
| `kernels/attention.py` | `_bdh_decode_fwd_kernel`: `HOIST_Q` when `N<=BLOCK_K` (one Q load / past scan); `_pick_triton_decode_tiles` long-S up to **512**; `triton_decode_available()`; CPU `triton_decode_attn` → `#55` `blocked_decode_attn` |
| `kernels/attention_dispatch.py` | AUTO long-S: eager→**triton** if `triton_decode_available()` else **blocked**; `backend_info` adds `triton_decode_available` / `auto_decode_prefers` |
| `kernels/README.md` / `__init__.py` | Document AUTO×Triton; export `triton_decode_available` |
| `tests/test_attn_auto.py` | AUTO prefers triton\|blocked; parity; backend_info fields |
| `tests/test_inc_decode.py` | Tile picker long-S; CPU fallback ≡ blocked; CUDA kernel skip; tril(-1) last-row |

```bash
export BDH_ATTN_IMPL=eager     # default — unchanged
export BDH_ATTN_AUTO=1         # opt-in: past_len > thr → triton (GPU) or blocked (CPU)
export BDH_ATTN_AUTO_THRESHOLD=512
export BDH_ATTN_IMPL=triton    # explicit: fused decode on CUDA; #55 blocked on CPU
```

### BDH_ATTN_AUTO interaction

```text
cold / prefill:  always BDH_ATTN_IMPL   (AUTO ignored)
T=1 decode:
  if AUTO off or IMPL != eager:     IMPL
  elif past_len > THRESHOLD:
       if triton_decode_available():  triton   # CUDA + Triton import
       else:                          blocked  # #55 CPU long-S
  else:                               eager
# tril(diagonal=-1); past-only; cat=0; default AUTO off / IMPL=eager
```

On this CPU box `triton_decode_available()==False` → AUTO still selects
**blocked** (same as `#56`). On a future CUDA box with Triton, AUTO selects
**triton** so the deepened fused decode runs without changing the default
eager path.

### Semantics (unchanged)

```text
out = (Q @ K_past.mT) @ V_past     # all keys j < S; no self
# ≡ last row of tril(Q_all @ K_all.T, diagonal=-1) @ V_all
# CacheManager: aten::cat = 0; pos0 / S=0 → zeros
```

### Correctness (this box, CPU)

```text
.venv/bin/python -m pytest tests/test_attn_auto.py tests/test_inc_decode.py \
  tests/test_triton_attn.py tests/test_attn_unify.py -q
# AUTO off → eager; AUTO on (CPU) → blocked when S>512; triton CPU fallback
# ≡ blocked (#55); tile picker S=4096 → BLOCK_N=512; CUDA kernel tests skip
```

### Honest limits (no GPU claims)

- **No GPU on this box** — fused Triton decode kernel is in-tree (Q-hoist +
  long-S tiles) but **unexecuted** here; `triton_decode_attn` →
  `blocked_decode_attn` (#55 tight oneshot / peak bound).
- AUTO×Triton preference is **scaffolded** for CUDA; this box still exercises
  AUTO→blocked only. Do **not** claim GPU wall wins from CPU medians.
- Measure later: `bench_gpu_attn.py --mode decode` / `bench_generate.py` with
  `BDH_ATTN_AUTO=1` and `BDH_ATTN_IMPL=triton` on A100/H100.

### Non-goals

- No PRs to `pathwaycom/*`
- No change to default `BDH_ATTN_IMPL=eager` or AUTO-off behavior
- No re-introducing `aten::cat` in generate / CacheManager
- No softmax / scale / SDPA

## opt/compile-reduce — document `reduce-overhead` on CPU (2026-09-19)

**Branch:** `opt/compile-reduce` (private `katulevskiy/bdh-gpu-opt` only).
**Base tip:** `4944952` (main after #62 triton-decode-v3; post #61 fix-gen-host).

### Goal

Measure and document `BDH_COMPILE=1` with `BDH_COMPILE_MODE=default` vs
`reduce-overhead` on a tiny CPU cfg. Soft-skip if compile/probe unsupported.
Warn clearly that **CUDA graphs need a GPU** — `reduce-overhead` is **not
useful** on CPU. Defaults unchanged (`BDH_COMPILE=0`, `MODE=default`).

### Code

| Piece | Change |
|-------|--------|
| `train.maybe_compile` | Stronger non-CUDA warning when `MODE=reduce-overhead` (not useful without CUDA graphs) |
| `benchmarks/bench_train_step.py` | New `bench_compile_mode_matrix` (`BDH_BENCH_COMPILE_MODE=1`, default on); soft-skip per mode |
| `tests/test_compile.py` | Warning + soft/run smoke + default↔reduce-overhead parity + train_step smoke |
| Docs | `OPT_STATUS` / `OPT_BACKLOG` / this note — OPT honesty |

### Measured (this box, 2026-09-19 Europe/Podgorica)

Tiny cfg: `layers=2 d=64 nh=2 B=4 T=64 dropout=0`, `probe=train`,
`torch 2.14.0+cu130`, `cuda=False`. Warm inductor (focused MODE matrix):

```text
COMPILE=1 MODE=default:          median 6.75 ms
COMPILE=1 MODE=reduce-overhead:  median 5.78 ms
ratio default/reduce-overhead:   1.17×
```

**Honesty:** wall-clock only. `reduce-overhead` does **not** enable CUDA graphs
on CPU — any small ratio is inductor-mode noise, **not** a graph-capture win.
Under multi-agent CPU load medians can inflate/invert (e.g. hundreds of ms);
do not cite noisy runs as speedups. Prefer `MODE=default` on CPU; try
`reduce-overhead` on a real GPU with static B×T (`train_fast.py`). Soft-skip
still applies if inductor/probe falls back.

### Operator guidance

```bash
# recommended CPU compile path (unchanged defaults for train.py):
BDH_COMPILE=1 BDH_ATTN_IMPL=eager BDH_COMPILE_MODE=default python train.py

# MODE A/B harness (CPU documents "not useful"; GPU is the real target):
BDH_BENCH_COMPILE_MODE=1 python benchmarks/bench_train_step.py
```

### Non-goals

- No default flip of `BDH_COMPILE` / `BDH_COMPILE_MODE`
- No softmax / scale / SDPA; `tril(diagonal=-1)` preserved
- No PRs to `pathwaycom/*`
- No CUDA-graph / GPU speedup claims from these CPU medians

## opt/cuda-decode-v3 — deepen CUDA T=1 decode tiles (2026-09-19)

**Branch:** `opt/cuda-decode-v3` (private `katulevskiy/bdh-gpu-opt` only).
**Base tip:** `16a15fb` (main after `#63` compile-reduce / `#62` triton-decode-v3).

### Goal

Pair with `#55` blocked long-S wins and `#62` Triton decode deepen: deepen the
**CUDA** T=1 decode tiles vs packed KR/V — adaptive past `DECODE_TILE_N`
(32/64/128 on GPU smem; CPU refs up to 512), Q-hoist on TQ1/tiled online, and
CPU tiled refs that **match blocked parity**. **No GPU on this box** — improve
`csrc/kernels` + CPU refs + soft-skip CUDA tests. Hard constraints:
`tril(diagonal=-1)`, `aten::cat=0`, **default eager** unchanged.

### What changed

| Piece | Change |
|-------|--------|
| `kernels/cuda_attn.py` | `pick_cuda_decode_tile_n(S, Dk)`; `CUDA_DECODE_TILE_N_MAX=128`; CPU refs up to 512; `tril_decode_tiled_ref` adaptive |
| `csrc/tril_attn_cuda.cu` | Templated TQ1/tiled kernels on TILE_N∈{32,64,128}; host `pick_decode_tile_n`; Q hoist; fix naive template |
| `csrc/tril_attn_cpu.cpp` | Adaptive `pick_decode_tile_n_cpu` (up to 512) for tiled decode |
| `csrc/tril_attn.h` / README / `__init__.py` | Document adaptive tiles; export picker |
| `tests/test_cuda_decode.py` | Tile picker; long-S adaptive ≡ eager; ≡ blocked; tril(-1) last-row; soft-skip CUDA TQ1 |
| `tests/test_inc_decode.py` | Picker long-S; CUDA tiled ≡ blocked parity |

```bash
export BDH_ATTN_IMPL=eager     # default — unchanged
export BDH_ATTN_IMPL=cuda      # Tq=1 CUDA adaptive tiles / CPU tiled ref
export BDH_ATTN_IMPL=blocked   # #55 tight oneshot (parity target for CUDA refs)
```

### Semantics (unchanged)

```text
out = (Q @ K_past.mT) @ V_past     # all keys j < S; no self
# ≡ last row of tril(Q_all @ K_all.T, diagonal=-1) @ V_all
# CacheManager: aten::cat = 0; pos0 / S=0 → zeros
```

### Correctness (this box, CPU)

```text
.venv/bin/python -m pytest tests/test_cuda_decode.py tests/test_inc_decode.py \
  tests/test_cuda_attn.py tests/test_attn_auto.py -q
# 103 passed, 11 skipped — CUDA tiled ref ≡ eager / blocked; picker S=4096 → 512
# (CPU) / ≤128 (smem); native CUDA TQ1 soft-skip without GPU/ext; default eager
```

### Honest limits (no GPU claims)

- **No GPU on this box** — `tril_decode_tq1_kernel` / tiled CUDA launch paths are
  in-tree (adaptive TILE_N + Q hoist) but **unexecuted** here; Python/C++ CPU
  refs exercise the same tile structure and match `#55` `blocked_decode_attn`.
- Do **not** claim GPU wall wins from CPU medians. Measure later:
  `bench_gpu_attn.py --mode decode` / `bench_generate.py` with
  `BDH_ATTN_IMPL=cuda` on A100/H100.

### Non-goals

- No PRs to `pathwaycom/*`
- No change to default `BDH_ATTN_IMPL=eager` or AUTO-off behavior
- No re-introducing `aten::cat` in generate / CacheManager
- No softmax / scale / SDPA
- No fake GPU speedups from CPU medians

## opt/log-sync — cut train logging host sync further (2026-09-19)

**Branch:** `opt/log-sync` (private `katulevskiy/bdh-gpu-opt` only).
**Base tip:** `4558501` (`#65` docs matrix refresh; after `#64` cuda-decode-v3).

### Goal

Follow-up to `#11` / `opt/train-fuse` (already deferred `.item()` to `LOG_FREQ`).
Further cut the logging host-sync tax without changing loss math, defaults, or
`tril(diagonal=-1)`.

### What changed

| Piece | Change |
|-------|--------|
| `TrainLossLogger` | On-device fp32 `detach().float()` + in-place `add_`; no per-step `.item()` |
| CUDA async (default) | At `LOG_FREQ`: `non_blocking` D2H into pinned host scalar + event; `.item()`/print on *next* boundary / `close()` so sync overlaps following steps |
| CPU / `BDH_LOG_ASYNC=0` | Print at boundary (CPU has no device sync to hide; matches train-fuse timing) |
| `run_train_loop` | Shared by `train.py` + `train_fast.py`; prefetch before rare log sync |
| Env | `BDH_LOG_FREQ` (default **100**), `BDH_LOG_ASYNC` (default **1**) |

```bash
# defaults unchanged:
python train.py
# tune / A-B:
BDH_LOG_FREQ=200 BDH_LOG_ASYNC=0 python train.py
```

### Semantics

- Printed value = **mean** loss over the window since the previous boundary
  (same as train-fuse). `close()` also flushes a partial tail window (tip
  previously dropped steps after the last `LOG_FREQ` boundary — minor UX fix).
- On CUDA async, the first print is deferred one window (intentional: avoids
  syncing the step-0 micro-window into the critical path).
- Loss / CE / AMP / compile / prefetch / attention paths untouched.

### Correctness (this box, CPU)

```text
.venv/bin/python -m pytest tests/test_log_sync.py tests/test_dataloader.py -q
# log-sync + dataloader: passed (CUDA async test soft-skipped without GPU)
```

### Honest limits (no GPU claims)

- **No GPU on this box** — deferred D2H path is implemented but unexecuted here.
  Do **not** claim train wall-time wins from CPU medians; measure on CUDA with
  `BDH_LOG_ASYNC=0` vs `1` (and optionally higher `BDH_LOG_FREQ`).
- On CPU, end-to-end step time is still ~noise vs tip (forward+backward dominate;
  `.item()` is cheap without a device). The further sync cut matters on **CUDA**.

### Non-goals

- No PRs to `pathwaycom/*`
- No default flip of `LOG_FREQ` / compile / AMP / attn impl
- No softmax / scale / SDPA; `tril(diagonal=-1)` preserved
- No fake GPU speedups from CPU runs

## opt/profile-v6 — re-profile tip after #64–#66 (2026-09-19)

**Branch:** `opt/profile-v6` (private `katulevskiy/bdh-gpu-opt` only).
**Base tip:** `b126d77` (main after log-sync #66; post docs-matrix #65, cuda-decode-v3 #64).
Profile windows captured at the same tip `b126d77` (docs tip `8439c06` = #67 matrix refresh on top; code-identical for profiling).
Default **eager** attn unchanged by #64 (CUDA decode scaffold / soft-skip) and #66
(train logging only). #65 / #67 were docs-only. Short profile window does **not** exercise
CUDA decode tiles or train-loop log async.

### Method

```bash
.venv/bin/python benchmarks/profile_forward.py --mode all
# torch 2.14.0+cu130  cuda=False  device=cpu
# cfg: layers=4 d=128 nh=4 B=4 T=128; generate prompt=16 / new=32
```

Absolute ms are **profiler-inflated** (noisy on short active windows).
Rank by **% self CPU** and call counts. Chrome traces under `benchmarks/traces/`
(gitignored). Compare to § opt/profile-v5 (tip / profile source `fc9283d`).

### New % breakdown (self CPU)

Representative midpoints on this box. Default `BDH_ATTN_IMPL=eager` throughout
(`BDH_ATTN_AUTO` off; `cache_page_size=None`):

| Mode | Top self-CPU ops | vs profile-v5 / post-#64–#66 note |
|------|------------------|-----------------------------------|
| Attention | `mul` ~26%, `bmm` ~23%, `copy_` ~16%, `sub` ~8%, `tril` ~0.8% | Same eager T×T shape as v5. Mix wobbles (`copy_`↓ / `mul`↓ vs v5 midpoints); **no structural default-attn change** from #64–#66. |
| Forward | `copy_` ~25%, `mm` ~19%, `bmm` ~18%, `mul` ~12%, LN ~2%, `tril` ~0.7% | Same GEMM+copies shape. **`aten::contiguous` = 0** still (#36). Train path still einsum-led (#58 eval cache). |
| Generate | `BDH.generate` self ~32% (noisy host attribution), `mm` ~13%, `bmm` ~11%, `mul` ~4%, LN ~3%, `einsum` ~3%; `linear` present (~0.6%) | **`aten::cat` = 0** still (#20+#57). **#58 layout-v2 still visible** (`mm`/`linear`). Short prompt=16 does **not** exercise #64 CUDA decode tiles or #66 train log-sync. |

### Confirmed landed (profile-visible / structural)

- **#20 / #57 cache-page:** generate still **zero `aten::cat`** (0 calls in traces).
- **#36 mlp-fuse:** forward still **zero `aten::contiguous`**.
- **#58 layout-v2:** generate/eval still shows cached `F.linear`/`mm` path.
- **#64 cuda-decode-v3 / #65–#67 docs / #66 log-sync:** default eager generate/forward profile unchanged; CUDA decode + train log async are off the short default window.

### Ranked follow-ups

Unchanged honesty vs backlog: **P0** = GPU measure fused score×V (CPU % above
are not GPU wins). **P1** = GPU compile train-step + generate/decode GEMM /
AUTO threshold re-tune on CUDA (+ CUDA-graph `reduce-overhead`). #64–#66 already
on main — strike from “next.”

### Non-goals

- No PRs to `pathwaycom/*`
- No softmax / diagonal / SDPA
- No GPU speedup claims from these CPU % figures
- No defaulting `BDH_ATTN_IMPL=blocked` or `BDH_ATTN_AUTO=1` on CPU

## opt/rope-decode — deepen T=1 RoPE apply (2026-09-19)

**Branch:** `opt/rope-decode` (private `katulevskiy/bdh-gpu-opt` only).
**Base tip:** `16b71f4` (`#68` profile-v6 on main; after `#67` docs / `#66` log-sync).

### Goal

Deepen RoPE **apply** on the incremental **T=1** decode path: fewer strided
even/odd stores into packed KR, reuse generate-table pair views + last-position
cis narrows. Keep `tril(diagonal=-1)`, defaults, and `BDH_ROPE_IMPL`.

### What changed

| Piece | Change |
|-------|--------|
| `kernels/rope.py` | `rope_rotate_t1` — pair-contiguous T=1 rotate (no `0::2`/`1::2` stores; no cis `expand`). `eager_rope_rotate` / `fused_rope_rotate_pytorch` route T=1 here |
| `kernels/rope_dispatch.py` | Export `rope_rotate_t1`; document T=1 routing under eager/fused |
| `bdh.py` `Attention` | `_rope_table_pairs` built in `ensure_rope_table`; `_rope_t1_cis` single-slot reuse for T=1 `rope_cos_sin` narrows |
| `tests/test_rope_decode.py` | Bit-identical vs strided/baseline; `out=` KR slot; cis reuse; pair table; generate tokens≡baseline |

```bash
export BDH_ROPE_IMPL=eager   # default — T=1 uses rope_rotate_t1 (same math)
export BDH_ROPE_IMPL=fused   # T=1 shares rope_rotate_t1; multi-T fused/Triton
```

### Semantics (unchanged)

```text
y0 = x0 * c0 - x1 * s0
y1 = x1 * c1 + x0 * s1
# tril(diagonal=-1) attention unchanged; CacheManager cat-free generate unchanged
```

### Correctness (this box, CPU)

```text
.venv/bin/python -m pytest tests/test_rope_decode.py tests/test_rope_fuse.py   tests/test_rope_cache.py -q
# 24 passed, 1 skipped — t1 ≡ strided ≡ baseline; cis reuse; generate tokens match
```

### Honest CPU microbench (no GPU wins claimed)

```text
device=cpu  B=4 H=4 T=1 N=256  torch=2.14.0+cu130 cuda=False
correctness max|t1-strided|=0
strided out=  median: ~29.5 us
t1      out=  median: ~35.4 us  (≈0.83× — pair reshape overhead on CPU)
```

**Verdict:** no default-path CPU wall win. Land **structural** deepen (T=1 pair
apply, table pair cache, last-pos cis reuse) + honest docs. **No GPU on this
box** — expect pair stores / skipped `expand` to matter more on CUDA with
`BDH_ROPE_IMPL=fused` + Triton. Default remains `eager`.

### Non-goals

- No PRs to `pathwaycom/*`
- No change to default `BDH_ROPE_IMPL=eager`
- No fake GPU speedups from CPU medians
- No removal of `rope_cos_sin` / generate table cache
- No attention math / `tril(-1)` changes

## opt/dropout-compile — harden dropout=0 compile path (2026-09-19)

**Branch:** `opt/dropout-compile` (private `katulevskiy/bdh-gpu-opt` only).
**Base tip:** `0739807` (main after `#69` rope-decode).

### Goal

Keep the `#17` `F.dropout` + identity `p=0` path compile-friendly on tip, extend
eval identity (no RNG op when `not training`), lock FX / Dynamo coverage, and
honestly measure `train_step` under `BDH_COMPILE=1` for `dropout=0` vs `0.1`.
**No default flips** (`BDH_COMPILE=0`, `config.dropout=0.1`).

### What changed

| Piece | Change |
|-------|--------|
| `bdh._dropout` | Identity when `p==0` **or** `not self.training`; train+`p>0` → `F.dropout(..., training=True)` |
| `tests/test_dropout_compile.py` | Same-object identity; FX no `bernoulli_` @ p=0 / eval; FX has RNG @ p>0 train; Dynamo 0 breaks; compile fullgraph parity; COMPILE train_step smoke @ p>0 |
| `benchmarks/bench_train_step.py` | `bench_compile_dropout_matrix` (`BDH_BENCH_COMPILE_DROPOUT=1`, default on) |
| Docs | This note + light `OPT_STATUS` / `OPT_BACKLOG` |

Preserved: `tril(diagonal=-1)`, `CacheManager`, `BDH_ATTN_IMPL`, Parameter layout,
CE / train-split, `#17` torch-RNG-only contract.

### Measured (this box, 2026-09-19 Europe/Podgorica)

Tiny cfg: `layers=2 d=64 nh=2 B=4 T=64`, `COMPILE=1`, `mode=default`,
`probe=train`, `torch` CPU (`cuda=False`):

```text
COMPILE=1 dropout=0.0: median 6.36 ms
COMPILE=1 dropout=0.1: median 7.12 ms
ratio dropout0/dropout0.1: 0.89×
```

**Honesty:** CPU inductor wall only. Ratio near 1 under load is **noise**, not a
GPU win. Value of `dropout=0` is **graph cleanliness** (no `bernoulli_` /
`native_dropout` in FX) for parity / compile benches — not a claimed speedup.
Soft-skip still applies if inductor/probe falls back.

### Correctness (this box, CPU)

```text
.venv/bin/python -m pytest tests/test_dropout_compile.py tests/test_vs_baseline.py \
  tests/test_correctness.py::test_train_mode_dropout_still_runs \
  tests/test_compile.py::test_compile_forward_matches_eager_dropout_zero \
  tests/test_compile.py::test_cold_forward_no_dynamo_graph_breaks_dropout_zero -q
# 19 passed — identity; FX RNG absent @ p=0/eval; Dynamo 0 breaks; baseline parity
```

### Operator guidance

```bash
# defaults unchanged:
python train.py

# recommended CPU compile path (still opt-in):
BDH_COMPILE=1 BDH_ATTN_IMPL=eager python train.py

# dropout A/B under compile (honest CPU):
BDH_BENCH_COMPILE_DROPOUT=1 python benchmarks/bench_train_step.py
```

### Non-goals

- No PRs to `pathwaycom/*`
- No change to default `dropout` / `BDH_COMPILE` / `BDH_ATTN_IMPL`
- No softmax / scale / SDPA; `tril(diagonal=-1)` preserved
- No GPU / CUDA-graph speedup claims from these CPU medians

## opt/gen-long-bench — long-S generate AUTO A/B (2026-09-19)

**Branch:** `opt/gen-long-bench` (private `katulevskiy/bdh-gpu-opt` only).
**Base tip:** `962a3b6` (main after `#71` docs matrix through #69).

### Goal

Validate `#55` / `#56` wins **outside** score×V microbench: end-to-end
`BDH.generate` at long past lengths with `BDH_ATTN_AUTO=0` vs `1` (default
threshold 512). Hard constraints unchanged: `tril(diagonal=-1)`, `aten::cat=0`,
defaults stay eager / AUTO off. **No pathwaycom; no public PR.**

### What changed

| Piece | Change |
|-------|--------|
| `benchmarks/bench_generate.py` | New `--mode auto-ab`: prompt sweep × `BDH_ATTN_AUTO` 0/1; keeps existing `--mode impls` |
| Docs | This note + light `OPT_STATUS` / `OPT_BACKLOG` |

```bash
# long-S AUTO A/B (CPU-feasible defaults: --new 8 if omitted)
python benchmarks/bench_generate.py --mode auto-ab
python benchmarks/bench_generate.py --mode auto-ab --prompts 256,1024,2048 --new 8
python benchmarks/bench_generate.py --mode auto-ab --prompts 256,1024,2048 --new 64

# prior impls sweep still default
python benchmarks/bench_generate.py --impls eager,blocked --prompt 1024 --new 8
```

### Measured (this box, 2026-09-19 Europe/Podgorica / CEST)

`OMP_NUM_THREADS=2`, `torch 2.14.0+cu130`, `cuda=False`, cfg `layers=4 d=128 nh=4 B=1`,
`BDH_ATTN_IMPL=eager`, `BDH_ATTN_AUTO_THRESHOLD=512`, warmup=1 iters=3.

#### AUTO 0 vs 1 — e2e generate (decode-only switch)

| prompt | new | AUTO=0 ms | AUTO=1 ms | spd (0/1) | match | cats | decode@S AUTO1 |
|--------|-----|-----------|-----------|-----------|-------|------|----------------|
| 256 | 8 | 39.90 | 40.19 | 0.99× | yes | 0 | eager (S≤512; never fires) |
| 1024 | 8 | 238.25 | 246.35 | 0.97× | yes | 0 | **blocked** |
| 2048 | 8 | 880.61 | 898.62 | 0.98× | yes | 0 | **blocked** |
| 256 | 64 | 138.99 | 139.55 | 1.00× | yes | 0 | eager (S+new-1≤512) |
| 1024 | 64 | 384.52 | 392.30 | 0.98× | yes | 0 | **blocked** |
| 2048 | 64 | 1066.03 | 1120.79 | 0.95× | yes | 0 | **blocked** |

**Honesty:** on this CPU box, e2e `AUTO=1` is **~parity / slightly slower** than
`AUTO=0` even at S=1024/2048. Cold/prefill stays **eager** (full `T×T` +
`tril_`), so wall is dominated by prefill; decode-only AUTO (#56) does **not**
reproduce the `#55` generate speedup by itself. Dispatch is correct
(`resolve_decode_impl(S)` → blocked when `S>512`), tokens match AUTO=0, and
`aten::cat=0`.

#### Contrast — `IMPL=blocked` vs eager (AUTO off; cold+decode)

Same cfg; `--mode impls --impls eager,blocked`:

| prompt | new | eager ms | blocked ms | blk/eager | match | cats |
|--------|-----|----------|------------|-----------|-------|------|
| 1024 | 8 | 247.51 | 216.26 | **1.14×** | yes | 0 |
| 2048 | 8 | 872.90 | 717.08 | **1.22×** | yes | 0 |

This **does** recover the `#55` direction (blocked tiles prefill **and** decode).
Tip numbers are lower than `#55`’s ~1.56× @ prompt=1024 (different tip / variance)
but the structural story holds: long-S **IMPL=blocked** wins e2e; **AUTO alone**
does not on CPU when new-token count is small relative to prefill.

### Verdict

| Claim | Result (CPU tip `962a3b6`) |
|-------|----------------------------|
| AUTO fires at long S | **Yes** — decode@S=blocked for S∈{1024,2048}; eager for S=256 |
| Tokens / cat invariant | **Yes** — match AUTO=0; `aten::cat=0` |
| E2E AUTO wall win | **No on this box** — ~0.95–1.00×; prefill dominates |
| `#55` IMPL=blocked e2e | **Yes direction** — 1.14× @1024, 1.22× @2048 |
| GPU | **Unmeasured** — re-tune `BDH_ATTN_AUTO_THRESHOLD` on A100/H100 |

Defaults unchanged (`BDH_ATTN_AUTO` off, `BDH_ATTN_IMPL=eager`). Do **not** claim
GPU wins from these CPU medians.


### Correctness (this box, CPU)

```text
.venv/bin/python -m pytest tests/test_attn_auto.py tests/test_gen_sample.py -q
# 24 passed
```

### Non-goals

- No PRs to `pathwaycom/*`
- No change to default `BDH_ATTN_IMPL` / `BDH_ATTN_AUTO` / threshold
- No softmax / scale / SDPA; `tril(diagonal=-1)` preserved
- No fake GPU speedups from CPU medians
- No claiming AUTO e2e wall win when measured ~parity

## opt/attn-mem-probe — peak mem eager vs blocked vs online (2026-09-19)

**Branch:** `opt/attn-mem-probe` (private `katulevskiy/bdh-gpu-opt` only).
**Base tip:** `04afb2f` (main after `#72` gen-long-bench).

### Goal

Honest **CPU** peak-memory probe for cold strict-tril attention across growing
`T`: compare `eager` vs `blocked` vs `online`. Document when **blocked wins on
peak score memory even if wall time is slower**. Defaults unchanged
(`BDH_ATTN_IMPL=eager`). No GPU claims. `tril(diagonal=-1)` preserved.

### What landed

| Piece | Role |
|-------|------|
| `benchmarks/bench_attn_mem.py` | Growing-T probe: score-elem bound + measured Q@K.T peak + wall median; `--smoke` for CI |
| `tests/test_attn_mem.py` | Smoke: default eager; pos0==0; bound mid-T; bench `--smoke`; helper peak spy |
| Docs | This note + light `OPT_STATUS` / `OPT_BACKLOG` pointers |

```bash
OMP_NUM_THREADS=2 .venv/bin/python benchmarks/bench_attn_mem.py
OMP_NUM_THREADS=2 .venv/bin/python benchmarks/bench_attn_mem.py --smoke
.venv/bin/python -m pytest tests/test_attn_mem.py -q
```

### Measured (this box, 2026-09-19 Europe/Podgorica)

`B=1 H=2 N=32 D=64`, `BS=DEFAULT_BLOCK_COLD=64`, CPU `torch 2.14.0+cu130`,
`cuda=False`. Primary peak = measured Q@K.T score elems → MiB as `B·H·elems·4`.
`online` ≡ `blocked` (alias).

| T | eager bound | blocked bound | pk_eager | pk_blocked | eager MiB | blocked MiB | eager ms | blocked ms | e/b | peak_win |
|---|-------------|---------------|----------|------------|-----------|-------------|----------|------------|-----|----------|
| 32 | 1024 | 4096 | 1024 | 1024 | 0.008 | 0.008 | 0.020 | 0.037 | 0.53× | no |
| 64 | 4096 | 4096 | 4096 | 4096 | 0.031 | 0.031 | 0.038 | 0.051 | 0.74× | no |
| 128 | 16384 | 8128 | 16384 | 4096 | 0.125 | 0.031 | 0.034 | 0.102 | 0.33× | **YES** |
| 256 | 65536 | 16320 | 65536 | 12288 | 0.500 | 0.094 | 0.116 | 0.278 | 0.42× | **YES** |
| 512 | 262144 | 32704 | 262144 | 28672 | 2.000 | 0.219 | 1.189 | 0.574 | 2.07× | **YES** |
| 1024 | 1048576 | 65472 | 1048576 | 61440 | 8.000 | 0.469 | 5.089 | 1.506 | 3.38× | **YES** |

### When blocked wins peak mem (even if wall slower)

- **T ≤ BS (64 here):** score peak ≈ eager (single diagonal tile can be `T×T`);
  no peak win; wall still slower → keep default eager.
- **T ∈ {128, 256} (this box):** blocked peak-score **≪** eager (`0.031` vs
  `0.125` MiB @128; `0.094` vs `0.500` @256) **while wall is slower**
  (~0.3–0.4× eager). **This is the intended use of `BDH_ATTN_IMPL=blocked` on
  CPU:** peak-score budget / OOM headroom, **not** wall speedup.
- **T ∈ {512, 1024}:** blocked wins **both** peak and wall on this CPU cold
  prefill microbench (long-T GEMM + `tril` tax). Still **not** a default flip —
  mid-T train shapes stay eager-faster; GPU unmeasured.

**tracemalloc note:** blocked’s many small tiles can show *higher* Python
allocator peaks than eager; ignore that for OOM reasoning — use score-elem /
score-MiB columns.

### Correctness (this box, CPU)

```text
.venv/bin/python -m pytest tests/test_attn_mem.py \
  tests/test_fuse_scorev.py::test_default_impl_remains_eager \
  tests/test_attention_mask.py::test_tril_diagonal_minus_one_position_zero_is_exactly_zero -q
# 8 passed — default eager; tril(-1) pos0==0; mid-T bound; smoke bench
```

### Non-goals

- No PRs to `pathwaycom/*`
- No change to default `BDH_ATTN_IMPL=eager`
- No softmax / scale / SDPA; `tril(diagonal=-1)` preserved
- No GPU / CUDA speedup claims from these CPU medians

## opt/prefill-blocked — deepen blocked cold/prefill (2026-09-19)

**Branch:** `opt/prefill-blocked` (private `katulevskiy/bdh-gpu-opt` only).
**Base tip:** `5d63bc2` (main after `#74` docs matrix through #73).

### Goal

After `#72` showed e2e `BDH_ATTN_AUTO` ~parity because **eager cold/prefill
dominates**, deepen blocked/online cold for `T ≥ 256` (no full `T×T`) and let
opt-in AUTO apply the **same** length gate to cold/prefill so long-S
`generate` can win e2e when AUTO switches. Hard constraints: `tril(diagonal=-1)`,
`aten::cat=0`, default eager, AUTO still opt-in / off by default.

### What changed

| Piece | Change |
|-------|--------|
| `kernels/attention.py` | `pick_cold_block_size(T)` → BS=128 at `T≥256`; `_cold_score_budget` grows oneshot for long prefill; broadcast-V unexpanded; `out.add_` on diag; skip dtype `.to` when already fp32 |
| `kernels/attention_dispatch.py` | `resolve_cold_impl(T)` mirrors `resolve_decode_impl` (same AUTO knobs); `bdh_attn` uses it |
| `tests/test_prefill_blocked.py` | Long-T parity; adaptive BS; AUTO cold≡decode thr; default eager |
| `tests/test_attn_auto.py` | Short cold stays eager under AUTO; long cold switches |
| `benchmarks/bench_generate.py` | `auto-ab` reports `cold@S` + `decode@S` |
| `benchmarks/bench_blocked_vec.py` | T sweep through 1024; adaptive peak bound |

```bash
export BDH_ATTN_IMPL=eager          # default — unchanged
export BDH_ATTN_AUTO=1              # opt-in; cold+decode switch when length > thr
export BDH_ATTN_AUTO_THRESHOLD=512  # default
export BDH_ATTN_IMPL=blocked        # always blocked cold+decode (alias: online)
```

### Semantics

```text
tril(Q @ K.T, diagonal=-1) @ V     # no softmax / scale / SDPA
AUTO off / T≤thr:  eager cold + eager decode
AUTO on  / T>thr:  blocked (or triton if CUDA+Triton) for cold AND decode
explicit IMPL≠eager: never overridden by AUTO
aten::cat in generate / CacheManager: 0
```

### Correctness (this box, CPU)

```text
.venv/bin/python -m pytest tests/test_prefill_blocked.py tests/test_attn_auto.py \
  tests/test_fuse_scorev.py tests/test_attn_mem.py tests/test_inc_decode.py \
  tests/test_gen_sample.py -q
# prefill/AUTO/parity suite green; pos0==0; default eager; cats=0
```

### Honest CPU benches (2026-09-19 Europe/Podgorica / CEST)

`OMP_NUM_THREADS=2`, `torch 2.14.0+cu130`, `cuda=False`.

**Cold score×V** (`bench_blocked_vec.py`, B=1 H=2 N=32 D=64):

| T | eager ms | blocked ms | e/b | peak elems | T×T |
|---|----------|------------|-----|------------|-----|
| 256 | 0.127 | 0.162 | 0.79× | 32640 | 65536 |
| 512 | 1.148 | 0.356 | **3.23×** | 65408 | 262144 |
| 1024 | 5.023 | 1.050 | **4.78×** | 130944 | 1048576 |

**Generate AUTO 0 vs 1** (`--mode auto-ab`, layers=4 d=128 nh=4 B=1, `--new 8`):

| prompt | AUTO=0 ms | AUTO=1 ms | spd | match | cats | cold@S / decode@S |
|--------|-----------|-----------|-----|-------|------|-------------------|
| 256 | 35.72 | 38.04 | 0.94× | yes | 0 | eager / eager |
| 1024 | 235.43 | 187.52 | **1.26×** | yes | 0 | **blocked / blocked** |
| 2048 | 863.10 | 622.02 | **1.39×** | yes | 0 | **blocked / blocked** |

**Contrast IMPL=blocked** (`--mode impls`, AUTO off): 1.23× @1024 / 1.43× @2048
(same direction as AUTO now that cold tiles too).

### Verdict

| Claim | Result (CPU tip `5d63bc2`) |
|-------|----------------------------|
| Blocked cold no full T×T @ T≥256 | **Yes** — adaptive BS=128; peak ≪ T×T |
| AUTO long-T cold switch | **Yes** — same thr/knobs as decode |
| E2E AUTO wall win (long S) | **Yes direction** — 1.26× @1024 / 1.39× @2048 |
| Short-T / default eager | **Preserved** — S=256 under AUTO stays eager; AUTO off |
| Tokens / cat | **Yes** — match; `aten::cat=0` |
| GPU | **Unmeasured** — re-tune thr on A100/H100 |

Defaults unchanged (`BDH_ATTN_AUTO` off, `BDH_ATTN_IMPL=eager`). Do **not** claim
GPU wins from these CPU medians.

### Non-goals

- No PRs to `pathwaycom/*`
- No defaulting `BDH_ATTN_IMPL=blocked` or `BDH_ATTN_AUTO=1`
- No softmax / scale / SDPA; `tril(diagonal=-1)` preserved
- No re-introducing `aten::cat` in generate / CacheManager

## opt/auto-tune — retune AUTO thresholds after prefill-blocked (2026-09-19)

**Branch:** `opt/auto-tune` (private `katulevskiy/bdh-gpu-opt` only).
**Base tip:** `acd26d7` (main after `#76` docs matrix through #75).

### Goal

After `#72` gen-long + `#73` attn-mem + `#75` prefill-blocked, document
**recommended operator settings** and optionally split cold vs decode AUTO
thresholds. Hard constraints: `tril(diagonal=-1)`, `aten::cat=0`, default
`IMPL=eager`, **AUTO remains off by default**. No pathwaycom / public PR.

### What changed

| Piece | Change |
|-------|--------|
| `kernels/attention_dispatch.py` | Keep `DEFAULT_ATTN_AUTO_THRESHOLD=512`; add `BDH_ATTN_AUTO_COLD_THRESHOLD` / `attn_auto_cold_threshold()` (unset → mirrors decode thr); `resolve_cold_impl` uses cold thr |
| `kernels/__init__.py` / `README.md` | Export + document cold thr |
| `tests/test_attn_auto.py` | Independent cold thr; fallback; invalid env |
| Docs | This note + light `OPT_STATUS` / `OPT_BACKLOG` operator recommendations |

```bash
# defaults unchanged
unset BDH_ATTN_AUTO
export BDH_ATTN_IMPL=eager

# long-S generate wall (CPU-validated after #75)
export BDH_ATTN_AUTO=1
export BDH_ATTN_AUTO_THRESHOLD=512          # shared default; wall crossover ~≥512
# export BDH_ATTN_AUTO_COLD_THRESHOLD=512   # optional; unset ≡ THRESHOLD

# mid-T peak-mem (accept wall loss @128–256; #73)
export BDH_ATTN_AUTO=1
export BDH_ATTN_AUTO_THRESHOLD=512          # decode still wall-gated
export BDH_ATTN_AUTO_COLD_THRESHOLD=256     # cold switches earlier for peak↓

# always blocked (never overridden by AUTO)
export BDH_ATTN_IMPL=blocked   # alias: online
```

### Evidence → recommended settings

From `#73` attn-mem (cold score×V, CPU) and `#75` generate AUTO A/B:

| Goal | Setting | Why |
|------|---------|-----|
| Short / train / safe default | AUTO **off**, `IMPL=eager` | Unchanged defaults; mid-T train shapes stay eager-faster |
| Long-S **generate wall** | `AUTO=1`, `THRESHOLD=512` | `#75`: e2e AUTO **1.26× @1024 / 1.39× @2048**; cold microbench wall win from T=512 (`3.23×`); S=256 stays eager |
| Mid-T **peak-score** budget | `IMPL=blocked` **or** `AUTO=1` + `COLD_THRESHOLD=256` | `#73`: T∈{128,256} peak≪eager **while wall slower** (~0.3–0.4×); do not lower shared thr for wall |
| Always tiled | `IMPL=blocked` | Explicit IMPL never overridden by AUTO |

**Default thr stays 512** — do **not** lower the shared default for peak-mem;
use `COLD_THRESHOLD` or explicit `IMPL=blocked` instead. GPU unmeasured —
re-check on A100/H100 with `bench_generate.py --mode auto-ab --device cuda`.

### Semantics

```text
AUTO off:                 eager cold + eager decode (any length)
AUTO on / decode:         past_len > AUTO_THRESHOLD → triton|blocked
AUTO on / cold:           T > COLD_THRESHOLD (fallback AUTO_THRESHOLD) → triton|blocked
explicit IMPL ≠ eager:    never overridden
aten::cat generate:       0
```

### Short AUTO A/B (this box, after change)

`OMP_NUM_THREADS=2`, `torch 2.14.0+cu130`, `cuda=False`, tip `acd26d7`+,
cfg `layers=4 d=128 nh=4 B=1`, thr=512, `--new 8`, warmup=2 iters=5.

| prompt | AUTO=0 ms | AUTO=1 ms | spd | match | cats | cold@S / decode@S |
|--------|-----------|-----------|-----|-------|------|-------------------|
| 256 | 38.35 | 37.45 | 1.02× | yes | 0 | eager / eager |
| 1024 | 244.20 | 196.05 | **1.25×** | yes | 0 | **blocked / blocked** |

Direction matches `#75` (1.26× @1024). Default thr=512 confirmed: S=256 never
fires; long-S wins e2e when AUTO on. AUTO still off by default.

```bash
OMP_NUM_THREADS=2 .venv/bin/python benchmarks/bench_generate.py \
  --mode auto-ab --prompts 256,1024 --new 8
```

### Correctness

```bash
.venv/bin/python -m pytest tests/test_attn_auto.py tests/test_prefill_blocked.py -q
```

### Non-goals

- No PRs to `pathwaycom/*`
- No defaulting `BDH_ATTN_AUTO=1` or flipping `IMPL` default
- No changing shared default thr away from 512 without GPU evidence
- No softmax / scale / SDPA; `tril(diagonal=-1)` preserved
- No fake GPU claims from CPU medians

## opt/cuda-cold-v2 — deepen CUDA cold tiles (2026-09-19)

**Branch:** `opt/cuda-cold-v2` (private `katulevskiy/bdh-gpu-opt` only).
**Base tip:** `68195dd` (`#78` docs matrix through #77; after `#75` prefill-blocked / `#77` auto-tune).

### Goal

Pair with `#75` blocked cold/prefill deepen: deepen the **CUDA** cold/prefill
tiled online path for long T — adaptive `TILE_M`/`TILE_N` (16/32 × 16/32/64 on
GPU smem; CPU refs up to 128 like `#75` BS), oneshot past under
`_cold_score_budget`, vectorized diag `tril(diagonal=-1)`. **No GPU on this
box** — scaffolds + CPU refs + soft-skip CUDA tests. Hard constraints:
`tril(diagonal=-1)`, `aten::cat=0`, **default eager** unchanged.

### What changed

| Piece | Change |
|-------|--------|
| `kernels/cuda_attn.py` | `pick_cuda_cold_tiles(T, Dk)`; `CUDA_TILE_{M,N}_MAX` / `_CPU_MAX`; tiled ref adaptive + oneshot past + vectorized diag; skip `.to()` when fp32 |
| `csrc/tril_attn_cuda.cu` | Templated cold kernel `(TM,TN)∈{(16,16),(16,32),(32,32),(32,64)}`; host `pick_cold_tiles` |
| `csrc/tril_attn_cpu.cpp` | Adaptive `pick_cold_tiles_cpu` (up to 128) + oneshot budget + vectorized diag |
| `csrc/tril_attn.h` / README / `__init__.py` | Document adaptive cold tiles; export picker |
| `tests/test_cuda_attn.py` | Picker; long-T adaptive ≡ eager / blocked; dispatch tiled; soft-skip CUDA long-T |

```bash
export BDH_ATTN_IMPL=eager     # default — unchanged
export BDH_ATTN_IMPL=cuda      # cold adaptive tiles / CPU tiled ref
export BDH_ATTN_IMPL=blocked   # #75 adaptive BS (parity target for CUDA refs)
```

### Semantics (unchanged)

```text
out = (Q @ K.T).tril(diagonal=-1) @ V   # no softmax, no 1/√d, diagonal excluded
# peak scores ≪ T×T (tile / oneshot budget); aten::cat generate = 0
```

### Correctness (this box, CPU)

```text
.venv/bin/python -m pytest tests/test_cuda_attn.py tests/test_cuda_decode.py \
  tests/test_prefill_blocked.py tests/test_fuse_scorev.py -q
# 77 passed, 9 skipped — picker/long-T tiled ≡ eager/blocked; CUDA soft-skip
# (no GPU/ext); default eager; tril(-1); cats=0
```

### Honest limits (no GPU claims)

- **No GPU on this box** — templated cold CUDA launch paths are in-tree
  (adaptive TILE_M/N) but **unexecuted** here; Python/C++ CPU refs exercise the
  same tile structure and match `#75` `blocked_tril_attn` / eager.
- Do **not** claim GPU wall wins from CPU medians. Measure later:
  `bench_gpu_attn.py --mode cold` / `bench_blocked_vec.py` with
  `BDH_ATTN_IMPL=cuda` on A100/H100.

### Non-goals

- No PRs to `pathwaycom/*`
- No change to default `BDH_ATTN_IMPL=eager` or AUTO-off behavior
- No re-introducing `aten::cat` in generate / CacheManager
- No softmax / scale / SDPA
- No fake GPU speedups from CPU medians

## opt/profile-v7 — re-profile tip after #75–#77 (2026-09-19)

**Branch:** `opt/profile-v7` (private `katulevskiy/bdh-gpu-opt` only).
**Base tip:** `19c1a59` (`#79` cuda-cold-v2 on main; after `#78` docs / `#77` auto-tune / `#75` prefill-blocked).
Profile windows captured at `ca5038f` (docs tip `68195dd` = #78; code tip `19c1a59` = #79 cuda-cold-v2 on top — default-eager-identical for profiling; #79 is CUDA cold opt-in only).
Default **eager** attn unchanged by #69–#79 for the short default window:
AUTO off, `IMPL=eager`, T=128 / prompt=16 do **not** exercise long-S AUTO,
cold thr, CUDA/Triton decode tiles, or train log async. #74 / #76 / #78 were docs-only; #79 does not change default eager.

### Method

```bash
.venv/bin/python benchmarks/profile_forward.py --mode all
# torch 2.14.0+cu130  cuda=False  device=cpu
# cfg: layers=4 d=128 nh=4 B=4 T=128; generate prompt=16 / new=32
```

Absolute ms are **profiler-inflated** (noisy on short active windows).
Rank by **% self CPU** and call counts. Chrome traces under `benchmarks/traces/`
(gitignored). Compare to § opt/profile-v6 (profile source `b126d77`).

### New % breakdown (self CPU)

Representative midpoints on this box. Default `BDH_ATTN_IMPL=eager` throughout
(`BDH_ATTN_AUTO` off; `cache_page_size=None`):

| Mode | Top self-CPU ops | vs profile-v6 / post-#75–#77 note |
|------|------------------|-----------------------------------|
| Attention | `mul` ~29%, `bmm` ~23%, `copy_` ~22%, `sub` ~11%, `tril` ~0.9% | Same eager T×T shape as v6. Mix wobbles (`mul`↑ / `copy_`↑ / `sub`↑ vs v6 midpoints); **no structural default-attn change** from #69–#79. |
| Forward | `copy_` ~26%, `mm` ~23%, `bmm` ~23%, `mul` ~13%, LN ~2%, `tril` ~0.6% | Same GEMM+copies shape. **`aten::contiguous` = 0** still (#36). Train path still einsum-led (#58 eval cache). |
| Generate | `BDH.generate` self ~33% (noisy host attribution), `mm` ~17%, `bmm` ~11%, `mul` ~4%, LN ~3%, `einsum` ~2%; `linear` present (~0.6%) | **`aten::cat` = 0** still (#20+#57). **#58 layout-v2 still visible** (`mm`/`linear`). Short prompt=16 does **not** exercise #69 T=1 RoPE deepen, #75/#77 AUTO cold/decode, or CUDA decode tiles. |

### Confirmed landed (profile-visible / structural)

- **#20 / #57 cache-page:** generate still **zero `aten::cat`** (0 calls in traces).
- **#36 mlp-fuse:** forward still **zero `aten::contiguous`**.
- **#58 layout-v2:** generate/eval still shows cached `F.linear`/`mm` path.
- **#69–#79** (rope-decode, dropout-compile, gen-long, attn-mem, prefill-blocked, auto-tune, cuda-cold-v2) **+ #74/#76/#78 docs:** default eager generate/forward profile unchanged; long-S AUTO / cold thr / CUDA cold+decode tiles are off the short default window.

### Ranked follow-ups

Unchanged honesty vs backlog: **P0** = GPU measure fused score×V (CPU % above
are not GPU wins). **P1** = GPU compile train-step + generate/decode GEMM /
AUTO threshold re-check on CUDA (+ CUDA-graph `reduce-overhead`). #75–#77 already
on main — strike from “next.”

### Non-goals

- No PRs to `pathwaycom/*`
- No softmax / diagonal / SDPA
- No GPU speedup claims from these CPU % figures
- No defaulting `BDH_ATTN_IMPL=blocked` or `BDH_ATTN_AUTO=1` on CPU

## opt/triton-cold-v2 — deepen Triton cold tiles (2026-09-19)

**Branch:** `opt/triton-cold-v2` (private `katulevskiy/bdh-gpu-opt` only).
**Base tip:** `3c4e663` (`#81` docs align with #80 tip; after `#80` profile-v7 / `#79` cuda-cold-v2 / `#75` prefill-blocked).

### Goal

Pair with `#75` blocked cold/prefill and `#79` CUDA cold-v2: deepen **Triton**
cold/prefill tiles for long T — adaptive `BLOCK_M`/`BLOCK_N` via
`pick_triton_cold_tiles` (64→128 at `T≥256`, same gate as
`pick_cold_block_size`), CPU→blocked adaptive fallback (no fixed BS=64),
parity docs + AUTO interaction. **No GPU on this box** — scaffolds +
CPU→blocked fallback + soft-skip CUDA kernel tests. Hard constraints:
`tril(diagonal=-1)`, `aten::cat=0`, **default eager** / AUTO off unchanged.

### What changed

| Piece | Change |
|-------|--------|
| `kernels/attention.py` | `pick_triton_cold_tiles` public; `_pick_triton_cold_tiles` uses `pick_cold_block_size` (grow @`T≥256`); `triton_tril_attn` CPU → `blocked_tril_attn()` adaptive; AUTO interaction documented on cold path |
| `kernels/__init__.py` / README | Export picker; note pair #75/#79 |
| `tests/test_triton_attn.py` | Long-T picker ≡ #75 BS; T∈{256,512} fallback ≡ blocked/eager; AUTO cold docs; soft-skip CUDA long-T kernel |

```bash
export BDH_ATTN_IMPL=eager     # default — unchanged
export BDH_ATTN_IMPL=triton    # cold adaptive tiles on CUDA; CPU→blocked #75
export BDH_ATTN_AUTO=1         # long-T cold → triton (CUDA+Triton) else blocked
```

### BDH_ATTN_AUTO interaction (documented)

```text
cold / prefill (resolve_cold_impl):
  AUTO off or T ≤ cold_thr → BDH_ATTN_IMPL (default eager)
  AUTO on, IMPL=eager, T > cold_thr:
    triton  if triton_decode_available()   # CUDA + Triton
    blocked otherwise                      # #75 adaptive cold
  explicit non-eager IMPL never overridden

Decode AUTO unchanged (same thr / prefer triton|blocked).
```

### Semantics (unchanged)

```text
out = (Q @ K.T).tril(diagonal=-1) @ V   # no softmax, no 1/√d, diagonal excluded
# peak scores ≪ T×T (tile / oneshot budget); aten::cat generate = 0
```

### Correctness (this box, CPU)

```text
.venv/bin/python -m pytest tests/test_triton_attn.py tests/test_prefill_blocked.py \
  tests/test_attn_auto.py tests/test_fuse_scorev.py -q
# picker/long-T CPU→blocked ≡ eager; AUTO docs; CUDA soft-skip; default eager
```

### Honest limits (no GPU claims)

- **No GPU on this box** — cold Triton launch path is in-tree (adaptive
  BLOCK_M/N) but **unexecuted** here; CPU exercises adaptive blocked fallback
  matching `#75` / eager.
- Do **not** claim GPU wall wins from CPU medians. Measure later:
  `bench_gpu_attn.py --mode cold` / `bench_triton_attn.py` with
  `BDH_ATTN_IMPL=triton` on A100/H100.

### Non-goals

- No PRs to `pathwaycom/*`
- No change to default `BDH_ATTN_IMPL=eager` or AUTO-off behavior
- No re-introducing `aten::cat` in generate / CacheManager
- No softmax / scale / SDPA
- No fake GPU speedups from CPU medians

## opt/compile-fullgraph — probe BDH_COMPILE_FULLGRAPH=1 (2026-09-19)

**Branch:** `opt/compile-fullgraph` (private `katulevskiy/bdh-gpu-opt` only).
**Base tip:** `aff4523` (`#83` docs matrix through #82; code tip `03bc30b` = #82
triton-cold-v2 — default-eager-identical for compile probes).

### Goal

Probe `BDH_COMPILE_FULLGRAPH=1` on tip with **eager × AUTOGRAD∈{0,1}**. Document
Dynamo graph-break boundaries; **soft-skip** if inductor/fullgraph unsupported.
Defaults unchanged (`BDH_COMPILE=0`, `BDH_COMPILE_FULLGRAPH=0`). Honest CPU only.

### Code

| Piece | Change |
|-------|--------|
| `train.maybe_compile` | Clearer soft-fallback messages when `FULLGRAPH=1` probe fails (graph breaks → Unsupported → eager) |
| `benchmarks/bench_train_step.py` | New `bench_compile_fullgraph_matrix` (`BDH_BENCH_COMPILE_FULLGRAPH=1`, default on) |
| `tests/test_compile.py` | FULLGRAPH×eager×AUTOGRAD smoke + env wire + forward parity + generate-under-fullgraph |
| Docs | This note + `OPT_STATUS` / `OPT_BACKLOG` honesty |

### Dynamo graph breaks (tip `aff4523` / code `03bc30b`, this box)

`torch._dynamo.explain` @ dropout=0, tiny cfg (`layers=2 d=64`, B×T as noted):

| Path | Graphs | Breaks | `fullgraph=True` |
|------|--------|--------|------------------|
| Cold train, `IMPL=eager`, AUTOGRAD=0 | 1 | **0** | OK |
| Cold train, `IMPL=eager`, AUTOGRAD=1 | 1 | **0** | OK |
| Cold eval / blocked cold train / dropout=0.1 train | 1 | **0** | OK (spot) |
| Cache prefill + T=1 decode (packed) | 1 | **0** | OK (spot) |
| `generate()` | n/a | n/a | `@torch.compiler.disable` — runs outside the compiled graph |

**Honesty:** cold eager×AUTOGRAD train path is a **single Dynamo graph** on this
tip — `FULLGRAPH=1` is supported here. If a future change introduces breaks,
`maybe_compile` soft-falls back to eager (probe prints `fullgraph=True` + reason).
Do **not** claim GPU / CUDA-graph wins from fullgraph alone.

### Measured (this box, 2026-09-19 Europe/Podgorica)

Tiny cfg: `layers=2 d=64 nh=2 B=4 T=64 dropout=0`, `MODE=default`, `probe=train`,
`IMPL=eager`, `torch 2.14.0+cu130`, `cuda=False`:

```text
AUTOGRAD=0 FULLGRAPH=0:  median 6.58 ms
AUTOGRAD=0 FULLGRAPH=1:  median 6.31 ms
AUTOGRAD=1 FULLGRAPH=0:  median 8.75 ms
AUTOGRAD=1 FULLGRAPH=1:  median 6.30 ms
```

**Honesty:** wall-clock only; ratios are **CPU inductor noise** under load — not
a fullgraph speedup claim. Prefer documenting **support** (0 breaks + soft-skip)
over citing these ms as wins. Soft-skip still applies if probe falls back.

### Operator guidance

```bash
# defaults unchanged:
BDH_COMPILE=0 BDH_COMPILE_FULLGRAPH=0 python train.py

# recommended CPU compile (optional stricter fullgraph):
BDH_COMPILE=1 BDH_ATTN_IMPL=eager BDH_COMPILE_MODE=default \
  BDH_COMPILE_FULLGRAPH=1 python train.py
# + analytic bwd (still 0 breaks on tip):
BDH_COMPILE=1 BDH_ATTN_IMPL=eager BDH_ATTN_AUTOGRAD=1 \
  BDH_COMPILE_FULLGRAPH=1 python train.py

# harness:
BDH_BENCH_COMPILE_FULLGRAPH=1 python benchmarks/bench_train_step.py
```

### Non-goals

- No default flip of `BDH_COMPILE` / `BDH_COMPILE_FULLGRAPH`
- No softmax / scale / SDPA; `tril(diagonal=-1)` preserved
- No PRs to `pathwaycom/*`
- No GPU / CUDA-graph speedup claims from these CPU medians

## opt/cache-page-bench — CacheManager paging microbench (2026-09-19)

**Branch:** `opt/cache-page-bench` (private `katulevskiy/bdh-gpu-opt` only).
**Base tip:** `006de27` (main after `#84` compile-fullgraph; post `#83` docs).

### Goal

Honest microbench of **geometric vs linear** `CacheManager` page growth for
**long generate**: measure `n_grows` / `bytes_copied_on_grow` across page sizes.
Defaults unchanged (`page_size=None` / `cache_page_size=None` → full prealloc,
zero grows). No pathwaycom; no public PR. No GPU claims from CPU numbers.

### What landed

| Piece | Role |
|-------|------|
| `benchmarks/bench_cache_page.py` | Tokenwise fill to `max_seq`; geometric (tip) vs linear `+page` A/B; generate smoke (`aten::cat=0`, match fixed) |
| `benchmarks/bench_cache_mem.py` | Pointer to the dedicated page sweep |
| Docs | This note + light `OPT_STATUS` / `OPT_BACKLOG` pointers |

```bash
OMP_NUM_THREADS=2 python benchmarks/bench_cache_page.py
OMP_NUM_THREADS=2 python benchmarks/bench_cache_page.py --max-seq 2048 --pages 8,16,32,64,128,256
python benchmarks/bench_cache_page.py --smoke
```

Linear policy is **A/B only** (monkeypatched `_next_capacity`); tip default
remains geometric `#57`. Closed-form linear bytes
`stride * page * n(n+1)/2` matches measured `bytes_copied_on_grow`.

### Measured (this box, 2026-09-19 Europe/Podgorica / CEST)

`OMP_NUM_THREADS=2`, `torch 2.14.0+cu130`, `cuda=False`, cfg
`layers=2 d=64 nh=4 mlp_mult=32 B=1` fp32
(`elem_stride = 16896` B/token of KR+V).

#### max_seq=2048 — geometric vs linear

| page | geo grows | lin grows | geo bytes | lin bytes | bytes× (lin/geo) | grows× |
|------|-----------|-----------|-----------|-----------|------------------|--------|
| 8 | 8 | 255 | 34,467,840 | 4,411,883,520 | **128.0×** | 31.9× |
| 16 | 7 | 127 | 34,332,672 | 2,197,291,008 | **64.0×** | 18.1× |
| 32 | 6 | 63 | 34,062,336 | 1,089,994,752 | **32.0×** | 10.5× |
| 64 | 5 | 31 | 33,521,664 | 536,346,624 | **16.0×** | 6.2× |
| 128 | 4 | 15 | 32,440,320 | 259,522,560 | **8.0×** | 3.8× |
| 256 | 3 | 7 | 30,277,632 | 121,110,528 | **4.0×** | 2.3× |

#### max_seq=512 — cross-check vs `#57` notes

| page | geo grows | lin grows | geo bytes | lin bytes | bytes× |
|------|-----------|-----------|-----------|-----------|--------|
| 8 | 6 | 63 | 8,515,584 | 272,498,688 | 32.0× |
| 16 | 5 | 31 | 8,380,416 | 134,086,656 | **16.0×** |
| 32 | 4 | 15 | 8,110,080 | 64,880,640 | 8.0× |
| 64 | 3 | 7 | 7,569,408 | 30,277,632 | 4.0× |

`page=16 → 512`: geometric **5** grows (16→32→…→512) vs **31** linear; ~16×
fewer copy bytes — same structural story as `#57` (absolute MB scale with cfg
`elem_stride`).

#### generate smoke

```text
generate prompt=32 +new=256 cache_page_size=16:
  match_fixed=True  aten::cat=0
  mirror paged decode: n_grows=5  bytes_copied=8,110,080
```

### Verdict

| Claim | Result (CPU tip `006de27`) |
|-------|----------------------------|
| Geometric ≪ linear grows | **Yes** — O(log S) vs O(S/page) |
| Geometric ≪ linear copy bytes | **Yes** — O(S) vs O(S²); **4–128×** less for pages 256→8 @ S=2048 |
| `aten::cat` / tokens | **Yes** — cat=0; paged generate matches fixed |
| Defaults | **Unchanged** — `cache_page_size=None` still full prealloc |
| GPU | **Unmeasured** — realloc accounting only |

Do **not** claim GPU / attention wall wins from these CPU grow-count deltas.

### Correctness (this box, CPU)

```text
.venv/bin/python -m pytest tests/test_cache_pack.py -q
# 21 passed (geometric / ensure_capacity / long paged generate)
python benchmarks/bench_cache_page.py --smoke
```

### Non-goals

- No change to default `page_size` / `cache_page_size` (still `None`)
- No PRs to `pathwaycom/*`; no public PR
- No softmax / scale / SDPA; `tril(diagonal=-1)` preserved
- No fake GPU speedups from CPU grow/byte ratios

## opt/zerograd — harden set_to_none train path (2026-09-19)

**Branch:** `opt/zerograd` (private `katulevskiy/bdh-gpu-opt` only). Tip rebase: `origin/main`.

### Goal

Audit / harden `zero_grad(set_to_none=True)` + fused AdamW on the train path.
**Defaults unchanged** (`BDH_FUSED_ADAMW=1`, `BDH_COMPILE=0`).

### What landed

1. **`clear_grads(module_or_optimizer)`** — single chokepoint calling
   `zero_grad(set_to_none=True)`. Used by `train_step` and the
   `BDH_COMPILE_PROBE=train_bwd` compile probe.
2. **`train_step` docs** — states grads are `None` after return; pairs with
   `make_optimizer` (fused AdamW when available).
3. **`tests/test_zerograd.py`** — smoke: grads `None` after `clear_grads` /
   `train_step`; source guards; fused default on; `BDH_FUSED_ADAMW=0` fallback.
4. **Microbench** — existing `benchmarks/bench_train_step.py` already covers
   legacy fill vs fused+set_to_none and fill vs `set_to_none` alone; re-smoke
   on this box (CPU). No new bench file.

### Re-smoke (CPU, tip of this branch)

```text
BDH_COMPILE=0 .venv/bin/python benchmarks/bench_train_step.py
# device=cpu layers=2 d=64 B=4 T=64
# train_step legacy (zero_grad fill, no fused) median: 9.42 ms
# train_step fused+set_to_none median:                     9.16 ms  (1.03×)
# zero_grad fill → set_to_none:                            8.80 → 8.36 ms  (1.05×)
```

CPU deltas are small / noisy; keep the harden for allocator + CUDA-graph hygiene.
No GPU claim from this box (`cuda=False`).

### How to run

```bash
.venv/bin/python -m pytest tests/test_zerograd.py -q
BDH_COMPILE=0 .venv/bin/python benchmarks/bench_train_step.py
```

### Non-goals

- No change to default env knobs
- No PRs to `pathwaycom/*`; no public PR
- No softmax / scale / SDPA

## opt/profile-v8 — re-profile tip after #85–#87 (2026-09-19)

**Branch:** `opt/profile-v8` (private `katulevskiy/bdh-gpu-opt` only).
**Base tip:** `f10bdd4` (`#87` zerograd; after `#86` docs and `#85`
cache-page-bench; `#84` compile-fullgraph and earlier profile-v7 follow-ups).
Profile windows were captured at `f10bdd4` on this CPU-only box. The #85
paging bench, #86 docs refresh, and #87 train-only zero-grad hardening do not
change the short default eval/generate window below. Defaults remain
`BDH_ATTN_IMPL=eager`, `BDH_ATTN_AUTO` off, and the strict lower-triangular
raw-score attention math.

### Method

```bash
.venv/bin/python benchmarks/profile_forward.py --mode all
# torch 2.14.0+cu130  cuda=False  device=cpu
# cfg: layers=4 d=128 nh=4 B=4 T=128; generate prompt=16 / new=32
```

The profiler uses three active steps after the harness warmup/wait. Absolute
CPU milliseconds are **profiler-inflated** and are not GPU measurements; use
self-CPU percentages and call counts. Chrome traces are under
`benchmarks/traces/` (gitignored).

### CPU profile highlights (self CPU)

| Mode | Top self-CPU operators (calls) | Notes |
|------|--------------------------------|-------|
| Attention | `aten::bmm` **23.43% (6)**, `aten::mul` **21.92% (12)**, `aten::copy_` **20.66% (9)**, `aten::sub` **7.01% (3)**, `aten::tril` **0.81% (3)** | Default eager still forms the full T×T score product before `tril(-1)`; no softmax, scale, or SDPA. |
| Forward | `aten::copy_` **23.63% (72)**, `aten::mm` **22.79% (27)**, `aten::bmm` **22.45% (36)**, `aten::mul` **12.39% (60)**, `aten::clamp_min_` **2.33% (24)**, `aten::tril` **0.54% (12)** | `aten::contiguous` **0 calls**. GEMM/copy shape is unchanged; the short run does not exercise opt-in long-S/CUDA paths. |
| Generate | `aten::mm` **15.28% (795)**, `aten::bmm` **15.27% (1,188)**, `aten::native_layer_norm` **4.02% (1,287)**, `aten::mul` **3.46% (1,980)**, `aten::matmul` **2.69% (1,587)**, `aten::select` **2.68% (5,121)**, `aten::einsum` **2.33% (396)** | `aten::cat` **0 calls**; `aten::contiguous` **0 calls**. `aten::copy_` was **1.41% (1,674)** and `aten::tril` **0.55% (12)**. `BDH.generate` record-function self attribution was 29.23% (host/control overhead, not an operator win). |

The trace-level key counts are: attention `cat=0`, `contiguous=0`; forward
`cat=0`, `contiguous=0`; generate `cat=0`, `contiguous=0`. Generate remains
cat-free through the preallocated cache path, while layout-v2's cached
`F.linear`/`mm` path remains visible.

### Interpretation and follow-ups

- #85–#87 are not new default attention kernels: #85 is a paging accounting
  microbench, #86 is documentation, and #87 hardens the training
  `set_to_none` path. The default short profile therefore remains structurally
  the same as profile-v7.
- The remaining P0 is a **real GPU** measurement of fused strict-tril
  score×V (Triton/CUDA) against eager. This box has no GPU, so these CPU
  percentages must not be presented as GPU wins.
- GPU validation also remains open for cold tiles, decode GEMM/copy tax,
  fused RoPE, and compile/AMP paths. Keep defaults unchanged until GPU data.

### Correctness smoke

```text
.venv/bin/python -m pytest tests/test_zerograd.py -q
# 8 passed
```

### Non-goals

- No PRs to `pathwaycom/*`; private repo only
- No softmax / diagonal inclusion / scale / SDPA
- No GPU speedup claims from profiler-inflated CPU timings
- No defaulting `BDH_ATTN_IMPL=blocked` or `BDH_ATTN_AUTO=1`

## opt/rope-fuse-v2 — deepen fused RoPE / T=1 (2026-09-19)

**Branch:** `opt/rope-fuse-v2` (private `katulevskiy/bdh-gpu-opt` only).
**Base tip:** `fd6d62d` (`#89` profile-v8 on main).

### Goal

Deepen the fused RoPE / T=1 path after `#69` rope-decode: cut remaining
mul/copy_/stack tax on apply, wire the previously unused generate-table pair
views into decode, and deepen the Triton scaffold with a CPU blocked tile
fallback + parity tests. **Defaults unchanged** (`BDH_ROPE_IMPL=eager`).

### Audit (remaining tax)

| Path | Tax found |
|------|-----------|
| T=1 `rope_rotate_t1` | `stack→reshape` when `out=None`; per-step flat→pair cis reshape |
| `_rope_table_pairs` | Built in `ensure_rope_table` but **unused** on apply |
| Fused T>1 pytorch | Needless `cos.expand(v.shape)` before pair reshape; `stack` alloc |
| Triton host | Always `expand+contiguous` cis; CPU fallback was pytorch (no tile scaffold) |

### What changed

| Piece | Change |
|-------|--------|
| `kernels/rope.py` | `_store_pairs` (no stack); `rope_rotate_paired`; fused pytorch **no expand**; `fused_rope_rotate_blocked` tile scaffold; Triton CPU → blocked; lighter cis staging helper |
| `kernels/rope_dispatch.py` | Export paired/blocked; document fallback chain |
| `bdh.py` | `_rope_t1_cis_pairs` + `t1_cis_pairs()`; `Attention.forward` T=1 uses paired cis when generate table is warm |
| `tests/test_rope_decode.py` / `test_rope_fuse.py` | Paired ≡ strided; blocked ≡ eager/fused; Triton CPU→blocked; generate cache continuity under eager+fused |

```bash
export BDH_ROPE_IMPL=eager   # default — T=1 uses paired table path when warmed
export BDH_ROPE_IMPL=fused   # pytorch pair path (no expand/stack); Triton on CUDA
```

### Correctness (this box, CPU)

```text
.venv/bin/python -m pytest tests/test_rope_decode.py tests/test_rope_fuse.py \
  tests/test_rope_cache.py tests/test_vs_baseline.py tests/test_correctness.py \
  tests/test_attention_mask.py -q
# 48 passed, 1 skipped — paired/blocked ≡ eager ≡ baseline; generate tokens match
```

### Honest CPU microbench (no GPU wins claimed)

```text
device=cpu  B=4 H=4 N=256  torch=2.14.0+cu130 cuda=False
T=1 strided out=  median: ~26 us
T=1 t1      out=  median: ~33 us  (≈0.79× — pair path still no CPU wall win)
T=1 paired  out=  median: ~35 us  (≈0.74× — structural; skips cis reshape)
T=128 eager_rotate   median: ~100 ms
T=128 fused_pytorch  median: ~71 ms   (≈1.41× vs eager — no-expand/no-stack)
T=128 blocked_tile   median: ~143 ms  (≈0.70× — scaffold only, not a win claim)
```

**Verdict:** keep **default eager**. Opt-in `BDH_ROPE_IMPL=fused` shows a real
CPU rotate win on cold T>1 after dropping expand/stack; T=1 paired is structural
(for CUDA / fewer reshape nodes). **No GPU on this box** — Triton still
unexecuted; blocked is the CPU tile parity fallback.

### Non-goals

- No PRs to `pathwaycom/*`; no public PR
- No change to default `BDH_ROPE_IMPL=eager`
- No fake GPU speedups from CPU medians
- No attention math / `tril(-1)` / CacheManager changes

## opt/prefetch-h2d — CUDA side-stream staging scaffold (2026-09-19)

**Branch:** `opt/prefetch-h2d` (private `katulevskiy/bdh-gpu-opt` only).
**Base tip:** `fd6d62d` (`origin/main`, profile-v8 / #89).

`BatchPrefetcher` now keeps one CUDA batch staged ahead when async host
prefetching is enabled: producer batches are pinned, `x`/`y` H2D copies are
queued on a dedicated side stream, and the caller stream waits on an event only
when consuming that batch. `record_stream` keeps pinned host storage safe for
the asynchronous copy. CPU remains a no-op for the CUDA path and keeps the
existing host-thread behavior.

**Env:** `BDH_PREFETCH_H2D=1` (default; CUDA side-stream lookahead),
`BDH_PREFETCH_H2D=0` (host prefetch retained, H2D on caller stream),
`BDH_PREFETCH_ASYNC=0` (synchronous debug/A-B). The constructor's
`cuda_staging=` keyword overrides the H2D flag for tests/A-B and is ignored on
CPU.

**Validation (this CPU-only box):** `python -m pytest -q tests/test_dataloader.py`
→ 14 passed, 1 skipped (CUDA-specific); `python benchmarks/bench_prefetch.py`
→ gather 0.069 ms, synthetic overlap 1.56×, tiny train loop sync 1.999 ms vs
async 2.180 ms. These are CPU smoke/noise measurements; GPU H2D overlap and
throughput remain unmeasured.

## opt/blocked-tile-v2 — flatten long CPU cold heads (2026-09-19)

**Branch:** `opt/blocked-tile-v2` (private `katulevskiy/bdh-gpu-opt` only).
**Base tip:** `7ff9b15` (`main` / prefetch-h2d; rebased before PR).

### Goal and audit

The existing blocked/online cold path already avoided a full `T×T` score matrix,
but its long-CPU critical path still used 4-D `matmul` for every past/diagonal
tile. With broadcast `V=(B,1,T,D)`, each tile paid the head-broadcast iterator
again; the score mask also allocated a second tensor.

### What changed

- `kernels/attention.py`: for CPU `T >= 256`, flatten `(B,H)` once and run
  dense `torch.bmm` for score and score×V tiles; broadcast V is expanded once
  for this staging path.
- Chunked past accumulation uses in-place `add_`; diagonal scores use
  `tril_(-1)` before the fused score×V bmm.
- `T < 256` and CUDA retain the existing device-generic path; `BDH_ATTN_IMPL`,
  `BDH_ATTN_AUTO`, and default eager behavior are unchanged.

Strict `tril(diagonal=-1)` raw-score semantics remain unchanged; position 0 is
zero and no softmax/scale/SDPA was introduced.

### Correctness and honest CPU bench

```text
python -m pytest tests/test_prefill_blocked.py tests/test_fuse_scorev.py \
  tests/test_attn_mem.py tests/test_attn_auto.py -q
# 71 passed

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python benchmarks/bench_blocked_vec.py
# torch 2.14.0+cu130, cuda=False, B=1 H=2 N=32 D=64
# T=32:   eager 0.016 ms, blocked 0.029 ms (0.56x)
# T=64:   eager 0.029 ms, blocked 0.042 ms (0.69x)
# T=128:  eager 0.066 ms, blocked 0.119 ms (0.56x)
# T=256:  eager 0.216 ms, blocked 0.174 ms (1.25x)
# T=512:  eager 1.747 ms, blocked 0.437 ms (4.00x)
# T=1024: eager 8.183 ms, blocked 1.395 ms (5.87x)
```

This is a CPU-only, single-thread microbench: blocked/online still loses on
short T, and no GPU speedup is claimed. Default `BDH_ATTN_IMPL` remains eager.

### Non-goals

- No public or `pathwaycom/*` PRs
- No default `BDH_ATTN_IMPL` / `BDH_ATTN_AUTO` change
- No full score materialization, softmax, scale, or diagonal inclusion

## opt/copy-tax-v1 — cut forward QR GEMM copy_ tax (2026-09-19)

**Branch:** `opt/copy-tax-v1` (private `katulevskiy/bdh-gpu-opt` only).
**Base tip:** `3cac6d7` (`origin/main`, docs-v17 / #94).

### Audit (profile-v8 shape, default eager train/eval forward)

On the profile harness cfg (`layers=4 d=128 nh=4 B=4 T=128`), one forward had
**24× `aten::copy_`**. Shape attribution:

| Shape | Count | Source |
|-------|------:|--------|
| `(B,H,T,N/2)` pair stores | 8 | RoPE write into QR (unavoidable) |
| `(B,H,T,N)` + `(B,H,N,T)` | 8 | **QR / QR.mT contig clones** before score GEMM |
| `(B,H,T,T)` or `(B,H,T,D)` | 4 | broadcast `scores @ V` materializes expanded V |
| misc | 4 | einsum / other |

Root cause of the 8 GEMM clones: train feeds RoPE a **non-contiguous**
`(B,nh,T,N)` permute view of contiguous `(B,T,nh,N)`.
`torch.empty_like(v)` preserves that layout (`preserve_format`), so QR is
non-contiguous and BLAS clones both QR and `QR.mT`.

`eager_tril_attn` also used out-of-place `scores.tril(...)` despite docs /
OPT_NOTES claiming in-place `tril_`.

### Changes (defaults unchanged)

1. **`kernels/rope.py`**: `_alloc_rope_out(v)` → `torch.empty(shape,…)`
   (contiguous). Used by `_store_pairs`, eager multi-T, fused blocked, and
   Triton host alloc when `out is None`. Caller `out=` (CacheManager KR slot)
   unchanged.
2. **`kernels/attention.py`**: `eager_tril_attn` and blocked diagonal tile use
   `scores.tril_(diagonal=-1)` on the fresh matmul buffer.
3. **`tests/test_copy_tax.py`**: contiguous QR from permute view; no QR/KT
   contig copies in forward profile; logits/grads @ dropout=0 vs baseline;
   generate token parity.

Left on the table (not safe / not bit-identical grads): replace
`scores @ V` broadcast with `einsum("bhij,bjd->bhid", …)` — drops V expand
`copy_` but gV differs by ~1e-4 association.

### Measured (CPU-only, honest)

| Metric | Before | After |
|--------|-------:|------:|
| Forward `aten::copy_` (1 run, profile cfg) | **24** | **18** |
| QR `(B,H,T,N)` / `(B,H,N,T)` contig copies | 8 | **0** |
| RoPE+score GEMM clone/copy micro (permute v) | 9 | 2 (pair stores only) |

No GPU claim. Wall % will still show RoPE stores + V broadcast + GEMM; this
cuts the **avoidable** QR layout tax. `aten::contiguous` remains 0; generate
`aten::cat` remains 0.

### Non-goals

- No PRs to `pathwaycom/*`; no public PR
- No default `BDH_ATTN_IMPL` / `BDH_ROPE_IMPL` change
- No softmax / scale / SDPA / diagonal inclusion
- No fake GPU wins from CPU copy_ counts


## opt/attn-bwd-gpu-scaffold — CUDA analytic-train harness (2026-09-19)

**Branch:** `opt/attn-bwd-gpu-scaffold` (private `katulevskiy/bdh-gpu-opt` only).
**Base tip:** `8e7a4d2` (`main`, copy-tax-v1 / #95).
**Base tip:** `3cac6d7` (`main`, docs-v17 / #94).

### Audit and scaffold

- `bdh.Attention.forward` already routes cold and multi-token-with-past train
  work through `bdh_attn` when `BDH_ATTN_AUTOGRAD=1`; T=1 decode remains the
  no-grad CacheManager path. No default train wiring change was needed.
- `benchmarks/bench_attn_bwd.py` now gates on `torch.cuda.is_available()` before
  allocation and exits 0 with `SKIP` on CPU. On CUDA it runs the full
  `IMPL=eager|blocked|triton|cuda` × `AUTOGRAD=0|1` matrix, reports effective
  fallbacks, and parity-checks loss and every parameter gradient before timing.
- `--dtype` / `BDH_ATTN_BWD_DTYPE` (float32 default; bf16/fp16 supported),
  batch/tokens, warmup, and repetitions are explicit. Default B=4/T=64 remains
  a smoke case; use B=8/T=256+ for occupancy and T>=512 for memory studies.
  Native Triton/CUDA forward paths without an autograd graph are reported as
  skipped for AUTOGRAD=0, never as speed wins.

### Validation

```text
python benchmarks/bench_attn_bwd.py
# SKIP: CUDA unavailable; bench_attn_bwd requires a real CUDA device
# exit 0
python -m pytest tests/test_attn_bwd.py tests/test_attn_bwd_bench.py -q
# CPU parity matrix passes; CUDA-only harness skip contract passes
```

No GPU timing or speedup is claimed from this CPU-only box. Defaults remain
`BDH_ATTN_IMPL=eager`, `BDH_ATTN_AUTOGRAD=0`, and no attention math changes.

## opt/attn-bwd-v2 — keep blocked/online train spellings in the GPU matrix (2026-09-19)

**Branch:** `opt/attn-bwd-v2` (private `katulevskiy/bdh-gpu-opt` only).
**Base tip:** `45b4afe` (`main`).

The #96 harness matrix now includes the documented `IMPL=online` alias next
to `IMPL=blocked`, each crossed with `AUTOGRAD=0|1`. This is a runbook-level
parity guard: `online` resolves to the same tiled backend, so it must not be
omitted from GPU train-step comparisons. The CPU contract test checks the
matrix and the CUDA-only benchmark still exits cleanly with `SKIP` before
allocation when CUDA is unavailable.

```text
python benchmarks/bench_attn_bwd.py
# CPU: SKIP: CUDA unavailable; ... (exit 0)
python -m pytest tests/test_attn_bwd.py tests/test_attn_bwd_bench.py -q
```

No GPU timing, speedup, or kernel claim is added. Defaults remain eager with
`BDH_ATTN_AUTOGRAD=0`; attention remains raw scores × `tril(diagonal=-1)`.

## opt/profile-v9 — re-profile copy-tax-v1 tip (2026-09-19)

**Branch:** `opt/profile-v9` (private `katulevskiy/bdh-gpu-opt` only).
**Profile source:** `8e7a4d2` (`#95` copy-tax-v1); the branch is rebased onto
`06e25d2` (`#97` docs-v18, including #96's CUDA scaffold) before publication.
The profile window itself was captured at the requested #95 tip, before that
rebase; #96/#97 do not change the default short eval/generate path.

### Method

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python \
  benchmarks/profile_forward.py --mode all
# torch 2.14.0+cu130  cuda=False  device=cpu
# cfg: layers=4 d=128 nh=4 B=4 T=128; generate prompt=16 / new=32
```

The harness uses two warmups, one wait, and three active steps. Absolute CPU
milliseconds are profiler-inflated; the table reports self-CPU percentages and
aggregate calls over the three active steps. Chrome traces remain under
`benchmarks/traces/` (gitignored).

### CPU profile highlights (self CPU)

| Mode | Top self-CPU operators (calls) | Notes |
|------|--------------------------------|-------|
| Attention | `aten::mul` **30.78% (12)**, `aten::bmm` **30.50% (6)**, `aten::copy_` **19.45% (9)**, `aten::sub` **8.56% (3)**, `aten::add` **3.16% (3)**, `aten::tril_` **0.32% (3)** | Default eager still computes raw scores then applies strict `tril_(diagonal=-1)`; no softmax, scale, or SDPA. |
| Forward | `aten::bmm` **34.12% (36)**, `aten::mm` **24.93% (27)**, `aten::mul` **17.69% (60)**, `aten::copy_` **9.09% (48)**, `aten::clamp_min_` **3.24% (24)**, `aten::sub` **2.56% (12)** | `aten::cat=0`, `aten::contiguous=0`; warmed harness active window is **48 copy_ / 3 = 16 per active call**. |
| Generate | `aten::mm` **22.16% (795)**, `aten::bmm` **14.72% (1,188)**, `aten::mul` **3.00% (1,980)**, `aten::matmul` **2.45% (1,587)**, `aten::native_layer_norm` **2.36% (1,287)**, `aten::einsum` **2.02% (396)**, `aten::copy_` **1.14% (1,674)** | `aten::cat=0`, `aten::contiguous=0`; copy_ is **558 per active generate call**, and the preallocated cache path remains cat-free. |

A separate one-forward CPU profiler check at the same cfg reports
`aten::copy_=18`, `aten::cat=0`, and `aten::contiguous=0`, reproducing the
#95 single-run claim of **24 → 18**. The default harness's warmed active
window reports 16 per call because its counts are taken after the two warmups
and one wait; both are CPU operator counts, not GPU performance evidence.

### Smoke and verdict

```text
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m pytest \
  tests/test_copy_tax.py -q
# 8 passed
```

Defaults are unchanged (`BDH_ATTN_IMPL=eager`, `BDH_ROPE_IMPL=eager`), and
attention remains raw scores × strict lower-triangular `tril(-1)`. This box has
no GPU, so no GPU timing, kernel win, or speedup is claimed. The main remaining
question is real-GPU validation of fused score×V / RoPE / layout effects.

### Non-goals

- No PRs to `pathwaycom/*`; private repo only; no public PR
- No default attention/RoPE implementation change
- No softmax / scale / SDPA / diagonal inclusion
- No GPU claims from CPU profiler percentages or copy_ counts


## opt/scorev-fuse-v2 — deepen CPU blocked/online score×V (2026-09-19)

**Branch:** `opt/scorev-fuse-v2` (private `katulevskiy/bdh-gpu-opt` only).
**Base:** `82d5697` (`main`, docs-v19 / #99).

### Audit and deepen

The long CPU cold path already flattened `(B,H)` into one dense `bmm` batch,
but each score tile still held a score tensor plus a separate `bmm(scores,V)`
result until the add/copy epilogue. The remaining Python loops are the query
tile loop, chunked-past loop when the score budget is exceeded, and the
T=1 decode past-tile loop; they are required to keep peak score storage below
full `T×T` / `Tq×S`. The generic short/device path retains its broadcast-V
matmul behavior rather than paying a new expansion.

This revision adds `_score_v_into`: inference/no-grad CPU tiles use
`baddbmm(..., out=target)` so score×V writes directly into the flattened
output tile; autograd uses the graph-safe add/copy fallback. The T=1
head-matched decode path reuses the flattened output view, and the broadcast-V
oneshot budget tightens from 2048 to **1024 score elements**. Strict
`tril(diagonal=-1)` semantics, raw scores×V math, and default eager dispatch
are unchanged.

### Correctness (CPU)

```text
.venv/bin/python -m pytest -q
# 490 passed, 18 skipped (CUDA/native paths), 3 warnings
# blocked/online parity vs eager tril(-1); position 0 == 0; default eager unchanged
```

### Honest CPU microbench (single-thread; no GPU claim)

```text
# cold: B=1 H=2 N=32 D=64, eager / blocked, median ms
T=256: 0.139 / 0.198  (0.70×)
T=512: 0.516 / 0.397  (1.30×)
T=1024: 2.622 / 1.756  (1.49×)

# decode: B=1 H=2 Tq=1 N=32 D=64, eager / blocked, median ms
S=512:  0.020 / 0.022  (0.94×), peak 512
S=1024: 0.021 / 0.023  (0.90×), peak 1024
S=1536: 0.026 / 0.129  (0.20×), peak 256
S=4096: 0.073 / 0.134  (0.55×), peak 1024
```

CPU blocked is still slower at some short/mid shapes; the structural win is
lower score peak and fewer inference epilogue buffers. These measurements are
not GPU evidence. Re-measure Triton/CUDA on real hardware before claiming a
kernel or speedup win.

### Non-goals

- No default `BDH_ATTN_IMPL` change; eager remains the default.
- No softmax, scale, SDPA, diagonal inclusion, or full score materialization.
- No GPU claims from CPU timings.

## opt/gen-copy-tax-v1 — cut generate RoPE + sample copy_ (2026-09-19)

**Branch:** `opt/gen-copy-tax-v1` (private `katulevskiy/bdh-gpu-opt` only).
**Base tip:** `a0dd4db` (main after #101 docs-v20 / #100 scorev-fuse-v2 / #98 profile-v9).

### Goal

Cut generate-path `aten::copy_` tax from profile-v9 (~1674 / **558** per active
generate). Keep `tril(diagonal=-1)`, **cat-free** generate, defaults unchanged.
No fake GPU wins.

### Audit (CPU, cfg layers=4 d=128 nh=4, prompt=16 / +32)

| Bucket | ~copy_/gen | Notes |
|--------|------------|-------|
| RoPE `_store_pairs` into packed KR | 256 | 2× pair setitem / layer-step |
| V cache slot writes | 128 | necessary |
| multinomial `_to_copy` / `any` | ~128 | sampler internals |
| token `out[:, i] =` | 32 | assign after multinomial |
| prefill + prompt | ~14 | |

Rejected: `torch.stack→reshape` halves RoPE `copy_` but **`aten::stack` cats
under the hood** — violates generate `cat=0`.

### What landed

| Piece | Detail |
|-------|--------|
| `kernels/rope.py` `_store_pairs` | fp32/fp64: `view_as_complex(out).copy_(complex(y0,y1))` — **1** `copy_`, **0** `cat`, bit-identical to dual setitem; other dtypes keep setitem |
| `BDH._sample_from_logits(..., idx_out=)` | multinomial / top-k gather write into optional `(B,1)` buffer |
| `BDH.generate` | `idx_next = out.narrow(1, prompt+t, 1)` passed as `idx_out` — drops per-step token assign `copy_` |
| Defaults | unchanged |
| `tests/test_gen_copy_tax.py` | store parity, 1 `copy_`/store, sample RNG parity, generate≡baseline, cache continuity, profiler gate |

### Profile (this box, CPU)

| Metric | Before (profile-v9) | After |
|--------|---------------------|-------|
| `aten::copy_` / generate | **~558** | **~398** |
| `aten::cat` | 0 | **0** |

Breakdown of the cut: RoPE store 256→128 + token assign 32→0 ≈ **-160**/gen.
Remaining: V writes + sampler internals (not safely removable without RNG /
layout changes).

### Correctness

```bash
.venv/bin/python -m pytest tests/test_gen_copy_tax.py tests/test_rope_decode.py \
  tests/test_rope_fuse.py tests/test_cache_pack.py tests/test_copy_tax.py \
  tests/test_gen_host.py -q
```

### Non-goals

- Softmax / diagonal / SDPA
- PRs to `pathwaycom/*`
- Claiming GPU speedups from CPU copy_ counts
- Changing default sampling distribution / RNG (only the write destination)

## opt/profile-v10 — re-profile gen-copy-tax-v1 tip (2026-09-19)

**Branch:** `opt/profile-v10` (private `katulevskiy/bdh-gpu-opt` only).
**Profile source:** `1363794` (`#102` gen-copy-tax-v1); this branch is rebased
onto `234de2a` (`#103` docs-v21) before publication. The rebase is docs-only
and does not change the default eval/generate path.

### Method

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python \
  benchmarks/profile_forward.py --mode all
# torch 2.14.0+cu130  cuda=False  device=cpu
# cfg: layers=4 d=128 nh=4 B=4 T=128; generate prompt=16 / new=32
```

The harness uses two warmups, one wait, and three active steps. Absolute CPU
milliseconds are profiler-inflated; the table reports self-CPU percentages and
aggregate calls over the three active steps. Chrome traces remain under
`benchmarks/traces/` (gitignored).

### CPU profile highlights (self CPU)

| Mode | Top self-CPU operators (calls) | Notes |
|------|--------------------------------|-------|
| Attention | `aten::bmm` **30.87% (6)**, `aten::mul` **30.19% (12)**, `aten::copy_` **19.04% (9)**, `aten::sub` **8.28% (3)**, `aten::add` **2.96% (3)**, `aten::tril_` **0.31% (3)** | Default eager still computes raw scores then applies strict `tril_(diagonal=-1)`; no softmax, scale, or SDPA. |
| Forward | `aten::bmm` **36.00% (36)**, `aten::mul` **21.86% (60)**, `aten::mm` **19.15% (27)**, `aten::copy_` **11.27% (48)**, `aten::clamp_min_` **3.00% (24)**, `aten::sub` **2.45% (12)** | `aten::cat=0`, `aten::contiguous=0`; warmed harness active window is **48 copy_ / 3 = 16 per active call**. |
| Generate | `aten::mm` **14.76% (795)**, `aten::bmm` **12.80% (1,188)**, `aten::matmul` **7.33% (1,587)**, `aten::mul` **4.70% (1,980)**, `aten::native_layer_norm` **2.55% (1,287)**, `aten::einsum` **2.54% (396)**, `aten::copy_` **0.50% (1,194)** | `aten::cat=0`, `aten::contiguous=0`; generate `copy_` is **398 per active call**, down from profile-v9's 558, while the preallocated cache path remains cat-free. |

A separate one-forward CPU profiler check at the same cfg reports
`aten::copy_=18`, `aten::cat=0`, and `aten::contiguous=0`. A standalone
one-shot generate check reports `aten::copy_=400`, while the scheduled harness
reports **1,194 / 3 = 398 per active call**; the latter is the comparable
profile-v10 number. These are CPU operator counts, not GPU performance
measurements.

### Smoke and verdict

```text
.venv/bin/python -m pytest tests/test_gen_copy_tax.py tests/test_rope_decode.py \
  tests/test_rope_fuse.py tests/test_cache_pack.py tests/test_copy_tax.py \
  tests/test_gen_host.py -q
# 74 passed, 1 skipped
```

The tip confirms the #102 generate copy-tax cut (~558 → ~398 `aten::copy_` per
active generate call) with `aten::cat=0` and `aten::contiguous=0`; forward's
isolated `copy_` remains 18. Defaults are unchanged (`BDH_ATTN_IMPL=eager`,
`BDH_ROPE_IMPL=eager`), and attention remains raw scores × strict
lower-triangular `tril(-1)`. This box has no GPU, so no GPU timing, kernel win,
or speedup is claimed.

## opt/ln-resid-v2 — residual/LN audit probe (2026-09-19)

**Branch:** `opt/ln-resid-v2` (private `katulevskiy/bdh-gpu-opt` only).
**Base tip:** `6f91417` (`main`, #104 profile-v10).

### Audit verdict

This is a **probe-only** revision: the tip already contains the safe #51
residual deepen (`LN(yMLP)` → `y.add_(x)` → `LN(y)`). On CPU with torch
2.14.0+cu130, the residual-only profile is exactly **2×
`aten::native_layer_norm` + 1× `aten::add_`**, with no `aten::add`,
`aten::copy_`, `aten::cat`, `aten::to`, or `aten::_to_copy`. The model inputs
remain unmodified.

The remaining two LayerNorm calls each necessarily return a fresh normalized
output; the native schema is:

```text
aten::native_layer_norm(Tensor, SymInt[], Tensor?, Tensor?, float)
  -> (Tensor, Tensor, Tensor)
```

There is no eager affine-free fused add+LayerNorm or out-buffer LayerNorm API
in this torch build. A manual in-place LayerNorm would change accumulation /
gradient behavior and is not a safe default. The fp32, fp16, and bf16 probes
show no explicit dtype bounce on the existing path.

### What landed

- `tests/test_ln_resid_probe.py` locks the operator-count and dtype audit.
- No model/default/attention behavior changed; no fake optimization was landed.
- Existing `F.layer_norm` + inner-output reuse remains the recommended path
  until a validated fused kernel or newer safe API is available.

### Correctness / limits

```bash
python -m pytest tests/test_ln_resid_probe.py tests/test_compile.py \
  tests/test_vs_baseline.py -q
```

The box is CPU-only (`cuda=False`), so this records no GPU claim.

### Non-goals

- No PRs to `pathwaycom/*`; private repo only; no public PR
- No default attention/RoPE implementation change
- No softmax / scale / SDPA / diagonal inclusion
- No GPU claims from CPU profiler percentages or copy_ counts
- No RMSNorm / affine-LN semantic change
- No attention math or default change
- No GPU claims from CPU operator counts

## opt/triton-cold-v3 — adaptive wide-head tiles and copy-free CPU fallback (2026-09-19)

**Branch:** `opt/triton-cold-v3` (private `katulevskiy/bdh-gpu-opt` only).
**Base tip:** `7c27817`; pair with merged `#82` Triton cold-v2 and `#93`
blocked-tile-v2. The branch is rebased onto the current `main` before PR.

### Audit / deepen

- Cold Triton tile selection still grows 64→128 at `T≥256` for the normal
  BDH head shape (`N≤64`, `D≤128`), but keeps wide-head query/key tiles at
  64×64 to bound qk/accumulator register pressure instead of applying the
  long-T width blindly. Explicit `block_m`/`block_n` overrides remain honored.
- The long CPU fallback from `#93` still flattens `(B,H)` for score `bmm`,
  while shared `V=(B,1,T,D)` now remains a single `(B,T,D)` view during each
  score×V tile; it no longer materializes a `(B*H,T,D)` V staging copy.
- Triton CUDA launch/scaffold behavior remains unchanged except for the
  dimension-aware picker; CPU-only fallback remains the exercised validation.

### Correctness / limits

```text
/workspace/bdh-gpu-opt/.venv/bin/python -m pytest \
  tests/test_triton_attn.py tests/test_prefill_blocked.py -q
# 45 passed, 3 skipped (CPU; CUDA+Triton soft-skipped)
```

The tests cover eager parity, strict `tril(diagonal=-1)` / `pos0==0`, long CPU
→ blocked fallback, broadcast and per-head V, adaptive tile thresholds, and
explicit tile overrides. This box has no usable GPU; no GPU timing, kernel
validation, or speedup claim is made. Defaults remain eager and AUTO remains
off.

### Non-goals

- No public or `pathwaycom/*` PRs; private repo only.
- No default `BDH_ATTN_IMPL` / `BDH_ATTN_AUTO` change.
- No softmax, scale, diagonal inclusion, or full-score materialization.

## opt/triton-cold-v4 — actionable CPU-safe Triton skip diagnostics (2026-09-19)

**Branch:** `opt/triton-cold-v4` (private `katulevskiy/bdh-gpu-opt` only).
**Base tip:** `c557b55` (`main`, after #148 docs through #147).

### Audit / deepen

- Added `triton_cold_skip_reason()` to report the exact import/device gate
  without allocating a CUDA tensor or attempting a kernel launch.
- Triton cold CUDA tests now use that diagnostic in their `skipif` reason,
  distinguishing `Triton unavailable: import failed (...)` from
  `CUDA unavailable: torch.cuda.is_available() is false`.
- Added a CPU-safe unit test for the diagnostic; cold fallback, raw scores ×
  strict `tril(diagonal=-1)`, eager default, and AUTO-off behavior are unchanged.

### Validation (CPU-only)

```text
/workspace/bdh-gpu-opt/.venv/bin/python -m pytest tests/test_triton_attn.py -q -rs
# 36 passed, 3 skipped (CUDA unavailable: torch.cuda.is_available() is false)
```

No GPU timing, compilation, kernel validation, correctness, speedup, or other
GPU claim is made. Defaults remain eager and `BDH_ATTN_AUTO` remains off.

### Non-goals

- No public or `pathwaycom/*` PRs; private repo only.
- No default `BDH_ATTN_IMPL` / `BDH_ATTN_AUTO` change.
- No softmax, scale, diagonal inclusion, or full-score materialization.

## opt/gen-vcopy-v1 — V/sampler probe + T>1 RoPE pair store (2026-09-19)

**Branch:** `opt/gen-vcopy-v1` (private `katulevskiy/bdh-gpu-opt` only).
**Base tip:** `f430b53` (`main`, #109 docs-v23; includes #108).

### Goal

After #102 (~558→~398 generate `aten::copy_`), attribute the remainder and cut
anything safe from V-cache writes / sampler internals. Keep `tril(diagonal=-1)`,
`aten::cat=0`, defaults unchanged. No GPU claims.

### Attribution (CPU, cfg layers=4 d=128 nh=4, prompt=16 / +32)

One generate call (~398 before this PR):

| Bucket | ~copy_/gen | Notes |
|--------|------------|-------|
| V slot `dst_v.copy_(v_tok)` | 132 | 4×(1 prefill + 32 decode) — necessary x→cache snapshot; residual updates `x` so aliasing is unsafe |
| RoPE decode `_store_pairs` | 128 | 4×32 — already 1/call after #102 |
| Prefill RoPE T>1 strided setitem | 8 | 4×2 — **cut by this PR** (→ 4 via `_store_pairs`) |
| multinomial internals | 128 | 4 `copy_`/sample × 32 — ATen C++; `_to_copy`×3 + `any`×1; RNG-coupled |
| prompt `out[:, :T].copy_` | 1 | |
| **Total** | **~397** | matches profile-v10 harness |

**Not safely removable (defaults / RNG / layout):**

- **V writes:** each layer must snapshot `x` into packed `(B,1,T,D)` before
  residual LN replaces `x`. Squeeze-view `copy_` is still one `copy_`. Layout
  change to drop the singleton head dim does not remove the snapshot.
- **Default sampler:** `torch.multinomial(..., out=idx_out)` still emits ~4
  `aten::copy_` + 3 `_to_copy` + 1 `any` per step inside ATen. Custom
  Gumbel-max / Categorical would change the RNG stream vs baseline.

### What landed (1 safe reduce + small top-k cleanup)

| Piece | Detail |
|-------|--------|
| `kernels/rope.py` `eager_rope_rotate` | T>1 uses pair reshape + `_store_pairs` (same as T=1/fused) — bit-identical to historical strided setitem; **1** fp32 `copy_`, **0** `cat` |
| `BDH._sample_from_logits` top-k | `torch.gather(..., out=idx_out)` when set — drops gather+`copy_` (non-default `top_k` only) |
| Defaults | unchanged (`BDH_ROPE_IMPL=eager`, `top_k=None`) |
| Tests | T>1 single-copy + strided parity; top-k gather parity; attribution bucket sum; generate/cache gates |

### Profile (this box, CPU)

| Metric | profile-v10 (#102 tip) | After gen-vcopy-v1 |
|--------|------------------------|--------------------|
| `aten::copy_` / generate | **~398** | **~394** |
| `aten::cat` | 0 | **0** |

Delta ≈ **−4**/gen from prefill RoPE 8→4. V (132) + multinomial (128) remain.

### Correctness

```bash
.venv/bin/python -m pytest tests/test_gen_copy_tax.py tests/test_cache_pack.py \
  tests/test_rope_decode.py tests/test_rope_fuse.py tests/test_copy_tax.py \
  tests/test_gen_sample.py tests/test_gen_host.py -q
```

### Non-goals

- Softmax / diagonal / SDPA
- Changing default multinomial RNG or V cache layout
- PRs to `pathwaycom/*`
- Claiming GPU speedups from CPU copy_ counts

## opt/profile-v11 — re-profile gen-vcopy-v1 tip (2026-09-19)

**Branch:** `opt/profile-v11` (private `katulevskiy/bdh-gpu-opt` only).
**Profile source:** `4963b0f` (`#110` gen-vcopy-v1); this docs-only branch is
rebased onto `acf6e09` (`#111` docs-v24). The rebase does not change the
default eval/generate path.

### Method

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python \
  benchmarks/profile_forward.py --mode all
# torch 2.14.0+cu130  cuda=False  device=cpu
# cfg: layers=4 d=128 nh=4 B=4 T=128; generate prompt=16 / new=32
```

The harness uses two warmups, one wait, and three active steps. Absolute CPU
milliseconds are profiler-inflated; the table reports self-CPU percentages and
aggregate calls over the three active steps. Chrome traces remain under
`benchmarks/traces/` (gitignored).

### CPU profile highlights (self CPU)

| Mode | Top self-CPU operators (calls) | Notes |
|------|--------------------------------|-------|
| Attention | `aten::mul` **25.29% (12)**, `aten::bmm` **22.61% (6)**, `aten::complex` **16.19% (6)**, `aten::copy_` **14.68% (6)**, `aten::add` **6.92% (3)**, `aten::sub` **6.60% (3)** | Default eager still computes raw scores then applies strict `tril_(diagonal=-1)`; no softmax, scale, or SDPA. |
| Forward | `aten::bmm` **28.69% (36)**, `aten::mm` **24.63% (27)**, `aten::mul` **16.40% (60)**, `aten::complex` **10.23% (24)**, `aten::copy_` **6.65% (36)**, `aten::clamp_min_` **3.06% (24)** | `aten::cat=0`, `aten::contiguous=0`; warmed harness is **36 copy_ / 3 = 12 per active call**. |
| Generate | `aten::mm` **21.92% (795)**, `aten::bmm` **13.60% (1,188)**, `aten::mul` **3.03% (1,980)**, `aten::native_layer_norm` **2.29% (1,287)**, `aten::einsum` **2.07% (396)**, `aten::copy_` **0.87% (1,182)** | `aten::cat=0`, `aten::contiguous=0`; generate `copy_` is **1,182 / 3 = 394 per active call**, down from profile-v10's 398. |

Separate one-shot CPU profiler checks at the same cfg report forward
`aten::copy_=14`, generate `aten::copy_=396`, and `aten::cat=0` /
`aten::contiguous=0` in both modes. The isolated forward count is lower than
the profile-v10 ~18 because #110 changes the T>1 RoPE store to `_store_pairs`;
the scheduled harness count is the comparable number for this run. These are
CPU operator counts, not GPU performance measurements.

### Smoke and verdict

```text
.venv/bin/python -m pytest tests/test_gen_copy_tax.py tests/test_cache_pack.py \
  tests/test_rope_decode.py tests/test_rope_fuse.py tests/test_copy_tax.py \
  tests/test_gen_sample.py tests/test_gen_host.py -q
# 84 passed, 1 skipped
```

The tip confirms the #110 V/sampler probe: generate `aten::copy_` is ~394 per
active call, while V-slot writes (~132/gen) and multinomial internals (~128/gen)
remain coupled to cache-snapshot layout and the default RNG stream. Forward and
generate remain cat-free and contiguous-free. Defaults are unchanged
(`BDH_ATTN_IMPL=eager`, `BDH_ROPE_IMPL=eager`), and attention remains raw scores
× strict lower-triangular `tril(-1)`. This box has no GPU, so no GPU timing,
kernel win, or speedup is claimed.

### Non-goals

- No PRs to `pathwaycom/*`; private repo only; no public PR
- No default attention/RoPE implementation change
- No softmax / scale / SDPA / diagonal inclusion
- No GPU claims from CPU profiler percentages or copy_ counts

## opt/cuda-cold-v3 — bounded wide-head CUDA tiles (2026-09-19)

**Branch:** `opt/cuda-cold-v3` (private `katulevskiy/bdh-gpu-opt` only).
**Base tip:** `0f7af06` (`main`, #113 docs-v25); pair with merged #79
`opt/cuda-cold-v2` and #108 `opt/triton-cold-v3`.

### Audit / deepen

- The CUDA cold kernel already stages only the active `K`/`V` tiles in shared
  memory. Broadcast `V=(B,1,T,Dv)` selects `hv=0` per head; it does not
  materialize a `(B*H,T,Dv)` copy. The CPU tiled reference likewise keeps the
  broadcast value tensor unexpanded.
- The missing #108 parity was the dimension-aware long-T tile guard. The
  picker now treats `Dk > 64` or `Dv > 128` as a wide head: default long-T
  tiles stay at 32×32 for CUDA shared memory and 64×64 for the CPU mirror,
  while normal heads retain 32×64 CUDA / 128×128 CPU preferences. Explicit
  tile overrides remain authoritative.
- C++ and Python pickers receive the value width so the CUDA launch and CPU
  reference make the same wide-head decision. Strict raw-score × strict
  `tril(diagonal=-1)` semantics and default eager dispatch are unchanged.

### Smoke / correctness

```text
/workspace/bdh-gpu-opt/.venv/bin/python -m pytest \
  tests/test_cuda_attn.py tests/test_cuda_decode.py -q
# 37 passed, 9 skipped (CUDA unavailable)

BDH_BUILD_EXT=1 BDH_FORCE_CPU_EXT=1 \
  /workspace/bdh-gpu-opt/.venv/bin/python setup.py build_ext --inplace
# succeeded with the PyTorch 2.14 C++20 headers
```

The full CPU suite and the native CUDA tests are the final checks for this
branch. This box has no CUDA device, so CUDA tests soft-skip and no GPU timing,
kernel correctness on hardware, or speedup claim is made.

### Non-goals

- No public or `pathwaycom/*` PRs; private repo only.
- No default `BDH_ATTN_IMPL` / `BDH_ATTN_AUTO` change.
- No softmax, scale, diagonal inclusion, or full-score materialization.
- No GPU claims from CPU reference parity or build smoke.


## opt/online-decode-t1-v2 — direct shared-V T=1 decode epilogue (2026-09-19)

**Branch:** `opt/online-decode-t1-v2` (private `katulevskiy/bdh-gpu-opt` only).
**Base:** `f24a5d8` (`opt/cuda-cold-v3` / #114 tip; rebase onto current
`main` / #115 before opening the private PR). This pairs the blocked online
long-S path from #55 with the direct score×V epilogue from #100.

### Audit / deepen

The existing #55 path already tightens broadcast-V T=1 decode to a 1024-score
oneshot budget and tiles longer packed past scans. The #100 `baddbmm` epilogue
already writes head-matched V tiles into the flattened output. The remaining
CacheManager hot path was broadcast `V=(B,1,S,D)`: both its oneshot and its
long-S tile loop still used generic 4-D matmul, retaining a score×V product
buffer after each score tile.

This revision keeps the shared V unexpanded, flattens T=1 Q/K scores over
`B*H`, and accumulates each shared-V tile with per-sample zero-stride V views
and `baddbmm(..., out=target)` in inference/no-grad mode. The output tile is
therefore the epilogue destination for both the oneshot and long-S paths;
autograd uses the existing graph-safe matmul fallback. No score tile exceeds
the #55 budget, and no full `T×S`/`T×T` score is retained on the tiled path.

Strict raw-score × `tril(diagonal=-1)` semantics are unchanged: packed past
keys are all earlier than the T=1 query, and `S=0` returns zeros. `online`
remains the `blocked` alias. `BDH_ATTN_IMPL` remains opt-in; default eager is
unchanged.

### Correctness (CPU-only box; no GPU claims)

```bash
/workspace/bdh-gpu-opt/.venv/bin/python -m pytest \
  tests/test_inc_decode.py tests/test_gen_sample.py -q
# 66 passed, 3 skipped
```

Coverage includes blocked/online parity against eager, exact position-zero
zero output, broadcast-V oneshot and long-S tile parity, autograd fallback,
and blocked `generate()` token parity with eager while `torch.cat` remains 0.

### Non-goals

- No default `BDH_ATTN_IMPL` change; eager remains the default.
- No softmax, scale, SDPA, or diagonal inclusion.
- No V expansion into `(B,H,S,D)` and no full score materialization.
- No GPU timing, kernel claim, or speedup claim from this CPU-only box.
- No public PR and no PRs to `pathwaycom/*`; private repo only.
## opt/compile-train-v2 — backward-aware compile probe (2026-09-19)

**Branch:** `opt/compile-train-v2` (private `katulevskiy/bdh-gpu-opt` only).
**Base tip:** `d193d78` (`origin/main`, #115). No pathwaycom or public PR.

### Audit / deepen

The train path already kept data loading, logging, `generate()`, and optimizer
control outside the compiled module. The remaining gap was probe depth:
`BDH_COMPILE_PROBE=train` warmed only the forward graph, so a compile failure
in the backward portion could still first appear in the real `train_step`.

This revision makes `BDH_COMPILE_PROBE=train_bwd` the opt-in compile default.
It runs the train forward with targets, calls `loss.backward()`, and clears the
probe gradients through `zero_grad(set_to_none=True)` before returning. The
forward-only `train` and `eval` probes remain available for diagnostics. If the
backward probe is selected without `example_y`, `maybe_compile` now returns the
original module with an explicit soft-fallback message instead of returning an
unprobed `OptimizedModule`.

`train_step` documents the boundary explicitly: the compiled module covers
forward/backward, while `optimizer.step()` and gradient clearing remain eager.
This preserves the current optimizer semantics and avoids claiming that the
whole Python train-step function is captured.

### Bench / matrix contract

`benchmarks/bench_train_step.py` now imports with
`BDH_COMPILE_PROBE=train_bwd` unless the operator explicitly sets a probe. Its
compile A/B and FULLGRAPH×eager×AUTOGRAD sections print the effective probe,
soft-skip compile/probe/train-step failures, and retain the honest CPU-only
matrix. The matrix dimensions are:

```text
COMPILE ∈ {0,1}
IMPL ∈ {eager, blocked}
AUTOGRAD ∈ {0,1}
FULLGRAPH ∈ {0,1} × IMPL=eager
MODE ∈ {default, reduce-overhead}
```

A warm single-process smoke on this CPU box (`layers=2 d=64 B=4 T=64`,
`IMPL=eager`, `MODE=default`) reported `COMPILE=0` at 19.08 ms and
`COMPILE=1` with the new `train_bwd` probe at 31.23 ms. This is a local CPU
wall measurement, not a speedup claim; the stronger probe adds intentional
backward compile/startup work.

On this CPU-only box, use only `BDH_COMPILE=1 BDH_ATTN_IMPL=eager
BDH_COMPILE_MODE=default` as the recommended compile combination. CPU wall
medians are not GPU or CUDA-graph claims; `COMPILE=0` remains the repository
default.

### Tests

```bash
.venv/bin/python -m pytest tests/test_compile.py -q
.venv/bin/python -m pytest tests/ -q
```

Coverage adds the backward-probe default contract and verifies that missing
probe targets soft-fall back to eager without requiring inductor. Existing
FULLGRAPH×eager×AUTOGRAD parity/smoke coverage remains unchanged. No GPU was
available for this work, so no GPU compile, CUDA-graph, or speedup claim is made.

### Non-goals

- No default `BDH_COMPILE=1`; `COMPILE=0` remains unchanged.
- No default attention/RoPE/autograd implementation change.
- No attention math change: raw scores × strict `tril(diagonal=-1)` remains.
- No GPU claims and no PRs to `pathwaycom/*` / no public PR.


## opt/amp-train-v2 — harden AMP train configuration and benchmark matrix (2026-09-19)

**Branch:** `opt/amp-train-v2` (private `katulevskiy/bdh-gpu-opt` only).
**Base tip:** `c6c8dfd` (`#118` docs-v27, after `#117`).

### Audit and deepen

The opt-in `BDH_AMP_DTYPE` path now validates CPU autocast with a real matmul
smoke even when a version capability probe exists. Unsupported CPU bf16/fp16
requests raise an actionable error; the benchmark catches that error and records
a visible `soft-skip` instead of aborting the matrix. AMP configuration is
transactional, so a failed opt-in leaves the prior fp32 context intact.

The train benchmark now measures the same matrix for each supported dtype:
`full` autocast (the existing behavior) and `forward_only` (logits autocast,
fp32 CE). It reports scaler state and vs-fp32 ratios, while retaining the CPU
warning that these timings are not GPU throughput evidence.

GradScaler activation is gated by all three conditions: resolved dtype
`float16`, `device.type == "cuda"`, and a live `torch.cuda.is_available()`.
Non-CUDA configurations construct a disabled CPU scaler for a stable inspection
surface; `train_step` additionally requires `scaler.is_enabled()` before using
the scaling path. bf16 and CPU remain unscaled.

### Correctness and CPU-safe validation

```text
pytest -q tests/test_bf16_train.py
# 18 passed

OMP_NUM_THREADS=2 BDH_BENCH_COMPILE=0 BDH_BENCH_COMPILE_BLOCKED=0 \
BDH_BENCH_COMPILE_MODE=0 BDH_BENCH_COMPILE_DROPOUT=0 \
BDH_BENCH_COMPILE_FULLGRAPH=0 BDH_BENCH_AMP=1 \
python benchmarks/bench_train_step.py
```

Observed on this CPU box (device-local medians; **not** GPU claims):

```text
                         median_ms  vs_fp32
float32/full                 13.32     1.00x
bfloat16/full                 9.07     0.68x
bfloat16/forward_only         9.41     0.71x
float16/full                11.04     0.83x
float16/forward_only        10.87     0.82x
amp_claim=none; scaler=False for all CPU rows
```

The numbers are local CPU noise/cast behavior only. There is no GPU in this
runner, so no AMP speedup or Tensor Core claim is made. Defaults remain AMP off
(`BDH_AMP_DTYPE=float32`), compile off, eager, and `tril(diagonal=-1)`.

### Non-goals

- No default AMP enablement or dtype change.
- No attention math, compile, or optimizer-default changes.
- No GPU claims from CPU medians; CUDA validation remains open.
- No PRs to `pathwaycom/*`; private repo only.
## opt/auto-thr-v2 — strict cold/decode gates and A/B harness (2026-09-19)

**Branch:** `opt/auto-thr-v2` (private `katulevskiy/bdh-gpu-opt` only).
**Base tip:** `c6c8dfd` (`main`, #118 docs-v27). No `pathwaycom/*` and no public PR.

### Audit / deepen

The AUTO audit keeps the safety contract from #77 and makes the two length
checks explicit in one helper:

```text
AUTO off:                         eager cold + eager T=1 decode
AUTO on + IMPL=eager:
  decode:                         past_len > BDH_ATTN_AUTO_THRESHOLD
  cold/prefill:                   T > BDH_ATTN_AUTO_COLD_THRESHOLD
  unset COLD_THRESHOLD:            mirrors AUTO_THRESHOLD
  equality or shorter length:     eager
IMPL != eager:                    explicit backend is never overridden
```

`BDH_ATTN_AUTO` remains opt-in and `BDH_ATTN_IMPL=eager` remains the default.
The long-path preference is Triton only when CUDA+Triton is available, otherwise
blocked; the strict gate itself is shared by both resolvers, while the cold
threshold may be lowered independently for peak-score-memory experiments.
Attention math is unchanged: raw scores × strict `tril(diagonal=-1)`.

### Harness polish

`benchmarks/bench_generate.py --mode auto-ab` now accepts
`--auto-cold-threshold` (default: mirror `--auto-threshold`) and reports both
effective gates at every prompt. It checks the cold and decode resolver results
independently, retains AUTO0/AUTO1 token parity and `aten::cat == 0`, and marks
a decode gate crossed during newly generated tokens. This avoids treating a
cold-only override as if it also moved the decode crossover.

Example short A/B smoke (CPU wall only):

```bash
OMP_NUM_THREADS=2 python benchmarks/bench_generate.py \
  --mode auto-ab --prompts 16,40 --new 2 --layers 1 --d 32 \
  --heads 2 --mlp-mult 4 --warmup 0 --iters 1 \
  --auto-threshold 32 --auto-cold-threshold 8
```

The smoke completed on CPU with AUTO0/AUTO1 token parity and `aten::cat=0`
for both prompts. At `S=16`, AUTO selected `cold=blocked, decode=eager`; at
`S=40`, it selected `cold=blocked, decode=blocked`, confirming the independent
gates. The printed medians are local CPU wall timings only.

Do not infer GPU/kernel wins from CPU timings. Re-tune thresholds only with
actual target-device measurements; no GPU was available for this work.

### Tests

```bash
pytest -q tests/test_attn_auto.py tests/test_prefill_blocked.py tests/test_attention_mask.py
```

Coverage includes AUTO-enabled cold parity with an independent cold gate,
AUTO-disabled eager cold parity, decode parity, strict threshold boundaries,
explicit backend non-override, invalid env values, and the default eager
contract. No GPU claims are made.

### Non-goals

- No default `BDH_ATTN_AUTO=1` and no change to the eager default.
- No attention mask, score scaling, softmax, or cache layout changes.
- No GPU claims, no public PR, and no PRs to `pathwaycom/*`.


## opt/profile-v12 — re-profile post-#116–#120 tip (2026-09-19)

**Branch:** `opt/profile-v12` (private `katulevskiy/bdh-gpu-opt` only; no public
PR).
**Profile source:** `f34adc0` (`#120` auto-thr-v2), after `#119` amp-train-v2,
`#118` docs-v27, `#117` compile-train-v2, and `#116` online-decode-t1-v2.
This is a docs-only profile branch; the default eval/generate path is unchanged.

### Method

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python \
  benchmarks/profile_forward.py --mode all
# torch 2.14.0+cu130  cuda=False  device=cpu
# cfg: layers=4 d=128 nh=4 B=4 T=128; generate prompt=16 / new=32
```

The harness uses two warmups, one wait, and three active steps. Absolute CPU
milliseconds are profiler-inflated; the table reports self-CPU percentages and
aggregate calls over the three active steps. Chrome traces remain under
`benchmarks/traces/` (gitignored).

### CPU profile highlights (self CPU)

| Mode | Top self-CPU operators (calls) | Notes |
|------|--------------------------------|-------|
| Attention | `aten::mul` **25.84% (12)**, `aten::bmm` **22.99% (6)**, `aten::complex` **15.11% (6)**, `aten::copy_` **13.50% (6)**, `aten::add` **7.32% (3)**, `aten::sub` **7.30% (3)** | Default eager still computes raw scores then applies strict `tril_(diagonal=-1)`; no softmax, scale, or SDPA. |
| Forward | `aten::bmm` **28.06% (36)**, `aten::mm` **23.22% (27)**, `aten::mul` **16.39% (60)**, `aten::complex` **11.32% (24)**, `aten::copy_` **7.86% (36)**, `aten::clamp_min_` **3.01% (24)** | `aten::cat=0`, `aten::contiguous=0`; warmed harness is **36 copy_ / 3 = 12 per active call**. |
| Generate | `aten::mm` **22.75% (795)**, `aten::bmm` **14.65% (1,188)**, `aten::mul` **3.01% (1,980)**, `aten::native_layer_norm` **2.34% (1,287)**, `aten::matmul` **2.33% (1,587)**, `aten::einsum` **1.93% (396)**, `aten::copy_` **0.85% (1,182)** | `aten::cat=0`, `aten::contiguous=0`; generate `copy_` is **1,182 / 3 = 394 per active call**. |

Separate one-shot CPU profiler checks at the same cfg report forward
`aten::copy_=14` and generate `aten::copy_=396`, with `aten::cat=0` /
`aten::contiguous=0` in both modes. These are CPU operator counts, not GPU
performance measurements.

### Smoke and verdict

```text
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m pytest \
  tests/test_gen_copy_tax.py tests/test_cache_pack.py tests/test_rope_decode.py \
  tests/test_rope_fuse.py tests/test_copy_tax.py tests/test_gen_sample.py \
  tests/test_gen_host.py tests/test_attn_auto.py -q
# 108 passed, 1 skipped in 6.91s
```

The post-#116–#120 tip preserves the cat-free, contiguous-free generate and
forward paths, with scheduled generate `copy_` at ~394 per active call and
isolated forward `copy_` at 14. Defaults remain unchanged: unset
`BDH_ATTN_IMPL` resolves to `eager`, AUTO remains off, and unset
`BDH_ROPE_IMPL` remains eager. This box has no GPU, so no GPU timing, kernel
win, or speedup is claimed; attention remains raw scores × strict lower-
triangular `tril(-1)`.

### Non-goals

- No PRs to `pathwaycom/*`; private repo only; no public PR
- No default attention/RoPE implementation change
- No softmax / scale / SDPA / diagonal inclusion
- No GPU claims from CPU profiler percentages or copy_ counts
## opt/sparse-v2 — gated density re-smoke and keep-OFF guardrails (2026-09-19)

**Branch:** `opt/sparse-v2` (private `katulevskiy/bdh-gpu-opt` only).
**Base tip:** `f34adc0` (`main`). Sparse remains P2 and default OFF.

### Audit / deepen

`bdh_sparse.py` remains an opt-in helper module; `BDH.forward` does not import
or dispatch to it. The optional encoder materialization hook now has a clear
runtime gate:

```text
BDH_SPARSE_PROBE unset / 0 / false:  sparse probe disabled (default)
BDH_SPARSE_PROBE=1 (or true/on):    optional probe hook may run
```

An ungated `encoder_relu_matmul(..., use_sparse=True)` raises an actionable
error instead of silently creating a second model path. The primitive sparse
and masked-GEMM helpers remain directly testable for numerical equivalence.

`benchmarks/bench_sparse_probe.py` uses the same gate and exits before corpus
loading, training, or timing when it is unset. `--density-only` makes the
short-train density re-smoke explicit; `--enforce-density-guardrail` checks a
conservative floor below the prior observation (~27% `x`, ~12% `xy`) without
claiming that a short CPU run reproduces the paper's ~5% trained density.

### Density re-smoke (CPU-only)

```bash
BDH_SPARSE_PROBE=1 python benchmarks/bench_sparse_probe.py \
  --steps 150 --log-every 150 --batch 8 --block 64 \
  --density-only --enforce-density-guardrail
```

Observed at this tip:

```text
step=0    x=0.5018  y=0.4856  xy=0.2427
step=150  x=0.2663  y=0.4269  xy=0.1137
 density_guardrail=pass (x>=0.2000, xy>=0.0800)
```

The probe still observes density well above the paper reference at 150 steps.
This is a local CPU density observation, not a GPU or trained-model claim.
The prior CPU crossover conclusion remains unchanged: sparse paths did not
beat dense at the measured workload, so no sparse path is enabled by default.

### Tests

```bash
python -m pytest tests/test_sparse.py -q
# 13 passed, 1 warning (PyTorch sparse CSR beta notice)
BDH_SPARSE_PROBE=1 python benchmarks/bench_sparse_probe.py --steps 1 \
  --log-every 1 --batch 2 --block 32 --density-only
```

Coverage includes the default-off gate, gated sparse round-trip smoke, dense
BDH forward isolation, COO/CSR/masked equivalence, and decoder layout parity.
No GPU was available; no GPU timing, kernel, or speedup claim is made.

### Non-goals

- No sparse wiring into `BDH.forward`; dense remains the default path.
- No default enablement or threshold auto-selection; default BDH callers stay
  unchanged. The optional encoder hook now intentionally requires the gate.
- No GPU claims from CPU density or timing observations.


## opt/rope-gpu-scaffold — paired T=1 Triton launch (2026-09-19)

**Branch:** `opt/rope-gpu-scaffold` (private `katulevskiy/bdh-gpu-opt` only).
**Base:** `aea2501` (current `origin/main`, after #123).

### Audit after #90 (`opt/rope-fuse-v2`)

#90 added the row-wise flat Triton kernel and the CPU blocked parity scaffold,
but two GPU-readiness gaps remained: flat Triton staging expanded the
broadcast `(1,1,1,N)` T=1 cis to every `(B,H,1,N)` row, and the warmed paired
T=1 `Attention` path called `rope_rotate_paired` directly, bypassing
`BDH_ROPE_IMPL=fused`.

### Deepen

- Add `fused_rope_rotate_paired` and `bdh_rope_rotate_paired`.
- Route warmed T=1 decode through that dispatch without changing eager math.
- Reuse the existing Triton kernel with a zero cis row stride for the common
  one-row paired table; no host `expand` for the T=1 CUDA scaffold.
- Keep a general broadcast fallback, CPU paired implementation, `out=`, and an
  analytic V backward for the opt-in Triton path.

`BDH_ROPE_IMPL=eager` remains the default and uses the paired reference path;
`BDH_ROPE_IMPL=fused` is still opt-in. No GPU performance or correctness claim
is made on this CPU-only box.

### Tests

- CPU paired T=1 parity remains exact against eager.
- CUDA/Triton tests are skip-gated on both `torch.cuda.is_available()` and
  Triton availability, so CPU pytest stays clean.
- The new CUDA test covers the zero-stride paired launch against eager when a
  CUDA+Triton environment is present.

## opt/decode-gemm-v2 — strided packed KR/V views in T=1 Triton decode (2026-09-19)

**Branch:** `opt/decode-gemm-v2` (private `katulevskiy/bdh-gpu-opt` only).
**Base:** `f16115c` (`origin/main`, tip #125).

### Audit / deepen

The T=1 Triton decode kernel already accepts explicit element strides, but its
host launcher unconditionally staged `Q`, `K`, and `V` through contiguous
tensors. With `CacheManager`, the live KR/V prefix is capacity-padded and
therefore strided across batches; staging it every decode step adds a
full-prefix device copy before the score GEMM.

The launcher now flattens view-compatible Q/K/V shapes with `reshape` and
passes their native strides to the existing kernel. Broadcast values remain
`(B,1,S,D)` / `(B,S,D)` and per-head values retain their native batch stride;
non-view-compatible inputs still use the normal reshape fallback. No kernel
semantics change: decode remains raw `(Q @ K_past.T) @ V_past`, with packed past
keys excluding the new token. Default eager dispatch is unchanged.

### Correctness (CPU-only; no GPU claims)

```text
/workspace/bdh-gpu-opt/.venv/bin/python -m pytest \
  tests/test_inc_decode.py tests/test_cuda_decode.py tests/test_gen_sample.py -q
# 87 passed, 7 skipped in 2.80s

Full suite: 525 passed, 19 skipped, 3 warnings in 57.35s
```

Coverage includes eager/blocked/Triton/CUDA-ref parity, exact `pos0 == 0`,
capacity-padded packed-cache T=1 views, and eager-vs-opt-in `generate()` with
`torch.cat` count zero. CUDA/Triton hardware execution remains skip-gated on
this CPU-only box; no GPU timing, kernel-correctness, or speedup claim is made.

### Non-goals

- Default eager behavior remains unchanged.
- No softmax, scaling, diagonal inclusion, or full score materialization.
- No public PR and no PRs to `pathwaycom/*`; private repo only.
- No GPU claims from CPU tests.


## opt/profile-v13 — CPU re-profile after #128 (2026-09-19)

**Branch:** `opt/profile-v13` (private `katulevskiy/bdh-gpu-opt` only; no public
PR).
**Profile source:** `45b4afe` (`#128`, packed Triton decode KR/V strides), after
`#127` profile CI scaffolding and `#126` docs update. The profile does not make
GPU claims and does not alter attention semantics.

### Method

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /workspace/bdh-gpu-opt/.venv/bin/python \
  benchmarks/profile_forward.py --device cpu --mode all
# torch 2.14.0+cu130  cuda=False  device=cpu
# cfg: layers=4 d=128 nh=4 B=4 T=128; generate prompt=16 / new=32
```

The harness uses two warmups, one wait, and three active steps. Percentages are
self-CPU percentages from the three active steps; operator counts below are the
aggregate counts followed by the per-active-call count. `cat` and `contiguous`
were absent from all three CPU traces (zero calls).

### CPU profile highlights (self CPU)

| Mode | Top self-CPU operators | copy_ / cat / contiguous |
|------|-------------------------|---------------------------|
| Attention | `aten::bmm` **26.82%**, `aten::mul` **25.20%**, `aten::complex` **15.25%**, `aten::copy_` **11.32%**, `aten::add` **6.48%**, `aten::sub` **6.40%** | `copy_` **6 / 3 = 2 per call**; `cat=0`; `contiguous=0` |
| Forward | `aten::bmm` **28.14%**, `aten::mm` **22.93%**, `aten::mul` **16.48%**, `aten::complex` **11.82%**, `aten::copy_` **7.20%** | `copy_` **36 / 3 = 12 per call**; `cat=0`; `contiguous=0` |
| Generate | `aten::mm` **21.81%**, `aten::bmm` **14.39%**, `aten::mul` **3.06%**, `aten::matmul` **2.41%**, `aten::native_layer_norm` **2.28%**, `aten::einsum` **2.00%**, `aten::copy_` **0.83%** | `copy_` **1,182 / 3 = 394 per call**; `cat=0`; `contiguous=0` |

Attention remains raw scores × strict `tril(diagonal=-1)`: no softmax, scale,
or SDPA. CPU profiler percentages and copy counts are not GPU performance
measurements.

### Smoke and verdict

```text
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /workspace/bdh-gpu-opt/.venv/bin/python -m pytest \
  tests/test_cache_pack.py tests/test_inc_decode.py tests/test_attention_mask.py \
  tests/test_gen_copy_tax.py tests/test_gen_sample.py tests/test_cuda_decode.py -q
# 130 passed, 7 skipped in 7.55s
```

The post-#128 CPU profile preserves the prior cat-free/contiguous-free forward
and generate paths. No GPU timing, kernel win, correctness, or speedup claim is
made on this CPU-only box.

### Non-goals

- No PRs to `pathwaycom/*`; private repo only; no public PR.
- No softmax, scale, diagonal inclusion, or SDPA substitution.
- No GPU claims from CPU profiler percentages or copy counts.


## Generate microbench v2 (2026-09-19)

- `opt/gen-bench-v2` starts from tip `5200b4f` and deepens
  `benchmarks/bench_generate.py` with `--auto-threshold-sweep`. The helper
  repeats the existing long-S AUTO A/B for a stable, de-duplicated list of
  decode thresholds; cold thresholds mirror each value unless explicitly
  overridden. Every run retains seeded token parity, `aten::cat=0`, and strict
  cold/decode resolver checks.
- The harness remains CPU-safe and default-eager: no environment defaults or
  generate semantics change when the sweep option is omitted. CPU wall medians
  are diagnostic only; this change makes no GPU timing or kernel claim.
## opt/cuda-build-v2 — CPU-safe CUDA build configuration smoke (2026-09-19)

**Branch:** `opt/cuda-build-v2` (private `katulevskiy/bdh-gpu-opt` only).
**Base tip:** `891b7c5` (`main`, after #131). No pathwaycom or public PR.

### Audit / deepen

The optional `csrc/` build already uses C++20 and `kernels.cuda_attn` already
soft-imports a missing `bdh_cuda_ext`. The remaining CPU-box gap was the
CUDA-enabled torch-wheel case: `BDH_BUILD_EXT=1 BDH_BUILD_CUDA=1` could enter
`CUDAExtension` setup even when the box had no `nvcc`, producing an opaque
ninja/toolchain traceback. `setup.py` now checks `CUDA_HOME`/`CUDA_PATH` and
`PATH` for `nvcc` before constructing the CUDA extension. A missing compiler
prints an explicit skip and leaves the pure-Python CPU refs available;
`BDH_FORCE_CPU_EXT=1` remains the explicit C++-only route.

The new CPU-safe configuration tests cover the default pure-Python no-op and
the forced-CUDA/no-`nvcc` skip without compiling or claiming GPU behavior.
Attention semantics remain raw scores × strict `tril(diagonal=-1)`, and all
defaults are unchanged.

### Tests

```text
python -m pytest tests/test_cuda_build.py tests/test_cuda_attn.py tests/test_cuda_decode.py -q
# expected on this box: CPU tests pass; CUDA-only tests skip cleanly
```

No GPU is available here, so this is build configuration/import smoke only;
no CUDA compilation, kernel correctness-on-hardware, timing, or speedup claim
is made.

### Non-goals

- No default `BDH_BUILD_EXT`, attention, decode, or implementation change.
- No softmax, scale, diagonal inclusion, or full-score materialization.
- No public or `pathwaycom/*` PRs; private repository only.


## opt/scorev-fuse-v3 — B=1 direct shared-V T=1 epilogue (2026-09-19)

**Branch:** `opt/scorev-fuse-v3` (private `katulevskiy/bdh-gpu-opt` only; no
`pathwaycom/*` or public PR).
**Base tip:** `dba5f75` (`main`, after #134 CUDA-build-v2).

### Audit / deepen

The default eager T=1 decode remains `_two_gemm_decode`; it is unchanged. The
blocked/online CPU-safe path and the CPU fallback behind `triton` already keep
CacheManager's broadcast `V=(B,1,S,D)` unexpanded and use a direct `baddbmm`
out epilogue for score×V tiles. The CUDA/Triton launcher remains stride-aware
for packed KR/V views; this follow-on does not alter its GPU scaffold.

The remaining small hot-path staging was the common `B=1` case: the shared-V
epilogue rebuilt `(B,H,1,D)` / `(B,H,1,Bj)` views and entered a one-iteration
Python batch loop even though the score and output were already flattened as
`(B*H,1,*)`. The B=1 branch now reuses those flattened views and sends one
zero-stride shared-V view directly to `torch.baddbmm(..., out=target)`. `B>1`
keeps the existing per-sample zero-stride path, and autograd keeps the
allocation-safe matmul fallback because `out=` operators are not differentiable.

Math is unchanged: raw scores × strict `tril(diagonal=-1)`, no softmax, no
scale, no SDPA. `pos0==0`, cat-free generate, default eager, and all resolver
behavior remain unchanged.

### Tests (CPU-only; no GPU claims)

```text
/workspace/bdh-gpu-opt/.venv/bin/python -m pytest \
  tests/test_inc_decode.py tests/test_gen_sample.py tests/test_fuse_scorev.py -q
# 111 passed, 3 skipped in 20.40s

/workspace/bdh-gpu-opt/.venv/bin/python -m pytest -q
# 533 passed, 19 skipped, 3 warnings in 121.09s
```

Coverage includes eager-vs-blocked/online/triton/cuda-ref decode parity, exact
position-zero zeros, the B=1 direct out-buffer epilogue, and generate token
parity with `torch.cat` count zero. This CPU box has no GPU; no timing,
kernel-on-hardware, correctness-on-GPU, or speedup claim is made.

### Non-goals

- No default eager, attention math, or generate behavior change.
- No softmax, scale, diagonal inclusion, or full score materialization.
- No GPU claims; no public PR and no PRs to `pathwaycom/*`.

## opt/compile-train-v3 — clarify CPU-safe compile fallback (2026-09-19)

**Branch:** `opt/compile-train-v3` (private `katulevskiy/bdh-gpu-opt` only).
**Base tip:** `cc62e0b` (`#135` scorev-fuse-v3, after #134). No public PR.

### Audit / deepen

`train.maybe_compile()` now labels every compile/probe soft-fallback with the
requested device, mode, probe, and `fullgraph` setting, explicitly naming the
original eager module as the returned fallback. This makes construction failure,
missing `example_y` for `train_bwd`, and first-probe failure distinguishable
without changing the safe behavior.

The `maybe_compile()` contract also spells out the FULLGRAPH/AUTOGRAD
interaction: `FULLGRAPH=1` constrains the module probe, `AUTOGRAD=1` plus
`train_bwd` exercises the analytic attention backward path, and optimizer step /
gradient clearing remain eager. A graph break or unsupported backward remains a
soft fallback, not a hard failure. Defaults remain `BDH_COMPILE=0`,
`BDH_COMPILE_PROBE=train_bwd`, and `BDH_COMPILE_FULLGRAPH=0`.

### CPU-safe validation

```text
/workspace/bdh-gpu-opt/.venv/bin/python -m pytest tests/test_compile.py -q
# 36 passed, 1 skipped, 1 warning in 56.36s

OMP_NUM_THREADS=2 BDH_BENCH_COMPILE=0 BDH_BENCH_COMPILE_BLOCKED=0 \
  BDH_BENCH_COMPILE_MODE=0 BDH_BENCH_COMPILE_DROPOUT=0 \
  BDH_BENCH_COMPILE_FULLGRAPH=0 BDH_BENCH_AMP=0 \
  /workspace/bdh-gpu-opt/.venv/bin/python benchmarks/bench_train_step.py
# CPU baseline smoke completed; compile/matrix/AMP sections explicitly skipped

OMP_NUM_THREADS=2 /workspace/bdh-gpu-opt/.venv/bin/python -m pytest tests/ -q
# 533 passed, 19 skipped, 4 warnings in 164.89s
```

No GPU or CUDA-graph measurement was available or added. Attention remains raw
scores × strict `tril(diagonal=-1)`.


## opt/rope-gpu-v2 — conservative Triton gates and T>1 CPU parity (2026-09-19)

**Branch:** `opt/rope-gpu-v2` (private `katulevskiy/bdh-gpu-opt` only; no
public PR and no `pathwaycom/*`).
**Base tip:** `6a4c856` (`main`, after #137 docs).

### Audit / deepen

The default `BDH_ROPE_IMPL=eager` path is unchanged. The fused RoPE scaffold
now gates Triton on the complete tensor set instead of checking only `v`:
`v`, cis, and optional `out` must be on the same CUDA device and use the
conservative fp16/bf16/fp32 set. Unsupported or mixed-device inputs therefore
skip to the existing PyTorch/blocked fallback rather than reaching a kernel
launch with a device or dtype mismatch. `backend_info(cuda)` also reports a
clean skip without trying to initialize an unavailable CUDA runtime.

The CPU coverage adds T>1 parity for transposed leading dimensions and an
explicit out buffer, alongside the no-CUDA/Triton gate and diagnostic checks.
RoPE math, cache behavior, strict lower-triangular attention, default eager
dispatch, and generate behavior are unchanged.

### Tests (CPU-only; no GPU claims)

```text
python -m pytest tests/test_rope_fuse.py tests/test_rope_decode.py -q
# 33 passed, 2 skipped in 2.08s

python -m pytest -q
# 536 passed, 19 skipped, 3 warnings in 60.69s
```

This CPU box has no usable CUDA/Triton execution target; no timing, kernel-on-
hardware, GPU correctness, or speedup claim is made.

### Non-goals

- No default eager, attention math, cache, or generate behavior change.
- No GPU claims; no public PR and no PRs to `pathwaycom/*`.


## opt/prefetch-h2d-v2 — make the CPU no-op contract explicit (2026-09-19)

**Branch:** `opt/prefetch-h2d-v2` (private `katulevskiy/bdh-gpu-opt` only).
**Base tip:** `da664ec` (`#138` rope-gpu-v2, after `#137`). No public PR.

### Audit / deepen

The H2D staging flag remains unchanged (`BDH_PREFETCH_H2D=1` and
`BDH_PREFETCH_ASYNC=1`). `BatchPrefetcher` now documents and tests the CPU
contract explicitly: the flag is gated by `device.type` before any CUDA stream
or event is constructed, the staged-device lookahead stays absent, and the
CPU transfer preserves the original tensor objects rather than adding a copy or
synchronization point. The constructor override is covered for unset/default,
`False`, and `True` values.

No attention or training math changed: attention remains raw scores × strict
`tril(diagonal=-1)`. Prefetch remains opt-in to the existing harness; no GPU
throughput, overlap, or correctness claim is made.

### CPU-safe validation

```text
/workspace/bdh-gpu-opt/.venv/bin/python -m pytest tests/test_dataloader.py -q
# 17 passed, 1 skipped

OMP_NUM_THREADS=2 /workspace/bdh-gpu-opt/.venv/bin/python -m pytest tests/ -q
# 539 passed, 19 skipped, 3 warnings in 69.09s
```

### Non-goals

- No default prefetch or H2D flag changes.
- No GPU timing or throughput claims.
- No public PR and no PRs to `pathwaycom/*`.

## opt/blocked-tile-v3 — wide-head CPU cold parity (2026-09-19)

**Branch:** `opt/blocked-tile-v3` (private `katulevskiy/bdh-gpu-opt` only; no
public PR and no `pathwaycom/*`).
**Base tip:** `6fd7950` (`main`, after #139).

### Audit / deepen

The default `BDH_ATTN_IMPL=eager` path is unchanged. The long CPU cold branch
(`T >= 256`) continues to flatten the query/key batch to `B*H` dense `bmm`
inputs. For shared `V=(B,1,T,D)`, it preserves the `(B,T,D)` view and
broadcasts it only for each score×V tile; head-matched `V=(B,H,T,D)` remains a
dense flattened batch. The implementation doc now states this tile-bound and
staging behavior explicitly.

CPU parity coverage now includes the wide-head shape `N=128`, `D=256` at
`T=512`, for both shared-V and head-matched-V layouts. This exercises the
long-T blocked path while retaining strict raw-score × `tril(diagonal=-1)` and
position-zero semantics. The Triton/CUDA wide-head tile policy and explicit
overrides remain unchanged; no GPU timing or claim is added.

### Tests (CPU-only; no GPU claims)

```text
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
/workspace/bdh-gpu-opt/.venv/bin/python -m pytest \
  tests/test_prefill_blocked.py tests/test_triton_attn.py -q
# 47 passed, 3 skipped in 4.39s

OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
/workspace/bdh-gpu-opt/.venv/bin/python -m pytest -q
# 541 passed, 19 skipped, 3 warnings in 101.13s
```

Defaults remain eager and AUTO-off; no attention math, cache, generate, or
GPU behavior is claimed to change.

### Non-goals

- No default eager or AUTO behavior change.
- No softmax, scaling, diagonal inclusion, or full-score materialization.
- No GPU claims; no public PR and no PRs to `pathwaycom/*`.

## opt/amp-train-v3 — explicit CPU-safe AMP dtype matrix (2026-09-19)

**Branch:** `opt/amp-train-v3` (private `katulevskiy/bdh-gpu-opt` only; no
public PR and no `pathwaycom/*`).
**Base tip:** `40eac80` (`main`, after #140 docs).

### Audit / deepen

The AMP train harness now has one parameterized configuration matrix covering
`float32`, `bfloat16`, and `float16`. Each case asserts the resolved PyTorch
dtype, the forward-only flag, and the GradScaler gate; CPU cases skip only when
the requested autocast backend fails its existing smoke probe. This makes the
CPU-safe dtype contract explicit without changing train execution behavior.
Defaults remain fp32 / AMP-off, and GradScaler remains restricted to
float16 + CUDA. Attention remains raw scores × strict `tril(diagonal=-1)`.

### Tests (CPU-only; no GPU claims)

```text
python -m pytest tests/test_bf16_train.py -q
# 21 passed in 2.12s
```

No GPU is available here; no throughput, kernel-on-hardware, GPU correctness,
or speedup claim is made.

### Non-goals

- No default AMP, optimizer, attention, or mask behavior change.
- No GPU claims; no public PR and no PRs to `pathwaycom/*`.

## opt/profile-v14 — CPU re-profile after #141/#142 (2026-09-19)

**Branch:** `opt/profile-v14` (private `katulevskiy/bdh-gpu-opt` only; no
public PR).
**Profile source:** `68f949a` (`#142`, after `#141` blocked-tile-v3,
`#140` docs, and `#139` prefetch-h2d-v2). The profile is CPU-only and makes no
GPU claim.

### Method

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
/workspace/bdh-gpu-opt/.venv/bin/python benchmarks/profile_forward.py \
  --device cpu --mode all
# torch 2.14.0+cu130  cuda=False  device=cpu
# cfg: layers=4 d=128 nh=4 B=4 T=128; generate prompt=16 / new=32
```

The harness uses two warmups, one wait, and three active steps. Percentages are
self-CPU percentages from the three active steps; operator counts below are the
aggregate counts followed by the per-active-call count. `cat` and `contiguous`
were absent from all three CPU traces (zero calls).

### CPU profile highlights (self CPU)

| Mode | Top self-CPU operators | copy_ / cat / contiguous |
|------|-------------------------|---------------------------|
| Attention | `aten::bmm` **26.11%**, `aten::mul` **23.93%**, `aten::complex` **15.84%**, `aten::copy_` **12.50%**, `aten::add` **7.89%**, `aten::sub` **5.85%** | `copy_` **6 / 3 = 2 per call**; `cat=0`; `contiguous=0` |
| Forward | `aten::bmm` **28.32%**, `aten::mm` **23.89%**, `aten::mul` **15.36%**, `aten::complex` **11.70%**, `aten::copy_` **6.83%**, `aten::clamp_min_` **3.02%** | `copy_` **36 / 3 = 12 per call**; `cat=0`; `contiguous=0` |
| Generate | `aten::mm` **20.32%**, `aten::bmm` **13.16%**, `aten::mul` **3.05%**, `aten::matmul` **2.59%**, `aten::native_layer_norm` **2.38%**, `aten::einsum` **2.18%**, `aten::copy_` **0.88%** | `copy_` **1,182 / 3 = 394 per call**; `cat=0`; `contiguous=0` |

The attention path remains raw scores × strict `tril(diagonal=-1)`: no softmax,
scale, or SDPA. Relative to profile-v13, the short CPU window preserves the
same `copy_` counts and cat-free/contiguous-free forward and generate paths.
These CPU profiler percentages and operator counts are not GPU performance
measurements.

### Smoke and verdict

```text
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
/workspace/bdh-gpu-opt/.venv/bin/python -m pytest \
  tests/test_bf16_train.py tests/test_prefill_blocked.py tests/test_cache_pack.py \
  tests/test_inc_decode.py tests/test_gen_sample.py -q
# 131 passed, 3 skipped in 3.09s
```

No GPU is available on this box, so no GPU timing, kernel win, correctness, or
speedup claim is made. Defaults and attention semantics remain unchanged.

### Non-goals

- No PRs to `pathwaycom/*`; private repo only; no public PR.
- No softmax, scale, diagonal inclusion, or SDPA substitution.
- No GPU claims from CPU profiler percentages or copy counts.

## opt/sparse-v3 — make sparse probe outcomes scriptable (2026-09-19)

**Branch:** `opt/sparse-v3` (private `katulevskiy/bdh-gpu-opt` only; no public
PR and no PRs to `pathwaycom/*`).
**Base tip:** `ca353f2` (`main`, after #144 profile-v14 / merged #123 follow-on).

### Audit / deepen

The sparse harness remains explicitly gated by `BDH_SPARSE_PROBE=1`, and the
production `bdh.py` path remains dense. This follow-on gives the probe a stable
script contract: the default-off no-op returns exit `0`, a successful optional
run returns exit `0`, and an enforced density guardrail failure returns exit
`2` with a concise stderr diagnostic. This makes CI or shell callers able to
distinguish “not requested” from a failed guardrail without changing defaults.

Attention remains raw scores × strict `tril(diagonal=-1)`; no softmax, scale,
diagonal inclusion, GPU timing, or GPU correctness claim is added.

### CPU-safe validation

```text
python -m pytest tests/test_sparse.py -q
# 14 passed

BDH_SPARSE_PROBE=0 python benchmarks/bench_sparse_probe.py --skip-train --skip-crossover
# exit 0; SPARSE PROBE DISABLED (default)
```

Density stays OFF by default; no GPU claim is made from this CPU-only change.

## opt/online-decode-v3 — reuse shared-V views for online T=1 (2026-09-19)

**Branch:** `opt/online-decode-v3` (private `katulevskiy/bdh-gpu-opt` only; no
public PR).
**Base tip:** `2a2b1fb` (main after #145 sparse-v3; rebased before publishing).

### Audit / deepen

The default eager path remains unchanged. The opt-in blocked/online T=1 path
now takes one reused `(B,S,D)` view of CacheManager's shared
`V=(B,1,S,D)` layout before scanning past tiles, instead of rebuilding a
four-dimensional slice on every tile. It never expands that view to `B*H`.

For B=1, the existing zero-stride `baddbmm(..., out=target)` epilogue remains
the no-product-buffer fast path. For B>1, the shared-V epilogue now uses one
broadcast matmul over `(B,H,1,*)` and adds the small output tile, removing the
per-sample Python loop while retaining the same raw score×V accumulation.
The #128 Triton launcher continues to consume view-compatible packed KR/V
strides without host staging; unsupported layouts retain the existing reshape
fallback.

Attention math is unchanged: raw scores × strict `tril(diagonal=-1)`, no
softmax, no scale, and no self-attention for a packed decode prefix. Default
eager dispatch remains unchanged. This CPU-only validation makes no GPU claim.

### Tests

Focused coverage exercises eager-vs-blocked/online decode parity, exact
`pos0 == 0`, the B=1 direct output epilogue, the B>1 shared-V path, padded
non-contiguous cache views, and cat-free generate parity.

No GPU is available on this box; CUDA/Triton hardware tests remain skip-gated.

### Non-goals

- No default eager, cache, generate, or attention math change.
- No softmax, scaling, diagonal inclusion, or full-score materialization.
- No GPU timing, kernel-on-hardware, GPU correctness, or speedup claim.
## opt/cuda-cold-v4 — actionable CPU-safe CUDA skips (2026-09-19)

**Branch:** `opt/cuda-cold-v4` (private `katulevskiy/bdh-gpu-opt` only; no
public PR and no PRs to `pathwaycom/*`).
**Base tip:** `2a2b1fb` (`main`, after #145 sparse-v3; audited from #114 tip `ca353f2`).

### Audit / deepen

The #114 CUDA cold-tile tests now report an actionable collection-time reason
for every native CUDA skip: missing CUDA device, missing `bdh_cuda_ext`, or a
loaded extension without a CUDA kernel. A CPU-only run still exercises all
reference, tiled, wide-head bound, strict raw-score × `tril(diagonal=-1)`, and
build-smoke coverage; only native CUDA execution is skipped. Defaults remain
eager and AUTO-off. This is a skip-diagnostic polish only: it adds no CUDA
compilation, timing, correctness, or speedup claim.

### Tests (CPU-only; no GPU claims)

```text
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
/workspace/bdh-gpu-opt/.venv/bin/python -m pytest tests/test_cuda_attn.py -q -rs
# 18 passed, 5 skipped in 1.18s
# Native CUDA skips name the missing device/extension condition.

OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
/workspace/bdh-gpu-opt/.venv/bin/python -m pytest -q
# 545 passed, 19 skipped, 3 warnings in 60.92s
```

No GPU is available on this box.

### Non-goals

- No default eager or AUTO behavior change.
- No softmax, scaling, diagonal inclusion, or change to raw score × strict
  `tril(diagonal=-1)` semantics.
- No GPU claims; no public PR and no PRs to `pathwaycom/*`.


## opt/auto-thr-v3 — CPU-safe AUTO threshold sweep smoke (2026-09-19)

**Branch:** `opt/auto-thr-v3` (private `katulevskiy/bdh-gpu-opt` only; no
public PR and no PRs to `pathwaycom/*`).
**Base tip:** `c557b55` (main after #148 docs refresh; code tip #146).

### Audit / deepen

The #131 `bench_generate.py --auto-threshold-sweep` harness now has focused
CPU-safe smoke coverage. The test drives the sweep dispatcher without a model
run, confirming that threshold order is stable, duplicate decode thresholds are
deduplicated, recursive sweep dispatch is disabled for each child run, and an
explicit independent cold threshold is preserved across every decode threshold.
Malformed, empty, and negative sweep items are also rejected. This hardens the
reporting/control surface for independent strict gates without changing the
AUTO resolver or any defaults.

Attention remains raw scores × strict `tril(diagonal=-1)`; eager remains the
default and `BDH_ATTN_AUTO` remains off. This CPU-only test adds no GPU timing,
CUDA/Triton execution, kernel correctness, or speedup claim.

### CPU validation

```text
python -m pytest tests/test_auto_threshold_sweep.py tests/test_attn_auto.py -q
# 28 passed in 2.81s

OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m pytest -q
# 552 passed, 19 skipped, 3 warnings in 67.18s
```

### Non-goals

- No default eager/AUTO behavior change.
- No softmax, scaling, diagonal inclusion, or change to raw score × strict
  `tril(diagonal=-1)` semantics.
- No GPU claims; no public PR and no PRs to `pathwaycom/*`.

## opt/gen-copy-v2 — confirm post-#110 generate copy ceiling (2026-09-19)

**Branch:** `opt/gen-copy-v2` (private `katulevskiy/bdh-gpu-opt` only).
**Base tip:** `748a138` (`main` after #150 triton-cold-v4; current through #150).

### Audit / verdict

The post-#110 generate path was audited again with the default eager,
`temperature=1.0`, `top_k=None` contract. No further default-safe cut was
found, so this PR is a small CPU probe/documentation update; `bdh.py` is
unchanged.

| Remaining source | Current contract | Safe cut status |
|---|---|---|
| Packed V slot `dst_v.copy_(v_tok)` | Owns the `(B, 1, T, D)` cache snapshot before residual LN reuses/replaces `x`; about one write per layer and forward step | Required unless cache ownership/layout changes |
| RoPE pair store | `_store_pairs` already writes one fp32 `copy_` per layer/step with no `cat`; T>1 prefill and T=1 decode share this path | Already at the safe eager floor |
| Default `torch.multinomial(..., out=idx_out)` | `out=` removes the caller-side token assignment, but ATen still performs internal copy/conversion work (about four `copy_` events per B=1 sample on this CPU build) | Removing it requires a custom sampler or RNG/layout change |
| Prompt seed copy | One initial `out[:, :prompt_len].copy_(idx)` establishes owned output storage | Required for the preallocated cat-free output contract |

The tiny `test_remaining_generate_copy_ceiling_probe` probe checks both
remaining classes directly: an owned V destination must retain its snapshot
after the source is reused, and `torch.multinomial(out=...)` still emits
internal `aten::copy_` work. The existing attribution probe continues to
account for the warmed generate total at about **394 `copy_`/call** with
`aten::cat=0`; the exact count is CPU-build evidence, not a GPU claim.

### Validation / non-goals

- Default eager dispatch, strict raw `tril(diagonal=-1)`, token parity, and
  cat-free generate remain unchanged.
- No custom sampler, cache-layout change, default RNG-stream change, or GPU
  timing claim.
- No public or `pathwaycom/*` PRs; private repository only.

## opt/rope-fuse-v3 — reuse warmed T>1 table narrows (2026-09-19)

**Branch:** `opt/rope-fuse-v3` (private `katulevskiy/bdh-gpu-opt` only; no
public PR and no `pathwaycom/*`).
**Base tip:** `d60380f` (`main`, after #152; includes #90/#124/#138).

### Audit / deepen

The CPU-safe RoPE fuse keeps the default `BDH_ROPE_IMPL=eager` behavior and
adds one narrow cache for repeated table-backed `T>1` `(cos, sin)` ranges.
When a caller does not hoist `cos_sin` itself, the covered range now reuses the
same view until the single-slot range changes; rebuilding the generate table
invalidates the view. No RoPE math, attention semantics, GPU path, or timing
claim changes. The fused T>1 pair store and paired T=1 path remain as landed.

### CPU validation

```text
python -m pytest tests/test_rope_cache.py tests/test_rope_fuse.py tests/test_rope_decode.py -q
# 39 passed, 2 skipped in 2.04s
```

No GPU is available here, so no GPU timing, kernel-on-hardware correctness, or
speedup claim is made. Attention remains raw scores × strict
`tril(diagonal=-1)`.

### Non-goals

- No default eager, attention math, cache table contents, or generate behavior change.
- No GPU claims; no public PR and no PRs to `pathwaycom/*`.

## opt/layout-v3 — CPU probe remaining layout materializations (2026-09-19)

**Branch:** `opt/layout-v3` (private `katulevskiy/bdh-gpu-opt` only).
**Base tip:** `3d96337` (`main`, after #155 docs for #154).

### Audit / verdict

After layout-v2 and the warmed T>1 RoPE-table narrow cache, the remaining
contiguous/layout work was probed on CPU without flipping any default layout.
The eval encoder cache intentionally materializes one contiguous `(nh*N, D)`
weight during `eval()`/cache refresh; it is outside the forward hot path. Once
warmed, the default eager forward probe reports **0 `aten::contiguous`** and
**0 `aten::cat`**. It still reports two clone-backed layout materializations
per layer at the current CPU build, with shapes `(B, nh, T, D)` and
`(nh, B, T, D, 1)` from eager score×V/broadcast handling. These are retained
as an evidence boundary rather than “fixed” with an unsafe activation-layout
flip.

The default packed generate probe reports the remaining explicit contiguous
hotspots inside ATen sampling: one `(B, V)` materialization under softmax and
one `(B,)` materialization per sampled token under index selection. The model
path remains cat-free. Removing those would require a custom sampler/RNG or
other contract change, so no default-safe view reuse was found.

### CPU validation

```text
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  python -m pytest tests/test_layout_v2.py tests/test_layout_v3.py \
    tests/test_rope_cache.py tests/test_rope_fuse.py tests/test_gen_copy_tax.py -q -rs
# 45 passed, 2 skipped in 6.92s

OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m pytest -q -rs
# 557 passed, 19 skipped, 3 warnings in 62.35s
```

`tests/test_layout_v3.py` is a small `torch.profiler` probe that checks the
one-time eval-cache materialization, the warm forward clone signature, and the
generate sampler contiguous shapes. It makes no GPU timing, CUDA/Triton
execution, kernel correctness, or speedup claim.

### Non-goals

- No default eager, cache, generate, or attention math change.
- No unsafe layout flip, sampler rewrite, custom RNG path, or channels-last path.
- No GPU claims and no repository/public-PR scope beyond the private target.

## opt/profile-v15 — CPU re-profile after #152/#154 (2026-09-19)

**Branch:** `opt/profile-v15` (private `katulevskiy/bdh-gpu-opt` only; no
public PR).
**Base tip:** `e1248ce` (`main`, after #156 layout-v3; includes #152
generate-copy ceiling and #154 rope-fuse-v3). This profile is CPU-only and
makes no GPU claim.

### Method

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
/workspace/bdh-gpu-opt/.venv/bin/python benchmarks/profile_forward.py \
  --device cpu --mode all
# torch 2.14.0+cu130  cuda=False  device=cpu
# cfg: layers=4 d=128 nh=4 B=4 T=128; generate prompt=16 / new=32
```

The harness uses two warmups, one wait, and three active steps. Percentages are
self-CPU percentages from the three active steps; counts are aggregate counts
followed by the per-active-call count. `cat` and `contiguous` were absent from
all three CPU traces (zero calls).

### CPU profile highlights (self CPU)

| Mode | Top self-CPU operators | copy_ / cat / contiguous |
|------|-------------------------|---------------------------|
| Attention | `aten::bmm` **31.99%**, `aten::mul` **22.22%**, `aten::complex` **19.12%**, `aten::copy_` **8.87%**, `aten::add` **7.14%**, `aten::sub` **5.44%** | `copy_` **6 / 3 = 2 per call**; `cat=0`; `contiguous=0` |
| Forward | `aten::bmm` **29.05%**, `aten::mm` **23.15%**, `aten::mul` **15.99%**, `aten::complex` **11.42%**, `aten::copy_` **7.05%**, `aten::clamp_min_` **2.79%** | `copy_` **36 / 3 = 12 per call**; `cat=0`; `contiguous=0` |
| Generate | `aten::mm` **21.81%**, `aten::bmm` **14.15%**, `aten::mul` **3.05%**, `aten::matmul` **2.40%**, `aten::native_layer_norm` **2.36%**, `aten::einsum` **2.04%**, `aten::copy_` **0.86%** | `copy_` **1,182 / 3 = 394 per call**; `cat=0`; `contiguous=0` |

The tip preserves raw scores × strict `tril(diagonal=-1)` attention: no
softmax, scale, or SDPA. The observed copy counts remain at the warmed
CPU-build floor from profile-v14, with no `cat` or `contiguous` calls. These
CPU percentages and operator counts are not GPU performance measurements.

### Verdict / non-goals

- CPU profile recorded after #152/#154 on the post-#156 main tip.
- No GPU timing, kernel-on-hardware result, correctness, or speedup claim.
- No attention-semantic change, public PR, or PR to `pathwaycom/*`.

## opt/cache-bench-v2 — report packed-cache footprint per page (2026-09-19)

**Branch:** `opt/cache-bench-v2` (private `katulevskiy/bdh-gpu-opt` only).
**Base tip:** `86315f0` (`main`, after #152 / #157).

### Goal

Deepen the post-#85 CPU-only cache-page bench without changing cache or
`generate` defaults. The page sweep now reports the packed `CacheManager`
capacity transition (`initial→final`) and final packed KR+V allocation in KiB,
alongside geometric-vs-linear grow/copy counts. This makes a small page and a
large page directly comparable without inferring the allocation from `max_seq`
and dtype.

### What changed

- `benchmarks/bench_cache_page.py`: `_run_policy()` now captures initial and
  final capacity, live bytes, and `CacheManager.bytes_allocated`; the sweep
  prints `capacity` and `alloc_KiB` for each page size.
- The benchmark asserts that the geometric and linear A/B policies reach the
  same final packed capacity and allocation; only the grow/copy policy differs.
- No model, attention, page-growth, or default behavior changed. The linear
  policy remains a benchmark-only monkeypatch.

```bash
OMP_NUM_THREADS=2 .venv/bin/python benchmarks/bench_cache_page.py --smoke
OMP_NUM_THREADS=2 .venv/bin/python benchmarks/bench_cache_page.py \
  --max-seq 2048 --pages 8,16,32,64,128,256
```

### CPU evidence

The smoke/sweep reports remain CPU accounting only. `alloc_KiB` is the final
packed KR+V tensor footprint for the fp32 benchmark configuration; it is not a
GPU memory measurement or a claim about attention/generate wall time. Defaults
remain unchanged (`page_size=None` / `cache_page_size=None` still preallocate
`max_seq`).

### Non-goals

- No GPU/CUDA claims or measurements
- No softmax / scale / SDPA; raw scores × `tril(diagonal=-1)` preserved
- No default change and no generate-path change
- No PRs to `pathwaycom/*`; private repo only

## opt/profile-v16 — CPU re-profile after #159/#160 (2026-09-19)

**Branch:** `opt/profile-v16` (private `katulevskiy/bdh-gpu-opt` only; no
public PR).
**Base tip:** `717c38e` (`main`, after #159 zerograd logger fallback and #160
cache-bench-v2; includes #158 docs and #157 profile-v15). This profile is
CPU-only and makes no GPU claim.

### Method

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
/workspace/bdh-gpu-opt/.venv/bin/python benchmarks/profile_forward.py \
  --device cpu --mode all --warmup 2 --wait 1 --active 3
# torch 2.14.0+cu130  cuda=False  device=cpu
# cfg: layers=4 d=128 nh=4 B=4 T=128; generate prompt=16 / new=32
```

The harness uses two warmups, one wait, and three active steps, matching
profile-v15. Percentages below are self-CPU percentages from those three
active steps; counts are aggregate counts followed by the per-active-call
count. `cat` and `contiguous` were absent from all three CPU traces (zero
calls).

### CPU profile highlights (self CPU)

| Mode | Top self-CPU operators | copy_ / cat / contiguous |
|------|-------------------------|---------------------------|
| Attention | `aten::bmm` **29.74%**, `aten::mul` **22.64%**, `aten::complex` **16.18%**, `aten::copy_` **11.32%**, `aten::add` **7.60%**, `aten::sub` **6.09%** | `copy_` **6 / 3 = 2 per call**; `cat=0`; `contiguous=0` |
| Forward | `aten::bmm` **28.15%**, `aten::mm` **23.15%**, `aten::mul` **15.20%**, `aten::complex` **12.13%**, `aten::copy_` **7.77%**, `aten::clamp_min_` **2.97%** | `copy_` **36 / 3 = 12 per call**; `cat=0`; `contiguous=0` |
| Generate | `aten::mm` **21.68%**, `aten::bmm` **14.24%**, `aten::mul` **3.10%**, `aten::matmul` **2.37%**, `aten::native_layer_norm` **2.34%**, `aten::einsum` **2.00%**, `aten::copy_` **0.87%** | `copy_` **1,182 / 3 = 394 per call**; `cat=0`; `contiguous=0` |

The tip preserves raw scores × strict `tril(diagonal=-1)` attention: no
softmax, scale, or SDPA. Relative to profile-v15, all copy/cat/contiguous
counts are unchanged. The self-CPU mix moves modestly within this short CPU
trace (attention bmm 31.99%→29.74%, forward bmm 29.05%→28.15%, generate mm
21.81%→21.68%), so the honest result is **flat versus v15**, not a speedup
claim. Absolute profiler milliseconds are omitted because they are CPU-box
and profiler dependent; these are not GPU performance measurements.

### Verdict / non-goals

- CPU profile recorded after #159/#160 on the `717c38e` main tip.
- Defaults remain unchanged; no attention semantic change was made.
- No GPU timing, kernel-on-hardware result, correctness, or speedup claim.
- P0 remains real GPU measurement, including cold CUDA/Triton validation.
- No PRs to `pathwaycom/*`; private repository only.

## opt/decode-gemm-v3 — preserve packed T=1 decode views (2026-09-19)

**Base tip:** `7a9b14f` (main after #162).

The Triton T=1 decode launcher now makes its packed-layout contract explicit: it
tries a no-copy `view` when flattening batch/head dimensions and keeps the prior
`reshape` fallback only for layouts that cannot be viewed directly. This preserves
capacity-strided `CacheManager` KR/V views without an unconditional recontiguous
step. CPU coverage now runs the fake-launcher stride contract even without Triton
and checks blocked/online/triton fallback parity for B=1 and B=2 exactly at and
just above the decode oneshot score budget.

Attention remains raw score × `tril(diagonal=-1)` × V with no softmax or scale;
eager remains the default. Validation is CPU-only and reports no GPU timing or
GPU win. P0 remains real CUDA/Triton measurement.
## opt/scorev-fuse-v4 — B>1 shared-V decode epilogue (2026-09-19)

**Branch:** `opt/scorev-fuse-v4` on the private repository. This is a small
follow-up to the shared-V decode work and keeps eager defaults unchanged.

### Change

The CPU-safe blocked decode helper now handles accumulated B>1 shared-V tiles
by reshaping the score/output rows to `(B, H*Bi, *)` and using `baddbmm(...,
out=...)`. The `(B,1,S,D)` cache view stays unexpanded, and beta=1 tiles no
longer allocate a separate score×V product before accumulation. The beta=0
oneshot case keeps the existing broadcast `matmul(..., out=...)` path.

The grad-enabled path still avoids `out=` operators and uses the existing
matmul-plus-add fallback, so autograd remains graph-safe. Raw scores × strict
`tril(diagonal=-1)` semantics and eager defaults are unchanged.

### CPU validation

```bash
pytest -q tests/test_scorev_fuse_v4.py
pytest -q tests/test_fuse_scorev.py tests/test_inc_decode.py
```

These tests are CPU parity/graph-safety checks only. No GPU timing or hardware
correctness claim is made.
## opt/compile-train-v4 — first-probe fallback diagnostics (2026-09-19)

**Branch:** `opt/compile-train-v4` (private `katulevskiy/bdh-gpu-opt` only;
no public PR). **Base:** `7719e9a` (#164).

`train.maybe_compile()` now distinguishes an unprobed compile from a failed
first probe, and the `train_bwd` missing-target fallback states the exact
remediation: pass `example_y` or select the explicit forward-only `train`
probe. First-probe failures identify the original eager fallback and point to
probe inputs/backend checks. Defaults remain `BDH_COMPILE=0`,
`BDH_COMPILE_PROBE=train_bwd`, `IMPL=eager`; no attention math or GPU claim
changed.

CPU validation:

```bash
.venv/bin/python -m pytest tests/test_compile.py -q
OMP_NUM_THREADS=2 .venv/bin/python -m pytest tests/ -q
```

## RoPE GPU scaffold v3 (2026-09-19)

- Added `triton_rope_skip_reason()` and `backend_info()["triton_skip_reason"]` so CPU/CUDA/Triton gate decisions are explicit instead of silent fallback.
- Added CPU contracts for T=1 paired rotation into a non-contiguous cache-slot-like `out=` buffer and for the existing T>1 tile parity path.
- Tests remain CPU-only on this box; no fused-GPU timing or speedup claim is made. Defaults and strict `tril(-1)` attention semantics are unchanged.
## opt/prefetch-h2d-v3 — deepen CPU no-op and CUDA lifetime contract (2026-09-19)

**Branch:** `opt/prefetch-h2d-v3` (private `katulevskiy/bdh-gpu-opt` only;
no public PR).
**Base tip:** `fb698fc` (after #165).

### Audit / deepen

The H2D opt-in remains unchanged and CPU-safe. `BatchPrefetcher` coverage now
runs both async and sync host modes across unset, disabled, and enabled H2D
settings. On the CPU path, the test forbids construction of CUDA stream/event
objects, asserts that no device lookahead is retained, and checks that the
caller transfer preserves tensor identity. The CUDA-side staging doc now makes
the event/lifetime handoff explicit: the staged tuple retains pinned host
sources until the recorded event is consumed.

### CPU-safe validation

```text
/workspace/bdh-gpu-opt/.venv/bin/python -m pytest tests/test_dataloader.py -q
# 20 passed, 1 skipped in 2.34s

OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
/workspace/bdh-gpu-opt/.venv/bin/python -m pytest -q
# 571 passed, 19 skipped, 3 warnings in 89.39s
```

This CPU-only box does not exercise CUDA H2D overlap or provide timing data;
no GPU correctness, throughput, overlap, or speedup claim is made. Defaults
remain unchanged (`BDH_PREFETCH_ASYNC=1`, `BDH_PREFETCH_H2D=1`).

### Non-goals

- No default prefetch or H2D flag changes.
- No GPU timing or overlap claims.
- No public PR.

## opt/online-decode-v4 — retain shared-V packed views (2026-09-19)

**Branch:** `opt/online-decode-v4` on the private repository. **Base tip:**
`62acaa7` (#169).

The opt-in blocked/online T=1 shared-V decode now keeps a non-flattenable
`(B,H,S,N)` key view on the 4-D CPU score path instead of forcing a
`B*H` reshape copy. The reused `(B,S,D)` shared-V view, direct inference
accumulation, and grad-enabled fallback remain unchanged. Raw scores ×
`tril(diagonal=-1)` semantics, eager defaults, and cat-free generate remain
unmodified.

CPU validation covers B>1 long-S tiles, capacity-strided cache parity,
non-flattenable key views, autograd, incremental decode, and generate. This
box has no GPU; no GPU timing, correctness, or win claim is made.


## opt/attn-bwd-v3 — deepen CPU analytic parity matrix (2026-09-19)

**Branch:** `opt/attn-bwd-v3` (private `katulevskiy/bdh-gpu-opt` only).
**Base tip:** `1d12071` (#170).

The post-#130 CPU contract now crosses `eager|blocked|online|triton|cuda`
with `AUTOGRAD` off/on across shared-V and full-head-V layouts, a non-tile
sequence shape, and randomized output gradients. The CUDA-only harness now
labels expected native `triton`/`cuda` + `AUTOGRAD=0` skips explicitly instead
of printing only the raw autograd exception.

```text
python -m pytest tests/test_attn_bwd.py tests/test_attn_bwd_bench.py -q
python benchmarks/bench_attn_bwd.py
# CPU: SKIP: CUDA unavailable; ... (exit 0)
```

No GPU timing or training claim is made on this CPU-only box. Defaults remain
`BDH_ATTN_IMPL=eager`, `BDH_ATTN_AUTOGRAD=0`, and raw scores ×
`tril(diagonal=-1)`.
## opt/cuda-cold-v5 — explicit CPU-only extension smoke contract (2026-09-19)

**Branch:** `opt/cuda-cold-v5` (private `katulevskiy/bdh-gpu-opt` only; no
public PR and no PRs to `pathwaycom/*`).
**Base tip:** `f95c143` (main after #173; rebased from the requested #170 tip
`1d12071`).

### Audit / deepen

The CUDA cold path keeps the #146 actionable native skip diagnostics and strict
raw score × `tril(diagonal=-1)` CPU mirrors. Build smoke now also covers the
explicit `BDH_FORCE_CPU_EXT=1` branch: setup selects the CPU-only native
extension configuration without entering CUDA setup, while the default remains
pure Python and missing-`nvcc` CUDA requests remain a clear no-op. No CUDA
compiler, hardware execution, timing, or speedup is claimed on this CPU-only
box.

### CPU-safe validation

```text
/workspace/bdh-gpu-opt/.venv/bin/python -m pytest tests/test_cuda_build.py tests/test_cuda_attn.py -q -rs
# 21 passed, 5 skipped

OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
/workspace/bdh-gpu-opt/.venv/bin/python -m pytest -q -rs
# 575 passed, 19 skipped, 3 warnings
```

### Non-goals

- No default eager or AUTO behavior change.
- No softmax, scaling, diagonal inclusion, or change to raw score × strict
  `tril(diagonal=-1)` semantics.
- No GPU claim, timing, or public PR.
## opt/blocked-tile-v4 — deepen CPU wide-head cold parity (2026-09-19)

**Branch:** `opt/blocked-tile-v4` on the private repository. **Base tip:**
`bc3967b` (#174).

The blocked cold-path parity matrix now includes a wider, batched shape at the
128-row tile boundary: `B=2, H=3, T=257, N=160, D=192`. Both shared-V
(`value_heads=1`) and head-matched-V (`value_heads=3`) layouts are compared
with the eager raw-score reference, including the partial final tile and exact
zero output at position 0.

Validation is CPU-only. Defaults remain eager, attention remains raw
`(Q @ K.T).tril(diagonal=-1) @ V`, and no GPU timing or win claim is made.


## opt/triton-cold-v5 — CPU-safe skip marker and long-T fallback parity (2026-09-19)

**Branch:** `opt/triton-cold-v5` on private `katulevskiy/bdh-gpu-opt`.
**Base tip:** `4e13111` (`main`, post-#175).

Cold Triton diagnostics now append the explicit `CPU-safe skip` marker to both
the import-gate and CUDA-availability reasons, matching the native cold-path
diagnostic contract. A focused CPU matrix also covers the long-T (`T=257`)
Triton fallback against blocked and eager strict raw score ×
`tril(diagonal=-1)` references for both shared-V and per-head-V layouts.

### CPU-safe validation

```text
python -m pytest tests/test_triton_cold_v5.py tests/test_triton_attn.py -q -rs
# 39 passed, 3 skipped

OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m pytest -q -rs
# 593 passed, 19 skipped, 3 warnings
```

No CUDA allocation, kernel launch, timing, or GPU claim is added. Defaults stay
eager and `BDH_ATTN_AUTO` remains opt-in.
