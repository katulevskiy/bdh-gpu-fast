# OPT backlog — ranked remaining work

Private sandbox only (`katulevskiy/bdh-gpu-opt`). **Do not** PR to `pathwaycom/bdh`.

Constraint (hard): attention stays **raw scores** × **strict lower-triangular**
`tril(diagonal=-1)` — **no softmax**, **no `1/√d`**, **no**
`F.scaled_dot_product_attention`.

Profile source: `benchmarks/profile_forward.py` on CPU
(`torch 2.14.0+cu130`, `cuda=False`), tip `b160469`, cfg `layers=4 d=128 nh=4 B=4 T=128`,
generate prompt=16 / new=32. Absolute ms are **profiler-inflated**; use **%
self CPU** and call counts. Re-run on GPU before claiming kernel wins.

Post-#36/#39 re-profile (`opt/profile-v3`): generate still **0× `aten::cat`**; forward **0× `aten::contiguous`** (mlp-fuse); default eager still pays full T×T `bmm`+`tril`. See `OPT_NOTES.md` § opt/profile-v3.

Post-#19–#22 re-profile (`opt/profile-v2`): generate **0× `aten::cat`**; default
eager still pays full T×T `bmm`+`tril`. See `OPT_NOTES.md` § opt/profile-v2.

## Already landed (main)

- Analytic attn train path (`BDH_ATTN_AUTOGRAD` / `StrictTrilAttnFn`) — **landed** `opt/attn-bwd-train` (#39): cold+multi-token wiring, `bench_attn_bwd.py`, grad parity @ dropout=0
- Blocked/online + **tiled** analytic bwd train (`opt/blocked-autograd`): no full T×T in fwd or bwd; `online` alias; GPU train A/B still open

- RoPE without `stack→view`; skip redundant dtype casts
- RoPE cos/sin table cache by (T, head_dim, device, dtype) (`opt/rope-cache` #18)
- KV-style cache + incremental `generate`
- MLP `permute → contiguous → view`
- Vectorized `train.get_batch` (+ pin/non_blocking CUDA; optional DataLoader workers #14)
- Triton + blocked pure-PyTorch attn dispatch (`BDH_ATTN_IMPL`) — CPU blocked still < eager after `opt/blocked-vec`; GPU unmeasured
- Experimental sparse ReLU matmul (default off); **short-train density + CPU crossover** (`opt/sparse-probe`) — keep OFF
- CUDA extension scaffold for tril score×V (optional build)
- Weight layout: `(B,T,nh,N)` encoder einsum, decoder view+`F.linear`, optional bias fuse (#13)
- Contiguous vocab proj + optional `tie_weights` (`opt/embed-tie` #16)
- `F.dropout` + identity `p=0` for compile (`opt/dropout-fuse` #17)
- CUDA/C++ decode vs packed KR/V under `BDH_ATTN_IMPL=cuda` (scaffold; CPU ref always) (#15)
- Blocked/Triton decode polish vs packed KR/V (`opt/triton-decode2` #19)
- CacheManager v2: layer-contiguous KR/V, page growth, generate **0× aten::cat** (`opt/cache-v2` #20)
- Decode-copy: `reserve` + in-place RoPE into packed KR; `narrow` past views; no intermediate `.to()` on `copy_` (`opt/decode-copy`)
- Decode GEMM polish: blocked/Triton/CUDA T=1 score×V vs packed KR/V — broadcast-V, Triton staging, tiled CUDA (`opt/decode-gemm`)
- Online fused strict-tril score×V (no full T×T) under `blocked`/`online` (`opt/fuse-scorev` #21)
- Vectorized CPU blocked/online tiles (`opt/blocked-vec`) — beats old Python-row blocked wall; still < eager on CPU
- GPU attn microbench harness `benchmarks/bench_gpu_attn.py` (eager|blocked|online|triton|cuda; clean CPU skip) (`opt/gpu-bench`)
- Cold CUDA tril score×V **tiled online** scaffold (no global T×T; `opt/cuda-cold`) — GPU measure still P0
- `torch.compile` harden: train probe, graph-break docs, CPU inductor parity (`opt/compile-harden` #22)
- Compile-friendly LN+residual: `F.layer_norm` only, functional residual, no `is_grad_enabled` product branch (`opt/ln-compile`)
- CPU train-step `BDH_COMPILE=0` vs `1` microbench via `maybe_compile` (soft-skip if inductor missing) (`opt/compile-bench`)
- Generate microbench `benchmarks/bench_generate.py`: CacheManager generate × attn impls eager|blocked|triton|cuda when avail; honest CPU numbers (`opt/gen-bench`)
- OPT_STATUS operator matrix + README pointer (`opt/docs-matrix` #34)
- MLP fuse: bias+ReLU + merge without unconditional `.contiguous()` (`opt/mlp-fuse` #36); profile-v3: `aten::contiguous`=0
- CUDA attn CPU refs + build smoke deepen (`opt/cuda-ref-v2` #37)

## Ranked next work

| P | Item | Why (from profile / notes) | Target | Risk |
|---|------|----------------------------|--------|------|
| **P0** | **Measure Triton/CUDA fused tril-score×V on real GPU** | Default **eager** still (post-#36 profile-v3): attn self `bmm` ~16–32% + RoPE `mul`/`copy_` + `tril`; forward `bmm` ~15–28% + `copy_` ~18–25% + `mm` ~10–18%.
| **P0** | **Cold Triton tile/staging validation** | **Landed `opt/triton-cold`:** adaptive power-of-2 tiles, fused strict-tril score×V, and broadcast-V staging; GPU validation remains open. | A100/H100 microbench; bit-identical | Env blocker |
| **P1** | **`torch.compile` GPU train-step next** | Forward still `copy_` ~20%, `mm` ~12%, `mul`/`mul_` ~12%, LN ~4%. **CPU** `BDH_COMPILE=0` vs `1` train-step bench landed (`opt/compile-bench`, soft-skip if inductor missing). **GPU inductor / CUDA graphs still unmeasured**. | A100/H100: `BDH_COMPILE=0` vs `1` (+ `reduce-overhead`) via `benchmarks/bench_train_step.py` | Low |
| **P1** | **Decode GEMM / copy tax on generate** | Generate (profile-v3): `bmm` ~14–44%, `copy_` ~6–11% (post-#20 **cats=0**); |blocked|triton|cuda; honest CPU ~noise, `aten::cat=0`, tokens match eager. Default still eager. | A100/H100: `bench_generate.py` + `bench_gpu_attn.py --mode decode`; keep cat-free | Medium |
| **P2** | **Fused RoPE kernel** | Attn: `mul`/`copy_` from strided rotate. **Cached tables** (#18); **fused rotate landed** `opt/rope-fuse` (`BDH_ROPE_IMPL`). | GPU Triton microbench still open | Low–medium |
| **P2** | **Sparsity follow-through** | **Density measured** `opt/sparse-probe`: short-train x~27% xy~12% @150 steps (≫ paper 5%); CPU sparse **never reliably beat dense** → **keep OFF**. GPU sparse still open. | GPU sparse bench if density ≪10% | Speculative |
| **P2** | **Memory layout** | Forward contiguous/clone copies significant. | **Landed `opt/weight-layout`** + **`opt/mlp-fuse`** — `(B,T,nh,N)` encoder, free decoder view, fused bias+ReLU / `_mlp_merge`; embed-tie #16; CPU wall ~noise vs tip | Low |
| **P3** | **Hardware / dtype** | **Landed `opt/bf16-train`:** opt-in `BDH_AMP_DTYPE` + GradScaler fp16+CUDA only; CPU parity tests. GPU train bench still open. | GPU box microbench | Env |
| **P3** | **Profiler CI artifact** | Traces gitignored; optional nightly upload. | CI | N/A |

### Done (struck from ranked queue)

| Was | Status |
|-----|--------|
| Memory layout / mlp-fuse | **Landed** #13+#16+#36 — profile-v3: forward **`aten::contiguous`=0**; CPU ~noise |
| Generate × attn impls microbench | **Landed** #35 `opt/gen-bench` — GPU e2e still P1 |
| OPT_STATUS docs matrix | **Landed** #34 `opt/docs-matrix` |
| CUDA attn CPU refs deepen | **Landed** #37 `opt/cuda-ref-v2` |
| Vectorized CPU blocked tiles | **Landed** #38 `opt/blocked-vec` — still < eager; GPU measure open |
| Re-profile post-mlp-fuse | **Landed** `opt/profile-v3` (this PR) — tip `b160469`; cats=0; contiguous=0 |
| Analytic tril attn train path | **Landed** #39 `opt/attn-bwd-train` — default AUTOGRAD off; eager profile unchanged |
| Blocked/online tiled analytic bwd | **This PR** `opt/blocked-autograd` — blocked|online+AUTOGRAD=1; dense M-recompute only for eager |
| Cache packing / fewer cats | **Done** #19–#20 — generate `aten::cat` **0** (was ~10% self / ~864 calls pre-pack) |
| Fuse score×V epilogue (no materialize T×T) | **Landed** #21; **CPU vectorized** `opt/blocked-vec` (~18–36× vs old blocked wall; still slower than eager) |
| `torch.compile` / inductor CPU harden | **Landed** #17+#22; CPU 0-vs-1 train bench `opt/compile-bench`; remaining = **GPU** measure (P1) |
| Fused RoPE rotate (`BDH_ROPE_IMPL`) | **Landed** `opt/rope-fuse` — default eager; fused PyTorch + optional Triton |
| Decode GEMM vs packed KR/V | **Landed** `opt/decode-gemm` — blocked/triton/cuda decode polish; GPU measure still open |
| Memory layout / embed path | **Landed** #12–#13+#16 |

| **P2** | **Analytic attn train on GPU** | CPU blocked|online + tiled analytic bwd landed (`opt/blocked-autograd`). **GPU** train-step with `IMPL=blocked|triton|cuda` + AUTOGRAD=1 unmeasured. | A100/H100 `bench_attn_bwd.py` | Low |

## Explicit non-goals

- Softmax / diagonal inclusion
- PRs to `pathwaycom/*`
- Claiming GPU speedups from CPU profiler absolute times
- Defaulting `BDH_ATTN_IMPL=blocked` on CPU (still slower than eager after `opt/blocked-vec`; use for peak-score memory / parity)
- Re-introducing `aten::cat` in `generate` / packed cache path

## Suggested order

1. GPU: bench eager vs blocked/online vs Triton vs CUDA fused score×V (cold + T=1 decode)
2. GPU: end-to-end `bench_generate.py` (CacheManager) across attn impls + `bench_gpu_attn.py --mode decode`
3. GPU: `BDH_COMPILE=0` vs `1` train-step (CPU harness ready in `bench_train_step.py`; try `reduce-overhead` on CUDA)
4. ~~Cache preallocate / cat-free generate~~ (**done** `opt/cache-v2` #20)
5. ~~RoPE fuse~~ (**done** `opt/rope-fuse`) / ~~sparsity density CPU~~ (`opt/sparse-probe`) / dtype

```bash
python benchmarks/profile_forward.py --mode all
```

## GPU microbench (A100 / H100)

Harness skips cleanly when `torch.cuda.is_available()` is false (this sandbox).
On a CUDA box, compare **eager | blocked | online | triton | cuda** and print
bit-identical / `allclose@1e-4` vs eager:

```bash
# cold-path tril(diagonal=-1) score×V — default shapes B=2 H=4 T=128 N=64 D=128
python benchmarks/bench_gpu_attn.py

# longer seq / more heads (typical A100/H100 sweep)
python benchmarks/bench_gpu_attn.py --B 4 --H 8 --T 512 --N 64 --D 128 --warmup 20 --iters 100

# optional dtype sweep
python benchmarks/bench_gpu_attn.py --dtype bfloat16
python benchmarks/bench_gpu_attn.py --dtype float16

# optional native CUDA extension (otherwise `cuda` backend uses pure-PyTorch ref)
BDH_BUILD_EXT=1 BDH_BUILD_CUDA=1 pip install -e . --no-build-isolation
python benchmarks/bench_gpu_attn.py
```

Record: GPU name, torch/CUDA versions, median ms per backend, `bit_identical`
and `max|Δ|` vs eager. **Do not** claim wins from CPU medians; keep default
`BDH_ATTN_IMPL=eager` until GPU data lands. Private repo only — not pathwaycom.

## GPU next step — `BDH_COMPILE` train-step (after CPU harness)

CPU sandbox now has an **honest** A/B in `benchmarks/bench_train_step.py`
(`BDH_COMPILE=0` vs `1` via `train.maybe_compile`; soft-skip if inductor/CXX
missing). **Do not** treat CPU medians as GPU wins.

On an A100/H100 box:

```bash
# eager vs compiled train-step (default mode=default, probe=train)
BDH_BENCH_COMPILE=1 python benchmarks/bench_train_step.py

# CUDA-graph-oriented (static B×T; see train_fast.py)
BDH_BENCH_COMPILE=1 BDH_COMPILE_MODE=reduce-overhead python benchmarks/bench_train_step.py
```

Record: GPU name, torch/CUDA, median ms for `BDH_COMPILE=0` and `=1`, whether
`_orig_mod` stuck (true compile) vs soft fallback, and `reduce-overhead` note.
Private repo only — never `pathwaycom/*`.

## Generate microbench — CacheManager × attn impls (`opt/gen-bench`)

End-to-end `BDH.generate` (packed KR/V CacheManager, cat-free) across
`BDH_ATTN_IMPL=eager|blocked|triton|cuda`. Labels **effective** backend
(triton→blocked on CPU; cuda→`cuda_ref` without native ext). Prints median ms,
tok/s, tokens-match-eager, `aten::cat` count.

```bash
python benchmarks/bench_generate.py
python benchmarks/bench_generate.py --prompt 32 --new 64 --warmup 2 --iters 5
# A100/H100:
python benchmarks/bench_generate.py --device cuda --warmup 5 --iters 20
```

**CPU honesty (this box):** medians ~noise vs eager; all match eager; `aten::cat=0`.
**Do not** claim Triton/CUDA wins from CPU. GPU e2e + decode kernel measure = P1.
Private repo only — never `pathwaycom/*`.
