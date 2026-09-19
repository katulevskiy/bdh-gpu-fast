# OPT status — landed work (#1–#114)

Private sandbox only: [`katulevskiy/bdh-gpu-opt`](https://github.com/katulevskiy/bdh-gpu-opt).
**Do not** open PRs against `pathwaycom/bdh` or any `pathwaycom/*` repo.

Tip documented here: `f24a5d8` (`#114` opt/cuda-cold-v3 / `#113` docs-matrix-v25 / `#112` profile-v11 / `#111` docs-matrix-v24 / `#110` gen-vcopy-v1 / `#109` docs-matrix-v23 / `#108` triton-cold-v3 / `#107` docs-matrix-v22 / `#106` docs-matrix-v22 / `#105` opt/ln-resid-v2 / `#104` profile-v10 / `#103` docs-matrix-v21 / `#102` gen-copy-tax-v1 / `#101` docs-matrix-v20 / `#100` scorev-fuse-v2 / `#99` docs-matrix-v19 / `#98` profile-v9 / `#97` docs-v18 / `#96` attn-bwd-gpu-scaffold / `#95` copy-tax-v1 / `#94` docs through #93 / `#93` blocked-tile-v2 / `#92` prefetch-h2d / `#91` docs through #90 / `#90` rope-fuse-v2 / `#89` profile-v8 / `#88` docs through #87 / `#87` zero-grad harden / `#86` docs through #84 / `#85` cache-page-bench / `#84` compile-fullgraph / `#83` docs matrix / `#82` triton-cold-v2 / `#81` docs align / `#80` profile-v7 / `#79` cuda-cold-v2; `#77` auto-tune / `#75` prefill-blocked). Profile-v11 source is `4963b0f` (`#110`); profile-v10 source is `1363794` (`#102`), and the current code tip is `f24a5d8`. Default eager remains unchanged; #92 is CPU no-op, #93 is an opt-in CPU cold-path win at T≥256, #95/profile-v9 report CPU-only copy-call evidence (isolated forward `copy_`=18; warmed harness 16/call; generate remains copy_-heavy), #100 deepens blocked/online CPU score×V epilogues plus T=1 decode tiling, #102 introduced and #104 re-profiled the complex-view copy for RoPE pairs plus sampled `idx_out` into preallocated `out.narrow`, with generate `copy_` ~558→~398 and `cat` still 0, #105 confirms the existing eager residual-LN path is already the safe deepen, #108 keeps wide-head Triton cold tiles bounded while preserving shared CPU V as a view in the blocked fallback, and #110 cuts generate `copy_` ~398→~394 by using `_store_pairs` for T>1 RoPE plus a top-k `gather(out=)` path. The #110 probe ceiling is the remaining V-slot writes (~132/gen) and multinomial internals (~128/gen), which are not safely removable without changing cache-snapshot layout or RNG behavior. #109 and #111 are docs-only; #112 re-profiles the #110 tip with warmed forward `copy_`=12/call, isolated `copy_`=14, generate `copy_`=394/call, and `cat`/`contiguous`=0. #113 is docs-only; #114 bounds wide-head CUDA cold tiles with the #108-aligned 64×64 guard, keeps CPU parity, clean CUDA skips, and a C++20 build smoke. No GPU speedup evidence was added, so GPU measurement and cold CUDA/Triton validation remain open.
Detail / benches: [`OPT_NOTES.md`](OPT_NOTES.md). Ranked remaining: [`OPT_BACKLOG.md`](OPT_BACKLOG.md).

Hard constraint (all opts): attention stays **raw scores** × **strict lower-triangular**
`tril(diagonal=-1)` — **no softmax**, **no `1/√d`**, **no**
`F.scaled_dot_product_attention`.

---

## Honest CPU vs GPU (read this first)

| Claim | Status on this sandbox |
|-------|------------------------|
| Device | **CPU-only** (`torch 2.14.0+cu130`, `cuda=False`) |
| Algorithmic wins measured on CPU | **Yes** — KV / CacheManager generate (~2.5–3×), vectorized `get_batch`, cat-free decode path, compile-friendly structure |
| Kernel wins (Triton / CUDA fused score×V, fused RoPE, AMP train throughput) | **Not measured** — scaffolds + harnesses in-tree; GPU benches skip cleanly |
| Default train / attn path | Still **eager** + **fp32** — opt-in env flags only |
| CPU `blocked` / `triton` (→blocked) attn | Mid-S often **slower**; **long S** decode/generate can beat eager (#55); keep for peak-mem / long-S, not default |
| CPU `BDH_COMPILE=1` | **Only with `IMPL=eager`**: ~1.5× warm train-step (#46: 5.25 vs 7.76 ms). `COMPILE=1`+blocked|online|triton = measured regression (~70–100×); `maybe_compile` warns. GPU compile still open |
| CPU AMP (`BDH_AMP_DTYPE`) | Correctness smoke (#24+#54); often **slower** than fp32 on CPU; **throughput claim = GPU-only** |
| Sparse ReLU | **Default OFF**; short-train densifies but not to paper ~5%; CPU sparse≪dense |

**Rule:** never cite CPU profiler absolute ms or CPU microbench medians as GPU speedups.
Re-run on A100/H100 via `benchmarks/bench_gpu_attn.py` before claiming kernel wins.

---

## Env flags (operator-facing)

### `BDH_ATTN_IMPL` — attention backend (default `eager`)

| Value | Cold / prefill | T=1 decode vs packed KR/V | Notes |
|-------|----------------|---------------------------|-------|
| `eager` | Full `T×T` then `tril_(diagonal=-1)` | `_two_gemm_decode` (Tq=1 BH-bmm / 4D @) | **Default**; reference math |
| `blocked` | Online / tiled fused score×V (no full `T×T`) | Online tiled decode (tight oneshot; peak ~Tq×tile) | Lower peak; #100 deepens no-grad score×V epilogues and T=1 tiling; long-S CPU wall can beat eager |
| `triton` | Triton fused on CUDA; else blocked | Triton decode + `V_BROADCAST`; else blocked | Needs CUDA + Triton to run kernel |
| `cuda` | Native ext if built (`BDH_BUILD_EXT=1`), else PyTorch ref | `tril_decode` Tq=1 + adaptive `DECODE_TILE_N` (v3) | Deepened tiles; GPU measure open |

Also: `BDH_ATTN_AUTOGRAD=1` → `StrictTrilAttnFn` analytic Q/K/V backward (opt-in; #7/#39/#41). Default **off**. With `IMPL=blocked|online|triton|cuda`, bwd is **tiled** (no full T×T); eager keeps dense M-recompute. T=1 CacheManager decode / `generate` stay on the decode path.

### `BDH_ATTN_AUTO` — opt-in long-S backend (default **off**)

When set (`1|true|yes|on`) and `BDH_ATTN_IMPL` is `eager`:
- **T=1 decode** → **triton** (CUDA+Triton) else **blocked** once
  `past_len > BDH_ATTN_AUTO_THRESHOLD` (default **512**, retained after
  `#55`/`#72`/`#73`/`#75`/`#77` — CPU wall crossover ~≥512).
- **Cold / prefill** → same prefer rule once
  `T > BDH_ATTN_AUTO_COLD_THRESHOLD` (unset → mirrors `AUTO_THRESHOLD`).
  Lower cold thr (e.g. 256) only when mid-T **peak-score** budget matters more
  than wall (`#73`/`#77`: T∈{128,256} peak↓ wall↑).

Explicit `IMPL∈{blocked,online,triton,cuda}` is never overridden. Default
(AUTO unset) = eager cold and decode at every length.

```bash
export BDH_ATTN_IMPL=eager     # default
export BDH_ATTN_IMPL=blocked
export BDH_ATTN_IMPL=triton
export BDH_ATTN_IMPL=cuda

# opt-in long-T cold + long-S decode → blocked|triton (#75/#77/#82)
export BDH_ATTN_AUTO=1
export BDH_ATTN_AUTO_THRESHOLD=512
# optional independent cold gate (unset → same as THRESHOLD):
# export BDH_ATTN_AUTO_COLD_THRESHOLD=256   # peak-mem mid-T; accept wall loss
```

### `BDH_ROPE_IMPL` — RoPE rotate (default `eager`)

| Value | Behavior |
|-------|----------|
| `eager` | Historical strided even/odd rotate (bit-identical default) |
| `fused` | Pair-contiguous PyTorch; Triton on CUDA when usable |

Cos/sin **tables** are cached in `Attention` regardless (#18). This flag only picks how cos/sin apply to `v` (#29).

```bash
export BDH_ROPE_IMPL=eager   # default
export BDH_ROPE_IMPL=fused
```

### `BDH_COMPILE` — `torch.compile` (default `0` / off)

| Env | Default | Meaning |
|-----|---------|---------|
| `BDH_COMPILE` | `0` | Set `1` / `true` to enable `train.maybe_compile` |
| `BDH_COMPILE_MODE` | `default` | `default` \| `reduce-overhead` \| `max-autotune` |
| `BDH_COMPILE_PROBE` | `train_bwd` | `eval` \| `train` \| `train_bwd` |
| `BDH_COMPILE_FULLGRAPH` | `0` | `fullgraph=True` when set; soft-fallback to eager on graph breaks (#84 probe: cold eager×AUTOGRAD holds 0 breaks on tip) |

`generate()` is `@torch.compiler.disable`. Changing `BDH_ATTN_IMPL` after compile → recompile. On CPU, `reduce-overhead` is **not useful** — CUDA graphs need a real GPU (`maybe_compile` warns; prefer `MODE=default`).

**Operator guidance (CPU, after #46 / #63 / #84):** recommend `BDH_COMPILE=1` **only** with
`BDH_ATTN_IMPL=eager` and `BDH_COMPILE_MODE=default` (optionally `BDH_ATTN_AUTOGRAD=1` and/or `BDH_COMPILE_FULLGRAPH=1` — tip cold path
still 0 Dynamo graph breaks; FULLGRAPH soft-falls back if Unsupported). `maybe_compile` logs a clear warning if `COMPILE=1` with
`IMPL∈{blocked,online,triton}` (measured CPU regression) **or** `MODE=reduce-overhead` on non-CUDA.
Defaults unchanged (`COMPILE=0`, `MODE=default`, `FULLGRAPH=0`, `IMPL=eager`). **GPU** inductor / CUDA graphs still unmeasured.

```bash
BDH_COMPILE=0 python train.py
# recommended CPU compile train path:
BDH_COMPILE=1 BDH_ATTN_IMPL=eager BDH_COMPILE_MODE=default BDH_COMPILE_PROBE=train_bwd python train.py
# optional analytic bwd (still 0 graph breaks on eager):
BDH_COMPILE=1 BDH_ATTN_IMPL=eager BDH_ATTN_AUTOGRAD=1 python train.py
# optional fullgraph (soft-fallback if Dynamo breaks; tip holds):
BDH_COMPILE=1 BDH_ATTN_IMPL=eager BDH_COMPILE_FULLGRAPH=1 python train.py
# warns (CPU regression) — do not use as default train:
BDH_COMPILE=1 BDH_ATTN_IMPL=blocked python train.py
# warns on CPU (no CUDA graphs) — prefer default; try on GPU:
BDH_COMPILE=1 BDH_COMPILE_MODE=reduce-overhead python train.py
```

### `BDH_PREFETCH_ASYNC` — host batch prefetch (default `1`)

| Env | Default | Meaning |
|-----|---------|---------|
| `BDH_PREFETCH_ASYNC` | `1` | `BatchPrefetcher` uses a daemon host producer and depth-1 queue; set `0` for synchronous preload / A-B |

Async mode overlaps the next numpy host gather (and CUDA pinning) with
`train_step`; CUDA H2D is issued on the caller side stream. Public `get_batch`
semantics are unchanged. Real gather is already ~0.07 ms on this CPU box, so
e2e CPU gains are usually small/noisy; GPU pin/H2D overlap remains unmeasured.

```bash
BDH_PREFETCH_ASYNC=1 python train.py  # default
BDH_PREFETCH_ASYNC=0 python train.py  # sync debug / A-B
```

### `BDH_PREFETCH_H2D` — CUDA side-stream H2D lookahead (default `1`)

| Env | Default | Meaning |
|-----|---------|---------|
| `BDH_PREFETCH_H2D` | `1` | On CUDA, stage one pinned batch ahead on a dedicated side stream and hand it to the caller stream via an event; `0` keeps H2D on the caller stream |

This flag is a CPU no-op. It applies to the async host-prefetch path;
`BDH_PREFETCH_ASYNC=0` remains the synchronous debug/A-B mode. GPU H2D
overlap and throughput are unmeasured. The `cuda_staging=` constructor override
is available for tests/A-B and is ignored on CPU.

```bash
BDH_PREFETCH_H2D=1 python train.py  # default on CUDA
BDH_PREFETCH_H2D=0 python train.py  # caller-stream H2D
```

### `BDH_AMP_DTYPE` — train autocast (default `float32` / unset)

| Value | Autocast | GradScaler |
|-------|----------|------------|
| `float32` / `fp32` / `off` / unset | Off | Off |
| `bfloat16` / `bf16` | Yes | **Never** |
| `float16` / `fp16` / `half` | Yes | Only if **CUDA** |

Optional `BDH_AMP_FORWARD_ONLY=1` (#54): autocast **logits only**; CE in fp32 outside.
Default `0` keeps `model(x,y)` under the same autocast ctx.

CPU AMP is for parity smoke, not speed (`amp_throughput_claim_device() → none` here).
**AMP helps on GPU** (Tensor Cores / HBM); never cite CPU medians as speedups.
Decode path AMP (older #5) is separate from this train knob (#24+#54).

```bash
BDH_AMP_DTYPE=float32 python train.py
BDH_AMP_DTYPE=bfloat16 python train.py   # needs bf16 support on device
BDH_AMP_DTYPE=float16 python train.py    # GradScaler only on CUDA
BDH_AMP_FORWARD_ONLY=1 BDH_AMP_DTYPE=bfloat16 python train.py  # CE in fp32
BDH_BENCH_AMP=1 python benchmarks/bench_train_step.py          # honest A/B
```

---

## Landed opts (#1–#114)

| # | Branch / title | What landed | CPU | GPU |
|---|----------------|-------------|-----|-----|
| **1** | `opt/kernels` | KV-style cache + incremental `generate`; RoPE without `stack→view`; MLP `permute→contiguous→view`; preserve `tril(-1)` | Generate ~2.5× vs full recompute; forward ~1.3× (layout/RoPE) | Kernel rewrite N/A yet |
| **2** | `opt/triton-attn` | Triton + blocked pure-PyTorch strict-tril score@V; `BDH_ATTN_IMPL` | Blocked ≪ eager wall; Triton→blocked | Triton in-tree, unexecuted here |
| **3** | `opt/cuda-ext` | CUDA/C++ scaffold + CPU ref for tril score×V | CPU ref correct | Ext optional build; no GPU run |
| **4** | `opt/profiler` | Operator CPU profiler + `OPT_BACKLOG.md` | Profiles this box | CUDA activities when available |
| **5** | `opt/decode-amp` | CacheManager decode path + optional AMP on generate | Cache path solid | AMP train throughput open |
| **6** | `opt/qkv-fuse` | Shared RoPE cis; fuse rope `out=`; inplace ReLU proj | Fewer allocs / noise-level | — |
| **7** | `opt/attn-bwd` | Analytic `StrictTrilAttnFn` (`BDH_ATTN_AUTOGRAD`) | Gradcheck / fp64 match | — |
| **8** | `opt/attn-unify` | Unified `BDH_ATTN_IMPL` eager\|blocked\|triton\|cuda | Cold+decode dispatch | CUDA path scaffold |
| **9** | `opt/inc-decode` | Blocked/Triton T=1 decode vs packed KR/V | ≡ eager last row | GPU decode bench open |
| **10** | `opt/compile-train` | Fused AdamW, `set_to_none`, prefetch, compile path | Train microbench fused path | — |
| **11** | `opt/train-fuse` | Opt-in `BDH_COMPILE` (default 0); defer loss `.item()` | Sync-light logging | — |
| **12** | `opt/ln-fuse` | Residual LN reuse + encoder proj helpers | Small LN slice win | — |
| **13** | `opt/weight-layout` | Contiguous `(B,T,nh,N)` encoder; decoder view + `F.linear` | ~noise vs tip | Layout for GPU GEMM |
| **14** | `opt/dataloader` | Pin / non_blocking + optional DataLoader workers | Host gather ≪ model | H2D wins need CUDA |
| **15** | `opt/cuda-decode` | CUDA scaffold T=1 decode vs packed KR/V | CPU ref always | Ext optional |
| **16** | `opt/embed-tie` | Contiguous vocab proj + optional `tie_weights` | Layout tidy | — |
| **17** | `opt/dropout-fuse` | `F.dropout` + identity `p=0` for compile | Inductor-friendly | — |
| **18** | `opt/rope-cache` | Cache RoPE cos/sin by `(T, head_dim, device, dtype)` | Fewer phase rebuilds | — |
| **19** | `opt/triton-decode2` | Polish blocked/Triton decode vs packed KR/V | Parity vs eager | GPU open |
| **20** | `opt/cache-v2` | CacheManager deepen — generate **0× `aten::cat`** | Cat tax gone | Keep cat-free on GPU |
| **21** | `opt/fuse-scorev` | Online fused strict-tril score×V (`blocked`/`online`) | CPU slower; lower peak mem | **P0** GPU measure |
| **22** | `opt/compile-harden` | Train probe, graph-break docs, CPU inductor parity | Compile↔eager @ dropout=0 | CUDA graphs unmeasured |
| **23** | `opt/profile-v2` | Re-profile tip; refresh backlog post-#19–#22 | Docs only | Points GPU P0s |
| **24** | `opt/bf16-train` | Opt-in `BDH_AMP_DTYPE` + GradScaler fp16+CUDA | CPU AMP smoke | Train bench open |
| **25** | `opt/triton-cold` | Better cold Triton tiles + less staging; deepened by triton-cold-v2 | →blocked on CPU | **P0** GPU validate |
| **26** | `opt/gpu-bench` | `benchmarks/bench_gpu_attn.py` + backlog runbook | Clean skip if no CUDA | Harness ready |
| **27** | `opt/cuda-cold` | Tiled online cold CUDA tril score×V (no global `T×T`); deepened by cuda-cold-v2 | Ref path | **P0** GPU measure |
| **28** | `opt/decode-copy` | `CacheManager.reserve` + in-place RoPE; fewer `copy_` | `copy_` calls cut ~2× | Remaining: decode GEMM |
| **29** | `opt/rope-fuse` | Fused RoPE rotate (`BDH_ROPE_IMPL`) | Default eager; fused PT | Triton GPU open |
| **30** | `opt/ln-compile` | `F.layer_norm` residual; fewer Dynamo specializations | Compile-friendlier | — |
| **31** | `opt/compile-bench` | Honest CPU `BDH_COMPILE=0` vs `1` train-step harness | Warm ~1.5× tiny cfg | GPU A/B = P1 |
| **32** | `opt/sparse-probe` | Short-train ReLU density + CPU sparse crossover | Density↓ but sparse≪dense; **default OFF** | Sparse GPU speculative |
| **33** | `opt/decode-gemm` | T=1 decode score×V polish (blocked/Triton/CUDA) | ≡ eager; ~parity wall | `--mode decode` GPU open |
| **34** | `opt/docs-matrix` | `OPT_STATUS.md` (#1–#33) + README pointer | Docs only | Points GPU P0s |
| **35** | `opt/gen-bench` | `benchmarks/bench_generate.py` CacheManager × attn impls | CPU ~noise; `aten::cat=0`; tokens≡eager | E2E GPU = P1 |
| **36** | `opt/mlp-fuse` | Fused bias+ReLU + MLP merge without contiguous copies | Forward/generate **`aten::contiguous`=0**; wall ~noise | Epilogue hygiene for compile/GPU |
| **37** | `opt/cuda-ref-v2` | Deepen CUDA attn CPU refs + build smoke | Soft-skip without ext | **P0** GPU measure still open |
| **38** | `opt/blocked-vec` | Vectorized CPU blocked/online tril tiles | ~18–36× vs old blocked wall; still < eager | Default eager unchanged |
| **39** | `opt/attn-bwd-train` | Analytic `StrictTrilAttnFn` first-class train path (`BDH_ATTN_AUTOGRAD`); cold+multi-token wiring; `bench_attn_bwd.py` | Grad parity @ dropout=0; CPU tiny train_step ~noise | GPU train unmeasured |
| **40** | `opt/profile-v3` | Re-profiled post-#36 through #39; refreshed `OPT_STATUS` / `OPT_BACKLOG` / `OPT_NOTES` | Docs/profile only; generate `aten::cat`=0, forward `aten::contiguous`=0 | No GPU measurements; defaults unchanged |
| **41** | `opt/blocked-autograd` | Blocked/online forward + **tiled** analytic backward under `BDH_ATTN_AUTOGRAD=1`; `online` aliases to `blocked` | CPU grad parity; tiny train-step ~1.9× vs AUTOGRAD-off in one honest run | GPU train A/B open |
| **42** | `opt/prefetch-v2` | Host-thread depth-1 batch prefetch, numpy producer gather, optional CUDA pin/H2D overlap; `BDH_PREFETCH_ASYNC=0` sync A/B | Synthetic overlap ~1.7×; real CPU e2e ~noise/modest | GPU pin/H2D overlap unmeasured |
| **43** | `opt/docs-matrix-v2` | Refreshed `OPT_STATUS` / `OPT_BACKLOG` through #40 and README pointer | Docs only | — |
| **44** | `opt/gen-host` | Warm full-span RoPE table; hoist dispatch/sampling; `inference_mode`; preserve cat-free generate and default eager math | CPU generate ~52.34→39.84 ms (~1.31×); `arange` 33→1; tokens match | No GPU measurement |
| **45** | `opt/docs-matrix-v3` | Refresh OPT matrix / backlog through #44 | Docs only | — |
| **46** | `opt/compile-blocked` | Compile×blocked×AUTOGRAD train matrix; `StrictTrilSelfAttnFn` Dynamo fix | COMPILE+eager ~1.5×; COMPILE+blocked ~70–100× slower; 0 graph breaks | GPU open |
| **47** | `opt/encoder-fuse` | Hybrid encoder: default einsum; optional bias → `F.linear` epilogue | Always-on linear lost on CPU; hybrid landed | GPU linear epilogue open |
| **48** | `opt/gen-sample` | Fuse T=1 lm_head+sample / optional fused top-k; reuse logits/probs buffers | Large-V top_k ~1.5×; default-V wall ~noise; tokens-match | GPU decode open |
| **49** | `opt/compile-guidance` | Warn COMPILE+blocked/online/triton; document eager compile path | Advisory warn; defaults unchanged | GPU still open |
| **50** | `opt/profile-v4` | Re-profile tip after gen-sample #48 (+ #49 guidance); refresh `OPT_NOTES` / `OPT_BACKLOG` / `OPT_STATUS` tip SHAs | Docs/profile only; `aten::cat`=0; `aten::contiguous`=0; gen-host host self ~1.4% | No GPU measurements; defaults unchanged |
| **51** | `opt/ln-deepen` | Residual LN: `F.layer_norm` + inner-out `add_` reuse (cut add temp) | Residual micro small; e2e ~noise; Dynamo 0 breaks | No fused GPU LN claim |
| **52** | `opt/docs-matrix` (docs refresh) | Refresh OPT matrix / backlog through #51 | Docs only | — |
| **53** | `opt/decode-mm` | T=1 decode deepen (`_two_gemm_decode`, CUDA Tq=1 + `DECODE_TILE_N`); B=1 lm_head `mv` | ≡ eager last row; cats=0; CPU wall ~noise | GPU `--mode decode` open |
| **54** | `opt/amp-deepen` | Harden opt-in AMP: fp16 CPU smoke, `BDH_AMP_FORWARD_ONLY`, honest AMP train_step bench, GPU-only claim | CPU smoke + bench; often ≲/≳ fp32 | GPU train AMP still open |
| **55** | `opt/decode-online-v2` | Deepen blocked/online T=1 decode (`_DECODE_ONESHOT_ELEMS`, peak helper) | ≡ eager; cats=0; long-S peak↓ + wall↑ | GPU `--mode decode` open |
| **56** | `opt/attn-auto` | Opt-in `BDH_ATTN_AUTO` long-S T=1 decode → blocked (thr=512); cold deepened in #75 | default eager; AUTO parity; cats=0 | GPU threshold re-tune open |
| **57** | `opt/cache-page` | Geometric CacheManager page growth + empty+prefix `copy_`; `ensure_capacity`; long-S grow stats | fewer grows/bytes vs linear; cats=0; defaults unchanged | GPU long-S peak still open |
| **58** | `opt/layout-v2` | Eval cached contiguous encoder `(nh*N,D)` + `F.linear`; train einsum; compile traces einsum | T=1/32 eval ~1.3–1.4× vs train path; T=128 ~noise; gen uses cache | GPU layout still open |
| **59** | `opt/profile-v5` | Re-profile tip after #55–#58; refresh `OPT_NOTES` / `OPT_BACKLOG` / `OPT_STATUS` tip SHAs | Docs/profile only; `aten::cat`=0; `aten::contiguous`=0; #58 layout visible on generate | No GPU measurements; defaults unchanged |
| **60** | `opt/docs-matrix-v5` | Refresh optimization matrix through #59; update documented tip/profile metadata | Docs only | — |
| **61** | `opt/fix-gen-host-test` | Repair environ-hoist assertion after #56 `BDH_ATTN_AUTO` decode probes; generate semantics unchanged | 372 passed, 9 skipped; test-only | — |
| **62** | `opt/triton-decode-v3` | Deepen Triton T=1 decode scaffold (long-S tiles, Q-hoist); AUTO→triton when CUDA else #55 blocked | ≡ eager/blocked on CPU; CUDA tests skip; default eager | GPU `--mode decode` open |
| **63** | `opt/compile-reduce` | Document/measure `BDH_COMPILE_MODE=reduce-overhead` vs `default` on CPU; warn no CUDA graphs | CPU MODE A/B in `bench_train_step`; reduce-overhead not useful on CPU | GPU CUDA graphs still P1 |
| **64** | `opt/cuda-decode-v3` | Deepen CUDA T=1 decode tiles (adaptive DECODE_TILE_N 32/64/128 + TQ1 Q-hoist); ≡ blocked parity | ≡ eager/blocked; soft-skip GPU; default eager | GPU `--mode decode` open |
| **65** | `opt/docs-matrix-v6` | Refresh optimization matrix through #64; update documented tip/profile metadata | Docs only | — |
| **66** | `opt/log-sync` | Further cut train logging host sync (`TrainLossLogger`; CUDA deferred D2H; `BDH_LOG_FREQ`/`BDH_LOG_ASYNC`) | CPU: sync path tested; e2e ~noise | CUDA defer unmeasured |
| **67** | `opt/docs-matrix-v7` | Refresh OPT matrix / backlog through #66; documented tip/profile metadata | Docs only | — |
| **68** | `opt/profile-v6` | Re-profile tip after #64–#66; refresh `OPT_NOTES` / `OPT_BACKLOG` / `OPT_STATUS` tip SHAs | Docs/profile only; `aten::cat`=0; `aten::contiguous`=0; #64–#66 off short default window | No GPU measurements; defaults unchanged |
| **69** | `opt/rope-decode` | Deepen T=1 RoPE apply (`rope_rotate_t1`); table pair cache + last-pos cis reuse; keep `BDH_ROPE_IMPL` default eager | t1≡strided; CPU wall ~0.83× (no win claim) | GPU fused/Triton T=1 open |
| **70** | `opt/dropout-compile` | Harden dropout=0 / eval identity for compile; FX/Dynamo tests; COMPILE=1 dropout A/B bench | Identity + 0 Dynamo breaks; CPU dropout A/B ~noise | GPU compile still P1 |
| **71** | `opt/docs-matrix` (through #69) | Refresh optimization matrix through #69 (`962a3b6`) | Docs only | — |
| **72** | `opt/gen-long-bench` | `bench_generate.py --mode auto-ab`: long-S generate AUTO 0/1 (S∈{256,1024,2048}); validate #55/#56 outside microbench | AUTO fires; match; cats=0; e2e then deepened by #75 | GPU thr re-tune open |
| **73** | `opt/attn-mem-probe` | CPU peak-mem probe eager vs blocked vs online (`bench_attn_mem.py`) | Mid-T: peak↓ wall↑; long-T both; default eager | No GPU claims |
| **74** | `opt/docs-matrix` (through #73) | Refresh optimization matrix through #73 (`5d63bc2`) | Docs only | — |
| **75** | `opt/prefill-blocked` | Deepen blocked/online cold (adaptive BS@T≥256); AUTO long-T cold+decode | AUTO e2e 1.26×@1024 / 1.39×@2048; cats=0; default eager | GPU thr re-tune open |
| **76** | `opt/docs-matrix-v10` | Refresh `OPT_STATUS.md` / `OPT_BACKLOG.md` through #75; update documented tip metadata | Docs only | — |
| **77** | `opt/auto-tune` | Keep shared thr=512; add `BDH_ATTN_AUTO_COLD_THRESHOLD`; operator recs from #72/#73/#75 | short AUTO A/B 1.25×@1024; default AUTO off | GPU thr re-check open |
| **78** | `opt/docs-matrix-v11` | Refresh optimization matrix through #77 | Docs only | — |
| **79** | `opt/cuda-cold-v2` | Deepen CUDA cold tiles (adaptive TILE_M/N long-T; pair #75); CPU refs ≡ blocked/eager | ≡ eager/blocked; soft-skip GPU; default eager | GPU `--mode cold` open |
| **80** | `opt/profile-v7` | Re-profile tip after #75–#77 (+ #79 on tip); refresh `OPT_NOTES` / `OPT_BACKLOG` / `OPT_STATUS` tip SHAs | Docs/profile only; `aten::cat`=0; `aten::contiguous`=0; #69–#79 off short default window | No GPU measurements; defaults unchanged |
| **81** | docs align (#80 tip) | Align optimization docs with #80 tip | Docs only | — |
| **82** | `opt/triton-cold-v2` | Deepen Triton cold tiles (adaptive BLOCK_M/N @T≥256; pair #75/#79); CPU→blocked adaptive; AUTO docs | ≡ eager/blocked; soft-skip GPU; default eager | GPU `--mode cold` open |
| **83** | docs matrix (#82 tip) | Refresh optimization docs through #82 | Docs only | — |
| **84** | `opt/compile-fullgraph` | Probe `BDH_COMPILE_FULLGRAPH=1` eager×AUTOGRAD; graph-break docs; soft-skip | 0 breaks cold eager; bench matrix; soft-fallback | GPU inductor still P1 |
| **85** | `opt/cache-page-bench` | CacheManager geometric-vs-linear page-growth microbench; pointer from `bench_cache_mem.py`; documented tables | CPU: 128×→4× fewer bytes copied across page sizes 8→256; generate smoke `aten::cat=0`, tokens match | No GPU claims |
| **86** | `docs/opt-status-backlog-v84` | Refresh `OPT_STATUS.md` / `OPT_BACKLOG.md` through #84 | Docs only | — |
| **87** | `opt/zerograd` | Harden `zero_grad(set_to_none=True)` and fused AdamW via `clear_grads()` chokepoint; compile `train_bwd` probe; `tests/test_zerograd.py` | CPU re-smoke: fused+set-to-none ~1.03× vs legacy; set-to-none alone ~1.05× vs fill | GPU train measurement open; defaults unchanged (`BDH_FUSED_ADAMW=1`, `BDH_COMPILE=0`) |
| **88** | `opt/docs-matrix-v15` | Refresh `OPT_STATUS.md` / `OPT_BACKLOG.md` through #87 | Docs only | — |
| **89** | `opt/profile-v8` | Re-profile the post-#85–#87 tip; refresh profile/status/backlog evidence and keep the GPU blocker explicit | Docs/profile only; `aten::cat`=0 and `aten::contiguous`=0 on the short window | No GPU measurements; defaults unchanged |
| **90** | `opt/rope-fuse-v2` | Deepen fused RoPE and T=1 pair apply: pair-contiguous stores, no expand/stack allocs, cached table-pair path, blocked Triton CPU fallback/scaffold, and parity tests | Fused T>1 rotate ~1.4× in the honest CPU microbench; T=1 parity; no default change | GPU Triton/CUDA validation open; default `BDH_ROPE_IMPL=eager` |
| **91** | `opt/docs-matrix-v16` | Refresh `OPT_STATUS.md` / `OPT_BACKLOG.md` through #90 | Docs only | — |
| **92** | `opt/prefetch-h2d` | Opt-in CUDA pinned-batch, side-stream H2D lookahead with event handoff; CPU path is a no-op; `BDH_PREFETCH_H2D=1` default | CPU smoke only; no throughput claim | GPU H2D overlap unmeasured; default CPU behavior unchanged |
| **93** | `opt/blocked-tile-v2` | Flatten long CPU cold blocked/online `(B,H)` heads once and use dense `bmm` score and score×V tiles; preserve strict `tril(-1)` and default eager | Blocked wins at T≥256: 1.25× @256, 4.00× @512, 5.87× @1024 in the honest single-thread bench; short T still loses | CUDA path and GPU speedup unmeasured; default eager unchanged |
| **94** | `opt/docs-matrix-v17` | Refresh `OPT_STATUS.md` / `OPT_BACKLOG.md` through #93 | Docs only | — |
| **95** | `opt/copy-tax-v1` | Allocate contiguous QR for RoPE on the encoder permute view; use in-place `tril_` on fresh score buffers; reduce avoidable forward `aten::copy_` 24→18 and QR contig copies 8→0 | CPU call-count deltas only; correctness/parity tests pass; no wall-speed or GPU claim | GPU copy/layout impact unmeasured; defaults `BDH_ATTN_IMPL`/`BDH_ROPE_IMPL` unchanged |
| **96** | `opt/attn-bwd-gpu-scaffold` | Add CUDA analytic-attention train harness with parity checks, dtype/config controls, and clean CPU skip | CPU skip contract; no GPU timing here | Real CUDA validation remains open; defaults unchanged |
| **97** | `opt/docs-matrix-v18` | Refresh optimization matrix through #95/#96 | Docs only | — |
| **98** | `opt/profile-v9` | Re-profile the #95 copy-tax-v1 tip after #96/#97; record isolated and warmed CPU operator counts without overclaiming GPU impact | Isolated one-forward `aten::copy_`=18 (reproduces 24→18); warmed harness 16/call; `aten::cat`=0 and `aten::contiguous`=0; generate `copy_` remains heavy | No GPU timing or speedup; GPU validation remains open |
| **99** | `opt/docs-matrix-v19` | Refresh `OPT_STATUS.md` / `OPT_BACKLOG.md` through #98 | Docs only | — |
| **100** | `opt/scorev-fuse-v2` | Direct `baddbmm(..., out=target)` score×V epilogues for CPU inference/no-grad blocked/online tiles; reuse the flattened T=1 output and tighten the decode oneshot budget to 1024; default eager unchanged | Correctness pass (490 passed, 18 skipped); CPU cold/decode behavior remains shape-dependent, with lower score peak and fewer inference epilogue buffers | No GPU timing or speedup; Triton/CUDA validation remains open |
| **101** | `opt/docs-matrix-v20` | Refresh `OPT_STATUS.md` / `OPT_BACKLOG.md` through #100 | Docs only | — |
| **102** | `opt/gen-copy-tax-v1` | Use a complex-view copy for fp32 RoPE pair stores; sample into preallocated `out.narrow` via `idx_out`; keep defaults unchanged and generate cat-free | CPU generate `aten::copy_` ~558→~398; `aten::cat`=0; parity tests pass | No GPU timing; copy/layout impact unmeasured; GPU validation remains open |
| **103** | `opt/docs-matrix-v21` | Refresh `OPT_STATUS.md` / `OPT_BACKLOG.md` through #102 | Docs only | — |
| **104** | `opt/profile-v10` | Re-profile the #102 gen-copy-tax-v1 tip; record warmed and isolated CPU operator counts without overclaiming GPU impact | Generate `aten::copy_`=398/call (1,194/3); isolated forward `copy_`=18; `aten::cat`=0 and `aten::contiguous`=0 | No GPU timing or speedup; GPU validation remains open |
| **105** | `opt/ln-resid-v2` | Audit remaining residual LayerNorm temporaries; confirm the existing safe eager deepen and avoid an unsafe default change | CPU probe: exactly 2× `aten::native_layer_norm` + 1× `aten::add_`; no residual `add`, `copy_`, `cat`, `to`, or `_to_copy`; dtype probes show no explicit bounce | No GPU timing or speedup; defaults unchanged |
| **106** | `opt/docs-matrix-v22` | Refresh `OPT_STATUS.md` / `OPT_BACKLOG.md` through #104 | Docs only | — |
| **107** | `opt/docs-matrix-v22` | Record the #105 residual-LN audit in `OPT_STATUS.md` / `OPT_BACKLOG.md` | Docs only | — |
| **108** | `opt/triton-cold-v3` | Bound wide-head Triton cold query/key tiles at 64×64 while preserving explicit overrides; keep shared CPU `V=(B,1,T,D)` as a `(B,T,D)` view in the flattened blocked score×V fallback | CPU parity: 45 passed, 3 skipped; no GPU timing or kernel claim; defaults eager and AUTO-off unchanged | GPU/Triton validation remains open |
| **109** | `opt/docs-matrix-v23` | Refresh `OPT_STATUS.md` / `OPT_BACKLOG.md` through #108 | Docs only | — |
| **110** | `opt/gen-vcopy-v1` | Probe remaining generate `aten::copy_`; use `_store_pairs` for T>1 RoPE and `gather(out=)` for top-k without changing defaults or RNG behavior | CPU generate `aten::copy_` ~398→~394, `aten::cat`=0; V-slot (~132/gen) and multinomial (~128/gen) copies remain the safe-removal ceiling | No GPU timing or speedup; GPU measurement remains open |
| **111** | `opt/docs-matrix-v24` | Refresh `OPT_STATUS.md` / `OPT_BACKLOG.md` through #110 | Docs only | — |
| **112** | `opt/profile-v11` | Re-profile the #110 gen-vcopy tip; record warmed and isolated CPU operator counts without overclaiming GPU impact | CPU generate `aten::copy_`=394/call (1,182/3; isolated 396), forward `copy_`=12/call warmed and 14 isolated; `aten::cat`=0 and `aten::contiguous`=0 | No GPU timing or speedup; GPU validation remains open |
| **113** | `opt/docs-matrix-v25` | Refresh `OPT_STATUS.md` / `OPT_BACKLOG.md` through #112 and carry profile-v11 evidence forward | Docs only | — |
| **114** | `opt/cuda-cold-v3` | Align wide-head CUDA cold tile bounds with Triton #108; retain CPU reference parity, clean CUDA skip coverage, and C++20 build smoke | CPU parity; no GPU timing | **P0** GPU measure / cold CUDA validation |


Related early landings without a #1–#33 slot (still on main, documented in notes):

- **`opt/sparse-relu`** — experimental `bdh_sparse.py` helpers; **default OFF**
- **`opt/cache-pack`** — preallocated KR/V CacheManager (superseded/deepened by #20)

---

## Defaults vs opt-in (cheat sheet)

| Knob | Production-safe default | When to flip |
|------|-------------------------|--------------|
| `BDH_ATTN_IMPL` | `eager` | GPU after microbench win; or `blocked` for peak-mem experiments |
| `BDH_ATTN_AUTOGRAD` | off / unset | `1` when training with non-eager attn; blocked/online → tiled analytic bwd |
| `BDH_ATTN_AUTO` | off / unset | `1` for long-T cold + long-S decode→triton (CUDA) or blocked (#55/#75/#77); `THRESHOLD` default 512; optional `COLD_THRESHOLD` (#auto-tune) |
| `BDH_ROPE_IMPL` | `eager` | `fused` after GPU RoPE bench |
| `BDH_COMPILE` / `MODE` / `FULLGRAPH` | `0` / `default` / `0` | CPU: `COMPILE=1` **only with `IMPL=eager`** + `MODE=default` (#46/#49/#63); optional `FULLGRAPH=1` (#84; soft-fallback); `reduce-overhead` needs GPU CUDA graphs; GPU inductor still open |
| `BDH_PREFETCH_ASYNC` | `1` | `0` for synchronous preload / A-B; GPU pin/H2D overlap still needs measurement |
| `BDH_PREFETCH_H2D` | `1` on CUDA | `0` to keep H2D on the caller stream; CPU no-op; GPU overlap still needs measurement |
| `BDH_AMP_DTYPE` | `float32` | `bf16`/`fp16` on **CUDA** train boxes (CPU = smoke only) |
| `BDH_AMP_FORWARD_ONLY` | `0` | `1` for logits-only autocast + fp32 CE |
| Sparse ReLU | OFF | Only if density + GPU sparse kernel win |

---

## Quick re-run

```bash
cd /workspace/bdh-gpu-opt   # or this worktree
source .venv/bin/activate
python -m pytest tests/ -q
python benchmarks/profile_forward.py --mode all
BDH_BENCH_COMPILE=1 BDH_BENCH_COMPILE_MODE=1 BDH_BENCH_COMPILE_FULLGRAPH=1 BDH_BENCH_AMP=1 python benchmarks/bench_train_step.py
python benchmarks/bench_sparse_probe.py
python benchmarks/bench_attn_mem.py --smoke
python benchmarks/bench_cache_page.py --smoke
# on a CUDA box:
python benchmarks/bench_gpu_attn.py
python benchmarks/bench_gpu_attn.py --mode decode --T 512
```

---

## Explicit non-goals

- Softmax / diagonal inclusion / SDPA drop-in
- PRs or pushes to `pathwaycom/*`
- Claiming GPU speedups from CPU medians or profiler-inflated absolute times
- Defaulting `BDH_ATTN_IMPL=blocked` on CPU
- Recommending `BDH_COMPILE=1` with `blocked`/`online`/`triton` on CPU (warns; #46/#49)
- Recommending `BDH_COMPILE_MODE=reduce-overhead` on CPU (no CUDA graphs; #63)
- Re-introducing `aten::cat` in packed `generate`
- Wiring sparse ReLU into default `BDH.forward` without a measured win
