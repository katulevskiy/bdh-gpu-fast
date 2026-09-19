# OPT status — landed work (#1–#273; #160 docs scope retained)

Private sandbox only: [`katulevskiy/bdh-gpu-opt`](https://github.com/katulevskiy/bdh-gpu-opt).
**Do not** open PRs against `pathwaycom/bdh` or any `pathwaycom/*` repo.

Tip pointer: `3b3c082` (#273 blocked-tile raw-score contract, current tip) follows #272 sparse guardrail terminal coverage (`b9bf6c8`), #271 run-level GPU skip status (`75597c9`), #270 explicit prefetch H2D opt-out coverage (`dae2e07`), #268 tiled packed per-head decode gradients (`aebc449`), #266 strict-tril backward boundary queries (`cbf0831`), #265 CUDA_PATH missing-nvcc skip contract (`6d6aaab`), and profile-v20 (`f4c7cab`). Profile-v20 remains the matched CPU-only re-profile: attention `copy_`=2/call, forward `copy_`=12/call, generate `copy_`=394/call, with `cat=0` and `contiguous=0`; #270–#273 add only CPU-safe contract/skip coverage. None adds CUDA timing or GPU speedup evidence. Real GPU measurement remains the P0 blocker and cold CUDA/Triton validation remains open.
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
| `blocked` | Online / tiled fused score×V (no full `T×T`) | Online tiled decode (tight oneshot; peak ~Tq×tile) | Lower peak; #100/#116/#135 deepen no-grad score×V epilogues and shared-V T=1 tiling; long-S CPU wall can beat eager |
| `triton` | Triton fused on CUDA; else blocked | Triton decode + `V_BROADCAST`; else blocked | Needs CUDA + Triton to run kernel |
| `cuda` | Native ext if built (`BDH_BUILD_EXT=1`), else PyTorch ref | `tril_decode` Tq=1 + adaptive `DECODE_TILE_N` (v3) | Deepened tiles; GPU measure open |

Also: `BDH_ATTN_AUTOGRAD=1` → `StrictTrilAttnFn` analytic Q/K/V backward (opt-in; #7/#39/#41). Default **off**. With `IMPL=blocked|online|triton|cuda`, bwd is **tiled** (no full T×T); eager keeps dense M-recompute. T=1 CacheManager decode / `generate` stay on the decode path.

### `BDH_ATTN_AUTO` — opt-in long-S backend (default **off**)

When set (`1|true|yes|on`) and `BDH_ATTN_IMPL` is `eager`:
- **T=1 decode** → **triton** (CUDA+Triton) else **blocked** once
  `past_len > BDH_ATTN_AUTO_THRESHOLD` (default **512**, retained after
  `#55`/`#72`/`#73`/`#75`/`#77`/`#120` — CPU wall crossover ~≥512).
- **Cold / prefill** → same prefer rule once
  `T > BDH_ATTN_AUTO_COLD_THRESHOLD` (unset → mirrors `AUTO_THRESHOLD`).
  Lower cold thr (e.g. 256) only when mid-T **peak-score** budget matters more
  than wall (`#73`/`#77`/`#120`: T∈{128,256} peak↓ wall↑).

Explicit `IMPL∈{blocked,online,triton,cuda}` is never overridden. Default
(AUTO unset) = eager cold and decode at every length.

```bash
export BDH_ATTN_IMPL=eager     # default
export BDH_ATTN_IMPL=blocked
export BDH_ATTN_IMPL=triton
export BDH_ATTN_IMPL=cuda

# opt-in long-T cold + long-S decode → blocked|triton (#75/#77/#82/#120)
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

Cos/sin **tables** are cached in `Attention` regardless (#18). This flag only picks how cos/sin apply to `v` (#29). On the #138 tip, Triton is selected only when `v`, cis, and optional `out` share the same CUDA device and use the conservative fp16/bf16/fp32 set; unsupported or mixed inputs skip to the existing PyTorch/blocked fallback, and `backend_info(cuda)` skips cleanly without initializing an unavailable runtime.

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

**Operator guidance (CPU, after #46 / #63 / #84 / #117 / #136):** recommend `BDH_COMPILE=1` **only** with
`BDH_ATTN_IMPL=eager` and `BDH_COMPILE_MODE=default` (optionally `BDH_ATTN_AUTOGRAD=1` and/or `BDH_COMPILE_FULLGRAPH=1` — tip cold path
still 0 Dynamo graph breaks; FULLGRAPH soft-falls back if Unsupported). `maybe_compile` logs a clear warning if `COMPILE=1` with
`IMPL∈{blocked,online,triton}` (measured CPU regression) **or** `MODE=reduce-overhead` on non-CUDA.
Defaults unchanged (`COMPILE=0`, `MODE=default`, `PROBE=train_bwd`, `FULLGRAPH=0`, `IMPL=eager`). `train`/`eval` remain diagnostic probes; missing `example_y` soft-falls to eager. **GPU** inductor / CUDA graphs still unmeasured.

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

This flag is a CPU no-op. #139 makes that contract explicit, and #203 extends it
to an unavailable CUDA runtime: the device/runtime gate runs before any CUDA
stream/event construction, CPU reports `device-not-cuda: cpu`, and a requested
CUDA device without a live runtime reports `CUDA unavailable:
torch.cuda.is_available() is false`. No staged device lookahead is created on
CPU or unavailable-CUDA paths, and `_to_device` preserves the original CPU
tensor objects. It applies to the async host-prefetch path;
`BDH_PREFETCH_ASYNC=0` remains the synchronous debug/A-B mode. GPU H2D
overlap and throughput are unmeasured. The `cuda_staging=` constructor override
is available for tests/A-B and is ignored on CPU or unavailable-CUDA paths.

```bash
BDH_PREFETCH_H2D=1 python train.py  # default on CUDA
BDH_PREFETCH_H2D=0 python train.py  # caller-stream H2D
```

### `BDH_AMP_DTYPE` — train autocast (default `float32` / unset)

| Value | Autocast | GradScaler |
|-------|----------|------------|
| `float32` / `fp32` / `off` / unset | Off | Off |
| `bfloat16` / `bf16` | Yes | **Never** |
| `float16` / `fp16` / `half` | Yes | Only if **live CUDA** |

Optional `BDH_AMP_FORWARD_ONLY=1` (#54/#119): autocast **logits only**; CE in fp32 outside.
Default `0` keeps `model(x,y)` under the same autocast ctx.

CPU AMP is for parity smoke, not speed (`amp_throughput_claim_device() → none` here). #119/#142/#179 make capability checks transactional, soft-skip unsupported matrix arms, and report full vs forward-only scope plus scaler state; #207 verifies an unavailable CPU request preserves the complete prior AMP configuration state.
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

## Landed opts (#1–#273)

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
| **115** | `opt/docs-matrix-v26` | Refresh `OPT_STATUS.md` / `OPT_BACKLOG.md` through #114 | Docs only | — |
| **116** | `opt/online-decode-t1-v2` | Deepen opt-in blocked/online T=1 shared-V decode: flatten score tiles over `B*H` and write inference/no-grad score×V directly with `baddbmm(..., out=target)`; preserve autograd fallback, strict `tril(-1)`, `S=0` zeros, cat-free generate, and default eager | CPU 507 passed, 18 skipped, 3 warnings; parity and cat-free generate coverage | No GPU timing or speedup; GPU validation remains open |
| **117** | `opt/compile-train-v2` | Make `BDH_COMPILE_PROBE=train_bwd` the opt-in default; probe forward+backward, soft-fallback without targets, document the optimizer boundary and CPU bench matrix | CPU 506 passed, 18 skipped; CPU A/B 19.08 ms uncompiled vs 31.23 ms compile with `train_bwd` (not a speedup claim) | No GPU timing or speedup; GPU compile/CUDA graphs remain open |
| **118** | `opt/docs-matrix-v27` | Refresh `OPT_STATUS.md` / `OPT_BACKLOG.md` through #117 | Docs only | — |
| **119** | `opt/amp-train-v2` | Harden opt-in AMP capability/transactional gating, expand full vs forward-only train coverage, and require an enabled CUDA float16 GradScaler; fp32/AMP-off defaults unchanged | CPU parity/soft-skip matrix only; no throughput claim | GPU AMP train bench open |
| **120** | `opt/auto-thr-v2` | Centralize strict independent AUTO cold/decode gates; add independent-threshold generate A/B and parity/default-eager coverage | CPU parity + `aten::cat=0`; no kernel claim | **P0** GPU threshold/tile validation open |
| **121** | `opt/profile-v12` | Re-profile the #116–#120 tip and carry CPU operator evidence into status/backlog | Forward warmed `copy_`=12/call, isolated=14; generate=394/call; `cat`/`contiguous`=0 | No GPU timing or speedup; GPU validation remains open |
| **122** | `docs/opt-status-backlog-v28` | Prior docs-only refresh through #120; superseded/carried forward by this v29 refresh | Docs only | — |
| **123** | `opt/sparse-v2` | Gate `BDH_SPARSE_PROBE`, add conservative density guardrails, and test that production BDH stays dense/default-off | CPU density re-smoke x=26.63% / xy=11.37% at step 150; sparse did not beat dense; keep OFF | No GPU timing or sparse-kernel claim; GPU validation remains open |
| **124** | `opt/rope-gpu-scaffold` | Route paired T=1 RoPE through the selected `BDH_ROPE_IMPL`; add a zero-stride cis-row Triton launch without expanding broadcast rows; retain eager default and CPU fallback/parity | CPU paired parity; CUDA/Triton coverage is skip-gated here; no GPU timing | **P0** GPU fused-RoPE validation remains open |
| **125** | `opt/docs-matrix-v29` | Docs-only refresh through #123 on the #124 code tip; this v30 refresh carries the #124 scaffold into the matrix and updates the tip pointer | Docs only | — |
| **126** | `opt/docs-matrix-v30` | Docs-only refresh carrying the #124/#125 documentation tip forward and keeping the private code/docs pointer current | Docs only | — |
| **127** | `opt/profile-ci` | Add CPU profile smoke plus optional manual/nightly Chrome trace upload; keep traces gitignored and avoid PR-triggered or GPU claims | CPU smoke only; no GPU timing | — |
| **128** | `opt/decode-gemm-v2` | Preserve capacity-padded packed CacheManager Q/K/V strides in the T=1 Triton decode launcher; avoid per-step contiguous staging while retaining eager defaults, raw score×V, and strict past-only decode semantics | CPU parity; CUDA/Triton coverage remains skip-gated; no GPU timing | **P0** GPU decode measure / cold CUDA-Triton validation |
| **129** | `opt/profile-v13` | Re-profile the clean #128 tip and record attention/forward/generate operator evidence without changing semantics | CPU-only profile: attention `copy_`=2/call, forward=12/call, generate=394/call; `cat=0`, `contiguous=0` | No GPU timing or speedup claim |
| **130** | `opt/attn-bwd-v2` | Include the `online` alias alongside `blocked` in the analytic-attention GPU train matrix, crossed with `AUTOGRAD=0|1`, while retaining the CPU-safe parity gate | CPU parity/skip contract; full smoke 525 passed, 19 skipped; GPU matrix unrun | **P2** GPU analytic-attention train measure remains open |
| **131** | `opt/auto-threshold-sweep` | Deepen the CPU AUTO generate microbench with `--auto-threshold-sweep` over stable, de-duplicated decode thresholds; preserve seeded token parity, strict cold/decode resolver checks, `aten::cat=0`, and default eager behavior | CPU harness/tests only; no GPU timing | — |
| **132** | docs refresh | Refresh `OPT_STATUS.md` / `OPT_BACKLOG.md` through #130 | Docs only | — |
| **133** | docs refresh | Refresh `OPT_STATUS.md` / `OPT_BACKLOG.md` through #131 | Docs only | — |
| **134** | `opt/cuda-build-v2` | Skip optional `CUDAExtension` setup cleanly when `nvcc` is unavailable; add default/no-`nvcc` subprocess smoke coverage | CPU configuration tests pass; CUDA-only tests skip cleanly | No GPU build, timing, or kernel claim; P0 GPU validation remains open |
| **135** | `opt/scorev-fuse-v3` | Deepen the common B=1, T=1 shared-V decode epilogue by reusing flattened score/output views and writing score×V with `baddbmm(..., out=target)`; preserve broadcast-V layout, autograd fallback, eager default, strict raw `tril`, and cat-free generate | Focused CPU: 111 passed, 3 skipped; full suite: 533 passed, 19 skipped, 3 warnings | No GPU timing or speedup; GPU/Triton/CUDA validation remains P0 |
| **136** | `opt/compile-train-v3` | Clarify CPU-safe compile/probe soft-fallback diagnostics, identify the original eager module on fallback, and document FULLGRAPH/AUTOGRAD/probe boundaries without changing defaults | `test_compile.py`: 36 passed, 1 skipped, 1 warning; full suite: 533 passed, 19 skipped, 4 warnings; baseline smoke skips compile matrix/AMP sections | No GPU or CUDA-graph measurement; compile validation remains open |
| **137** | docs refresh | Refresh `OPT_STATUS.md` / `OPT_BACKLOG.md` through #136 | Docs only | — |
| **138** | `opt/rope-gpu-v2` | Harden Triton RoPE dispatch with complete tensor/device/dtype skip gates; preserve eager default and PyTorch/blocked fallback | Focused CPU: 33 passed, 2 skipped; T>1 parity and out-buffer coverage; no GPU timing | **P0** GPU fused-RoPE validation remains open |
| **139** | `opt/prefetch-h2d-v2` | Make the CPU H2D-prefetch contract explicit: gate by device before CUDA objects, keep no device lookahead, and preserve CPU tensor identity | CPU: 17 passed, 1 skipped; full suite: 539 passed, 19 skipped, 3 warnings; CPU path is a no-op | GPU H2D overlap/throughput remains unmeasured |
| **140** | docs refresh | Refresh `OPT_STATUS.md` / `OPT_BACKLOG.md` through #139 | Docs only | — |
| **141** | `opt/blocked-tile-v3` | Add wide-head CPU cold parity coverage with shared and head-matched V layouts; document bounded shared-V tile behavior without changing eager defaults | CPU parity coverage; no GPU timing or kernel claim | **P0** GPU measure / cold CUDA-Triton validation |
| **142** | `opt/amp-train-v3` | Add explicit float32/bf16/fp16 AMP configuration coverage with CPU-safe backend skips; preserve fp32 defaults and CUDA-only GradScaler gating | CPU-safe configuration/parity coverage; no throughput claim | GPU AMP train measurement remains open |
| **143** | docs refresh | Refresh `OPT_STATUS.md` / `OPT_BACKLOG.md` through #142 | Docs only | — |
| **144** | `opt/profile-v14` | Re-profile the #141/#142 tip and record CPU-only evidence without changing semantics | Flat versus profile-v13: attention `copy_`=2/call, forward `copy_`=12/call, generate `copy_`=394/call; `cat=0`, `contiguous=0`; no GPU claim | **P0** GPU measure / cold CUDA-Triton validation |
| **145** | `opt/sparse-v3` | Make sparse-probe success and guardrail-failure outcomes scriptable with stable exit codes, coverage, and notes; keep the CPU-only probe default-off | Sparse remains **OFF**; no production-path change or GPU sparse claim | GPU sparse validation remains open; **P0** GPU measure still outstanding |
| **146** | `opt/cuda-cold-v4` | Clarify CPU-safe CUDA skip behavior for the cold-attention tests without changing the CUDA implementation or defaults | CPU configuration/skip coverage; no GPU build or timing claim | **P0** GPU measure / cold CUDA validation remains open |
| **147** | `opt/online-decode-v3` | Deepen opt-in blocked/online T=1 shared-V decode by reusing shared cache views and vectorizing the B>1 epilogue; preserve strict raw `tril(-1)`, autograd fallback, cat-free generate, and default eager | CPU parity/coverage; no GPU timing or speedup claim | **P0** GPU/CUDA-Triton validation remains open |
| **148** | `opt/docs-matrix-v35` | Refresh `OPT_STATUS.md` / `OPT_BACKLOG.md` through #147 | Docs only | — |
| **149** | `opt/auto-thr-v3` | Add CPU-safe smoke coverage for the #131 `--auto-threshold-sweep` dispatcher: stable threshold de-duplication, recursive-dispatch suppression, malformed-input rejection, and independent cold-gate preservation | CPU-only smoke; eager and AUTO-off defaults unchanged; no GPU timing or performance claim | **P0** GPU measure / threshold-tile validation remains open |
| **150** | `opt/triton-cold-v4` | Add `triton_cold_skip_reason()` for actionable CPU-safe cold Triton import/device diagnostics without CUDA allocation or launch; preserve strict raw `tril(-1)`, eager defaults, AUTO-off behavior, and CPU blocked fallback | CPU targeted: 127 passed, 6 skipped; full suite: 548 passed, 19 skipped, 3 warnings; cold skips report `CUDA unavailable: torch.cuda.is_available() is false` | **P0** GPU measure / cold CUDA-Triton validation remains open |
| **151** | docs refresh | Refresh `OPT_STATUS.md` / `OPT_BACKLOG.md` through #149 on the #150 tip | Docs only | —
| **152** | `opt/gen-copy-v2` | Confirm the post-#110 default-eager generate copy ceiling: packed V snapshots, RoPE pair stores, prompt ownership, and ATen multinomial internals remain accounted for; preserve cat-free output, token parity, and strict raw `tril` semantics | CPU: 554 passed, 19 skipped, 3 warnings; focused copy/parity suite: 91 passed, 2 skipped; generate remains ~394 `copy_`/call with `cat=0` | **P0** GPU measure / copy-tax impact remains open |
| **153** | docs refresh | Refresh `OPT_STATUS.md` / `OPT_BACKLOG.md` through #152 on the #154 code tip | Docs only | — |
| **154** | `opt/rope-fuse-v3` | Reuse the warmed T>1 table-backed `(cos, sin)` narrow for callers that do not hoist `cos_sin`; invalidate the view cache when the generate table rebuilds; preserve eager defaults and CPU parity | CPU cache-hit/rebuild parity coverage; no GPU timing | **P0** GPU fused-RoPE validation remains open |
| **155** | `opt/docs-matrix-v38` | Refresh `OPT_STATUS.md` / `OPT_BACKLOG.md` through #154 | Docs only | — |
| **156** | `opt/layout-v3` | CPU `torch.profiler` probe of remaining layout materializations: one-time eval encoder-cache materialization, warm eager clone signatures, and ATen sampler contiguous hotspots; no unsafe layout flip or sampler/RNG rewrite | CPU-only evidence: warm eager forward remains `aten::contiguous`=0 and `aten::cat`=0, with two clone-backed shapes per layer; generate sampler retains `(B,V)` and `(B,)` contiguous hotspots | **P0** GPU measure / cold CUDA-Triton validation remains open |
| **157** | `opt/profile-v15` | CPU re-profile after #152/#154 and #156; record self-CPU operator percentages plus `copy_`, `cat`, and `contiguous` counts without changing semantics | Attention `copy_`=2/call, forward `copy_`=12/call, generate `copy_`=394/call; `cat=0`, `contiguous=0`; CPU-only evidence | **P0** GPU measure / cold CUDA-Triton validation remains open |
| **158** | `opt/docs-matrix-v39` | Refresh `OPT_STATUS.md` / `OPT_BACKLOG.md` through #157 on the #160 code tip | Docs only | — |
| **159** | `opt/zerograd-v2` | Make `TrainLossLogger` CUDA-deferred mode explicit: `async_cuda=True` is a no-op on CPU; add resolved-state property and synchronous CPU coverage while preserving logging and `set_to_none` defaults | CPU fallback coverage; no GPU timing or speedup claim | **P0** GPU measure / cold CUDA-Triton validation remains open |
| **160** | `opt/cache-bench-v2` | Deepen the CPU cache-page bench with `initial→final` packed-cache capacity and final KR/V allocation in KiB; assert geometric/linear policies reach the same final capacity/allocation without changing defaults | CPU accounting only; cache/generate defaults unchanged; no GPU claim | **P0** GPU measure remains open |
| **161** | `opt/profile-v16` | CPU re-profile after #159/#160 using the profile-v15 warmup/active-step schedule; record self-CPU mix and copy/cat/contiguous counts without changing semantics | Flat versus v15: attention `copy_`=2/call, forward `copy_`=12/call, generate `copy_`=394/call; `cat=0`, `contiguous=0`; CPU-only evidence | **P0** GPU measure / cold CUDA-Triton validation remains open |
| **163** | `opt/gpu-measure-v2` | Deepen structured GPU-attention measurement for cold and T=1 decode paths with JSON fields, parity deltas, and CPU-safe skip diagnostics; preserve defaults and raw strict `tril(-1)` math | CPU tests/skip path only here; no CUDA run or GPU win claim | **P0** run on real GPU remains open |
| **164** | `opt/decode-gemm-v3` | Preserve capacity-strided packed CacheManager Q/K/V views in T=1 Triton decode; avoid an unconditional staging copy while retaining eager defaults, raw strict `tril(-1)`, and past-only decode semantics | CPU parity/stride contract; no GPU timing | **P0** GPU decode measure / cold CUDA-Triton validation |
| **165** | `opt/scorev-fuse-v4` | Accumulate B>1 shared-V decode tiles in place, removing the accumulated shared-V score×V staging product while preserving CPU-safe strict-tril parity, autograd fallback, and eager defaults | CPU parity/coverage; no GPU timing | **P0** GPU decode measure / cold CUDA-Triton validation |
| **166** | `opt/compile-train-v4` | Clarify unprobed, missing-target, and first-probe `torch.compile` fallbacks with actionable retry guidance; preserve compile/probe defaults and eager attention | `test_compile.py`: 37 passed, 1 skipped; full suite: 561 passed, 19 skipped; CPU-only diagnostics | No GPU/CUDA-graph measurement; compile validation remains open |
| **167** | `opt/docs-v41` | Refresh `OPT_STATUS.md` / `OPT_BACKLOG.md` through #165 on the #168 predecessor tip | Docs only | — |
| **168** | `opt/rope-gpu-v3` | Expose stable Triton/CUDA skip reasons and preserve T=1 paired RoPE parity for a strided cache-slot-like `out=` buffer without changing eager defaults | Focused CPU: 41 passed, 2 skipped; full suite: 569 passed, 19 skipped, 3 warnings; no GPU timing | **P0** GPU fused-RoPE validation remains open |
| **169** | `opt/prefetch-h2d-v3` | Expand CPU no-op coverage across async/sync host modes and H2D overrides; document CUDA staged-buffer event/lifetime handoff without changing defaults | CPU: 20 passed, 1 skipped; full suite: 571 passed, 19 skipped, 3 warnings; no GPU timing | GPU H2D overlap/throughput remains unmeasured |
| **170** | `opt/online-decode-v4` | Preserve packed key views in opt-in online T=1 shared-V decode when flattening would copy; retain shared value views, strict causal parity, autograd fallback, eager defaults, and cat-free generate | CPU focused: 94 passed, 3 skipped; additional CPU checks: 54 passed, 1 unrelated existing failure; no GPU timing or correctness claim | **P0** GPU/CUDA-Triton validation remains open |
| **171** | `opt/docs-v42` | Refresh `OPT_STATUS.md` / `OPT_BACKLOG.md` through #169 | Docs only | — |
| **172** | `opt/docs-v42` | Refresh `OPT_STATUS.md` / `OPT_BACKLOG.md` through #170 | Docs only | — |
| **173** | `opt/attn-bwd-v3` | Deepen the CPU analytic-attention parity matrix across eager/blocked/online/triton/cuda, AUTOGRAD off/on, shared/full-head V layouts, non-tile sequence shape, and randomized output gradients; make expected native CUDA-backend skips explicit | CPU parity/skip coverage only; no GPU timing or training claim | **P2** GPU analytic-attention train measure remains open |
| **174** | `opt/cuda-cold-v5` | Add explicit CPU-only native-extension setup smoke while preserving no-CUDA/no-nvcc skips, strict raw `tril(-1)` references, and unchanged defaults | CPU: 21 passed, 5 skipped focused; full suite: 575 passed, 19 skipped, 3 warnings; no GPU timing | **P0** GPU measure / cold CUDA validation remains open |
| **175** | `opt/docs-v43` | Refresh `OPT_STATUS.md` / `OPT_BACKLOG.md` through #174 while preserving profile-v16 counts and the real-GPU P0 blocker | Docs only | — |
| **176** | `opt/blocked-tile-v4` | Add bounded wide-head CPU blocked-tile parity at the partial 128-row tile boundary for shared-V and head-matched-V layouts; defaults and raw strict-tril math unchanged | CPU-only parity at `B=2,H=3,T=257,N=160,D=192`; no GPU timing or win claim | **P0** GPU measure / cold CUDA-Triton validation remains open |
| **177** | docs refresh | Refresh `OPT_STATUS.md` / `OPT_BACKLOG.md` through #176 while preserving profile-v16 counts and the real-GPU P0 blocker | Docs only | — |
| **178** | `opt/triton-cold-v5` | Mark unavailable cold Triton paths with an explicit CPU-safe skip marker and add long-T blocked-fallback parity for shared-V and per-head-V layouts; defaults and raw strict-tril math unchanged | CPU: 39 passed, 3 skipped focused; full suite: 593 passed, 19 skipped, 3 warnings; no CUDA allocation, launch, timing, or win claim | **P0** GPU measure / cold CUDA-Triton validation remains open |
| **179** | `opt/amp-train-v4` | Add CPU-safe `float32|bfloat16|float16` AMP contract coverage, actionable unsupported-CPU skips, and the CUDA-only float16 GradScaler gate; preserve fp32 / AMP-off defaults | CPU-only contract coverage; no GPU timing or throughput claim | **P3** GPU AMP train measurement remains open |
| **180** | docs refresh | Refresh `OPT_STATUS.md` / `OPT_BACKLOG.md` through #179 while preserving profile-v16 counts and the real-GPU P0 blocker | Docs only | — |
| **181** | `opt/cache-bench-v3` | Expose initial→final packed-cache capacity and final packed KR/V allocation in KiB in the page sweep; assert the capacity/allocation invariant before and after CPU page growth without changing defaults | CPU-only packed-footprint/accounting evidence; generate smoke matches fixed-capacity output with `aten::cat=0`; no GPU claim | **P0** GPU measure / cold CUDA-Triton validation remains open |
| **182** | `opt/profile-v17` | Re-profile the #176–#181 tip with the profile-v15/v16 schedule and record self-CPU percentages plus `copy_`, `cat`, and `contiguous` counts without changing semantics | Flat versus profile-v16: attention `copy_`=2/call, forward `copy_`=12/call, generate `copy_`=394/call; `cat=0`, `contiguous=0`; CPU-only evidence | **P0** GPU measure / cold CUDA-Triton validation remains open |
| **183** | docs refresh | Refresh `OPT_STATUS.md` / `OPT_BACKLOG.md` through #182 while preserving profile-v17 counts and the real-GPU P0 blocker | Docs only | — |
| **184** | `opt/sparse-probe` | Add explicit `exit_code` / `reason` markers for disabled/complete (`0`), failing density guardrail (`2`), and guardrail enforcement without samples (`3`); add CPU smoke coverage while preserving the default-off opt-in and dense production path | CPU smoke only; sparse remains OFF; no GPU claim | **P0** GPU measure / sparse validation remains open |
| **185** | `opt/auto-thr-v4` | Deepen CPU AUTO threshold-sweep gate smoke: an omitted cold threshold mirrors each de-duplicated decode threshold, including zero, per child; malformed input errors before dispatch | CPU-only control-flow smoke; eager/AUTO-off defaults unchanged; no GPU claim | **P0** GPU measure / threshold/tile validation remains open |
| **186** | `opt/gen-bench-v3` | Harden generate sweeps with eager as the token reference, explicit PASS/FAIL rows, fail-closed token/cat checks, separate AUTO `cat0`/`cat1` summary columns, and strict threshold smoke coverage | CPU-only validation; defaults remain eager; no GPU timing or win claim | **P1** GPU generate measurement remains open |
| **187** | docs refresh | Refresh `OPT_STATUS.md` / `OPT_BACKLOG.md` through #186 while preserving profile-v17 counts and the real-GPU P0 blocker | Docs only | — |
| **188** | `opt/layout-v4` | Deepen the CPU sampler layout probe across scaled sampling, narrow `top_k`, and the `top_k=vocab` fallback; preserve sampler/RNG behavior, defaults, and raw strict-tril attention semantics | CPU-only profiler coverage: each opt-in case has one expected `(B,)` index materialization, no `aten::cat`, and no new `(B,V)` contiguous signature; no GPU claim | **P0** GPU measure / cold CUDA-Triton validation remains open |
| **189** | `opt/online-decode-v5` | Add B>1 long-S packed shared-V CPU parity coverage across blocked, online, Triton-fallback, and CUDA-reference dispatch while preserving eager defaults, strict lower-triangular semantics, and the autograd fallback | CPU-only parity/stride coverage at the long-S tiled boundary; no GPU timing or win claim | **P0** GPU measure / cold CUDA-Triton validation remains open |
| **190** | docs refresh | Refresh `OPT_STATUS.md` / `OPT_BACKLOG.md` through #188 while preserving the profile-v17 counts and the real-GPU P0 blocker | Docs only | — |
| **191** | `opt/rope-fuse-v4` | Keep paired and flat T=1 RoPE cache narrows aligned across position switches; add CPU T>1 narrow and paired-cache invalidation coverage while preserving eager defaults and raw strict-tril attention semantics | CPU-only cache correctness/parity coverage; no GPU timing or RoPE win claim | **P0** GPU fused-RoPE validation remains open |
| **193** | docs refresh | Refresh `OPT_STATUS.md` / `OPT_BACKLOG.md` through #191 while preserving profile-v17 counts and the real-GPU P0 blocker | Docs only | — |
| **194** | `opt/scorev-v5` | Deepen B=1 shared-V score×V epilogue coverage for packed-cache `out=` accumulation and the grad-safe fallback while preserving eager defaults and raw strict-tril attention semantics | CPU-only test coverage; no GPU timing or speedup claim | **P0** GPU measure / cold CUDA-Triton validation remains open |
| **195** | `opt/compile-v5` | Sharpen optional `torch.compile` missing-target and first-probe soft-fallback diagnostics; add CPU eval-probe coverage that restores the caller training mode without changing defaults | CPU-only diagnostics/smoke; no GPU or CUDA-graph timing claim | **P1** GPU inductor / CUDA-graph validation remains open |
| **196** | `opt/cuda-build-v3` | Check CUDA_HOME/CUDA_PATH and PATH before constructing the optional CUDA extension; keep missing-nvcc setup a clear CPU-safe no-op and retain explicit CPU-only setup smoke | CPU-only setup/test coverage; no GPU build or timing claim | **P0** GPU measure / cold CUDA-Triton validation remains open |
| **197** | docs refresh | Refresh `OPT_STATUS.md` / `OPT_BACKLOG.md` through #196 while preserving profile-v17 counts and the real-GPU P0 blocker | Docs only | — |
| **198** | `opt/attn-bwd-v4` | Add T=1 strict-tril backward contract coverage across eager, blocked, online, Triton fallback, and CUDA-reference dispatch for shared-V and full-head-V layouts; assert exact zero output and Q/K/V gradients when no past key exists | CPU-only contract coverage; no GPU timing or speedup claim | **P0** GPU measure / cold CUDA-Triton validation remains open |
| **199** | `opt/decode-gemm-v4` | Add CPU parity coverage for T=1 packed, capacity-strided per-head K/V views at and just above the decode oneshot boundary; lock the per-head decode GEMM flattening shape/stride and view-preserving contract | CPU-only parity/stride coverage; no GPU timing or performance claim | **P0** GPU decode measure / cold CUDA-Triton validation remains open |
| **200** | `opt/docs-v50` | Refresh `OPT_STATUS.md` / `OPT_BACKLOG.md` through #198 while preserving profile-v17 counts and the real-GPU P0 blocker | Docs only | — |
| **201** | `opt/gpu-measure-v3` | Distinguish real CUDA summaries from `--force-cpu` smoke output in schema version 2; report actual device, timing scope, and reason, with CPU contract coverage and an explicit measurement boundary | CPU-only smoke/contract coverage; no GPU timing or speedup claim | **P0** real GPU measurement / cold CUDA-Triton validation remains open |
| **202** | `opt/docs-v51` | Refresh `OPT_STATUS.md` / `OPT_BACKLOG.md` through #201 while preserving the real-GPU P0 blocker | Docs only | — |
| **203** | `opt/prefetch-v4` | Gate opt-in H2D staging on both CUDA device type and live CUDA availability; expose actionable CPU-safe skip reasons before stream/event construction | CPU-only skip/identity coverage; no H2D timing or correctness claim | **P1** GPU H2D overlap/throughput remains unmeasured |
| **204** | `opt/profile-v18` | Re-profile the #199–#201 tip with the matched profile-v17 schedule; record unchanged attention/forward/generate `copy_` counts and `cat`/`contiguous` floors | CPU-only: `copy_`=2/12/394 per call; `cat=0`; `contiguous=0`; flat versus v17 | **P0** real GPU measurement / cold CUDA-Triton validation remains open |
| **205** | `test/zerograd-v3` | Deepen the CPU-safe `TrainLossLogger` lifecycle contract: flush a partial window exactly once on `close()` and reject updates after close | CPU-only lifecycle coverage; no GPU timing or speedup claim | **P0** real GPU measurement / cold CUDA-Triton validation remains open |
| **206** | `opt/docs-v52` | Refresh `OPT_STATUS.md` / `OPT_BACKLOG.md` through #204 while preserving profile-v18 counts and the real-GPU P0 blocker | Docs only | — |
| **207** | `test/amp-train-contracts-v4` | Deepen the CPU AMP failure-state contract so an unavailable request reports an actionable error without mutating the complete prior configuration | CPU-only failure-state coverage; no GPU timing or throughput claim | **P0** real GPU measurement / cold CUDA-Triton validation remains open |
| **208** | `test/prefill-blocked` | Deepen CPU long-path blocked attention parity with exact autograd gradient checks at the tiled boundary; preserve raw strict-tril semantics and defaults | CPU-only forward/gradient parity coverage; no GPU timing or speedup claim | **P0** real GPU measurement / cold CUDA-Triton validation remains open |
| **209** | `test/rope-fuse` | Stabilize the Triton RoPE skip-reason contract by validating the primary input before optional runtime probing; preserve eager defaults and safe fallbacks | CPU-only skip-diagnostic coverage; no GPU timing or speedup claim | **P0** real GPU measurement / cold CUDA-Triton validation remains open |
| **210** | docs refresh | Refresh `OPT_STATUS.md` / `OPT_BACKLOG.md` through #209 | Docs only | — |
| **211** | `opt/cache-bench-v4` | Cover linear cache-page accounting when `max_seq` ends on a partial page; lock analytic grow/copy-byte accounting to the CPU run without changing cache or generate defaults | CPU-only accounting; no GPU timing, memory, or speedup claim | **P0** real GPU measurement / cold CUDA-Triton validation remains open |
| **212** | `opt/cuda-cold-v6` | Validate malformed cold-attention shapes before optional native dispatch; preserve stable CPU-safe `ValueError` contracts and unchanged strict raw `tril(-1)` semantics | CPU-only contract coverage; no GPU timing or kernel claim | **P0** real GPU measurement / cold CUDA-Triton validation remains open |
| **213** | `opt/triton-cold-v6` | Exercise Triton import, CUDA-unavailable, and available skip-reason branches deterministically without CUDA work | CPU-only gate coverage; no GPU timing or kernel claim | **P0** real GPU measurement / cold CUDA-Triton validation remains open |
| **214** | `test/sparse-probe-contract-v5` | Make an enforced sparse-density guardrail failure terminal before CPU crossover work; keep sparse opt-in and production wiring unchanged | CPU-only terminal exit contract; no attention, GPU timing, or sparse-kernel claim | **P0** real GPU measurement / cold CUDA-Triton validation remains open |
| **215** | `test/bench-generate` | Verify interrupted generate-benchmark dispatch and AUTO threshold scopes restore their environment contracts | CPU-only environment-contract coverage; no GPU timing or generate-performance claim | **P1** GPU generate measurement remains open |
| **216** | `opt/auto-thr-v5` | Reject negative `BDH_ATTN_AUTO_THRESHOLD` and `BDH_ATTN_AUTO_COLD_THRESHOLD` values before dispatch | CPU-only threshold-contract coverage; no GPU timing or performance claim | **P0** GPU measure / cold CUDA-Triton validation remains open |
| **217** | `opt/docs-v54` | Refresh `OPT_STATUS.md` / `OPT_BACKLOG.md` through #215 while preserving profile-v18 counts and the real-GPU P0 blocker | Docs only | — |
| **218** | `opt/online-v6` | Lock CPU online decode raw-score semantics across blocked, online, Triton-fallback, and CUDA-reference dispatch; no softmax, scale, or fused attention op | CPU-only contract coverage; no GPU timing or speedup claim | **P0** GPU measure / cold CUDA-Triton validation remains open |
| **219** | `opt/scorev-v6` | Add CPU parity coverage for B=1 non-flat packed K views on the score×V path; preserve 4-D strides, avoid staging, and match eager decode | CPU-only parity/stride coverage; no GPU timing or performance claim | **P0** GPU measure / cold CUDA-Triton validation remains open |
| **220** | `opt/scorev-v6` | Align the B=1 non-flat K-view contract with the tiled `bmm` score×V path while preserving eager parity | CPU-only tiled shape/parity coverage; no GPU timing or performance claim | **P0** GPU measure / cold CUDA-Triton validation remains open |
| **221** | `opt/layout-v5` | Deepen the multi-step sampler-layout contract: generate two tokens and require one `(B,)` sampler result per token while keeping `aten::cat=0` and no `(B,V)` materialization | CPU-only profiler contract coverage; no GPU timing or performance claim | **P0** GPU measure / cold CUDA-Triton validation remains open |
| **224** | `opt/rope-fuse-v5` | Guard paired T=1 RoPE table lookups at negative and out-of-range positions without poisoning the last valid flat or paired narrow cache | CPU-only boundary/skip contract coverage; no GPU timing or performance claim | **P0** GPU measure / cold CUDA-Triton validation remains open |
| **225** | `opt/compile-v6` | Clear partial parameter gradients when a `train_bwd` compile probe fails before returning the eager fallback; preserve caller training mode | CPU-only failing-backward fallback coverage; no GPU compile or performance claim | **P1** GPU inductor / CUDA-graph validation remains open |
| **228** | `opt/profile-v19` | Record a matched CPU profile-v19 after #227 with unchanged operator counts | CPU-only: attention/forward/generate `copy_`=2/12/394 per call; `cat=0`, `contiguous=0`; flat versus v18 | **P0** real GPU measurement / cold CUDA-Triton validation remains open |
| **229** | `opt/attn-bwd-v5` | Deepen strict-tril backward coverage: a single query loss reaches only its matching Q row and strict-past K/V rows across eager, blocked, online, Triton fallback, and CUDA-reference dispatch | CPU-only contract coverage; no GPU timing or performance claim | **P0** GPU measure / cold CUDA-Triton validation remains open |
| **230** | `opt/decode-gemm-v5` | Add CPU-safe autograd parity for capacity-strided per-head K/V decode GEMMs at the tiled oneshot boundary across eager, blocked, online, and Triton-fallback dispatch | CPU-only parity/stride/gradient coverage; no GPU timing or performance claim | **P0** GPU measure / cold CUDA-Triton validation remains open |
| **231** | `opt/docs-v57` | Refresh `OPT_STATUS.md` / `OPT_BACKLOG.md` through #230 while preserving profile-v19 counts and the real-GPU P0 blocker | Docs only | — |
| **232** | `opt/gpu-measure-v4` | Add schema-v3 structured run/backend skip records, explicit per-backend status/reason, and null timing for unavailable backends while preserving CPU smoke labeling | CPU-only smoke/contract coverage; no GPU timing or speedup claim | **P0** real GPU measurement / cold CUDA-Triton validation remains open |
| **233** | `opt/prefetch-v5` | Harden the CPU-safe H2D skip gate when `torch.cuda.is_available()` raises; preserve allocation-free behavior and existing prefetch defaults | CPU-only no-allocation/skip contract coverage; no CUDA H2D timing or overlap claim | **P1** GPU H2D overlap measurement remains open |
| **234** | `test/amp-train-contracts-v4` | Add parameterized CPU-only skip-reason coverage for unavailable bfloat16 and float16 AMP configuration while preserving requested dtype/context, attention math, and defaults | CPU-only AMP configuration coverage; no GPU timing or throughput claim | **P0** real GPU measurement / cold CUDA-Triton validation remains open |
| **235** | `opt/blocked-tile-v6` | Extend long-path CPU forward/gradient parity from blocked tiles to the public online alias for shared and head-matched V layouts at T=257 | CPU-only blocked/online gradient parity; no GPU timing or performance claim | **P0** real GPU measurement / cold CUDA-Triton validation remains open |
| **236** | `opt/docs-v58` | Refresh `OPT_STATUS.md` / `OPT_BACKLOG.md` through #235 while preserving profile-v19 counts and the real-GPU P0 blocker | Docs only | —
| **237** | `opt/sparse-v6` | Add a CPU-safe density-only sparse probe contract: a passing guardrail completes without starting the CPU crossover sweep; keep the probe opt-in and production sparse wiring unchanged | CPU-only contract coverage; sparse remains OFF; no GPU claim | **P0** real GPU measurement / cold CUDA-Triton validation remains open
| **238** | `test/bench-generate` | Verify interrupted AUTO runs restore variables that were initially unset | CPU-only environment-contract coverage; no GPU timing or generate-performance claim | **P1** GPU generate measurement remains open
| **239** | `test/online-decode-v5` | Extend the raw-score decode contract to the eager backend alongside blocked, online, Triton-fallback, and CUDA-reference dispatch; preserve no-softmax/no-scale semantics | CPU-only decode contract coverage; no GPU timing or performance claim | **P0** real GPU measurement / cold CUDA-Triton validation remains open
| **240** | `opt/auto-thr-v6` | Verify explicit backends remain authoritative on both AUTO cold and decode gates | CPU-only AUTO contract coverage; eager/AUTO-off defaults unchanged; no GPU timing or performance claim | **P0** GPU measure / cold CUDA-Triton validation remains open
| **241** | `test/layout_v3.py` | Extend the CPU sampler-layout contract to the `top_k=1` boundary alongside scaled, narrow, and full-vocabulary cases | CPU-only profiler contract coverage; no GPU timing or performance claim | **P0** GPU measure / cold CUDA-Triton validation remains open
| **242** | `test/fuse_scorev.py` | Add CPU parity coverage for distinct, non-aliased query/key tensors across eager, blocked, and online score×V paths | CPU-only parity coverage; no GPU timing or performance claim | **P0** GPU measure / cold CUDA-Triton validation remains open
| **244** | `test/rope_fuse.py` | Enforce positive blocked RoPE tile sizes before the T=1 shortcut and cover the contract at T=1 and T>1 | CPU-only boundary/skip contract coverage; no GPU timing or performance claim | **P0** real GPU measurement / cold CUDA-Triton validation remains open
| **245** | `test/compile.py` | Verify a successful `train_bwd` compile probe restores an eval caller mode and clears parameter gradients | CPU-only compile-probe cleanup coverage; no GPU compile or performance claim | **P1** GPU inductor / CUDA-graph validation remains open
| **249** | `tests/test_inc_decode.py` | Extend the packed per-head T=1 decode contract through the CPU CUDA-mirror tiled path with forward and Q/K/V autograd parity coverage | CPU-only CUDA-mirror parity coverage; no CUDA timing or performance claim | **P0** real GPU measurement / cold CUDA-Triton validation remains open
| **251** | `benchmarks/bench_gpu_attn.py` / `tests/test_bench_gpu_attn.py` | Keep `--force-cpu` execution on CPU even when CUDA is available; schema-v3 smoke contract remains honest | CPU-only harness/test coverage; no GPU timing or speedup claim | **P0** real GPU measurement / cold CUDA-Triton validation remains open
| **252** | `tests/test_dataloader.py` | Verify the CPU H2D gate short-circuits before probing CUDA availability | CPU-only no-probe contract coverage; no H2D timing or overlap claim | **P1** GPU H2D overlap remains unmeasured
| **253** | `OPT_STATUS.md` / `OPT_BACKLOG.md` | Refresh the optimization matrix through profile-v20, #251, and #252 | Docs only; no GPU evidence | —
| **254** | `tests/test_amp_train_contracts_v4.py` | Preserve the complete active AMP configuration when an unavailable CPU dtype switch fails | CPU-only failure-state coverage; no GPU timing or throughput claim | **P0** real GPU measurement / cold CUDA-Triton validation remains open
| **255** | `tests/test_bench_generate.py` | Verify interrupted generate-implementation sweeps restore an environment variable that was initially absent | CPU-only environment-contract coverage; no GPU timing or generate-performance claim | **P1** GPU generate measurement remains open
| **256** | `tests/test_prefill_blocked.py` | Extend long blocked/online gradient parity to `B=2` while retaining wide-head shared and head-matched V layouts | CPU-only batched forward/gradient parity; no GPU timing or performance claim | **P0** real GPU measurement / cold CUDA-Triton validation remains open
| **257** | `tests/test_sparse_probe_contract_v7.py` | Cover the density-only sparse-probe crossover boundary without changing opt-in or production sparse wiring | CPU-only sparse contract coverage; sparse remains OFF; no GPU claim | **P0** real GPU measurement / cold CUDA-Triton validation remains open
| **258** | `tests/test_auto_threshold_contract_v7.py` | Keep an unset cold AUTO threshold synchronized with later decode-threshold changes while preserving strict cold/decode gates | CPU-only AUTO contract coverage; no GPU timing or performance claim | **P0** real GPU measurement / cold CUDA-Triton validation remains open
| **259** | `tests/test_fuse_scorev.py` | Verify blocked/online score×V preserves Q/K/V gradients for distinct query and key tensors | CPU-only gradient parity; no GPU timing or performance claim | **P0** real GPU measurement / cold CUDA-Triton validation remains open
| **260** | `tests/test_online_decode_v5.py` | Cover signed raw-score decode with independent per-head values across eager, blocked, online, Triton-fallback, and CUDA-reference dispatch | CPU-only decode contract coverage; no GPU timing or performance claim | **P0** real GPU measurement / cold CUDA-Triton validation remains open
| **266** | `tests/test_attn_bwd_v4_contract.py` | Extend strict-tril backward locality coverage to first, interior, and final query rows | CPU-only backward contract coverage; no GPU timing or performance claim | **P0** real GPU measurement / cold CUDA-Triton validation remains open
| **268** | `tests/test_inc_decode.py` | Cover tiled packed per-head T=1 decode GEMM autograd parity across blocked, online, and Triton-fallback paths | CPU-only decode gradient parity; no GPU timing or performance claim | **P0** real GPU measurement / cold CUDA-Triton validation remains open
| **270** | `tests/test_dataloader.py` | Explicit `BDH_PREFETCH_H2D=0` skips CUDA availability probes and stream/event allocation; defaults and runtime behavior unchanged | CPU-only no-probe/skip coverage; no H2D timing or overlap claim | **P1** GPU H2D overlap/throughput remains unmeasured
| **271** | `benchmarks/bench_gpu_attn.py` | Mark run-level unavailable-device summaries with `status: skip` and carry the schema-v4 contract | CPU-safe skip coverage; no GPU timing or speedup claim | **P0** real GPU measurement remains open
| **272** | `tests/test_sparse_probe_contract_v7.py` | Make an enforced sparse-density guardrail failure terminal before CPU crossover work; sparse remains opt-in/default-off | CPU-only terminal contract; no sparse-kernel or GPU claim | **P0** real GPU measurement / sparse validation remains open
| **273** | `tests/test_prefill_blocked.py` | Add a deterministic blocked/online tile-boundary contract for raw `Q @ K.T` × `tril(diagonal=-1)` × `V` | CPU-only test coverage; no GPU timing or performance claim | **P0** real GPU measurement / cold CUDA-Triton validation remains open
| **tip** | `OPT_NOTES.md` (profile-v20) | Retain matched CPU operator counts through `3b3c082`: attention/forward/generate `copy_`=2/12/394 per call, `cat=0`, `contiguous=0`; #270–#273 add CPU-safe contract/skip coverage only | CPU-only profile evidence and contract coverage; no GPU timing or speedup claim | **P0** real GPU measurement / cold CUDA-Triton validation remains open
### Concurrent main updates

- **#159** `opt/zerograd-v2` merged as `717c38e`; it was in-flight while the original docs branch was prepared but is landed on the current main tip.
- **#161** `opt/profile-v16` merged as `8bd6b17`; **#163** `opt/gpu-measure-v2` merged as `d0e667b` with structured GPU measurement scaffolding but no CUDA results on this box.
- **#164** `opt/decode-gemm-v3` merged as `7719e9a`; preserves capacity-strided packed T=1 decode views without an unconditional staging copy, with CPU parity only and no CUDA timing.
- **#165** `opt/scorev-fuse-v4` merged as `aa73a43`; deepens B>1 shared-V decode accumulation in place with CPU-safe strict-tril parity, no CUDA timing.
- **#166** `opt/compile-train-v4` merged as `79eef11`; clarifies missing-target and first-probe soft-fallback guidance without changing compile/probe defaults or making a GPU claim.
- **#167** `opt/docs-v41` merged as `fb698fc`; refreshes the status/backlog docs through #165.
- **#168** `opt/rope-gpu-v3` merged as `c3233ed`; exposes actionable Triton skip reasons and adds CPU parity for strided T=1 RoPE output buffers, with no GPU timing.
- **#169** `opt/prefetch-h2d-v3` merged as `62acaa7`; expands CPU no-op coverage across host/H2D settings and documents CUDA staged-buffer lifetime, with no GPU timing or overlap claim.
- **#170** `opt/online-decode-v4` merged as `1d12071`; retains packed key/value views for opt-in online decode with CPU parity only and no GPU timing or win claim.
- **#171** `opt/docs-v42` merged as `9319670`; refreshes the status/backlog docs through #169.
- **#172** `opt/docs-v42` merged as `11972ae`; refreshes the status/backlog docs through #170.
- **#173** `opt/attn-bwd-v3` merged as `f95c143`; deepens the CPU analytic-attention parity matrix with explicit expected CUDA-backend skips and no GPU claim.
- **#174** `opt/cuda-cold-v5` merged as `bc3967b`; adds explicit CPU-only extension setup smoke while preserving actionable no-CUDA/no-nvcc skips and strict CPU references, with no GPU claim.
- **#175** `opt/docs-v43` merged as `0af0690`; refreshes the status/backlog docs through #174 while preserving profile-v16 counts and the real-GPU blocker.
- **#176** `opt/blocked-tile-v4` is at `4e13111`; adds CPU-only wide-head partial-tile parity coverage with no GPU timing or win claim.
- **#177** docs refresh is at `5507747`; carries the matrix through the blocked-tile tip.
- **#178** `opt/triton-cold-v5` is at `f72033e`; adds CPU-safe cold-Triton skip markers and long-T fallback parity with no CUDA timing or win claim.
- **#179** `opt/amp-train-v4` is at `0983326`; adds CPU-only AMP dtype/scaler contracts with no GPU timing or throughput claim.
- **#180** docs refresh is at `f92090f`; carries the matrix through #179 and preserves profile-v16 counts.
- **#181** `opt/cache-bench-v3` is at `f34d908`; adds CPU-only initial→final packed-cache footprint accounting with no GPU memory, timing, or speedup claim.
- **#182** `opt/profile-v17` is at `4eea9ad`; matched CPU evidence is flat versus profile-v16 with `copy_` counts 2/12/394 and `cat=0`, `contiguous=0`.
- **#183** docs refresh is at `86ba7df`; carries the matrix through profile-v17 and preserves the real-GPU P0 blocker.
- **#184** `opt/sparse-probe` is at `63b25cc`; adds CPU-only sparse-probe exit guardrails and smoke coverage while sparse remains OFF.
- **#185** `opt/auto-thr-v4` is at `e7943ba`; adds CPU-only per-child AUTO threshold-gate fallback and malformed-input dispatch smoke.
- **#187** docs refresh is at `2088c90`; carries the matrix through #186 and preserves profile-v17 counts plus the real-GPU P0 blocker.
- **#188** `opt/layout-v4` is at `87ccac9`; deepens CPU sampler-layout probe coverage for scaled and top-k signatures without changing defaults or sampler/RNG behavior.
- **#189** `opt/online-decode-v5` is at `da79013`; adds B>1 long-S packed shared-V CPU parity across blocked, online, Triton-fallback, and CUDA-reference dispatch with no GPU timing claim.
- **#190** docs refresh is at `a618356`; carries the matrix through #188 and preserves profile-v17 counts plus the real-GPU P0 blocker.
- **#193** docs refresh merged as `4bd410a`; carries the matrix through #191 and preserves profile-v17 counts plus the real-GPU P0 blocker.
- **#194** `opt/scorev-v5` is at `4fda79f`; adds CPU-only packed-cache `out=` accumulation and grad-safe fallback coverage on the B=1 shared-V score×V path, with no GPU timing claim.
- **#195** `opt/compile-v5` is at `4ae8ab2`; clarifies missing-target and first-probe soft fallbacks and adds CPU eval-probe training-mode restoration coverage, with no GPU/CUDA-graph timing claim.
- **#196** `opt/cuda-build-v3` is at `4e9d3e4`; clarifies missing-nvcc setup discovery and keeps the CPU-safe setup smoke explicit, with no GPU build or timing claim.
- **#197** docs refresh merged as `ccadc3a`; carries the matrix through #196 and preserves profile-v17 counts plus the real-GPU P0 blocker.
- **#198** `opt/attn-bwd-v4` is at `9ad9015`; adds CPU-only T=1 strict-tril backward contract coverage across eager, blocked, online, Triton-fallback, and CUDA-reference dispatch for shared-V and full-head-V layouts, with no GPU timing or speedup claim.
- **#199** `opt/decode-gemm-v4` merged as `269eb2d`; adds CPU-only packed per-head T=1 decode GEMM parity and stride/view contract coverage, with no GPU timing or performance claim.
- **#200** docs refresh merged as `e253bba`; carries the matrix through #198 while preserving profile-v17 counts and the real-GPU P0 blocker.
- **#201** `opt/gpu-measure-v3` merged as `c25aa9f`; marks forced-CPU smoke summaries with schema version 2, actual device, timing scope, and reason, with no GPU timing or speedup claim.
- **#202** docs refresh merged as `3d3d1fe`; carries the matrix through #201 and preserves the real-GPU P0 blocker.
- **#203** `opt/prefetch-v4` merged as `9669e06`; gates unavailable-CUDA H2D staging before stream/event construction and adds CPU-safe skip-reason coverage, with no GPU timing or overlap claim.
- **#204** `opt/profile-v18` merged as `5995a7c`; matched CPU counts remain flat versus profile-v17 (`copy_`=2/12/394 per call, `cat=0`, `contiguous=0`), with no GPU timing or speedup claim.
- **#205** zerograd logger lifecycle tests merged as `e8ad099`; CPU-only coverage locks one-shot partial flush and rejects updates after close, with no GPU timing or speedup claim.
- **#206** docs refresh merged as `6e0f313`; carries the matrix through #204 and preserves profile-v18 counts plus the real-GPU blocker.
- **#207** CPU AMP failure-state tests merged as `dd11718`; unavailable requests preserve the complete prior AMP configuration, with no GPU timing or throughput claim.
- **#208** blocked-tile long-path gradient tests merged as `dd4fc90`; CPU-only forward/gradient parity covers shared and head-matched V layouts at the tiled boundary, with no GPU timing or speedup claim.
- **#209** Triton RoPE skip-reason tests merged as `a48b7e1`; invalid primary inputs are diagnosed before runtime probing, with no GPU timing or speedup claim.
- **#210** docs refresh merged as `fecfba4`; carries the matrix through #209.
- **#211** cache-page accounting tests merged as `d3a37b7`; covers a partial final page with CPU-only closed-form grow/copy-byte validation and no GPU claim.
- **#212** CUDA cold shape-contract tests merged as `61af5e0`; validates malformed inputs before optional native dispatch with no GPU timing or kernel claim.
- **#213** Triton cold skip-gate tests merged as `bd809b9`; covers import/device branches deterministically without CUDA work or GPU claims.
- **#214** sparse probe guardrail tests merged as `8ec4b16`; failed enforced density guardrails stop before CPU crossover work, with no attention or GPU claim.
- **#215** generate-benchmark environment tests merged as `eafe65c`; interrupted dispatch and AUTO threshold scopes restore their environment contracts without GPU timing or performance claims.
- **#216** AUTO threshold tests merged as `265b25c`; both decode and cold thresholds reject negative values before dispatch, with CPU-only contract coverage.
- **#217** docs refresh merged as `fb80c34`; carries the matrix through #215 while preserving profile-v18 counts and the real-GPU P0 blocker.
- **#218** online decode score tests merged as `50d40f5`; blocked, online, Triton-fallback, and CUDA-reference paths retain raw scores without softmax or scale, with CPU-only contract coverage.
- **#219** B=1 non-flat score×V tests merged as `d218150`; CPU parity confirms non-flat packed K views retain the 4-D path without staging, with no GPU timing or performance claim.
- **#220** B=1 score×V view tests merged as `f344494`; non-flat K views retain tiled `bmm` shape parity with eager decode, with no GPU timing or performance claim.
- **#221** multi-step sampler layout tests merged as `39ca536`; two-token CPU profiling preserves one `(B,)` result per token, `aten::cat=0`, and no `(B,V)` sampler materialization.
- **#224** paired RoPE table-boundary tests merged as `ad7a50b`; rejected negative/out-of-range lookups do not poison valid flat or paired cache narrows, with no GPU timing or performance claim.
- **#225** failed compile-backward probe tests merged as `258b24c`; partial parameter gradients are cleared before the eager fallback, caller training mode is preserved, and CPU-only regression coverage adds no GPU compile result.
- **#228** profile-v19 merged as `1ad6b01`; matched CPU operator counts remain attention/forward/generate `copy_`=2/12/394 per call with `cat=0` and `contiguous=0`.
- **#229** strict-tril backward support tests merged as `26d0b4f`; single-query gradients reach only the matching Q row and strict-past K/V rows across the supported dispatches, with no GPU timing claim.
- **#230** packed per-head decode GEMM tests merged as `0a42a67`; CPU autograd parity covers capacity-strided K/V views at the tiled oneshot boundary, with no GPU timing claim.
- **#231** docs refresh merged as `d8eb80e`; carries the matrix through #230 while preserving profile-v19 counts and the real-GPU P0 blocker.
- **#232** structured backend skip tests merged as `efbb1f3`; schema-v3 summaries record run/backend skips, explicit result status/reason, and null timing for unavailable backends, with CPU-only coverage and no GPU timing claim.
- **#233** prefetch-v5 merged as `ff5b071`; CUDA availability probe errors remain allocation-free skip outcomes before stream/event construction, with CPU-only coverage and no H2D overlap claim.
- **#234** CPU AMP skip-reason contract merged as `1f1edd9`; parameterized unavailable bfloat16/float16 requests retain requested dtype and backend context, with no GPU timing or throughput claim.
- **#235** long blocked/online gradient parity merged as `ee586d0`; T=257 shared and head-matched V CPU cases cover both blocked and online aliases, with no GPU performance claim.
- **#236** docs refresh merged as `02bcf68`; carries the matrix through #235 while preserving profile-v19 counts and the real-GPU P0 blocker.
- **#237** sparse density-only probe contract merged as `66c1680`; a passing guardrail completes without starting CPU crossover work, with sparse still opt-in/default-off and no GPU claim.
- **#238** generate-AUTO environment cleanup tests merged as `95ecae0`; interrupted runs restore variables that were initially unset, with CPU-only coverage and no GPU claim.
- **#239** eager raw-score decode contract merged as `34a0d9f`; the eager backend joins the no-softmax/no-scale decode parity contract, with no GPU timing or performance claim.
- **#240** `opt/auto-thr-v6` merged as `2a004c3`; CPU-safe coverage keeps explicit backends authoritative on both AUTO gates, with no GPU timing or performance claim.
- **#241** top-k-one sampler-layout coverage merged as `8a2ea37`; the CPU probe now covers the `top_k=1` boundary alongside scaled, narrow, and full-vocabulary cases, with no GPU timing or performance claim.
- **#242** distinct-Q/K score-V parity coverage merged as `c20f440`; eager, blocked, and online paths are checked against the expected raw score×V result for non-aliased query/key tensors, with no GPU timing or performance claim.
- **#244** blocked RoPE tile contract merged as `9d2fcb5`; nonpositive tiles are rejected before the T=1 shortcut and covered at T=1 and T>1, with no GPU timing or performance claim.
- **#245** successful compile-backward probe cleanup merged as `4047ce9`; a successful `train_bwd` probe restores eval caller mode and clears parameter gradients, with CPU-only coverage and no GPU compile result.
- **Current CUDA-build tip** `d6a5455` deepens the missing-`nvcc` skip contract; a directory named `nvcc` remains a clear CPU-safe no-op before CUDAExtension setup, with no GPU build or timing claim.
- **#247** shared-V strict-tril backward contract landed as `77a1ca4`; CPU-only parity coverage, no GPU timing claim.
- **#248** docs refresh landed as `0c7c2f1`; carries the matrix through the prior tip.
- **#249** CUDA-mirror packed decode gradients landed as `567c1bf`; the CPU CUDA-mirror tiled path now has forward and Q/K/V autograd parity coverage for packed per-head T=1 decode, with no CUDA timing or performance claim.
- **Profile-v20** landed as `f4c7cab`; matched CPU counts remain flat versus profile-v19 (`copy_`=2/12/394, `cat=0`, `contiguous=0`), with no GPU timing or speedup claim.
- **#251** `opt/gpu-measure-v5` merged as `84bd3f6`; forced-CPU smoke selects CPU even when CUDA is available, with schema-v3 contract coverage and no GPU timing claim.
- **#252** `opt/prefetch-v6` merged as `0baa35d`; the CPU H2D gate returns before probing CUDA availability, with no H2D timing or overlap claim.
- **#253** `opt/docs-v61` merged as `53491cf`; refreshes the matrix through profile-v20, #251, and #252.
- **#254** `opt/amp-train-v7` merged as `94a4e55`; an unavailable CPU dtype switch preserves the complete active AMP configuration, with no GPU throughput claim.
- **#255** `opt/gen-bench-v6` merged as `1b67d13`; interrupted generate-implementation sweeps restore an initially absent environment variable, with no GPU performance claim.
- **#257** `opt/sparse-v7` merged as `042a01d`; covers the CPU density-only sparse-probe crossover boundary while sparse remains opt-in/default-off.
- **#256** `opt/blocked-tile-v7` merged as `b00dbd1`; extends CPU blocked/online gradient parity to batched `B=2` wide-head layouts, with no GPU timing claim.
- **#258** `opt/auto-thr-v7` merged as `6abd53b`; an unset cold AUTO threshold follows later decode-threshold changes, with CPU-only contract coverage and no GPU timing claim.
- **#259** `opt/scorev-v8` merged as `f5a9cea`; distinct query/key score-V paths preserve Q/K/V gradients across blocked and online CPU coverage, with no GPU timing claim.
- **#260** `opt/online-v8` merged as `9b6d0a8`; signed raw-score decode with independent per-head values remains CPU-parity coverage across supported dispatches, with no GPU timing claim.
- **#262** sampler-layout overflow coverage landed as `1384ca5`; the CPU contract now covers the `top_k` greater-than-vocabulary clamp boundary without changing attention semantics or defaults.
- **#263** mixed-dtype RoPE output coverage landed as `7c0e969`; CPU entrypoints cover fp16 RoPE math written into fp32 cache-style output slots across sequence shapes, with no GPU timing claim.
- **#264** failed compile-probe eval cleanup landed as `3747356`; a failed `train_bwd` probe restores eval caller mode and clears partial parameter gradients before retaining the eager module.
- **#265** CUDA_PATH missing-nvcc skip coverage landed as `6d6aaab`; a stale non-executable `CUDA_PATH` `nvcc` stub remains a clear CPU-safe no-op before CUDAExtension setup.
- **#266** strict-tril backward boundary coverage landed as `cbf0831`; CPU tests now cover first, interior, and final query rows across the supported dispatches and V layouts.
- **#268** tiled packed per-head decode gradient coverage landed as `aebc449`; CPU tests cover blocked, online, and Triton-fallback autograd parity for capacity-strided K/V views.
- **#270** `dae2e07` adds explicit `BDH_PREFETCH_H2D=0` CPU-safe coverage: the opt-out returns before CUDA availability probing or stream/event allocation; no H2D timing or overlap claim.
- **#271** `75597c9` marks run-level unavailable-device GPU summaries with `status: skip` under schema v4; no GPU timing or speedup evidence.
- **#272** `b9bf6c8` makes an enforced sparse-density guardrail failure terminal before CPU crossover work; sparse remains opt-in/default-off with no GPU sparse claim.
- **#273** `3b3c082` adds deterministic blocked/online tile-boundary coverage for raw strict-tril score×V semantics; CPU-only tests with no GPU timing or performance claim.
- **Current tip** `3b3c082` carries the flat profile-v20 counts plus the CPU-only contracts through #273; the matrix remains CPU/docs evidence only and the real-GPU P0 blocker is unchanged.

Related early landings without a #1–#33 slot (still on main, documented in notes):

- **`opt/sparse-relu`** — experimental `bdh_sparse.py` helpers; **default OFF**
- **`opt/cache-pack`** — preallocated KR/V CacheManager (superseded/deepened by #20)

---

## Defaults vs opt-in (cheat sheet)

| Knob | Production-safe default | When to flip |
|------|-------------------------|--------------|
| `BDH_ATTN_IMPL` | `eager` | GPU after microbench win; or `blocked` for peak-mem experiments |
| `BDH_ATTN_AUTOGRAD` | off / unset | `1` when training with non-eager attn; blocked/online → tiled analytic bwd |
| `BDH_ATTN_AUTO` | off / unset | `1` for long-T cold + long-S decode→triton (CUDA) or blocked (#55/#75/#77/#120); strict decode/cold gates, `THRESHOLD` default 512, optional `COLD_THRESHOLD` |
| `BDH_ROPE_IMPL` | `eager` | `fused` after GPU RoPE bench |
| `BDH_COMPILE` / `MODE` / `FULLGRAPH` | `0` / `default` / `0` | CPU: `COMPILE=1` **only with `IMPL=eager`** + `MODE=default` (#46/#49/#63); optional `FULLGRAPH=1` (#84; soft-fallback); `reduce-overhead` needs GPU CUDA graphs; GPU inductor still open |
| `BDH_PREFETCH_ASYNC` | `1` | `0` for synchronous preload / A-B; GPU pin/H2D overlap still needs measurement |
| `BDH_PREFETCH_H2D` | `1` on CUDA | `0` to keep H2D on the caller stream; CPU no-op; GPU overlap still needs measurement |
| `BDH_AMP_DTYPE` | `float32` | `bf16`/`fp16` on **CUDA** train boxes (CPU = smoke only) |
| `BDH_AMP_FORWARD_ONLY` | `0` | `1` for logits-only autocast + fp32 CE |
| Sparse ReLU | OFF | Only with `BDH_SPARSE_PROBE=1` for an explicit probe; enable production path only after density + GPU sparse-kernel win |

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
- Treating CPU density or sparse crossover as a GPU sparse-kernel result; `BDH_SPARSE_PROBE` remains explicit opt-in