# OPT backlog — ranked remaining work

Private sandbox only (`katulevskiy/bdh-gpu-opt`). **Do not** PR to `pathwaycom/bdh`.

Constraint (hard): attention stays **raw scores** × **strict lower-triangular**
`tril(diagonal=-1)` — **no softmax**, **no `1/√d`**, **no**
`F.scaled_dot_product_attention`.

Profile source: `benchmarks/profile_forward.py` on CPU
(`torch 2.14.0+cu130`, `cuda=False`), profile tip `b126d77` / documented tip `0739807` (post #64–#66 cuda-decode-v3 + docs-matrix + log-sync; #67 docs refresh; #68 profile-v6; #69 rope-decode), cfg `layers=4 d=128 nh=4 B=4 T=128`,
generate prompt=16 / new=32. Absolute ms are **profiler-inflated**; use **%
self CPU** and call counts. Re-run on GPU before claiming kernel wins.

Post-#64–#66 re-profile (`opt/profile-v6`): generate still **0× `aten::cat`**; forward **0× `aten::contiguous`**; default eager still full T×T `bmm`+`tril`. #58 layout-v2 still visible on generate (`mm`/`linear`); #64–#66 not exercised on short default window (CUDA decode / train log). **GPU still the blocker.** See `OPT_NOTES.md` § opt/profile-v6.

Post-#55–#58 re-profile (`opt/profile-v5`): generate still **0× `aten::cat`**; forward **0× `aten::contiguous`**; default eager still full T×T `bmm`+`tril`. #58 layout-v2 visible on generate (`mm`/`linear`); #55–#57 opt-ins not exercised on short default window. **GPU still the blocker.** See `OPT_NOTES.md` § opt/profile-v5.

Post-#48/#49 re-profile (`opt/profile-v4`): generate still **0× `aten::cat`**; forward **0× `aten::contiguous`**; default eager still full T×T `bmm`+`tril`. gen-host host self ~1.4% (was ~26% in v3). **GPU still the blocker.** See `OPT_NOTES.md` § opt/profile-v4.

Post-#36/#39 re-profile (`opt/profile-v3`): generate still **0× `aten::cat`**; forward **0× `aten::contiguous`** (mlp-fuse); default eager still pays full T×T `bmm`+`tril`. See `OPT_NOTES.md` § opt/profile-v3.

Post-#19–#22 re-profile (`opt/profile-v2`): generate **0× `aten::cat`**; default
eager still pays full T×T `bmm`+`tril`. See `OPT_NOTES.md` § opt/profile-v2.

## Already landed (main)

- Train logging sync deepen: `TrainLossLogger` + CUDA deferred D2H (`opt/log-sync` #66); defaults LOG_FREQ=100 / ASYNC=1

- Analytic attn train path (`BDH_ATTN_AUTOGRAD` / `StrictTrilAttnFn`) — **landed** `opt/attn-bwd-train` (#39): cold+multi-token wiring, `bench_attn_bwd.py`, grad parity @ dropout=0
- Blocked/online + **tiled** analytic bwd train (`opt/blocked-autograd` #41): no full T×T in fwd or bwd; `online` alias; GPU train A/B still open

- RoPE without `stack→view`; skip redundant dtype casts
- RoPE cos/sin table cache by (T, head_dim, device, dtype) (`opt/rope-cache` #18)
- KV-style cache + incremental `generate`
- MLP `permute → contiguous → view`
- Vectorized `train.get_batch` (+ pin/non_blocking CUDA; optional DataLoader workers #14)
- BatchPrefetcher v2: host-thread queue double-buffer + numpy producer gather (`opt/prefetch-v2` #42); synthetic overlap ~1.7×; e2e CPU ~noise; GPU pin/H2D overlap open
- Triton + blocked pure-PyTorch attn dispatch (`BDH_ATTN_IMPL`) — CPU blocked still < eager after `opt/blocked-vec`; GPU unmeasured
- Experimental sparse ReLU matmul (default off); **short-train density + CPU crossover** (`opt/sparse-probe`) — keep OFF
- Peak-mem probe eager vs blocked vs online (`opt/attn-mem-probe`): mid-T peak↓ even when wall slower; default eager unchanged
- CUDA extension scaffold for tril score×V (optional build)
- Weight layout: `(B,T,nh,N)` encoder einsum, decoder view+`F.linear`, optional bias fuse (#13)
- Contiguous vocab proj + optional `tie_weights` (`opt/embed-tie` #16)
- `F.dropout` + identity `p=0` for compile (`opt/dropout-fuse` #17)
- CUDA/C++ decode vs packed KR/V under `BDH_ATTN_IMPL=cuda` (scaffold; CPU ref always) (#15)
- Blocked/Triton decode polish vs packed KR/V (`opt/triton-decode2` #19)
- CacheManager v2: layer-contiguous KR/V, page growth, generate **0× aten::cat** (`opt/cache-v2` #20)
- Decode-copy: `reserve` + in-place RoPE into packed KR; `narrow` past views; no intermediate `.to()` on `copy_` (`opt/decode-copy`)
- Cache-page: geometric page growth + empty+prefix copy_; `ensure_capacity`; fewer realloc copies on long S (`opt/cache-page`)
- Decode GEMM polish: blocked/Triton/CUDA T=1 score×V vs packed KR/V — broadcast-V, Triton staging, tiled CUDA (`opt/decode-gemm`)
- Decode-mm: `_two_gemm_decode` + CUDA Tq=1/`DECODE_TILE_N` + B=1 lm_head `mv`; default eager (`opt/decode-mm`)
- Decode-online-v2: blocked/online T=1 tight oneshot + long-S tiles; peak helper (`opt/decode-online-v2`)
- Attn-auto: opt-in `BDH_ATTN_AUTO` long-S T=1 decode → blocked (thr=512 from #55); cold stays IMPL (`opt/attn-auto`)
- Triton decode-v3: long-S tiles + Q-hoist scaffold; AUTO prefers triton when CUDA+Triton else #55 blocked (`opt/triton-decode-v3`)
- CUDA decode-v3: adaptive DECODE_TILE_N (32/64/128 GPU; CPU refs ≤512) + TQ1 Q-hoist; ≡ blocked parity (`opt/cuda-decode-v3`)
- Online fused strict-tril score×V (no full T×T) under `blocked`/`online` (`opt/fuse-scorev` #21)
- Vectorized CPU blocked/online tiles (`opt/blocked-vec`) — beats old Python-row blocked wall; still < eager on CPU
- GPU attn microbench harness `benchmarks/bench_gpu_attn.py` (eager|blocked|online|triton|cuda; clean CPU skip) (`opt/gpu-bench`)
- Cold CUDA tril score×V **tiled online** scaffold (no global T×T; `opt/cuda-cold`) — GPU measure still P0
- `torch.compile` harden: train probe, graph-break docs, CPU inductor parity (`opt/compile-harden` #22)
- Compile-friendly LN+residual: `F.layer_norm` only, no `is_grad_enabled` product branch (`opt/ln-compile` #30); deepen: inner-LN buffer `add_` reuse (`opt/ln-deepen`)
- CPU train-step `BDH_COMPILE=0` vs `1` microbench via `maybe_compile` (soft-skip if inductor missing) (`opt/compile-bench`)
- Generate microbench `benchmarks/bench_generate.py`: CacheManager generate × attn impls eager|blocked|triton|cuda when avail; honest CPU numbers (`opt/gen-bench`)
- OPT_STATUS operator matrix + README pointer (`opt/docs-matrix` #34)
- MLP fuse: bias+ReLU + merge without unconditional `.contiguous()` (`opt/mlp-fuse` #36); profile-v3: `aten::contiguous`=0
- CUDA attn CPU refs + build smoke deepen (`opt/cuda-ref-v2` #37)
- Profile-v3 docs refresh (`opt/profile-v3` #40) — documented tip `d3ff475`; profile source tip `b160469`; no code or GPU measurement
- OPT status matrix refresh (`opt/docs-matrix-v2` #43) — docs-only through #40
- Generate Python/host tax (`opt/gen-host` #44) + `opt/gen-sample` T=1 lm_head+sample fuse; remaining P1 is GPU decode GEMM
- Compile × blocked × AUTOGRAD train matrix + SelfAttnFn Dynamo fix (`opt/compile-blocked` #46) — CPU; compile+blocked not a win
- Encoder fuse attempt (`opt/encoder-fuse` #47): default stays einsum; optional bias → `F.linear` epilogue; always-on linear lost on CPU (weight transpose-copy)
- Compile guidance: warn COMPILE+blocked|online|triton; recommend COMPILE=1 only with eager on CPU (`opt/compile-guidance` #49) — GPU compile still P1
- Profile-v4 docs refresh (`opt/profile-v4` #50) — tip `527ead2`; profile source `c7a7471`; cats=0; contiguous=0; no GPU measurement
- Profile-v5 docs refresh (`opt/profile-v5`) — tip `fc9283d`; profile source `fc9283d`; cats=0; contiguous=0; no GPU measurement
- Docs matrix v7 refresh (`opt/docs-matrix-v7` #67) — docs-only through #66; documented tip `8439c06`
- Profile-v6 docs refresh (`opt/profile-v6` #68) — tip `8439c06`; profile source `b126d77`; cats=0; contiguous=0; no GPU measurement
- T=1 RoPE apply deepen (`opt/rope-decode` #69) — `rope_rotate_t1` pair stores; table-pair / last-position cis reuse; CPU wall ~0.83× (no win claim); GPU fused/Triton T=1 open
- Docs matrix v5 refresh (`opt/docs-matrix-v5` #60) — docs-only through #59; documented tip `71ba3a4`
- Generate host-test repair (`opt/fix-gen-host-test` #61) — updated AUTO decode environ-get floor; semantics unchanged
- Residual LN deepen: reuse inner LN out via `add_` (fewer add temps; `F.layer_norm` #30 path kept) (`opt/ln-deepen`) — CPU e2e ~noise
- Compile reduce-overhead guidance (`opt/compile-reduce` #63) — CPU mode matrix and non-CUDA warning; GPU CUDA graphs still open
- Docs matrix v6 refresh (`opt/docs-matrix-v6` #65) — docs-only through #64; documented tip `4558501`

## Ranked next work

| P | Item | Why (from profile / notes) | Target | Risk |
|---|------|----------------------------|--------|------|
| **P0** | **Measure Triton/CUDA fused tril-score×V on real GPU** | Default **eager** still (**GPU blocker**; profile-v6): attn `mul`~26% `bmm`~23% `copy_`~16% `tril`~0.8%; forward `copy_`~25% `mm`~19% `bmm`~18% `mul`~12%. | A100/H100: `bench_gpu_attn.py` (+ fused score×V) | Env blocker |
| **P0** | **Cold Triton tile/staging validation** | **Landed `opt/triton-cold`:** adaptive power-of-2 tiles, fused strict-tril score×V, and broadcast-V staging; GPU validation remains open. | A100/H100 microbench; bit-identical | Env blocker |
| **P1** | **`torch.compile` GPU train-step next** | Forward still `copy_` ~20%, `mm` ~12%, `mul`/`mul_` ~12%, LN ~4%. **CPU**: recommend `COMPILE=1` **only with eager** + `MODE=default` (#46/#49/#63; blocked/`reduce-overhead` warn). **GPU inductor / CUDA graphs still unmeasured**. | A100/H100: `BDH_COMPILE=0` vs `1` + `MODE=default` vs `reduce-overhead` via `benchmarks/bench_train_step.py` | Low |
| **P1** | **Decode GEMM / copy tax on generate** | **Host tax cut** #44+#48; **decode-mm** + **decode-online-v2** + **attn-auto** + **triton-decode-v3** + **cuda-decode-v3** scaffold (AUTO→triton when CUDA; cuda adaptive tiles). CPU long-S **IMPL=blocked** e2e win; opt-in AUTO thr=512. **`opt/gen-long-bench`**: e2e AUTO 0/1 @ S∈{256,1024,2048} — AUTO fires + parity, but AUTO e2e wall ~parity on CPU (eager prefill dominates); IMPL=blocked still 1.14–1.22×. Remaining = **GPU** measure / re-tune thr. Default still eager. | A100/H100: `bench_generate.py --mode auto-ab` + `--mode impls` + `bench_gpu_attn.py --mode decode`; keep cat-free | Medium |

| **P2** | **Fused RoPE kernel** | Attn: `mul`/`copy_` from strided rotate. **Cached tables** (#18); **fused rotate landed** `opt/rope-fuse` (`BDH_ROPE_IMPL`). | GPU Triton microbench still open | Low–medium |
| **P2** | **Sparsity follow-through** | **Density measured** `opt/sparse-probe`: short-train x~27% xy~12% @150 steps (≫ paper 5%); CPU sparse **never reliably beat dense** → **keep OFF**. GPU sparse still open. | GPU sparse bench if density ≪10% | Speculative |
| **P2** | **Memory layout** | Forward contiguous/clone copies significant. | **Landed #13+#16+#36+#47 + `opt/layout-v2`** — eval cached encoder `F.linear`; train einsum; channels_last rejected; CPU short-T win / long-T ~noise | Low |
| **P3** | **Hardware / dtype** | **Landed `opt/bf16-train`+#54 `opt/amp-deepen`:** opt-in `BDH_AMP_DTYPE` + GradScaler fp16+CUDA; CPU smoke/bench honest (often slower); `BDH_AMP_FORWARD_ONLY`. GPU train throughput still open. | GPU box microbench | Env |
| **P3** | **Profiler CI artifact** | Traces gitignored; optional nightly upload. | CI | N/A |

### Done (struck from ranked queue)

| Was | Status |
|-----|--------|
| Memory layout / mlp-fuse | **Landed** #13+#16+#36 — profile-v3: forward **`aten::contiguous`=0**; CPU ~noise |
| Generate × attn impls microbench | **Landed** #35 `opt/gen-bench` — GPU e2e still P1 |
| OPT_STATUS docs matrix | **Landed** #34 `opt/docs-matrix` |
| CUDA attn CPU refs deepen | **Landed** #37 `opt/cuda-ref-v2` |
| Vectorized CPU blocked tiles | **Landed** #38 `opt/blocked-vec` — still < eager; GPU measure open |
| Re-profile post-mlp-fuse | **Landed** #40 `opt/profile-v3` — profile source tip `b160469`; documented tip `d3ff475`; cats=0; contiguous=0 |
| Re-profile post-gen-sample | **Landed** `opt/profile-v4` #50 — tip `527ead2`; profile source `c7a7471`; cats=0; contiguous=0; gen-host host self ↓ |
| Re-profile post-#55–#58 | **Landed** `opt/profile-v5` #59 — tip `fc9283d`; cats=0; contiguous=0; layout-v2 visible on generate |
| Re-profile post-#64–#66 | **Landed** `opt/profile-v6` #68 — tip `8439c06`; profile source `b126d77`; cats=0; contiguous=0; #64–#66 off short default window |
| Analytic tril attn train path | **Landed** #39 `opt/attn-bwd-train` — default AUTOGRAD off; eager profile unchanged |
| Blocked/online tiled analytic bwd | **Landed** #41 `opt/blocked-autograd` — blocked|online+AUTOGRAD=1; dense M-recompute only for eager |
| Batch prefetch overlap | **Landed** #42 `opt/prefetch-v2` — host queue/numpy producer; GPU pin/H2D overlap remains open |
| OPT status docs refresh | **Landed** #43 `opt/docs-matrix-v2` — matrix through #40 |
| Cache packing / fewer cats | **Done** #19–#20 — generate `aten::cat` **0** (was ~10% self / ~864 calls pre-pack) |
| Fuse score×V epilogue (no materialize T×T) | **Landed** #21; **CPU vectorized** `opt/blocked-vec` (~18–36× vs old blocked wall; still slower than eager) |
| `torch.compile` / inductor CPU harden | **Landed** #17+#22+#31+#46+#49+#63+#70; COMPILE+eager only win on CPU; warn on COMPILE+blocked; **CPU `reduce-overhead` not useful** (no CUDA graphs); dropout=0 / eval identity hardened (#70); remaining = **GPU** measure (P1) |
| Fused RoPE rotate (`BDH_ROPE_IMPL`) | **Landed** `opt/rope-fuse` — default eager; fused PyTorch + optional Triton |
| T=1 RoPE apply deepen | **Landed** `opt/rope-decode` — `rope_rotate_t1` + table pairs / cis reuse; CPU wall ~noise; GPU open |
| Decode GEMM vs packed KR/V | **Landed** `opt/decode-gemm` — blocked/triton/cuda decode polish; GPU measure still open |
| Decode-mm T=1 / lm_head mv | **Landed** `opt/decode-mm` — `_two_gemm_decode`, CUDA Tq=1 + `DECODE_TILE_N`, B=1 `mv`; CPU wall ~noise; GPU open |
| Decode-online-v2 blocked T=1 | **Landed** `opt/decode-online-v2` — tight broadcast oneshot; long-S peak↓ + wall↑ on CPU; GPU open |
| Attn-auto long-S decode | **Landed** `opt/attn-auto` — `BDH_ATTN_AUTO` + thr=512; default eager; GPU threshold re-tune open |
| Long-S generate AUTO A/B | **Landed** `opt/gen-long-bench` — AUTO fires @ S>512; e2e AUTO ~parity (CPU); IMPL=blocked 1.14–1.22×; GPU open |
| Triton decode-v3 | **Landed** `opt/triton-decode-v3` — long-S tiles/Q-hoist scaffold; AUTO prefers triton when avail; CPU→#55 blocked; GPU measure open |
| CUDA decode-v3 | **Landed** `opt/cuda-decode-v3` — adaptive DECODE_TILE_N + TQ1 Q-hoist; ≡ blocked parity; GPU measure open |
| Memory layout / embed path | **Landed** #12–#13+#16 |
| Generate Python/host tax | **Landed** #44 `opt/gen-host`; follow-up `opt/gen-sample` fuses T=1 lm_head+sample / top-k (large-V top_k ~1.5×; default V wall ~noise). GPU decode GEMM remains P1 |

| **P2** | **Analytic attn train on GPU** | CPU blocked|online + tiled analytic bwd landed (`opt/blocked-autograd` #41). **GPU** train-step with `IMPL=blocked|triton|cuda` + AUTOGRAD=1 unmeasured. | A100/H100 `bench_attn_bwd.py` | Low |

## Explicit non-goals

- Softmax / diagonal inclusion
- PRs to `pathwaycom/*`
- Claiming GPU speedups from CPU profiler absolute times
- Defaulting `BDH_ATTN_IMPL=blocked` on CPU (still slower than eager after `opt/blocked-vec`; use for peak-score memory / parity)
- Recommending `BDH_COMPILE=1` with `blocked`/`online`/`triton` on CPU (#46 regression; #49 warns)
- Recommending `BDH_COMPILE_MODE=reduce-overhead` on CPU (no CUDA graphs; #63 documents)
- Re-introducing `aten::cat` in `generate` / packed cache path

## Suggested order

1. GPU: bench eager vs blocked/online vs Triton vs CUDA fused score×V (cold + T=1 decode)
2. GPU: end-to-end `bench_generate.py` (CacheManager) across attn impls + `bench_gpu_attn.py --mode decode`
3. GPU: `BDH_COMPILE=0` vs `1` train-step (CPU: only eager is a win; harness in `bench_train_step.py`; try `reduce-overhead` on CUDA)
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
(`BDH_COMPILE=0` vs `1`, plus `MODE=default` vs `reduce-overhead` via
`train.maybe_compile`; soft-skip if inductor/CXX missing). On CPU,
`reduce-overhead` is **not useful** (no CUDA graphs) — measured for honesty
only (#63). **Do not** treat CPU medians as GPU / CUDA-graph wins.

On an A100/H100 box:

```bash
# eager vs compiled train-step (default mode=default, probe=train)
BDH_BENCH_COMPILE=1 python benchmarks/bench_train_step.py

# MODE matrix (default vs reduce-overhead; CUDA graphs need GPU + static B×T)
BDH_BENCH_COMPILE_MODE=1 python benchmarks/bench_train_step.py

# Force reduce-overhead arm only via env (still runs full harness sections)
BDH_BENCH_COMPILE=1 BDH_COMPILE_MODE=reduce-overhead python benchmarks/bench_train_step.py
```

Record: GPU name, torch/CUDA, median ms for `BDH_COMPILE=0` and `=1`,
`MODE=default` vs `reduce-overhead`, whether `_orig_mod` stuck (true compile)
vs soft fallback. Private repo only — never `pathwaycom/*`.

## Generate microbench — CacheManager × attn impls + long-S AUTO (`opt/gen-bench` / `opt/gen-long-bench`)

End-to-end `BDH.generate` (packed KR/V CacheManager, cat-free) across
`BDH_ATTN_IMPL=eager|blocked|triton|cuda` (`--mode impls`, default). Labels
**effective** backend (triton→blocked on CPU; cuda→`cuda_ref` without native
ext). Prints median ms, tok/s, tokens-match-eager, `aten::cat` count.

`--mode auto-ab` (`opt/gen-long-bench`): prompt `S∈{256,1024,2048}` ×
`BDH_ATTN_AUTO=0|1` (thr=512). Cold stays eager; decode switches when
`past_len > thr`. Validates `#55`/`#56` outside score×V microbench.

```bash
python benchmarks/bench_generate.py
python benchmarks/bench_generate.py --prompt 32 --new 64 --warmup 2 --iters 5
python benchmarks/bench_generate.py --mode auto-ab --prompts 256,1024,2048 --new 8
python benchmarks/bench_generate.py --mode auto-ab --prompts 256,1024,2048 --new 64
# A100/H100:
python benchmarks/bench_generate.py --device cuda --warmup 5 --iters 20
python benchmarks/bench_generate.py --mode auto-ab --device cuda
```

**CPU honesty (this box / tip `962a3b6`):**
- `--mode impls` short prompt: medians ~noise vs eager; match; `aten::cat=0`
- `--mode auto-ab`: AUTO fires @ S>512; tokens match; cats=0; **e2e AUTO wall ~parity**
  (eager prefill dominates); **IMPL=blocked** still **1.14× @1024 / 1.22× @2048**
- **Do not** claim Triton/CUDA / AUTO e2e wins from CPU. GPU thr re-tune = P1.
Private repo only — never `pathwaycom/*`.
