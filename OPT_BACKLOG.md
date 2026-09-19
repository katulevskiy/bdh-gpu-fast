# OPT backlog — ranked remaining work

Private sandbox only (`katulevskiy/bdh-gpu-opt`). **Do not** PR to `pathwaycom/bdh`.

Constraint (hard): attention stays **raw scores** × **strict lower-triangular**
`tril(diagonal=-1)` — **no softmax**, **no `1/√d`**, **no**
`F.scaled_dot_product_attention`.

Profile source: `benchmarks/profile_forward.py` on CPU
(`torch 2.14.0+cu130`, `cuda=False`), profile-v12 source tip `f34adc0` / current code tip `aea2501` after #116 online-decode-t1-v2, #117 compile-train-v2, #118 docs-v27, #119 amp-train-v2, #120 auto-thr-v2, #121 profile-v12, the closed #122 docs-v28 attempt, and #123 sparse-v2; profile-v11 source tip `4963b0f`; #115 docs-matrix-v26; #114 cuda-cold-v3; #113 docs-matrix-v25; profile-v10 source tip `1363794`; docs #106/#107 on #104/#105; rebased documented code tip `234de2a`; profile-v9 source tip `8e7a4d2`; post #85 cache-page-bench, #86 docs, #87 zerograd; #88 docs-v15; #89 profile-v8; #90 rope-fuse-v2; #91 docs-v16; #92 prefetch-h2d; #93 blocked-tile-v2; #94 docs-v17; #95 copy-tax-v1; #96 attn-bwd scaffold; #97 docs-v18; #98 profile-v9; #99 docs-v19; #100 scorev-fuse-v2; #101 docs-v20; #102 gen-copy-tax-v1; #103 docs-v21; #104 profile-v10; #105 ln-resid-v2; #106 docs through #104; #107 docs through #105; #108 triton-cold-v3; #109 docs-v23; #110 gen-vcopy-v1; #111 docs-v24; #112 profile-v11; #84 compile-fullgraph and earlier profile-v7 follow-ups), cfg `layers=4 d=128 nh=4 B=4 T=128`,
generate prompt=16 / new=32. Absolute ms are **profiler-inflated**; use **%
self CPU** and call counts. Re-run on GPU before claiming kernel wins.

Post-#95 re-profile (`opt/profile-v9`, source `8e7a4d2`; current docs tip `1363794` after #96–#102): forward warmed harness `aten::copy_` is 48 over three active calls (16/call), while an isolated one-forward check reproduces #95's **18**; forward and generate remain `aten::cat=0` and `aten::contiguous=0`. Generate `aten::copy_` is 1,674 over three active calls (558/call), so generate remains copy_-heavy. CPU-only evidence; **GPU still the blocker.** See `OPT_NOTES.md` § opt/profile-v9.

Post-#100 score×V deepen (`opt/scorev-fuse-v2`, source `90b609f`): CPU inference/no-grad blocked/online tiles use direct `baddbmm(..., out=target)` score×V epilogues, T=1 decode reuses the flattened output view, and the broadcast-V oneshot budget is 1024 score elements. Cold/decode CPU behavior remains shape-dependent; this is lower-peak/epilogue evidence, not GPU timing. Default eager remains unchanged; **GPU still the blocker.** See `OPT_NOTES.md` § opt/scorev-fuse-v2.

Post-#102 generate copy-tax cut (`opt/gen-copy-tax-v1`, source `1363794`): fp32 RoPE pair stores use a complex-view copy, and generate samples into the preallocated output via `idx_out`/`out.narrow`. CPU generate `aten::copy_` falls ~558→~398 while `aten::cat` stays 0; defaults remain unchanged. This is CPU call-count evidence only, not GPU timing; **GPU still the blocker.**

Post-#102 re-profile (`opt/profile-v10`, source `1363794`, rebased onto `234de2a`): attention `bmm` 30.87% / `mul` 30.19% / `copy_` 19.04%; forward `bmm` 36.00% / `mul` 21.86% / `mm` 19.15% / `copy_` 11.27% with isolated forward `copy_`=18; generate `mm` 14.76% / `bmm` 12.80% / `matmul` 7.33%, with `copy_` 1,194 over three active calls (398/call). Forward and generate remain `aten::cat=0` and `aten::contiguous=0`; CPU-only evidence, **GPU still the blocker.** See `OPT_NOTES.md` § opt/profile-v10.
Post-#105 residual-LN audit (`opt/ln-resid-v2`): the current eager path is already the safe deepen (`LN(yMLP)` → `y.add_(x)` → `LN(y)`). CPU residual-only probes show exactly 2× `aten::native_layer_norm` + 1× `aten::add_`, with no residual `add`, `copy_`, `cat`, `to`, or `_to_copy`; fp32/fp16/bf16 probes show no explicit dtype bounce. No further default-safe change was found; defaults and generation behavior remain unchanged, and GPU validation is still open.

Post-#108 Triton cold v3 (`opt/triton-cold-v3`, source `e11c187`): normal long-T cold tiles remain 128, while wide heads (`N>64` or `D>128`) stay at bounded 64×64 query/key tiles unless explicitly overridden. The flattened CPU blocked fallback keeps shared `V=(B,1,T,D)` as a `(B,T,D)` view instead of materializing `(B*H,T,D)` staging; parity coverage passes (45 passed, 3 skipped on CPU). No CUDA/Triton execution is available here; **GPU measurement and cold Triton validation remain P0 blockers.** Defaults remain eager and AUTO-off.

Post-#110 generate V/sampler probe (`opt/gen-vcopy-v1`, source `4963b0f`): T>1 RoPE now uses `_store_pairs` and the top-k path can use `gather(out=)`, reducing CPU generate `aten::copy_` ~398→~394 while `aten::cat` remains 0. **Probe ceiling:** remaining V-slot writes (~132/gen) are required cache snapshots, and multinomial internals (~128/gen) are RNG/layout-coupled; removing them would require changing cache layout or the default RNG stream. This is CPU-only call-count evidence; defaults remain unchanged and **GPU measurement is still the P0 blocker.** See `OPT_NOTES.md` § opt/gen-vcopy-v1.

Post-#110 re-profile (`opt/profile-v11`, source `4963b0f`, rebased onto `acf6e09`): attention `mul` 25.29% / `bmm` 22.61% / `complex` 16.19% / `copy_` 14.68%; forward `bmm` 28.69% / `mm` 24.63% / `mul` 16.40% / `complex` 10.23% / `copy_` 6.65% with warmed `copy_`=12/call and isolated `copy_`=14; generate `mm` 21.92% / `bmm` 13.60% / `mul` 3.03%, with `copy_` 1,182 over three active calls (394/call; isolated 396). Forward and generate remain `aten::cat=0` and `aten::contiguous=0`; CPU-only evidence, **GPU still the blocker.** See `OPT_NOTES.md` § opt/profile-v11.

Post-#111 docs refresh (`opt/docs-matrix-v24`, source `acf6e09`): align `OPT_STATUS.md` / `OPT_BACKLOG.md` through #110 and carry the profile-v11 evidence forward. Docs-only; no code or GPU evidence.

Post-#113 docs refresh (`opt/docs-matrix-v25`): align `OPT_STATUS.md` / `OPT_BACKLOG.md` through #112 and carry the profile-v11 evidence forward. Docs-only; no code or GPU evidence.

Post-#114 CUDA cold v3 (`opt/cuda-cold-v3`): align wide-head CUDA cold tile bounds with Triton #108, retain CPU reference parity, clean CUDA skips, and add a C++20 build smoke. No GPU timing or speedup evidence was added; **GPU measurement and cold CUDA validation remain P0 blockers.** Defaults remain eager and AUTO-off.

Post-#115 docs refresh (`opt/docs-matrix-v26`): refresh `OPT_STATUS.md` / `OPT_BACKLOG.md` through #114. Docs-only; no code or GPU evidence.

Post-#116 online decode T=1 (`opt/online-decode-t1-v2`): flatten broadcast-V score tiles over `B*H` and write inference/no-grad score×V directly into the output with `baddbmm(..., out=target)` for oneshot and long-S paths; retain the graph-safe autograd fallback, strict lower-triangular semantics, `S=0` zeros, cat-free generate, and default eager. CPU tests reported 507 passed, 18 skipped, 3 warnings; no GPU timing or speedup claim.

Post-#117 compile train v2 (`opt/compile-train-v2`): make `BDH_COMPILE_PROBE=train_bwd` the opt-in default, probe forward plus backward, soft-fallback to eager when backward targets are missing, and document the compiled forward/backward versus eager optimizer boundary. CPU A/B smoke reported 19.08 ms uncompiled vs 31.23 ms with the stronger probe; 506 tests passed and 18 skipped. No GPU/CUDA-graph or speedup claim; `COMPILE=0` remains the repository default.

Post-#118 docs refresh (`opt/docs-matrix-v27`): refresh `OPT_STATUS.md` / `OPT_BACKLOG.md` through #117. Docs-only; no code or GPU evidence.

Post-#119 AMP train v2 (`opt/amp-train-v2`): make CPU AMP capability checks transactional and explicit, soft-skip unsupported benchmark arms, expand full versus logits-only autocast coverage, and gate GradScaler to live CUDA float16 plus an enabled scaler. CPU validation is parity/bench evidence only; defaults remain fp32 and AMP-off, with no GPU throughput claim.

Post-#120 AUTO threshold v2 (`opt/auto-thr-v2`): centralize strict `past_len > decode_threshold` and `T > cold_threshold` gates, preserve explicit-backend non-override and AUTO-off eager defaults, and extend `bench_generate.py --mode auto-ab` with independent cold-threshold reporting and parity checks. CPU smoke retained token parity and `aten::cat=0`; no GPU timing or kernel claim was added, so GPU measurement and cold CUDA/Triton validation remain P0.

Post-#121 re-profile (`opt/profile-v12`, source `f34adc0`): attention `mul` 25.84% / `bmm` 22.99% / `complex` 15.11% / `copy_` 13.50%; forward `bmm` 28.06% / `mm` 23.22% / `mul` 16.39% / `complex` 11.32% / `copy_` 7.86% with warmed `copy_`=12/call and isolated `copy_`=14; generate `mm` 22.75% / `bmm` 14.65% / `matmul` 2.33%, with `copy_` 1,182 over three active calls (394/call; isolated 396). Forward and generate remain `aten::cat=0` and `aten::contiguous=0`; CPU-only evidence, **GPU still the blocker.** See `OPT_NOTES.md` § opt/profile-v12.

Post-#122 docs refresh (`docs/opt-status-backlog-v28`): the closed docs-only attempt through #120 is superseded by this v29 refresh, which carries the #121 profile evidence and #123 sparse result forward.

Post-#123 sparse v2 (`opt/sparse-v2`): gate the optional helper/benchmark behind `BDH_SPARSE_PROBE=1`; unset/false exits before training or timing, while ungated sparse encoder materialization raises an actionable error. The 150-step CPU density re-smoke observed x=0.2663 and xy=0.1137 with the conservative guardrail passing; the prior CPU sparse crossover remains a non-win, so dense BDH and sparse OFF are unchanged. No GPU timing or sparse-kernel claim was added; GPU measurement remains the P0 blocker.

Post-#85–#87 re-profile (`opt/profile-v8`, #89): attention `bmm` 23.43% / `mul` 21.92% / `copy_` 20.66%; forward `copy_` 23.63% / `mm` 22.79% / `bmm` 22.45% / `mul` 12.39%; generate `mm` 15.28% / `bmm` 15.27%. Generate remains **0× `aten::cat`**; forward and generate remain **0× `aten::contiguous`**; default eager still full T×T `bmm`+`tril`. #85–#90 do not alter this short default eval/generate window; #92 CUDA staging is not exercised on CPU, #93 is an opt-in cold path at T≥256, and #95 adds only CPU copy-call evidence (forward `copy_` 24→18; QR contig copies 8→0). No GPU timing or speedup claim was added. **GPU still the blocker.** See `OPT_NOTES.md` § opt/profile-v8.

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
- Cache-page-bench (#85): geometric vs linear grows/bytes microbench (`benchmarks/bench_cache_page.py`) (`opt/cache-page-bench`)
- Decode GEMM polish: blocked/Triton/CUDA T=1 score×V vs packed KR/V — broadcast-V, Triton staging, tiled CUDA (`opt/decode-gemm`)
- Decode-mm: `_two_gemm_decode` + CUDA Tq=1/`DECODE_TILE_N` + B=1 lm_head `mv`; default eager (`opt/decode-mm`)
- Decode-online-v2: blocked/online T=1 tight oneshot + long-S tiles; peak helper (`opt/decode-online-v2`)
- Attn-auto + prefill-blocked + auto-tune: opt-in `BDH_ATTN_AUTO` long-T cold + long-S decode → blocked|triton (shared thr=512; optional independent cold thr); (`opt/attn-auto` / `opt/prefill-blocked` / `opt/auto-tune`)
- Triton decode-v3: long-S tiles + Q-hoist scaffold; AUTO prefers triton when CUDA+Triton else #55 blocked (`opt/triton-decode-v3`)
- CUDA decode-v3: adaptive DECODE_TILE_N (32/64/128 GPU; CPU refs ≤512) + TQ1 Q-hoist; ≡ blocked parity (`opt/cuda-decode-v3`)
- Online fused strict-tril score×V (no full T×T) under `blocked`/`online` (`opt/fuse-scorev` #21)
- Vectorized CPU blocked/online tiles (`opt/blocked-vec`) — beats old Python-row blocked wall; still < eager on CPU
- GPU attn microbench harness `benchmarks/bench_gpu_attn.py` (eager|blocked|online|triton|cuda; clean CPU skip) (`opt/gpu-bench`)
- Cold CUDA tril score×V **tiled online** scaffold (no global T×T; `opt/cuda-cold` + `opt/cuda-cold-v2` + #114 `opt/cuda-cold-v3` adaptive/bounded long-T) — GPU measure and cold CUDA validation still P0
- Docs matrix v25 (#113) — docs-only refresh through #112; current docs align to #114 tip
- Docs matrix v26 (#115) — docs-only refresh through #114
- Online decode T=1 (#116) — opt-in blocked/online shared-V direct inference epilogue; CPU parity only
- Compile train v2 (#117) — backward-aware `train_bwd` probe default and eager optimizer boundary; CPU-only evidence
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
- Docs matrix v12 refresh (`opt/docs-matrix-v12` #83) — docs-only through #82; documented tip `03bc30b`
- Profile-v6 docs refresh (`opt/profile-v6` #68) — tip `8439c06`; profile source `b126d77`; cats=0; contiguous=0; no GPU measurement
- Profile-v7 docs refresh (`opt/profile-v7`) — tip `b8067f5`; profile source `ca5038f`; cats=0; contiguous=0; no GPU measurement
- T=1 RoPE apply deepen (`opt/rope-decode` #69) — `rope_rotate_t1` pair stores; table-pair / last-position cis reuse; CPU wall ~0.83× (no win claim); GPU fused/Triton T=1 open
- Docs matrix v9 refresh (`opt/docs-matrix-v9`) — docs-only through #73; documented tip `0215ca8`
- Docs matrix v10 refresh (`opt/docs-matrix-v10`) — docs-only through #75; documented tip `43948b7`
- Auto-tune follow-up (`opt/auto-tune` #77) — keeps shared threshold 512, adds optional cold threshold and operator recommendations; CPU A/B 1.25× @1024; GPU threshold re-check open
- Docs matrix v5 refresh (`opt/docs-matrix-v5` #60) — docs-only through #59; documented tip `71ba3a4`
- Generate host-test repair (`opt/fix-gen-host-test` #61) — updated AUTO decode environ-get floor; semantics unchanged
- Residual LN deepen: reuse inner LN out via `add_` (fewer add temps; `F.layer_norm` #30 path kept) (`opt/ln-deepen`) — CPU e2e ~noise
- Compile reduce-overhead guidance (`opt/compile-reduce` #63) — CPU mode matrix and non-CUDA warning; GPU CUDA graphs still open
- Compile fullgraph probe (`opt/compile-fullgraph` #84) — FULLGRAPH=1 × eager×AUTOGRAD; 0 Dynamo breaks on tip cold path; soft-fallback; GPU still open
- Docs refresh (#86) — docs-only alignment through #84; documented tip `006de27`
- Zero-grad train-path hardening (#87, `opt/zerograd`) — `clear_grads()` chokepoint, `set_to_none=True` + fused AdamW path, compile `train_bwd` probe, and 8 smoke tests; CPU re-smoke ~1.03× fused+set-to-none vs legacy; defaults unchanged
- Docs matrix v15 (#88) — refresh through #87; documented tip `f10bdd4`
- Profile-v8 (#89) — CPU profile evidence for the #85–#87 tip; no GPU measurements; defaults unchanged
- Rope-fuse-v2 (#90, `opt/rope-fuse-v2`) — no-expand/stack fused rotate, cached table-pair T=1 apply, blocked Triton CPU scaffold, and parity tests; default `BDH_ROPE_IMPL=eager`; GPU validation remains open
- Docs matrix (#91, `opt/docs-matrix-v16`) — refresh through #90; docs only
- Prefetch H2D (#92, `opt/prefetch-h2d`) — pinned one-batch CUDA side-stream lookahead, `BDH_PREFETCH_H2D=1` default; CPU no-op; GPU H2D overlap/throughput remains unmeasured
- Blocked tile v2 (#93, `opt/blocked-tile-v2`) — flatten long CPU cold `(B,H)` heads into dense `bmm` tiles; CPU blocked wins from T≥256, while CUDA/default eager behavior is unchanged
- Docs matrix v17 (#94, `opt/docs-matrix-v17`) — refresh `OPT_STATUS.md` / `OPT_BACKLOG.md` through #93; docs only
- Copy-tax v1 (#95, `opt/copy-tax-v1`) — contiguous QR avoids RoPE clones from the encoder permute view, fresh score buffers use in-place `tril_`, and CPU profile call counts improve (`copy_` 24→18; QR contig copies 8→0); correctness/parity pass, defaults unchanged, GPU impact unmeasured
- Docs matrix v6 refresh (`opt/docs-matrix-v6` #65) — docs-only through #64; documented tip `4558501`

## Ranked next work

| P | Item | Why (from profile / notes) | Target | Risk |
|---|------|----------------------------|--------|------|
| **P0** | **Measure Triton/CUDA fused tril-score×V on real GPU** | Default **eager** still (**GPU blocker**; profile-v9, #100, #102, #104, #105, #110, #119, #120, #121, and #123 are CPU/docs evidence only): attention `bmm` 30.50% / `mul` 30.78% / `copy_` 19.45%; forward `bmm` 34.12% / `mm` 24.93% / `mul` 17.69% / `copy_` 9.09%. #96–#123 add no GPU timing or speedup claim; #95 isolated forward `copy_`=24→18, while profile-v9 warmed harness is 16/call and generate is 558/call before #102. #100 lowers blocked/online inference epilogue buffering and tightens T=1 decode tiling; #102/#104 cut and re-measure generate `copy_` ~558→~398 with `cat` at 0; #105 confirms the residual-LN path without GPU evidence; #108 keeps wide-head cold Triton tiles bounded and removes shared-V CPU staging without GPU evidence; #110 cuts generate `copy_` ~398→~394 but leaves V-slot (~132/gen) and multinomial (~128/gen) copies at the probe ceiling; #114 adds only bounded CUDA cold-tile/build-smoke coverage; #116 adds only CPU shared-V T=1 epilogue/parity coverage; #117 adds only CPU compile-probe coverage; #119 adds only CPU AMP gating/bench coverage; #120 adds only CPU AUTO parity/harness coverage; #121 adds only CPU profile evidence; #123 adds only gated CPU sparse density evidence. GPU validation remains open. #93, #108, #110, #112, #116, #119, #121, and #123 remain CPU-only evidence; #111, #113, #115, #118, and #122 are docs-only. | A100/H100: `bench_gpu_attn.py` (+ fused score×V) | Env blocker
| **P0** | **Cold CUDA/Triton tile/staging validation** | **Landed `opt/triton-cold` + `opt/triton-cold-v2` + #93 CPU tile flattening + #108 `opt/triton-cold-v3` + #114 `opt/cuda-cold-v3` + #120 strict AUTO gates:** adaptive power-of-2 tiles (grow @T≥256 pair #75/#79), wide-head 64×64 bounds, fused strict-tril score×V, shared-V view fallback without `(B*H,T,D)` staging, CPU→blocked adaptive, and #114-aligned CUDA cold bounds; #93 CPU blocked bench is 1.25× @256 / 4.00× @512 / 5.87× @1024, but GPU measurement and cold CUDA/Triton validation remain open. | A100/H100 microbench; bit-identical | Env blocker |
| **P1** | **`torch.compile` GPU train-step next** | **Landed #117**: `BDH_COMPILE_PROBE=train_bwd` now probes the compiled forward/backward region; optimizer step and grad clearing remain eager. CPU smoke is honest but slower with the stronger probe (19.08 ms `COMPILE=0` vs 31.23 ms `COMPILE=1` on the cited tiny config). CPU guidance remains `COMPILE=1` **only with eager** + `MODE=default`; blocked/`reduce-overhead` warn. **GPU inductor / CUDA graphs still unmeasured.** | A100/H100: `BDH_COMPILE=0` vs `1` + `MODE=default` vs `reduce-overhead` + `FULLGRAPH` via `benchmarks/bench_train_step.py` | Low
| **P1** | **Decode GEMM / copy tax on generate** | **Host tax cut** #44+#48; **decode-mm** + **decode-online-v2** + **attn-auto** + **triton-decode-v3** + **cuda-decode-v3** + **`opt/prefill-blocked`** + **`opt/auto-tune`**: AUTO long-T cold+decode (adaptive blocked cold @T≥256), with optional independent cold threshold. CPU e2e AUTO **1.26×@1024 / 1.39×@2048**; IMPL=blocked ~1.23–1.43×; short A/B ~1.25× @1024. Remaining = **GPU** measure / re-tune thr. Default still eager. | A100/H100: `bench_generate.py --mode auto-ab` + `--mode impls` + `bench_gpu_attn.py --mode decode`; keep cat-free | Medium |
| **P1** | **CUDA prefetch H2D overlap** | **Landed #92** `BDH_PREFETCH_H2D=1`: one pinned batch staged on a side stream with event handoff; CPU is a no-op and the CPU bench has no throughput evidence. | CUDA host: measure H2D overlap, lifetime safety, and end-to-end train throughput; keep `BDH_PREFETCH_H2D=1` opt-in to measurement only | Env blocker |
| **P2** | **Fused RoPE kernel** | Attn: `mul`/`copy_` from strided rotate. **Landed #90** `opt/rope-fuse-v2` deepens `opt/rope-fuse`: no expand/stack, table-pair T=1, blocked Triton fallback; default remains eager. GPU validation is still open. | GPU Triton microbench still open | Low–medium |
| **P2** | **Sparsity follow-through** | **Gated density re-smoke landed** `opt/sparse-v2`: `BDH_SPARSE_PROBE` is explicit opt-in, short-train x=26.63% / xy=11.37% at step 150, and the conservative x≥20% / xy≥8% guardrail passes. CPU sparse **never reliably beat dense** → dense/default path and sparse **stay OFF**; GPU sparse remains unmeasured. | GPU sparse bench only if density ≪10% and a kernel win is demonstrated | Speculative |
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
| Re-profile post-#75–#77 | **Landed** `opt/profile-v7` (#80) — tip `b8067f5`; profile source `ca5038f`; cats=0; contiguous=0; #69–#79 off short default window |
| Re-profile post-#85–#87 | **Landed** #89 `opt/profile-v8` — tip `f10bdd4`; profile source `f10bdd4`; cats=0; contiguous=0; #85–#87 off or outside the short default window; no GPU measurements |
| Docs matrix through #90 | **Landed** #91 `opt/docs-matrix-v16` — docs-only; documented code tip `24f44a6` |
| CUDA H2D staging | **Landed** #92 `opt/prefetch-h2d` — `BDH_PREFETCH_H2D=1` default on CUDA; CPU no-op; GPU overlap/throughput remains unmeasured |
| Long CPU blocked tile flattening | **Landed** #93 `opt/blocked-tile-v2` — dense `(B,H)` `bmm` staging for T≥256; CPU-only 1.25×/4.00×/5.87× at T=256/512/1024; default eager and CUDA path unchanged |
| Docs matrix v17 | **Landed** #94 `opt/docs-matrix-v17` — docs-only refresh through #93 |
| Copy-tax / RoPE layout | **Landed** #95 `opt/copy-tax-v1` — CPU forward `aten::copy_` 24→18 and QR contiguous copies 8→0; correctness/parity pass; no GPU claim; defaults unchanged |
| Attn-bwd GPU scaffold | **Landed** #96 `opt/attn-bwd-gpu-scaffold` — CUDA analytic-attention train harness with parity/config controls and clean CPU skip; no GPU timing
| Docs matrix v18 | **Landed** #97 — refresh through #95/#96; docs only
| Re-profile post-#95 | **Landed** #98 `opt/profile-v9` — source `8e7a4d2`, current docs tip `90b609f`; isolated forward `copy_`=18, warmed harness 16/call, generate 558/call, `cat`/`contiguous`=0; no GPU measurement
| Docs matrix v19 | **Landed** #99 — refresh through #98; docs only
| Score×V epilogue deepen | **Landed** #100 `opt/scorev-fuse-v2` — direct no-grad CPU `baddbmm(..., out=target)` epilogues, flattened T=1 output reuse, 1024-element decode oneshot budget; default eager unchanged; no GPU measurement
| Docs matrix v20 | **Landed** #101 — refresh through #100; docs only
| Generate copy-tax v1 | **Landed** #102 `opt/gen-copy-tax-v1` — complex-view RoPE pair copy plus `idx_out`/`out.narrow` sampling; CPU generate `copy_` ~558→~398, `cat`=0; defaults unchanged; no GPU measurement
| Docs matrix v21 | **Landed** #103 `opt/docs-matrix-v21` — refresh through #102; docs only
| Re-profile post-#102 | **Landed** #104 `opt/profile-v10` — source `1363794`, current docs tip `6f91417`; generate `copy_`=398/call (1,194/3), isolated forward `copy_`=18, `cat`/`contiguous`=0; no GPU measurement
| Residual LN audit | **Landed** #105 `opt/ln-resid-v2` — current eager residual path is already the safe deepen; CPU probes show 2× `native_layer_norm` + 1× `add_`, no residual copy/cat/dtype bounce; no further default-safe change; no GPU measurement
| Docs matrix v22 through #104 | **Landed** #106 `opt/docs-matrix-v22` — docs-only refresh through #104
| Docs matrix v22 through #105 | **Landed** #107 `opt/docs-matrix-v22` — docs-only residual-LN audit update
| Triton cold v3 | **Landed** #108 `opt/triton-cold-v3` — bounded wide-head cold tiles and copy-free shared-V CPU blocked fallback; parity pass on CPU, no GPU measurement
| Docs matrix v23 | **Landed** #109 `opt/docs-matrix-v23` — docs-only refresh through #108
| Generate V/sampler probe | **Landed** #110 `opt/gen-vcopy-v1` — T>1 RoPE `_store_pairs` plus top-k `gather(out=)` cut CPU generate `copy_` ~398→~394; V-slot and multinomial copies remain the safe-removal ceiling; no GPU measurement
| Docs matrix v24 | **Landed** #111 `opt/docs-matrix-v24` — docs-only refresh through #110
| Re-profile post-#110 | **Landed** #112 `opt/profile-v11` — source `4963b0f`, warmed forward `copy_`=12/call, isolated forward `copy_`=14, generate `copy_`=394/call (1,182/3; isolated 396), `cat`/`contiguous`=0; no GPU measurement
| Docs matrix v25 | **Landed** #113 `opt/docs-matrix-v25` — refresh through #112; docs only
| CUDA cold v3 | **Landed** #114 `opt/cuda-cold-v3` — #108-aligned wide-head CUDA/CPU tile bounds, CPU parity, clean CUDA skip, and C++20 build smoke; no GPU measurement
| Docs matrix v26 | **Landed** #115 `opt/docs-matrix-v26` — refresh through #114; docs only
| Online decode T=1 | **Landed** #116 `opt/online-decode-t1-v2` — direct inference/no-grad shared-V score×V epilogue for blocked/online oneshot and long-S decode, graph-safe autograd fallback, parity and cat-free generate; no GPU measurement
| Compile train v2 | **Landed** #117 `opt/compile-train-v2` — backward-aware `train_bwd` compile probe default, missing-target eager fallback, explicit optimizer boundary, and CPU-only bench matrix; no GPU measurement
| Docs matrix v27 | **Landed** #118 `opt/docs-matrix-v27` — refresh through #117; docs only
| AMP train v2 | **Landed** #119 `opt/amp-train-v2` — transactional CPU capability checks, full/forward-only matrix, enabled CUDA float16 GradScaler gate; defaults unchanged; no GPU measurement
| AUTO threshold v2 | **Landed** #120 `opt/auto-thr-v2` — strict independent cold/decode gates, independent-threshold generate A/B harness, parity/default-eager coverage; no GPU measurement
| Profile v12 | **Landed** #121 `opt/profile-v12` — CPU re-profile of the #116–#120 tip; warmed forward `copy_`=12/call, isolated=14, generate=394/call; `cat`/`contiguous`=0; no GPU measurement
| Docs matrix v28 | **Carried forward** #122 `docs/opt-status-backlog-v28` — prior closed docs-only attempt; superseded by this v29 refresh
| Sparse v2 | **Landed** #123 `opt/sparse-v2` — explicit `BDH_SPARSE_PROBE` gate, x~27% / xy~11% density guardrails, dense-default tests; no GPU measurement; sparse remains OFF
| Analytic tril attn train path | **Landed** #39 `opt/attn-bwd-train` — default AUTOGRAD off; eager profile unchanged |
| Blocked/online tiled analytic bwd | **Landed** #41 `opt/blocked-autograd` — blocked|online+AUTOGRAD=1; dense M-recompute only for eager |
| Batch prefetch overlap | **Landed** #42 `opt/prefetch-v2` — host queue/numpy producer; GPU pin/H2D overlap remains open |
| OPT status docs refresh | **Landed** #43 `opt/docs-matrix-v2` — matrix through #40 |
| Cache packing / fewer cats | **Done** #19–#20 — generate `aten::cat` **0** (was ~10% self / ~864 calls pre-pack) |
| Fuse score×V epilogue (no materialize T×T) | **Landed** #21; **CPU vectorized** `opt/blocked-vec` (~18–36× vs old blocked wall; still slower than eager) |
| `torch.compile` / inductor CPU harden | **Landed** #17+#22+#31+#46+#49+#63+#70+#84; COMPILE+eager only win on CPU; warn on COMPILE+blocked; **CPU `reduce-overhead` not useful** (no CUDA graphs); dropout=0 / eval identity hardened (#70); **FULLGRAPH=1** probed (#84; 0 breaks cold eager×AUTOGRAD; soft-fallback); remaining = **GPU** measure (P1) |
| ~~Zero-grad / `set_to_none` train-path hardening~~ | **Landed** #87 `opt/zerograd` — `clear_grads()` chokepoint plus compile `train_bwd` probe; CPU re-smoke ~1.03× fused+set-to-none vs legacy; defaults unchanged |
| Fused RoPE rotate (`BDH_ROPE_IMPL`) | **Landed** `opt/rope-fuse` + **#90 `opt/rope-fuse-v2`** — default eager; no-expand/stack fuse; table-pair T=1; blocked Triton CPU scaffold; GPU validation open |
| T=1 RoPE apply deepen | **Landed** `opt/rope-decode` + **#90 `opt/rope-fuse-v2`** pair-apply wire — `rope_rotate_paired` from `_rope_table_pairs`; CPU parity / no GPU claim |
| Decode GEMM vs packed KR/V | **Landed** `opt/decode-gemm` — blocked/triton/cuda decode polish; GPU measure still open |
| Decode-mm T=1 / lm_head mv | **Landed** `opt/decode-mm` — `_two_gemm_decode`, CUDA Tq=1 + `DECODE_TILE_N`, B=1 `mv`; CPU wall ~noise; GPU open |
| Decode-online-v2 blocked T=1 | **Landed** `opt/decode-online-v2` — tight broadcast oneshot; long-S peak↓ + wall↑ on CPU; GPU open |
| Attn-auto long-S decode | **Landed** `opt/attn-auto` + `opt/prefill-blocked` + `opt/auto-tune` — `BDH_ATTN_AUTO` shared thr=512, adaptive cold, optional `BDH_ATTN_AUTO_COLD_THRESHOLD`; default eager; GPU thr re-tune open |
| Long-S generate AUTO A/B | **Landed** `opt/gen-long-bench` — harness; deepened by `opt/prefill-blocked` / `opt/auto-tune` e2e AUTO 1.26–1.39×; GPU open |
| Prefill-blocked cold AUTO | **Landed** `opt/prefill-blocked` + `opt/auto-tune` — adaptive cold BS@T≥256; AUTO cold+decode; shared thr=512 plus optional cold gate; e2e AUTO 1.26–1.39×; GPU open |
| Triton decode-v3 | **Landed** `opt/triton-decode-v3` — long-S tiles/Q-hoist scaffold; AUTO prefers triton when avail; CPU→#55 blocked; GPU measure open |
| CUDA decode-v3 | **Landed** `opt/cuda-decode-v3` — adaptive DECODE_TILE_N + TQ1 Q-hoist; ≡ blocked parity; GPU measure open |
| CUDA cold-v2 | **Landed** `opt/cuda-cold-v2` — adaptive cold TILE_M/N (pair #75); ≡ blocked/eager; GPU measure open |
| Triton cold-v2 | **Landed** `opt/triton-cold-v2` — adaptive cold BLOCK_M/N (pair #75/#79); CPU→blocked adaptive; GPU measure open |
| Memory layout / embed path | **Landed** #12–#13+#16 |
| Generate Python/host tax | **Landed** #44 `opt/gen-host`; follow-up `opt/gen-sample` fuses T=1 lm_head+sample / top-k (large-V top_k ~1.5×; default V wall ~noise). GPU decode GEMM remains P1 |

| **P2** | **Analytic attn train on GPU** | CPU blocked|online + tiled analytic bwd landed (`opt/blocked-autograd` #41), and #96 adds the CUDA parity/config harness with a clean CPU skip. **GPU** train-step with `IMPL=blocked|triton|cuda` + AUTOGRAD=1 remains unmeasured. | A100/H100 `bench_attn_bwd.py` / #96 scaffold | Low |

## Explicit non-goals

- Softmax / diagonal inclusion
- PRs to `pathwaycom/*`
- Claiming GPU speedups from CPU profiler absolute times
- Defaulting `BDH_ATTN_IMPL=blocked` on CPU (still slower than eager after `opt/blocked-vec`; use for peak-score memory / parity)
- Recommending `BDH_COMPILE=1` with `blocked`/`online`/`triton` on CPU (#46 regression; #49 warns)
- Recommending `BDH_COMPILE_MODE=reduce-overhead` on CPU (no CUDA graphs; #63 documents)
- Re-introducing `aten::cat` in `generate` / packed cache path
- Treating CPU sparse density/crossover as a GPU result; `BDH_SPARSE_PROBE` remains explicit opt-in and sparse stays OFF

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
(`BDH_COMPILE=0` vs `1`, with the stronger `BDH_COMPILE_PROBE=train_bwd` default, plus `MODE=default` vs `reduce-overhead` via
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
`BDH_ATTN_AUTO=0|1` (thr=512). Cold+decode switch when
`past_len > thr`. Validates `#55`/`#56`/`#75`/`#77` outside score×V microbench.

```bash
python benchmarks/bench_generate.py
python benchmarks/bench_generate.py --prompt 32 --new 64 --warmup 2 --iters 5
python benchmarks/bench_generate.py --mode auto-ab --prompts 256,1024,2048 --new 8
python benchmarks/bench_generate.py --mode auto-ab --prompts 256,1024,2048 --new 64
# A100/H100:
python benchmarks/bench_generate.py --device cuda --warmup 5 --iters 20
python benchmarks/bench_generate.py --mode auto-ab --device cuda
```

**CPU honesty (this box / profile source `f10bdd4`; documented code tip `bcf2458`):**
- `--mode impls` short prompt: medians ~noise vs eager; match; `aten::cat=0`
- `--mode auto-ab`: AUTO fires @ S>512; tokens match; cats=0; **e2e AUTO**
  **1.26× @1024 / 1.39× @2048** after `opt/prefill-blocked` (cold+decode);
  IMPL=blocked ~1.23–1.43×
- **Do not** claim Triton/CUDA / AUTO e2e wins from CPU. GPU thr re-tune = P1.
Private repo only — never `pathwaycom/*`.
