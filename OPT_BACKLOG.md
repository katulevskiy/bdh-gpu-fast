# OPT backlog — ranked remaining work

Private sandbox only (`katulevskiy/bdh-gpu-opt`). **Do not** PR to `pathwaycom/bdh`.

Constraint (hard): attention stays **raw scores** × **strict lower-triangular**
`tril(diagonal=-1)` — **no softmax**, **no `1/√d`**, **no**
`F.scaled_dot_product_attention`.

Profile source: `benchmarks/profile_forward.py` on CPU (`torch 2.14.0+cu130`, `cuda=False`), profile-v15 source tip `86315f0` / profile-v16 source tip `717c38e` / profile-v17 source tip `f34d908` / current code tip `da79013` after #159 zerograd-v2, #160 cache-bench-v2, #161 profile-v16, #163 structured GPU-measurement harness, #164 packed T=1 decode-view deepen, #165 shared-V decode epilogue deepen, #166 compile-train-v4 fallback guidance, #167 docs refresh, #168 fused-RoPE scaffold v3, #169 prefetch H2D v3, #170 online-decode-v4, #171/#172 docs refreshes, #173 attn-bwd-v3, #174 cuda-cold-v5, #175 docs refresh through #174, #176 blocked-tile-v4, #177 docs refresh through #176, #178 triton-cold-v5, #179 amp-train-v4, #180 docs refresh through #179, #181 cache-bench-v3, #182 profile-v17, #183 docs refresh through #182, #184 sparse-probe exit guardrails, #185 auto-thr-v4, #186 gen-bench-v3, #187 docs refresh through #186, #188 layout-v4, and #189 online-decode-v5. Profile-v17 remains flat versus v16/v15: attention `copy_`=2/call, forward `copy_`=12/call, generate `copy_`=394/call, with `cat=0` and `contiguous=0`. #163 adds structured cold/decode/dtype measurement and CPU-safe skip coverage but no CUDA run or GPU result. #164 preserves capacity-strided packed KR/V decode views without an unconditional staging copy; #165 accumulates B>1 shared-V decode tiles in place; #166 adds CPU-only compile-probe fallback diagnostics; #167, #171, #172, #180, and #187 are docs-only; #168 adds CPU-only Triton skip-reason and strided T=1 RoPE output parity coverage; #169 expands CPU no-op coverage and documents CUDA staged-buffer lifetime without measuring H2D overlap; #170 preserves packed key/value views in opt-in online decode with CPU parity only; #173 deepens the CPU analytic-attention parity matrix; #174 adds explicit CPU-only extension setup smoke; #175 and #177 are docs-only; #176 adds only CPU wide-head partial-tile parity; #178 adds only CPU-safe skip-marker and long-T fallback parity; #179 adds only CPU AMP dtype/scaler contract coverage; #181 adds only CPU packed-footprint accounting; #182 adds only matched CPU profile-v17 evidence; #184 adds CPU-only sparse-probe guardrails; #185 adds CPU-only AUTO threshold-gate smoke; #186 hardens CPU generate validation; #188 adds CPU-only sampler-layout probe coverage; and #189 adds CPU-only packed shared-V decode parity coverage. None adds a CUDA run or GPU result. Absolute ms are **profiler-inflated**; use **% self CPU** and call counts. Re-run on GPU before claiming kernel wins.

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

Post-#124 rope GPU scaffold (`opt/rope-gpu-scaffold`): route paired T=1 RoPE through `BDH_ROPE_IMPL=fused`, reuse the Triton kernel with a zero cis-row stride for the broadcast table, and retain the eager default, CPU paired parity, and CUDA+Triton skip gates. This is GPU-readiness scaffolding only: no GPU timing, correctness, or speedup evidence was collected, so fused-RoPE measurement remains P0.

Post-#125 docs refresh (`opt/docs-matrix-v29`): docs-only refresh through #123 on the #124 code tip `f16115c`; this v30 update records #124 and keeps the private-repo pointer current.

Post-#126 docs refresh (`opt/docs-matrix-v30`): docs-only refresh carrying the #124/#125 documentation tip forward; no code or GPU evidence was added.

Post-#127 profile CI (`opt/profile-ci`, source `0d717c9`): add a CPU-only profiler smoke and optional manual/nightly Chrome trace upload scaffold while keeping `benchmarks/traces/` gitignored and avoiding PR-triggered or GPU performance claims. This is CI/profile plumbing only; no GPU measurement was added.

Post-#128 decode GEMM v2 (`opt/decode-gemm-v2`, source `45b4afe`): preserve capacity-padded packed CacheManager Q/K/V strides in the T=1 Triton decode launcher instead of staging contiguous copies each step. Eager remains the default, raw score×V and strict past-only decode semantics remain unchanged, and parity/packed-view/stride-launcher/cat-free coverage is CPU-only here; GPU decode measurement and cold CUDA/Triton validation remain P0 blockers.

Post-#129 profile-v13 (`opt/profile-v13`, source `45b4afe`): clean CPU re-profile after #128 records attention `copy_`=2/call, forward `copy_`=12/call, and generate `copy_`=394/call, with `cat=0` and `contiguous=0` across the profiled modes. The 130-pass/7-skip targeted smoke is CPU-only evidence; no GPU timing, kernel win, correctness, or speedup claim was added.

Post-#130 attn-bwd v2 (`opt/attn-bwd-v2`, source `5200b4f`): keep `IMPL=online` beside `blocked` in the analytic-attention GPU train matrix crossed with `AUTOGRAD=0|1`, with a clean CPU skip/parity contract. CPU validation remains smoke/parity only (525 passed, 19 skipped); the expanded CUDA matrix is still unrun, so GPU analytic-attention training remains open.

Post-#131 AUTO threshold sweep (`opt/auto-threshold-sweep`, source `891b7c5`): add `--auto-threshold-sweep` to repeat the CPU long-S AUTO A/B harness over stable, de-duplicated decode thresholds while preserving seeded token parity, strict cold/decode resolver checks, `aten::cat=0`, and default eager behavior. CPU tests/smoke only; no GPU timing or performance claim.

Post-#132 docs refresh (`ce75f3e`): refresh `OPT_STATUS.md` / `OPT_BACKLOG.md` through #130. Docs-only; no code or GPU evidence.

Post-#133 docs refresh (`70adef8`): refresh `OPT_STATUS.md` / `OPT_BACKLOG.md` through #131. Docs-only; no code or GPU evidence.

Post-#134 CUDA build v2 (`opt/cuda-build-v2`, source `dba5f75`): check for `nvcc` before constructing the optional `CUDAExtension`, so forced CUDA setup skips explicitly on runtime-only CPU boxes instead of failing with an opaque toolchain error. Default/no-`nvcc` subprocess smoke is covered; CPU refs remain available, defaults and attention semantics are unchanged, and CUDA-only tests skip cleanly. No GPU compilation, timing, or kernel claim was added.

Post-#135 score×V v3 (`opt/scorev-fuse-v3`, source `cc62e0b`): deepen the common B=1, T=1 shared-V decode epilogue by reusing flattened score/output views and writing directly with `baddbmm(..., out=target)`. B>1 keeps the existing path and autograd keeps the allocation-safe fallback; eager/default resolver behavior, strict raw `tril`, broadcast-V layout, and cat-free generate are unchanged. CPU validation reports 111 focused passes / 3 skips and 533 full-suite passes / 19 skips / 3 warnings; no GPU timing or speedup claim was added, so GPU measurement remains the P0 blocker.

Post-#136 compile train v3 (`opt/compile-train-v3`, source `546192d`): clarify that `maybe_compile()` labels construction, missing-target, and first-probe soft fallbacks with requested device/mode/probe/FULLGRAPH settings and the original eager module. Document the FULLGRAPH/AUTOGRAD/probe boundaries while keeping `BDH_COMPILE=0`, `BDH_COMPILE_PROBE=train_bwd`, and `BDH_COMPILE_FULLGRAPH=0` unchanged. CPU validation reports 36 passed / 1 skipped / 1 warning for `test_compile.py`, a baseline smoke with compile/matrix/AMP sections skipped, and 533 passed / 19 skipped / 4 warnings for the full suite; no GPU or CUDA-graph measurement was added.

Post-#137 docs refresh (`6a4c856`): refresh `OPT_STATUS.md` / `OPT_BACKLOG.md` through #136 on the clean #136 code tip. Docs-only; no code or GPU evidence.

Post-#138 rope GPU v2 (`opt/rope-gpu-v2`, source `da664ec`): harden the Triton RoPE gate so `v`, cis, and optional `out` must share a CUDA device and use fp16/bf16/fp32; unsupported or mixed inputs skip to the existing PyTorch/blocked fallback, and `backend_info(cuda)` skips cleanly without initializing unavailable CUDA. CPU T>1 transposed-leading-dim and out-buffer parity coverage reports 33 passed / 2 skipped; the eager default and RoPE/attention math are unchanged. No GPU correctness, timing, or speedup evidence was added; fused-RoPE validation remains a P0 blocker.

Post-#139 prefetch H2D v2 (`opt/prefetch-h2d-v2`, source `6fd7950`): make the CPU no-op contract explicit while keeping `BDH_PREFETCH_H2D=1` and `BDH_PREFETCH_ASYNC=1` unchanged. The device check happens before CUDA stream/event construction; CPU creates no device lookahead and `_to_device` returns the original CPU tensors. CPU validation reports 17 passed / 1 skipped for `test_dataloader.py` and 539 passed / 19 skipped / 3 warnings for the full suite. No GPU H2D overlap, throughput, or correctness claim was added; real GPU measurement remains the P0 blocker.

Post-#140 docs refresh (`40eac80`): refresh `OPT_STATUS.md` / `OPT_BACKLOG.md` through #139 on the #141 code tip. Docs-only; no code or GPU evidence.

Post-#141 blocked-tile-v3 (`ddf58df`): add wide-head CPU cold parity coverage for shared and head-matched V layouts and document the bounded shared-V tile behavior. Eager defaults remain unchanged, with no GPU timing or kernel claim; real GPU measurement and cold CUDA/Triton validation remain P0 blockers.

Post-#142 amp-train-v3 (`68f949a`): exercise explicit float32, bf16, and fp16 AMP configuration contracts with CPU-safe backend skips while preserving fp32 defaults and CUDA-only GradScaler gating. No GPU AMP throughput evidence was added; real GPU measurement remains the P0 blocker.

Post-#143 docs refresh: refresh `OPT_STATUS.md` / `OPT_BACKLOG.md` through #142. Docs-only; no code or GPU evidence.

Post-#144 profile-v14: re-profile the #141/#142 CPU tip. The short window is flat versus profile-v13: attention `bmm` 26.11% / `mul` 23.93% / `complex` 15.84% / `copy_` 12.50%; forward `bmm` 28.32% / `mm` 23.89% / `mul` 15.36% / `complex` 11.70% / `copy_` 6.83%; generate `mm` 20.32% / `bmm` 13.16% / `mul` 3.05% / `matmul` 2.59%. Counts remain attention `copy_`=2/call, forward `copy_`=12/call, and generate `copy_`=394/call; `aten::cat=0` and `aten::contiguous=0`. CPU-only evidence; no GPU timing or speedup claim.

Post-#145 sparse-v3: make sparse-probe success and guardrail-failure outcomes scriptable with stable exit codes, coverage, and notes. The probe remains CPU-only and default-off; sparse stays OFF because CPU sparse did not beat dense, and no GPU sparse-kernel claim was added.

Post-#146 cuda-cold-v4: clarify CPU-safe CUDA skip behavior in the cold-attention tests before any CUDA runtime/build is attempted. This is configuration/skip coverage only; no GPU build, timing, or kernel claim was added, so cold CUDA validation and GPU measurement remain P0 blockers.

Post-#147 online-decode-v3: deepen the opt-in blocked/online T=1 shared-V path by reusing one `(B,S,D)` CacheManager view across past tiles and vectorizing the B>1 shared-V epilogue while preserving the B=1 direct `baddbmm(..., out=target)` path, strict raw `tril(-1)` decode math, autograd fallback, cat-free generate, and default eager dispatch. Focused CPU parity/coverage passes; CUDA/Triton hardware remains skip-gated, so no GPU timing or speedup claim was added and GPU measurement remains the P0 blocker.

Post-#148 docs refresh (`opt/docs-matrix-v35`, source `c557b55`): refresh `OPT_STATUS.md` / `OPT_BACKLOG.md` through #147. Docs-only; no code or GPU evidence.

Post-#149 AUTO threshold sweep smoke (`opt/auto-thr-v3`, source `8f6ec4a`): add CPU-safe smoke coverage for the #131 `--auto-threshold-sweep` dispatcher, verifying stable threshold de-duplication, recursive-dispatch suppression, malformed-input rejection, and preservation of an independent cold threshold. Eager and AUTO-off defaults remain unchanged; this is CPU smoke only with no GPU timing or performance claim, so GPU measurement and threshold/tile validation remain P0 blockers.

Post-#150 Triton cold-v4 (`opt/triton-cold-v4`, source `748a138`): add `triton_cold_skip_reason()` so cold Triton skip markers report the import/device gate without allocating CUDA tensors or launching kernels during pytest collection; this CPU box reports `CUDA unavailable: torch.cuda.is_available() is false`. Targeted tests report 127 passed / 6 skipped and the full suite 548 passed / 19 skipped / 3 warnings. No GPU timing, compilation, kernel validation, correctness, speedup, or other GPU claim was added; cold CUDA/Triton validation and real GPU measurement remain P0 blockers.

Post-#151 docs refresh (source `ec851b3`): refresh `OPT_STATUS.md` / `OPT_BACKLOG.md` through #149 on the #150 code tip. Docs-only; no code or GPU evidence.

Post-#152 generate copy-v2 (`opt/gen-copy-v2`, source `d60380f`): re-audit the default-eager post-#110 generate path and confirm the CPU-safe copy ceiling: packed V snapshots remain required for cache ownership, RoPE pair stores are at the safe eager floor, prompt seed copy preserves owned cat-free output, and ATen `torch.multinomial(out=...)` retains internal copy work. The focused probe reports 91 passed / 2 skipped; the full suite reports 554 passed / 19 skipped / 3 warnings. No cache-layout, RNG-stream, custom-sampler, GPU timing, or speedup claim was added; real GPU measurement and copy-tax impact remain P0 blockers.

Post-#153 docs refresh: refresh `OPT_STATUS.md` / `OPT_BACKLOG.md` through #152 on the #154 code tip. Docs-only; no code or GPU evidence.

Post-#154 rope-fuse-v3 (`opt/rope-fuse-v3`, source `6d2dff9`): reuse the last table-backed T>1 `(cos, sin)` narrow for callers that do not hoist `cos_sin`, invalidate the view cache when the warmed generate table is rebuilt, and retain the eager default with CPU cache-hit/rebuild parity coverage. No GPU timing or speedup claim was added; fused-RoPE validation and real GPU measurement remain P0 blockers.
Post-#155 docs refresh (`opt/docs-matrix-v38`): refresh `OPT_STATUS.md` / `OPT_BACKLOG.md` through #154 on the #156 code tip. Docs-only; no code or GPU evidence.

Post-#156 layout-v3 (`opt/layout-v3`, source `e1248ce`): add a CPU `torch.profiler` probe for the remaining layout materializations after layout-v2 and #154. The eval encoder cache intentionally pays one contiguous `(nh*N,D)` materialization outside the forward hot path; warm default-eager forward remains `aten::contiguous=0` and `aten::cat=0` while reporting two clone-backed signatures per layer, `(B,nh,T,D)` and `(nh,B,T,D,1)`. Packed generate remains cat-free but retains ATen sampler contiguous hotspots, one `(B,V)` softmax buffer materialization and one `(B,)` index result per sampled token. No safe default view reuse, layout flip, sampler/RNG rewrite, GPU timing, or speedup claim was added; **P0 GPU measurement remains the blocker.**

Post-#157 profile-v15 (`opt/profile-v15`, source `86315f0`): re-profile the post-#156 CPU tip with one warmup wait and three active steps. Attention self-CPU top operators are `bmm` 31.99%, `mul` 22.22%, `complex` 19.12%, `copy_` 8.87%; forward `bmm` 29.05%, `mm` 23.15%, `mul` 15.99%, `complex` 11.42%, `copy_` 7.05%; generate `mm` 21.81%, `bmm` 14.15%, `mul` 3.05%, `matmul` 2.40%, `copy_` 0.86%. Counts remain attention `copy_`=2/call, forward=12/call, generate=394/call, with `cat=0` and `contiguous=0`. CPU-only evidence; no GPU timing or speedup claim, and **P0 GPU measurement remains the blocker.**
Post-#158 docs refresh (`opt/docs-matrix-v39`, merge `64086c8`): refresh `OPT_STATUS.md` / `OPT_BACKLOG.md` through #157 on the #160 code tip. Docs-only; no code or GPU evidence.

Post-#159 zerograd-v2 (`opt/zerograd-v2`, merge `717c38e`): make `TrainLossLogger` CUDA-deferred mode explicit, with `async_cuda=True` a CPU no-op, a resolved-state property, and synchronous CPU coverage without changing logging or `set_to_none` defaults. CPU validation only; no GPU timing or speedup claim.

Post-#160 cache-bench-v2 (`opt/cache-bench-v2`, merge `055c524`): deepen the CPU-only page-growth bench to report each page policy's `initial→final` packed-cache capacity and final KR/V allocation in KiB, and assert geometric/linear A/B policies reach the same final capacity and allocation. Grow/copy accounting remains the comparison; cache/generate defaults and page behavior are unchanged. This is CPU memory accounting only, not a GPU memory or wall-time claim; **P0 GPU measurement remains the blocker.**

Post-#161 profile-v16 (`opt/profile-v16`, merge `8bd6b17`): re-profile the #159/#160 tip with two warmups, one wait, and three active CPU steps. Attention self-CPU top operators are `bmm` 29.74%, `mul` 22.64%, `complex` 16.18%, `copy_` 11.32%; forward `bmm` 28.15%, `mm` 23.15%, `mul` 15.20%, `complex` 12.13%, `copy_` 7.77%; generate `mm` 21.68%, `bmm` 14.24%, `mul` 3.10%, `matmul` 2.37%, `copy_` 0.87%. Counts remain attention `copy_`=2/call, forward=12/call, generate=394/call, with `cat=0` and `contiguous=0`; the modest mix movement is flat trace noise, not a speedup claim.

Post-#163 structured GPU measurement (`opt/gpu-measure-v2`, merge `d0e667b`): deepen `bench_gpu_attn.py` for structured cold/decode/dtype output, parity deltas, median fields, and CPU-safe skip diagnostics, with focused CPU tests. No CUDA box was available and no GPU timing or win is claimed; **P0 remains real GPU measurement, including cold CUDA/Triton validation.**

Post-#164 packed T=1 decode views (`opt/decode-gemm-v3`, merge `7719e9a`): preserve capacity-strided packed CacheManager Q/K/V views in the T=1 Triton decode path and avoid an unconditional per-step staging copy while retaining eager defaults, raw strict `tril(-1)` score×V semantics, and past-only decode behavior. CPU parity/stride coverage only; no CUDA timing or speedup claim was added, so **P0 remains real GPU measurement, including cold CUDA/Triton validation.**

Post-#165 shared-V decode epilogue (`opt/scorev-fuse-v4`, merge `aa73a43`): accumulate B>1 shared-V decode tiles in place, removing the accumulated score×V staging product while preserving CPU-safe strict-tril decode parity, the autograd fallback, and eager defaults. No CUDA timing or speedup claim was added; **P0 remains real GPU measurement, including cold CUDA/Triton validation.**

Post-#166 compile-train-v4 (`opt/compile-train-v4`, merge `79eef11`): distinguish unprobed compilation, missing `example_y` for the `train_bwd` probe, and first-probe failures; emit actionable retry guidance while soft-falling back to the original eager module. CPU validation reports 37 passed / 1 skipped for `test_compile.py` and 561 passed / 19 skipped for the full suite. Defaults remain `BDH_COMPILE=0`, `BDH_COMPILE_PROBE=train_bwd`, and `IMPL=eager`; no GPU or CUDA-graph measurement was added.

Post-#167 docs refresh (`opt/docs-v41`, merge `fb698fc`): refresh `OPT_STATUS.md` / `OPT_BACKLOG.md` through #165 on the #168 predecessor tip. Docs-only; no code or GPU evidence.

Post-#168 fused-RoPE scaffold v3 (`opt/rope-gpu-v3`, merge `c3233ed`): expose stable Triton/CUDA skip reasons through `triton_rope_skip_reason()` and `backend_info()`, and preserve CPU parity for T=1 paired rotation into a strided cache-slot-like `out=` buffer. Focused CPU validation reports 41 passed / 2 skipped and the full suite 569 passed / 19 skipped / 3 warnings. Eager defaults and strict `tril(-1)` attention semantics remain unchanged; no fused-GPU timing or speedup claim was added, so **P0 remains real GPU measurement, including cold CUDA/Triton validation.**

Post-#169 prefetch H2D v3 (`opt/prefetch-h2d-v3`, merge `62acaa7`): expand CPU-only coverage across async and sync host modes with unset, disabled, and enabled H2D settings; forbid CPU CUDA stream/event construction, assert no device lookahead, preserve CPU tensor identity, and document the CUDA staged-buffer event/lifetime handoff. CPU validation reports 20 passed / 1 skipped for `test_dataloader.py` and 571 passed / 19 skipped / 3 warnings for the full suite. Defaults remain `BDH_PREFETCH_ASYNC=1` and `BDH_PREFETCH_H2D=1`; no GPU H2D overlap, throughput, correctness, or speedup claim was added, so **P0 remains real GPU measurement, including cold CUDA/Triton validation.**

Post-#170 online-decode-v4 (`opt/online-decode-v4`, merge `1d12071`): keep the opt-in blocked/online T=1 shared-V CPU decode on its original 4-D key view when B*H flattening would require a copy, reuse the shared value view, and preserve direct inference accumulation, the autograd fallback, eager defaults, cat-free generate, and strict causal parity. Focused CPU validation reports 94 passed / 3 skipped; the additional CPU check reports 54 passed / 1 unrelated existing failure. No GPU timing, correctness, or win claim was added; **P0 remains real GPU measurement, including cold CUDA/Triton validation.**

Post-#173 attn-bwd-v3 (`opt/attn-bwd-v3`, merge `f95c143`): deepen the CPU analytic-attention parity matrix across `eager|blocked|online|triton|cuda`, `AUTOGRAD=0|1`, shared-V and full-head-V layouts, a non-tile sequence shape, and randomized output gradients. The CUDA-only harness labels expected native `triton`/`cuda` plus `AUTOGRAD=0` skips explicitly instead of surfacing only the raw autograd exception. No GPU timing or training claim was added; **P0 remains real GPU measurement, including cold CUDA/Triton validation.**

Post-#174 cuda-cold-v5 (`opt/cuda-cold-v5`, merge `bc3967b`): add explicit `BDH_FORCE_CPU_EXT=1` setup smoke so the CPU-only native-extension configuration is selected without entering CUDA setup, while the default remains pure Python and missing-`nvcc` CUDA requests remain a clear no-op. Focused CPU validation reports 21 passed / 5 skipped for `test_cuda_build.py` + `test_cuda_attn.py`; the full suite reports 575 passed / 19 skipped / 3 warnings. No GPU compilation, hardware execution, timing, or speedup claim was added; **P0 remains real GPU measurement, including cold CUDA/Triton validation.**

Post-#175/#176/#177/#178/#179 (`0af0690` / `4e13111` / `5507747` / `f72033e` / `0983326`): the docs refresh carries the matrix through #179 while preserving profile-v16 counts. The blocked cold-path parity matrix adds `B=2, H=3, T=257, N=160, D=192` at the partial 128-row tile boundary for shared-V (`value_heads=1`) and head-matched-V (`value_heads=3`) layouts, compared with the eager raw-score reference and exact zero output at position 0. The triton-cold-v5 follow-up marks both import-gate and CUDA-unavailable diagnostics with the explicit `CPU-safe skip` marker and checks long-T (`T=257`) fallback parity against blocked and eager references for shared-V and per-head-V layouts. Focused CPU validation reports 39 passed / 3 skipped; the full suite reports 593 passed / 19 skipped / 3 warnings. The amp-train-v4 follow-up adds CPU-safe `float32|bfloat16|float16` configuration coverage, actionable unsupported-CPU skips, and a CUDA-only float16 GradScaler gate; defaults remain fp32 / AMP-off. These are CPU-only parity/contract results; no CUDA allocation or launch occurs on unavailable paths, and no GPU timing, throughput, or win claim is made.

Post-#180/#181 (`f92090f` / `f34d908`): #180 refreshes `OPT_STATUS.md` / `OPT_BACKLOG.md` through #179 while preserving the profile-v16 counts and the real-GPU P0 blocker. #181 extends the CPU-only cache-page report with initial→final packed-cache capacity and allocation accounting; the smoke reports page 8 as `8→128` capacity and `132.0→2112.0 KiB` allocation, page 16 as `16→128` and `264.0→2112.0 KiB`, and the opt-in generate smoke as `8→40`, `132.0→660.0 KiB`, `match_fixed=True`, `aten::cat=0`. Geometric-versus-linear grow/copy counts remain CPU accounting only; defaults, cache policy, attention semantics, and generate behavior are unchanged. No GPU memory measurement, timing, or speedup claim is added; **P0 remains real GPU measurement, including cold CUDA/Triton validation.**

Post-#182 profile-v17 (`4eea9ad`): run the matched profile-v15/v16 schedule after #176–#181 (two warmups, one wait, three active CPU steps). Self-CPU highlights are attention `mul` 24.91% / `bmm` 24.65% / `complex` 14.56% / `copy_` 13.32%; forward `bmm` 30.16% / `mm` 23.01% / `mul` 16.60% / `complex` 9.47% / `copy_` 7.23%; generate `mm` 20.36% / `bmm` 13.29% / `mul` 3.03% / `matmul` 2.48%. Counts match profile-v16: attention `copy_`=2/call, forward `copy_`=12/call, generate `copy_`=394/call, with `cat=0` and `contiguous=0`; the mix is flat trace movement, not a speedup claim. No GPU timing or cold CUDA/Triton validation was added; **P0 remains real GPU measurement.**

Post-#183 docs refresh (`86ba7df`): refresh `OPT_STATUS.md` / `OPT_BACKLOG.md` through #182 while preserving the profile-v17 counts and the real-GPU P0 blocker. Docs-only; no code or GPU evidence.

Post-#184 sparse probe guardrails (`63b25cc`): add stable `exit_code` / `reason` markers for the default-off or completed probe (`0`), an observed density guardrail failure (`2`), and enforcement without density samples (`3`). CPU smoke coverage proves the default-off path remains a hard no-op and that enforcement cannot silently pass without samples. Sparse remains OFF; no GPU timing or sparse-kernel win claim was added.

Post-#185 AUTO threshold gate smoke (`e7943ba`): materialize each sweep child's cold threshold so an omitted override mirrors its de-duplicated decode threshold, including the zero boundary. Malformed threshold input reports a CLI error without dispatching an AUTO child run. This is CPU-only control-flow coverage; eager and `BDH_ATTN_AUTO`-off defaults remain unchanged, with no GPU timing, correctness, or win claim.

Post-#186 generate-benchmark hardening (`9a86f03`): require `eager` as the token reference even for custom `--impls` orders, label implementation and AUTO rows `PASS`/`FAIL`, and return non-zero on token mismatch, decode/cold resolution failure, or any non-zero `aten::cat` count. The AUTO summary now separates `cat0` and `cat1`; the CPU smoke keeps equality at the strict AUTO threshold eager and selects `blocked`/`triton` only when length is greater. Defaults remain eager and attention remains raw scores × strict `tril(diagonal=-1)`; validation is CPU-only with no GPU timing or win claim.

Post-#187 docs refresh (`2088c90`): refresh `OPT_STATUS.md` / `OPT_BACKLOG.md` through #186 while preserving the profile-v17 counts and the real-GPU P0 blocker. Docs-only; no code or GPU evidence.

Post-#188 sampler layout v4 (`87ccac9`): deepen the CPU profiler probe across scaled sampling (`temperature=0.7`), narrow `top_k=8`, and the `top_k=vocab` fallback. Each opt-in branch remains cat-free and shows only the expected one `(B,)` index materialization; no new default `(B,V)` sampler contiguous signature is introduced. Defaults, sampler/RNG behavior, and raw scores × strict `tril(diagonal=-1)` attention remain unchanged. CPU profiler coverage only; no CUDA timing, cold validation, or GPU win claim.

Post-#189 online-decode-v5 (`da79013`): add a CPU parity probe for a B=3, long-S packed CacheManager prefix with shared V at `S=2049`. It checks packed KR/V strides and compares blocked, online, Triton-fallback, and CUDA-reference dispatch against eager decode across a tiled boundary. Eager remains the default, raw scores × strict `tril(diagonal=-1)` is unchanged, and the autograd fallback remains available. CPU-only validation; no GPU decode timing or win claim.

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
- Decode GEMM v2 (#128, `opt/decode-gemm-v2`) — preserve packed CacheManager Q/K/V strides in T=1 Triton decode; avoid per-step contiguous staging; GPU measurement remains open
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
- Docs matrix v30 (#126) — docs-only refresh carrying the #124/#125 tip forward
- Profile CI (#127, `opt/profile-ci`) — CPU profile smoke and optional trace upload scaffold; marked done in the ranked queue
- AUTO threshold sweep (#131, `opt/auto-threshold-sweep`, with #149 `opt/auto-thr-v3` smoke) — stable de-duplicated CPU long-S A/B harness plus dispatcher smoke with parity/cat-free/default-eager checks; GPU threshold re-tune remains open
- CUDA build v2 (#134, `opt/cuda-build-v2`) — explicit no-`nvcc` skip before `CUDAExtension` setup; CPU-safe configuration smoke only; no GPU build claim
- Score×V v3 (#135, `opt/scorev-fuse-v3`) — B=1 T=1 shared-V flattened-view reuse with direct `baddbmm(..., out=target)`; CPU parity only, no GPU timing
- Compile train v3 (#136, `opt/compile-train-v3`) — explicit CPU-safe compile/probe soft-fallback diagnostics and FULLGRAPH/AUTOGRAD boundary docs; no GPU/CUDA-graph measurement
- Docs matrix #140 (`40eac80`) — docs-only refresh through #139 on the #141 code tip
- Blocked-tile-v3 #141 (`ddf58df`) — wide-head CPU cold parity for shared/head-matched V layouts; bounded shared-V behavior; no GPU timing
- AMP train v3 #142 (`68f949a`) — explicit float32/bf16/fp16 configuration contracts with CPU-safe skips; fp32 defaults and CUDA-only GradScaler gating; no GPU throughput claim
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
| **P0** | **Measure Triton/CUDA fused tril-score×V on real GPU** | Default **eager** still (**GPU blocker**; profile-v9, #100, #102, #104, #105, #110, #119, #120, #121, #123, #126, #127, #128, #129, #130, #131, #132, #133, #134, #135, and #136–#140 are CPU/docs/CI evidence only, and #141–#142 add CPU-only parity/configuration evidence, #143 is docs-only, #144 profile-v14 is flat CPU evidence, #145 adds only scriptable sparse-probe exit outcomes, #146 adds CPU-safe skip clarification, and #147 adds CPU-only online-decode parity, #148 is docs-only, #149 adds CPU-only AUTO sweep smoke, #150 adds CPU-safe Triton skip diagnostics, #151 is docs-only, and #152 confirms only the CPU generate copy ceiling; #153 is docs-only; #154 adds only CPU T>1 RoPE narrow reuse/parity coverage; #155 is docs-only; #156 adds only a CPU layout-materialization probe; #157 adds only CPU profile-v15 evidence; #158 is docs-only; #159 adds only CPU logger-fallback coverage; #160 adds only CPU packed-cache footprint accounting; #161 adds only CPU profile-v16 evidence; and #163 adds only structured GPU-measurement/skip scaffolding, #164 adds only packed-view decode preservation/parity coverage, and #165 adds only B>1 shared-V decode parity/accumulation coverage, #166 adds only CPU compile-probe fallback diagnostics, #167 is docs-only, #168 adds only CPU Triton skip-reason/strided-RoPE parity coverage, and #169 adds only CPU prefetch no-op/lifetime documentation, #170 adds only CPU packed-key-view online-decode parity; #171 and #172 are docs-only; #173 adds only CPU analytic-attention parity-matrix coverage; #174 adds only explicit CPU-extension setup smoke, with no CUDA timing; #175 and #177 are docs-only; #176 adds only CPU wide-head partial-tile parity; #178 adds only CPU-safe skip diagnostics and long-T fallback parity; and #179 adds only CPU AMP dtype/scaler contracts; #180 is docs-only; and #181 adds only CPU packed-footprint accounting; #182 adds only matched CPU profile-v17 evidence): attention `bmm` 30.50% / `mul` 30.78% / `copy_` 19.45%; forward `bmm` 34.12% / `mm` 24.93% / `mul` 17.69% / `copy_` 9.09%. #96–#123 add no GPU timing or speedup claim; #126–#142 add no GPU timing or speedup claim; #95 isolated forward `copy_`=24→18, while profile-v9 warmed harness is 16/call and generate is 558/call before #102. #100 lowers blocked/online inference epilogue buffering and tightens T=1 decode tiling; #102/#104 cut and re-measure generate `copy_` ~558→~398 with `cat` at 0; #105 confirms the residual-LN path without GPU evidence; #108 keeps wide-head cold Triton tiles bounded and removes shared-V CPU staging without GPU evidence; #110 cuts generate `copy_` ~398→~394 but leaves V-slot (~132/gen) and multinomial (~128/gen) copies at the probe ceiling; #114 adds only bounded CUDA cold-tile/build-smoke coverage; #116 adds only CPU shared-V T=1 epilogue/parity coverage; #117 adds only CPU compile-probe coverage; #119 adds only CPU AMP gating/bench coverage; #120 adds only CPU AUTO parity/harness coverage; #121 adds only CPU profile evidence; #123 adds only gated CPU sparse density evidence; #124 adds only CPU parity and skip-gated zero-stride Triton RoPE scaffolding; #125 and #126 are docs-only; #127 is CPU CI/profile plumbing; #128 is CPU parity/stride coverage; #129 is CPU profile evidence; #130 is a CPU skip/parity contract; #131 is a CPU AUTO harness extension; #132 and #133 are docs-only; #134 is no-`nvcc` build/config smoke; #135 is CPU B=1 score×V parity coverage; #136 is CPU compile/probe fallback diagnostics; #144 adds no GPU evidence, #145 keeps sparse default-off, and #146/#147 add no GPU evidence; #148 and #151 are docs-only, #149 adds only CPU AUTO sweep smoke, #150 adds CPU-safe Triton skip diagnostics, and #152 confirms only the CPU generate copy ceiling. GPU validation remains open. #93, #108, #110, #112, #116, #119, #121, #123, #127, #128, #129, #130, #131, #134, #135, and #136, #166, #168, #169, and #170 remain CPU-only evidence; #111, #113, #115, #118, #122, #125, #126, #132, #133, #137, and #167 are docs-only; #124 and #168 remain skip-gated GPU-readiness evidence; #169 remains CPU-only prefetch evidence; #173 remains CPU-only analytic-attention parity evidence; #174 remains CPU-only extension setup evidence; #176 remains CPU-only partial-tile parity; #178 remains CPU-only skip/fallback parity; #179 remains CPU-only AMP contract coverage; #180 is docs-only; #181 remains CPU-only packed-footprint evidence; #182 remains CPU-only profile-v17 evidence; #183 is docs-only; #184 remains CPU-only sparse-probe guardrail evidence; #185 remains CPU-only AUTO threshold-gate smoke; #186 remains CPU-only generate-benchmark validation; #187 is docs-only; #188 adds only CPU sampler-layout probe coverage; #189 adds only CPU packed shared-V decode parity. | A100/H100: `bench_gpu_attn.py` (+ fused score×V) | Env blocker
| **P0** | **Cold CUDA/Triton tile/staging validation** | **Landed `opt/triton-cold` + `opt/triton-cold-v2` + #93 CPU tile flattening + #108 `opt/triton-cold-v3` + #114 `opt/cuda-cold-v3` + #120 strict AUTO gates:** adaptive power-of-2 tiles (grow @T≥256 pair #75/#79), wide-head 64×64 bounds, fused strict-tril score×V, shared-V view fallback without `(B*H,T,D)` staging, CPU→blocked adaptive, #114-aligned CUDA cold bounds, and #128 packed-stride T=1 decode coverage + #141 wide-head CPU parity + #170 packed-key-view coverage + #174 explicit CPU extension smoke + #176 blocked-tile-v4 partial-tile parity + #178 triton-cold-v5 skip/fallback parity; #177 is docs-only; #179 is CPU-only AMP contract coverage; #180 is docs-only; #181 is CPU-only packed-footprint accounting; #182 is CPU-only profile-v17 evidence; #93 CPU blocked bench is 1.25× @256 / 4.00× @512 / 5.87× @1024, but GPU measurement and cold CUDA/Triton validation remain open. | A100/H100 microbench; bit-identical | Env blocker |
| **P1** | **`torch.compile` GPU train-step next** | **Landed #117 + #136 + #166**: `BDH_COMPILE_PROBE=train_bwd` probes the compiled forward/backward region; #136 makes construction, missing-target, and first-probe soft fallbacks identify the original eager module and documents the FULLGRAPH/AUTOGRAD boundary; #166 adds explicit `example_y`/forward-probe retry guidance. Optimizer step and grad clearing remain eager. CPU guidance remains `COMPILE=1` **only with eager** + `MODE=default`; blocked/`reduce-overhead` warn. **GPU inductor / CUDA graphs still unmeasured.** | A100/H100: `BDH_COMPILE=0` vs `1` + `MODE=default` vs `reduce-overhead` + `FULLGRAPH` via `benchmarks/bench_train_step.py` | Low
| **P1** | **Decode GEMM / copy tax on generate** | **Host tax cut** #44+#48; **decode-gemm-v2** and #164 preserve packed KR/V strides in T=1 Triton decode without per-step contiguous staging, while #165 deepens B>1 shared-V accumulation in place and #170 preserves packed key/value views for non-flattenable online decode; #131 adds the CPU AUTO threshold sweep, #149 adds CPU-safe dispatcher smoke, #185 adds CPU-only per-child cold-gate fallback smoke, #186 adds CPU-only fail-closed generate checks, #187 is docs-only, #188 adds CPU-only sampler-layout probe coverage, #189 adds CPU-only packed shared-V decode parity, and #152 confirms the remaining CPU-safe copy ceiling, #153 is docs-only, #154 reuses the warmed T>1 RoPE narrow without GPU evidence, #155 is docs-only, #156 only probes remaining CPU layout materializations, and #157 only records CPU profile-v15 evidence; #135 deepens the B=1 T=1 shared-V epilogue alongside **decode-mm** + **decode-online-v2** + **attn-auto** + **triton-decode-v3** + **cuda-decode-v3** + **`opt/prefill-blocked`** + **`opt/auto-tune`**: AUTO long-T cold+decode (adaptive blocked cold @T≥256), with optional independent cold threshold. CPU e2e AUTO **1.26×@1024 / 1.39×@2048**; IMPL=blocked ~1.23–1.43×; short A/B ~1.25× @1024. Remaining = **GPU** measure / re-tune thr. Default still eager. | A100/H100: `bench_generate.py --mode auto-ab` + `--mode impls` + `bench_gpu_attn.py --mode decode`; keep cat-free | Medium |
| **P1** | **CUDA prefetch H2D overlap** | **Landed #92 + #139** `BDH_PREFETCH_H2D=1`: one pinned batch staged on a side stream with event handoff; #139 explicitly verifies that CPU is an identity/no-stream/event/device-lookahead no-op, with no CPU throughput evidence. | CUDA host: measure H2D overlap, lifetime safety, and end-to-end train throughput; keep `BDH_PREFETCH_H2D=1` opt-in to measurement only | Env blocker |
| **P2** | **Fused RoPE kernel** | Attn: `mul`/`copy_` from strided rotate. **Landed #90**, **#124**, **#138**, and **#168**: no expand/stack, table-pair T=1, zero-stride cis-row Triton scaffold, conservative complete-tensor/device/dtype skip gates, actionable skip reasons, strided T=1 output parity, and blocked/PyTorch CPU fallback; default remains eager. GPU fused-RoPE validation is still open. | GPU Triton microbench still open | Low–medium |
| **P2** | **Sparsity follow-through** | **Landed #145 sparse-v3**: stable success/guardrail-failure exit outcomes and coverage/notes build on the gated density probe from `opt/sparse-v2`; `BDH_SPARSE_PROBE` is explicit opt-in, short-train x=26.63% / xy=11.37% at step 150, and the conservative x≥20% / xy≥8% guardrail passes. CPU sparse **never reliably beat dense** → dense/default path and sparse **stay OFF**; GPU sparse remains unmeasured. | GPU sparse bench only if density ≪10% and a kernel win is demonstrated | Speculative |
| **P2** | **Memory layout** | #156 probes the remaining CPU layout materializations without an unsafe activation-layout flip; #157 confirms the warmed CPU operator-count floor. | **Landed #13+#16+#36+#47 + `opt/layout-v2` + #156/#157** — eval cached encoder `F.linear`; train einsum; warm eager forward `aten::contiguous=0`, `aten::cat=0`; remaining clone/sampler signatures are evidence boundaries; GPU impact still needs measurement | Low |
| **P3** | **Hardware / dtype** | **Landed `opt/bf16-train`+#54 `opt/amp-deepen`+#142 `opt/amp-train-v3`:** opt-in `BDH_AMP_DTYPE` with explicit float32/bf16/fp16 configuration coverage, CPU-safe backend skips, and GradScaler gated to CUDA fp16; fp32 defaults remain unchanged. GPU train throughput still open. | GPU box microbench | Env |

### CPU-safe next deepeners (non-blocking)

- Broaden the #160 cache-page sweep across `max_seq` and page sizes when useful, keeping `initial→final` capacity, final packed KR/V `alloc_KiB`, grow counts, and copied-byte accounting together. This remains CPU memory accounting only.
- Keep any further layout/sampler work observational and parity-gated; do not change cache layout, RNG behavior, or defaults without evidence. These CPU probes do not replace the real-GPU P0 work.

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
| CUDA H2D staging | **Landed** #92 + #139 `opt/prefetch-h2d-v2` — `BDH_PREFETCH_H2D=1` default on CUDA; #139 explicitly preserves CPU identity/no-stream/event/device-lookahead no-op; GPU overlap/throughput remains unmeasured |
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
| Rope GPU scaffold | **Landed** #124 `opt/rope-gpu-scaffold` — paired T=1 dispatch through opt-in fused RoPE, zero-stride Triton cis-row scaffold, CPU parity/clean skips; no GPU measurement
| Docs matrix v29 | **Landed** #125 `opt/docs-matrix-v29` — docs-only through #123 on the #124 code tip; superseded by this #124-aligned v30 refresh
| Docs matrix v30 | **Landed** #126 `opt/docs-matrix-v30` — docs-only refresh carrying the #124/#125 documentation tip forward
| Profiler CI artifact | **Landed** #127 `opt/profile-ci` — CPU profile smoke plus optional manual/nightly trace upload; traces remain gitignored; no PR-triggered or GPU claim
| Decode GEMM v2 | **Landed** #128 `opt/decode-gemm-v2` — packed CacheManager Q/K/V strides preserved in T=1 Triton decode; no per-step contiguous staging; no GPU measurement
| Profile v13 | **Landed** #129 `opt/profile-v13` — clean #128 CPU re-profile; attention `copy_`=2/call, forward=12/call, generate=394/call; `cat`/`contiguous`=0; no GPU measurement
| Analytic attn-bwd v2 | **Landed** #130 `opt/attn-bwd-v2` — include `online` beside `blocked` in the CUDA train matrix with `AUTOGRAD=0|1`; CPU skip/parity contract; GPU matrix remains open
| AUTO threshold sweep | **Landed** #131 `opt/auto-threshold-sweep` — stable de-duplicated CPU AUTO generate A/B sweep with seeded parity, strict gates, and `aten::cat=0`; no GPU measurement
| Docs refreshes | **Landed** #132/#133 — docs-only matrix refreshes through #130/#131
| CUDA build v2 | **Landed** #134 `opt/cuda-build-v2` — explicit no-`nvcc` skip before optional `CUDAExtension` setup; CPU-safe smoke, no GPU build claim
| Score×V v3 | **Landed** #135 `opt/scorev-fuse-v3` — B=1 T=1 shared-V flattened-view reuse and direct `baddbmm(..., out=target)`; CPU parity only, no GPU timing
| Compile train v3 | **Landed** #136 `opt/compile-train-v3` — CPU-safe compile/probe soft-fallback diagnostics and FULLGRAPH/AUTOGRAD boundary docs; no GPU/CUDA-graph measurement
| Docs matrix through #136 | **Landed** #137 — docs-only refresh; no code or GPU evidence
| Rope GPU v2 | **Landed** #138 `opt/rope-gpu-v2` — conservative Triton complete-tensor/device/dtype skip gates, CPU T>1/out-buffer parity, eager default; no GPU measurement
| Prefetch H2D v2 | **Landed** #139 `opt/prefetch-h2d-v2` — explicit CPU identity/no-stream/event/device-lookahead no-op contract; CUDA defaults unchanged; no GPU measurement
| Docs matrix through #139 | **Landed** #140 `40eac80` — docs-only refresh on the #141 code tip
| Blocked-tile-v3 | **Landed** #141 `ddf58df` — wide-head CPU cold parity with shared/head-matched V layouts; no GPU timing; GPU measure remains P0
| AMP train v3 | **Landed** #142 `68f949a` — explicit float32/bf16/fp16 configuration contracts with CPU-safe skips; no GPU AMP throughput claim
| Docs refresh | **Landed** #143 — docs-only refresh through #142
| Profile v14 | **Landed** #144 `opt/profile-v14` — flat CPU profile versus v13: attention `copy_`=2/call, forward=12/call, generate=394/call; `cat`/`contiguous`=0; no GPU measurement
| Sparse probe v3 | **Landed** #145 `opt/sparse-v3` — stable success/guardrail-failure exit outcomes plus coverage/notes; probe and sparse production path remain OFF; no GPU sparse claim
| CUDA cold v4 | **Landed** #146 `opt/cuda-cold-v4` — clarify CPU-safe CUDA skip behavior before runtime/build setup; no GPU build or timing claim; P0 GPU validation remains open
| Online decode v3 | **Landed** #147 `opt/online-decode-v3` — reused shared-V views and vectorized B>1 epilogue for opt-in blocked/online T=1 decode; CPU parity/coverage only; default eager and strict raw `tril(-1)` unchanged; no GPU measurement
| Docs refresh | **Landed** #148 `opt/docs-matrix-v35` — docs-only refresh through #147
| AUTO threshold sweep smoke | **Landed** #149 `opt/auto-thr-v3` — CPU-safe smoke for stable de-duplication, recursive-dispatch suppression, malformed-input rejection, and independent cold-gate preservation; eager/AUTO-off defaults unchanged; no GPU measurement
| Triton cold-v4 skip diagnostics | **Landed** #150 `opt/triton-cold-v4` — actionable CPU-safe import/device skip reasons without CUDA allocation or launch; targeted 127 passed / 6 skipped, full suite 548 passed / 19 skipped / 3 warnings; no GPU measurement
| Docs refresh | **Landed** #151 — docs-only refresh through #149 on the #150 code tip
| Generate copy ceiling | **Landed** #152 `opt/gen-copy-v2` — CPU probe confirms packed V snapshots, RoPE pair stores, prompt ownership, and ATen multinomial internals account for the remaining default-eager copy tax; full suite 554 passed / 19 skipped / 3 warnings; no GPU measurement
| Docs refresh | **Landed** #153 — docs-only refresh through #152 on the #154 code tip
| Rope-fuse-v3 | **Landed** #154 `opt/rope-fuse-v3` — reuse warmed T>1 table-backed RoPE narrows with cache invalidation on table rebuild; CPU cache-hit/rebuild parity only; no GPU measurement
| Docs refresh | **Landed** #155 `opt/docs-matrix-v38` — docs-only refresh through #154; no code or GPU evidence
| Layout-v3 materialization probe | **Landed** #156 `opt/layout-v3` — CPU `torch.profiler` evidence for one-time eval-cache materialization, warm eager clone signatures, and ATen sampler contiguous hotspots; no unsafe layout/sampler change or GPU measurement
| Profile-v15 | **Landed** #157 `opt/profile-v15` — CPU self-CPU profile after #156; counts remain attention `copy_`=2/call, forward=12/call, generate=394/call, with `cat=0` and `contiguous=0`; no GPU measurement
| Zerograd v2 | **Landed** #159 `opt/zerograd-v2` — explicit CPU logger fallback/resolved state; CPU-only coverage and defaults preserved; no GPU measurement
| Docs matrix v39 | **Landed** #158 — docs-only refresh through #157 on the #160 code tip
| Cache-page benchmark v2 | **Landed** #160 `opt/cache-bench-v2` — CPU-only `initial→final` capacity and final packed KR/V allocation reporting; geometric/linear final capacity/allocation equality asserted; defaults unchanged and no GPU claim
| Profile v16 | **Landed** #161 `opt/profile-v16` — flat CPU re-profile after #159/#160; attention `copy_`=2/call, forward=12/call, generate=394/call; `cat`/`contiguous`=0; no GPU measurement
| Structured GPU measurement v2 | **Landed** #163 `opt/gpu-measure-v2` — cold/decode/dtype JSON/parity/skip harness and CPU tests; no CUDA run or GPU result
| Packed-view T=1 decode deepen | **Landed** #164 `opt/decode-gemm-v3` — preserve capacity-strided packed CacheManager Q/K/V views without unconditional staging; CPU parity/stride coverage only, no CUDA timing
| Shared-V decode epilogue deepen | **Landed** #165 `opt/scorev-fuse-v4` — accumulate B>1 shared-V tiles in place; CPU strict-tril parity only, no CUDA timing
| Compile-train-v4 fallback guidance | **Landed** #166 `opt/compile-train-v4` — actionable missing-target/first-probe soft-fallback diagnostics; CPU-only, no GPU/CUDA-graph measurement
| Docs matrix v41 | **Landed** #167 `opt/docs-v41` — docs-only refresh through #165
| Fused-RoPE scaffold v3 | **Landed** #168 `opt/rope-gpu-v3` — actionable Triton/CUDA skip reasons plus strided T=1 output parity; CPU-only, no GPU timing
| Prefetch H2D v3 | **Landed** #169 `opt/prefetch-h2d-v3` — expanded CPU no-op coverage and CUDA staged-buffer lifetime documentation; no GPU overlap or throughput claim
| Online-decode v4 | **Landed** #170 `opt/online-decode-v4` — preserve packed key/value views when B*H flattening would copy; CPU parity only, no GPU timing
| Analytic attention backward v3 | **Landed** #173 `opt/attn-bwd-v3` — deepen CPU parity across backends, AUTOGRAD modes, V layouts, non-tile shape, and randomized output gradients; no GPU timing
| CUDA cold v5 | **Landed** #174 `opt/cuda-cold-v5` — explicit CPU-only extension setup smoke with no-CUDA/no-nvcc skips preserved; no GPU timing or claim
| Docs refresh #175 | **Landed** `opt/docs-v43` — docs-only refresh through #174; profile-v16 counts and the real-GPU P0 blocker preserved
| Blocked-tile-v4 (#176) | **Landed** `opt/blocked-tile-v4` at `4e13111` — CPU-only wide-head partial-tile parity for shared-V and head-matched-V layouts; no GPU timing or win claim
| Docs refresh #177 | **Landed** `5507747` — docs-only refresh through the blocked-tile-v4 tip; profile-v16 counts and the real-GPU P0 blocker preserved
| Triton cold-v5 (#178) | **Landed** `opt/triton-cold-v5` at `f72033e` — CPU-safe skip markers plus long-T fallback parity for shared-V and per-head-V layouts; no CUDA allocation/launch/timing or GPU win claim
| AMP train-v4 (#179, current tip) | **Landed** `opt/amp-train-v4` at `0983326` — CPU-safe dtype/scaler contracts with fp32 / AMP-off defaults; no GPU timing or throughput claim
| Docs refresh #180 | **Landed** `f92090f` — docs-only refresh through #179; profile-v16 counts and the real-GPU P0 blocker preserved
| Cache-page benchmark v3 (#181, current tip) | **Landed** `opt/cache-bench-v3` at `f34d908` — CPU-only initial→final packed-cache footprint/allocation reporting and invariant coverage; defaults unchanged, no GPU memory or timing claim
| Profile v17 (#182, current tip) | **Landed** `opt/profile-v17` at `4eea9ad` — matched CPU re-profile is flat versus v16; attention/forward/generate `copy_` counts remain 2/12/394 with `cat=0`, `contiguous=0`; no GPU measurement
| Docs refresh #183 | **Landed** `86ba7df` — docs-only refresh through #182; profile-v17 counts and the real-GPU P0 blocker preserved
| Sparse probe guardrails #184 | **Landed** `63b25cc` — CPU-only stable probe exit markers and default-off/no-samples smoke coverage; sparse remains OFF with no GPU claim
| AUTO threshold gate smoke #185 (current tip) | **Landed** `e7943ba` — CPU-only per-child cold-threshold fallback and malformed-input no-dispatch coverage; defaults unchanged and no GPU claim
| Generate benchmark v3 #186 (current tip) | **Landed** `9a86f03` — CPU-only eager-reference ordering, PASS/FAIL status, fail-closed token/cat checks, separate AUTO `cat0`/`cat1` summary, and strict threshold smoke; no GPU claim
| Docs refresh #187 | **Landed** `2088c90` — docs-only refresh through #186; profile-v17 counts and the real-GPU P0 blocker preserved
| Sampler layout v4 #188 (current tip) | **Landed** `87ccac9` — CPU-only scaled/top-k sampler layout probe coverage; one expected `(B,)` index materialization per opt-in case, no `aten::cat`, no new `(B,V)` contiguous signature, and no GPU claim
| Online decode v5 #189 (current tip) | **Landed** `da79013` — CPU-only B>1 long-S packed shared-V parity across blocked/online/Triton-fallback/CUDA-reference dispatch; no GPU claim
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

| **P2** | **Analytic attn train on GPU** | CPU blocked|online + tiled analytic bwd landed (`opt/blocked-autograd` #41); #96 adds the CUDA parity/config harness, and **#130** keeps the `online` alias beside `blocked` across `AUTOGRAD=0|1` with a clean CPU skip/parity contract. **GPU** train-step with `IMPL=blocked|online|triton|cuda` remains unmeasured. | A100/H100 `bench_attn_bwd.py` / #96 + #130 matrix | Low |

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
# eager vs compiled train-step (default mode=default, probe=train_bwd)
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

**CPU honesty (this box / profile source `f10bdd4`; documented code tip `6fd7950`):**
- `--mode impls` short prompt: medians ~noise vs eager; match; `aten::cat=0`
- `--mode auto-ab`: AUTO fires @ S>512; tokens match; cats=0; **e2e AUTO**
  **1.26× @1024 / 1.39× @2048** after `opt/prefill-blocked` (cold+decode);
  IMPL=blocked ~1.23–1.43×
- **Do not** claim Triton/CUDA / AUTO e2e wins from CPU. GPU thr re-tune = P1.
Private repo only — never `pathwaycom/*`.
