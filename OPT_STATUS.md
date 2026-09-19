# OPT status — landed work (#1–#463; docs-v80)

Private sandbox: `katulevskiy/bdh-gpu-opt`.

Documentation coverage: #418–#463 plus documented tip `d9703b0` (#434), with prior coverage retained. This branch is based on current main tip `6d29504` (#465); #464–#465 remain outside this refresh scope.

## Evidence boundary

Attention remains **raw scores × strict lower-triangular** `tril(diagonal=-1)`: no softmax, no `1/√d`, and no `F.scaled_dot_product_attention`.

This sandbox is CPU-only (`torch 2.14.0+cu130`, `cuda=False`). CPU tests establish correctness, layout, dispatch, and control-flow contracts only. They do not establish GPU speedups, GPU throughput, CUDA correctness, or Triton timing.

**P0 remains real GPU measurement plus cold CUDA–Triton validation.** Keep production defaults unchanged until that evidence exists.

## Current matrix

| Area | Current contract | Evidence boundary |
|---|---|---|
| Device | CPU-only; CUDA/Triton paths skip or fall back cleanly here | No GPU claim |
| Attention | `eager` default; `blocked`, `online`, `triton`, and `cuda` opt-in; strict-past backward now covers padded/strided/zero-stride Q/K/V, zero-stride Q/K gradient reduction including batch/head-broadcast Q/K views, strided upstream gradients, packed-decode V, and empty-sequence contracts on CPU | Raw strict-tril parity on CPU; GPU measure open |
| AUTO | Off by default; decode threshold remains independent from the optional cold threshold; blank/unset transitions, per-call requested-backend precedence, caller-argument isolation, and malformed/continuing threshold sweeps fail closed before execution | CPU dispatch/parity only |
| RoPE | `eager` default; fused path opt-in; paired output-slot aliasing and incompatible cis leading dimensions are rejected safely; mixed-dtype, strided-output parity is covered through the public T>1 dispatcher | CPU shape/parity only; fused GPU validation open |
| Compile | Off by default; invalid and normalized `BDH_COMPILE_PROBE` values fail closed to `train_bwd`, including missing-`example_y` training callers and missing-`example_x` no-probe callers, while failed probes and construction fallback preserve caller mode and gradients | CUDA graphs/inductor unmeasured |
| AMP | fp32/off by default; explicit `BDH_AMP_FORWARD_ONLY` off/on spellings remain CPU-safe, and a live CUDA runtime is required before exposing GPU throughput while bf16/fp16 CPU forward-mode contexts remain contract-covered and GradScaler stays disabled | CPU smoke only; GPU train throughput open |
| Sparse ReLU | Off by default; explicit probe guardrails distinguish no-sample and terminal density-floor failure exits, including CLI floor overrides, while a passing sample permits only a mocked CPU crossover sweep | CPU density/control flow only; no sparse-kernel result |
| DataLoader | Validation split forwarding and repeated worker-backed batches preserve CPU tensor identity; asynchronous producer failures reach `next()` with the host-gather cause; worker initialization locks full batches, bounded prefetch, persistent workers, and distinct reproducible CPU RNG streams, while negative `NUM_WORKERS` stays synchronous with pinning and persistent prefetch disabled | CPU identity/opt-out contracts only; H2D overlap unmeasured |
| Sampling | Strided logits and sampler probability buffers across multinomial, narrow/full/overflow-clamped top-k, zero-stride, and combined non-unit-vocabulary-stride singleton output views preserve RNG parity, identity, and neighbor isolation | CPU contract only |
| Packed generate | Cache path remains cat-free (`aten::cat=0`); interrupted cat probes restore their hook; decode-only AUTO overrides remain isolated; packed decode preserves raw score×V semantics across capacity-strided V layouts; threshold sweeps continue after individual failures while retaining aggregate failure | CPU operator contract, not GPU timing |

The retained profile-v20 baseline is `aten::copy_` 2/call for attention, 12/call for forward, and 394/call for generate, with `aten::cat=0` and `aten::contiguous=0`. The #403–#463 additions and tip follow-up are CPU-safe contracts, diagnostics, and docs; #450 adds schema-v5 CUDA-build/visible-device skip fields but no timing. These are CPU call-count observations, not GPU performance claims.

## Landed in this refresh

| PR | Branch / scope | What the matrix records | Result |
|---:|---|---|---|
| **#350** | `opt/docs-v72` / docs | Previous refresh through #349 | Docs-only; P0 unchanged |
| **#351** | `opt/rope-v12` / RoPE | CPU shape contracts for unpaired/paired entrypoints, odd dimensions, cis mismatches, malformed outputs, and paired-axis mismatches | CPU contract only |
| **#352** | `opt/gen-bench-v11` / AUTO | Explicit cold-threshold sweep remains independent of the decode threshold | CPU contract only |
| **#353** | `opt/compile-v13` / compile | Unsupported `BDH_COMPILE_PROBE` fails closed to `train_bwd` with clean probe gradients and mode preservation | CPU contract only |
| **#354** | `opt/cuda-build-v11` / CUDA build | `BDH_FORCE_CPU_EXT=1` is inert unless `BDH_BUILD_EXT=1`; pure-Python install remains safe | CPU contract only |
| **#355** | `opt/attn-bwd-v12` / attention backward | Strict-past backward parity at the default 64-row tile boundary, including diagonal/future gradient zeros | CPU contract only |
| **#356** | `opt/gpu-measure-v11` / GPU attention benchmark | Benchmark reference remains raw scores × strict `tril(diagonal=-1)` × V, without scaling or softmax | CPU contract only; no GPU timing |
| **#357** | `opt/prefetch-v12` / DataLoader | Worker-backed batches preserve CPU tensor identity through `_to_train_device` without CUDA setup | CPU contract only |
| **#358** | `opt/sparse-probe-v13` / sparse probe | Enforced guardrails have a distinct no-sample exit; crossover work stays skipped | CPU contract only; no GPU claim |
| **#359** | `opt/amp-train-v13` / AMP | CPU execution cannot expose a GPU throughput claim; throughput remains gated to real CUDA | CPU contract only |
| **#360** | `opt/auto-thr-v14` / AUTO | Blank/whitespace cold-threshold fallback keeps strict equality and above-threshold decode behavior aligned | CPU contract only |
| **#361** | `opt/blocked-tile-v13` / blocked prefill | Float16/bfloat16 CPU inputs retain dtype while preserving raw strict-tril parity across the partial 128-row tile | CPU contract only |
| **#362** | `opt/decode-gemm-v12` / decode | Eager, blocked, online, Triton-fallback, and CUDA dispatch preserve raw QK-transpose-times-V semantics without softmax or scaling | CPU contract only |
| **#363** | `opt/online-decode-v14` / online decode | Empty past tensors return exact zero score×V output with expected shape and dtype across backends | CPU contract only; no GPU claim |
| **#364** | `opt/scorev-v14` / score-V backward | Capacity-padded, non-contiguous shared-V views preserve blocked/online strict-past gradient parity without a staging copy | CPU contract only |
| **#365** | `opt/docs-v73` / docs | Previous matrix/backlog refresh through #363 | Docs-only; P0 unchanged |
| **#366** | `opt/rope-fuse-v13` / RoPE | Paired output-slot aliasing is rejected while CPU shape-contract coverage remains intact | CPU contract only |
| **#367** | `opt/layout-v13` / sampling | Strided `(B, 1)` output views preserve RNG parity, return identity, and isolate neighbors | CPU contract only |
| **#368** | `opt/gen-bench-v12` / AUTO | Decode-only threshold overrides leave an omitted cold threshold independent and restore it unchanged | CPU contract only |
| **#369** | `opt/cuda-build-v12` / CUDA build | Explicit CPU mode wins over discoverable `nvcc` and a simultaneous CUDA request | CPU contract only |
| **#370** | `opt/compile-v14` / compile | Case-insensitive, whitespace-tolerant probe names preserve `train_bwd` cleanup and training mode | CPU contract only |
| **#371** | `opt/attn-bwd-v13` / attention backward | Empty-prefix strict-past rows produce zero output and zero query gradient | CPU contract only |
| **#372** | `opt/prefetch-v13` / DataLoader | Repeated prefetched worker batches remain CPU identity paths without CUDA setup | CPU contract only |
| **#373** | `opt/gpu-measure-v12` / GPU decode benchmark | Packed-past decode coverage locks raw score×V semantics without scaling or softmax | CPU contract only; no GPU timing |
| **#374** | `opt/amp-train-v14` / AMP | Active CPU bf16/fp16 contexts never report a CUDA throughput claim | CPU contract only |
| **#375** | `opt/sparse-probe-v14` / sparse probe | Density-guardrail failure has a distinct exit and skips CPU crossover work | CPU contract only; no GPU claim |
| **#376** | `opt/blocked-tile-v14` / blocked prefill | Low-precision capacity-padded, non-contiguous V views preserve dtype, first-row masking, and raw strict-tril parity | CPU contract only |
| **#377** | `opt/auto-thr-v15` / AUTO | Clearing a previously parsed blank cold-threshold override resumes the live decode threshold, including strict equality and above-threshold behavior | CPU contract only |
| **#378** | `opt/online-decode-v15` / online decode | Nonempty float64 tiled decode preserves query dtype while matching eager raw score×V semantics | CPU contract only; no GPU claim |
| **#379** | `opt/docs-v74` / docs | Previous matrix/backlog refresh through #376 | Docs-only; P0 unchanged |
| **#380** | `opt/scorev-v15` / score-V | Capacity-strided Q/K/V views preserve blocked/online output parity with eager strict-tril score×V math | CPU contract only |
| **#381** | `opt/layout-v14` / sampling | Zero-stride `(B, 1)` outputs preserve multinomial/top-k identity, RNG parity, and neighbor isolation | CPU contract only |
| **#382** | `opt/gen-bench-v13` / generate benchmark | Interrupted `count_torch_cat` probes restore the temporary `torch.cat` hook | CPU contract only |
| **#383** | `opt/rope-fuse-v14` / RoPE | Paired T=1 fp16-input/fp32-cache output parity remains covered | CPU contract only; fused GPU validation open |
| **#384** | `opt/compile-v15` / compile | Unsupported probe names preserve the CPU-safe `train_bwd` cleanup contract | CPU contract only |
| **#385** | `opt/cuda-build-v13` / CUDA build | `nvcc` directory discovery remains covered without changing force-CPU opt-in behavior | CPU contract only |
| **#386** | `opt/attn-bwd-v14` / attention backward | Analytic strict-tril backward parity holds for non-contiguous Q/K/V views | CPU contract only |
| **#387** | `opt/prefetch-v14` / DataLoader | CPU DataLoader H2D opt-out remains explicit and clean under the deeper contract | CPU contract only; overlap unmeasured |
| **#388** | `opt/decode-gemm-v13` / decode | Shared and per-head capacity-strided V layouts preserve raw score×V semantics across CPU-safe decode dispatch | CPU contract only; no GPU timing |
| **#389** | `opt/gpu-measure-v13` / GPU decode benchmark | Packed decode layouts are included in the GPU-measure reference contract | CPU contract only; no GPU timing |
| **#390** | `opt/amp-train-v15` / AMP | CPU AMP never reports GPU throughput, even when CUDA is reported available; GradScaler remains disabled | CPU contract only |
| **#391** | `opt/docs-v75` / docs | Previous matrix/backlog refresh through #388 | Docs-only; P0 unchanged |
| **#392** | `opt/sparse-probe-v15` / sparse probe | Terminal `xy` density-floor failure remains distinct and skips crossover work | CPU contract only; no GPU claim |
| **#393** | `opt/blocked-tile-v15` / blocked prefill | Capacity-padded non-contiguous V preserves forward/backward parity and zero padded-column gradients | CPU contract only |
| **#394** | `opt/auto-thr-v16` / AUTO | Blank-to-explicit cold-threshold transitions keep the later cold override independent from decode changes | CPU contract only |
| **#395** | `opt/online-decode-v16` / online decode | Empty past with float64 queries returns typed, shaped zero score×V output across backends | CPU contract only; no GPU claim |
| **#396** | `opt/scorev-v16` / score-V | Blocked and online dispatch retain raw score×strict-tril score-V semantics, including row-zero behavior | CPU contract only |
| **#397** | `opt/layout-v15` / sampling | Strided `(B, T, V)` prefill logits slices preserve singleton output identity, RNG parity, and neighbor isolation | CPU contract only |
| **#398** | `opt/rope-fuse-v15` / RoPE | Paired T=1 RoPE preserves fp16-input/fp32-cache parity on non-contiguous output slots | CPU contract only; fused GPU validation open |
| **#399** | `opt/gen-bench-v14` / generate benchmark | Malformed AUTO threshold sweeps fail closed before any run or model setup | CPU contract only |
| **#400** | `opt/compile-v16` / compile | Failed `train_bwd` probes restore eval mode and clear partial gradients before eager fallback | CPU contract only |
| **#401** | `opt/cuda-build-v14` / CUDA build | A directory named `nvcc` under `CUDA_PATH` is rejected cleanly, preserving CPU fallback behavior | CPU contract only |
| **#402** | `opt/attn-bwd-v15` / attention backward | Strict-tril backward covers non-contiguous upstream gradients across eager, blocked, online, Triton fallback, and CUDA fallback paths | CPU contract only; no GPU claim |
| **#403** | `opt/amp-train-v16` / AMP | Live CUDA is required before exposing the AMP GPU throughput claim; stale CUDA selection remains CPU-safe | CPU contract only; no GPU claim |
| **#404** | `opt/gpu-measure-v14` / GPU measurement | Cold-path CUDA-unavailable diagnostics retain native handoff commands; CPU skips remain non-timed | CPU contract only; no GPU timing |
| **#405** | `opt/prefetch-v15` / DataLoader | Validation split forwarding preserves host batch gathering, pin-memory, and tensor identity behavior | CPU contract only |
| **#406** | `opt/docs-v76` / docs | Previous matrix/backlog refresh through #402, rebased onto #404 | Docs-only; P0 unchanged |
| **#407** | `opt/blocked-tile-v16` / blocked prefill | Capacity-padded, non-contiguous Q/K views preserve eager parity and strict-past masking on the long-T blocked path | CPU contract only |
| **#408** | `opt/sparse-probe-v16` / sparse probe | A passing density sample permits the mocked CPU crossover sweep while guardrails remain explicit | CPU contract only; no GPU claim |
| **#409** | `opt/auto-thr-v17` / AUTO | Removing an explicit cold threshold falls back live to the decode threshold with strict boundaries preserved | CPU contract only |
| **#410** | `opt/online-decode-v17` / online decode | Nonempty float64 packed-cache views preserve shape, dtype, and raw score×V parity across CPU-safe dispatch | CPU contract only; no GPU claim |
| **#411** | `opt/layout-v16` / sampling | Non-unit vocabulary strides preserve multinomial/top-k output isolation and contiguous-reference parity | CPU contract only |
| **#412** | `opt/scorev-v17` / score-V | Distinct Q/K with per-head V preserve raw score×strict-tril semantics, including row-zero behavior | CPU contract only; no GPU claim |
| **#413** | `opt/gen-bench-v15` / generate benchmark | Decode-only AUTO override does not create or leak an omitted cold-threshold variable | CPU contract only |
| **#414** | `opt/rope-fuse-v16` / RoPE | Paired T=1 mixed-dtype, strided-output parity remains covered through the public dispatcher | CPU contract only; fused GPU validation open |
| **#415** | `opt/compile-v17` / compile | Compile-construction fallback preserves caller mode and existing gradients | CPU contract only |
| **#416** | `opt/attn-bwd-v16` / attention backward | Empty-sequence strict-tril backward is covered across eager, blocked, online, Triton fallback, and CUDA fallback paths | CPU contract only; no GPU claim |
| **#417** | `opt/cuda-build-v15` / CUDA build | Executable `CUDA_PATH/bin/nvcc` discovery is covered through setup metadata without GPU claims | CPU contract only; no GPU claim |

## Refresh additions (#418–#433 and tip `d9703b0`)

| PR / tip | Branch / scope | What the matrix records | Result |
|---:|---|---|---|
| **#418** | `opt/prefetch-v16` / DataLoader | Asynchronous prefetch producer failures reach the consumer through `next()` with the original host-gather exception as cause | CPU contract only |
| **#419** | `opt/amp-train-v17` / AMP | A live CUDA runtime is the positive gate for a throughput claim; the contract remains testable without GPU execution | CPU contract only; no measured throughput |
| **#420** | `opt/gpu-measure-v15` / GPU measurement | CUDA-unavailable cold runs emit structured, machine-readable no-timing diagnostics | CPU contract only; no GPU timing |
| **#421** | `opt/docs-v77` / docs | Previous matrix/backlog refresh through #417, rebased onto #420 | Docs-only; P0 unchanged |
| **#422** | `opt/sparse-probe-v17` / sparse probe | The final density sample controls guardrail enforcement and blocks CPU crossover work on failure | CPU contract only; no GPU claim |
| **#423** | `opt/blocked-tile-v17` / online prefill | Capacity-padded Q/K coverage extends to online prefill with eager parity and strict-past masking | CPU contract only |
| **#424** | `opt/auto-thr-v18` / AUTO | An explicit cold threshold stays scoped to the cold gate while the shared decode threshold remains independent | CPU contract only |
| **#425** | `opt/online-decode-v18` / online decode | Multi-query public dispatch preserves score×V parity and read-only behavior for nonzero-offset packed K/V views | CPU contract only |
| **#426** | `opt/scorev-v18` / score-V | Distinct-Q/K, per-head-V raw score×V semantics hold through every public dispatch alias and CPU fallback | CPU contract only; no performance claim |
| **#427** | `opt/rope-fuse-v17` / RoPE | Public T>1 dispatch preserves mixed-dtype, strided-output parity for eager and fused implementations | CPU contract only; fused GPU validation open |
| **#428** | `opt/gen-bench-v16` / generate benchmark | Negative explicit AUTO cold thresholds fail before model setup | CPU contract only |
| **#429** | `opt/layout-v17` / sampling | Non-unit vocabulary strides combined with zero-stride outputs preserve RNG parity and isolation across sampler paths | CPU contract only |
| **#430** | `opt/compile-v18` / compile | Missing `example_y` uses the default `train_bwd` fallback without probing or returning a compiled wrapper; caller mode and gradients remain intact | CPU contract only |
| **#431** | `opt/cuda-build-v16` / CUDA build | Explicit CUDA builds can select executable `PATH/nvcc` while missing CUDA-home paths remain explicit | CPU setup contract only; no GPU claim |
| **#432** | `opt/attn-bwd-v17` / attention backward | Zero-stride per-head V forward parity and gradient reduction to the expanded base hold across dispatch variants | CPU contract only; no GPU claim |
| **#433** | `opt/gpu-measure-v16` / GPU measurement | CUDA-unavailable summaries preserve the requested benchmark mode and explicit cold/decode handoffs | CPU contract only; no GPU timing |
| **tip `d9703b0`** | DataLoader worker prefetch | Full host batches, bounded prefetch, persistent workers, and per-worker initialization are locked into the CPU-safe worker contract | CPU contract only; H2D overlap unmeasured |

## Refresh additions (#435–#448)

| PR | Branch / scope | What the matrix records | Result |
|---:|---|---|---|
| **#435** | `opt/amp-train-v18` / AMP | Full-forward and logits-only bf16/fp16 selection remains CPU-safe and mode-only | CPU contract only; no GPU throughput |
| **#436** | `opt/sparse-probe-v18` / sparse probe | CLI density floors override defaults and a custom floor failure stops the CPU crossover path | CPU contract only; no GPU claim |
| **#437** | `opt/docs-v78` / docs | Previous matrix/backlog refresh through #433 and tip `d9703b0` | Docs-only; P0 unchanged |
| **#438** | `opt/blocked-tile-v18` / attention backward | Capacity-padded Q/K views preserve blocked/online strict-tril parity, first-row zeroing, and padded-column gradients | CPU contract only |
| **#439** | `opt/auto-thr-v19` / AUTO | Explicit blocked, Triton, CUDA, and online selections remain unchanged across cold/decode gates | CPU contract only; no GPU claim |
| **#440** | `opt/online-decode-v19` / online decode | Direct tiled decode covers multi-query, per-head V, offset-packed K/V, and read-only inputs | CPU contract only |
| **#441** | `opt/scorev-v19` / score-V | Distinct-Q/K and per-head-V gradients preserve strict-tril parity across dispatch aliases | CPU contract only; no GPU claim |
| **#442** | `opt/rope-fuse-v18` / RoPE | Public T=1 dispatch preserves mixed-dtype writes into non-contiguous cache slots | CPU contract only; fused GPU validation open |
| **#443** | `opt/layout-v18` / sampling | Non-unit vocabulary stride plus unit batch stride preserves RNG parity, identity, and untouched-neighbor isolation | CPU contract only |
| **#444** | `opt/gen-bench-v17` / generate benchmark | AUTO threshold sweeps isolate per-run arguments without mutating caller state | CPU contract only |
| **#445** | `opt/cuda-build-v17` / CUDA build | Executable `CUDA_HOME/bin/nvcc` is selected while invalid CUDA_PATH/PATH entries stay isolated | CPU setup contract only; no GPU claim |
| **#446** | `opt/compile-v19` / compile | Training-mode missing-target callers skip probing and preserve existing gradients | CPU contract only |
| **#447** | `opt/attn-bwd-v18` / attention backward | Zero-stride Q/K head views reduce gradients to their underlying bases across implementation selectors | CPU contract only; no GPU claim |
| **#448** | `opt/prefetch-v18` / DataLoader | Worker initialization provides distinct, reproducible CPU RNG streams without changing prefetch code | CPU contract only; H2D overlap unmeasured |


## Refresh additions (#449–#463)

| PR | Branch / scope | What the matrix records | Result |
|---:|---|---|---|
| **#449** | `opt/amp-train-v19` / AMP | Explicit `BDH_AMP_FORWARD_ONLY` off/on spellings remain CPU-safe without a GPU throughput claim | CPU contract only; no GPU throughput |
| **#450** | `opt/gpu-measure-v17` / GPU measurement | CUDA build and visible-device skip diagnostics are exposed in schema v5 without substituting CPU timing | CPU contract only; no GPU timing |
| **#451** | `opt/sparse-probe-v19` / sparse probe | A custom `--min-final-xy` floor is reported and stops before CPU crossover work on failure | CPU contract only; no GPU claim |
| **#452** | `opt/blocked-tile-v19` / attention backward | Strict-tril blocked/online prefill backward at a tile boundary leaves diagonal/future sources gradient-free | CPU contract only |
| **#453** | `opt/docs-v79` / docs | Previous matrix/backlog refresh through #448 | Docs-only; P0 unchanged |
| **#454** | `opt/auto-thr-v20` / AUTO | Per-call requested backend precedence remains explicit against conflicting environment settings across cold/decode gates | CPU contract only |
| **#455** | `opt/scorev-v20` / score-V | Distinct-Q/K shared-V and per-head-V gradients preserve raw strict-tril score×V semantics across dispatch aliases | CPU contract only; no GPU claim |
| **#456** | `opt/online-decode-v20` / online decode | T=1 per-head-V decode preserves raw score×V parity for offset-packed K/V while inputs remain read-only | CPU contract only; no GPU claim |
| **#457** | `opt/layout-v19` / sampling | Strided sampler probability buffers preserve multinomial and top-k contracts, including overflow-clamped paths | CPU contract only |
| **#458** | `opt/gen-bench-v18` / generate benchmark | Threshold sweeps continue after an individual failure while retaining aggregate failure status | CPU contract only |
| **#459** | `opt/rope-fuse-v19` / RoPE | Incompatible cis leading dimensions are rejected before rotation math on regular and paired entrypoints | CPU contract only; fused GPU validation open |
| **#460** | `opt/compile-v20` / compile | Missing-`example_x` callers receive an unprobed wrapper without executing it; mode and gradients remain intact | CPU contract only |
| **#461** | `opt/attn-bwd-v19` / attention backward | Batch/head-broadcast zero-stride Q/K views reduce gradients to singleton bases across implementation selectors | CPU contract only; no GPU claim |
| **#462** | `opt/cuda-build-v18` / CUDA build | A stale non-executable `CUDA_HOME/bin/nvcc` falls back to executable `CUDA_PATH/bin/nvcc` in CPU-safe setup metadata | CPU setup contract only; no GPU claim |
| **#463** | `opt/prefetch-v19` / DataLoader | Negative `NUM_WORKERS` remains synchronous CPU loading with pinning and persistent-worker prefetch disabled | CPU identity/opt-out contract only |

## Defaults and operator guidance

- `BDH_ATTN_IMPL=eager` remains the default reference path.
- `BDH_ATTN_AUTO` remains unset/off; `BDH_ATTN_AUTO_THRESHOLD=512` is the retained decode threshold, with an optional independent cold threshold.
- `BDH_ATTN_AUTOGRAD` remains opt-in. Blocked/online paths use tiled analytic backward only when explicitly enabled.
- `BDH_ROPE_IMPL=eager`, `BDH_COMPILE=0`, `BDH_AMP_DTYPE=float32`, and sparse ReLU OFF remain unchanged.
- `BDH_FORCE_CPU_EXT=1` remains inert unless native extension build is explicitly enabled.
- Do not turn CPU medians, profiler times, density, skip results, or parity tests into GPU performance claims.

## Next measurement gate

On a real CUDA box, run the GPU attention harness for eager/blocked/online/Triton/CUDA on cold and T=1 decode shapes, then record GPU name, torch/CUDA versions, medians, and parity deltas. Keep the default eager path until those results and cold CUDA–Triton validation are available.

See `OPT_BACKLOG.md` for the ranked P0/P1/P2 work and `OPT_NOTES.md` for detailed historical measurements.
