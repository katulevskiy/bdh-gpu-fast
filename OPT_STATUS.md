# OPT status — landed work (#1–#568; docs-v87)

Private sandbox: `katulevskiy/bdh-gpu-opt`.

Documentation coverage: #418–#568 plus documented tips `d9703b0` (#434) and `6181ba6`, with prior coverage retained. This branch was started from requested tip `49b730e` (#568); #540–#568 are recorded below.

## Evidence boundary

Attention remains **raw scores × strict lower-triangular** `tril(diagonal=-1)`: no softmax, no `1/√d`, and no `F.scaled_dot_product_attention`.

This sandbox is CPU-only (`torch 2.14.0+cu130`, `cuda=False`). CPU tests establish correctness, layout, dispatch, and control-flow contracts only. They do not establish GPU speedups, GPU throughput, CUDA correctness, or Triton timing.

**P0 remains real GPU measurement plus cold CUDA–Triton validation.** Keep production defaults unchanged until that evidence exists.

## Current matrix

| Area | Current contract | Evidence boundary |
|---|---|---|
| Device | CPU-only; CUDA/Triton paths skip or fall back cleanly here; structured GPU skips classify build/device/runtime state, retain the requested configuration, keep schema-v9 stdout/JSON skip records aligned with `timing_scope=none`, and keep `BDH_BUILD_CUDA` exact (`1` only); only `BDH_FORCE_CPU_EXT=1` selects CPU-only extension setup, even when CUDA is detected | No GPU claim |
| Attention | `eager` default; `blocked`, `online`, `triton`, and `cuda` opt-in; strict-past backward now covers padded/strided/zero-stride Q/K/V, Q/K backing-storage gradients, shared-Q batch/head-broadcast and zero-stride Q/K/V gradient reduction, batch/head-broadcast upstream gradients, dispatcher-selected blocked/online partial long-prefill tiles, both sides of the first and second default 64-row tiles, strided upstream gradients, packed-decode V, empty-sequence contracts, and multi-query upstream rows spanning tile boundaries on CPU | Raw strict-tril parity on CPU; GPU measure open |
| AUTO | Off by default; decode threshold remains independent from the optional cold threshold; blank/unset and zero-threshold transitions, strict cold/decode boundaries, Triton preference when available, per-call requested-backend precedence including the `online` alias over a conflicting eager environment, runtime disable/re-enable transitions, threshold preservation, caller-argument isolation, and malformed/continuing threshold sweeps fail closed before execution; invalid cold thresholds do not poison the decode gate | CPU dispatch/parity only |
| RoPE | `eager` default; fused path opt-in; paired output-slot aliasing, overlapping direct/view/shifted output aliases, public T>1 cosine/sine aliases including cached-cis overlap rejection before stores, incompatible cis leading dimensions, and CPU-safe blocked/Triton-fallback backward with pair-specific upstream gradients and non-contiguous pair-strided inputs/output cache slots are covered; mixed-dtype, strided-output parity is covered through the public T>1 dispatcher | CPU shape/parity only; fused GPU validation open |
| Compile | Off by default; invalid and normalized `BDH_COMPILE_PROBE` values fail closed to `train_bwd`, including missing-`example_y` training callers and missing-`example_x` no-probe callers; input-only forward probes succeed without targets; successful probes clear probe and pre-existing gradients and restore caller training state, while failed probes and construction fallback preserve the eager module, caller mode, gradients, and compile options across both fullgraph settings | CUDA graphs/inductor unmeasured |
| AMP | fp32/off by default; explicit `BDH_AMP_FORWARD_ONLY` off/on spellings remain CPU-safe; switching from active bf16/fp16 forward-only mode back to fp32 clears that mode, logits-only forward runs inside the forward-only context with cross-entropy afterward in fp32, the default full-forward path passes targets inside the AMP context, and a live CUDA runtime is required before exposing GPU throughput while failed context construction preserves the complete prior AMP state, CPU contexts remain contract-covered, and GradScaler stays disabled | CPU smoke only; GPU train throughput open |
| Sparse ReLU | Off by default; the disabled gate wins over guardrail enforcement without starting training or crossover work; explicit probe guardrails distinguish no-sample and terminal density-floor failure exits, including CLI floor overrides, while a passing sample permits only a mocked CPU crossover sweep | CPU density/control flow only; no sparse-kernel result |
| DataLoader | Validation split forwarding and repeated worker-backed batches preserve CPU tensor identity; terminal asynchronous producer failures reach every pending `next()` promptly with the host-gather cause; worker initialization locks full batches, bounded one-slot prefetch, persistent workers, and distinct reproducible CPU RNG streams; zero-worker loading keeps `batch_size=None` so each item remains a full host batch; worker construction preserves the full-batch iterable and requested split, shutdown stops production and clears the one-slot queue, terminal producer failure leaves no live thread or queued resources and close is idempotent, while negative `NUM_WORKERS` stays synchronous with pinning and persistent prefetch disabled | CPU identity/opt-out contracts only; H2D overlap unmeasured |
| Sampling | Strided logits and sampler probability buffers across multinomial, narrow/full/overflow-clamped top-k, zero-stride, and combined padded non-unit-vocabulary scratch with strided singleton output views preserve RNG parity, identity, contiguous-reference parity, and neighbor isolation, including narrow top-k reuse, padded scratch protection, and the deeper padded narrow top-k case | CPU contract only |
| Packed generate | Cache path remains cat-free (`aten::cat=0`); interrupted cat probes restore their hook; decode-only AUTO overrides remain isolated; packed decode preserves raw score×V semantics across capacity-strided V layouts; threshold sweeps continue after individual failures while retaining aggregate failure; omitted decode thresholds are not invented or leaked; `--device auto` falls back to CPU, explicit `--device cpu` remains CPU even when CUDA is visible, and an unavailable explicit CUDA request skips cleanly | CPU operator contract, not GPU timing |

The retained profile-v20 baseline is `aten::copy_` 2/call for attention, 12/call for forward, and 394/call for generate, with `aten::cat=0` and `aten::contiguous=0`. The #403–#568 additions are CPU-safe contracts, diagnostics, and docs; #450/#465/#479/#495/#512/#525/#540 refine CUDA-build/device/runtime skip handoffs, exact build-flag handling, schema-v9 requested-config retention, stdout/JSON parity, and no-timing boundaries, while #552 and #566 harden explicit CPU extension selection. None adds CUDA timing, GPU throughput, sparse-kernel evidence, or cold CUDA–Triton validation. These are CPU call-count observations, not GPU performance claims.

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

## Refresh additions (#464–#478 and tip `6181ba6`)

| PR / tip | Branch / scope | What the matrix records | Result |
|---:|---|---|---|
| **#464** | `opt/amp-train-v20` / AMP | Explicit fp32 training remains full-forward, clears forward-only mode, and keeps GradScaler disabled | CPU contract only; no GPU throughput |
| **#465** | `opt/gpu-measure-v18` / GPU measurement | CUDA skip diagnostics classify build, visible-device, and runtime state without substituting CPU timing | CPU contract only; no GPU timing |
| **#466** | `opt/docs-v80` / docs | Previous matrix/backlog refresh through #463 | Docs-only; P0 unchanged |
| **#467** | `opt/sparse-probe-v20` / sparse probe | A passing density-only probe skips CPU crossover work while recording the successful guardrail | CPU contract only; no GPU claim |
| **#468** | `opt/blocked-tile-v20` / attention backward | Batched partial-tile strict-tril backward covers shared-value and head-matched-value layouts for blocked and online paths | CPU contract only |
| **#469** | `opt/auto-thr-v21` / AUTO | Runtime AUTO disable/re-enable transitions release and re-enter both cold and decode gates | CPU contract only |
| **#470** | `opt/online-decode-v21` / online decode | Shared-V multi-query decode preserves raw signed score×V results across CPU-safe dispatch and fallback paths | CPU contract only; no GPU claim |
| **#471** | `opt/scorev-v21` / score-V | Strided shared-V score×V preserves raw strict-tril outputs and Q/K/V gradients across dispatch aliases | CPU contract only; no GPU claim |
| **#472** | `opt/layout-v20` / sampling | Strided probability scratch stays within non-unit-vocabulary views across multinomial and full/overflow-clamped top-k paths | CPU contract only |
| **#473** | `opt/gen-bench-v19` / generate benchmark | Both AUTO threshold gates remain preserved when AUTO is toggled without overrides | CPU contract only |
| **#474** | `opt/rope-fuse-v20` / RoPE | Overlapping direct, view, and shifted output aliases are rejected before pair stores | CPU contract only; fused GPU validation open |
| **#475** | `opt/attn-bwd-v20` / attention backward | Batch/head-broadcast zero-stride V preserves strict-tril output and Q/K/V gradient parity across implementation selectors | CPU contract only; no GPU claim |
| **#476** | `opt/cuda-build-v19` / CUDA build | Missing-torch native-build setup remains a CPU-safe metadata no-op without compiler or CUDA requirements | CPU setup contract only; no GPU claim |
| **#477** | `opt/compile-v21` / compile | No-probe construction failure preserves the eager module, caller training mode, and existing gradients | CPU contract only; no GPU claim |
| **#478** | `opt/amp-train-v21` / AMP | Switching active bf16 forward-only AMP back to fp32 clears forward-only mode and keeps GradScaler disabled | CPU contract only; no GPU throughput |
| **tip `6181ba6`** | GPU measurement | CUDA-unavailable schema-v7 summaries retain the exact requested B/H/T/N/D/dtype/warmup/iters configuration | CPU contract only; no GPU timing |

## Refresh additions (#479–#494)

| PR / tip | Branch / scope | What the matrix records | Result |
|---:|---|---|---|
| **#479** | `opt/gpu-measure-v19` / GPU measurement | CUDA-unavailable summaries retain the exact requested shape, dtype, warmup, and iteration configuration | CPU skip contract only; no GPU timing |
| **#480** | `opt/prefetch-v20` / DataLoader | Zero-worker CPU loading does not install the worker-only initialization hook | CPU synchronous-loading contract only |
| **#481** | `opt/sparse-probe-v21` / sparse probe | A passing density-only probe with a custom floor still skips CPU crossover work | CPU guardrail only; no GPU claim |
| **#482** | `opt/blocked-tile-v21` / blocked prefill | Strict-tril raw-score prefill covers batched partial tiles with shared and head-matched value layouts | CPU contract only |
| **#483** | `opt/docs-v81` / docs | Previous matrix/backlog refresh through #478, rebased onto #482 | Docs-only; P0 unchanged |
| **#484** | `opt/auto-thr-v22` / AUTO | Cold/decode threshold boundaries and CPU blocked fallback remain explicit when Triton is unavailable | CPU dispatch contract only |
| **#485** | `opt/online-decode-v22` / online decode | Shared-value online decode remains deterministic across single and split past tiles | CPU raw score×V contract only; no GPU claim |
| **#486** | `opt/scorev-v22` / score-V | Feature-strided shared-V outputs and Q/K/V gradients remain stable across CPU-safe dispatch aliases | CPU contract only; no GPU claim |
| **#487** | `opt/rope-fuse-v21` / RoPE | Unpaired and paired paths reject `out=` buffers overlapping cosine/sine storage before pair stores | CPU alias contract only |
| **#488** | `opt/layout-v21` / sampling | Offset, padded probability scratch with unit vocabulary stride preserves full-vocabulary and top-k fallback neighbors | CPU contract only |
| **#489** | `opt/gen-bench-v20` / AUTO | A cold-only threshold override leaves the decode threshold unchanged and restores environment state | CPU contract only |
| **#490** | `opt/compile-v22` / compile | No-probe option coverage spans `fullgraph=False` and `fullgraph=True` while preserving caller state | CPU contract only; GPU path unmeasured |
| **#491** | `opt/cuda-build-v20` / CUDA build | A stale non-executable `CUDA_HOME/bin/nvcc` falls back to an executable `PATH/nvcc` | CPU setup contract only; no GPU claim |
| **#492** | `opt/attn-bwd-v21` / attention backward | Strict-tril backward covers head-broadcast upstream gradients across dispatch aliases | CPU contract only; no GPU claim |
| **#493** | `opt/amp-train-v22` / AMP | Failed backend construction leaves the complete prior AMP state intact | CPU contract only; no GPU throughput |
| **#494** | `opt/prefetch-v21` / DataLoader | Zero-worker DataLoader keeps `batch_size=None`, preserving each full host batch without outer re-collation | CPU contract only; H2D overlap unmeasured |

## Refresh additions (#495–#509)

| PR | Branch / scope | What the matrix records | Result |
|---:|---|---|---|
| **#495** | `opt/gpu-measure-v20` / GPU measurement | Run-level CUDA skips expose `timing_scope=none` and the classified CUDA runtime state in the CPU handoff | CPU diagnostics only; no GPU timing |
| **#496** | `opt/sparse-probe-v22` / sparse probe | A passing enforced density guardrail explicitly skips crossover work | CPU guardrail only; no GPU claim |
| **#497** | `opt/blocked-tile-v22` / blocked prefill | Dispatcher-selected blocked/online long-prefill paths cover shared and head-matched value layouts at partial tiles | CPU contract only |
| **#498** | `opt/docs-v82` / docs | Previous matrix/backlog refresh through #494 | Docs-only; P0 unchanged |
| **#499** | `opt/auto-thr-v23` / AUTO | Strict cold/decode threshold boundaries preserve Triton preference when the availability probe is true | CPU dispatch contract only; no GPU claim |
| **#500** | `opt/online-decode-v23` / online decode | Long T=1 per-head-V tiled decode preserves eager raw score×V parity and packed-cache views | CPU raw score×V contract only; no GPU claim |
| **#501** | `opt/scorev-v23` / score-V | Feature-strided per-head-V outputs and Q/K/V gradients match the eager strict-tril reference across public aliases | CPU contract only; no GPU claim |
| **#502** | `opt/layout-v22` / sampling | Offset, padded, non-unit-vocabulary scratch preserves full-vocabulary and top-k fallback neighbors | CPU contract only |
| **#503** | `opt/gen-bench-v21` / generate benchmark | An omitted decode threshold is neither invented nor leaked by a cold-only AUTO override | CPU contract only |
| **#504** | `opt/rope-fuse-v22` / RoPE | Public T>1 eager/fused dispatch rejects output aliases overlapping cosine or sine storage before any store | CPU alias contract only |
| **#505** | `opt/compile-v23` / compile | No-probe eval/train callers preserve compile options, caller mode, and gradients for both fullgraph settings | CPU contract only; GPU path unmeasured |
| **#506** | `opt/attn-bwd-v22` / attention backward | Shared Q views broadcasting across batch and heads reduce gradients to the underlying base under the raw strict-tril contract | CPU contract only; no GPU claim |
| **#507** | `opt/cuda-build-v21` / CUDA build | Non-`1` `BDH_BUILD_CUDA` values remain CPU-only setup requests, including when a GPU is visible | CPU setup contract only; no GPU claim |
| **#508** | `opt/prefetch-v22` / DataLoader | Worker construction preserves the full-batch iterable and requested split | CPU contract only; H2D overlap unmeasured |
| **#509** | `opt/amp-train-v23` / AMP | Failed autocast-context construction preserves every field of the prior AMP state | CPU contract only; no GPU throughput |

## Refresh additions (#510–#524)

| PR | Branch / scope | What the matrix records | Result |
|---:|---|---|---|
| **#510** | `opt/sparse-probe-v23` / sparse probe | The opted-in all-skip path completes without training or CPU crossover work | CPU guardrail only; no GPU claim |
| **#511** | `opt/docs-v83` / docs | Previous matrix/backlog refresh through #509 | Docs-only; P0 unchanged |
| **#512** | `opt/gpu-measure-v21` / GPU measurement | Run-level CUDA skips retain self-contained build/device diagnostics and `timing_scope=none` | CPU diagnostics only; no GPU timing |
| **#513** | `opt/blocked-tile-v23` / blocked prefill | Dispatcher-selected blocked/online partial long-prefill tiles cover shared and head-matched value layouts with raw-score output and Q/K/V gradient parity | CPU contract only; no GPU claim |
| **#514** | `opt/online-decode-v24` / online decode | Offset-packed K/V views support multi-query per-head-V dispatch across decode backends without mutating cache inputs | CPU raw score×V contract only; no GPU claim |
| **#515** | `opt/auto-thr-v24` / AUTO | The per-call `online` alias overrides a conflicting eager environment on cold and decode gates | CPU dispatch contract only; no GPU claim |
| **#516** | `opt/scorev-v24` / score-V | Capacity-strided per-head-V forward and Q/K/V gradients preserve raw score×strict-tril×V parity across dispatch aliases | CPU contract only; no GPU claim |
| **#517** | `opt/rope-fuse-v23` / RoPE | Public paired dispatch rejects output aliases of cached cis before any eager or fused store | CPU alias contract only; fused GPU validation open |
| **#518** | `opt/layout-v23` / sampling | Padded non-unit-vocabulary scratch combined with strided decode output preserves full/top-k/overflow paths and excluded neighbors | CPU contract only |
| **#519** | `opt/gen-bench-v22` / AUTO | Zero remains a valid strict cold/decode threshold and environment values restore cleanly | CPU dispatch contract only; no GPU claim |
| **#520** | `opt/compile-v24` / compile | Successful `train_bwd` compile probes clear probe gradients and restore caller training state | CPU contract only; GPU path unmeasured |
| **#521** | `opt/cuda-build-v22` / CUDA build | Empty, zero-padded, and whitespace `BDH_BUILD_CUDA` values remain CPU-only; only exact `1` selects CUDA setup | CPU setup contract only; no GPU claim |
| **#522** | `opt/prefetch-v23` / DataLoader | The asynchronous BatchPrefetcher queue remains bounded to one-slot lookahead | CPU contract only; H2D overlap unmeasured |
| **#523** | `opt/amp-train-v24` / AMP | Logits-only forward stays inside the AMP context and cross-entropy runs afterward in fp32 | CPU contract only; no GPU throughput |
| **#524** | `opt/attn-bwd-v23` / attention backward | Batch/head-broadcast upstream gradients preserve raw strict-tril parity across implementation selectors | CPU contract only; no GPU claim |

## Refresh additions (#525–#539)

| PR | Branch / scope | What the matrix records | Result |
|---:|---|---|---|
| **#525** | `opt/gpu-measure-v22` / GPU measurement | Run-level CUDA skip records include the availability probe in schema v9 while preserving CPU handoffs and raw strict-tril score×V semantics | CPU diagnostics only; no GPU timing |
| **#526** | `opt/sparse-probe-v24` / sparse probe | The default-off gate wins over guardrail enforcement without training or crossover work and reports its machine-readable exit | CPU guardrail only; no GPU claim |
| **#527** | `opt/docs-v84` / docs | Previous matrix/backlog refresh through #524 | Docs-only; P0 unchanged |
| **#528** | `opt/blocked-tile-v24` / blocked prefill | Dispatcher-selected blocked/online long-prefill paths cover padded non-contiguous V views, raw strict-tril parity, storage gradients, and untouched padding | CPU contract only; no GPU claim |
| **#529** | `opt/auto-thr-v25` / AUTO | Strict zero-threshold equality and transitions remain covered for cold and decode gates with unavailable-Triton fallback | CPU dispatch contract only; no GPU claim |
| **#530** | `opt/online-decode-v25` / online decode | Offset-packed K/V multi-query decode preserves float64 per-head-V dtype across CPU-safe dispatch | CPU raw score×V contract only; no GPU claim |
| **#531** | `opt/scorev-v25` / score-V | Capacity-strided shared and per-head V backing-storage gradients preserve raw score × strict-tril parity across public aliases | CPU contract only; no GPU claim |
| **#532** | `opt/layout-v24` / sampling | Narrow top-k reuses padded non-unit-stride probability storage without changing excluded neighbors or contiguous-reference parity | CPU contract only |
| **#533** | `opt/rope-fuse-v24` / RoPE | CPU-safe fused T>1 backward covers pair-specific upstream gradients against eager parity | CPU contract only; fused GPU validation open |
| **#534** | `opt/gen-bench-v23` / generate benchmark | `--device auto` falls back to CPU without CUDA; an explicit unavailable CUDA request skips without GPU work | CPU device-resolution contract only |
| **#535** | `opt/compile-v25` / compile | Successful `train_bwd` probes clean up pre-existing gradients while preserving caller state | CPU contract only; GPU path unmeasured |
| **#536** | `opt/prefetch-v24` / DataLoader | Closing the async prefetcher stops production and clears its one-slot queue | CPU contract only; H2D overlap unmeasured |
| **#537** | `opt/attn-bwd-v24` / attention backward | CPU-safe strict-tril backward covers both sides of the default 64-row tile boundary | CPU contract only; no GPU claim |
| **#538** | `opt/cuda-build-v23` / CUDA build | Non-exact `BDH_FORCE_CPU_EXT` values do not force CPU-only setup during an explicit CUDA request; only exact `1` does | CPU setup contract only; no GPU claim |
| **#539** | `opt/amp-train-v25` / AMP | Default full-forward training passes targets while keeping the full forward inside the AMP context | CPU contract only; no GPU throughput |

## Refresh additions (#540–#553)

| PR | Branch / scope | What the matrix records | Result |
|---:|---|---|---|
| **#540** | `opt/gpu-measure-v23` / GPU measurement | Decode CUDA skips keep stdout and JSON summaries identical, retain the requested configuration, and expose `timing_scope=none` without results or timing text | CPU diagnostics only; no GPU timing |
| **#541** | `opt/docs-v85` / docs | Previous matrix/backlog refresh through #539 | Docs-only; P0 unchanged |
| **#542** | `opt/sparse-probe-v25` / sparse probe | `--density-only --skip-train` suppresses density and crossover work while reporting successful probe completion | CPU guardrail only; no GPU claim |
| **#543** | `opt/blocked-tile-v25` / blocked prefill | Dispatcher-selected blocked/online paths preserve padded non-contiguous Q/K storage gradients and strict-tril row-zero/padding behavior | CPU contract only; no GPU claim |
| **#544** | `opt/auto-thr-v26` / AUTO | Non-integer and negative cold-threshold overrides fail with the documented error while leaving the independent decode gate usable | CPU dispatch contract only; no GPU claim |
| **#545** | `opt/online-decode-v26` / online decode | Per-head-V tiled decode remains invariant across block sizes at and across tile boundaries under raw score×V semantics | CPU raw score×V contract only; no GPU claim |
| **#546** | `opt/scorev-v26` / score-V | Capacity-strided Q/K views accumulate backing-storage gradients matching eager across blocked/online/Triton/CUDA aliases | CPU contract only; no GPU claim |
| **#547** | `opt/rope-v25` / RoPE | CPU-safe blocked and Triton fallback entries preserve fused backward parity for arbitrary pair-specific upstream gradients | CPU contract only; fused GPU validation open |
| **#548** | `opt/layout-v25` / sampling | Padded non-unit probability scratch and zero-stride singleton output preserve selected indices and isolate padding/neighbors | CPU contract only |
| **#549** | `opt/gen-bench-v24` / generate benchmark | Explicit `--device cpu` stays CPU even when CUDA is reported available | CPU device-resolution contract only |
| **#550** | `opt/compile-v26` / compile | A successful `train_bwd` probe restores the caller training mode on the returned wrapper as well as the original module | CPU contract only; GPU path unmeasured |
| **#551** | `opt/attn-bwd-v25` / attention backward | Strict-past backward covers queries in the first and second 64-row tiles, including the diagonal/future exclusion boundary | CPU contract only; no GPU claim |
| **#552** | `opt/cuda-build-v24` / CUDA build | Exact `BDH_FORCE_CPU_EXT=1` selects CPU-only extension setup even when torch reports CUDA available | CPU setup contract only; no GPU claim |
| **#553** | `opt/prefetch-v25` / DataLoader | A terminal host-gather failure is surfaced promptly and repeatedly through `next()` with its cause preserved | CPU contract only; H2D overlap unmeasured |

## Refresh additions (#559–#568)

| PR | Branch / scope | What the matrix records | Result |
|---:|---|---|---|
| **#559** | `opt/auto-thr-v27` / AUTO | Removing an invalid cold-threshold override restores the live shared threshold while preserving strict equality and above-threshold decode behavior | CPU dispatch contract only; no GPU claim |
| **#560** | `opt/online-decode-v27` / online decode | Direct empty-past decode returns typed zero outputs for shared-V and per-head-V layouts across block sizes | CPU raw score×V contract only; no GPU claim |
| **#561** | `opt/scorev-v27` / score-V | Capacity-strided Q/K backing-storage gradients remain correct for shared-V and per-head-V layouts across public dispatch aliases | CPU contract only; no GPU claim |
| **#562** | `opt/rope-v26` / RoPE | Non-contiguous pair-strided inputs and output cache slots preserve parity through PyTorch, blocked, and CPU Triton-fallback paths | CPU contract only; fused GPU validation open |
| **#563** | `opt/layout-v26` / sampling | Padded narrow top-k probability storage remains isolated, including the full scratch-buffer neighbor check | CPU contract only |
| **#564** | `opt/gen-bench-v25` / generate benchmark | A CPU Triton request reports an explicit blocked fallback rather than implying a CUDA Triton kernel | CPU device-resolution contract only; no GPU claim |
| **#565** | `opt/compile-v27` / compile | Input-only forward probes succeed without targets and restore the caller training state | CPU contract only; GPU path unmeasured |
| **#566** | `opt/cuda-build-v25` / CUDA build | Exact force-CPU selection wins over detected CUDA and a valid `nvcc` during extension setup | CPU setup contract only; no GPU claim |
| **#567** | `opt/attn-bwd-v26` / attention backward | Multiple upstream query rows spanning tile boundaries accumulate only strict-past gradients across implementation selectors | CPU contract only; no GPU claim |
| **#568** | `opt/prefetch-v26` / DataLoader | Terminal producer failure leaves no live thread or queued resources, and shutdown remains idempotent | CPU contract only; H2D overlap unmeasured |

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
