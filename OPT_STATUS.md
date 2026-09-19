# OPT status — landed work (#1–#376; docs-v74)

Private sandbox: `katulevskiy/bdh-gpu-opt`.

Documentation coverage: #350–#376. This branch is based on current main tip `5c32568` (#378); #377 and #378 landed after the requested refresh range and remain outside this documentation scope. The documented landing tip is `f2a8203` (#376).

## Evidence boundary

Attention remains **raw scores × strict lower-triangular** `tril(diagonal=-1)`: no softmax, no `1/√d`, and no `F.scaled_dot_product_attention`.

This sandbox is CPU-only (`torch 2.14.0+cu130`, `cuda=False`). CPU tests establish correctness, layout, dispatch, and control-flow contracts only. They do not establish GPU speedups, GPU throughput, CUDA correctness, or Triton timing.

**P0 remains real GPU measurement plus cold CUDA–Triton validation.** Keep production defaults unchanged until that evidence exists.

## Current matrix

| Area | Current contract | Evidence boundary |
|---|---|---|
| Device | CPU-only; CUDA/Triton paths skip or fall back cleanly here | No GPU claim |
| Attention | `eager` default; `blocked`, `online`, `triton`, and `cuda` opt-in; strict-past backward and padded/strided V contracts covered on CPU | Raw strict-tril parity on CPU; GPU measure open |
| AUTO | Off by default; decode threshold remains independent from the optional cold threshold; blank/whitespace fallback and decode-only overrides preserve the gates | CPU dispatch/parity only |
| RoPE | `eager` default; fused path opt-in; paired output-slot aliasing is rejected safely | CPU shape/parity only; fused GPU validation open |
| Compile | Off by default; invalid and normalized `BDH_COMPILE_PROBE` values fail closed to `train_bwd` without changing defaults | CUDA graphs/inductor unmeasured |
| AMP | fp32/off by default; bf16/fp16 CPU contexts are contract-covered | CPU smoke only; GPU train throughput open |
| Sparse ReLU | Off by default; explicit probe guardrails distinguish no-sample and density failure exits and skip crossover work | CPU density/control flow only; no sparse-kernel result |
| DataLoader | Repeated worker-backed batches preserve CPU tensor identity without CUDA setup | CPU identity contract only; H2D overlap unmeasured |
| Sampling | Strided singleton output views preserve RNG parity, identity, and neighbor isolation | CPU contract only |
| Packed generate | Cache path remains cat-free (`aten::cat=0`) | CPU operator contract, not GPU timing |

The retained profile-v20 baseline is `aten::copy_` 2/call for attention, 12/call for forward, and 394/call for generate, with `aten::cat=0` and `aten::contiguous=0`. These are CPU call-count observations, not GPU performance claims.

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
