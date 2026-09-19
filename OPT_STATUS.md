# OPT status — landed work (#1–#349; docs-v72)

Private sandbox: `katulevskiy/bdh-gpu-opt`.

Tip: `8a5a94b` (#349, narrow top-k sampler-output layout) is the current tip. This refresh carries the requested #343–#348 matrix forward and includes #349 because it landed while the docs branch was being prepared.

## Evidence boundary

Attention remains **raw scores × strict lower-triangular** `tril(diagonal=-1)`: no softmax, no `1/√d`, and no `F.scaled_dot_product_attention`.

This sandbox is CPU-only (`torch 2.14.0+cu130`, `cuda=False`). CPU tests establish correctness, layout, dispatch, and control-flow contracts only. They do not establish GPU speedups, GPU throughput, CUDA correctness, or Triton timing.

**P0 remains real GPU measurement plus cold CUDA–Triton validation.** Keep production defaults unchanged until that evidence exists.

## Current matrix

| Area | Current contract | Evidence boundary |
|---|---|---|
| Device | CPU-only; CUDA/Triton paths skip or fall back cleanly here | No GPU claim |
| Attention | `eager` default; `blocked`, `online`, `triton`, and `cuda` opt-in | Raw strict-tril parity on CPU; GPU measure open |
| AUTO | Off by default; long decode gate uses `past_len > 512`; cold gate may be independent | CPU dispatch/parity only |
| RoPE | `eager` default; fused path opt-in | CPU parity only; fused GPU validation open |
| Compile | Off by default; CPU guidance is eager-only with default mode | CUDA graphs/inductor unmeasured |
| AMP | fp32/off by default; bf16/fp16 opt-in | CPU smoke only; GPU train throughput open |
| Sparse ReLU | Off by default; `BDH_SPARSE_PROBE=1` is explicit | CPU density only; no sparse-kernel result |
| Packed generate | Cache path remains cat-free (`aten::cat=0`) | CPU operator contract, not GPU timing |

The retained profile-v20 baseline is `aten::copy_` 2/call for attention, 12/call for forward, and 394/call for generate, with `aten::cat=0` and `aten::contiguous=0`. These are CPU call-count observations, not GPU performance claims.

## Landed since docs-v71

| PR | Branch / scope | What the matrix records | Result |
|---:|---|---|---|
| **#343** | `opt/sparse-v12` / sparse probe | Covers every supported truthy `BDH_SPARSE_PROBE` token with case/whitespace normalization; density-only work stays CPU-safe and crossover work remains skipped | Sparse remains opt-in/default-off; no GPU claim |
| **#344** | `opt/docs-v71` / docs | Refreshed `OPT_STATUS.md` and `OPT_BACKLOG.md` through #342 | Docs-only; P0 unchanged |
| **#345** | `opt/blocked-v12` / blocked prefill | Covers capacity-padded, non-contiguous V views for shared and head-matched layouts with raw strict-tril parity | CPU contract only |
| **#346** | `opt/auto-thr-v12` / AUTO | Keeps invalid cold-only threshold overrides isolated from the shared decode gate and verifies recovery plus strict boundaries | CPU contract only |
| **#347** | `opt/online-v13` / online decode | Covers multi-query reads from non-zero-offset packed K/V views and preserves the input views | CPU parity only; no GPU timing |
| **#348** | `opt/scorev-v13` / score×V | Covers long-path shared-V output and Q/K/V gradient parity for blocked and online paths against eager strict-tril raw score×V | CPU contract only |
| **#349** | `opt/layout-v12` / sampler layout | Adds the narrow top-k (`k=8`) strided-output branch and protects neighboring backing values | CPU contract only; current tip |

## Defaults and operator guidance

- `BDH_ATTN_IMPL=eager` remains the default reference path.
- `BDH_ATTN_AUTO` remains unset/off; `BDH_ATTN_AUTO_THRESHOLD=512` is the retained decode threshold, with an optional independent cold threshold.
- `BDH_ATTN_AUTOGRAD` remains opt-in. Blocked/online paths use tiled analytic backward only when explicitly enabled.
- `BDH_ROPE_IMPL=eager`, `BDH_COMPILE=0`, `BDH_AMP_DTYPE=float32`, and sparse ReLU OFF remain unchanged.
- Do not turn CPU medians, profiler times, density, skip results, or parity tests into GPU performance claims.

## Next measurement gate

On a real CUDA box, run the GPU attention harness for eager/blocked/online/Triton/CUDA on cold and T=1 decode shapes, then record GPU name, torch/CUDA versions, medians, and parity deltas. Keep the default eager path until those results and cold CUDA–Triton validation are available.

See `OPT_BACKLOG.md` for the ranked P0/P1/P2 work and `OPT_NOTES.md` for detailed historical measurements.