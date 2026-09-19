# OPT backlog — ranked remaining work (docs-v72)

Private sandbox: `katulevskiy/bdh-gpu-opt`.

Current tip: `8a5a94b` (#349). The requested #343–#348 refresh is included below; #349 is also recorded because it landed before this branch was written.

## Hard constraints

- Attention is raw scores × strict lower-triangular `tril(diagonal=-1)`.
- No softmax, `1/√d`, or `F.scaled_dot_product_attention`.
- CPU-only evidence never becomes a GPU speedup or throughput claim.
- Production defaults remain eager/fp32/AUTO-off/sparse-off until hardware evidence changes them.

## Ranked work

| Priority | Work | Current state | Acceptance evidence |
|---|---|---|---|
| **P0** | Real GPU cold attention | Blocked by absent live CUDA hardware | A100/H100 cold eager vs blocked/online/Triton/CUDA medians plus strict-tril parity |
| **P0** | Cold CUDA–Triton validation | Scaffolds and CPU fallbacks are present; no launch has been measured here | Successful build/launch, cold compile behavior, output parity, and median timing |
| **P0** | T=1 decode on GPU | Packed K/V and shared/per-head score×V CPU contracts are covered | Real-GPU decode medians, peak memory, and parity at tile/oneshot boundaries |
| **P1** | Long-T AUTO threshold/tile retune | CPU gates and independent cold threshold are covered; AUTO remains opt-in | GPU cold/decode sweep that justifies threshold and tile choices |
| **P1** | Fused RoPE on GPU | CPU paired/layout contracts and skip gates are covered | Real-GPU fused/eager parity and timing across supported dtypes/layouts |
| **P1** | AMP train and H2D overlap | CPU configuration/identity contracts are covered | CUDA bf16/fp16 train throughput and pinned H2D overlap measurements |
| **P1** | Compile train step | CPU eager guidance and fallback contracts are covered | GPU inductor/CUDA-graph A/B with `BDH_COMPILE=0|1` |
| **P2** | Analytic attention backward on GPU | CPU blocked/online tiled backward and gradient contracts are covered | GPU train-step parity and timing across eager/blocked/online/Triton/CUDA |
| **P2** | Sparse production path | Probe is explicit and default-off; CPU density is not a kernel result | Density plus a measured GPU sparse-kernel win before any default change |

## Latest landed contracts

| PR | Scope | Recorded outcome |
|---:|---|---|
| **#343** | Sparse truthy gate | Supported truthy tokens normalize correctly; density-only CPU probe remains opt-in and crossover work stays skipped |
| **#344** | docs-v71 | Matrix/backlog refreshed through #342; no GPU evidence added |
| **#345** | Blocked prefill | Capacity-padded non-contiguous shared/head-matched V views preserve raw strict-tril parity on CPU |
| **#346** | AUTO threshold | Cold-only malformed overrides do not corrupt the shared decode gate; valid updates recover at strict boundaries |
| **#347** | Online decode | Non-zero-offset packed K/V views support multi-query CPU parity without mutating inputs |
| **#348** | Score×V | Long-path shared-V outputs and Q/K/V gradients match eager strict-tril raw score×V on CPU |
| **#349** | Sampler layout | Narrow top-k strided output and neighboring backing values are protected; current tip |

## Current evidence summary

The profile-v20 call-count baseline remains attention `copy_` 2/call, forward `copy_` 12/call, generate `copy_` 394/call, `cat=0`, and `contiguous=0`. The #343–#349 additions are CPU-safe tests/docs/layout contracts; none adds CUDA timing, GPU throughput, sparse-kernel evidence, or cold CUDA–Triton validation.

Defaults remain:

- `BDH_ATTN_IMPL=eager`
- `BDH_ATTN_AUTO` unset/off, decode threshold 512, optional independent cold threshold
- `BDH_ROPE_IMPL=eager`
- `BDH_COMPILE=0` with CPU compile guidance limited to eager/default mode
- `BDH_AMP_DTYPE=float32`
- Sparse ReLU OFF

## GPU measurement checklist

On a real CUDA box:

```bash
python benchmarks/bench_gpu_attn.py
python benchmarks/bench_gpu_attn.py --mode decode --T 512
python benchmarks/bench_gpu_attn.py --dtype bfloat16
python benchmarks/bench_gpu_attn.py --dtype float16
```

Record GPU name, torch/CUDA versions, cold and decode medians for eager/blocked/online/Triton/CUDA, peak memory, `bit_identical`, and `max|Δ|` versus eager. A CPU run or unavailable-CUDA skip is not a P0 result.

For the remaining train gate, compare `BDH_COMPILE=0|1` and AMP modes on a real CUDA device, then measure H2D overlap and analytic backward. Do not change defaults from CPU evidence alone.

## Explicit non-goals

- Softmax, diagonal inclusion, or SDPA substitution.
- GPU claims from CPU profiler absolute times, CPU medians, density, or skip output.
- Defaulting blocked/online/triton on CPU.
- Enabling sparse ReLU in production without density and GPU-kernel evidence.
