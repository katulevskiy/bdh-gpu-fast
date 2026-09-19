# OPT backlog — ranked remaining work (docs-v73)

Private sandbox: `katulevskiy/bdh-gpu-opt`.

Documentation coverage: #350–#363, with documented tip `8cc215c` (#363). This branch is based on current main tip `a59b542` (#364); #364 is outside this refresh scope.

## Hard constraints

- Attention is raw scores × strict lower-triangular `tril(diagonal=-1)`.
- No softmax, `1/√d`, or `F.scaled_dot_product_attention`.
- CPU-only evidence never becomes a GPU speedup or throughput claim.
- Production defaults remain eager/fp32/AUTO-off/sparse-off until hardware evidence changes them.

## Ranked work

| Priority | Work | Current state | Acceptance evidence |
|---|---|---|---|
| **P0** | Real GPU cold attention | CPU reference and benchmark contracts are covered; no live CUDA measurement | A100/H100 cold eager vs blocked/online/Triton/CUDA medians plus strict-tril parity |
| **P0** | Cold CUDA–Triton validation | Scaffolds and CPU fallbacks are present; no launch has been measured here | Successful build/launch, cold compile behavior, output parity, and median timing |
| **P0** | T=1 decode on GPU | CPU score×V, packed-view, and empty-past contracts are covered by #362/#363 | Real-GPU decode medians, peak memory, and parity at tile/oneshot boundaries |
| **P1** | Long-T AUTO threshold/tile retune | CPU cold-threshold sweep and blank/whitespace fallback are covered; AUTO remains opt-in | GPU cold/decode sweep that justifies threshold and tile choices |
| **P1** | Fused RoPE on GPU | CPU shape/layout contracts are covered by #351 | Real-GPU fused/eager parity and timing across supported dtypes/layouts |
| **P1** | AMP train and H2D overlap | CPU AMP throughput and worker-backed H2D identity contracts are covered; no CUDA throughput evidence | CUDA bf16/fp16 train throughput and pinned H2D overlap measurements |
| **P1** | Compile train step | CPU invalid-probe fallback and force-CPU build guard are covered; GPU path unmeasured | GPU inductor/CUDA-graph A/B with `BDH_COMPILE=0|1` |
| **P2** | Analytic attention backward on GPU | CPU strict-past tile-boundary and gradient contracts are covered; GPU path unmeasured | GPU train-step parity and timing across eager/blocked/online/Triton/CUDA |
| **P2** | Sparse production path | No-sample guardrail is covered; probe is explicit and default-off | Density plus a measured GPU sparse-kernel win before any default change |

## Latest landed contracts

| PR | Scope | Recorded outcome |
|---:|---|---|
| **#350** | Docs-v72 | Matrix/backlog refreshed through #349; no GPU evidence added |
| **#351** | RoPE shape contracts | Unpaired/paired malformed-shape cases are rejected safely on CPU |
| **#352** | AUTO cold threshold | Explicit cold-threshold sweeps do not alter the independent decode gate |
| **#353** | Compile probe fallback | Invalid probe values fail closed to `train_bwd` without changing defaults |
| **#354** | CUDA build guard | Force-CPU mode remains opt-in-only and does not invoke native setup by itself |
| **#355** | Attention backward tile boundary | Strict-past raw score×V gradients match the eager reference at the 64-row boundary |
| **#356** | GPU attention measurement contract | CPU reference locks raw score×strict-tril×V semantics; no GPU timing was added |
| **#357** | DataLoader worker identity | CPU worker-backed batches preserve tensor identity without CUDA setup |
| **#358** | Sparse guardrail | No-sample exit is distinct and skips crossover work |
| **#359** | AMP train contract | CPU execution cannot be reported as GPU throughput evidence |
| **#360** | AUTO blank threshold | Blank/whitespace cold-threshold fallback preserves strict decode boundaries |
| **#361** | Blocked prefill low precision | CPU float16/bfloat16 parity and input-dtype preservation hold across the partial tile |
| **#362** | Decode score×V | CPU backend contracts preserve raw QK-transpose-times-V arithmetic and past-only semantics |
| **#363** | Online decode empty past | Empty past returns exact zero score×V output with expected shape and dtype |

## Current evidence summary

The profile-v20 call-count baseline remains attention `copy_` 2/call, forward `copy_` 12/call, generate `copy_` 394/call, `cat=0`, and `contiguous=0`. The #350–#363 additions are CPU-safe docs/tests/contracts; #356 names the GPU benchmark reference but adds no CUDA timing. None adds GPU throughput, sparse-kernel evidence, or cold CUDA–Triton validation.

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
