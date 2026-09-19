# OPT backlog — ranked remaining work (docs-v74)

Private sandbox: `katulevskiy/bdh-gpu-opt`.

Documentation coverage: #350–#376, with documented landing tip `f2a8203` (#376). The branch is based on current main tip `5c32568` (#378); #377 and #378 remain outside this requested refresh range.

## Hard constraints

- Attention is raw scores × strict lower-triangular `tril(diagonal=-1)`.
- No softmax, `1/√d`, or `F.scaled_dot_product_attention`.
- CPU-only evidence never becomes a GPU speedup or throughput claim.
- Production defaults remain eager/fp32/AUTO-off/sparse-off until hardware evidence changes them.

## Ranked work

| Priority | Work | Current state | Acceptance evidence |
|---|---|---|---|
| **P0** | Real GPU cold attention | CPU reference, padded/strided-V parity, and benchmark contracts are covered; no live CUDA measurement | A100/H100 cold eager vs blocked/online/Triton/CUDA medians plus strict-tril parity |
| **P0** | Cold CUDA–Triton validation | Scaffolds and CPU fallbacks are present; no launch has been measured here | Successful build/launch, cold compile behavior, output parity, and median timing |
| **P0** | T=1 decode on GPU | CPU score×V, packed-past, and empty-past contracts are covered by #362/#363/#373 | Real-GPU decode medians, peak memory, and parity at tile/oneshot boundaries |
| **P1** | Long-T AUTO threshold/tile retune | CPU cold/decode threshold independence, blank fallback, and decode-only overrides are covered by #352/#360/#368; AUTO remains opt-in | GPU cold/decode sweep that justifies threshold and tile choices |
| **P1** | Fused RoPE on GPU | CPU shape/layout and paired-output alias contracts are covered by #351/#366 | Real-GPU fused/eager parity and timing across supported dtypes/layouts |
| **P1** | AMP train and H2D overlap | CPU bf16/fp16 throughput contracts and repeated worker identity are covered by #359/#372/#374; no CUDA throughput evidence | CUDA bf16/fp16 train throughput and pinned H2D overlap measurements |
| **P1** | Compile train step | CPU invalid/normalized-probe fallback and force-CPU build guards are covered by #353/#354/#369/#370; GPU path unmeasured | GPU inductor/CUDA-graph A/B with `BDH_COMPILE=0|1` |
| **P2** | Analytic attention backward on GPU | CPU strict-past tile-boundary, strided-V, and empty-prefix gradient contracts are covered by #355/#364/#371; GPU path unmeasured | GPU train-step parity and timing across eager/blocked/online/Triton/CUDA |
| **P2** | Sparse production path | No-sample and density-failure guardrails are covered by #358/#375; probe is explicit and default-off | Density plus a measured GPU sparse-kernel win before any default change |

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
| **#364** | Strided shared-V score-V | Blocked/online strict-past parity holds for a capacity-padded non-contiguous shared-V view |
| **#365** | Docs-v73 | Matrix/backlog refreshed through #363; no GPU evidence added |
| **#366** | Paired RoPE output alias | Aliased paired output storage is rejected safely on CPU |
| **#367** | Sampler output stride | Strided singleton outputs preserve RNG parity, identity, and neighbor isolation |
| **#368** | AUTO decode-only threshold | Decode-only overrides leave the cold threshold independent and restore it unchanged |
| **#369** | CUDA build precedence | Explicit CPU mode wins over discoverable `nvcc` and a CUDA request |
| **#370** | Compile probe normalization | Case/whitespace-normalized names preserve clean `train_bwd` fallback behavior |
| **#371** | Empty-prefix backward | Empty strict-past rows produce zero output and zero query gradient |
| **#372** | DataLoader worker identity | Repeated prefetched batches remain CPU identity paths without CUDA setup |
| **#373** | GPU decode measurement contract | Packed-past decode remains raw score×V without scaling or softmax |
| **#374** | AMP train contract | Active CPU bf16/fp16 contexts never report CUDA throughput |
| **#375** | Sparse guardrail failure | Density failure is distinct and skips CPU crossover work |
| **#376** | Blocked prefill padded V | Low-precision non-contiguous V preserves dtype, masking, and raw strict-tril parity |

## Current evidence summary

The profile-v20 call-count baseline remains attention `copy_` 2/call, forward `copy_` 12/call, generate `copy_` 394/call, `cat=0`, and `contiguous=0`. The #350–#376 additions are CPU-safe docs/tests/contracts; #356 and #373 define GPU benchmark references but add no CUDA timing. None adds GPU throughput, sparse-kernel evidence, or cold CUDA–Triton validation.

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
