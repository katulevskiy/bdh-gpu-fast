# OPT backlog — ranked remaining work (docs-v76)

Private sandbox: `katulevskiy/bdh-gpu-opt`.

Documentation coverage: #389–#402, with documented landing tip `9ae8df5` (#402).

## Hard constraints

- Attention is raw scores × strict lower-triangular `tril(diagonal=-1)`.
- No softmax, `1/√d`, or `F.scaled_dot_product_attention`.
- CPU-only evidence never becomes a GPU speedup or throughput claim.
- Production defaults remain eager/fp32/AUTO-off/sparse-off until hardware evidence changes them.

## Ranked work

| Priority | Work | Current state | Acceptance evidence |
|---|---|---|---|
| **P0** | Real GPU cold attention | CPU reference, padded/strided-Q/K/V parity, strided-gradient and packed-decode contracts, and benchmark contracts are covered; no live CUDA measurement | A100/H100 cold eager vs blocked/online/Triton/CUDA medians plus strict-tril parity |
| **P0** | Cold CUDA–Triton validation | Scaffolds and CPU fallbacks are present; no launch has been measured here | Successful build/launch, cold compile behavior, output parity, and median timing |
| **P0** | T=1 decode on GPU | CPU score×V, packed-past, empty-past including float64, and capacity-strided-V contracts are covered by #362/#363/#378/#388/#389/#395 | Real-GPU decode medians, peak memory, and parity at tile/oneshot boundaries |
| **P1** | Long-T AUTO threshold/tile retune | CPU cold/decode threshold independence, blank transitions, decode-only overrides, and pre-run sweep validation are covered by #352/#360/#368/#377/#394/#399; AUTO remains opt-in | GPU cold/decode sweep that justifies threshold and tile choices |
| **P1** | Fused RoPE on GPU | CPU shape/layout, paired-output alias, and fp16-input/fp32-cache T=1 contracts are covered by #351/#366/#383/#398 | Real-GPU fused/eager parity and timing across supported dtypes/layouts |
| **P1** | AMP train and H2D overlap | CPU bf16/fp16 throughput contracts, no-claim gating, disabled CPU GradScaler, and repeated worker/H2D opt-out identity are covered by #359/#372/#374/#387/#390; no CUDA throughput evidence | CUDA bf16/fp16 train throughput and pinned H2D overlap measurements |
| **P1** | Compile train step | CPU invalid/normalized-probe cleanup, failed-probe restoration, and force-CPU/nvcc build guards are covered by #353/#354/#369/#370/#384/#385/#400/#401; GPU path unmeasured | GPU inductor/CUDA-graph A/B with `BDH_COMPILE=0|1` |
| **P2** | Analytic attention backward on GPU | CPU strict-past tile-boundary, strided-V, empty-prefix, non-contiguous-Q/K/V, padded-V, and strided-upstream-gradient contracts are covered by #355/#364/#371/#386/#393/#402; GPU path unmeasured | GPU train-step parity and timing across eager/blocked/online/Triton/CUDA |
| **P2** | Sparse production path | No-sample and terminal density-floor failure guardrails are covered by #358/#375/#392; probe is explicit and default-off | Density plus a measured GPU sparse-kernel win before any default change |

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
| **#377** | AUTO cold-threshold recovery | Clearing a parsed blank override resumes the live decode gate at strict boundaries |
| **#378** | Online decode dtype | Float64 tiled decode preserves query dtype and eager raw score×V semantics |
| **#379** | Docs-v74 | Matrix/backlog refreshed through #376; no GPU evidence added |
| **#380** | Capacity-strided score-V | Blocked/online outputs match eager strict-tril score×V math for strided Q/K/V views |
| **#381** | Zero-stride sampler output | Multinomial/top-k paths preserve identity, RNG parity, and neighbor isolation |
| **#382** | Generate cat-hook cleanup | Interrupted cat probes restore the temporary `torch.cat` hook |
| **#383** | Paired RoPE mixed dtype | CPU paired T=1 fp16-input/fp32-cache output parity remains covered |
| **#384** | Compile probe fallback | Unsupported names preserve the CPU-safe `train_bwd` cleanup contract |
| **#385** | CUDA nvcc directory contract | nvcc directory discovery is covered without changing force-CPU opt-in behavior |
| **#386** | Strided attention backward | Analytic strict-tril backward parity holds for non-contiguous Q/K/V views |
| **#387** | DataLoader H2D opt-out | CPU H2D opt-out remains explicit and clean under the deeper contract |
| **#388** | Packed decode GEMM | Shared and per-head capacity-strided V layouts preserve raw score×V semantics across CPU-safe dispatch |
| **#389** | Packed decode measurement contract | Packed decode layouts are included in the GPU-measure reference contract; no timing was added |
| **#390** | AMP throughput claim contract | CPU AMP cannot report GPU throughput and keeps GradScaler disabled |
| **#391** | Docs-v75 | Matrix/backlog refreshed through #388; no GPU evidence added |
| **#392** | Sparse guardrail floor | Terminal `xy` density-floor failure is distinct and skips crossover work |
| **#393** | Padded-V backward | Capacity-padded non-contiguous V preserves blocked/online forward and gradient parity |
| **#394** | AUTO cold-threshold transition | Blank-to-explicit transitions keep the later cold override independent |
| **#395** | Online decode empty dtype | Float64 empty-past decode returns typed, shaped zero score×V output |
| **#396** | Score-V dispatch | Blocked/online paths retain raw score×strict-tril semantics including row zero |
| **#397** | Strided sampler layout | Strided prefill logits slices preserve singleton output identity, RNG parity, and neighbor isolation |
| **#398** | Paired RoPE mixed dtype | Non-contiguous paired T=1 output slots preserve fp16-input/fp32-cache parity |
| **#399** | AUTO sweep validation | Malformed threshold sweeps fail closed before execution or model setup |
| **#400** | Compile probe failure | Failed probes restore mode and clear partial gradients before eager fallback |
| **#401** | CUDA nvcc directory contract | A `CUDA_PATH/nvcc` directory is rejected cleanly while CPU fallback remains safe |
| **#402** | Strided upstream attention gradients | Strict-tril backward covers non-contiguous upstream gradients across CPU-safe dispatch paths |

## Current evidence summary

The profile-v20 call-count baseline remains attention `copy_` 2/call, forward `copy_` 12/call, generate `copy_` 394/call, `cat=0`, and `contiguous=0`. The #350–#402 additions are CPU-safe docs/tests/contracts; #356, #373, and #389 define GPU benchmark references but add no CUDA timing. None adds GPU throughput, sparse-kernel evidence, or cold CUDA–Triton validation.

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
