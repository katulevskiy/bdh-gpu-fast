# OPT backlog — ranked remaining work (docs-v80)

Private sandbox: `katulevskiy/bdh-gpu-opt`.

Documentation coverage: #418–#463 plus documented tip `d9703b0` (#434), with prior coverage retained. This branch is based on current main tip `6d29504` (#465); #464–#465 remain outside this refresh scope.

## Hard constraints

- Attention is raw scores × strict lower-triangular `tril(diagonal=-1)`.
- No softmax, `1/√d`, or `F.scaled_dot_product_attention`.
- CPU-only evidence never becomes a GPU speedup or throughput claim.
- Production defaults remain eager/fp32/AUTO-off/sparse-off until hardware evidence changes them.

## Ranked work

| Priority | Work | Current state | Acceptance evidence |
|---|---|---|---|
| **P0** | Real GPU cold attention | CPU reference, padded/strided/zero-stride-Q/K/V parity, batch/head-broadcast zero-stride-Q/K gradient reduction, strided-gradient, empty-sequence, packed-decode, and benchmark contracts are covered; no live CUDA measurement | A100/H100 cold eager vs blocked/online/Triton/CUDA medians plus strict-tril parity |
| **P0** | Cold CUDA–Triton validation | Scaffolds, executable `CUDA_PATH/bin/nvcc` and `PATH/nvcc` discovery, requested-mode skip diagnostics, and CPU fallbacks are covered; no launch has been measured here | Successful build/launch, cold compile behavior, output parity, and median timing |
| **P0** | T=1 decode on GPU | CPU score×V, packed-past views, empty-past including float64, capacity-strided-V, float64 packed-cache, multi-query/per-head-V views, T=1 offset-packed per-head-V dispatch, and public dispatch contracts are covered by #362/#363/#378/#388/#389/#395/#398/#410/#414/#425/#426/#440/#456 | Real-GPU decode medians, peak memory, and parity at tile/oneshot boundaries |
| **P1** | Long-T AUTO threshold/tile retune | CPU cold/decode threshold independence, blank/unset transitions, per-call backend precedence, caller-argument isolation, decode-only overrides, scoped cold gates, pre-run validation, and failure-continuing sweeps are covered by #352/#360/#368/#377/#394/#399/#409/#413/#424/#428/#439/#444/#454/#458; AUTO remains opt-in | GPU cold/decode sweep that justifies threshold and tile choices |
| **P1** | Fused RoPE on GPU | CPU shape/layout, paired-output alias, incompatible cis leading dimensions, fp16-input/fp32-cache T=1, non-contiguous output, and public T>1 dispatch contracts are covered by #351/#366/#383/#398/#414/#427/#442/#459 | Real-GPU fused/eager parity and timing across supported dtypes/layouts |
| **P1** | AMP train and H2D overlap | CPU bf16/fp16 forward-mode contracts, explicit forward-only environment selection, live-CUDA claim gating, disabled CPU GradScaler, validation-split forwarding, repeated worker/H2D opt-out identity, reproducible worker RNG streams, and negative-worker opt-out behavior are covered by #359/#372/#374/#387/#390/#403/#405/#419/#435/#448/#449/#463; no CUDA throughput evidence | CUDA bf16/fp16 train throughput and pinned H2D overlap measurements |
| **P1** | Compile train step | CPU invalid/normalized-probe cleanup, missing-target training and no-`example_x` fallbacks, failed-probe restoration, compile-construction fallback, and force-CPU/nvcc build guards are covered by #353/#354/#369/#370/#384/#385/#400/#401/#415/#417/#430/#431/#445/#446/#460; GPU path unmeasured | GPU inductor/CUDA-graph A/B with `BDH_COMPILE=0|1` |
| **P2** | Analytic attention backward on GPU | CPU strict-past tile-boundary, strided/zero-stride-Q/K/V, batch/head-broadcast zero-stride-Q/K gradient reduction, empty-prefix, non-contiguous-Q/K/V, padded-V, strided-upstream-gradient, and empty-sequence contracts are covered by #355/#364/#371/#386/#393/#402/#407/#412/#416/#423/#432/#438/#447/#452/#461; GPU path unmeasured | GPU train-step parity and timing across eager/blocked/online/Triton/CUDA |
| **P2** | Sparse production path | No-sample, passing-sample, final-sample, terminal density-floor, custom `--min-final-xy` floor, and CLI floor-override guardrails are covered by #358/#375/#392/#408/#422/#436/#451; the probe is explicit and default-off | Density plus a measured GPU sparse-kernel win before any default change |

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
| **#403** | AMP live-CUDA gate | A live CUDA runtime is required before exposing the AMP GPU throughput claim; stale CUDA selection remains CPU-safe |
| **#404** | Cold GPU-measure skip | CUDA-unavailable diagnostics retain native handoff commands and CPU skips remain non-timed |
| **#405** | DataLoader validation split | Host validation gathering preserves pin-memory and tensor identity behavior |
| **#406** | Docs-v76 | Matrix/backlog refreshed through #402 and rebased onto #404 |
| **#407** | Blocked prefill Q/K layout | Capacity-padded, non-contiguous Q/K views preserve eager parity and strict-past masking |
| **#408** | Sparse passing guardrail | A passing density sample permits the mocked CPU crossover sweep without a performance claim |
| **#409** | AUTO cold-threshold fallback | Removing an explicit cold threshold falls back to the live decode threshold at strict boundaries |
| **#410** | Online decode dispatch dtype | Nonempty float64 packed-cache views preserve shape, dtype, and raw score×V parity |
| **#411** | Non-unit-vocabulary sampler layout | Strided multinomial/top-k outputs preserve isolation and contiguous-reference parity |
| **#412** | Score-V per-head dispatch | Distinct Q/K with per-head V preserve raw score×strict-tril semantics, including row zero |
| **#413** | Generate omitted cold threshold | Decode-only AUTO override does not create or leak an omitted cold-threshold variable |
| **#414** | Public paired RoPE dispatch | Paired T=1 mixed-dtype, strided-output parity is covered through the public dispatcher |
| **#415** | Compile construction fallback | Caller mode and existing gradients are preserved when `torch.compile` construction fails |
| **#416** | Empty attention backward | Empty-sequence strict-tril backward is covered across CPU-safe dispatch paths |
| **#417** | Executable CUDA_PATH nvcc | Setup metadata covers executable `CUDA_PATH/bin/nvcc` discovery without GPU claims |

## Refresh additions (#418–#433 and tip `d9703b0`)

| PR / tip | Scope | Recorded outcome |
|---:|---|---|
| **#418** | DataLoader producer failure | Prefetch producer exceptions surface through `next()` with the original host-gather cause |
| **#419** | AMP live-runtime gate | Positive live-CUDA throughput eligibility is covered without executing a GPU |
| **#420** | Cold GPU-measure skip | Unavailable-CUDA runs produce structured no-timing diagnostics |
| **#421** | Docs-v77 | Matrix/backlog refreshed through #417; no GPU evidence added |
| **#422** | Sparse final-sample guardrail | A failing final density sample blocks CPU crossover work |
| **#423** | Online padded-QK prefill | Online prefill preserves eager parity and strict-past masking for capacity-padded Q/K |
| **#424** | AUTO threshold scope | Explicit cold threshold remains independent from the shared decode gate |
| **#425** | Packed online decode | Multi-query dispatch preserves packed-view score×V parity and read-only inputs |
| **#426** | Score-V public dispatch | Distinct-Q/K, per-head-V raw score×V semantics hold across public aliases and CPU fallbacks |
| **#427** | Public RoPE dispatch | T>1 eager/fused paths preserve mixed-dtype, strided-output parity on CPU |
| **#428** | Generate threshold validation | Negative explicit cold thresholds fail before model setup |
| **#429** | Sampler layout | Non-unit vocabulary stride plus zero-stride outputs preserve RNG parity and isolation |
| **#430** | Compile missing-target fallback | Absent `example_y` does not run a partial probe or return a compiled wrapper; caller state is preserved |
| **#431** | CUDA build selection | Explicit CUDA build covers executable `PATH/nvcc` selection |
| **#432** | Attention backward | Zero-stride per-head V gradients reduce correctly to the expanded base across dispatch variants |
| **#433** | GPU-measure mode diagnostics | CUDA-unavailable summaries retain the requested mode and explicit handoffs |
| **tip `d9703b0`** | DataLoader worker prefetch | Full host batches, bounded prefetch, persistent workers, and per-worker initialization are locked down |

## Refresh additions (#435–#448)

| PR / scope | Recorded outcome |
|---:|---|
| **#435** AMP forward mode | Full-forward and logits-only bf16/fp16 selection remains a CPU-safe mode contract. |
| **#436** Sparse CLI floor | CLI density floors override defaults and custom floor failure stops CPU crossover work. |
| **#437** Docs-v78 | Matrix/backlog refreshed through #433 and tip `d9703b0`; no GPU evidence added. |
| **#438** Padded Q/K backward | Capacity-padded Q/K views preserve strict-tril parity and gradients into padded storage columns. |
| **#439** AUTO explicit backend | Explicit blocked/Triton/CUDA/online selections remain stable across cold and decode gates. |
| **#440** Online decode views | Multi-query per-head-V decode preserves eager raw score×V parity for offset-packed K/V. |
| **#441** Score-V gradients | Distinct-Q/K and per-head-V gradients preserve strict-tril parity across dispatch aliases. |
| **#442** T=1 RoPE cache slots | Mixed-dtype writes into non-contiguous cache slots remain CPU-safe through eager and fused implementations. |
| **#443** Sampler layout | Non-unit vocabulary stride plus unit batch stride preserves RNG parity, identity, and neighbor isolation. |
| **#444** Generate sweep isolation | AUTO threshold sweeps use isolated per-run arguments and leave caller state unchanged. |
| **#445** CUDA_HOME nvcc | Executable `CUDA_HOME/bin/nvcc` selection is covered without GPU claims. |
| **#446** Compile missing target | Training-mode callers skip the missing-target probe and preserve all pre-existing gradients. |
| **#447** Zero-stride Q/K backward | Q/K gradients reduce to underlying bases across all implementation selectors. |
| **#448** DataLoader worker RNG | Worker initialization yields distinct, reproducible CPU RNG streams without changing prefetch behavior. |


## Refresh additions (#449–#463)

| PR / scope | Recorded outcome |
|---:|---|
| **#449** AMP forward-only environment | Explicit `BDH_AMP_FORWARD_ONLY` off/on spellings remain a CPU-safe mode contract without GPU throughput evidence. |
| **#450** GPU-measure skip diagnostics | CUDA build and visible-device fields are included in schema-v5 skipped-run summaries; CPU timing is never substituted. |
| **#451** Sparse custom floor | An effective `--min-final-xy` floor is reported and a failing final floor stops before CPU crossover work. |
| **#452** Prefill backward tile boundary | Strict-tril blocked/online backward leaves diagonal and future sources gradient-free at the tile boundary. |
| **#453** Docs-v79 | Matrix/backlog refreshed through #448; no GPU evidence added. |
| **#454** AUTO requested backend | Per-call requested backend takes precedence explicitly against conflicting environment settings. |
| **#455** Shared-V score-V gradients | Distinct-Q/K shared-V and per-head-V layouts preserve raw strict-tril score×V gradients across dispatch aliases. |
| **#456** T=1 online decode | Per-head-V decode preserves raw score×V parity for offset-packed K/V and read-only inputs. |
| **#457** Sampler scratch layout | Strided probability buffers preserve multinomial, narrow/full top-k, overflow-clamped top-k, and decode-write contracts. |
| **#458** Generate sweep continuation | Threshold sweeps continue after individual failures while retaining aggregate failure status. |
| **#459** RoPE broadcast shape | Incompatible cis leading dimensions fail before rotation math on regular and paired entrypoints. |
| **#460** Compile no-probe path | Missing-`example_x` callers receive an unprobed wrapper without execution and preserve caller state. |
| **#461** Batch/head zero-stride backward | Broadcast Q/K gradients reduce to singleton bases across implementation selectors. |
| **#462** CUDA nvcc fallback | A stale non-executable `CUDA_HOME/bin/nvcc` falls back to executable `CUDA_PATH/bin/nvcc` in setup metadata. |
| **#463** Negative DataLoader workers | Negative `NUM_WORKERS` stays synchronous on CPU with pinning and persistent-worker prefetch disabled. |

## Current evidence summary

The profile-v20 call-count baseline remains attention `copy_` 2/call, forward `copy_` 12/call, generate `copy_` 394/call, `cat=0`, and `contiguous=0`. The #350–#463 additions and tip follow-up are CPU-safe docs/tests/contracts and diagnostics; #356, #373, #389, #404, #420, #433, and #450 define GPU benchmark references or handoffs but add no CUDA timing. None adds GPU throughput, sparse-kernel evidence, or cold CUDA–Triton validation.

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
