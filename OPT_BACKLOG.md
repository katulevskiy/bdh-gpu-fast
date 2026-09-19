# OPT backlog — ranked remaining work

Private sandbox only (`katulevskiy/bdh-gpu-opt`). **Do not** PR to `pathwaycom/bdh`.

Constraint (hard): attention stays **raw scores** × **strict lower-triangular**
`tril(diagonal=-1)` — **no softmax**, **no `1/√d`**, **no**
`F.scaled_dot_product_attention`.

Profile source: `benchmarks/profile_forward.py` on CPU
(`torch 2.14.0+cu130`, `cuda=False`), cfg `layers=4 d=128 nh=4 B=4 T=128`,
generate prompt=16 / new=32. Absolute ms are **profiler-inflated**; use **%
self CPU** and call counts. Re-run on GPU before claiming kernel wins.

## Already landed (main)

- RoPE without `stack→view`; skip redundant dtype casts
- KV-style cache + incremental `generate`
- MLP `permute → contiguous → view`
- Vectorized `train.get_batch`
- Triton + blocked pure-PyTorch attn dispatch (`BDH_ATTN_IMPL`) — CPU blocked slower; GPU unmeasured
- Experimental sparse ReLU matmul (default off)
- CUDA extension scaffold for tril score×V (optional build)

## Ranked next work

| P | Item | Why (from profile / notes) | Target | Risk |
|---|------|----------------------------|--------|------|
| **P0** | **Measure Triton/CUDA fused tril-score×V on real GPU** | Eager path still pays full TxT `bmm` (~12% self) + `tril_` (~6%). Kernels in-tree; no CUDA on this box. | A100/H100 microbench vs eager; bit-identical | Env blocker |
| **P0** | **Fuse score×V epilogue (no materialize T×T)** | Ideal: `sum_{j<i} (Q_i·K_j) V_j` without full score tensor. | Complete Triton/CUDA fused kernel | High impact |
| **P1** | **Cache packing / fewer cats** | Generate: `aten::cat` ~10% self CPU, 864 calls, ~201 MB — per-step cat on `kr`/`v`. | Prefill-sized buffer + write ptr | Medium |
| **P1** | **`torch.compile` / inductor** | Forward: `copy_` ~23%, `mul` ~16%, LN ~9%, ReLU ~7%. | compile + parity tests | Low |
| **P2** | **Fused / cached RoPE** | Attn: mul/copy/trig/neg ~60% combined self. | Phase table or fused RoPE | Low–medium |
| **P2** | **Sparsity follow-through** | Sparse path experimental — measure density; keep only if GPU win. | Density + GPU bench | Speculative |
| **P2** | **Memory layout** | Forward contiguous/clone copies significant. | Layout audit | Medium |
| **P3** | **Hardware / dtype** | No GPU here; bf16/fp16 train optional later. | GPU box | Env |
| **P3** | **Profiler CI artifact** | Traces gitignored; optional nightly upload. | CI | N/A |

## Explicit non-goals

- Softmax / diagonal inclusion
- PRs to `pathwaycom/*`
- Claiming GPU speedups from CPU profiler absolute times
- Defaulting `BDH_ATTN_IMPL=blocked` on CPU (measured slower)

## Suggested order

1. GPU: bench eager vs triton vs CUDA fused score×V
2. Cache preallocate (kill generate cat storm)
3. torch.compile on forward
4. RoPE fuse / layout / sparsity density

```bash
python benchmarks/profile_forward.py --mode all
```
