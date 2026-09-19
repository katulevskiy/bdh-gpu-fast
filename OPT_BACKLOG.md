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
- RoPE cos/sin table cache by (T, head_dim, device, dtype) (`opt/rope-cache`)
- KV-style cache + incremental `generate`
- MLP `permute → contiguous → view`
- Vectorized `train.get_batch` (+ pin/non_blocking CUDA; optional DataLoader workers)
- Triton + blocked pure-PyTorch attn dispatch (`BDH_ATTN_IMPL`) — CPU blocked slower; GPU unmeasured
- Experimental sparse ReLU matmul (default off)
- CUDA extension scaffold for tril score×V (optional build)
- Weight layout: `(B,T,nh,N)` encoder einsum, decoder view+`F.linear`, optional bias fuse (`opt/weight-layout`)
- CUDA/C++ decode vs packed KR/V under `BDH_ATTN_IMPL=cuda` (scaffold; CPU ref always)
- CacheManager v2: layer-contiguous KR/V, page growth, generate **0× aten::cat** (`opt/cache-v2`)

## Ranked next work

| P | Item | Why (from profile / notes) | Target | Risk |
|---|------|----------------------------|--------|------|
| **P0** | **Measure Triton/CUDA fused tril-score×V on real GPU** | Eager path still pays full TxT `bmm` (~12% self) + `tril_` (~6%). Kernels in-tree; no CUDA on this box. | A100/H100 microbench vs eager; bit-identical | Env blocker |
| **P0** | **Fuse score×V epilogue (no materialize T×T)** | **Landed `opt/fuse-scorev`:** blocked/online strict-tril accumulation keeps peak score storage ≪ T×T; GPU Triton/CUDA measure remains open. | GPU microbench vs eager; keep deepening CUDA/Triton | High impact |
| **P1** | ~~Cache packing / fewer cats~~ | **Landed cache-pack + cache-v2**: generate `aten::cat` **0** (was 32 post-pack / ~864 pre-pack). Layer-contiguous + optional `page_size`. | done | — |
| **P1** | **`torch.compile` / inductor** | Forward: `copy_` ~23%, `mul` ~16%, LN ~9%, ReLU ~7%. **train-fuse:** optional `BDH_COMPILE=1` + fused AdamW + sync-light loop landed; measure on GPU. | GPU compile parity | Low |
| **P2** | **Fused / cached RoPE** | Attn: mul/copy/trig/neg ~60% combined self. | **Cached tables landed `opt/rope-cache`**; fused kernel still open | Low–medium |
| **P2** | **Sparsity follow-through** | Sparse path experimental — measure density; keep only if GPU win. | Density + GPU bench | Speculative |
| **P2** | **Memory layout** | Forward contiguous/clone copies significant. | **Landed `opt/weight-layout`** — `(B,T,nh,N)` encoder path, free decoder view, `F.linear`+bias hooks; embed/lm_head path: see `opt/embed-tie`; CPU wall ~noise vs tip | Low |
| **P3** | **Hardware / dtype** | No GPU here; bf16/fp16 train optional later. | GPU box | Env |
| **P3** | **Profiler CI artifact** | Traces gitignored; optional nightly upload. | CI | N/A |

## Explicit non-goals

- Softmax / diagonal inclusion
- PRs to `pathwaycom/*`
- Claiming GPU speedups from CPU profiler absolute times
- Defaulting `BDH_ATTN_IMPL=blocked` on CPU (measured slower)

## Suggested order

1. GPU: bench eager vs triton vs CUDA fused score×V
2. ~~Cache preallocate / cat-free generate~~ (**done** `opt/cache-v2`)
3. torch.compile on forward
4. RoPE fuse / layout / sparsity density

```bash
python benchmarks/profile_forward.py --mode all
```
