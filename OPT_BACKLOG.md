# OPT backlog — ranked remaining work

Private sandbox only (`katulevskiy/bdh-gpu-opt`). **Do not** PR to `pathwaycom/bdh`.

Constraint (hard): attention stays **raw scores** × **strict lower-triangular**
`tril(diagonal=-1)` — **no softmax**, **no `1/√d`**, **no**
`F.scaled_dot_product_attention`.

Profile source: `benchmarks/profile_forward.py` on CPU
(`torch 2.14.0+cu130`, `cuda=False`), tip `3d3ed2b`, cfg `layers=4 d=128 nh=4 B=4 T=128`,
generate prompt=16 / new=32. Absolute ms are **profiler-inflated**; use **%
self CPU** and call counts. Re-run on GPU before claiming kernel wins.

Post-#19–#22 re-profile (`opt/profile-v2`): generate **0× `aten::cat`**; default
eager still pays full T×T `bmm`+`tril`. See `OPT_NOTES.md` § opt/profile-v2.

## Already landed (main)

- RoPE without `stack→view`; skip redundant dtype casts
- RoPE cos/sin table cache by (T, head_dim, device, dtype) (`opt/rope-cache` #18)
- KV-style cache + incremental `generate`
- MLP `permute → contiguous → view`
- Vectorized `train.get_batch` (+ pin/non_blocking CUDA; optional DataLoader workers #14)
- Triton + blocked pure-PyTorch attn dispatch (`BDH_ATTN_IMPL`) — CPU blocked slower; GPU unmeasured
- Experimental sparse ReLU matmul (default off)
- CUDA extension scaffold for tril score×V (optional build)
- Weight layout: `(B,T,nh,N)` encoder einsum, decoder view+`F.linear`, optional bias fuse (#13)
- Contiguous vocab proj + optional `tie_weights` (`opt/embed-tie` #16)
- `F.dropout` + identity `p=0` for compile (`opt/dropout-fuse` #17)
- CUDA/C++ decode vs packed KR/V under `BDH_ATTN_IMPL=cuda` (scaffold; CPU ref always) (#15)
- Blocked/Triton decode polish vs packed KR/V (`opt/triton-decode2` #19)
- CacheManager v2: layer-contiguous KR/V, page growth, generate **0× aten::cat** (`opt/cache-v2` #20)
- Online fused strict-tril score×V (no full T×T) under `blocked`/`online` (`opt/fuse-scorev` #21)
- `torch.compile` harden: train probe, graph-break docs, CPU inductor parity (`opt/compile-harden` #22)

## Ranked next work

| P | Item | Why (from profile / notes) | Target | Risk |
|---|------|----------------------------|--------|------|
| **P0** | **Measure Triton/CUDA fused tril-score×V on real GPU** | Default **eager** still: attn self `bmm` ~36% + `tril` ~7%; forward `bmm` ~29% + `tril` ~3%. Online/blocked (#21) + CUDA/Triton scaffolds in-tree; **no CUDA on this box**. | A100/H100 microbench vs eager; bit-identical | Env blocker |
| **P1** | **`torch.compile` GPU parity / train bench** | Forward still `copy_` ~20%, `mm` ~12%, `mul`/`mul_` ~12%, LN ~4%. Compile path hardened (#22); **GPU inductor / CUDA graphs unmeasured**. | GPU compile train step vs eager | Low |
| **P1** | **Decode GEMM / copy tax on generate** | Generate: Python `BDH.generate` ~26%, `bmm` ~20%, `copy_` ~8%, `mm` ~4%, `einsum` ~4%, `slice` ~3%. **Cats gone** (#20). Remaining: incremental score×V kernel + fewer host copies. | GPU decode kernel bench; keep cat-free | Medium |
| **P2** | **Fused RoPE kernel** | Attn: `mul` ~19% + `copy_` ~11% + `sub`/`add` (RoPE). **Cached tables landed** (#18); fused rotate kernel still open. | Optional fused RoPE on GPU | Low–medium |
| **P2** | **Sparsity follow-through** | Sparse path experimental — measure density; keep only if GPU win. | Density + GPU bench | Speculative |
| **P3** | **Hardware / dtype** | No GPU here; bf16/fp16 train optional later. | GPU box | Env |
| **P3** | **Profiler CI artifact** | Traces gitignored; optional nightly upload. | CI | N/A |

### Done (struck from ranked queue)

| Was | Status |
|-----|--------|
| Cache packing / fewer cats | **Done** #19–#20 — generate `aten::cat` **0** (was ~10% self / ~864 calls pre-pack) |
| Fuse score×V epilogue (no materialize T×T) | **Landed** #21 under `BDH_ATTN_IMPL=blocked\|online` (CPU slower; GPU measure = P0) |
| `torch.compile` / inductor CPU harden | **Landed** #17+#22; remaining = GPU measure (P1) |
| Memory layout / embed path | **Landed** #12–#13+#16 |

## Explicit non-goals

- Softmax / diagonal inclusion
- PRs to `pathwaycom/*`
- Claiming GPU speedups from CPU profiler absolute times
- Defaulting `BDH_ATTN_IMPL=blocked` on CPU (measured slower)
- Re-introducing `aten::cat` in `generate` / packed cache path

## Suggested order

1. GPU: bench eager vs blocked/online vs Triton vs CUDA fused score×V (cold + T=1 decode)
2. GPU: `BDH_COMPILE=1` train-step vs eager (after #22 probe)
3. ~~Cache preallocate / cat-free generate~~ (**done** `opt/cache-v2` #20)
4. RoPE fuse / sparsity density / dtype

```bash
python benchmarks/profile_forward.py --mode all
```
