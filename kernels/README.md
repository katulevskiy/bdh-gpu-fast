# Attention kernels (strict tril score×V)

BDH attention is **not** SDPA:

```text
scores = (Q @ K.mT).tril(diagonal=-1)
out    = scores @ V
```

No softmax, no `1/sqrt(d)`, diagonal excluded.

## Unified dispatch (`BDH_ATTN_IMPL`)

| Value | Backend |
|-------|---------|
| `eager` (default) | Full T×T then `tril(diagonal=-1)` |
| `blocked` | Tiled pure PyTorch (no full upper triangle) |
| `triton` | Triton fused on CUDA; blocked fallback otherwise |
| `cuda` | `kernels.cuda_attn.tril_score_v` (native ext if built, else ref) |

```bash
export BDH_ATTN_IMPL=eager     # default
export BDH_ATTN_IMPL=blocked
export BDH_ATTN_IMPL=triton
export BDH_ATTN_IMPL=cuda
```

Cold `Attention.forward` (no cache) goes through `kernels.attention_dispatch.bdh_attn`.
T=1 decode against packed past KR/V uses `bdh_attn_decode` when
`BDH_ATTN_IMPL` is `blocked` / `triton` / `cuda` (`cuda` → `tril_decode`).
Blocked/triton decode share `_tiled_score_v` with the cold past region;
Triton has a dedicated decode kernel on CUDA (blocked fallback on CPU).
Default remains **eager**.

## Modules

| Path | Role |
|------|------|
| `attention.py` | cold tril + decode: `blocked_*` / `online_*`, `triton_*`, shared `_tiled_score_v` |
| `attention_dispatch.py` | `BDH_ATTN_IMPL` → `bdh_attn()` / `bdh_attn_decode()` |
| `attention_bwd.py` | Optional `StrictTrilAttnFn` + analytic Q/K/V bwd (`BDH_ATTN_AUTOGRAD=1`) |
| `cuda_attn.py` | Optional native CUDA/C++ ext + always-on CPU ref (full + decode) |
| `rope.py` | RoPE rotate: eager / fused PyTorch / optional Triton |
| `rope_dispatch.py` | `BDH_ROPE_IMPL` → `bdh_rope_rotate()` |

## Python API (`kernels/cuda_attn.py`)

| Symbol | Role |
|--------|------|
| `tril_score_v_ref(q,k,v)` | Pure PyTorch full tril score×V — **always works** |
| `tril_score_v(q,k,v)` | Native ext if built, else reference |
| `tril_decode_ref(q,k_past,v)` | Pure PyTorch decode vs packed past — **always** |
| `tril_decode(q,k_past,v)` | Native decode ext if built, else reference |
| `has_cuda_ext()` / `has_cuda_kernel()` | Capability probes |

## Optional native build (`csrc/`)

```bash
# default: no compile — CPU ref only
pip install -e .

# compile native (needs matching torch headers + g++/nvcc)
BDH_BUILD_EXT=1 pip install -e . --no-build-isolation
BDH_BUILD_EXT=1 BDH_BUILD_CUDA=1 pip install -e . --no-build-isolation  # force .cu
```

On this CPU sandbox, native compile may fail (torch/g++ ABI); that is OK — tests
skip CUDA paths and still validate the CPU reference.

### Cold CUDA kernel shape (`opt/cuda-cold`)

`tril_attn_cuda.cu` cold path is a **tiled online** scaffold:

- Grid `(ceil(T/16), B·H, ceil(Dv/32))`, block `(32, 16)`
- Shared Q/K/V tiles; accumulate `score×V` in registers — **no global T×T**
- Falls back to a per-element fused loop if dynamic smem would exceed 48 KiB
- Decode kernel unchanged (packed past; see `opt/cuda-decode`)

## RoPE rotate (`BDH_ROPE_IMPL`)

| Value | Backend |
|-------|---------|
| `eager` (default) | Strided even/odd (historical `Attention.rope`) |
| `fused` | Pair-contiguous pure PyTorch; Triton on CUDA when usable |

`rope_cos_sin` caching stays in `bdh.Attention`. Modules: `rope.py`, `rope_dispatch.py`.
