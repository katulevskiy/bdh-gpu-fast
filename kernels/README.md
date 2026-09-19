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
| `online` | Alias of `blocked` |
| `triton` | Triton fused on CUDA; blocked fallback otherwise |
| `cuda` | `kernels.cuda_attn.tril_score_v` (native ext if built, else CPU ref) |

```bash
export BDH_ATTN_IMPL=eager     # default
export BDH_ATTN_IMPL=blocked
export BDH_ATTN_IMPL=online    # alias of blocked
export BDH_ATTN_IMPL=triton
export BDH_ATTN_IMPL=cuda

# train with fused/blocked forward + analytic bwd (tiled for non-eager)
export BDH_ATTN_AUTOGRAD=1
export BDH_ATTN_IMPL=blocked

# opt-in: long-T cold + long-S decode → triton (CUDA) or blocked (CPU)
# threshold from #55 CPU benches (mid-S stay eager; S≫512 prefer)
export BDH_ATTN_AUTO=1
export BDH_ATTN_AUTO_THRESHOLD=512   # optional; default 512; switch when length > thr
```

Cold `Attention.forward` (no cache) goes through `kernels.attention_dispatch.bdh_attn`.
T=1 decode against packed past KR/V uses `bdh_attn_decode` for **all**
impls including eager (`_two_gemm_decode`). With `BDH_ATTN_AUTO=1` and default eager, length above
`BDH_ATTN_AUTO_THRESHOLD` (default 512) switches **cold/prefill** (`T`) and
**T=1 decode** (`past_len`) to **triton** if CUDA+Triton are available
(`triton_decode_available`), else **blocked** (#55 / `opt/prefill-blocked`).
Short sequences stay eager; explicit non-eager `IMPL` is never overridden.
Blocked/online/triton share
`_tiled_score_v` (broadcast-V tight `_DECODE_ONESHOT_ELEMS`, `out.add_`
tiles, peak ~Tq×tile on long S); Triton decode-v3 uses `V_BROADCAST` +
long-S tiles (up to 512) + optional Q-hoist on CUDA (CPU → #55 blocked
fallback); CUDA decode-v3 uses **adaptive** `DECODE_TILE_N` (32/64/128 on
GPU smem; CPU refs up to 512, pair #55/#62) + dedicated **Tq=1** kernel
when the ext is built. Default remains **eager**.

## Modules

| Path | Role |
|------|------|
| `attention.py` | cold tril + decode: `_two_gemm_decode`, `blocked_*` / `online_*`, `triton_*`, `_tiled_score_v` |
| `attention_dispatch.py` | `BDH_ATTN_IMPL` → `bdh_attn()` / `bdh_attn_decode()`; opt-in `BDH_ATTN_AUTO` long-T cold + long-S decode |
| `attention_bwd.py` | Optional `StrictTrilAttnFn` + analytic Q/K/V bwd (`BDH_ATTN_AUTOGRAD=1`); dense M-recompute for eager, **tiled** analytic for blocked/online/triton/cuda (no full T×T) |
| `cuda_attn.py` | Optional native CUDA/C++ ext + always-on CPU refs (eager + tiled online) |
| `rope.py` | RoPE rotate: eager / fused PyTorch / optional Triton |
| `rope_dispatch.py` | `BDH_ROPE_IMPL` → `bdh_rope_rotate()` |

## Python API (`kernels/cuda_attn.py`)

| Symbol | Role |
|--------|------|
| `tril_score_v_ref(q,k,v)` | Golden eager full tril — **bit-identical**, always works |
| `tril_score_v_tiled_ref(...)` | CPU mirror of CUDA tiled online cold (TILE_M/N=16; no full T×T) |
| `tril_score_v(q,k,v)` | Native ext if built; else tiled (large T) / eager (small T) |
| `tril_decode_ref(q,k_past,v)` | Decode ref — vectorized small S; tiles large past |
| `tril_decode_tiled_ref(...)` | CPU mirror of CUDA tiled online decode (adaptive tile) |
| `pick_cuda_decode_tile_n(S, Dk)` | Long-S past tile picker (pair #55/#62) |
| `tril_decode(q,k_past,v)` | Native decode ext if built, else decode ref |
| `has_cuda_ext()` / `has_cuda_kernel()` | Capability probes (soft — import never raises) |
| `ext_status()` | Human-readable build / load smoke string |

**Honesty:** CPU refs validate correctness on boxes without a GPU / without a
successful `BDH_BUILD_EXT` compile. They are **not** a claim of GPU speedups.
When native is missing, dispatch prefers tiled online for large T to avoid
materializing a full T×T score buffer (same structure the `.cu` scaffold uses).

## Optional native build (`csrc/`)

```bash
# default: no compile — CPU ref only (BDH_BUILD_EXT unset)
pip install -e .

# compile native (needs matching torch headers + g++/nvcc)
BDH_BUILD_EXT=1 pip install -e . --no-build-isolation
BDH_BUILD_EXT=1 BDH_BUILD_CUDA=1 pip install -e . --no-build-isolation  # force .cu
BDH_BUILD_EXT=1 BDH_FORCE_CPU_EXT=1 pip install -e . --no-build-isolation  # CPU ext only
```

Documented in `setup.py` docstring + `pyproject.toml` comments. On this CPU
sandbox, native compile may fail (torch/g++ ABI); that is OK — `import
kernels.cuda_attn` never raises, tests skip CUDA paths, and CPU refs still run.

### Cold / decode CUDA kernel shape

`tril_attn_cuda.cu`:

- Cold: **tiled online** — grid `(ceil(T/16), B·H, ceil(Dv/32))`, block `(32, 16)`;
  shared Q/K/V tiles; register `score×V`; smem>48 KiB → fused naive (still no T×T)
- Decode: **tiled online** packed-past score×V; **adaptive** past tiles
  (32/64/128 via `pick_decode_tile_n`); Tq=1 thin grid + Q hoist; naive fallback

CPU tiled refs in `cuda_attn.py` / `tril_attn_cpu.cpp` use the same TILE_M/N=16
cold tiles and adaptive decode tiles (up to 512 on CPU) so a GPU drop-in build
can reuse the tested online structure. Numerically ≡ blocked (#55) / eager.

## RoPE rotate (`BDH_ROPE_IMPL`)

| Value | Backend |
|-------|---------|
| `eager` (default) | Strided even/odd (historical `Attention.rope`) |
| `fused` | Pair-contiguous pure PyTorch; Triton on CUDA when usable |

`rope_cos_sin` caching stays in `bdh.Attention`. Modules: `rope.py`, `rope_dispatch.py`.
