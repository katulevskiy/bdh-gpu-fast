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
The KV-cache / `generate()` decode path stays **eager** — custom kernels do not
yet support incremental attention.

## Modules

| Path | Role |
|------|------|
<<<<<<< HEAD
| `attention.py` / `attention_dispatch.py` | Triton + blocked/eager PyTorch (`BDH_ATTN_IMPL`) |
| `attention_bwd.py` | Optional `StrictTrilAttnFn` + analytic Q/K/V bwd (`BDH_ATTN_AUTOGRAD=1`) |
=======
| `attention.py` | `eager_tril_attn`, `blocked_tril_attn`, `triton_tril_attn` |
| `attention_dispatch.py` | `BDH_ATTN_IMPL` → `bdh_attn()` / `resolve_attn_impl()` |
>>>>>>> f30de27 (opt/attn-unify: single BDH_ATTN_IMPL dispatch (eager|blocked|triton|cuda))
| `cuda_attn.py` | Optional native CUDA/C++ ext + always-on CPU ref |

## Python API (`kernels/cuda_attn.py`)

| Symbol | Role |
|--------|------|
| `tril_score_v_ref(q,k,v)` | Pure PyTorch reference — **always works** (CPU/CUDA tensors) |
| `tril_score_v(q,k,v)` | Native ext if built, else reference |
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
