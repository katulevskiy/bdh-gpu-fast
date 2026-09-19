# Attention kernels (strict tril score×V)

BDH attention is **not** SDPA:

```text
scores = (Q @ K.mT).tril(diagonal=-1)
out    = scores @ V
```

No softmax, no `1/sqrt(d)`, diagonal excluded.

## Modules

| Path | Role |
|------|------|
| `attention.py` / `attention_dispatch.py` | Triton + blocked/eager PyTorch (`BDH_ATTN_IMPL`) |
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
