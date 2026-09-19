# OPT status — landed work (#1–#40)

Private sandbox only: [`katulevskiy/bdh-gpu-opt`](https://github.com/katulevskiy/bdh-gpu-opt).
**Do not** open PRs against `pathwaycom/bdh` or any `pathwaycom/*` repo.

Tip documented here: `d3ff475` (`opt/profile-v3` #40 on `main`; profile captured at `b160469` after #39).
Detail / benches: [`OPT_NOTES.md`](OPT_NOTES.md). Ranked remaining: [`OPT_BACKLOG.md`](OPT_BACKLOG.md).

Hard constraint (all opts): attention stays **raw scores** × **strict lower-triangular**
`tril(diagonal=-1)` — **no softmax**, **no `1/√d`**, **no**
`F.scaled_dot_product_attention`.

---

## Honest CPU vs GPU (read this first)

| Claim | Status on this sandbox |
|-------|------------------------|
| Device | **CPU-only** (`torch 2.14.0+cu130`, `cuda=False`) |
| Algorithmic wins measured on CPU | **Yes** — KV / CacheManager generate (~2.5–3×), vectorized `get_batch`, cat-free decode path, compile-friendly structure |
| Kernel wins (Triton / CUDA fused score×V, fused RoPE, AMP train throughput) | **Not measured** — scaffolds + harnesses in-tree; GPU benches skip cleanly |
| Default train / attn path | Still **eager** + **fp32** — opt-in env flags only |
| CPU `blocked` / `triton` (→blocked) attn | Usually **slower** than eager (Python tile loop); keep for parity / peak-memory, not default |
| CPU `BDH_COMPILE=1` | Small warm-inductor train-step win possible (~1.5× on tiny cfg after cache); **not** a CUDA-graph / GPU claim |
| CPU AMP (`BDH_AMP_DTYPE`) | Correctness smoke; often **slower** than fp32 on CPU |
| Sparse ReLU | **Default OFF**; short-train densifies but not to paper ~5%; CPU sparse≪dense |

**Rule:** never cite CPU profiler absolute ms or CPU microbench medians as GPU speedups.
Re-run on A100/H100 via `benchmarks/bench_gpu_attn.py` before claiming kernel wins.

---

## Env flags (operator-facing)

### `BDH_ATTN_IMPL` — attention backend (default `eager`)

| Value | Cold / prefill | T=1 decode vs packed KR/V | Notes |
|-------|----------------|---------------------------|-------|
| `eager` | Full `T×T` then `tril_(diagonal=-1)` | Two-GEMM `(Q@K.mT)@V` | **Default**; reference math |
| `blocked` | Online / tiled fused score×V (no full `T×T`) | Tiled decode, broadcast V | Lower peak score mem; CPU often slower |
| `triton` | Triton fused on CUDA; else blocked | Triton decode + `V_BROADCAST`; else blocked | Needs CUDA + Triton to run kernel |
| `cuda` | Native ext if built (`BDH_BUILD_EXT=1`), else PyTorch ref | `tril_decode` tiled / ref | Scaffold; GPU measure open |

Also: `BDH_ATTN_AUTOGRAD=1` → `StrictTrilAttnFn` analytic Q/K/V backward (opt-in; #7/#39). Default **off**. With `IMPL=blocked|online|triton|cuda`, bwd is **tiled** (no full T×T); eager keeps dense M-recompute. T=1 CacheManager decode / `generate` stay on the decode path.

```bash
export BDH_ATTN_IMPL=eager     # default
export BDH_ATTN_IMPL=blocked
export BDH_ATTN_IMPL=triton
export BDH_ATTN_IMPL=cuda
```

### `BDH_ROPE_IMPL` — RoPE rotate (default `eager`)

| Value | Behavior |
|-------|----------|
| `eager` | Historical strided even/odd rotate (bit-identical default) |
| `fused` | Pair-contiguous PyTorch; Triton on CUDA when usable |

Cos/sin **tables** are cached in `Attention` regardless (#18). This flag only picks how cos/sin apply to `v` (#29).

```bash
export BDH_ROPE_IMPL=eager   # default
export BDH_ROPE_IMPL=fused
```

### `BDH_COMPILE` — `torch.compile` (default `0` / off)

| Env | Default | Meaning |
|-----|---------|---------|
| `BDH_COMPILE` | `0` | Set `1` / `true` to enable `train.maybe_compile` |
| `BDH_COMPILE_MODE` | `default` | `default` \| `reduce-overhead` \| `max-autotune` |
| `BDH_COMPILE_PROBE` | `train` | `eval` \| `train` \| `train_bwd` |
| `BDH_COMPILE_FULLGRAPH` | `0` | `fullgraph=True` when set |

`generate()` is `@torch.compiler.disable`. Changing `BDH_ATTN_IMPL` after compile → recompile. On CPU, `reduce-overhead` does **not** give CUDA graphs.

```bash
BDH_COMPILE=0 python train.py
BDH_COMPILE=1 BDH_COMPILE_PROBE=train python train.py
```

### `BDH_AMP_DTYPE` — train autocast (default `float32` / unset)

| Value | Autocast | GradScaler |
|-------|----------|------------|
| `float32` / `fp32` / `off` / unset | Off | Off |
| `bfloat16` / `bf16` | Yes | **Never** |
| `float16` / `fp16` / `half` | Yes | Only if **CUDA** |

CPU AMP is for parity smoke, not speed. Decode path AMP (older #5) is separate from this train knob (#24).

```bash
BDH_AMP_DTYPE=float32 python train.py
BDH_AMP_DTYPE=bfloat16 python train.py   # needs bf16 support on device
BDH_AMP_DTYPE=float16 python train.py    # GradScaler only on CUDA
```

---

## Landed opts (#1–#40)

| # | Branch / title | What landed | CPU | GPU |
|---|----------------|-------------|-----|-----|
| **1** | `opt/kernels` | KV-style cache + incremental `generate`; RoPE without `stack→view`; MLP `permute→contiguous→view`; preserve `tril(-1)` | Generate ~2.5× vs full recompute; forward ~1.3× (layout/RoPE) | Kernel rewrite N/A yet |
| **2** | `opt/triton-attn` | Triton + blocked pure-PyTorch strict-tril score@V; `BDH_ATTN_IMPL` | Blocked ≪ eager wall; Triton→blocked | Triton in-tree, unexecuted here |
| **3** | `opt/cuda-ext` | CUDA/C++ scaffold + CPU ref for tril score×V | CPU ref correct | Ext optional build; no GPU run |
| **4** | `opt/profiler` | Operator CPU profiler + `OPT_BACKLOG.md` | Profiles this box | CUDA activities when available |
| **5** | `opt/decode-amp` | CacheManager decode path + optional AMP on generate | Cache path solid | AMP train throughput open |
| **6** | `opt/qkv-fuse` | Shared RoPE cis; fuse rope `out=`; inplace ReLU proj | Fewer allocs / noise-level | — |
| **7** | `opt/attn-bwd` | Analytic `StrictTrilAttnFn` (`BDH_ATTN_AUTOGRAD`) | Gradcheck / fp64 match | — |
| **8** | `opt/attn-unify` | Unified `BDH_ATTN_IMPL` eager\|blocked\|triton\|cuda | Cold+decode dispatch | CUDA path scaffold |
| **9** | `opt/inc-decode` | Blocked/Triton T=1 decode vs packed KR/V | ≡ eager last row | GPU decode bench open |
| **10** | `opt/compile-train` | Fused AdamW, `set_to_none`, prefetch, compile path | Train microbench fused path | — |
| **11** | `opt/train-fuse` | Opt-in `BDH_COMPILE` (default 0); defer loss `.item()` | Sync-light logging | — |
| **12** | `opt/ln-fuse` | Residual LN reuse + encoder proj helpers | Small LN slice win | — |
| **13** | `opt/weight-layout` | Contiguous `(B,T,nh,N)` encoder; decoder view + `F.linear` | ~noise vs tip | Layout for GPU GEMM |
| **14** | `opt/dataloader` | Pin / non_blocking + optional DataLoader workers | Host gather ≪ model | H2D wins need CUDA |
| **15** | `opt/cuda-decode` | CUDA scaffold T=1 decode vs packed KR/V | CPU ref always | Ext optional |
| **16** | `opt/embed-tie` | Contiguous vocab proj + optional `tie_weights` | Layout tidy | — |
| **17** | `opt/dropout-fuse` | `F.dropout` + identity `p=0` for compile | Inductor-friendly | — |
| **18** | `opt/rope-cache` | Cache RoPE cos/sin by `(T, head_dim, device, dtype)` | Fewer phase rebuilds | — |
| **19** | `opt/triton-decode2` | Polish blocked/Triton decode vs packed KR/V | Parity vs eager | GPU open |
| **20** | `opt/cache-v2` | CacheManager deepen — generate **0× `aten::cat`** | Cat tax gone | Keep cat-free on GPU |
| **21** | `opt/fuse-scorev` | Online fused strict-tril score×V (`blocked`/`online`) | CPU slower; lower peak mem | **P0** GPU measure |
| **22** | `opt/compile-harden` | Train probe, graph-break docs, CPU inductor parity | Compile↔eager @ dropout=0 | CUDA graphs unmeasured |
| **23** | `opt/profile-v2` | Re-profile tip; refresh backlog post-#19–#22 | Docs only | Points GPU P0s |
| **24** | `opt/bf16-train` | Opt-in `BDH_AMP_DTYPE` + GradScaler fp16+CUDA | CPU AMP smoke | Train bench open |
| **25** | `opt/triton-cold` | Better cold Triton tiles + less staging | →blocked on CPU | **P0** GPU validate |
| **26** | `opt/gpu-bench` | `benchmarks/bench_gpu_attn.py` + backlog runbook | Clean skip if no CUDA | Harness ready |
| **27** | `opt/cuda-cold` | Tiled online cold CUDA tril score×V (no global `T×T`) | Ref path | **P0** GPU measure |
| **28** | `opt/decode-copy` | `CacheManager.reserve` + in-place RoPE; fewer `copy_` | `copy_` calls cut ~2× | Remaining: decode GEMM |
| **29** | `opt/rope-fuse` | Fused RoPE rotate (`BDH_ROPE_IMPL`) | Default eager; fused PT | Triton GPU open |
| **30** | `opt/ln-compile` | `F.layer_norm` residual; fewer Dynamo specializations | Compile-friendlier | — |
| **31** | `opt/compile-bench` | Honest CPU `BDH_COMPILE=0` vs `1` train-step harness | Warm ~1.5× tiny cfg | GPU A/B = P1 |
| **32** | `opt/sparse-probe` | Short-train ReLU density + CPU sparse crossover | Density↓ but sparse≪dense; **default OFF** | Sparse GPU speculative |
| **33** | `opt/decode-gemm` | T=1 decode score×V polish (blocked/Triton/CUDA) | ≡ eager; ~parity wall | `--mode decode` GPU open |
| **34** | `opt/docs-matrix` | `OPT_STATUS.md` (#1–#33) + README pointer | Docs only | Points GPU P0s |
| **35** | `opt/gen-bench` | `benchmarks/bench_generate.py` CacheManager × attn impls | CPU ~noise; `aten::cat=0`; tokens≡eager | E2E GPU = P1 |
| **36** | `opt/mlp-fuse` | Fused bias+ReLU + MLP merge without contiguous copies | Forward/generate **`aten::contiguous`=0**; wall ~noise | Epilogue hygiene for compile/GPU |
| **37** | `opt/cuda-ref-v2` | Deepen CUDA attn CPU refs + build smoke | Soft-skip without ext | **P0** GPU measure still open |
| **38** | `opt/blocked-vec` | Vectorized CPU blocked/online tril tiles | ~18–36× vs old blocked wall; still < eager | Default eager unchanged |
| **39** | `opt/attn-bwd-train` | Analytic `StrictTrilAttnFn` first-class train path (`BDH_ATTN_AUTOGRAD`); cold+multi-token wiring; `bench_attn_bwd.py` | Grad parity @ dropout=0; CPU tiny train_step ~noise | GPU train unmeasured |
| **40** | `opt/profile-v3` | Re-profiled post-#36 through #39; refreshed `OPT_STATUS` / `OPT_BACKLOG` / `OPT_NOTES` | Docs/profile only; generate `aten::cat`=0, forward `aten::contiguous`=0 | No GPU measurements; defaults unchanged |

Related early landings without a #1–#33 slot (still on main, documented in notes):

- **`opt/sparse-relu`** — experimental `bdh_sparse.py` helpers; **default OFF**
- **`opt/cache-pack`** — preallocated KR/V CacheManager (superseded/deepened by #20)

---

## Defaults vs opt-in (cheat sheet)

| Knob | Production-safe default | When to flip |
|------|-------------------------|--------------|
| `BDH_ATTN_IMPL` | `eager` | GPU after microbench win; or `blocked` for peak-mem experiments |
| `BDH_ATTN_AUTOGRAD` | off / unset | `1` when training with non-eager attn; blocked/online → tiled analytic bwd |
| `BDH_ROPE_IMPL` | `eager` | `fused` after GPU RoPE bench |
| `BDH_COMPILE` | `0` | `1` after probe succeeds; prefer GPU for real win |
| `BDH_AMP_DTYPE` | `float32` | `bf16`/`fp16` on CUDA train boxes |
| Sparse ReLU | OFF | Only if density + GPU sparse kernel win |

---

## Quick re-run

```bash
cd /workspace/bdh-gpu-opt   # or this worktree
source .venv/bin/activate
python -m pytest tests/ -q
python benchmarks/profile_forward.py --mode all
BDH_BENCH_COMPILE=1 python benchmarks/bench_train_step.py
python benchmarks/bench_sparse_probe.py
# on a CUDA box:
python benchmarks/bench_gpu_attn.py
python benchmarks/bench_gpu_attn.py --mode decode --T 512
```

---

## Explicit non-goals

- Softmax / diagonal inclusion / SDPA drop-in
- PRs or pushes to `pathwaycom/*`
- Claiming GPU speedups from CPU medians or profiler-inflated absolute times
- Defaulting `BDH_ATTN_IMPL=blocked` on CPU
- Re-introducing `aten::cat` in packed `generate`
- Wiring sparse ReLU into default `BDH.forward` without a measured win
