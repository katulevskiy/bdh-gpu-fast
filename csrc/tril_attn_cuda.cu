#include "tril_attn.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>

// Cold-path CUDA: tiled online strict-tril score×V (no global T×T scores).
// Semantics: out[i] = sum_{j < i} (Q[i]·K[j]) * V[j]  — NO softmax, NO 1/sqrt(d).
//
// Grid:  (ceil(T / TILE_M), B*H, ceil(Dv / TILE_D))
// Block: (TILE_D, TILE_M)  — thread (tx,ty) owns query row i0+ty and Dv lane d0+tx
// Shared: Q tile [TILE_M, Dk], K tile [TILE_N, Dk], V tile [TILE_N, TILE_D]
// Online: for each key tile, ephemeral BM×BN scores stay in registers; ×V accumulates
// into Out. Never allocates or writes a T×T score matrix.
//
// Decode kernels below: tiled online packed-past score×V
// (opt/decode-gemm + opt/decode-mm + opt/cuda-decode-v3: adaptive DECODE_TILE_N
//  + dedicated Tq=1 path; long-S tiles pair with #55/#62).

namespace {
constexpr int TILE_M = 16;  // query rows per block (cold + multi-Tq decode)
constexpr int TILE_N = 16;  // key cols per online tile (cold)
constexpr int TILE_D = 32;  // Dv columns per block (blockDim.x)
// Decode past tiles: base 32; cuda-decode-v3 adaptive long-S up to 128 (smem).
constexpr int DECODE_TILE_N = 32;
constexpr int DECODE_TILE_N_MAX = 128;
// Soft cap: fall back to naive if dynamic smem would exceed this.
constexpr size_t SMEM_CAP = 48 * 1024;

// Pair with #55/#62: prefer larger past tiles on long packed caches.
// Shrink until float-sized smem estimate fits SMEM_CAP (half/bf16 are smaller).
__host__ inline int pick_decode_tile_n(int S, int Dk) {
  int want;
  if (S <= 64) {
    want = 32;
  } else if (S <= 256) {
    want = 64;
  } else {
    want = 128;  // long S → fewer K/V tile iterations
  }
  int tn = DECODE_TILE_N;
  while (tn < want && tn < DECODE_TILE_N_MAX) {
    tn <<= 1;
  }
  if (tn > DECODE_TILE_N_MAX) {
    tn = DECODE_TILE_N_MAX;
  }
  // Clamp to next power-of-2 not exceeding S (avoid oversizing tiny past).
  while (tn > DECODE_TILE_N && tn / 2 >= S) {
    tn >>= 1;
  }
  auto smem_bytes = [&](int tile) -> size_t {
    // Worst-case float: Qs[Dk] + Ks[TN*Dk] + Vs[TN*TILE_D]
    return sizeof(float) *
           (static_cast<size_t>(Dk) +
            static_cast<size_t>(tile) * static_cast<size_t>(Dk) +
            static_cast<size_t>(tile) * static_cast<size_t>(TILE_D));
  };
  while (tn > DECODE_TILE_N && smem_bytes(tn) > SMEM_CAP) {
    tn >>= 1;
  }
  return tn;
}
}  // namespace

template <typename scalar_t>
__global__ void tril_score_v_tiled_kernel(
    const scalar_t* __restrict__ Q,
    const scalar_t* __restrict__ K,
    const scalar_t* __restrict__ V,
    scalar_t* __restrict__ Out,
    int B, int H, int T, int Dk, int Dv, int Hv) {
  extern __shared__ char smem_raw[];
  scalar_t* Qs = reinterpret_cast<scalar_t*>(smem_raw);
  scalar_t* Ks = Qs + TILE_M * Dk;
  scalar_t* Vs = Ks + TILE_N * Dk;

  const int bh = static_cast<int>(blockIdx.y);
  const int b = bh / H;
  const int h = bh % H;
  const int hv = (Hv == 1) ? 0 : h;

  const int i0 = static_cast<int>(blockIdx.x) * TILE_M;
  const int d0 = static_cast<int>(blockIdx.z) * TILE_D;
  const int ty = static_cast<int>(threadIdx.y);
  const int tx = static_cast<int>(threadIdx.x);
  const int i = i0 + ty;
  const int d = d0 + tx;

  const int64_t q_head = (static_cast<int64_t>(b) * H + h) * T;
  const int64_t v_head = (static_cast<int64_t>(b) * Hv + hv) * T;

  // Cooperative load of Q tile (all Dk) for rows [i0, i0+TILE_M).
  for (int e = tx; e < Dk; e += TILE_D) {
    if (i < T) {
      Qs[ty * Dk + e] = Q[(q_head + i) * Dk + e];
    } else {
      Qs[ty * Dk + e] = scalar_t(0);
    }
  }
  __syncthreads();

  float acc = 0.f;
  // Key tiles that can contribute to any row in this query tile: j < i0+TILE_M.
  const int j_lim = (i0 + TILE_M < T) ? (i0 + TILE_M) : T;
  for (int j0 = 0; j0 < j_lim; j0 += TILE_N) {
    // Load K[j0:j0+TILE_N, :] and V[j0:j0+TILE_N, d0:d0+TILE_D) into shared.
    for (int idx = ty * TILE_D + tx; idx < TILE_N * Dk; idx += TILE_M * TILE_D) {
      const int jl = idx / Dk;
      const int e = idx % Dk;
      const int j = j0 + jl;
      Ks[jl * Dk + e] =
          (j < T) ? K[(q_head + j) * Dk + e] : scalar_t(0);
    }
    for (int idx = ty * TILE_D + tx; idx < TILE_N * TILE_D; idx += TILE_M * TILE_D) {
      const int jl = idx / TILE_D;
      const int dc = idx % TILE_D;
      const int j = j0 + jl;
      const int dd = d0 + dc;
      Vs[jl * TILE_D + dc] =
          (j < T && dd < Dv) ? V[(v_head + j) * Dv + dd] : scalar_t(0);
    }
    __syncthreads();

    if (i < T && d < Dv && i > 0) {
      for (int jl = 0; jl < TILE_N; ++jl) {
        const int j = j0 + jl;
        if (j >= i) {
          continue;  // strict lower-triangular: j < i only
        }
        float score = 0.f;
        for (int e = 0; e < Dk; ++e) {
          score += static_cast<float>(Qs[ty * Dk + e]) *
                   static_cast<float>(Ks[jl * Dk + e]);
        }
        acc += score * static_cast<float>(Vs[jl * TILE_D + tx]);
      }
    }
    __syncthreads();
  }

  if (i < T && d < Dv) {
    Out[(q_head + i) * Dv + d] = static_cast<scalar_t>(acc);
  }
}

// Naive fallback: one thread per (b,h,i,d). Used when tiled smem would be too large.
template <typename scalar_t>
__global__ void tril_score_v_naive_kernel(
    const scalar_t* __restrict__ Q,
    const scalar_t* __restrict__ K,
    const scalar_t* __restrict__ V,
    scalar_t* __restrict__ Out,
    int B, int H, int T, int Dk, int Dv, int Hv) {
  const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t n = static_cast<int64_t>(B) * H * T * Dv;
  if (idx >= n) return;

  const int d = idx % Dv;
  int tmp = idx / Dv;
  const int i = tmp % T;
  tmp /= T;
  const int h = tmp % H;
  const int b = tmp / H;

  if (i == 0) {
    Out[idx] = scalar_t(0);
    return;
  }

  const int hv = (Hv == 1) ? 0 : h;
  float acc = 0.f;
  const int64_t q_base = (((static_cast<int64_t>(b) * H + h) * T + i) * Dk);
  for (int j = 0; j < i; ++j) {
    float score = 0.f;
    const int64_t k_base = (((static_cast<int64_t>(b) * H + h) * T + j) * Dk);
    for (int e = 0; e < Dk; ++e) {
      score += static_cast<float>(Q[q_base + e]) *
               static_cast<float>(K[k_base + e]);
    }
    const int64_t v_base = (((static_cast<int64_t>(b) * Hv + hv) * T + j) * Dv);
    acc += score * static_cast<float>(V[v_base + d]);
  }
  Out[idx] = static_cast<scalar_t>(acc);
}

torch::Tensor tril_score_v_cuda(torch::Tensor q, torch::Tensor k, torch::Tensor v) {
  TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda(), "CUDA tensors required");
  TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4, "q,k,v must be 4D");
  TORCH_CHECK(q.sizes() == k.sizes(), "q and k shapes must match");
  TORCH_CHECK(v.size(1) == q.size(1) || v.size(1) == 1, "v heads must match or broadcast");

  const auto B = q.size(0);
  const auto H = q.size(1);
  const auto T = q.size(2);
  const auto Dk = q.size(3);
  const auto Dv = v.size(3);
  const auto Hv = v.size(1);

  auto qc = q.contiguous();
  auto kc = k.contiguous();
  auto vc = v.contiguous();
  auto out = torch::empty({B, H, T, Dv}, qc.options());

  if (T == 0 || B == 0 || H == 0) {
    return out;
  }

  const at::cuda::CUDAGuard guard(qc.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, qc.scalar_type(),
      "tril_score_v_cuda",
      [&] {
        const size_t smem = sizeof(scalar_t) *
            (static_cast<size_t>(TILE_M) * static_cast<size_t>(Dk) +
             static_cast<size_t>(TILE_N) * static_cast<size_t>(Dk) +
             static_cast<size_t>(TILE_N) * static_cast<size_t>(TILE_D));

        if (smem <= SMEM_CAP) {
          dim3 block(TILE_D, TILE_M);
          dim3 grid(
              static_cast<unsigned>((T + TILE_M - 1) / TILE_M),
              static_cast<unsigned>(B * H),
              static_cast<unsigned>((Dv + TILE_D - 1) / TILE_D));
          tril_score_v_tiled_kernel<scalar_t><<<grid, block, smem, stream>>>(
              qc.data_ptr<scalar_t>(),
              kc.data_ptr<scalar_t>(),
              vc.data_ptr<scalar_t>(),
              out.data_ptr<scalar_t>(),
              (int)B, (int)H, (int)T, (int)Dk, (int)Dv, (int)Hv);
        } else {
          // Huge Dk: keep fused online math (no T×T) via naive per-element loop.
          const int64_t n = B * H * T * Dv;
          const int threads = 256;
          const int blocks = static_cast<int>((n + threads - 1) / threads);
          tril_score_v_naive_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
              qc.data_ptr<scalar_t>(),
              kc.data_ptr<scalar_t>(),
              vc.data_ptr<scalar_t>(),
              out.data_ptr<scalar_t>(),
              (int)B, (int)H, (int)T, (int)Dk, (int)Dv, (int)Hv);
        }
      });

  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

// Decode vs packed past KR/V (opt/decode-gemm + opt/decode-mm + cuda-decode-v3):
// tiled online with adaptive past TILE_N (32/64/128).
// out[b,h,i,d] = sum_{j=0}^{S-1} (Q[b,h,i]·K[b,h,j]) * V[b,hv,j,d]
// All past keys valid (no causal mask) — same as tril(diagonal=-1) at T=1.
// Grid: (ceil(Tq/TILE_M), B*H, ceil(Dv/TILE_D)); block (TILE_D, TILE_M).
// Tq==1 dispatches tril_decode_tq1_kernel (thin grid, Q hoisted once / past scan).
// Falls back to naive per-element fused loop if smem would exceed SMEM_CAP.

template <typename scalar_t, int TILE_N>
__global__ void tril_decode_tiled_kernel(
    const scalar_t* __restrict__ Q,
    const scalar_t* __restrict__ K,
    const scalar_t* __restrict__ V,
    scalar_t* __restrict__ Out,
    int B, int H, int Tq, int S, int Dk, int Dv, int Hv) {
  extern __shared__ char smem_raw[];
  scalar_t* Qs = reinterpret_cast<scalar_t*>(smem_raw);
  scalar_t* Ks = Qs + TILE_M * Dk;
  scalar_t* Vs = Ks + TILE_N * Dk;

  const int bh = static_cast<int>(blockIdx.y);
  const int b = bh / H;
  const int h = bh % H;
  const int hv = (Hv == 1) ? 0 : h;

  const int i0 = static_cast<int>(blockIdx.x) * TILE_M;
  const int d0 = static_cast<int>(blockIdx.z) * TILE_D;
  const int ty = static_cast<int>(threadIdx.y);
  const int tx = static_cast<int>(threadIdx.x);
  const int i = i0 + ty;
  const int d = d0 + tx;

  const int64_t q_head = (static_cast<int64_t>(b) * H + h) * Tq;
  const int64_t k_head = (static_cast<int64_t>(b) * H + h) * S;
  const int64_t v_head = (static_cast<int64_t>(b) * Hv + hv) * S;

  // Hoist Q tile once before past scan (pair with Triton HOIST_Q).
  for (int e = tx; e < Dk; e += TILE_D) {
    if (i < Tq) {
      Qs[ty * Dk + e] = Q[(q_head + i) * Dk + e];
    } else {
      Qs[ty * Dk + e] = scalar_t(0);
    }
  }
  __syncthreads();

  float acc = 0.f;
  for (int j0 = 0; j0 < S; j0 += TILE_N) {
    for (int idx = ty * TILE_D + tx; idx < TILE_N * Dk; idx += TILE_M * TILE_D) {
      const int jl = idx / Dk;
      const int e = idx % Dk;
      const int j = j0 + jl;
      Ks[jl * Dk + e] =
          (j < S) ? K[(k_head + j) * Dk + e] : scalar_t(0);
    }
    for (int idx = ty * TILE_D + tx; idx < TILE_N * TILE_D; idx += TILE_M * TILE_D) {
      const int jl = idx / TILE_D;
      const int dc = idx % TILE_D;
      const int j = j0 + jl;
      const int dd = d0 + dc;
      Vs[jl * TILE_D + dc] =
          (j < S && dd < Dv) ? V[(v_head + j) * Dv + dd] : scalar_t(0);
    }
    __syncthreads();

    if (i < Tq && d < Dv) {
      for (int jl = 0; jl < TILE_N; ++jl) {
        const int j = j0 + jl;
        if (j >= S) {
          continue;
        }
        float score = 0.f;
        for (int e = 0; e < Dk; ++e) {
          score += static_cast<float>(Qs[ty * Dk + e]) *
                   static_cast<float>(Ks[jl * Dk + e]);
        }
        acc += score * static_cast<float>(Vs[jl * TILE_D + tx]);
      }
    }
    __syncthreads();
  }

  if (i < Tq && d < Dv) {
    Out[(q_head + i) * Dv + d] = static_cast<scalar_t>(acc);
  }
}

// Tq=1 specialized decode (opt/decode-mm + cuda-decode-v3): one query row,
// adaptive TILE_N past tiles. Q hoisted once into smem before the past scan.
// Grid: (1, B*H, ceil(Dv/TILE_D)); block (TILE_D, 1) — no wasted TILE_M rows.
template <typename scalar_t, int TILE_N>
__global__ void tril_decode_tq1_kernel(
    const scalar_t* __restrict__ Q,
    const scalar_t* __restrict__ K,
    const scalar_t* __restrict__ V,
    scalar_t* __restrict__ Out,
    int B, int H, int S, int Dk, int Dv, int Hv) {
  extern __shared__ char smem_raw[];
  scalar_t* Qs = reinterpret_cast<scalar_t*>(smem_raw);
  scalar_t* Ks = Qs + Dk;  // single query row
  scalar_t* Vs = Ks + TILE_N * Dk;

  const int bh = static_cast<int>(blockIdx.y);
  const int b = bh / H;
  const int h = bh % H;
  const int hv = (Hv == 1) ? 0 : h;
  const int d0 = static_cast<int>(blockIdx.z) * TILE_D;
  const int tx = static_cast<int>(threadIdx.x);
  const int d = d0 + tx;

  const int64_t q_base = (static_cast<int64_t>(b) * H + h) * Dk;  // Tq=1
  const int64_t k_head = (static_cast<int64_t>(b) * H + h) * S;
  const int64_t v_head = (static_cast<int64_t>(b) * Hv + hv) * S;

  // Hoist the single query into smem once (before past K/V scan).
  for (int e = tx; e < Dk; e += TILE_D) {
    Qs[e] = Q[q_base + e];
  }
  __syncthreads();

  float acc = 0.f;
  for (int j0 = 0; j0 < S; j0 += TILE_N) {
    for (int idx = tx; idx < TILE_N * Dk; idx += TILE_D) {
      const int jl = idx / Dk;
      const int e = idx % Dk;
      const int j = j0 + jl;
      Ks[jl * Dk + e] =
          (j < S) ? K[(k_head + j) * Dk + e] : scalar_t(0);
    }
    for (int idx = tx; idx < TILE_N * TILE_D; idx += TILE_D) {
      const int jl = idx / TILE_D;
      const int dc = idx % TILE_D;
      const int j = j0 + jl;
      const int dd = d0 + dc;
      Vs[jl * TILE_D + dc] =
          (j < S && dd < Dv) ? V[(v_head + j) * Dv + dd] : scalar_t(0);
    }
    __syncthreads();

    if (d < Dv) {
      for (int jl = 0; jl < TILE_N; ++jl) {
        const int j = j0 + jl;
        if (j >= S) {
          continue;
        }
        float score = 0.f;
        for (int e = 0; e < Dk; ++e) {
          score += static_cast<float>(Qs[e]) *
                   static_cast<float>(Ks[jl * Dk + e]);
        }
        acc += score * static_cast<float>(Vs[jl * TILE_D + tx]);
      }
    }
    __syncthreads();
  }

  if (d < Dv) {
    // Out (B,H,1,Dv) contiguous: ((b*H+h)*1 + 0)*Dv + d
    Out[((static_cast<int64_t>(b) * H + h) * Dv) + d] = static_cast<scalar_t>(acc);
  }
}

// Naive fallback: one thread per (b,h,i,d). Still fused online (no Tq×S alloc).
template <typename scalar_t>
__global__ void tril_decode_naive_kernel(
    const scalar_t* __restrict__ Q,
    const scalar_t* __restrict__ K,
    const scalar_t* __restrict__ V,
    scalar_t* __restrict__ Out,
    int B, int H, int Tq, int S, int Dk, int Dv, int Hv) {
  const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t n = (int64_t)B * H * Tq * Dv;
  if (idx >= n) return;

  const int d = idx % Dv;
  int tmp = idx / Dv;
  const int i = tmp % Tq;
  tmp /= Tq;
  const int h = tmp % H;
  const int b = tmp / H;

  if (S == 0) {
    Out[idx] = scalar_t(0);
    return;
  }

  const int hv = (Hv == 1) ? 0 : h;
  float acc = 0.f;
  const int64_t q_base = (((int64_t)b * H + h) * Tq + i) * Dk;

  for (int j = 0; j < S; ++j) {
    float score = 0.f;
    const int64_t k_base = (((int64_t)b * H + h) * S + j) * Dk;
    for (int e = 0; e < Dk; ++e) {
      score += static_cast<float>(Q[q_base + e]) *
               static_cast<float>(K[k_base + e]);
    }
    const int64_t v_base = (((int64_t)b * Hv + hv) * S + j) * Dv;
    acc += score * static_cast<float>(V[v_base + d]);
  }
  Out[idx] = static_cast<scalar_t>(acc);
}

torch::Tensor tril_decode_cuda(torch::Tensor q, torch::Tensor k_past, torch::Tensor v_past) {
  TORCH_CHECK(q.is_cuda() && k_past.is_cuda() && v_past.is_cuda(), "CUDA tensors required");
  TORCH_CHECK(q.dim() == 4 && k_past.dim() == 4 && v_past.dim() == 4, "q,k,v must be 4D");
  TORCH_CHECK(q.size(0) == k_past.size(0) && q.size(1) == k_past.size(1),
              "q and k_past batch/heads must match");
  TORCH_CHECK(q.size(3) == k_past.size(3), "Dk must match");
  TORCH_CHECK(v_past.size(0) == q.size(0) && v_past.size(2) == k_past.size(2),
              "v_past batch/seq must match k_past");
  TORCH_CHECK(v_past.size(1) == q.size(1) || v_past.size(1) == 1,
              "v heads must match or broadcast");

  const auto B = q.size(0);
  const auto H = q.size(1);
  const auto Tq = q.size(2);
  const auto S = k_past.size(2);
  const auto Dk = q.size(3);
  const auto Dv = v_past.size(3);
  const auto Hv = v_past.size(1);

  auto qc = q.contiguous();
  auto kc = k_past.contiguous();
  auto vc = v_past.contiguous();
  auto out = torch::empty({B, H, Tq, Dv}, qc.options());

  if (S == 0 || Tq == 0 || B == 0 || H == 0) {
    out.zero_();
    return out;
  }

  const at::cuda::CUDAGuard guard(qc.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const int tile_n = pick_decode_tile_n(static_cast<int>(S), static_cast<int>(Dk));

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, qc.scalar_type(),
      "tril_decode_cuda",
      [&] {
        const size_t esz = sizeof(scalar_t);
        const size_t smem_tq1 =
            esz *
            (static_cast<size_t>(Dk) +
             static_cast<size_t>(tile_n) * static_cast<size_t>(Dk) +
             static_cast<size_t>(tile_n) * static_cast<size_t>(TILE_D));
        const size_t smem_multi =
            esz *
            (static_cast<size_t>(TILE_M) * static_cast<size_t>(Dk) +
             static_cast<size_t>(tile_n) * static_cast<size_t>(Dk) +
             static_cast<size_t>(tile_n) * static_cast<size_t>(TILE_D));

        if (Tq == 1 && smem_tq1 <= SMEM_CAP) {
          dim3 block(TILE_D, 1);
          dim3 grid(
              1u,
              static_cast<unsigned>(B * H),
              static_cast<unsigned>((Dv + TILE_D - 1) / TILE_D));
          if (tile_n >= 128) {
            tril_decode_tq1_kernel<scalar_t, 128>
                <<<grid, block, smem_tq1, stream>>>(
                    qc.data_ptr<scalar_t>(), kc.data_ptr<scalar_t>(),
                    vc.data_ptr<scalar_t>(), out.data_ptr<scalar_t>(),
                    (int)B, (int)H, (int)S, (int)Dk, (int)Dv, (int)Hv);
          } else if (tile_n >= 64) {
            tril_decode_tq1_kernel<scalar_t, 64>
                <<<grid, block, smem_tq1, stream>>>(
                    qc.data_ptr<scalar_t>(), kc.data_ptr<scalar_t>(),
                    vc.data_ptr<scalar_t>(), out.data_ptr<scalar_t>(),
                    (int)B, (int)H, (int)S, (int)Dk, (int)Dv, (int)Hv);
          } else {
            tril_decode_tq1_kernel<scalar_t, 32>
                <<<grid, block, smem_tq1, stream>>>(
                    qc.data_ptr<scalar_t>(), kc.data_ptr<scalar_t>(),
                    vc.data_ptr<scalar_t>(), out.data_ptr<scalar_t>(),
                    (int)B, (int)H, (int)S, (int)Dk, (int)Dv, (int)Hv);
          }
          return;
        }

        if (smem_multi <= SMEM_CAP) {
          dim3 block(TILE_D, TILE_M);
          dim3 grid(
              static_cast<unsigned>((Tq + TILE_M - 1) / TILE_M),
              static_cast<unsigned>(B * H),
              static_cast<unsigned>((Dv + TILE_D - 1) / TILE_D));
          if (tile_n >= 128) {
            tril_decode_tiled_kernel<scalar_t, 128>
                <<<grid, block, smem_multi, stream>>>(
                    qc.data_ptr<scalar_t>(), kc.data_ptr<scalar_t>(),
                    vc.data_ptr<scalar_t>(), out.data_ptr<scalar_t>(),
                    (int)B, (int)H, (int)Tq, (int)S, (int)Dk, (int)Dv, (int)Hv);
          } else if (tile_n >= 64) {
            tril_decode_tiled_kernel<scalar_t, 64>
                <<<grid, block, smem_multi, stream>>>(
                    qc.data_ptr<scalar_t>(), kc.data_ptr<scalar_t>(),
                    vc.data_ptr<scalar_t>(), out.data_ptr<scalar_t>(),
                    (int)B, (int)H, (int)Tq, (int)S, (int)Dk, (int)Dv, (int)Hv);
          } else {
            tril_decode_tiled_kernel<scalar_t, 32>
                <<<grid, block, smem_multi, stream>>>(
                    qc.data_ptr<scalar_t>(), kc.data_ptr<scalar_t>(),
                    vc.data_ptr<scalar_t>(), out.data_ptr<scalar_t>(),
                    (int)B, (int)H, (int)Tq, (int)S, (int)Dk, (int)Dv, (int)Hv);
          }
        } else {
          const int64_t n = B * H * Tq * Dv;
          const int threads = 256;
          const int blocks = static_cast<int>((n + threads - 1) / threads);
          tril_decode_naive_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
              qc.data_ptr<scalar_t>(),
              kc.data_ptr<scalar_t>(),
              vc.data_ptr<scalar_t>(),
              out.data_ptr<scalar_t>(),
              (int)B, (int)H, (int)Tq, (int)S, (int)Dk, (int)Dv, (int)Hv);
        }
      });

  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}
