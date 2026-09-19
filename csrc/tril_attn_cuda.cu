#include "tril_attn.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>

// Naive CUDA scaffold: one thread per (b,h,i,d_v).
// Computes out[b,h,i,d] = sum_{j < i} (sum_e Q[b,h,i,e]*K[b,h,j,e]) * V[b,hv,j,d]
// Ready to replace with tiled/shared-mem / flash-style later.
// Semantics: strict lower-triangular (diagonal excluded), NO softmax, NO 1/sqrt(d).

template <typename scalar_t>
__global__ void tril_score_v_kernel(
    const scalar_t* __restrict__ Q,
    const scalar_t* __restrict__ K,
    const scalar_t* __restrict__ V,
    scalar_t* __restrict__ Out,
    int B, int H, int T, int Dk, int Dv, int Hv) {
  // Linear index over (B, H, T, Dv)
  const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t n = (int64_t)B * H * T * Dv;
  if (idx >= n) return;

  const int d = idx % Dv;
  int tmp = idx / Dv;
  const int i = tmp % T;
  tmp /= T;
  const int h = tmp % H;
  const int b = tmp / H;

  // Position 0 (and any i with no prior keys) → zero
  if (i == 0) {
    Out[idx] = scalar_t(0);
    return;
  }

  const int hv = (Hv == 1) ? 0 : h;
  scalar_t acc = scalar_t(0);

  for (int j = 0; j < i; ++j) {
    scalar_t score = scalar_t(0);
    const int64_t q_base = (((int64_t)b * H + h) * T + i) * Dk;
    const int64_t k_base = (((int64_t)b * H + h) * T + j) * Dk;
    for (int e = 0; e < Dk; ++e) {
      score += Q[q_base + e] * K[k_base + e];
    }
    const int64_t v_base = (((int64_t)b * Hv + hv) * T + j) * Dv;
    acc += score * V[v_base + d];
  }
  Out[idx] = acc;
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

  const int64_t n = B * H * T * Dv;
  const int threads = 256;
  const int blocks = static_cast<int>((n + threads - 1) / threads);

  const at::cuda::CUDAGuard guard(qc.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, qc.scalar_type(),
      "tril_score_v_cuda",
      [&] {
        tril_score_v_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
            qc.data_ptr<scalar_t>(),
            kc.data_ptr<scalar_t>(),
            vc.data_ptr<scalar_t>(),
            out.data_ptr<scalar_t>(),
            (int)B, (int)H, (int)T, (int)Dk, (int)Dv, (int)Hv);
      });

  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}
