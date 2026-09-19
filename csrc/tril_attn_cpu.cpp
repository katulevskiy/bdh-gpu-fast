#include "tril_attn.h"

#include <ATen/ATen.h>
#include <algorithm>
#include <cmath>

namespace {
// Match csrc/tril_attn_cuda.cu tile sizes for CPU online scaffolds.
constexpr int64_t TILE_M = 16;
constexpr int64_t TILE_N = 16;
constexpr int64_t DECODE_TILE_N = 32;  // decode-mm: larger past tiles (no causal diag)
// Below this score footprint, prefer a single vectorized matmul (eager-shaped).
constexpr int64_t SCORE_ELEMS_EAGER_OK = 256 * 256;

// Widen half/bf16 to fp32 for accumulation; leave f32/f64 alone (no copy).
inline torch::Tensor maybe_acc(const torch::Tensor& t) {
  if (t.scalar_type() == torch::kHalf || t.scalar_type() == torch::kBFloat16) {
    return t.contiguous().to(torch::kFloat32);
  }
  return t.is_contiguous() ? t : t.contiguous();
}

}  // namespace

// Golden / small-T path: materialize scores then mask then matmul.
// Matches PyTorch: (Q @ K.mT).tril(diagonal=-1) @ V
// No V expand — matmul broadcasts (B,1,T,Dv) over heads.
static torch::Tensor tril_score_v_cpu_eager(torch::Tensor q, torch::Tensor k,
                                            torch::Tensor v) {
  auto qf = maybe_acc(q);
  auto kf = maybe_acc(k);
  auto vf = maybe_acc(v);

  auto scores = at::matmul(qf, kf.transpose(-2, -1));
  scores = scores.tril(/*diagonal=*/-1);
  auto out = at::matmul(scores, vf);
  return out.to(q.scalar_type());
}

// Tiled online cold (mirrors CUDA tril_score_v_tiled_kernel):
// past tiles j < i0 + diagonal row-wise strict tril; no full T×T retained.
static torch::Tensor tril_score_v_cpu_tiled(torch::Tensor q, torch::Tensor k,
                                            torch::Tensor v) {
  const auto B = q.size(0);
  const auto H = q.size(1);
  const auto T = q.size(2);
  const auto Dv = v.size(3);

  auto qf = maybe_acc(q);
  auto kf = maybe_acc(k);
  auto vf = maybe_acc(v);

  auto out = torch::zeros({B, H, T, Dv}, qf.options());
  if (T == 0) {
    return out.to(q.scalar_type());
  }

  for (int64_t i0 = 0; i0 < T; i0 += TILE_M) {
    const int64_t i1 = std::min(i0 + TILE_M, T);
    auto Qi = qf.narrow(/*dim=*/2, i0, i1 - i0);

    // Past key tiles: every j < i0 is strictly before queries in [i0,i1).
    for (int64_t j0 = 0; j0 < i0; j0 += TILE_N) {
      const int64_t j1 = std::min(j0 + TILE_N, i0);
      auto Kj = kf.narrow(/*dim=*/2, j0, j1 - j0);
      auto Vj = vf.narrow(/*dim=*/2, j0, j1 - j0);
      auto tile = at::matmul(at::matmul(Qi, Kj.transpose(-2, -1)), Vj);
      out.narrow(/*dim=*/2, i0, i1 - i0).add_(tile);
    }

    // Diagonal: online rows with j < i inside the block (tril diagonal=-1).
    const int64_t Bi = i1 - i0;
    for (int64_t r = 1; r < Bi; ++r) {
      auto qi = qf.narrow(/*dim=*/2, i0 + r, 1);
      auto Kj = kf.narrow(/*dim=*/2, i0, r);
      auto Vj = vf.narrow(/*dim=*/2, i0, r);
      auto row = at::matmul(at::matmul(qi, Kj.transpose(-2, -1)), Vj);
      out.narrow(/*dim=*/2, i0 + r, 1).add_(row);
    }
  }
  return out.to(q.scalar_type());
}

torch::Tensor tril_score_v_cpu(torch::Tensor q, torch::Tensor k, torch::Tensor v) {
  TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4,
              "q,k,v must be 4D (B,H,T,D)");
  TORCH_CHECK(q.sizes() == k.sizes(), "q and k must have the same shape");
  TORCH_CHECK(q.size(0) == v.size(0) && q.size(2) == v.size(2),
              "batch and seq dims of v must match q");
  TORCH_CHECK(v.size(1) == q.size(1) || v.size(1) == 1,
              "v heads must equal q heads or 1 (broadcast)");

  const auto T = q.size(2);
  if (T * T > SCORE_ELEMS_EAGER_OK) {
    return tril_score_v_cpu_tiled(q, k, v);
  }
  return tril_score_v_cpu_eager(q, k, v);
}

// Decode against packed past KR/V: (Q @ K_past.mT) @ V_past
// Q: (B,H,Tq,Dk), K: (B,H,S,Dk), V: (B,H|1,S,Dv). All past keys are valid (j < query).
static torch::Tensor tril_decode_cpu_eager(torch::Tensor q, torch::Tensor k_past,
                                           torch::Tensor v_past) {
  auto qf = maybe_acc(q);
  auto kf = maybe_acc(k_past);
  auto vf = maybe_acc(v_past);
  // scores ephemeral: (B,H,Tq,S) — no expand of V (matmul broadcasts).
  auto scores = at::matmul(qf, kf.transpose(-2, -1));
  auto out = at::matmul(scores, vf);
  return out.to(q.scalar_type());
}

static torch::Tensor tril_decode_cpu_tiled(torch::Tensor q, torch::Tensor k_past,
                                           torch::Tensor v_past) {
  const auto B = q.size(0);
  const auto H = q.size(1);
  const auto Tq = q.size(2);
  const auto S = k_past.size(2);
  const auto Dv = v_past.size(3);

  auto qf = maybe_acc(q);
  auto kf = maybe_acc(k_past);
  auto vf = maybe_acc(v_past);
  auto out = torch::zeros({B, H, Tq, Dv}, qf.options());

  for (int64_t j0 = 0; j0 < S; j0 += DECODE_TILE_N) {
    const int64_t j1 = std::min(j0 + DECODE_TILE_N, S);
    auto Kj = kf.narrow(/*dim=*/2, j0, j1 - j0);
    auto Vj = vf.narrow(/*dim=*/2, j0, j1 - j0);
    out.add_(at::matmul(at::matmul(qf, Kj.transpose(-2, -1)), Vj));
  }
  return out.to(q.scalar_type());
}

torch::Tensor tril_decode_cpu(torch::Tensor q, torch::Tensor k_past,
                              torch::Tensor v_past) {
  TORCH_CHECK(q.dim() == 4 && k_past.dim() == 4 && v_past.dim() == 4,
              "q,k_past,v_past must be 4D");
  TORCH_CHECK(q.size(0) == k_past.size(0) && q.size(1) == k_past.size(1),
              "q and k_past batch/heads must match");
  TORCH_CHECK(q.size(3) == k_past.size(3), "q and k_past Dk must match");
  TORCH_CHECK(v_past.size(0) == q.size(0) && v_past.size(2) == k_past.size(2),
              "v_past batch/seq must match k_past");
  TORCH_CHECK(v_past.size(1) == q.size(1) || v_past.size(1) == 1,
              "v_past heads must equal q heads or 1 (broadcast)");

  const auto B = q.size(0);
  const auto H = q.size(1);
  const auto Tq = q.size(2);
  const auto S = k_past.size(2);
  const auto Dv = v_past.size(3);

  if (S == 0) {
    return torch::zeros({B, H, Tq, Dv}, q.options());
  }

  if (Tq * S > SCORE_ELEMS_EAGER_OK) {
    return tril_decode_cpu_tiled(q, k_past, v_past);
  }
  return tril_decode_cpu_eager(q, k_past, v_past);
}
