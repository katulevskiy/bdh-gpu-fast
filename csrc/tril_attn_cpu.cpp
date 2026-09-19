#include "tril_attn.h"

#include <ATen/ATen.h>
#include <cmath>

// Reference C++ path: materialize scores then mask then matmul.
// Matches PyTorch: (Q @ K.mT).tril(diagonal=-1) @ V
torch::Tensor tril_score_v_cpu(torch::Tensor q, torch::Tensor k, torch::Tensor v) {
  TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4,
              "q,k,v must be 4D (B,H,T,D)");
  TORCH_CHECK(q.sizes() == k.sizes(), "q and k must have the same shape");
  TORCH_CHECK(q.size(0) == v.size(0) && q.size(2) == v.size(2),
              "batch and seq dims of v must match q");
  TORCH_CHECK(v.size(1) == q.size(1) || v.size(1) == 1,
              "v heads must equal q heads or 1 (broadcast)");

  auto qf = q.contiguous().to(torch::kFloat32);
  auto kf = k.contiguous().to(torch::kFloat32);
  auto vf = v.contiguous().to(torch::kFloat32);

  // scores: (B, H, T, T)
  auto scores = at::matmul(qf, kf.transpose(-2, -1));
  scores = scores.tril(/*diagonal=*/-1);

  if (vf.size(1) == 1 && qf.size(1) > 1) {
    vf = vf.expand({vf.size(0), qf.size(1), vf.size(2), vf.size(3)});
  }

  auto out = at::matmul(scores, vf);
  return out.to(q.scalar_type());
}

// Decode against packed past KR/V: (Q @ K_past.mT) @ V_past
// Q: (B,H,Tq,Dk), K: (B,H,S,Dk), V: (B,H|1,S,Dv). All past keys are valid (j < query).
torch::Tensor tril_decode_cpu(torch::Tensor q, torch::Tensor k_past, torch::Tensor v_past) {
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

  auto qf = q.contiguous().to(torch::kFloat32);
  auto kf = k_past.contiguous().to(torch::kFloat32);
  auto vf = v_past.contiguous().to(torch::kFloat32);

  if (vf.size(1) == 1 && qf.size(1) > 1) {
    vf = vf.expand({vf.size(0), qf.size(1), vf.size(2), vf.size(3)});
  }

  // scores: (B, H, Tq, S) — every past key is strictly earlier than the query
  auto scores = at::matmul(qf, kf.transpose(-2, -1));
  auto out = at::matmul(scores, vf);
  return out.to(q.scalar_type());
}
