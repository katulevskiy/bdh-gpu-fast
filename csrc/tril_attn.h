// Strict lower-triangular score×V (tril diagonal=-1, no softmax, no scale).
// Full: out[b,h,i,:] = sum_{j < i} (Q[b,h,i,:] · K[b,h,j,:]) * V[b,h_or_1,j,:]
// Decode: out[b,h,i,:] = sum_{j < S} (Q[b,h,i,:] · K_past[b,h,j,:]) * V_past[...]
//   (K/V are packed past only — new token never attends to itself.)
#pragma once

#include <torch/extension.h>

torch::Tensor tril_score_v_cpu(torch::Tensor q, torch::Tensor k, torch::Tensor v);
torch::Tensor tril_decode_cpu(torch::Tensor q, torch::Tensor k_past, torch::Tensor v_past);

#ifdef WITH_CUDA
torch::Tensor tril_score_v_cuda(torch::Tensor q, torch::Tensor k, torch::Tensor v);
torch::Tensor tril_decode_cuda(torch::Tensor q, torch::Tensor k_past, torch::Tensor v_past);
#endif

torch::Tensor tril_score_v(torch::Tensor q, torch::Tensor k, torch::Tensor v);
torch::Tensor tril_decode(torch::Tensor q, torch::Tensor k_past, torch::Tensor v_past);
