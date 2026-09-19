// Strict lower-triangular score×V (tril diagonal=-1, no softmax, no scale).
// out[b,h,i,:] = sum_{j < i} (Q[b,h,i,:] · K[b,h,j,:]) * V[b,h_or_1,j,:]
#pragma once

#include <torch/extension.h>

torch::Tensor tril_score_v_cpu(torch::Tensor q, torch::Tensor k, torch::Tensor v);

#ifdef WITH_CUDA
torch::Tensor tril_score_v_cuda(torch::Tensor q, torch::Tensor k, torch::Tensor v);
#endif

torch::Tensor tril_score_v(torch::Tensor q, torch::Tensor k, torch::Tensor v);
