#include "tril_attn.h"

torch::Tensor tril_score_v(torch::Tensor q, torch::Tensor k, torch::Tensor v) {
  if (q.is_cuda()) {
#ifdef WITH_CUDA
    return tril_score_v_cuda(q, k, v);
#else
    TORCH_CHECK(false, "bdh_cuda_ext built without CUDA");
#endif
  }
  return tril_score_v_cpu(q, k, v);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "BDH strict lower-triangular score×V (no softmax)";
  m.def("tril_score_v", &tril_score_v,
        "out = tril(Q@K.T, diagonal=-1) @ V  (no softmax, no scale)");
  m.def("tril_score_v_cpu", &tril_score_v_cpu, "CPU reference path");
#ifdef WITH_CUDA
  m.def("tril_score_v_cuda", &tril_score_v_cuda, "CUDA kernel path");
  m.attr("has_cuda") = true;
#else
  m.attr("has_cuda") = false;
#endif
}
