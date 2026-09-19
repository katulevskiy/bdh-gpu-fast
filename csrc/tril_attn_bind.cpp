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

torch::Tensor tril_decode(torch::Tensor q, torch::Tensor k_past, torch::Tensor v_past) {
  if (q.is_cuda()) {
#ifdef WITH_CUDA
    return tril_decode_cuda(q, k_past, v_past);
#else
    TORCH_CHECK(false, "bdh_cuda_ext built without CUDA");
#endif
  }
  return tril_decode_cpu(q, k_past, v_past);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "BDH strict lower-triangular score×V + decode (no softmax)";
  m.def("tril_score_v", &tril_score_v,
        "out = tril(Q@K.T, diagonal=-1) @ V  (no softmax, no scale)");
  m.def("tril_score_v_cpu", &tril_score_v_cpu, "CPU reference path");
  m.def("tril_decode", &tril_decode,
        "out = (Q @ K_past.T) @ V_past  (packed past; no self-attend)");
  m.def("tril_decode_cpu", &tril_decode_cpu, "CPU decode reference");
#ifdef WITH_CUDA
  m.def("tril_score_v_cuda", &tril_score_v_cuda, "CUDA kernel path");
  m.def("tril_decode_cuda", &tril_decode_cuda, "CUDA decode kernel path");
  m.attr("has_cuda") = true;
#else
  m.attr("has_cuda") = false;
#endif
}
