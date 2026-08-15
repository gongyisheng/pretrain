#include <torch/extension.h>

namespace {

at::Tensor scaled_gemm_mxfp8_meta(
    const at::Tensor& a,
    const at::Tensor& b,
    const at::Tensor& scale_a,
    const at::Tensor& scale_b,
    const std::optional<at::Tensor>& bias) {
  static_cast<void>(scale_a);
  static_cast<void>(scale_b);
  static_cast<void>(bias);
  return at::empty_symint(
      {a.sym_size(0), b.sym_size(1)}, a.options().dtype(at::kBFloat16));
}

at::Tensor scaled_gemm_mxfp8_cuda(
    const at::Tensor& a,
    const at::Tensor& b,
    const at::Tensor& scale_a,
    const at::Tensor& scale_b,
    const std::optional<at::Tensor>& bias) {
  static_cast<void>(a);
  static_cast<void>(b);
  static_cast<void>(scale_a);
  static_cast<void>(scale_b);
  static_cast<void>(bias);
  TORCH_CHECK(false, "scaled_gemm_mxfp8: CUDA implementation not complete");
}

}  // namespace

TORCH_LIBRARY_FRAGMENT(aot_kernel, m) {
  m.def(
      "scaled_gemm_mxfp8(Tensor a, Tensor b, Tensor scale_a, "
      "Tensor scale_b, Tensor? bias=None) -> Tensor");
}

TORCH_LIBRARY_IMPL(aot_kernel, Meta, m) {
  m.impl("scaled_gemm_mxfp8", TORCH_FN(scaled_gemm_mxfp8_meta));
}

TORCH_LIBRARY_IMPL(aot_kernel, CUDA, m) {
  m.impl("scaled_gemm_mxfp8", TORCH_FN(scaled_gemm_mxfp8_cuda));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {}
