#include <cuda_runtime.h>

namespace {

__global__ void write_matmul_alpha_beta_from_global_scales_kernel(
    const float* global_a,
    const float* global_b,
    float* alpha_beta) {
  alpha_beta[0] = global_a[0] * global_b[0];
  alpha_beta[1] = 0.0f;
}

}  // namespace

cudaError_t write_matmul_alpha_beta_from_global_scales(
    const float* global_a,
    const float* global_b,
    float* alpha_beta,
    cudaStream_t stream) {
  write_matmul_alpha_beta_from_global_scales_kernel<<<1, 1, 0, stream>>>(
      global_a, global_b, alpha_beta);
  return cudaGetLastError();
}
