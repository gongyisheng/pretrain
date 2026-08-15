#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cublasLt.h>
#include <cstdint>
#include <optional>
#include <unordered_map>
#include <vector>

namespace {

struct CublasLtHandle {
  cublasLtHandle_t handle{nullptr};

  CublasLtHandle() {
    const auto status = cublasLtCreate(&handle);
    TORCH_CHECK(
        status == CUBLAS_STATUS_SUCCESS,
        "scaled_gemm_mxfp8: cublasLtCreate failed");
  }

  ~CublasLtHandle() {
    cublasLtDestroy(handle);
  }
};

template <typename T, cublasStatus_t (*Destroy)(T)>
struct CublasLtResource {
  T value{nullptr};

  CublasLtResource() = default;
  CublasLtResource(const CublasLtResource&) = delete;
  CublasLtResource& operator=(const CublasLtResource&) = delete;

  ~CublasLtResource() {
    if (value != nullptr) {
      Destroy(value);
    }
  }

  T get() const {
    return value;
  }

  T* address() {
    return &value;
  }
};

using CublasLtMatmulDesc =
    CublasLtResource<cublasLtMatmulDesc_t, cublasLtMatmulDescDestroy>;
using CublasLtMatrixLayout =
    CublasLtResource<cublasLtMatrixLayout_t, cublasLtMatrixLayoutDestroy>;
using CublasLtMatmulPreference =
    CublasLtResource<cublasLtMatmulPreference_t, cublasLtMatmulPreferenceDestroy>;

cublasLtHandle_t get_cublas_lt_handle(int device_index) {
  static thread_local std::unordered_map<int, CublasLtHandle> handles;
  return handles.try_emplace(device_index).first->second.handle;
}

void set_matrix_order(cublasLtMatrixLayout_t layout, cublasLtOrder_t order) {
  const auto status = cublasLtMatrixLayoutSetAttribute(
      layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order));
  TORCH_CHECK(
      status == CUBLAS_STATUS_SUCCESS,
      "scaled_gemm_mxfp8: failed to set matrix layout order");
}

int64_t mxfp8_scale_numel(int64_t rows, int64_t inner) {
  const int64_t rounded_rows = (rows + 127) / 128 * 128;
  const int64_t blocks = (inner + 31) / 32;
  const int64_t rounded_blocks = (blocks + 3) / 4 * 4;
  return rounded_rows * rounded_blocks;
}

uint32_t pointer_alignment(const void* pointer) {
  const auto address = reinterpret_cast<uintptr_t>(pointer);
  uint32_t alignment = 256;
  while (address % alignment != 0) {
    alignment /= 2;
  }
  return alignment;
}

void set_pointer_alignment(
    cublasLtMatmulPreference_t preference,
    cublasLtMatmulPreferenceAttributes_t attribute,
    const void* pointer,
    const char* matrix) {
  const uint32_t alignment = pointer_alignment(pointer);
  const auto status = cublasLtMatmulPreferenceSetAttribute(
      preference, attribute, &alignment, sizeof(alignment));
  TORCH_CHECK(
      status == CUBLAS_STATUS_SUCCESS,
      "scaled_gemm_mxfp8: failed to set ",
      matrix,
      " pointer alignment");
}

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
  TORCH_CHECK(
      a.is_cuda() && b.is_cuda() && scale_a.is_cuda() && scale_b.is_cuda() &&
          (!bias.has_value() || bias->is_cuda()),
      "scaled_gemm_mxfp8: operands, scales, and bias must be CUDA tensors");
  c10::cuda::CUDAGuard device_guard(a.device());
  TORCH_CHECK(
      a.device() == b.device() && a.device() == scale_a.device() &&
          a.device() == scale_b.device() &&
          (!bias.has_value() || a.device() == bias->device()),
      "scaled_gemm_mxfp8: operands, scales, and bias must be on the same device");
  TORCH_CHECK(
      a.dim() == 2 && b.dim() == 2,
      "scaled_gemm_mxfp8: A and B must be 2D");
  TORCH_CHECK(
      a.size(1) == b.size(0),
      "scaled_gemm_mxfp8: mismatched K dimension");
  TORCH_CHECK(
      a.scalar_type() == at::ScalarType::Float8_e4m3fn &&
          b.scalar_type() == at::ScalarType::Float8_e4m3fn,
      "scaled_gemm_mxfp8: operands must use float8_e4m3fn");

  const int64_t M = a.size(0);
  const int64_t K = a.size(1);
  const int64_t N = b.size(1);
  TORCH_CHECK(
      a.numel() == 0 || (a.stride(0) == K && a.stride(1) == 1),
      "scaled_gemm_mxfp8: A must be row-major");
  TORCH_CHECK(
      b.numel() == 0 || (b.stride(0) == 1 && b.stride(1) == K),
      "scaled_gemm_mxfp8: B must be column-major");
  TORCH_CHECK(
      K % 16 == 0 && N % 16 == 0,
      "scaled_gemm_mxfp8: K and N must be divisible by 16");
  TORCH_CHECK(
      scale_a.scalar_type() == at::ScalarType::Float8_e8m0fnu &&
          scale_b.scalar_type() == at::ScalarType::Float8_e8m0fnu,
      "scaled_gemm_mxfp8: scales must use float8_e8m0fnu");
  TORCH_CHECK(
      scale_a.dim() == 1 && scale_b.dim() == 1,
      "scaled_gemm_mxfp8: scales must be flat");
  TORCH_CHECK(
      scale_a.is_contiguous() && scale_b.is_contiguous(),
      "scaled_gemm_mxfp8: scales must be contiguous");
  TORCH_CHECK(
      (scale_a.numel() == 0 ||
       reinterpret_cast<uintptr_t>(scale_a.data_ptr()) % 16 == 0) &&
          (scale_b.numel() == 0 ||
           reinterpret_cast<uintptr_t>(scale_b.data_ptr()) % 16 == 0),
      "scaled_gemm_mxfp8: scales must be 16-byte aligned");
  const int64_t expected_a = mxfp8_scale_numel(M, K);
  const int64_t expected_b = mxfp8_scale_numel(N, K);
  TORCH_CHECK(
      scale_a.numel() == expected_a && scale_b.numel() == expected_b,
      "scaled_gemm_mxfp8: invalid MXFP8 scale storage: expected (",
      expected_a,
      ", ",
      expected_b,
      "), got (",
      scale_a.numel(),
      ", ",
      scale_b.numel(),
      ")");
  if (bias.has_value()) {
    TORCH_CHECK(
        bias->scalar_type() == at::ScalarType::BFloat16,
        "scaled_gemm_mxfp8: bias must use bfloat16");
    TORCH_CHECK(
        bias->dim() == 1 && bias->size(0) == N,
        "scaled_gemm_mxfp8: bias must have shape (N,)");
    TORCH_CHECK(
        bias->is_contiguous(),
        "scaled_gemm_mxfp8: bias must be contiguous");
  }

  cudaDeviceProp properties;
  const cudaError_t properties_status =
      cudaGetDeviceProperties(&properties, a.device().index());
  TORCH_CHECK(
      properties_status == cudaSuccess,
      "scaled_gemm_mxfp8: failed to get CUDA device properties");
  TORCH_CHECK(
      properties.major >= 10,
      "scaled_gemm_mxfp8: requires compute capability 10.0 or newer");

  if (M == 0 || N == 0 || K == 0) {
    at::Tensor out =
        at::zeros({M, N}, a.options().dtype(at::ScalarType::BFloat16));
    if (bias.has_value()) {
      out.add_(*bias);
    }
    return out;
  }

  at::Tensor aligned_a = a;
  if (reinterpret_cast<uintptr_t>(a.data_ptr()) % 16 != 0) {
    aligned_a = a.clone(at::MemoryFormat::Contiguous);
  }
  at::Tensor aligned_b = b;
  if (reinterpret_cast<uintptr_t>(b.data_ptr()) % 16 != 0) {
    aligned_b =
        b.transpose(0, 1)
            .clone(at::MemoryFormat::Contiguous)
            .transpose(0, 1);
  }

  at::Tensor out =
      at::empty({M, N}, a.options().dtype(at::ScalarType::BFloat16));
  TORCH_CHECK(
      reinterpret_cast<uintptr_t>(out.data_ptr()) % 16 == 0,
      "scaled_gemm_mxfp8: output must be 16-byte aligned");

  cublasLtHandle_t handle = get_cublas_lt_handle(a.device().index());
  const auto stream = at::cuda::getCurrentCUDAStream(a.device().index());
  cublasStatus_t status = CUBLAS_STATUS_SUCCESS;

  CublasLtMatmulDesc operation;
  status = cublasLtMatmulDescCreate(
      operation.address(), CUBLAS_COMPUTE_32F, CUDA_R_32F);
  TORCH_CHECK(
      status == CUBLAS_STATUS_SUCCESS,
      "scaled_gemm_mxfp8: cublasLtMatmulDescCreate failed");

  const cublasOperation_t transpose = CUBLAS_OP_N;
  status = cublasLtMatmulDescSetAttribute(
      operation.get(),
      CUBLASLT_MATMUL_DESC_TRANSA,
      &transpose,
      sizeof(transpose));
  TORCH_CHECK(
      status == CUBLAS_STATUS_SUCCESS,
      "scaled_gemm_mxfp8: failed to set TRANSA");
  status = cublasLtMatmulDescSetAttribute(
      operation.get(),
      CUBLASLT_MATMUL_DESC_TRANSB,
      &transpose,
      sizeof(transpose));
  TORCH_CHECK(
      status == CUBLAS_STATUS_SUCCESS,
      "scaled_gemm_mxfp8: failed to set TRANSB");

  const cublasLtMatmulMatrixScale_t scale_mode =
      CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0;
  status = cublasLtMatmulDescSetAttribute(
      operation.get(),
      CUBLASLT_MATMUL_DESC_A_SCALE_MODE,
      &scale_mode,
      sizeof(scale_mode));
  TORCH_CHECK(
      status == CUBLAS_STATUS_SUCCESS,
      "scaled_gemm_mxfp8: failed to set A scale mode");
  status = cublasLtMatmulDescSetAttribute(
      operation.get(),
      CUBLASLT_MATMUL_DESC_B_SCALE_MODE,
      &scale_mode,
      sizeof(scale_mode));
  TORCH_CHECK(
      status == CUBLAS_STATUS_SUCCESS,
      "scaled_gemm_mxfp8: failed to set B scale mode");
  // View row-major output as column-major D^T and compute B^T A^T.
  const void* scale_a_pointer = scale_b.data_ptr();
  const void* scale_b_pointer = scale_a.data_ptr();
  status = cublasLtMatmulDescSetAttribute(
      operation.get(),
      CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
      &scale_a_pointer,
      sizeof(scale_a_pointer));
  TORCH_CHECK(
      status == CUBLAS_STATUS_SUCCESS,
      "scaled_gemm_mxfp8: failed to set A scale pointer");
  status = cublasLtMatmulDescSetAttribute(
      operation.get(),
      CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
      &scale_b_pointer,
      sizeof(scale_b_pointer));
  TORCH_CHECK(
      status == CUBLAS_STATUS_SUCCESS,
      "scaled_gemm_mxfp8: failed to set B scale pointer");

  if (bias.has_value()) {
    const cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_BIAS;
    status = cublasLtMatmulDescSetAttribute(
        operation.get(),
        CUBLASLT_MATMUL_DESC_EPILOGUE,
        &epilogue,
        sizeof(epilogue));
    TORCH_CHECK(
        status == CUBLAS_STATUS_SUCCESS,
        "scaled_gemm_mxfp8: failed to set bias epilogue");
    const void* bias_pointer = bias->data_ptr();
    status = cublasLtMatmulDescSetAttribute(
        operation.get(),
        CUBLASLT_MATMUL_DESC_BIAS_POINTER,
        &bias_pointer,
        sizeof(bias_pointer));
    TORCH_CHECK(
        status == CUBLAS_STATUS_SUCCESS,
        "scaled_gemm_mxfp8: failed to set bias pointer");
  }

  CublasLtMatrixLayout a_layout;
  CublasLtMatrixLayout b_layout;
  CublasLtMatrixLayout c_layout;
  CublasLtMatrixLayout d_layout;
  status = cublasLtMatrixLayoutCreate(
      a_layout.address(), CUDA_R_8F_E4M3, N, K, aligned_b.stride(1));
  TORCH_CHECK(
      status == CUBLAS_STATUS_SUCCESS,
      "scaled_gemm_mxfp8: failed to create A matrix layout");
  set_matrix_order(a_layout.get(), CUBLASLT_ORDER_ROW);
  status = cublasLtMatrixLayoutCreate(
      b_layout.address(), CUDA_R_8F_E4M3, K, M, aligned_a.stride(0));
  TORCH_CHECK(
      status == CUBLAS_STATUS_SUCCESS,
      "scaled_gemm_mxfp8: failed to create B matrix layout");
  set_matrix_order(b_layout.get(), CUBLASLT_ORDER_COL);
  status = cublasLtMatrixLayoutCreate(
      c_layout.address(),
      CUDA_R_16BF,
      N,
      M,
      out.stride(0));
  TORCH_CHECK(
      status == CUBLAS_STATUS_SUCCESS,
      "scaled_gemm_mxfp8: failed to create C matrix layout");
  set_matrix_order(c_layout.get(), CUBLASLT_ORDER_COL);
  status = cublasLtMatrixLayoutCreate(
      d_layout.address(),
      CUDA_R_16BF,
      N,
      M,
      out.stride(0));
  TORCH_CHECK(
      status == CUBLAS_STATUS_SUCCESS,
      "scaled_gemm_mxfp8: failed to create D matrix layout");
  set_matrix_order(d_layout.get(), CUBLASLT_ORDER_COL);

  const float alpha = 1.0f;
  const float beta = 0.0f;

  CublasLtMatmulPreference preference;
  status = cublasLtMatmulPreferenceCreate(preference.address());
  TORCH_CHECK(
      status == CUBLAS_STATUS_SUCCESS,
      "scaled_gemm_mxfp8: failed to create matmul preference");

  size_t max_workspace = 1ull << 22;
  status = cublasLtMatmulPreferenceSetAttribute(
      preference.get(),
      CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
      &max_workspace,
      sizeof(max_workspace));
  TORCH_CHECK(
      status == CUBLAS_STATUS_SUCCESS,
      "scaled_gemm_mxfp8: failed to set workspace limit");
  set_pointer_alignment(
      preference.get(),
      CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES,
      aligned_b.data_ptr(),
      "A");
  set_pointer_alignment(
      preference.get(),
      CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES,
      aligned_a.data_ptr(),
      "B");
  set_pointer_alignment(
      preference.get(),
      CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES,
      out.data_ptr(),
      "C");
  set_pointer_alignment(
      preference.get(),
      CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES,
      out.data_ptr(),
      "D");

  constexpr int heuristic_capacity = 4;
  std::vector<cublasLtMatmulHeuristicResult_t> heuristics(heuristic_capacity);
  int returned_algorithms = 0;
  status = cublasLtMatmulAlgoGetHeuristic(
      handle,
      operation.get(),
      a_layout.get(),
      b_layout.get(),
      c_layout.get(),
      d_layout.get(),
      preference.get(),
      heuristic_capacity,
      heuristics.data(),
      &returned_algorithms);
  TORCH_CHECK(
      status == CUBLAS_STATUS_SUCCESS,
      "scaled_gemm_mxfp8: cublasLtMatmulAlgoGetHeuristic failed");
  TORCH_CHECK(
      returned_algorithms > 0,
      "scaled_gemm_mxfp8: no cuBLASLt algorithms available");

  const auto& selected = heuristics[0];
  TORCH_CHECK(
      selected.state == CUBLAS_STATUS_SUCCESS,
      "scaled_gemm_mxfp8: first cuBLASLt heuristic is invalid");
  const int64_t workspace_bytes =
      static_cast<int64_t>(selected.workspaceSize);
  at::Tensor workspace = at::empty(
      {workspace_bytes}, a.options().dtype(at::ScalarType::Byte));
  void* workspace_pointer =
      workspace_bytes > 0 ? workspace.data_ptr() : nullptr;

  status = cublasLtMatmul(
      handle,
      operation.get(),
      &alpha,
      aligned_b.data_ptr(),
      a_layout.get(),
      aligned_a.data_ptr(),
      b_layout.get(),
      &beta,
      out.data_ptr(),
      c_layout.get(),
      out.data_ptr(),
      d_layout.get(),
      &selected.algo,
      workspace_pointer,
      workspace_bytes,
      stream.stream());
  TORCH_CHECK(
      status == CUBLAS_STATUS_SUCCESS,
      "scaled_gemm_mxfp8: cublasLtMatmul failed");
  return out;
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
