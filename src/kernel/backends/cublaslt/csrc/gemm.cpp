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
        "scaled_mm_mxfp8: cublasLtCreate failed");
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
      "scaled_mm_mxfp8: failed to set matrix layout order");
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
      "scaled_mm_mxfp8: failed to set ",
      matrix,
      " pointer alignment");
}

int64_t round_up(int64_t value, int64_t multiple) {
  return ((value + multiple - 1) / multiple) * multiple;
}

int64_t required_scale_elements(int64_t rows, int64_t k) {
  const int64_t blocks = (k + 31) / 32;
  return round_up(rows, 128) * round_up(blocks, 4);
}

void check_scaled_mm_mxfp8_meta_inputs(
    const at::Tensor& a,
    const at::Tensor& b,
    const at::Tensor& scale_a,
    const at::Tensor& scale_b,
    const std::optional<at::Tensor>& bias) {
  TORCH_CHECK(a.dim() == 2, "scaled_mm_mxfp8: A must be rank 2");
  TORCH_CHECK(b.dim() == 2, "scaled_mm_mxfp8: B must be rank 2");
  TORCH_CHECK(
      a.size(1) == b.size(0),
      "scaled_mm_mxfp8: A and B contraction dimensions must match");
  TORCH_CHECK(
      a.size(1) % 16 == 0 && b.size(1) % 16 == 0,
      "scaled_mm_mxfp8: K and N must be multiples of 16");
  TORCH_CHECK(
      a.scalar_type() == at::ScalarType::Float8_e4m3fn,
      "scaled_mm_mxfp8: A must have float8_e4m3fn dtype");
  TORCH_CHECK(
      b.scalar_type() == at::ScalarType::Float8_e4m3fn,
      "scaled_mm_mxfp8: B must have float8_e4m3fn dtype");
  TORCH_CHECK(
      scale_a.scalar_type() == at::ScalarType::Float8_e8m0fnu,
      "scaled_mm_mxfp8: A scale must have float8_e8m0fnu dtype");
  TORCH_CHECK(
      scale_b.scalar_type() == at::ScalarType::Float8_e8m0fnu,
      "scaled_mm_mxfp8: B scale must have float8_e8m0fnu dtype");

  const int64_t M = a.size(0);
  const int64_t K = a.size(1);
  const int64_t N = b.size(1);
  TORCH_CHECK(
      scale_a.dim() == 1 && scale_a.is_contiguous(),
      "scaled_mm_mxfp8: A scale must be a contiguous rank-1 swizzled buffer");
  TORCH_CHECK(
      scale_b.dim() == 1 && scale_b.is_contiguous(),
      "scaled_mm_mxfp8: B scale must be a contiguous rank-1 swizzled buffer");
  TORCH_CHECK(
      scale_a.numel() >= required_scale_elements(M, K),
      "scaled_mm_mxfp8: A scale buffer is too small for the swizzled layout");
  TORCH_CHECK(
      scale_b.numel() >= required_scale_elements(N, K),
      "scaled_mm_mxfp8: B scale buffer is too small for the swizzled layout");
  if (bias.has_value()) {
    TORCH_CHECK(
        bias->dim() == 1 && bias->size(0) == N,
        "scaled_mm_mxfp8: bias must have shape (N,)");
    TORCH_CHECK(
        bias->scalar_type() == at::ScalarType::BFloat16,
        "scaled_mm_mxfp8: bias must have bfloat16 dtype");
    TORCH_CHECK(
        bias->is_contiguous(), "scaled_mm_mxfp8: bias must be contiguous");
  }
}

void check_scaled_mm_mxfp8_cuda_inputs(
    const at::Tensor& a,
    const at::Tensor& b,
    const at::Tensor& scale_a,
    const at::Tensor& scale_b,
    const std::optional<at::Tensor>& bias) {
  check_scaled_mm_mxfp8_meta_inputs(a, b, scale_a, scale_b, bias);

  TORCH_CHECK(a.is_cuda(), "scaled_mm_mxfp8: A must be a CUDA tensor");
  TORCH_CHECK(b.is_cuda(), "scaled_mm_mxfp8: B must be a CUDA tensor");
  TORCH_CHECK(
      scale_a.is_cuda(), "scaled_mm_mxfp8: A scale must be a CUDA tensor");
  TORCH_CHECK(
      scale_b.is_cuda(), "scaled_mm_mxfp8: B scale must be a CUDA tensor");
  TORCH_CHECK(
      a.device() == b.device() && a.device() == scale_a.device() &&
          a.device() == scale_b.device(),
      "scaled_mm_mxfp8: A, B, and scales must be on the same CUDA device");
  const auto* properties = at::cuda::getDeviceProperties(a.get_device());
  TORCH_CHECK(
      properties->major >= 10,
      "scaled_mm_mxfp8: requires SM100 or newer, got SM",
      properties->major,
      properties->minor);
  if (bias.has_value()) {
    TORCH_CHECK(
        bias->is_cuda(), "scaled_mm_mxfp8: bias must be a CUDA tensor");
    TORCH_CHECK(
        bias->device() == a.device(),
        "scaled_mm_mxfp8: bias must be on the same CUDA device as A");
  }

  const int64_t M = a.size(0);
  const int64_t K = a.size(1);
  const int64_t N = b.size(1);
  if (M != 0 && N != 0 && K != 0) {
    TORCH_CHECK(
        a.stride(0) == K && a.stride(1) == 1,
        "scaled_mm_mxfp8: A must be row-major contiguous");
    TORCH_CHECK(
        b.stride(0) == 1 && b.stride(1) == K,
        "scaled_mm_mxfp8: B must be column-major contiguous");
  }
}

at::Tensor scaled_mm_mxfp8_cublaslt_meta(
    const at::Tensor& a,
    const at::Tensor& b,
    const at::Tensor& scale_a,
    const at::Tensor& scale_b,
    const std::optional<at::Tensor>& bias) {
  check_scaled_mm_mxfp8_meta_inputs(a, b, scale_a, scale_b, bias);
  return at::empty_symint(
      {a.sym_size(0), b.sym_size(1)}, a.options().dtype(at::kBFloat16));
}

at::Tensor scaled_mm_mxfp8_cublaslt_cuda(
    const at::Tensor& a,
    const at::Tensor& b,
    const at::Tensor& scale_a,
    const at::Tensor& scale_b,
    const std::optional<at::Tensor>& bias) {
  check_scaled_mm_mxfp8_cuda_inputs(a, b, scale_a, scale_b, bias);
  c10::cuda::CUDAGuard device_guard(a.device());

  const int64_t M = a.size(0);
  const int64_t K = a.size(1);
  const int64_t N = b.size(1);

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

  cublasLtHandle_t handle = get_cublas_lt_handle(a.device().index());
  const auto stream = at::cuda::getCurrentCUDAStream(a.device().index());
  cublasStatus_t status = CUBLAS_STATUS_SUCCESS;

  CublasLtMatmulDesc operation;
  status = cublasLtMatmulDescCreate(
      operation.address(), CUBLAS_COMPUTE_32F, CUDA_R_32F);
  TORCH_CHECK(
      status == CUBLAS_STATUS_SUCCESS,
      "scaled_mm_mxfp8: cublasLtMatmulDescCreate failed");

  const cublasOperation_t transpose = CUBLAS_OP_N;
  status = cublasLtMatmulDescSetAttribute(
      operation.get(),
      CUBLASLT_MATMUL_DESC_TRANSA,
      &transpose,
      sizeof(transpose));
  TORCH_CHECK(
      status == CUBLAS_STATUS_SUCCESS,
      "scaled_mm_mxfp8: failed to set TRANSA");
  status = cublasLtMatmulDescSetAttribute(
      operation.get(),
      CUBLASLT_MATMUL_DESC_TRANSB,
      &transpose,
      sizeof(transpose));
  TORCH_CHECK(
      status == CUBLAS_STATUS_SUCCESS,
      "scaled_mm_mxfp8: failed to set TRANSB");

  const cublasLtMatmulMatrixScale_t scale_mode =
      CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0;
  status = cublasLtMatmulDescSetAttribute(
      operation.get(),
      CUBLASLT_MATMUL_DESC_A_SCALE_MODE,
      &scale_mode,
      sizeof(scale_mode));
  TORCH_CHECK(
      status == CUBLAS_STATUS_SUCCESS,
      "scaled_mm_mxfp8: failed to set A scale mode");
  status = cublasLtMatmulDescSetAttribute(
      operation.get(),
      CUBLASLT_MATMUL_DESC_B_SCALE_MODE,
      &scale_mode,
      sizeof(scale_mode));
  TORCH_CHECK(
      status == CUBLAS_STATUS_SUCCESS,
      "scaled_mm_mxfp8: failed to set B scale mode");
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
      "scaled_mm_mxfp8: failed to set A scale pointer");
  status = cublasLtMatmulDescSetAttribute(
      operation.get(),
      CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
      &scale_b_pointer,
      sizeof(scale_b_pointer));
  TORCH_CHECK(
      status == CUBLAS_STATUS_SUCCESS,
      "scaled_mm_mxfp8: failed to set B scale pointer");

  if (bias.has_value()) {
    const cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_BIAS;
    status = cublasLtMatmulDescSetAttribute(
        operation.get(),
        CUBLASLT_MATMUL_DESC_EPILOGUE,
        &epilogue,
        sizeof(epilogue));
    TORCH_CHECK(
        status == CUBLAS_STATUS_SUCCESS,
        "scaled_mm_mxfp8: failed to set bias epilogue");
    const void* bias_pointer = bias->data_ptr();
    status = cublasLtMatmulDescSetAttribute(
        operation.get(),
        CUBLASLT_MATMUL_DESC_BIAS_POINTER,
        &bias_pointer,
        sizeof(bias_pointer));
    TORCH_CHECK(
        status == CUBLAS_STATUS_SUCCESS,
        "scaled_mm_mxfp8: failed to set bias pointer");
  }

  CublasLtMatrixLayout a_layout;
  CublasLtMatrixLayout b_layout;
  CublasLtMatrixLayout c_layout;
  CublasLtMatrixLayout d_layout;
  status = cublasLtMatrixLayoutCreate(
      a_layout.address(), CUDA_R_8F_E4M3, N, K, aligned_b.stride(1));
  TORCH_CHECK(
      status == CUBLAS_STATUS_SUCCESS,
      "scaled_mm_mxfp8: failed to create A matrix layout");
  set_matrix_order(a_layout.get(), CUBLASLT_ORDER_ROW);
  status = cublasLtMatrixLayoutCreate(
      b_layout.address(), CUDA_R_8F_E4M3, K, M, aligned_a.stride(0));
  TORCH_CHECK(
      status == CUBLAS_STATUS_SUCCESS,
      "scaled_mm_mxfp8: failed to create B matrix layout");
  set_matrix_order(b_layout.get(), CUBLASLT_ORDER_COL);
  status = cublasLtMatrixLayoutCreate(
      c_layout.address(),
      CUDA_R_16BF,
      N,
      M,
      out.stride(0));
  TORCH_CHECK(
      status == CUBLAS_STATUS_SUCCESS,
      "scaled_mm_mxfp8: failed to create C matrix layout");
  set_matrix_order(c_layout.get(), CUBLASLT_ORDER_COL);
  status = cublasLtMatrixLayoutCreate(
      d_layout.address(),
      CUDA_R_16BF,
      N,
      M,
      out.stride(0));
  TORCH_CHECK(
      status == CUBLAS_STATUS_SUCCESS,
      "scaled_mm_mxfp8: failed to create D matrix layout");
  set_matrix_order(d_layout.get(), CUBLASLT_ORDER_COL);

  const float alpha = 1.0f;
  const float beta = 0.0f;

  CublasLtMatmulPreference preference;
  status = cublasLtMatmulPreferenceCreate(preference.address());
  TORCH_CHECK(
      status == CUBLAS_STATUS_SUCCESS,
      "scaled_mm_mxfp8: failed to create matmul preference");

  size_t max_workspace = 1ull << 22;
  status = cublasLtMatmulPreferenceSetAttribute(
      preference.get(),
      CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
      &max_workspace,
      sizeof(max_workspace));
  TORCH_CHECK(
      status == CUBLAS_STATUS_SUCCESS,
      "scaled_mm_mxfp8: failed to set workspace limit");
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
      "scaled_mm_mxfp8: cublasLtMatmulAlgoGetHeuristic failed");
  TORCH_CHECK(
      returned_algorithms > 0,
      "scaled_mm_mxfp8: no cuBLASLt algorithms available");

  const auto& selected = heuristics[0];
  TORCH_CHECK(
      selected.state == CUBLAS_STATUS_SUCCESS,
      "scaled_mm_mxfp8: first cuBLASLt heuristic is invalid");
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
      "scaled_mm_mxfp8: cublasLtMatmul failed");
  return out;
}

}  // namespace

TORCH_LIBRARY_FRAGMENT(aot_kernel, m) {
  m.def(
      "_scaled_mm_mxfp8_cublaslt(Tensor a, Tensor b, Tensor scale_a, "
      "Tensor scale_b, Tensor? bias=None) -> Tensor");
}

TORCH_LIBRARY_IMPL(aot_kernel, Meta, m) {
  m.impl(
      "_scaled_mm_mxfp8_cublaslt",
      TORCH_FN(scaled_mm_mxfp8_cublaslt_meta));
}

TORCH_LIBRARY_IMPL(aot_kernel, CUDA, m) {
  m.impl(
      "_scaled_mm_mxfp8_cublaslt",
      TORCH_FN(scaled_mm_mxfp8_cublaslt_cuda));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {}
