#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cublasLt.h>
#include "mxfp8_grouped.cuh"

#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <limits>
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

void check_grouped_tensor(
    const at::Tensor& tensor,
    const char* name,
    at::ScalarType dtype,
    int64_t dimensions,
    const at::Device& device) {
  TORCH_CHECK(tensor.is_cuda(), "scaled_grouped_gemm_mxfp8: ", name, " must be CUDA");
  TORCH_CHECK(
      tensor.device() == device,
      "scaled_grouped_gemm_mxfp8: ",
      name,
      " must be on ",
      device,
      ", got ",
      tensor.device());
  TORCH_CHECK(
      tensor.scalar_type() == dtype,
      "scaled_grouped_gemm_mxfp8: ",
      name,
      " has dtype ",
      tensor.scalar_type(),
      ", expected ",
      dtype);
  TORCH_CHECK(
      tensor.dim() == dimensions,
      "scaled_grouped_gemm_mxfp8: ",
      name,
      " must be ",
      dimensions,
      "D, got ",
      tensor.dim(),
      "D");
}

void check_16_byte_aligned(const at::Tensor& tensor, const char* name) {
  TORCH_CHECK(
      reinterpret_cast<uintptr_t>(tensor.data_ptr()) % 16 == 0,
      "scaled_grouped_gemm_mxfp8: ",
      name,
      " must be 16-byte aligned");
}

void check_offset_buffer(const at::Tensor& offsets) {
  const size_t required_bytes =
      static_cast<size_t>(offsets.numel()) * sizeof(int32_t);
  TORCH_CHECK(
      offsets.storage_offset() >= 0 &&
          static_cast<size_t>(offsets.storage_offset()) * sizeof(int32_t) +
                  required_bytes <=
              offsets.storage().nbytes(),
      "scaled_grouped_gemm_mxfp8: offsets storage is smaller than its logical buffer");
}

void set_grouped_dimension_width(cublasLtMatrixLayout_t layout) {
  const cublasLtIntegerWidth_t width = CUBLASLT_INTEGER_WIDTH_64;
  auto status = cublasLtMatrixLayoutSetAttribute(
      layout,
      CUBLASLT_GROUPED_MATRIX_LAYOUT_ROWS_COLS_ARRAY_INTEGER_WIDTH,
      &width,
      sizeof(width));
  TORCH_CHECK(
      status == CUBLAS_STATUS_SUCCESS,
      "scaled_grouped_gemm_mxfp8: failed to set grouped rows/columns integer width");
  status = cublasLtMatrixLayoutSetAttribute(
      layout,
      CUBLASLT_GROUPED_MATRIX_LAYOUT_LD_ARRAY_INTEGER_WIDTH,
      &width,
      sizeof(width));
  TORCH_CHECK(
      status == CUBLAS_STATUS_SUCCESS,
      "scaled_grouped_gemm_mxfp8: failed to set grouped leading-dimension integer width");
}

at::Tensor scaled_gemm_mxfp8_cublaslt_meta(
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

at::Tensor scaled_gemm_mxfp8_cublaslt_cuda(
    const at::Tensor& a,
    const at::Tensor& b,
    const at::Tensor& scale_a,
    const at::Tensor& scale_b,
    const std::optional<at::Tensor>& bias) {
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

at::Tensor scaled_grouped_gemm_mxfp8_cublaslt_meta(
    const at::Tensor& aq,
    const at::Tensor& bq,
    const at::Tensor& scale_a,
    const at::Tensor& scale_b,
    const at::Tensor& offsets,
    const std::optional<at::Tensor>& bias) {
  static_cast<void>(scale_a);
  static_cast<void>(scale_b);
  static_cast<void>(offsets);
  static_cast<void>(bias);
  return at::empty_symint(
      {aq.sym_size(0), bq.sym_size(2)}, aq.options().dtype(at::kBFloat16));
}

at::Tensor scaled_grouped_gemm_mxfp8_cublaslt_cuda(
    const at::Tensor& aq,
    const at::Tensor& bq,
    const at::Tensor& scale_a,
    const at::Tensor& scale_b,
    const at::Tensor& offsets,
    const std::optional<at::Tensor>& bias) {
  check_grouped_tensor(
      aq,
      "A",
      at::ScalarType::Float8_e4m3fn,
      2,
      aq.device());
  check_grouped_tensor(
      bq,
      "B",
      at::ScalarType::Float8_e4m3fn,
      3,
      aq.device());
  check_grouped_tensor(
      scale_a, "A scale", at::kFloat, 2, aq.device());
  check_grouped_tensor(
      scale_b, "B scale", at::kFloat, 3, aq.device());
  check_grouped_tensor(
      offsets, "offsets", at::kInt, 1, aq.device());
  TORCH_CHECK(
      !bias.has_value(),
      "scaled_grouped_gemm_mxfp8: bias is not supported by cuBLASLt grouped MXFP8");
  TORCH_CHECK(
      offsets.numel() > 0,
      "scaled_grouped_gemm_mxfp8: offsets must contain at least one group");
  TORCH_CHECK(
      aq.size(1) == bq.size(1),
      "scaled_grouped_gemm_mxfp8: A and B contraction dimensions must match");

  const int64_t m = aq.size(0);
  const int64_t k = aq.size(1);
  const int64_t n = bq.size(2);
  const int64_t groups = offsets.size(0);
  const int64_t scale_blocks = (k + 31) / 32;
  TORCH_CHECK(
      k > 0 && n > 0 && k % 16 == 0 && n % 16 == 0,
      "scaled_grouped_gemm_mxfp8: K and N must be positive multiples of 16");
  TORCH_CHECK(
      m <= std::numeric_limits<uint32_t>::max() &&
          n <= std::numeric_limits<uint32_t>::max() &&
          k <= std::numeric_limits<uint32_t>::max() &&
          groups <= std::numeric_limits<uint32_t>::max(),
      "scaled_grouped_gemm_mxfp8: dimensions exceed cuBLASLt grouped limits");
  TORCH_CHECK(
      bq.size(0) == groups,
      "scaled_grouped_gemm_mxfp8: B group count must match offsets");
  TORCH_CHECK(
      scale_a.sizes() == at::IntArrayRef({m, scale_blocks}),
      "scaled_grouped_gemm_mxfp8: A scales must have shape [M, ceil(K / 32)]");
  TORCH_CHECK(
      scale_b.sizes() == at::IntArrayRef({groups, scale_blocks, n}),
      "scaled_grouped_gemm_mxfp8: B scales must have shape [E, ceil(K / 32), N]");
  TORCH_CHECK(
      offsets.is_contiguous(),
      "scaled_grouped_gemm_mxfp8: offsets must be contiguous int32");
  check_offset_buffer(offsets);
  TORCH_CHECK(
      aq.stride(1) == 1 && aq.stride(0) == k,
      "scaled_grouped_gemm_mxfp8: A must be contiguous row-major");
  TORCH_CHECK(
      bq.stride(0) % n == 0 && bq.stride(0) / n == k &&
          bq.stride(1) == 1 && bq.stride(2) == k,
      "scaled_grouped_gemm_mxfp8: B must be batch-major column-major; call to_column_major first");
  TORCH_CHECK(
      scale_a.is_contiguous() && scale_b.is_contiguous(),
      "scaled_grouped_gemm_mxfp8: scale tensors must be contiguous");
  check_16_byte_aligned(aq, "A");
  check_16_byte_aligned(bq, "B");

  c10::cuda::CUDAGuard device_guard(aq.device());
  TORCH_CHECK(
      cublasLtGetVersion() >= 130200,
      "scaled_grouped_gemm_mxfp8: grouped MXFP8 requires cuBLASLt 13.2 or newer");
  cudaDeviceProp device_properties{};
  C10_CUDA_CHECK(cudaGetDeviceProperties(
      &device_properties, aq.device().index()));
  TORCH_CHECK(
      device_properties.major == 10 ||
          (device_properties.major == 11 && device_properties.minor == 0),
      "scaled_grouped_gemm_mxfp8: grouped MXFP8 is supported only on SM10.x and SM11.0; got SM",
      device_properties.major,
      device_properties.minor);
  validate_mxfp8_grouped_offsets(offsets, m);
  at::Tensor out = at::empty({m, n}, aq.options().dtype(at::kBFloat16));
  if (m == 0) {
    return out;
  }

  Mxfp8GroupedMetadata metadata = build_mxfp8_grouped_metadata(
      aq, bq, scale_a, scale_b, offsets, out);
  cublasLtHandle_t handle = get_cublas_lt_handle(aq.device().index());
  const auto stream = at::cuda::getCurrentCUDAStream(aq.device().index());
  cublasStatus_t status = CUBLAS_STATUS_SUCCESS;

  CublasLtMatmulDesc operation;
  status = cublasLtMatmulDescCreate(
      operation.address(), CUBLAS_COMPUTE_32F, CUDA_R_32F);
  TORCH_CHECK(
      status == CUBLAS_STATUS_SUCCESS,
      "scaled_grouped_gemm_mxfp8: cublasLtMatmulDescCreate failed");
  const cublasOperation_t transa = CUBLAS_OP_T;
  const cublasOperation_t transb = CUBLAS_OP_N;
  status = cublasLtMatmulDescSetAttribute(
      operation.get(), CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa));
  TORCH_CHECK(
      status == CUBLAS_STATUS_SUCCESS,
      "scaled_grouped_gemm_mxfp8: failed to set TRANSA");
  status = cublasLtMatmulDescSetAttribute(
      operation.get(), CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb));
  TORCH_CHECK(
      status == CUBLAS_STATUS_SUCCESS,
      "scaled_grouped_gemm_mxfp8: failed to set TRANSB");
  const cublasLtMatmulMatrixScale_t scale_mode =
      CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0;
  status = cublasLtMatmulDescSetAttribute(
      operation.get(),
      CUBLASLT_MATMUL_DESC_A_SCALE_MODE,
      &scale_mode,
      sizeof(scale_mode));
  TORCH_CHECK(
      status == CUBLAS_STATUS_SUCCESS,
      "scaled_grouped_gemm_mxfp8: failed to set A scale mode");
  status = cublasLtMatmulDescSetAttribute(
      operation.get(),
      CUBLASLT_MATMUL_DESC_B_SCALE_MODE,
      &scale_mode,
      sizeof(scale_mode));
  TORCH_CHECK(
      status == CUBLAS_STATUS_SUCCESS,
      "scaled_grouped_gemm_mxfp8: failed to set B scale mode");
  const void* a_scale_pointers = metadata.a_scale_pointers.data_ptr();
  const void* b_scale_pointers = metadata.b_scale_pointers.data_ptr();
  status = cublasLtMatmulDescSetAttribute(
      operation.get(),
      CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
      &a_scale_pointers,
      sizeof(a_scale_pointers));
  TORCH_CHECK(
      status == CUBLAS_STATUS_SUCCESS,
      "scaled_grouped_gemm_mxfp8: failed to set A scale pointers");
  status = cublasLtMatmulDescSetAttribute(
      operation.get(),
      CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
      &b_scale_pointers,
      sizeof(b_scale_pointers));
  TORCH_CHECK(
      status == CUBLAS_STATUS_SUCCESS,
      "scaled_grouped_gemm_mxfp8: failed to set B scale pointers");

  CublasLtMatrixLayout a_layout;
  CublasLtMatrixLayout b_layout;
  CublasLtMatrixLayout c_layout;
  CublasLtMatrixLayout d_layout;
  status = cublasLtGroupedMatrixLayoutCreate(
      a_layout.address(),
      CUDA_R_8F_E4M3,
      groups,
      metadata.k.data_ptr(),
      metadata.n.data_ptr(),
      metadata.ldb.data_ptr());
  TORCH_CHECK(
      status == CUBLAS_STATUS_SUCCESS,
      "scaled_grouped_gemm_mxfp8: failed to create grouped A layout");
  set_grouped_dimension_width(a_layout.get());
  status = cublasLtGroupedMatrixLayoutCreate(
      b_layout.address(),
      CUDA_R_8F_E4M3,
      groups,
      metadata.k.data_ptr(),
      metadata.m.data_ptr(),
      metadata.lda.data_ptr());
  TORCH_CHECK(
      status == CUBLAS_STATUS_SUCCESS,
      "scaled_grouped_gemm_mxfp8: failed to create grouped B layout");
  set_grouped_dimension_width(b_layout.get());
  status = cublasLtGroupedMatrixLayoutCreate(
      c_layout.address(),
      CUDA_R_16BF,
      groups,
      metadata.n.data_ptr(),
      metadata.m.data_ptr(),
      metadata.ldc.data_ptr());
  TORCH_CHECK(
      status == CUBLAS_STATUS_SUCCESS,
      "scaled_grouped_gemm_mxfp8: failed to create grouped C layout");
  set_grouped_dimension_width(c_layout.get());
  status = cublasLtGroupedMatrixLayoutCreate(
      d_layout.address(),
      CUDA_R_16BF,
      groups,
      metadata.n.data_ptr(),
      metadata.m.data_ptr(),
      metadata.ldd.data_ptr());
  TORCH_CHECK(
      status == CUBLAS_STATUS_SUCCESS,
      "scaled_grouped_gemm_mxfp8: failed to create grouped D layout");
  set_grouped_dimension_width(d_layout.get());

  CublasLtMatmulPreference preference;
  status = cublasLtMatmulPreferenceCreate(preference.address());
  TORCH_CHECK(
      status == CUBLAS_STATUS_SUCCESS,
      "scaled_grouped_gemm_mxfp8: failed to create matmul preference");
  const uint32_t average_m = static_cast<uint32_t>((m + groups - 1) / groups);
  const uint32_t average_n = static_cast<uint32_t>(n);
  const uint32_t average_k = static_cast<uint32_t>(k);
  const size_t max_workspace = 1ull << 22;
  for (const auto& [attribute, value] :
       std::initializer_list<std::pair<cublasLtMatmulPreferenceAttributes_t, const uint32_t*>>{
           {CUBLASLT_MATMUL_PREF_GROUPED_DESC_D_AVERAGE_ROWS, &average_n},
           {CUBLASLT_MATMUL_PREF_GROUPED_DESC_D_AVERAGE_COLS, &average_m},
           {CUBLASLT_MATMUL_PREF_GROUPED_AVERAGE_REDUCTION_DIM, &average_k}}) {
    status = cublasLtMatmulPreferenceSetAttribute(
        preference.get(), attribute, value, sizeof(*value));
    TORCH_CHECK(
        status == CUBLAS_STATUS_SUCCESS,
        "scaled_grouped_gemm_mxfp8: failed to set grouped heuristic dimensions");
  }
  status = cublasLtMatmulPreferenceSetAttribute(
      preference.get(),
      CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
      &max_workspace,
      sizeof(max_workspace));
  TORCH_CHECK(
      status == CUBLAS_STATUS_SUCCESS,
      "scaled_grouped_gemm_mxfp8: failed to set workspace limit");

  cublasLtMatmulHeuristicResult_t heuristic{};
  int returned_algorithms = 0;
  status = cublasLtMatmulAlgoGetHeuristic(
      handle,
      operation.get(),
      a_layout.get(),
      b_layout.get(),
      c_layout.get(),
      d_layout.get(),
      preference.get(),
      1,
      &heuristic,
      &returned_algorithms);
  TORCH_CHECK(
      status == CUBLAS_STATUS_SUCCESS && returned_algorithms > 0 &&
          heuristic.state == CUBLAS_STATUS_SUCCESS,
      "scaled_grouped_gemm_mxfp8: no cuBLASLt grouped MXFP8 algorithm is available");
  const int64_t workspace_bytes = static_cast<int64_t>(heuristic.workspaceSize);
  at::Tensor workspace = at::empty(
      {workspace_bytes}, aq.options().dtype(at::kByte));
  void* workspace_pointer =
      workspace_bytes == 0 ? nullptr : workspace.data_ptr();
  const float alpha = 1.0f;
  const float beta = 0.0f;
  status = cublasLtMatmul(
      handle,
      operation.get(),
      &alpha,
      metadata.a_pointers.data_ptr(),
      a_layout.get(),
      metadata.b_pointers.data_ptr(),
      b_layout.get(),
      &beta,
      metadata.c_pointers.data_ptr(),
      c_layout.get(),
      metadata.d_pointers.data_ptr(),
      d_layout.get(),
      &heuristic.algo,
      workspace_pointer,
      workspace_bytes,
      stream.stream());
  TORCH_CHECK(
      status == CUBLAS_STATUS_SUCCESS,
      "scaled_grouped_gemm_mxfp8: cublasLtMatmul failed");
  return out;
}

}  // namespace

TORCH_LIBRARY_FRAGMENT(aot_kernel, m) {
  m.def(
      "_scaled_gemm_mxfp8_cublaslt(Tensor a, Tensor b, Tensor scale_a, "
      "Tensor scale_b, Tensor? bias=None) -> Tensor");
  m.def(
      "_scaled_grouped_gemm_mxfp8_cublaslt(Tensor aq, Tensor bq, Tensor scale_a, "
      "Tensor scale_b, Tensor offsets, Tensor? bias=None) -> Tensor");
}

TORCH_LIBRARY_IMPL(aot_kernel, Meta, m) {
  m.impl(
      "_scaled_gemm_mxfp8_cublaslt",
      TORCH_FN(scaled_gemm_mxfp8_cublaslt_meta));
  m.impl(
      "_scaled_grouped_gemm_mxfp8_cublaslt",
      TORCH_FN(scaled_grouped_gemm_mxfp8_cublaslt_meta));
}

TORCH_LIBRARY_IMPL(aot_kernel, CUDA, m) {
  m.impl(
      "_scaled_gemm_mxfp8_cublaslt",
      TORCH_FN(scaled_gemm_mxfp8_cublaslt_cuda));
  m.impl(
      "_scaled_grouped_gemm_mxfp8_cublaslt",
      TORCH_FN(scaled_grouped_gemm_mxfp8_cublaslt_cuda));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {}
