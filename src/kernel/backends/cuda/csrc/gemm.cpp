#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDADataType.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cublasLt.h>

#include <cstdint>
#include <optional>
#include <unordered_map>
#include <vector>

cudaError_t write_matmul_alpha_beta_from_global_scales(
    const float* global_a,
    const float* global_b,
    float* alpha_beta,
    cudaStream_t stream);

namespace {

struct CublasLtHandle {
  cublasLtHandle_t handle{nullptr};

  CublasLtHandle() {
    const auto status = cublasLtCreate(&handle);
    TORCH_CHECK(
        status == CUBLAS_STATUS_SUCCESS,
        "cuBLASLt: cublasLtCreate failed");
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

struct MatmulScalars {
  const float alpha;
  const float beta;
  const void* alpha_pointer;
  const void* beta_pointer;
  at::Tensor device_values;

  MatmulScalars()
      : alpha(1.0f),
        beta(0.0f),
        alpha_pointer(&alpha),
        beta_pointer(&beta) {}
};

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

void prepare_global_scales(
    cublasLtMatmulDesc_t operation,
    const std::optional<at::Tensor>& global_a,
    const std::optional<at::Tensor>& global_b,
    const at::Tensor& a,
    cudaStream_t stream,
    const char* operation_name,
    MatmulScalars* scalars) {
  if (!global_a.has_value()) {
    return;
  }

  scalars->device_values =
      at::empty({2}, a.options().dtype(at::ScalarType::Float));
  const auto cuda_status = write_matmul_alpha_beta_from_global_scales(
      global_a->data_ptr<float>(),
      global_b->data_ptr<float>(),
      scalars->device_values.data_ptr<float>(),
      stream);
  TORCH_CHECK(
      cuda_status == cudaSuccess,
      operation_name,
      ": failed to prepare global scales: ",
      cudaGetErrorString(cuda_status));

  const cublasLtPointerMode_t pointer_mode = CUBLASLT_POINTER_MODE_DEVICE;
  const auto status = cublasLtMatmulDescSetAttribute(
      operation,
      CUBLASLT_MATMUL_DESC_POINTER_MODE,
      &pointer_mode,
      sizeof(pointer_mode));
  TORCH_CHECK(
      status == CUBLAS_STATUS_SUCCESS,
      operation_name,
      ": failed to set device pointer mode");
  scalars->alpha_pointer = scalars->device_values.data_ptr();
  scalars->beta_pointer = scalars->device_values.data_ptr<float>() + 1;
}

// Python validates values and layouts before transforming inputs for this ABI.
void check_scaled_mm_cuda_inputs(
    const at::Tensor& a,
    const at::Tensor& b,
    const at::Tensor& scale_a,
    const at::Tensor& scale_b,
    const std::optional<at::Tensor>& bias,
    const std::optional<at::Tensor>& global_a,
    const std::optional<at::Tensor>& global_b,
    const char* operation) {
  TORCH_CHECK(
      a.device() == b.device() && a.device() == scale_a.device() &&
          a.device() == scale_b.device(),
      operation,
      ": A, B, and scales must be on the same CUDA device");
  if (bias.has_value()) {
    TORCH_CHECK(
        bias->device() == a.device(),
        operation,
        ": bias must be on the same CUDA device as A");
  }
  TORCH_CHECK(
      global_a.has_value() == global_b.has_value(),
      operation,
      ": global_a and global_b must be provided together");
  for (const auto& [name, scale] : {
           std::pair{"global_a", global_a},
           std::pair{"global_b", global_b},
       }) {
    if (scale.has_value()) {
      TORCH_CHECK(
          scale->device() == a.device(),
          operation,
          ": ",
          name,
          " must be on A's CUDA device");
      TORCH_CHECK(
          scale->scalar_type() == at::ScalarType::Float,
          operation,
          ": ",
          name,
          " must have float32 dtype");
      TORCH_CHECK(
          scale->numel() == 1,
          operation,
          ": ",
          name,
          " must contain exactly one value");
    }
  }
  const auto* properties = at::cuda::getDeviceProperties(a.get_device());
  TORCH_CHECK(
      properties->major >= 10,
      operation,
      ": requires SM100 or newer, got SM",
      properties->major,
      properties->minor);
}

at::Tensor scaled_mm_mxfp8_cublaslt_meta(
    const at::Tensor& a,
    const at::Tensor& b,
    const at::Tensor& scale_a,
    const at::Tensor& scale_b,
    at::ScalarType out_dtype,
    const std::optional<at::Tensor>& bias,
    const std::optional<at::Tensor>& global_a,
    const std::optional<at::Tensor>& global_b) {
  return at::empty_symint(
      {a.sym_size(0), b.sym_size(1)}, a.options().dtype(out_dtype));
}

at::Tensor scaled_mm_mxfp8_cublaslt_cuda(
    const at::Tensor& a,
    const at::Tensor& b,
    const at::Tensor& scale_a,
    const at::Tensor& scale_b,
    at::ScalarType out_dtype,
    const std::optional<at::Tensor>& bias,
    const std::optional<at::Tensor>& global_a,
    const std::optional<at::Tensor>& global_b) {
  check_scaled_mm_cuda_inputs(
      a,
      b,
      scale_a,
      scale_b,
      bias,
      global_a,
      global_b,
      "scaled_mm_mxfp8");
  c10::cuda::CUDAGuard device_guard(a.device());

  const int64_t M = a.size(0);
  const int64_t K = a.size(1);
  const int64_t N = b.size(1);

  if (M == 0 || N == 0 || K == 0) {
    at::Tensor out = at::zeros({M, N}, a.options().dtype(out_dtype));
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

  at::Tensor out = at::empty({M, N}, a.options().dtype(out_dtype));

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
  const auto output_type = at::cuda::ScalarTypeToCudaDataType(out_dtype);
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
      output_type,
      N,
      M,
      out.stride(0));
  TORCH_CHECK(
      status == CUBLAS_STATUS_SUCCESS,
      "scaled_mm_mxfp8: failed to create C matrix layout");
  set_matrix_order(c_layout.get(), CUBLASLT_ORDER_COL);
  status = cublasLtMatrixLayoutCreate(
      d_layout.address(),
      output_type,
      N,
      M,
      out.stride(0));
  TORCH_CHECK(
      status == CUBLAS_STATUS_SUCCESS,
      "scaled_mm_mxfp8: failed to create D matrix layout");
  set_matrix_order(d_layout.get(), CUBLASLT_ORDER_COL);

  MatmulScalars scalars;
  prepare_global_scales(
      operation.get(),
      global_a,
      global_b,
      a,
      stream.stream(),
      "scaled_mm_mxfp8",
      &scalars);

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
      scalars.alpha_pointer,
      aligned_b.data_ptr(),
      a_layout.get(),
      aligned_a.data_ptr(),
      b_layout.get(),
      scalars.beta_pointer,
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

at::Tensor scaled_mm_nvfp4_cublaslt_meta(
    const at::Tensor& a,
    const at::Tensor& b,
    const at::Tensor& scale_a,
    const at::Tensor& scale_b,
    at::ScalarType out_dtype,
    const std::optional<at::Tensor>& bias,
    const std::optional<at::Tensor>& global_a,
    const std::optional<at::Tensor>& global_b) {
  return at::empty_symint(
      {a.sym_size(0), b.sym_size(1)}, a.options().dtype(out_dtype));
}

at::Tensor scaled_mm_nvfp4_cublaslt_cuda(
    const at::Tensor& a,
    const at::Tensor& b,
    const at::Tensor& scale_a,
    const at::Tensor& scale_b,
    at::ScalarType out_dtype,
    const std::optional<at::Tensor>& bias,
    const std::optional<at::Tensor>& global_a,
    const std::optional<at::Tensor>& global_b) {
  check_scaled_mm_cuda_inputs(
      a,
      b,
      scale_a,
      scale_b,
      bias,
      global_a,
      global_b,
      "scaled_mm_nvfp4");
  c10::cuda::CUDAGuard device_guard(a.device());

  const int64_t M = a.size(0);
  const int64_t K = a.size(1) * 2;
  const int64_t N = b.size(1);
  if (M == 0 || N == 0 || K == 0) {
    at::Tensor out = at::zeros({M, N}, a.options().dtype(out_dtype));
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
  at::Tensor out = at::empty({M, N}, a.options().dtype(out_dtype));

  cublasLtHandle_t handle = get_cublas_lt_handle(a.device().index());
  const auto stream = at::cuda::getCurrentCUDAStream(a.device().index());
  cublasStatus_t status = CUBLAS_STATUS_SUCCESS;

  CublasLtMatmulDesc operation;
  status = cublasLtMatmulDescCreate(
      operation.address(), CUBLAS_COMPUTE_32F, CUDA_R_32F);
  TORCH_CHECK(
      status == CUBLAS_STATUS_SUCCESS,
      "scaled_mm_nvfp4: cublasLtMatmulDescCreate failed");

  const cublasOperation_t transpose_a = CUBLAS_OP_T;
  const cublasOperation_t transpose_b = CUBLAS_OP_N;
  for (const auto& attribute : {
           std::pair{CUBLASLT_MATMUL_DESC_TRANSA, &transpose_a},
           std::pair{CUBLASLT_MATMUL_DESC_TRANSB, &transpose_b},
       }) {
    status = cublasLtMatmulDescSetAttribute(
        operation.get(), attribute.first, attribute.second, sizeof(transpose_a));
    TORCH_CHECK(
        status == CUBLAS_STATUS_SUCCESS,
        "scaled_mm_nvfp4: failed to set transpose attribute");
  }

  const cublasLtMatmulMatrixScale_t scale_mode =
      CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
  for (const auto attribute : {
           CUBLASLT_MATMUL_DESC_A_SCALE_MODE,
           CUBLASLT_MATMUL_DESC_B_SCALE_MODE,
       }) {
    status = cublasLtMatmulDescSetAttribute(
        operation.get(), attribute, &scale_mode, sizeof(scale_mode));
    TORCH_CHECK(
        status == CUBLAS_STATUS_SUCCESS,
        "scaled_mm_nvfp4: failed to set block scale mode");
  }
  const void* cublas_scale_a = scale_b.data_ptr();
  const void* cublas_scale_b = scale_a.data_ptr();
  status = cublasLtMatmulDescSetAttribute(
      operation.get(),
      CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
      &cublas_scale_a,
      sizeof(cublas_scale_a));
  TORCH_CHECK(
      status == CUBLAS_STATUS_SUCCESS,
      "scaled_mm_nvfp4: failed to set A scale pointer");
  status = cublasLtMatmulDescSetAttribute(
      operation.get(),
      CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
      &cublas_scale_b,
      sizeof(cublas_scale_b));
  TORCH_CHECK(
      status == CUBLAS_STATUS_SUCCESS,
      "scaled_mm_nvfp4: failed to set B scale pointer");

  const auto output_type = at::cuda::ScalarTypeToCudaDataType(out_dtype);
  if (bias.has_value()) {
    const cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_BIAS;
    status = cublasLtMatmulDescSetAttribute(
        operation.get(),
        CUBLASLT_MATMUL_DESC_EPILOGUE,
        &epilogue,
        sizeof(epilogue));
    TORCH_CHECK(
        status == CUBLAS_STATUS_SUCCESS,
        "scaled_mm_nvfp4: failed to set bias epilogue");
    const void* bias_pointer = bias->data_ptr();
    status = cublasLtMatmulDescSetAttribute(
        operation.get(),
        CUBLASLT_MATMUL_DESC_BIAS_POINTER,
        &bias_pointer,
        sizeof(bias_pointer));
    TORCH_CHECK(
        status == CUBLAS_STATUS_SUCCESS,
        "scaled_mm_nvfp4: failed to set bias pointer");
    status = cublasLtMatmulDescSetAttribute(
        operation.get(),
        CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE,
        &output_type,
        sizeof(output_type));
    TORCH_CHECK(
        status == CUBLAS_STATUS_SUCCESS,
        "scaled_mm_nvfp4: failed to set bias dtype");
  }

  CublasLtMatrixLayout a_layout;
  CublasLtMatrixLayout b_layout;
  CublasLtMatrixLayout c_layout;
  CublasLtMatrixLayout d_layout;
  status = cublasLtMatrixLayoutCreate(
      a_layout.address(), CUDA_R_4F_E2M1, K, N, K);
  TORCH_CHECK(
      status == CUBLAS_STATUS_SUCCESS,
      "scaled_mm_nvfp4: failed to create A matrix layout");
  status = cublasLtMatrixLayoutCreate(
      b_layout.address(), CUDA_R_4F_E2M1, K, M, K);
  TORCH_CHECK(
      status == CUBLAS_STATUS_SUCCESS,
      "scaled_mm_nvfp4: failed to create B matrix layout");
  status = cublasLtMatrixLayoutCreate(c_layout.address(), output_type, N, M, N);
  TORCH_CHECK(
      status == CUBLAS_STATUS_SUCCESS,
      "scaled_mm_nvfp4: failed to create C matrix layout");
  status = cublasLtMatrixLayoutCreate(d_layout.address(), output_type, N, M, N);
  TORCH_CHECK(
      status == CUBLAS_STATUS_SUCCESS,
      "scaled_mm_nvfp4: failed to create D matrix layout");

  MatmulScalars scalars;
  prepare_global_scales(
      operation.get(),
      global_a,
      global_b,
      a,
      stream.stream(),
      "scaled_mm_nvfp4",
      &scalars);

  CublasLtMatmulPreference preference;
  status = cublasLtMatmulPreferenceCreate(preference.address());
  TORCH_CHECK(
      status == CUBLAS_STATUS_SUCCESS,
      "scaled_mm_nvfp4: failed to create matmul preference");
  size_t max_workspace = 1ull << 22;
  status = cublasLtMatmulPreferenceSetAttribute(
      preference.get(),
      CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
      &max_workspace,
      sizeof(max_workspace));
  TORCH_CHECK(
      status == CUBLAS_STATUS_SUCCESS,
      "scaled_mm_nvfp4: failed to set workspace limit");
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
      "scaled_mm_nvfp4: cublasLtMatmulAlgoGetHeuristic failed");
  TORCH_CHECK(
      returned_algorithms > 0,
      "scaled_mm_nvfp4: no cuBLASLt algorithms available");
  const auto& selected = heuristics[0];
  TORCH_CHECK(
      selected.state == CUBLAS_STATUS_SUCCESS,
      "scaled_mm_nvfp4: first cuBLASLt heuristic is invalid");

  const int64_t workspace_bytes = static_cast<int64_t>(selected.workspaceSize);
  at::Tensor workspace = at::empty(
      {workspace_bytes}, a.options().dtype(at::ScalarType::Byte));
  void* workspace_pointer = workspace_bytes > 0 ? workspace.data_ptr() : nullptr;
  status = cublasLtMatmul(
      handle,
      operation.get(),
      scalars.alpha_pointer,
      aligned_b.data_ptr(),
      a_layout.get(),
      aligned_a.data_ptr(),
      b_layout.get(),
      scalars.beta_pointer,
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
      "scaled_mm_nvfp4: cublasLtMatmul failed");
  return out;
}

}  // namespace

TORCH_LIBRARY_FRAGMENT(aot_kernel, m) {
  m.def(
      "_scaled_mm_mxfp8_cublaslt(Tensor a, Tensor b, Tensor scale_a, "
      "Tensor scale_b, ScalarType out_dtype, Tensor? bias=None, "
      "Tensor? global_a=None, Tensor? global_b=None) -> Tensor");
  m.def(
      "_scaled_mm_nvfp4_cublaslt(Tensor a, Tensor b, Tensor scale_a, "
      "Tensor scale_b, ScalarType out_dtype, Tensor? bias=None, "
      "Tensor? global_a=None, Tensor? global_b=None) -> Tensor");
}

TORCH_LIBRARY_IMPL(aot_kernel, Meta, m) {
  m.impl(
      "_scaled_mm_mxfp8_cublaslt",
      TORCH_FN(scaled_mm_mxfp8_cublaslt_meta));
  m.impl(
      "_scaled_mm_nvfp4_cublaslt",
      TORCH_FN(scaled_mm_nvfp4_cublaslt_meta));
}

TORCH_LIBRARY_IMPL(aot_kernel, CUDA, m) {
  m.impl(
      "_scaled_mm_mxfp8_cublaslt",
      TORCH_FN(scaled_mm_mxfp8_cublaslt_cuda));
  m.impl(
      "_scaled_mm_nvfp4_cublaslt",
      TORCH_FN(scaled_mm_nvfp4_cublaslt_cuda));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {}
