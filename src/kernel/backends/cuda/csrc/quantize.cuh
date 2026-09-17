#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include <cstdint>
#include <limits>

// Strides use each tensor's storage elements, including packed NVFP4 codes.
struct QuantizeParams {
  const void* input = nullptr;
  void* codes = nullptr;
  void* scales = nullptr;
  const float* global_scale = nullptr;
  float* statistics = nullptr;
  float* statistics_partials = nullptr;
  int64_t batches = 0;
  int64_t rows = 0;
  int64_t cols = 0;
  int64_t input_batch_stride = 0;
  int64_t input_row_stride = 0;
  int64_t input_col_stride = 0;
  int64_t code_batch_stride = 0;
  int64_t code_row_stride = 0;
  int64_t code_col_stride = 0;
  int64_t scale_batch_stride = 0;
  int64_t scale_row_stride = 0;
  int64_t scale_col_stride = 0;
  int64_t rows_per_block = 0;
  int64_t cols_per_block = 0;
  int64_t row_block_count = 0;
  int64_t col_block_count = 0;
  int contract_dim = -1;

  template <typename T>
  __host__ __device__ const T* input_data() const {
    return static_cast<const T*>(input);
  }

  template <typename T>
  __host__ __device__ T* code_data() const {
    return static_cast<T*>(codes);
  }

  template <typename T>
  __host__ __device__ T* scale_data() const {
    return static_cast<T*>(scales);
  }
};

template <typename index_t = int64_t>
__device__ __forceinline__ int64_t input_offset(
    const QuantizeParams& params, int64_t batch, int64_t row, int64_t col) {
  const index_t local_offset = static_cast<index_t>(row) *
          static_cast<index_t>(params.input_row_stride) +
      static_cast<index_t>(col) * static_cast<index_t>(params.input_col_stride);
  return batch * params.input_batch_stride + static_cast<int64_t>(local_offset);
}

template <typename index_t = int64_t>
__device__ __forceinline__ int64_t output_offset(
    const QuantizeParams& params, int64_t batch, int64_t row, int64_t col) {
  const index_t local_offset = static_cast<index_t>(row) *
          static_cast<index_t>(params.code_row_stride) +
      static_cast<index_t>(col) * static_cast<index_t>(params.code_col_stride);
  return batch * params.code_batch_stride + static_cast<int64_t>(local_offset);
}

template <typename index_t = int64_t>
__device__ __forceinline__ int64_t scale_offset(
    const QuantizeParams& params, int64_t batch, int64_t row, int64_t col) {
  const index_t local_offset = static_cast<index_t>(row) *
          static_cast<index_t>(params.scale_row_stride) +
      static_cast<index_t>(col) * static_cast<index_t>(params.scale_col_stride);
  return batch * params.scale_batch_stride + static_cast<int64_t>(local_offset);
}

inline bool offsets_fit_int32(
    int64_t rows, int64_t cols, int64_t row_stride, int64_t col_stride) {
  constexpr int64_t limit = std::numeric_limits<int32_t>::max();
  if (rows <= 0 || cols <= 0 || row_stride < 0 || col_stride < 0 ||
      row_stride > limit || col_stride > limit) return false;
  if (row_stride != 0 && rows - 1 > limit / row_stride) return false;
  const int64_t row_offset = (rows - 1) * row_stride;
  return col_stride == 0 || cols - 1 <= (limit - row_offset) / col_stride;
}

inline bool can_use_int32_indices(const QuantizeParams& params) {
  // Leave room for the final loop increment and padded block boundaries.
  constexpr int64_t limit = std::numeric_limits<int32_t>::max() - 4096;
  if (params.rows <= 0 || params.cols <= 0 ||
      params.cols > limit || params.rows > limit / params.cols ||
      params.rows_per_block <= 0 || params.cols_per_block <= 0 ||
      params.rows_per_block > limit - params.rows ||
      params.cols_per_block > limit - params.cols) return false;
  return offsets_fit_int32(params.rows, params.cols,
                           params.input_row_stride, params.input_col_stride) &&
      offsets_fit_int32(params.rows, params.cols,
                         params.code_row_stride, params.code_col_stride) &&
      offsets_fit_int32(params.rows, params.cols,
                         params.scale_row_stride, params.scale_col_stride);
}

__device__ __forceinline__ int64_t grid_index() {
  return static_cast<int64_t>(blockIdx.x) +
      static_cast<int64_t>(blockIdx.y) * static_cast<int64_t>(gridDim.x);
}

__device__ __forceinline__ float strict_max(float lhs, float rhs) {
  return isnan(lhs) || isnan(rhs) ? nanf("") : fmaxf(lhs, rhs);
}

__device__ __forceinline__ float compute_scale(float amax, float qmax) {
  constexpr float kMinScale = 1.0e-30f;
  return isnan(amax) ? nanf("") : fmaxf(amax * (1.0f / qmax), kMinScale);
}

template <bool kE5m2>
__device__ __forceinline__ uint8_t encode_fp8_rne(float value) {
  return __nv_cvt_float_to_fp8(
      value, __NV_SATFINITE, kE5m2 ? __NV_E5M2 : __NV_E4M3);
}

template <bool kE5m2>
__device__ __forceinline__ uint8_t encode_fp8_stochastic(float value, uint32_t random) {
  if (isnan(value)) return encode_fp8_rne<kE5m2>(value);
  constexpr int kMantissaBits = kE5m2 ? 2 : 3;
  constexpr int kMinUlpExponent = kE5m2 ? -16 : -9;
  const int exponent = ((__float_as_int(value) >> 23) & 0xff) - kMantissaBits;
  const float ulp = __int_as_float(max(exponent, 127 + kMinUlpExponent) << 23);
  const float lower = floorf(value / ulp) * ulp;
  const float probability = (value - lower) / ulp;
  return encode_fp8_rne<kE5m2>(
      static_cast<float>(random) * 0x1.0p-32f < probability ? lower + ulp : lower);
}

template <bool kE5m2>
__device__ __forceinline__ float decode_fp8(uint8_t code) {
  const __half_raw raw = __nv_cvt_fp8_to_halfraw(
      static_cast<__nv_fp8_storage_t>(code), kE5m2 ? __NV_E5M2 : __NV_E4M3);
  __half value;
  value = raw;
  return __half2float(value);
}

struct QuantizationStatistics {
  float src_sq = 0.0f;
  float err_sq = 0.0f;
  float under = 0.0f;
  float nonzero = 0.0f;
};

__device__ __forceinline__ void accumulate_quantization_statistics(
    QuantizationStatistics* partial,
    float source,
    float reconstructed,
    bool quantized_zero) {
  const bool source_nonzero = source != 0.0f;
  partial->src_sq += source * source;
  const float error = source - reconstructed;
  partial->err_sq += error * error;
  partial->under += source_nonzero && quantized_zero ? 1.0f : 0.0f;
  partial->nonzero += source_nonzero ? 1.0f : 0.0f;
}
