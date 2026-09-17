#pragma once

#include <cuda_runtime.h>
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

template <bool kE5m2>
__device__ __forceinline__ uint8_t fp8_rne(float value) {
  return __nv_cvt_float_to_fp8(
      value, __NV_SATFINITE, kE5m2 ? __NV_E5M2 : __NV_E4M3);
}

template <bool kE5m2>
__device__ __forceinline__ uint8_t fp8_stochastic(float value, uint32_t random) {
  if (isnan(value)) return fp8_rne<kE5m2>(value);
  constexpr int kMantissaBits = kE5m2 ? 2 : 3;
  constexpr int kMinUlpExponent = kE5m2 ? -16 : -9;
  const int exponent = ((__float_as_int(value) >> 23) & 0xff) - kMantissaBits;
  const float ulp = __int_as_float(max(exponent, 127 + kMinUlpExponent) << 23);
  const float lower = floorf(value / ulp) * ulp;
  const float probability = (value - lower) / ulp;
  return fp8_rne<kE5m2>(
      static_cast<float>(random) * 0x1.0p-32f < probability ? lower + ulp : lower);
}
