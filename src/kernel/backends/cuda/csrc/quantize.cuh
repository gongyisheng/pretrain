#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include "reduce.cuh"

#include <cstdint>
#include <limits>

template <int dim0_size, int dim1_size>
struct BlockShape {
  static constexpr int dim0 = dim0_size;
  static constexpr int dim1 = dim1_size;
};

enum class Dtype : uint8_t {
  Fp32,
  Fp8E4M3,
  Fp8E5M2,
  Fp8E8M0,
  Int4,
  Int5,
  Int6,
  Int7,
  Int8,
  Fp4E2M1,
  Fp4E2M1_4Over6,
};

enum class Layout : uint8_t { RowMajor, ColumnMajor, Swizzled32_4_4 };

// Strides use each tensor's storage elements, including packed NVFP4 codes.
struct QuantizeParams {
  const void* input = nullptr;
  void* codes = nullptr;
  Dtype dtype = Dtype::Fp8E4M3;
  Layout output_layout = Layout::RowMajor;
  void* scale = nullptr;
  Dtype scale_dtype = Dtype::Fp32;
  Layout scale_layout = Layout::RowMajor;
  const float* global_scale = nullptr;
  bool enable_global_scale = false;
  bool stochastic_rounding = false;
  float* statistics = nullptr;
  bool collect_stats = false;
  int64_t batch = 0;
  int64_t row = 0;
  int64_t col = 0;
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
    return static_cast<T*>(scale);
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

__device__ __forceinline__ int64_t tile_element_offset(
    int32_t row, int32_t col, int64_t row_stride, int64_t col_stride,
    bool narrow_offsets
) {
  if (narrow_offsets) {
    return row * static_cast<int32_t>(row_stride) +
           col * static_cast<int32_t>(col_stride);
  }
  return static_cast<int64_t>(row) * row_stride +
         static_cast<int64_t>(col) * col_stride;
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
  if (params.row <= 0 || params.col <= 0 ||
      params.col > limit || params.row > limit / params.col ||
      params.rows_per_block <= 0 || params.cols_per_block <= 0 ||
      params.rows_per_block > limit - params.row ||
      params.cols_per_block > limit - params.col) return false;
  return offsets_fit_int32(params.row, params.col,
                           params.input_row_stride, params.input_col_stride) &&
      offsets_fit_int32(params.row, params.col,
                         params.code_row_stride, params.code_col_stride) &&
      offsets_fit_int32(params.row, params.col,
                         params.scale_row_stride, params.scale_col_stride);
}

__device__ __forceinline__ int64_t grid_index() {
  return static_cast<int64_t>(blockIdx.x) +
      static_cast<int64_t>(blockIdx.y) * static_cast<int64_t>(gridDim.x);
}

__device__ __forceinline__ void swizzled_32_4_4_scale_epilogue(
    uint8_t* scales,
    uint8_t value,
    int64_t logical_row,
    int64_t logical_col,
    int64_t logical_rows,
    int64_t logical_cols,
    bool padding_only = false) {
  const int64_t padded_rows = (logical_rows + 127) / 128 * 128;
  const int64_t padded_cols = (logical_cols + 3) / 4 * 4;
  const int64_t padded_row_count = (padded_rows - logical_rows) * padded_cols;
  const int64_t padded_col_count = logical_rows * (padded_cols - logical_cols);
  const int64_t first = padding_only
      ? grid_index() * blockDim.x + threadIdx.x : 0;
  const int64_t stride = padding_only
      ? static_cast<int64_t>(gridDim.x) * gridDim.y * blockDim.x : 1;
  const int64_t count = padding_only ? padded_row_count + padded_col_count : 1;
  for (int64_t index = first; index < count; index += stride) {
    const int64_t row = padding_only && index < padded_row_count
        ? logical_rows + index / padded_cols
        : padding_only ? (index - padded_row_count) /
              (padded_cols - logical_cols) : logical_row;
    const int64_t col = padding_only && index < padded_row_count
        ? index % padded_cols
        : padding_only ? logical_cols + (index - padded_row_count) %
              (padded_cols - logical_cols) : logical_col;
    const int64_t offset = ((((row / 128 * (padded_cols / 4) + col / 4) * 32 +
                              row % 32) *
                                 4 +
                             (row % 128) / 32) *
                                4 +
                            col % 4);
    scales[offset] = padding_only ? 0 : value;
  }
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

template <bool kE5m2>
__device__ __forceinline__ uint8_t encode_fp8_ceil(float value) {
  if (isnan(value)) return kE5m2 ? 0x7e : 0x7f;
  constexpr float kMinValue = kE5m2 ? 0x1p-16f : 0x1p-9f;
  constexpr float kMaxValue = kE5m2 ? 57344.0f : 448.0f;
  value = fminf(fmaxf(value, kMinValue), kMaxValue);
  const uint8_t code = encode_fp8_rne<kE5m2>(value);
  return code + static_cast<uint8_t>(decode_fp8<kE5m2>(code) < value);
}

__device__ __forceinline__ uint8_t encode_fp8_e8m0(float maximum) {
  if (isnan(maximum)) return 255;
  const float exponent = ceilf(log2f(maximum / 448.0f));
  return static_cast<uint8_t>(fminf(fmaxf(exponent, -127.0f), 127.0f) + 127.0f);
}

__device__ __forceinline__ float decode_fp8_e8m0(uint8_t code) {
  uint32_t bits;
  if (code == 0) {
    bits = 254u << 23;
  } else if (code == 254) {
    bits = 0x00400000;
  } else if (code == 255) {
    bits = 0x7f800001;
  } else {
    bits = static_cast<uint32_t>(254 - code) << 23;
  }
  return __int_as_float(static_cast<int>(bits));
}

struct QuantizationStatistics {
  float src_sq = 0.0f;
  float err_sq = 0.0f;
  float under = 0.0f;
  float nonzero = 0.0f;
};

__device__ __forceinline__ void accumulate_quantization_statistics(
    QuantizationStatistics* partial,
    float src,
    float deq) {
  const bool src_nonzero = src != 0.0f;
  partial->src_sq += src * src;
  const float error = src - deq;
  partial->err_sq += error * error;
  partial->under += src_nonzero && deq == 0.0f ? 1.0f : 0.0f;
  partial->nonzero += src_nonzero ? 1.0f : 0.0f;
}

// All threads in a 1D block must participate, including those with no valid elements.
// Statistics must be zero-initialized; atomic sums are not bitwise deterministic.
template <int kThreads>
__device__ __forceinline__ void quantization_statistics_epilogue(
    QuantizationStatistics partial, float* statistics, int64_t numel) {
  float values[4] = {
      partial.src_sq, partial.err_sq, partial.under, partial.nonzero};
  cuda_reduce::block_reduce<kThreads>(values, cuda_reduce::Sum<float>{});
  if (threadIdx.x == 0) {
    atomicAdd(statistics, values[0]);
    atomicAdd(statistics + 1, values[1]);
    atomicAdd(statistics + 2, values[2]);
    atomicAdd(statistics + 4, values[3]);
    if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
      statistics[3] = static_cast<float>(numel);
    }
  }
}
