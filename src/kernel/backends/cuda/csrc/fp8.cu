#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <ATen/cuda/PhiloxUtils.cuh>
#include <ATen/core/enum_tag.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>

#include <cuda_fp8.h>
#include <cuda_fp16.h>
#include <curand_kernel.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <mutex>
#include <string>
#include <tuple>
#include <vector>

namespace {

constexpr int kThreads = 256;
constexpr int kRowwiseSplit = 256;
constexpr int kTensorwiseValues = 8;
constexpr float kMinScale = 1.0e-30f;

struct QuantizeParams {
  const void* input;
  uint8_t* codes;
  float* scales;
  float* statistics;
  int64_t batches;
  int64_t rows;
  int64_t cols;
  int64_t input_batch_stride;
  int64_t input_row_stride;
  int64_t input_col_stride;
  int64_t code_batch_stride;
  int64_t code_row_stride;
  int64_t code_col_stride;
  int64_t outer_size;
  int64_t contract_size;
  int64_t block_outer;
  int64_t block_contract;
  int64_t outer_groups;
  int64_t contract_groups;
  bool contract_is_col;
};

struct LaunchGrid {
  dim3 grid;
  int64_t blocks_per_row;
};

LaunchGrid make_launch_grid(int64_t blocks) {
  constexpr int64_t kMaxGridX = std::numeric_limits<int32_t>::max();
  constexpr int64_t kMaxGridY = 65535;
  const int64_t blocks_per_row = std::min(blocks, kMaxGridX);
  const int64_t rows = (blocks + blocks_per_row - 1) / blocks_per_row;
  TORCH_CHECK(rows <= kMaxGridY, "quantize_fp8 tensor is too large to launch");
  return {dim3(blocks_per_row, rows), blocks_per_row};
}

__device__ __forceinline__ int64_t grid_index(int64_t blocks_per_row) {
  return static_cast<int64_t>(blockIdx.x) +
      static_cast<int64_t>(blockIdx.y) * blocks_per_row;
}

__device__ __forceinline__ float max_with_nan(float lhs, float rhs) {
  return isnan(lhs) || isnan(rhs) ? nanf("") : fmaxf(lhs, rhs);
}

__device__ __forceinline__ float scale_from_amax(float amax, float qmax) {
  return isnan(amax) ? nanf("") : fmaxf(amax * (1.0f / qmax), kMinScale);
}

template <bool kE5m2>
__device__ __forceinline__ uint8_t fp8_rne(float value) {
  return __nv_cvt_float_to_fp8(
      value, __NV_SATFINITE, kE5m2 ? __NV_E5M2 : __NV_E4M3);
}

template <bool kE5m2>
__device__ __forceinline__ float decode_fp8(uint8_t code) {
  const __half_raw raw = __nv_cvt_fp8_to_halfraw(
      static_cast<__nv_fp8_storage_t>(code), kE5m2 ? __NV_E5M2 : __NV_E4M3);
  __half value;
  value = raw;
  return __half2float(value);
}

struct StatisticsPartial {
  float src_sq = 0.0f;
  float err_sq = 0.0f;
  float under = 0.0f;
  float nonzero = 0.0f;
};

template <bool kE5m2>
__device__ __forceinline__ void accumulate_statistics(
    StatisticsPartial* partial, float source, uint8_t code, float scale) {
  const bool source_nonzero = source != 0.0f;
  partial->src_sq += source * source;
  const float error = source - decode_fp8<kE5m2>(code) * scale;
  partial->err_sq += error * error;
  partial->under += source_nonzero && (code & 0x7f) == 0 ? 1.0f : 0.0f;
  partial->nonzero += source_nonzero ? 1.0f : 0.0f;
}

__device__ __forceinline__ void reduce_statistics(
    StatisticsPartial partial, float* statistics, int64_t numel) {
  if (statistics == nullptr) return;
  __shared__ float values[4][kThreads];
  values[0][threadIdx.x] = partial.src_sq;
  values[1][threadIdx.x] = partial.err_sq;
  values[2][threadIdx.x] = partial.under;
  values[3][threadIdx.x] = partial.nonzero;
  __syncthreads();
  for (int width = kThreads / 2; width > 0; width >>= 1) {
    if (threadIdx.x < width) {
      values[0][threadIdx.x] += values[0][threadIdx.x + width];
      values[1][threadIdx.x] += values[1][threadIdx.x + width];
      values[2][threadIdx.x] += values[2][threadIdx.x + width];
      values[3][threadIdx.x] += values[3][threadIdx.x + width];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    const int64_t offset =
        (static_cast<int64_t>(blockIdx.y) * gridDim.x + blockIdx.x) * 4;
    statistics[offset] = values[0][0];
    statistics[offset + 1] = values[1][0];
    statistics[offset + 2] = values[2][0];
    statistics[offset + 3] = values[3][0];
  }
}

__global__ void finalize_statistics_kernel(
    const float* partials, int64_t partial_count, int64_t numel, float* statistics) {
  float values[4] = {};
  for (int64_t index = threadIdx.x; index < partial_count; index += blockDim.x) {
    #pragma unroll
    for (int field = 0; field < 4; ++field) values[field] += partials[index * 4 + field];
  }
  __shared__ float reduced[4][kThreads];
  #pragma unroll
  for (int field = 0; field < 4; ++field) reduced[field][threadIdx.x] = values[field];
  __syncthreads();
  for (int width = kThreads / 2; width > 0; width >>= 1) {
    if (threadIdx.x < width) {
      #pragma unroll
      for (int field = 0; field < 4; ++field) {
        reduced[field][threadIdx.x] += reduced[field][threadIdx.x + width];
      }
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    statistics[0] = reduced[0][0]; statistics[1] = reduced[1][0];
    statistics[2] = reduced[2][0]; statistics[3] = static_cast<float>(numel);
    statistics[4] = reduced[3][0];
  }
}

template <bool kE5m2>
__device__ __forceinline__ float normalize_fp8_rne(
    float value, float scale, float inverse_scale) {
  float normalized = __fmul_rn(value, inverse_scale);
  const unsigned int bits = __float_as_uint(normalized) & 0x7fffffff;
  const int exponent = bits >> 23;
  constexpr int kMantissaBits = kE5m2 ? 2 : 3;
  constexpr int kNormalExponent = kE5m2 ? 113 : 121;
  if (exponent >= kNormalExponent) {
    constexpr unsigned int kMask = (1u << (23 - kMantissaBits)) - 1;
    constexpr unsigned int kMidpoint = 1u << (22 - kMantissaBits);
    if ((bits & kMask) - kMidpoint + 16 <= 32) {
      normalized = value / scale;
    }
  } else if (exponent > 0) {
    constexpr int kMinUlpExponent = kE5m2 ? -16 : -9;
    const int shift = 150 + kMinUlpExponent - exponent;
    if (shift <= 25) {
      const unsigned int significand = (bits & 0x7fffff) | 0x800000;
      const unsigned int remainder = significand & ((1u << shift) - 1);
      const unsigned int midpoint = 1u << (shift - 1);
      // Reciprocal multiplication differs by a few FP32 ULPs. Preserve exact
      // FP8 rounding with division near a rounding midpoint (including ties).
      if (remainder - midpoint + 16 <= 32) {
        normalized = value / scale;
      }
    }
  }
  return normalized;
}

template <bool kE5m2>
__device__ __forceinline__ uint8_t fp8_stochastic(
    float value,
    curandStatePhilox4_32_10_t* state) {
  if (isnan(value)) {
    return fp8_rne<kE5m2>(value);
  }
  constexpr int kMantissaBits = kE5m2 ? 2 : 3;
  constexpr int kMinUlpExponent = kE5m2 ? -16 : -9;
  const int exponent = ((__float_as_int(value) >> 23) & 0xff) - kMantissaBits;
  const int ulp_exponent = max(exponent, 127 + kMinUlpExponent);
  const float ulp = __int_as_float(ulp_exponent << 23);
  const float lower = floorf(value / ulp) * ulp;
  const float probability = (value - lower) / ulp;
  const float random = static_cast<float>(curand(state)) * 0x1.0p-32f;
  return fp8_rne<kE5m2>(random < probability ? lower + ulp : lower);
}

__device__ __forceinline__ int64_t input_offset(
    const QuantizeParams& params, int64_t batch, int64_t row, int64_t col) {
  return batch * params.input_batch_stride + row * params.input_row_stride +
      col * params.input_col_stride;
}

__device__ __forceinline__ int64_t output_offset(
    const QuantizeParams& params, int64_t batch, int64_t row, int64_t col) {
  return batch * params.code_batch_stride + row * params.code_row_stride +
      col * params.code_col_stride;
}

__device__ __forceinline__ int64_t scale_offset(
    const QuantizeParams& params,
    int64_t batch,
    int64_t outer,
    int64_t contract) {
  if (params.contract_is_col) {
    return (batch * params.rows + outer) * params.contract_groups + contract;
  }
  return (batch * params.contract_groups + contract) * params.cols + outer;
}

template <typename input_t, bool kE5m2, bool kStochastic>
__global__ void quantize_tiled_kernel(
    QuantizeParams params,
    at::PhiloxCudaState philox,
    int64_t blocks_per_row) {
  const int64_t tile = grid_index(blocks_per_row);
  const int64_t tile_count =
      params.batches * params.outer_groups * params.contract_groups;
  if (tile >= tile_count) {
    return;
  }
  const int64_t contract_group = tile % params.contract_groups;
  const int64_t remaining = tile / params.contract_groups;
  const int64_t outer_group = remaining % params.outer_groups;
  const int64_t batch = remaining / params.outer_groups;
  const int64_t outer_start = outer_group * params.block_outer;
  const int64_t contract_start = contract_group * params.block_contract;
  const int64_t outer_end = min(outer_start + params.block_outer, params.outer_size);
  const int64_t contract_end =
      min(contract_start + params.block_contract, params.contract_size);

  float maximum = 0.0f;
  for (int64_t outer = outer_start; outer < outer_end; ++outer) {
    for (int64_t contract = contract_start + threadIdx.x; contract < contract_end;
         contract += blockDim.x) {
      const int64_t row = params.contract_is_col ? outer : contract;
      const int64_t col = params.contract_is_col ? contract : outer;
      const float value = static_cast<float>(
          static_cast<const input_t*>(params.input)[input_offset(params, batch, row, col)]);
      maximum = max_with_nan(maximum, fabsf(value));
    }
  }
  __shared__ float shared_maximum[kThreads];
  shared_maximum[threadIdx.x] = maximum;
  __syncthreads();
  for (int width = kThreads / 2; width > 0; width >>= 1) {
    if (threadIdx.x < width) {
      shared_maximum[threadIdx.x] =
          max_with_nan(shared_maximum[threadIdx.x], shared_maximum[threadIdx.x + width]);
    }
    __syncthreads();
  }
  __shared__ float shared_scale;
  if (threadIdx.x == 0) {
    shared_scale = scale_from_amax(
        shared_maximum[0], kE5m2 ? 57344.0f : 448.0f);
    for (int64_t outer = outer_start; outer < outer_end; ++outer) {
      params.scales[scale_offset(params, batch, outer, contract_group)] = shared_scale;
    }
  }
  __syncthreads();

  curandStatePhilox4_32_10_t random_state;
  StatisticsPartial statistics;
  if constexpr (kStochastic) {
    const auto seeds = at::cuda::philox::unpack(philox);
    curand_init(
        static_cast<unsigned long long>(std::get<0>(seeds)),
        static_cast<unsigned long long>(tile * blockDim.x + threadIdx.x),
        static_cast<unsigned long long>(std::get<1>(seeds)),
        &random_state);
  }
  constexpr float qmax = kE5m2 ? 57344.0f : 448.0f;
  for (int64_t outer = outer_start; outer < outer_end; ++outer) {
    for (int64_t contract = contract_start + threadIdx.x; contract < contract_end;
         contract += blockDim.x) {
      const int64_t row = params.contract_is_col ? outer : contract;
      const int64_t col = params.contract_is_col ? contract : outer;
      float value = static_cast<float>(
          static_cast<const input_t*>(params.input)[input_offset(params, batch, row, col)]);
      value /= shared_scale;
      if (!isnan(value)) {
        value = fminf(fmaxf(value, -qmax), qmax);
      }
      const int64_t code_offset = output_offset(params, batch, row, col);
      uint8_t code;
      if constexpr (kStochastic) code = fp8_stochastic<kE5m2>(value, &random_state);
      else code = fp8_rne<kE5m2>(value);
      params.codes[code_offset] = code;
      if (params.statistics != nullptr) accumulate_statistics<kE5m2>(&statistics,
          static_cast<float>(static_cast<const input_t*>(params.input)[
              input_offset(params, batch, row, col)]), code, shared_scale);
    }
  }
  reduce_statistics(statistics, params.statistics,
                    params.batches * params.rows * params.cols);
}

template <typename input_t, bool kE5m2, bool kStochastic, int kBlockContract>
__global__ void quantize_blockwise_1d_packed_kernel(
    QuantizeParams params,
    at::PhiloxCudaState philox,
    int64_t blocks_per_row) {
  constexpr int kWarps = kThreads / 32;
  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  const int64_t tile = grid_index(blocks_per_row) * kWarps + warp;
  const int64_t tile_count = params.batches * params.outer_size * params.contract_groups;
  if (tile >= tile_count) {
    return;
  }
  const int64_t contract_group = tile % params.contract_groups;
  const int64_t remaining = tile / params.contract_groups;
  const int64_t outer = remaining % params.outer_size;
  const int64_t batch = remaining / params.outer_size;
  constexpr int kValuesPerLane = (kBlockContract + 31) / 32;
  const int64_t contract_start = contract_group * kBlockContract;
  float maximum = 0.0f;
  StatisticsPartial statistics;
  float values[kValuesPerLane];
  #pragma unroll
  for (int value_index = 0; value_index < kValuesPerLane; ++value_index) {
    const int64_t contract = contract_start + lane + value_index * 32;
    if (contract < params.contract_size && value_index * 32 + lane < kBlockContract) {
      const int64_t row = params.contract_is_col ? outer : contract;
      const int64_t col = params.contract_is_col ? contract : outer;
      values[value_index] = static_cast<float>(static_cast<const input_t*>(params.input)[
          input_offset(params, batch, row, col)]);
      maximum = max_with_nan(maximum, fabsf(values[value_index]));
    }
  }
  constexpr unsigned int kWarpMask = 0xffffffff;
  for (int offset = 16; offset > 0; offset >>= 1) {
    maximum = max_with_nan(maximum, __shfl_down_sync(kWarpMask, maximum, offset));
  }
  maximum = __shfl_sync(kWarpMask, maximum, 0);
  const float scale = scale_from_amax(maximum, kE5m2 ? 57344.0f : 448.0f);
  if (lane == 0) {
    params.scales[scale_offset(params, batch, outer, contract_group)] = scale;
  }
  if constexpr (kStochastic) {
    const auto seeds = at::cuda::philox::unpack(philox);
    curandStatePhilox4_32_10_t random_state;
    curand_init(
        static_cast<unsigned long long>(std::get<0>(seeds)),
        static_cast<unsigned long long>(tile * 32 + lane),
        static_cast<unsigned long long>(std::get<1>(seeds)),
        &random_state);
    #pragma unroll
    for (int value_index = 0; value_index < kValuesPerLane; ++value_index) {
      const int64_t contract = contract_start + lane + value_index * 32;
      if (contract < params.contract_size && value_index * 32 + lane < kBlockContract) {
        float value = values[value_index] / scale;
        if (!isnan(value)) {
          constexpr float qmax = kE5m2 ? 57344.0f : 448.0f;
          value = fminf(fmaxf(value, -qmax), qmax);
        }
        const int64_t row = params.contract_is_col ? outer : contract;
        const int64_t col = params.contract_is_col ? contract : outer;
        const uint8_t code = fp8_stochastic<kE5m2>(value, &random_state);
        params.codes[output_offset(params, batch, row, col)] = code;
        if (params.statistics != nullptr) accumulate_statistics<kE5m2>(&statistics, values[value_index], code, scale);
      }
    }
  } else {
    #pragma unroll
    for (int value_index = 0; value_index < kValuesPerLane; ++value_index) {
      const int64_t contract = contract_start + lane + value_index * 32;
      if (contract < params.contract_size && value_index * 32 + lane < kBlockContract) {
        float value = values[value_index] / scale;
        if (!isnan(value)) {
          constexpr float qmax = kE5m2 ? 57344.0f : 448.0f;
          value = fminf(fmaxf(value, -qmax), qmax);
        }
        const int64_t row = params.contract_is_col ? outer : contract;
        const int64_t col = params.contract_is_col ? contract : outer;
        const uint8_t code = fp8_rne<kE5m2>(value);
        params.codes[output_offset(params, batch, row, col)] = code;
        if (params.statistics != nullptr) accumulate_statistics<kE5m2>(&statistics, values[value_index], code, scale);
      }
    }
  }
  if (params.statistics != nullptr) {
    const int64_t offset = (static_cast<int64_t>(blockIdx.y) * gridDim.x + blockIdx.x) * 4;
    atomicAdd(params.statistics + offset, statistics.src_sq);
    atomicAdd(params.statistics + offset + 1, statistics.err_sq);
    atomicAdd(params.statistics + offset + 2, statistics.under);
    atomicAdd(params.statistics + offset + 3, statistics.nonzero);
  }
}

template <typename input_t>
__device__ __forceinline__ void load_contiguous_four(
    const input_t* input, int offset, float values[4]) {
  values[0] = static_cast<float>(input[offset]);
  values[1] = static_cast<float>(input[offset + 1]);
  values[2] = static_cast<float>(input[offset + 2]);
  values[3] = static_cast<float>(input[offset + 3]);
}

template <>
__device__ __forceinline__ void load_contiguous_four<float>(
    const float* input, int offset, float values[4]) {
  const float4 packed = reinterpret_cast<const float4*>(input + offset)[0];
  values[0] = packed.x;
  values[1] = packed.y;
  values[2] = packed.z;
  values[3] = packed.w;
}

template <>
__device__ __forceinline__ void load_contiguous_four<at::Half>(
    const at::Half* input, int offset, float values[4]) {
  union PackedHalf {
    uint2 raw;
    at::Half elements[4];
  } packed;
  packed.raw = reinterpret_cast<const uint2*>(input + offset)[0];
  values[0] = static_cast<float>(packed.elements[0]);
  values[1] = static_cast<float>(packed.elements[1]);
  values[2] = static_cast<float>(packed.elements[2]);
  values[3] = static_cast<float>(packed.elements[3]);
}

template <>
__device__ __forceinline__ void load_contiguous_four<at::BFloat16>(
    const at::BFloat16* input, int offset, float values[4]) {
  union PackedBFloat16 {
    uint2 raw;
    at::BFloat16 elements[4];
  } packed;
  packed.raw = reinterpret_cast<const uint2*>(input + offset)[0];
  values[0] = static_cast<float>(packed.elements[0]);
  values[1] = static_cast<float>(packed.elements[1]);
  values[2] = static_cast<float>(packed.elements[2]);
  values[3] = static_cast<float>(packed.elements[3]);
}

template <bool kE5m2>
__device__ __forceinline__ uint32_t fp8x4_rne(
    float first, float second, float third, float fourth) {
  const __nv_fp8x2_storage_t low = __nv_cvt_float2_to_fp8x2(
      make_float2(first, second), __NV_SATFINITE,
      kE5m2 ? __NV_E5M2 : __NV_E4M3);
  const __nv_fp8x2_storage_t high = __nv_cvt_float2_to_fp8x2(
      make_float2(third, fourth), __NV_SATFINITE,
      kE5m2 ? __NV_E5M2 : __NV_E4M3);
  return static_cast<uint32_t>(low) | (static_cast<uint32_t>(high) << 16);
}

template <typename input_t, bool kE5m2, bool kStochastic, int kBlockContract>
__global__ void quantize_blockwise_1d_contiguous_kernel(
    const input_t* input,
    uint8_t* codes,
    float* scales,
    int rows,
    int cols,
    int contract_groups,
    at::PhiloxCudaState philox,
    int blocks_per_row,
    float* statistics,
    int64_t numel) {
  constexpr int kWarps = kThreads / 32;
  constexpr int kValuesPerLane = kBlockContract == 256 ? 8 : 4;
  constexpr int kVectorsPerLane = kValuesPerLane / 4;
  constexpr int kLanesPerTile = kBlockContract / kValuesPerLane;
  constexpr int kTilesPerWarp = 32 / kLanesPerTile;
  static_assert(kLanesPerTile > 0 && kLanesPerTile <= 32);
  static_assert(32 % kLanesPerTile == 0);

  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  const int tile_in_warp = lane / kLanesPerTile;
  const int lane_in_tile = lane % kLanesPerTile;
  const int warp_index =
      (static_cast<int>(blockIdx.y) * blocks_per_row + static_cast<int>(blockIdx.x)) *
          kWarps +
      warp;
  const int tile = warp_index * kTilesPerWarp + tile_in_warp;
  const int tile_count = rows * contract_groups;
  const bool valid_tile = tile < tile_count;
  const int offset = valid_tile
      ? tile * kBlockContract + lane_in_tile * kValuesPerLane
      : 0;

  float values[kValuesPerLane];
  StatisticsPartial partial;
  float maximum = 0.0f;
  if (valid_tile) {
    #pragma unroll
    for (int vector_index = 0; vector_index < kVectorsPerLane; ++vector_index) {
      load_contiguous_four(
          input, offset + vector_index * 4, values + vector_index * 4);
    }
    #pragma unroll
    for (int value_index = 0; value_index < kValuesPerLane; ++value_index) {
      maximum = max_with_nan(maximum, fabsf(values[value_index]));
    }
  }
  constexpr unsigned int kWarpMask = 0xffffffff;
  #pragma unroll
  for (int reduction_offset = kLanesPerTile / 2; reduction_offset > 0;
       reduction_offset >>= 1) {
    maximum = max_with_nan(
        maximum,
        __shfl_down_sync(kWarpMask, maximum, reduction_offset, kLanesPerTile));
  }
  const float scale = __shfl_sync(
      kWarpMask,
      scale_from_amax(maximum, kE5m2 ? 57344.0f : 448.0f),
      tile_in_warp * kLanesPerTile,
      kLanesPerTile);
  if (valid_tile && lane_in_tile == 0) {
    scales[tile] = scale;
  }
  if (!valid_tile && statistics == nullptr) {
    return;
  }

  if (valid_tile) if constexpr (kStochastic) {
    const auto seeds = at::cuda::philox::unpack(philox);
    curandStatePhilox4_32_10_t random_state;
    curand_init(
        static_cast<unsigned long long>(std::get<0>(seeds)),
        static_cast<unsigned long long>(tile * kLanesPerTile + lane_in_tile),
        static_cast<unsigned long long>(std::get<1>(seeds)),
        &random_state);
    constexpr float qmax = kE5m2 ? 57344.0f : 448.0f;
    #pragma unroll
    for (int vector_index = 0; vector_index < kVectorsPerLane; ++vector_index) {
      const int value_offset = vector_index * 4;
      uint32_t packed = 0;
      #pragma unroll
      for (int element = 0; element < 4; ++element) {
        float value = values[value_offset + element] / scale;
        if (!isnan(value)) {
          value = fminf(fmaxf(value, -qmax), qmax);
        }
        const uint8_t code = fp8_stochastic<kE5m2>(value, &random_state);
        packed |= static_cast<uint32_t>(code) << (element * 8);
        if (statistics != nullptr) {
          const float source = static_cast<float>(input[offset + value_offset + element]);
          accumulate_statistics<kE5m2>(&partial, source, code, scale);
        }
      }
      reinterpret_cast<uint32_t*>(codes + offset + value_offset)[0] = packed;
    }
  } else if (valid_tile) {
    const float inverse_scale = __frcp_rn(scale);
    #pragma unroll
    for (int value_index = 0; value_index < kValuesPerLane; ++value_index) {
      values[value_index] = normalize_fp8_rne<kE5m2>(
          values[value_index], scale, inverse_scale);
    }
    #pragma unroll
    for (int vector_index = 0; vector_index < kVectorsPerLane; ++vector_index) {
      const int value_offset = vector_index * 4;
      const uint32_t packed = fp8x4_rne<kE5m2>(
              values[value_offset], values[value_offset + 1],
              values[value_offset + 2], values[value_offset + 3]);
      reinterpret_cast<uint32_t*>(codes + offset + value_offset)[0] = packed;
      if (statistics != nullptr) {
        #pragma unroll
        for (int element = 0; element < 4; ++element) {
          const uint8_t code = static_cast<uint8_t>(packed >> (element * 8));
          const float source = static_cast<float>(input[offset + value_offset + element]);
          accumulate_statistics<kE5m2>(&partial, source, code, scale);
        }
      }
    }
  }
  reduce_statistics(partial, statistics, numel);
}

template <typename input_t, bool kE5m2, int kBlockContract>
__global__ void quantize_blockwise_1d_transposed_rne_kernel(
    const input_t* input,
    uint8_t* codes,
    float* scales,
    int batches,
    int rows,
    int cols,
    int contract_groups,
    int blocks_per_row,
    float* statistics,
    int64_t numel) {
  constexpr int kOuterTile = 32;
  constexpr int kThreadsPerOuter = 8;
  constexpr int kValuesPerThread = kBlockContract / kThreadsPerOuter;
  static_assert(kBlockContract % kThreadsPerOuter == 0);

  const int tile = static_cast<int>(blockIdx.y) * blocks_per_row + blockIdx.x;
  const int outer_tiles = (rows + kOuterTile - 1) / kOuterTile;
  const int tile_count = batches * contract_groups * outer_tiles;
  if (tile >= tile_count) {
    return;
  }
  const int outer_tile = tile % outer_tiles;
  const int remaining = tile / outer_tiles;
  const int contract_group = remaining % contract_groups;
  const int batch = remaining / contract_groups;
  const int outer_start = outer_tile * kOuterTile;
  const int contract_start = contract_group * kBlockContract;
  const int valid_outers = min(kOuterTile, rows - outer_start);
  StatisticsPartial partial;

  __shared__ float shared_values[kOuterTile * (kBlockContract + 1)];
  for (int flat = threadIdx.x; flat < kOuterTile * kBlockContract;
       flat += blockDim.x) {
    const int contract = flat / kOuterTile;
    const int outer = flat - contract * kOuterTile;
    if (outer < valid_outers) {
      const int input_offset = batch * rows * cols +
          (contract_start + contract) * rows + outer_start + outer;
      shared_values[outer * (kBlockContract + 1) + contract] =
          static_cast<float>(input[input_offset]);
    }
  }
  __syncthreads();

  const int outer = threadIdx.x / kThreadsPerOuter;
  const int segment = threadIdx.x % kThreadsPerOuter;
  const int outer_in_warp = (threadIdx.x % 32) / kThreadsPerOuter;
  const bool valid_outer = outer < valid_outers;
  float maximum = 0.0f;
  if (valid_outer) {
    #pragma unroll
    for (int value_index = 0; value_index < kValuesPerThread; ++value_index) {
      const int value_offset = kValuesPerThread == 2
          ? segment * 2 + value_index
          : segment * 4 + (value_index / 4) * kThreadsPerOuter * 4 +
              value_index % 4;
      const float value = shared_values[
          outer * (kBlockContract + 1) + value_offset];
      maximum = max_with_nan(maximum, fabsf(value));
    }
  }
  constexpr unsigned int kWarpMask = 0xffffffff;
  #pragma unroll
  for (int reduction_offset = kThreadsPerOuter / 2; reduction_offset > 0;
       reduction_offset >>= 1) {
    maximum = max_with_nan(
        maximum,
        __shfl_down_sync(kWarpMask, maximum, reduction_offset, kThreadsPerOuter));
  }
  const float scale = __shfl_sync(
      kWarpMask,
      scale_from_amax(maximum, kE5m2 ? 57344.0f : 448.0f),
      outer_in_warp * kThreadsPerOuter,
      kThreadsPerOuter);
  if (valid_outer && segment == 0) {
    scales[(batch * rows + outer_start + outer) * contract_groups + contract_group] =
        scale;
  }
  if (valid_outer) {
    const float inverse_scale = __frcp_rn(scale);
    const int output_offset =
        (batch * rows + outer_start + outer) * cols + contract_start;
    if constexpr (kValuesPerThread == 2) {
    const int value_offset = segment * 2;
    const int shared_offset = outer * (kBlockContract + 1) + value_offset;
    const uint16_t packed =
        __nv_cvt_float2_to_fp8x2(
            make_float2(
                normalize_fp8_rne<kE5m2>(
                    shared_values[shared_offset], scale, inverse_scale),
                normalize_fp8_rne<kE5m2>(
                    shared_values[shared_offset + 1], scale, inverse_scale)),
            __NV_SATFINITE,
            kE5m2 ? __NV_E5M2 : __NV_E4M3);
    reinterpret_cast<uint16_t*>(codes + output_offset + value_offset)[0] = packed;
    if (statistics != nullptr) {
      accumulate_statistics<kE5m2>(
          &partial, shared_values[shared_offset], static_cast<uint8_t>(packed), scale);
      accumulate_statistics<kE5m2>(
          &partial, shared_values[shared_offset + 1],
          static_cast<uint8_t>(packed >> 8), scale);
    }
    } else {
    #pragma unroll
    for (int vector_index = 0; vector_index < kValuesPerThread / 4; ++vector_index) {
      const int value_offset =
          segment * 4 + vector_index * kThreadsPerOuter * 4;
      const int shared_offset = outer * (kBlockContract + 1) + value_offset;
      const uint32_t packed =
          fp8x4_rne<kE5m2>(
              normalize_fp8_rne<kE5m2>(
                  shared_values[shared_offset], scale, inverse_scale),
              normalize_fp8_rne<kE5m2>(
                  shared_values[shared_offset + 1], scale, inverse_scale),
              normalize_fp8_rne<kE5m2>(
                  shared_values[shared_offset + 2], scale, inverse_scale),
              normalize_fp8_rne<kE5m2>(
                  shared_values[shared_offset + 3], scale, inverse_scale));
      reinterpret_cast<uint32_t*>(codes + output_offset + value_offset)[0] = packed;
      if (statistics != nullptr) {
        #pragma unroll
        for (int element = 0; element < 4; ++element) {
          accumulate_statistics<kE5m2>(
              &partial, shared_values[shared_offset + element],
              static_cast<uint8_t>(packed >> (element * 8)), scale);
        }
      }
    }
    }
  }
  if (statistics != nullptr) {
    reduce_statistics(partial, statistics, numel);
  }
}

template <typename input_t, bool kE5m2, int kBlockContract>
__global__ void quantize_blockwise_1d_tiled_column_rne_kernel(
    const input_t* input,
    uint8_t* codes,
    float* scales,
    int batches,
    int rows,
    int cols,
    int contract_groups,
    int blocks_per_row,
    float* statistics,
    int64_t numel) {
  constexpr int kOuterTile = 32;
  constexpr int kThreadsPerOuter = 8;
  constexpr int kValuesPerThread = kBlockContract / kThreadsPerOuter;
  static_assert(kBlockContract % kThreadsPerOuter == 0);

  const int tile = static_cast<int>(blockIdx.y) * blocks_per_row + blockIdx.x;
  const int outer_tiles = (cols + kOuterTile - 1) / kOuterTile;
  const int tile_count = batches * contract_groups * outer_tiles;
  if (tile >= tile_count) {
    return;
  }
  const int outer_tile = tile % outer_tiles;
  const int remaining = tile / outer_tiles;
  const int contract_group = remaining % contract_groups;
  const int batch = remaining / contract_groups;
  const int outer_start = outer_tile * kOuterTile;
  const int contract_start = contract_group * kBlockContract;
  const int valid_outers = min(kOuterTile, cols - outer_start);
  StatisticsPartial partial;

  __shared__ float shared_values[kOuterTile * (kBlockContract + 1)];
  __shared__ float shared_scales[kOuterTile];
  for (int flat = threadIdx.x; flat < kOuterTile * kBlockContract;
       flat += blockDim.x) {
    const int outer = flat / kBlockContract;
    const int contract = flat - outer * kBlockContract;
    if (outer < valid_outers) {
      const int input_offset = batch * rows * cols +
          (outer_start + outer) * rows + contract_start + contract;
      shared_values[outer * (kBlockContract + 1) + contract] =
          static_cast<float>(input[input_offset]);
    }
  }
  __syncthreads();

  const int outer = threadIdx.x / kThreadsPerOuter;
  const int segment = threadIdx.x % kThreadsPerOuter;
  const bool valid_outer = outer < valid_outers;
  float maximum = 0.0f;
  if (valid_outer) {
    #pragma unroll 1
    for (int value_index = 0; value_index < kValuesPerThread; ++value_index) {
      const int value_offset = kValuesPerThread == 2
          ? segment * 2 + value_index
          : segment * 4 + (value_index / 4) * kThreadsPerOuter * 4 +
              value_index % 4;
      maximum = max_with_nan(
          maximum,
          fabsf(shared_values[outer * (kBlockContract + 1) + value_offset]));
    }
  }
  constexpr unsigned int kWarpMask = 0xffffffff;
  #pragma unroll
  for (int reduction_offset = kThreadsPerOuter / 2; reduction_offset > 0;
       reduction_offset >>= 1) {
    maximum = max_with_nan(
        maximum,
        __shfl_down_sync(kWarpMask, maximum, reduction_offset, kThreadsPerOuter));
  }
  if (valid_outer && segment == 0) {
    const float scale = scale_from_amax(maximum, kE5m2 ? 57344.0f : 448.0f);
    shared_scales[outer] = scale;
    scales[(batch * contract_groups + contract_group) * cols + outer_start + outer] =
        scale;
  }
  __syncthreads();

  if (valid_outer) {
    const float scale = shared_scales[outer];
    const float inverse_scale = __frcp_rn(scale);
    #pragma unroll 1
    for (int value_index = 0; value_index < kValuesPerThread; ++value_index) {
      const int value_offset = kValuesPerThread == 2
          ? segment * 2 + value_index
          : segment * 4 + (value_index / 4) * kThreadsPerOuter * 4 +
              value_index % 4;
      const int shared_offset = outer * (kBlockContract + 1) + value_offset;
      shared_values[shared_offset] = normalize_fp8_rne<kE5m2>(
          shared_values[shared_offset], scale, inverse_scale);
    }
  }
  __syncthreads();

  const int pairs = kOuterTile * kBlockContract / 2;
  for (int pair = threadIdx.x; pair < pairs; pair += blockDim.x) {
    const int flat = pair * 2;
    const int contract = flat / kOuterTile;
    const int output_outer = flat - contract * kOuterTile;
    if (output_outer < valid_outers) {
      const int shared_offset = output_outer * (kBlockContract + 1) + contract;
      const float first = shared_values[shared_offset];
      const int output_offset =
          (batch * rows + contract_start + contract) * cols + outer_start + output_outer;
      if (output_outer + 1 < valid_outers) {
        const float second = shared_values[shared_offset + kBlockContract + 1];
        const uint16_t packed =
            __nv_cvt_float2_to_fp8x2(
                make_float2(first, second), __NV_SATFINITE,
                kE5m2 ? __NV_E5M2 : __NV_E4M3);
        reinterpret_cast<uint16_t*>(codes + output_offset)[0] = packed;
        if (statistics != nullptr) {
          const int input_offset = batch * rows * cols +
              (outer_start + output_outer) * rows + contract_start + contract;
          accumulate_statistics<kE5m2>(
              &partial, static_cast<float>(input[input_offset]),
              static_cast<uint8_t>(packed), shared_scales[output_outer]);
          accumulate_statistics<kE5m2>(
              &partial, static_cast<float>(input[input_offset + rows]),
              static_cast<uint8_t>(packed >> 8), shared_scales[output_outer + 1]);
        }
      } else {
        const uint8_t code = fp8_rne<kE5m2>(first);
        codes[output_offset] = code;
        if (statistics != nullptr) {
          const int input_offset = batch * rows * cols +
              (outer_start + output_outer) * rows + contract_start + contract;
          accumulate_statistics<kE5m2>(
              &partial, static_cast<float>(input[input_offset]), code,
              shared_scales[output_outer]);
        }
      }
    }
  }
  if (statistics != nullptr) {
    reduce_statistics(partial, statistics, numel);
  }
}

template <typename input_t, bool kE5m2, bool kStochastic>
__global__ void quantize_rows_coalesced_kernel(
    QuantizeParams params,
    at::PhiloxCudaState philox,
    int64_t blocks_per_row) {
  constexpr int kWarps = kThreads / 32;
  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  const int64_t outer_tiles = (params.outer_size + 31) / 32;
  const int64_t tile = grid_index(blocks_per_row) * kWarps + warp;
  const int64_t tile_count = params.batches * params.contract_groups * outer_tiles;
  if (tile >= tile_count) {
    return;
  }
  const int64_t outer_tile = tile % outer_tiles;
  const int64_t remaining = tile / outer_tiles;
  const int64_t contract_group = remaining % params.contract_groups;
  const int64_t batch = remaining / params.contract_groups;
  const int64_t outer = outer_tile * 32 + lane;
  if (outer >= params.outer_size) {
    return;
  }
  const int64_t contract_start = contract_group * params.block_contract;
  const int64_t contract_end =
      min(contract_start + params.block_contract, params.contract_size);
  float maximum = 0.0f;
  for (int64_t contract = contract_start; contract < contract_end; ++contract) {
    const float value = static_cast<float>(static_cast<const input_t*>(params.input)[
        input_offset(params, batch, contract, outer)]);
    maximum = max_with_nan(maximum, fabsf(value));
  }
  const float scale = scale_from_amax(maximum, kE5m2 ? 57344.0f : 448.0f);
  params.scales[scale_offset(params, batch, outer, contract_group)] = scale;
  curandStatePhilox4_32_10_t random_state;
  StatisticsPartial statistics;
  if constexpr (kStochastic) {
    const auto seeds = at::cuda::philox::unpack(philox);
    curand_init(
        static_cast<unsigned long long>(std::get<0>(seeds)),
        static_cast<unsigned long long>(tile * 32 + lane),
        static_cast<unsigned long long>(std::get<1>(seeds)),
        &random_state);
  }
  constexpr float qmax = kE5m2 ? 57344.0f : 448.0f;
  for (int64_t contract = contract_start; contract < contract_end; ++contract) {
    const float source = static_cast<float>(static_cast<const input_t*>(params.input)[
        input_offset(params, batch, contract, outer)]);
    float value = source;
    value /= scale;
    if (!isnan(value)) {
      value = fminf(fmaxf(value, -qmax), qmax);
    }
    const int64_t code_offset = output_offset(params, batch, contract, outer);
    uint8_t code;
    if constexpr (kStochastic) code = fp8_stochastic<kE5m2>(value, &random_state);
    else code = fp8_rne<kE5m2>(value);
    params.codes[code_offset] = code;
    if (params.statistics != nullptr) accumulate_statistics<kE5m2>(&statistics, source, code, scale);
  }
  if (params.statistics != nullptr) {
    const int64_t offset = (static_cast<int64_t>(blockIdx.y) * gridDim.x + blockIdx.x) * 4;
    atomicAdd(params.statistics + offset, statistics.src_sq);
    atomicAdd(params.statistics + offset + 1, statistics.err_sq);
    atomicAdd(params.statistics + offset + 2, statistics.under);
    atomicAdd(params.statistics + offset + 3, statistics.nonzero);
  }
}

template <typename input_t>
__global__ void rowwise_partial_amax_kernel(
    QuantizeParams params,
    float* partials,
    int64_t split_count,
    int64_t blocks_per_row) {
  constexpr int kWarps = kThreads / 32;
  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  const int64_t outer_tiles = (params.outer_size + 31) / 32;
  const int64_t tile = grid_index(blocks_per_row) * kWarps + warp;
  const int64_t tile_count = params.batches * split_count * outer_tiles;
  if (tile >= tile_count) {
    return;
  }
  const int64_t outer_tile = tile % outer_tiles;
  const int64_t remaining = tile / outer_tiles;
  const int64_t split = remaining % split_count;
  const int64_t batch = remaining / split_count;
  const int64_t outer = outer_tile * 32 + lane;
  if (outer >= params.outer_size) {
    return;
  }
  const int64_t contract_start = split * kRowwiseSplit;
  const int64_t contract_end = min(contract_start + kRowwiseSplit, params.contract_size);
  float maximum = 0.0f;
  for (int64_t contract = contract_start; contract < contract_end; ++contract) {
    const float value = static_cast<float>(static_cast<const input_t*>(params.input)[
        input_offset(params, batch, contract, outer)]);
    maximum = max_with_nan(maximum, fabsf(value));
  }
  partials[(batch * split_count + split) * params.outer_size + outer] = maximum;
}

template <bool kE5m2>
__global__ void rowwise_scale_kernel(
    QuantizeParams params,
    const float* partials,
    int64_t split_count) {
  const int64_t outer = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t batch = blockIdx.y;
  if (outer >= params.outer_size) {
    return;
  }
  float maximum = 0.0f;
  for (int64_t split = 0; split < split_count; ++split) {
    maximum = max_with_nan(
        maximum, partials[(batch * split_count + split) * params.outer_size + outer]);
  }
  params.scales[scale_offset(params, batch, outer, 0)] = scale_from_amax(
      maximum, kE5m2 ? 57344.0f : 448.0f);
}

template <typename input_t, bool kE5m2, bool kStochastic>
__global__ void rowwise_encode_kernel(
    QuantizeParams params,
    at::PhiloxCudaState philox,
    int64_t split_count,
    int64_t blocks_per_row) {
  constexpr int kWarps = kThreads / 32;
  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  const int64_t outer_tiles = (params.outer_size + 31) / 32;
  const int64_t tile = grid_index(blocks_per_row) * kWarps + warp;
  const int64_t tile_count = params.batches * split_count * outer_tiles;
  if (tile >= tile_count) {
    return;
  }
  const int64_t outer_tile = tile % outer_tiles;
  const int64_t remaining = tile / outer_tiles;
  const int64_t split = remaining % split_count;
  const int64_t batch = remaining / split_count;
  const int64_t outer = outer_tile * 32 + lane;
  if (outer >= params.outer_size) {
    return;
  }
  const float scale = params.scales[scale_offset(params, batch, outer, 0)];
  const int64_t contract_start = split * kRowwiseSplit;
  const int64_t contract_end = min(contract_start + kRowwiseSplit, params.contract_size);
  curandStatePhilox4_32_10_t random_state;
  StatisticsPartial statistics;
  if constexpr (kStochastic) {
    const auto seeds = at::cuda::philox::unpack(philox);
    curand_init(
        static_cast<unsigned long long>(std::get<0>(seeds)),
        static_cast<unsigned long long>(tile * 32 + lane),
        static_cast<unsigned long long>(std::get<1>(seeds)),
        &random_state);
  }
  constexpr float qmax = kE5m2 ? 57344.0f : 448.0f;
  for (int64_t contract = contract_start; contract < contract_end; ++contract) {
    const float source = static_cast<float>(static_cast<const input_t*>(params.input)[
        input_offset(params, batch, contract, outer)]);
    float value = source;
    value /= scale;
    if (!isnan(value)) {
      value = fminf(fmaxf(value, -qmax), qmax);
    }
    const int64_t code_offset = output_offset(params, batch, contract, outer);
    uint8_t code;
    if constexpr (kStochastic) code = fp8_stochastic<kE5m2>(value, &random_state);
    else code = fp8_rne<kE5m2>(value);
    params.codes[code_offset] = code;
    if (params.statistics != nullptr) accumulate_statistics<kE5m2>(&statistics, source, code, scale);
  }
  if (params.statistics != nullptr) {
    const int64_t offset = (static_cast<int64_t>(blockIdx.y) * gridDim.x + blockIdx.x) * 4;
    atomicAdd(params.statistics + offset, statistics.src_sq);
    atomicAdd(params.statistics + offset + 1, statistics.err_sq);
    atomicAdd(params.statistics + offset + 2, statistics.under);
    atomicAdd(params.statistics + offset + 3, statistics.nonzero);
  }
}

template <typename input_t>
__global__ void tensorwise_partial_amax_kernel(
    QuantizeParams params, float* partials, int64_t partials_per_batch) {
  const int64_t partial = static_cast<int64_t>(blockIdx.x);
  const int64_t batch = static_cast<int64_t>(blockIdx.y);
  int64_t index =
      (partial * kThreads + threadIdx.x) * kTensorwiseValues;
  const int64_t elements = params.rows * params.cols;
  float maximum = 0.0f;
  int64_t row = index / params.cols;
  int64_t col = index - row * params.cols;
  #pragma unroll
  for (int value_index = 0; value_index < kTensorwiseValues; ++value_index) {
    if (index < elements) {
      const float value = static_cast<float>(static_cast<const input_t*>(params.input)[
          input_offset(params, batch, row, col)]);
      maximum = max_with_nan(maximum, fabsf(value));
      ++col;
      if (col == params.cols) {
        col = 0;
        ++row;
      }
      ++index;
    }
  }
  constexpr unsigned int kWarpMask = 0xffffffff;
  for (int offset = 16; offset > 0; offset >>= 1) {
    maximum = max_with_nan(maximum, __shfl_down_sync(kWarpMask, maximum, offset));
  }
  __shared__ float warp_maximum[kThreads / 32];
  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  if (lane == 0) {
    warp_maximum[warp] = maximum;
  }
  __syncthreads();
  if (warp == 0) {
    maximum = lane < kThreads / 32 ? warp_maximum[lane] : 0.0f;
    for (int offset = 16; offset > 0; offset >>= 1) {
      maximum = max_with_nan(
          maximum, __shfl_down_sync(kWarpMask, maximum, offset));
    }
    if (lane == 0) {
      partials[batch * partials_per_batch + partial] = maximum;
    }
  }
}

template <typename input_t>
__global__ void tensorwise_partial_amax_dense_kernel(
    const input_t* input,
    float* partials,
    int elements,
    int64_t partials_per_batch) {
  const int64_t batch = blockIdx.y;
  const int partial = blockIdx.x;
  const int index = (partial * kThreads + threadIdx.x) * kTensorwiseValues;
  float maximum = 0.0f;
  if (index < elements) {
    float values[kTensorwiseValues];
    load_contiguous_four(input + batch * static_cast<int64_t>(elements), index, values);
    load_contiguous_four(
        input + batch * static_cast<int64_t>(elements), index + 4, values + 4);
    #pragma unroll
    for (int value_index = 0; value_index < kTensorwiseValues; ++value_index) {
      maximum = max_with_nan(maximum, fabsf(values[value_index]));
    }
  }
  constexpr unsigned int kWarpMask = 0xffffffff;
  for (int offset = 16; offset > 0; offset >>= 1) {
    maximum = max_with_nan(maximum, __shfl_down_sync(kWarpMask, maximum, offset));
  }
  __shared__ float warp_maximum[kThreads / 32];
  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  if (lane == 0) {
    warp_maximum[warp] = maximum;
  }
  __syncthreads();
  if (warp == 0) {
    maximum = lane < kThreads / 32 ? warp_maximum[lane] : 0.0f;
    for (int offset = 16; offset > 0; offset >>= 1) {
      maximum = max_with_nan(
          maximum, __shfl_down_sync(kWarpMask, maximum, offset));
    }
    if (lane == 0) {
      partials[batch * partials_per_batch + partial] = maximum;
    }
  }
}

template <bool kE5m2>
__global__ void tensorwise_scale_kernel(
    const float* partials,
    float* matrix_scales,
    int64_t partials_per_batch) {
  const int64_t batch = blockIdx.x;
  float maximum = 0.0f;
  for (int64_t index = threadIdx.x; index < partials_per_batch; index += blockDim.x) {
    maximum = max_with_nan(maximum, partials[batch * partials_per_batch + index]);
  }
  constexpr unsigned int kWarpMask = 0xffffffff;
  for (int offset = 16; offset > 0; offset >>= 1) {
    maximum = max_with_nan(maximum, __shfl_down_sync(kWarpMask, maximum, offset));
  }
  __shared__ float warp_maximum[kThreads / 32];
  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  if (lane == 0) {
    warp_maximum[warp] = maximum;
  }
  __syncthreads();
  if (warp == 0) {
    maximum = lane < kThreads / 32 ? warp_maximum[lane] : 0.0f;
    for (int offset = 16; offset > 0; offset >>= 1) {
      maximum = max_with_nan(
          maximum, __shfl_down_sync(kWarpMask, maximum, offset));
    }
    if (lane == 0) {
      matrix_scales[batch] = scale_from_amax(
          maximum, kE5m2 ? 57344.0f : 448.0f);
    }
  }
}

template <typename input_t, bool kE5m2, bool kStochastic>
__global__ void tensorwise_encode_kernel(
    QuantizeParams params,
    const float* matrix_scales,
    at::PhiloxCudaState philox) {
  const int64_t batch = blockIdx.y;
  const int64_t elements = params.rows * params.cols;
  int64_t index =
      (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) * kTensorwiseValues;
  int64_t row = index / params.cols;
  int64_t col = index - row * params.cols;
  constexpr float qmax = kE5m2 ? 57344.0f : 448.0f;
  const float scale = matrix_scales[batch];
  curandStatePhilox4_32_10_t random_state;
  StatisticsPartial statistics;
  if constexpr (kStochastic) {
    const auto seeds = at::cuda::philox::unpack(philox);
    curand_init(
        static_cast<unsigned long long>(std::get<0>(seeds)),
        static_cast<unsigned long long>(
            (static_cast<int64_t>(blockIdx.y) * gridDim.x + blockIdx.x) * blockDim.x +
            threadIdx.x),
        static_cast<unsigned long long>(std::get<1>(seeds)),
        &random_state);
  }
  #pragma unroll
  for (int value_index = 0; value_index < kTensorwiseValues; ++value_index) {
    if (index < elements) {
      const float source = static_cast<float>(static_cast<const input_t*>(params.input)[
          input_offset(params, batch, row, col)]);
      float value = source;
      value /= scale;
      if (!isnan(value)) {
        value = fminf(fmaxf(value, -qmax), qmax);
      }
      const int64_t code_offset = output_offset(params, batch, row, col);
      uint8_t code;
      if constexpr (kStochastic) code = fp8_stochastic<kE5m2>(value, &random_state);
      else code = fp8_rne<kE5m2>(value);
      params.codes[code_offset] = code;
      if (params.statistics != nullptr) accumulate_statistics<kE5m2>(&statistics, source, code, scale);
      ++col;
      if (col == params.cols) {
        col = 0;
        ++row;
      }
      ++index;
    }
  }
  reduce_statistics(statistics, params.statistics,
                    params.batches * params.rows * params.cols);
}

template <typename input_t, bool kE5m2, bool kStochastic>
__global__ void tensorwise_encode_dense_kernel(
    const input_t* input,
    uint8_t* codes,
    const float* matrix_scales,
    int elements,
    at::PhiloxCudaState philox,
    float* statistics,
    int64_t numel) {
  const int64_t batch = blockIdx.y;
  const int index = (blockIdx.x * blockDim.x + threadIdx.x) * kTensorwiseValues;
  StatisticsPartial partial;
  if (index < elements) {
    float values[kTensorwiseValues];
    const input_t* input_batch = input + batch * static_cast<int64_t>(elements);
    load_contiguous_four(input_batch, index, values);
    load_contiguous_four(input_batch, index + 4, values + 4);
    const float scale = matrix_scales[batch];
    if constexpr (kStochastic) {
    curandStatePhilox4_32_10_t random_state;
    const auto seeds = at::cuda::philox::unpack(philox);
    curand_init(
        static_cast<unsigned long long>(std::get<0>(seeds)),
        static_cast<unsigned long long>(
            (static_cast<int64_t>(blockIdx.y) * gridDim.x + blockIdx.x) * blockDim.x +
            threadIdx.x),
        static_cast<unsigned long long>(std::get<1>(seeds)),
        &random_state);
    constexpr float qmax = kE5m2 ? 57344.0f : 448.0f;
    #pragma unroll
    for (int value_index = 0; value_index < kTensorwiseValues; ++value_index) {
      float value = values[value_index] / scale;
      if (!isnan(value)) {
        value = fminf(fmaxf(value, -qmax), qmax);
      }
      const uint8_t code = fp8_stochastic<kE5m2>(value, &random_state);
      codes[batch * static_cast<int64_t>(elements) + index + value_index] = code;
      if (statistics != nullptr) {
        accumulate_statistics<kE5m2>(&partial, values[value_index], code, scale);
      }
    }
    } else {
    const float inverse_scale = __frcp_rn(scale);
    float source_values[kTensorwiseValues];
    if (statistics != nullptr) {
      #pragma unroll
      for (int value_index = 0; value_index < kTensorwiseValues; ++value_index) {
        source_values[value_index] = values[value_index];
      }
    }
    #pragma unroll
    for (int value_index = 0; value_index < kTensorwiseValues; ++value_index) {
      values[value_index] = normalize_fp8_rne<kE5m2>(
          values[value_index], scale, inverse_scale);
    }
    uint8_t* output = codes + batch * static_cast<int64_t>(elements) + index;
    reinterpret_cast<uint32_t*>(output)[0] = fp8x4_rne<kE5m2>(
        values[0], values[1], values[2], values[3]);
    reinterpret_cast<uint32_t*>(output + 4)[0] = fp8x4_rne<kE5m2>(
        values[4], values[5], values[6], values[7]);
    if (statistics != nullptr) {
      #pragma unroll
      for (int value_index = 0; value_index < kTensorwiseValues; ++value_index) {
        accumulate_statistics<kE5m2>(
            &partial, source_values[value_index], output[value_index], scale);
      }
    }
    }
  }
  if (statistics != nullptr) {
    reduce_statistics(partial, statistics, numel);
  }
}

std::vector<int64_t> scale_shape(
    const at::Tensor& input,
    int64_t contract_dim,
    int64_t block_outer,
    int64_t block_contract) {
  std::vector<int64_t> shape(input.sizes().begin(), input.sizes().end());
  const int64_t axis = contract_dim + input.dim();
  const int64_t outer_axis = axis == input.dim() - 1 ? axis - 1 : axis + 1;
  if (block_outer == 0 && block_contract == 0) {
    shape[axis] = 1;
    return shape;
  }
  if (block_outer > 1) {
    shape[outer_axis] = input.size(outer_axis);
  }
  if (block_contract == 0) {
    shape[axis] = 1;
  } else {
    shape[axis] = (input.size(axis) + block_contract - 1) / block_contract;
  }
  return shape;
}

std::vector<int64_t> tensorwise_storage_shape(
    const at::Tensor& input, int64_t contract_dim) {
  std::vector<int64_t> shape(input.sizes().begin(), input.sizes().end());
  const int64_t axis = contract_dim + input.dim();
  const int64_t outer_axis = axis == input.dim() - 1 ? axis - 1 : axis + 1;
  shape[axis] = 1;
  shape[outer_axis] = 1;
  return shape;
}

void check_arguments(
    const at::Tensor& input,
    int64_t contract_dim,
    at::ScalarType dtype,
    int64_t block_outer,
    int64_t block_contract) {
  TORCH_CHECK(input.dim() == 2 || input.dim() == 3,
              "quantize_fp8 requires a 2D or 3D tensor");
  TORCH_CHECK(input.scalar_type() == at::kFloat || input.scalar_type() == at::kHalf ||
                  input.scalar_type() == at::kBFloat16,
              "quantize_fp8 requires float32, float16, or bfloat16 input");
  TORCH_CHECK(contract_dim == -2 || contract_dim == -1,
              "quantize_fp8 contract_dim must be -2 or -1");
  TORCH_CHECK(dtype == at::kFloat8_e4m3fn || dtype == at::kFloat8_e5m2,
              "quantize_fp8 dtype must be float8_e4m3fn or float8_e5m2");
  const bool tensorwise = block_outer == 0 && block_contract == 0;
  const bool rowwise = block_outer == 1 && block_contract == 0;
  TORCH_CHECK(tensorwise || rowwise || (block_outer > 0 && block_contract > 0),
              "quantize_fp8 block shape must be (0, 0), (1, 0), or positive");
  TORCH_CHECK(input.numel() > 0, "quantize_fp8 requires a nonempty tensor");
  TORCH_CHECK(block_outer <= 1 || block_outer == block_contract,
              "quantize_fp8 2D blocks must be square");
  TORCH_CHECK(std::all_of(input.strides().begin(), input.strides().end(),
                          [](int64_t stride) { return stride >= 0; }),
              "quantize_fp8 requires nonnegative strides");
}

std::vector<int64_t> code_strides(
    const at::Tensor& input, const std::string& output_layout) {
  TORCH_CHECK(output_layout == "row_major" || output_layout == "column_major",
              "quantize_fp8 output_layout must be row_major or column_major");
  const int64_t rows = input.size(-2);
  const int64_t cols = input.size(-1);
  if (output_layout == "row_major") {
    if (input.dim() == 2) return {cols, 1};
    return {rows * cols, cols, 1};
  }
  if (input.dim() == 2) return {1, rows};
  return {rows * cols, 1, rows};
}

template <typename input_t, bool kE5m2, bool kStochastic>
void launch_tensorwise(
    const at::Tensor& scales,
    const at::Tensor& partials,
    QuantizeParams params,
    at::PhiloxCudaState philox,
    cudaStream_t stream) {
  const int64_t elements_per_batch = params.rows * params.cols;
  const int64_t partials_per_batch =
      (elements_per_batch + kThreads * kTensorwiseValues - 1) /
      (kThreads * kTensorwiseValues);
  const bool dense =
      elements_per_batch <= std::numeric_limits<int>::max() &&
      elements_per_batch % kTensorwiseValues == 0 &&
      params.input_col_stride == 1 && params.input_row_stride == params.cols &&
      (params.batches == 1 || params.input_batch_stride == elements_per_batch) &&
      params.code_col_stride == 1 && params.code_row_stride == params.cols &&
      (params.batches == 1 || params.code_batch_stride == elements_per_batch) &&
      reinterpret_cast<uintptr_t>(params.input) %
              (sizeof(input_t) == sizeof(float) ? alignof(float4) : alignof(uint2)) ==
          0 &&
      reinterpret_cast<uintptr_t>(params.codes) % alignof(uint32_t) == 0;
  if (dense) {
    tensorwise_partial_amax_dense_kernel<input_t>
        <<<dim3(partials_per_batch, params.batches), kThreads, 0, stream>>>(
            static_cast<const input_t*>(params.input),
            partials.data_ptr<float>(), static_cast<int>(elements_per_batch),
            partials_per_batch);
    tensorwise_scale_kernel<kE5m2><<<params.batches, kThreads, 0, stream>>>(
        partials.data_ptr<float>(), scales.data_ptr<float>(), partials_per_batch);
    const int64_t code_blocks =
        (elements_per_batch + kThreads * kTensorwiseValues - 1) /
        (kThreads * kTensorwiseValues);
    tensorwise_encode_dense_kernel<input_t, kE5m2, kStochastic>
        <<<dim3(code_blocks, params.batches), kThreads, 0, stream>>>(
            static_cast<const input_t*>(params.input), params.codes,
            scales.data_ptr<float>(), static_cast<int>(elements_per_batch), philox,
            params.statistics, params.batches * elements_per_batch);
    return;
  }
  tensorwise_partial_amax_kernel<input_t><<<dim3(partials_per_batch, params.batches),
                                             kThreads, 0, stream>>>(
      params, partials.data_ptr<float>(), partials_per_batch);
  tensorwise_scale_kernel<kE5m2><<<params.batches, kThreads, 0, stream>>>(
      partials.data_ptr<float>(), scales.data_ptr<float>(), partials_per_batch);
  const int64_t code_blocks =
      (elements_per_batch + kThreads * kTensorwiseValues - 1) /
      (kThreads * kTensorwiseValues);
  tensorwise_encode_kernel<input_t, kE5m2, kStochastic>
      <<<dim3(code_blocks, params.batches), kThreads, 0, stream>>>(
          params, scales.data_ptr<float>(), philox);
}

template <typename input_t, bool kE5m2, bool kStochastic>
void launch_blockwise1d(
    QuantizeParams params,
    at::PhiloxCudaState philox,
    cudaStream_t stream) {
  const int64_t tiles = params.batches * params.outer_groups * params.contract_groups;
  if (!params.contract_is_col) {
    constexpr int kWarps = kThreads / 32;
    const int64_t outer_tiles = (params.outer_size + 31) / 32;
    const int64_t groups = params.batches * params.contract_groups * outer_tiles;
    const LaunchGrid launch = make_launch_grid((groups + kWarps - 1) / kWarps);
    quantize_rows_coalesced_kernel<input_t, kE5m2, kStochastic>
        <<<launch.grid, kThreads, 0, stream>>>(params, philox, launch.blocks_per_row);
    return;
  }
  constexpr int kWarps = kThreads / 32;
  if (params.block_contract == 16) {
    const LaunchGrid launch = make_launch_grid((tiles + kWarps - 1) / kWarps);
    quantize_blockwise_1d_packed_kernel<input_t, kE5m2, kStochastic, 16>
        <<<launch.grid, kThreads, 0, stream>>>(params, philox, launch.blocks_per_row);
    return;
  }
  if (params.block_contract == 32) {
    const LaunchGrid launch = make_launch_grid((tiles + kWarps - 1) / kWarps);
    quantize_blockwise_1d_packed_kernel<input_t, kE5m2, kStochastic, 32>
        <<<launch.grid, kThreads, 0, stream>>>(params, philox, launch.blocks_per_row);
    return;
  }
  if (params.block_contract == 64) {
    const LaunchGrid launch = make_launch_grid((tiles + kWarps - 1) / kWarps);
    quantize_blockwise_1d_packed_kernel<input_t, kE5m2, kStochastic, 64>
        <<<launch.grid, kThreads, 0, stream>>>(params, philox, launch.blocks_per_row);
    return;
  }
  if (params.block_contract == 128) {
    const LaunchGrid launch = make_launch_grid((tiles + kWarps - 1) / kWarps);
    quantize_blockwise_1d_packed_kernel<input_t, kE5m2, kStochastic, 128>
        <<<launch.grid, kThreads, 0, stream>>>(params, philox, launch.blocks_per_row);
    return;
  }
  if (params.block_contract == 256) {
    const LaunchGrid launch = make_launch_grid((tiles + kWarps - 1) / kWarps);
    quantize_blockwise_1d_packed_kernel<input_t, kE5m2, kStochastic, 256>
        <<<launch.grid, kThreads, 0, stream>>>(params, philox, launch.blocks_per_row);
    return;
  }
  const LaunchGrid launch = make_launch_grid(tiles);
  quantize_tiled_kernel<input_t, kE5m2, kStochastic>
      <<<launch.grid, kThreads, 0, stream>>>(params, philox, launch.blocks_per_row);
}

template <typename input_t, bool kE5m2, bool kStochastic>
void launch_blockwise1d_contiguous(
    const at::Tensor& input,
    const at::Tensor& codes,
    const at::Tensor& scales,
    int block_contract,
    at::PhiloxCudaState philox,
    cudaStream_t stream,
    float* statistics) {
  constexpr int kWarps = kThreads / 32;
  const int rows = static_cast<int>(input.numel() / input.size(-1));
  const int cols = static_cast<int>(input.size(-1));
  const int contract_groups = cols / block_contract;
  const int tiles = rows * contract_groups;

  if (block_contract == 16) {
    constexpr int kTilesPerWarp = 8;
    const LaunchGrid launch = make_launch_grid(
        (tiles + kWarps * kTilesPerWarp - 1) / (kWarps * kTilesPerWarp));
    quantize_blockwise_1d_contiguous_kernel<input_t, kE5m2, kStochastic, 16>
        <<<launch.grid, kThreads, 0, stream>>>(
            input.const_data_ptr<input_t>(),
            reinterpret_cast<uint8_t*>(codes.data_ptr()),
            scales.data_ptr<float>(), rows, cols, contract_groups, philox,
            static_cast<int>(launch.blocks_per_row), statistics, input.numel());
    return;
  }
  if (block_contract == 32) {
    constexpr int kTilesPerWarp = 4;
    const LaunchGrid launch = make_launch_grid(
        (tiles + kWarps * kTilesPerWarp - 1) / (kWarps * kTilesPerWarp));
    quantize_blockwise_1d_contiguous_kernel<input_t, kE5m2, kStochastic, 32>
        <<<launch.grid, kThreads, 0, stream>>>(
            input.const_data_ptr<input_t>(),
            reinterpret_cast<uint8_t*>(codes.data_ptr()),
            scales.data_ptr<float>(), rows, cols, contract_groups, philox,
            static_cast<int>(launch.blocks_per_row), statistics, input.numel());
    return;
  }
  if (block_contract == 64) {
    constexpr int kTilesPerWarp = 2;
    const LaunchGrid launch = make_launch_grid(
        (tiles + kWarps * kTilesPerWarp - 1) / (kWarps * kTilesPerWarp));
    quantize_blockwise_1d_contiguous_kernel<input_t, kE5m2, kStochastic, 64>
        <<<launch.grid, kThreads, 0, stream>>>(
            input.const_data_ptr<input_t>(),
            reinterpret_cast<uint8_t*>(codes.data_ptr()),
            scales.data_ptr<float>(), rows, cols, contract_groups, philox,
            static_cast<int>(launch.blocks_per_row), statistics, input.numel());
    return;
  }
  if (block_contract == 128) {
    const LaunchGrid launch = make_launch_grid((tiles + kWarps - 1) / kWarps);
    quantize_blockwise_1d_contiguous_kernel<input_t, kE5m2, kStochastic, 128>
        <<<launch.grid, kThreads, 0, stream>>>(
            input.const_data_ptr<input_t>(),
            reinterpret_cast<uint8_t*>(codes.data_ptr()),
            scales.data_ptr<float>(), rows, cols, contract_groups, philox,
            static_cast<int>(launch.blocks_per_row), statistics, input.numel());
    return;
  }
  if (block_contract == 256) {
    const LaunchGrid launch = make_launch_grid((tiles + kWarps - 1) / kWarps);
    quantize_blockwise_1d_contiguous_kernel<input_t, kE5m2, kStochastic, 256>
        <<<launch.grid, kThreads, 0, stream>>>(
            input.const_data_ptr<input_t>(),
            reinterpret_cast<uint8_t*>(codes.data_ptr()),
            scales.data_ptr<float>(), rows, cols, contract_groups, philox,
            static_cast<int>(launch.blocks_per_row), statistics, input.numel());
  }
}

template <typename input_t, bool kE5m2>
void launch_blockwise1d_transposed_rne(
    const at::Tensor& input,
    const at::Tensor& codes,
    const at::Tensor& scales,
    int block_contract,
    cudaStream_t stream,
    float* statistics,
    int64_t numel) {
  const int batches = input.dim() == 3 ? static_cast<int>(input.size(0)) : 1;
  const int rows = static_cast<int>(input.size(-2));
  const int cols = static_cast<int>(input.size(-1));
  const int contract_groups = cols / block_contract;
  const int outer_tiles = (rows + 31) / 32;
  const int tiles = batches * contract_groups * outer_tiles;
  const LaunchGrid launch = make_launch_grid(tiles);
  if (block_contract == 16) {
    quantize_blockwise_1d_transposed_rne_kernel<input_t, kE5m2, 16>
        <<<launch.grid, kThreads, 0, stream>>>(
            input.const_data_ptr<input_t>(), reinterpret_cast<uint8_t*>(codes.data_ptr()),
            scales.data_ptr<float>(), batches, rows, cols, contract_groups,
            static_cast<int>(launch.blocks_per_row), statistics, numel);
  } else if (block_contract == 32) {
    quantize_blockwise_1d_transposed_rne_kernel<input_t, kE5m2, 32>
        <<<launch.grid, kThreads, 0, stream>>>(
            input.const_data_ptr<input_t>(), reinterpret_cast<uint8_t*>(codes.data_ptr()),
            scales.data_ptr<float>(), batches, rows, cols, contract_groups,
            static_cast<int>(launch.blocks_per_row), statistics, numel);
  } else if (block_contract == 64) {
    quantize_blockwise_1d_transposed_rne_kernel<input_t, kE5m2, 64>
        <<<launch.grid, kThreads, 0, stream>>>(
            input.const_data_ptr<input_t>(), reinterpret_cast<uint8_t*>(codes.data_ptr()),
            scales.data_ptr<float>(), batches, rows, cols, contract_groups,
            static_cast<int>(launch.blocks_per_row), statistics, numel);
  } else if (block_contract == 128) {
    quantize_blockwise_1d_transposed_rne_kernel<input_t, kE5m2, 128>
        <<<launch.grid, kThreads, 0, stream>>>(
            input.const_data_ptr<input_t>(), reinterpret_cast<uint8_t*>(codes.data_ptr()),
            scales.data_ptr<float>(), batches, rows, cols, contract_groups,
            static_cast<int>(launch.blocks_per_row), statistics, numel);
  } else {
    quantize_blockwise_1d_transposed_rne_kernel<input_t, kE5m2, 256>
        <<<launch.grid, kThreads, 0, stream>>>(
            input.const_data_ptr<input_t>(), reinterpret_cast<uint8_t*>(codes.data_ptr()),
            scales.data_ptr<float>(), batches, rows, cols, contract_groups,
            static_cast<int>(launch.blocks_per_row), statistics, numel);
  }
}

template <typename input_t, bool kE5m2>
void launch_blockwise1d_transposed_column_rne(
    const at::Tensor& input,
    const at::Tensor& codes,
    const at::Tensor& scales,
    int block_contract,
    cudaStream_t stream,
    float* statistics,
    int64_t numel) {
  const int batches = input.dim() == 3 ? static_cast<int>(input.size(0)) : 1;
  const int rows = static_cast<int>(input.size(-2));
  const int cols = static_cast<int>(input.size(-1));
  const int contract_groups = rows / block_contract;
  const int outer_tiles = (cols + 31) / 32;
  const int tiles = batches * contract_groups * outer_tiles;
  const LaunchGrid launch = make_launch_grid(tiles);
  if (block_contract == 16) {
    quantize_blockwise_1d_tiled_column_rne_kernel<input_t, kE5m2, 16>
        <<<launch.grid, kThreads, 0, stream>>>(
            input.const_data_ptr<input_t>(), reinterpret_cast<uint8_t*>(codes.data_ptr()),
            scales.data_ptr<float>(), batches, rows, cols, contract_groups,
            static_cast<int>(launch.blocks_per_row), statistics, numel);
  } else if (block_contract == 32) {
    quantize_blockwise_1d_tiled_column_rne_kernel<input_t, kE5m2, 32>
        <<<launch.grid, kThreads, 0, stream>>>(
            input.const_data_ptr<input_t>(), reinterpret_cast<uint8_t*>(codes.data_ptr()),
            scales.data_ptr<float>(), batches, rows, cols, contract_groups,
            static_cast<int>(launch.blocks_per_row), statistics, numel);
  } else if (block_contract == 64) {
    quantize_blockwise_1d_tiled_column_rne_kernel<input_t, kE5m2, 64>
        <<<launch.grid, kThreads, 0, stream>>>(
            input.const_data_ptr<input_t>(), reinterpret_cast<uint8_t*>(codes.data_ptr()),
            scales.data_ptr<float>(), batches, rows, cols, contract_groups,
            static_cast<int>(launch.blocks_per_row), statistics, numel);
  } else if (block_contract == 128) {
    quantize_blockwise_1d_tiled_column_rne_kernel<input_t, kE5m2, 128>
        <<<launch.grid, kThreads, 0, stream>>>(
            input.const_data_ptr<input_t>(), reinterpret_cast<uint8_t*>(codes.data_ptr()),
            scales.data_ptr<float>(), batches, rows, cols, contract_groups,
            static_cast<int>(launch.blocks_per_row), statistics, numel);
  } else {
    quantize_blockwise_1d_tiled_column_rne_kernel<input_t, kE5m2, 256>
        <<<launch.grid, kThreads, 0, stream>>>(
            input.const_data_ptr<input_t>(), reinterpret_cast<uint8_t*>(codes.data_ptr()),
            scales.data_ptr<float>(), batches, rows, cols, contract_groups,
            static_cast<int>(launch.blocks_per_row), statistics, numel);
  }
}

template <typename input_t>
void launch_blockwise1d_transposed_rne(
    const at::Tensor& input,
    const at::Tensor& codes,
    const at::Tensor& scales,
    int block_contract,
    bool e5m2,
    cudaStream_t stream,
    float* statistics,
    int64_t numel) {
  if (e5m2) {
    launch_blockwise1d_transposed_rne<input_t, true>(
        input, codes, scales, block_contract, stream, statistics, numel);
  } else {
    launch_blockwise1d_transposed_rne<input_t, false>(
        input, codes, scales, block_contract, stream, statistics, numel);
  }
}

template <typename input_t>
void launch_blockwise1d_transposed_column_rne(
    const at::Tensor& input,
    const at::Tensor& codes,
    const at::Tensor& scales,
    int block_contract,
    bool e5m2,
    cudaStream_t stream,
    float* statistics,
    int64_t numel) {
  if (e5m2) {
    launch_blockwise1d_transposed_column_rne<input_t, true>(
        input, codes, scales, block_contract, stream, statistics, numel);
  } else {
    launch_blockwise1d_transposed_column_rne<input_t, false>(
        input, codes, scales, block_contract, stream, statistics, numel);
  }
}

template <typename input_t, bool kE5m2, bool kStochastic>
void launch_rowwise(
    QuantizeParams params,
    const at::Tensor& partials,
    at::PhiloxCudaState philox,
    cudaStream_t stream) {
  if (params.contract_is_col) {
    const int64_t tiles = params.batches * params.outer_size;
    const LaunchGrid launch = make_launch_grid(tiles);
    quantize_tiled_kernel<input_t, kE5m2, kStochastic>
        <<<launch.grid, kThreads, 0, stream>>>(params, philox, launch.blocks_per_row);
    return;
  }
  if (params.contract_size < 1024) {
    constexpr int kWarps = kThreads / 32;
    const int64_t outer_tiles = (params.outer_size + 31) / 32;
    const int64_t groups = params.batches * outer_tiles;
    const LaunchGrid launch = make_launch_grid((groups + kWarps - 1) / kWarps);
    quantize_rows_coalesced_kernel<input_t, kE5m2, kStochastic>
        <<<launch.grid, kThreads, 0, stream>>>(params, philox, launch.blocks_per_row);
    return;
  }
  constexpr int kWarps = kThreads / 32;
  const int64_t split_count = (params.contract_size + kRowwiseSplit - 1) / kRowwiseSplit;
  const int64_t outer_tiles = (params.outer_size + 31) / 32;
  const int64_t groups = params.batches * split_count * outer_tiles;
  const LaunchGrid partial_launch = make_launch_grid((groups + kWarps - 1) / kWarps);
  rowwise_partial_amax_kernel<input_t><<<partial_launch.grid, kThreads, 0, stream>>>(
      params, partials.data_ptr<float>(), split_count, partial_launch.blocks_per_row);
  const int64_t scale_blocks = (params.outer_size + kThreads - 1) / kThreads;
  rowwise_scale_kernel<kE5m2><<<dim3(scale_blocks, params.batches), kThreads, 0, stream>>>(
      params, partials.data_ptr<float>(), split_count);
  rowwise_encode_kernel<input_t, kE5m2, kStochastic>
      <<<partial_launch.grid, kThreads, 0, stream>>>(
          params, philox, split_count, partial_launch.blocks_per_row);
}

template <typename input_t, bool kE5m2, bool kStochastic>
void launch_blockwise2d(
    QuantizeParams params,
    at::PhiloxCudaState philox,
    cudaStream_t stream) {
  const int64_t tiles = params.batches * params.outer_groups * params.contract_groups;
  const LaunchGrid launch = make_launch_grid(tiles);
  quantize_tiled_kernel<input_t, kE5m2, kStochastic>
      <<<launch.grid, kThreads, 0, stream>>>(params, philox, launch.blocks_per_row);
}

template <typename input_t, bool kE5m2, bool kStochastic>
void launch_quantize_impl(
    const at::Tensor& input,
    const at::Tensor& codes,
    const at::Tensor& scales,
    const at::Tensor& partials,
    QuantizeParams params,
    at::PhiloxCudaState philox,
    cudaStream_t stream,
    bool tensorwise,
    bool rowwise) {
  params.input = input.const_data_ptr<input_t>();
  params.codes = reinterpret_cast<uint8_t*>(codes.data_ptr());
  params.scales = scales.data_ptr<float>();
  if (tensorwise) {
    launch_tensorwise<input_t, kE5m2, kStochastic>(
        scales, partials, params, philox, stream);
  } else if (rowwise) {
    launch_rowwise<input_t, kE5m2, kStochastic>(params, partials, philox, stream);
  } else if (params.block_outer == 1) {
    launch_blockwise1d<input_t, kE5m2, kStochastic>(params, philox, stream);
  } else {
    launch_blockwise2d<input_t, kE5m2, kStochastic>(params, philox, stream);
  }
}

template <typename input_t>
void launch_quantize(
    const at::Tensor& input,
    const at::Tensor& codes,
    const at::Tensor& scales,
    const at::Tensor& partials,
    QuantizeParams params,
    bool e5m2,
    bool stochastic_rounding,
    at::PhiloxCudaState philox,
    cudaStream_t stream,
    bool tensorwise,
    bool rowwise) {
  if (e5m2) {
    if (stochastic_rounding) {
      launch_quantize_impl<input_t, true, true>(
          input, codes, scales, partials, params, philox, stream, tensorwise, rowwise);
    } else {
      launch_quantize_impl<input_t, true, false>(
          input, codes, scales, partials, params, philox, stream, tensorwise, rowwise);
    }
  } else if (stochastic_rounding) {
    launch_quantize_impl<input_t, false, true>(
        input, codes, scales, partials, params, philox, stream, tensorwise, rowwise);
  } else {
    launch_quantize_impl<input_t, false, false>(
        input, codes, scales, partials, params, philox, stream, tensorwise, rowwise);
  }
}

template <typename input_t>
void launch_blockwise1d_contiguous(
    const at::Tensor& input,
    const at::Tensor& codes,
    const at::Tensor& scales,
    int block_contract,
    bool e5m2,
    bool stochastic_rounding,
    at::PhiloxCudaState philox,
    cudaStream_t stream,
    float* statistics) {
  if (e5m2) {
    if (stochastic_rounding) {
      launch_blockwise1d_contiguous<input_t, true, true>(
          input, codes, scales, block_contract, philox, stream, statistics);
    } else {
      launch_blockwise1d_contiguous<input_t, true, false>(
          input, codes, scales, block_contract, philox, stream, statistics);
    }
  } else if (stochastic_rounding) {
    launch_blockwise1d_contiguous<input_t, false, true>(
        input, codes, scales, block_contract, philox, stream, statistics);
  } else {
    launch_blockwise1d_contiguous<input_t, false, false>(
        input, codes, scales, block_contract, philox, stream, statistics);
  }
}

std::tuple<at::Tensor, at::Tensor> quantize_fp8_cuda_impl(
    const at::Tensor& input,
    int64_t contract_dim,
    at::ScalarType dtype,
    int64_t block_outer,
    int64_t block_contract,
    bool stochastic_rounding,
    const std::string& output_layout,
    const at::Tensor& statistics) {
  TORCH_CHECK(input.is_cuda(), "quantize_fp8 requires a CUDA tensor");
  check_arguments(input, contract_dim, dtype, block_outer, block_contract);
  c10::cuda::CUDAGuard device_guard(input.device());
  const at::Tensor codes = at::empty_strided(
      input.sizes(), code_strides(input, output_layout), input.options().dtype(dtype));
  const bool tensorwise = block_outer == 0;
  const bool rowwise = block_contract == 0;
  const std::vector<int64_t> scales_shape =
      scale_shape(input, contract_dim, block_outer, block_contract);
  const at::Tensor scales = tensorwise
      ? at::empty(tensorwise_storage_shape(input, contract_dim),
                  input.options().dtype(at::kFloat)).expand(scales_shape)
      : at::empty(scales_shape, input.options().dtype(at::kFloat));

  const bool contract_is_col = contract_dim == -1;
  const int64_t rows = input.size(-2);
  const int64_t cols = input.size(-1);
  QuantizeParams params{
      nullptr,
      nullptr,
      nullptr,
      statistics.defined() ? statistics.data_ptr<float>() : nullptr,
      input.dim() == 3 ? input.size(0) : 1,
      rows,
      cols,
      input.dim() == 3 ? input.stride(0) : 0,
      input.stride(-2),
      input.stride(-1),
      input.dim() == 3 ? codes.stride(0) : 0,
      codes.stride(-2),
      codes.stride(-1),
      contract_is_col ? rows : cols,
      contract_is_col ? cols : rows,
      tensorwise ? 1 : block_outer,
      tensorwise || block_contract == 0 ? 1 : block_contract,
      0,
      0,
      contract_is_col};
  params.outer_groups = (params.outer_size + params.block_outer - 1) / params.block_outer;
  params.contract_groups = block_contract == 0 || tensorwise
      ? 1
      : (params.contract_size + params.block_contract - 1) / params.block_contract;
  if (block_contract == 0 && !tensorwise) {
    params.block_contract = params.contract_size;
  }
  const int64_t tensorwise_partials = tensorwise
      ? (rows * cols + kThreads * kTensorwiseValues - 1) /
          (kThreads * kTensorwiseValues)
      : 0;
  const int64_t rowwise_splits = rowwise && !contract_is_col && rows >= 1024
      ? (rows + kRowwiseSplit - 1) / kRowwiseSplit
      : 0;
  const at::Tensor partials = tensorwise
      ? at::empty({params.batches, tensorwise_partials}, input.options().dtype(at::kFloat))
      : rowwise_splits > 0
      ? at::empty({params.batches, rowwise_splits, cols}, input.options().dtype(at::kFloat))
      : at::Tensor();

  at::PhiloxCudaState philox;
  if (stochastic_rounding) {
    auto generator = at::cuda::detail::getDefaultCUDAGenerator(input.get_device());
    std::lock_guard<std::mutex> lock(generator.mutex());
    auto* cuda_generator = static_cast<at::CUDAGeneratorImpl*>(
        generator.unsafeGetGeneratorImpl());
    const uint64_t counters =
        ((static_cast<uint64_t>(input.numel()) + 3) / 4) * 4;
    philox = cuda_generator->philox_cuda_state(counters);
  }
  const auto stream = at::cuda::getCurrentCUDAStream(input.get_device());
  const bool e5m2 = dtype == at::kFloat8_e5m2;
  const bool common_block_size =
      block_contract == 16 || block_contract == 32 || block_contract == 64 ||
      block_contract == 128 || block_contract == 256;
  const bool row_major_output = output_layout == "row_major";
  const bool fast_blockwise1d =
      row_major_output &&
      contract_dim == -1 && block_outer == 1 &&
      block_contract > 0 && input.is_contiguous() &&
      input.numel() <= std::numeric_limits<int>::max() &&
      input.size(-1) % block_contract == 0 && common_block_size &&
      reinterpret_cast<uintptr_t>(input.const_data_ptr()) %
              (input.scalar_type() == at::kFloat ? alignof(float4) : alignof(uint2)) ==
          0 &&
      reinterpret_cast<uintptr_t>(codes.data_ptr()) % alignof(uint32_t) == 0;
  const bool fast_transposed_blockwise1d_rne =
      row_major_output &&
      !stochastic_rounding && contract_dim == -1 && block_outer == 1 &&
      block_contract > 0 && !input.is_contiguous() &&
      input.numel() <= std::numeric_limits<int>::max() && common_block_size &&
      input.size(-1) % block_contract == 0 && input.stride(-2) == 1 &&
      input.stride(-1) == input.size(-2) &&
      (input.dim() == 2 || input.stride(0) == input.size(-2) * input.size(-1)) &&
      reinterpret_cast<uintptr_t>(codes.data_ptr()) % alignof(uint32_t) == 0;
  const bool fast_transposed_column_blockwise1d_rne =
      row_major_output &&
      !stochastic_rounding && contract_dim == -2 && block_outer == 1 &&
      block_contract > 0 && !input.is_contiguous() &&
      input.numel() <= std::numeric_limits<int>::max() && common_block_size &&
      input.size(-2) % block_contract == 0 && input.size(-1) % 2 == 0 &&
      input.stride(-2) == 1 && input.stride(-1) == input.size(-2) &&
      (input.dim() == 2 || input.stride(0) == input.size(-2) * input.size(-1)) &&
      reinterpret_cast<uintptr_t>(codes.data_ptr()) % alignof(uint16_t) == 0;
  switch (input.scalar_type()) {
    case at::kFloat:
      if (fast_blockwise1d) {
        launch_blockwise1d_contiguous<float>(
            input, codes, scales, static_cast<int>(block_contract), e5m2,
            stochastic_rounding, philox, stream.stream(), params.statistics);
      } else if (fast_transposed_blockwise1d_rne) {
        launch_blockwise1d_transposed_rne<float>(
            input, codes, scales, static_cast<int>(block_contract), e5m2,
            stream.stream(), params.statistics, input.numel());
      } else if (fast_transposed_column_blockwise1d_rne) {
        launch_blockwise1d_transposed_column_rne<float>(
            input, codes, scales, static_cast<int>(block_contract), e5m2,
            stream.stream(), params.statistics, input.numel());
      } else {
        launch_quantize<float>(
            input, codes, scales, partials, params, e5m2, stochastic_rounding, philox,
            stream.stream(), tensorwise, rowwise);
      }
      break;
    case at::kHalf:
      if (fast_blockwise1d) {
        launch_blockwise1d_contiguous<at::Half>(
            input, codes, scales, static_cast<int>(block_contract), e5m2,
            stochastic_rounding, philox, stream.stream(), params.statistics);
      } else if (fast_transposed_blockwise1d_rne) {
        launch_blockwise1d_transposed_rne<at::Half>(
            input, codes, scales, static_cast<int>(block_contract), e5m2,
            stream.stream(), params.statistics, input.numel());
      } else if (fast_transposed_column_blockwise1d_rne) {
        launch_blockwise1d_transposed_column_rne<at::Half>(
            input, codes, scales, static_cast<int>(block_contract), e5m2,
            stream.stream(), params.statistics, input.numel());
      } else {
        launch_quantize<at::Half>(
            input, codes, scales, partials, params, e5m2, stochastic_rounding, philox,
            stream.stream(), tensorwise, rowwise);
      }
      break;
    case at::kBFloat16:
      if (fast_blockwise1d) {
        launch_blockwise1d_contiguous<at::BFloat16>(
            input, codes, scales, static_cast<int>(block_contract), e5m2,
            stochastic_rounding, philox, stream.stream(), params.statistics);
      } else if (fast_transposed_blockwise1d_rne) {
        launch_blockwise1d_transposed_rne<at::BFloat16>(
            input, codes, scales, static_cast<int>(block_contract), e5m2,
            stream.stream(), params.statistics, input.numel());
      } else if (fast_transposed_column_blockwise1d_rne) {
        launch_blockwise1d_transposed_column_rne<at::BFloat16>(
            input, codes, scales, static_cast<int>(block_contract), e5m2,
            stream.stream(), params.statistics, input.numel());
      } else {
        launch_quantize<at::BFloat16>(
            input, codes, scales, partials, params, e5m2, stochastic_rounding, philox,
            stream.stream(), tensorwise, rowwise);
      }
      break;
    default:
      TORCH_CHECK(false, "quantize_fp8 has an unsupported input dtype");
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {codes, scales};
}

std::tuple<at::Tensor, at::Tensor> quantize_fp8_cuda(
    const at::Tensor& input,
    int64_t contract_dim,
    at::ScalarType dtype,
    int64_t block_outer,
    int64_t block_contract,
    bool stochastic_rounding,
    const std::string& output_layout) {
  return quantize_fp8_cuda_impl(
      input, contract_dim, dtype, block_outer, block_contract, stochastic_rounding,
      output_layout, at::Tensor());
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> quantize_fp8_with_stats_cuda(
    const at::Tensor& input,
    int64_t contract_dim,
    at::ScalarType dtype,
    int64_t block_outer,
    int64_t block_contract,
    bool stochastic_rounding,
  const std::string& output_layout) {
  const at::Tensor statistics = at::zeros({5}, input.options().dtype(at::kFloat));
  const int64_t batches = input.dim() == 3 ? input.size(0) : 1;
  const int64_t rows = input.size(-2);
  const int64_t cols = input.size(-1);
  const bool contract_is_col = contract_dim == -1;
  const int64_t outer = contract_is_col ? rows : cols;
  const int64_t contract = contract_is_col ? cols : rows;
  const int64_t partial_count = block_outer == 0
      ? batches * ((rows * cols + kThreads * kTensorwiseValues - 1) /
                   (kThreads * kTensorwiseValues))
      : block_contract == 0
      ? batches * outer * ((contract + kRowwiseSplit - 1) / kRowwiseSplit)
      : batches * ((outer + block_outer - 1) / block_outer) *
            ((contract + block_contract - 1) / block_contract);
  const at::Tensor partials = at::zeros(
      {partial_count, 4}, input.options().dtype(at::kFloat));
  auto [codes, scales] = quantize_fp8_cuda_impl(
      input, contract_dim, dtype, block_outer, block_contract, stochastic_rounding,
      output_layout, partials);
  const auto stream = at::cuda::getCurrentCUDAStream(input.get_device());
  finalize_statistics_kernel<<<1, kThreads, 0, stream.stream()>>>(
      partials.data_ptr<float>(), partial_count, input.numel(), statistics.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {codes, scales, statistics};
}

std::tuple<at::Tensor, at::Tensor> quantize_fp8_meta(
    const at::Tensor& input,
    int64_t contract_dim,
    at::ScalarType dtype,
    int64_t block_outer,
    int64_t block_contract,
    bool stochastic_rounding,
    const std::string& output_layout) {
  check_arguments(input, contract_dim, dtype, block_outer, block_contract);
  const bool tensorwise = block_outer == 0;
  const std::vector<int64_t> output_shape =
      scale_shape(input, contract_dim, block_outer, block_contract);
  const at::Tensor scales = tensorwise
      ? at::empty(tensorwise_storage_shape(input, contract_dim),
                  input.options().dtype(at::kFloat)).expand(output_shape)
      : at::empty(output_shape, input.options().dtype(at::kFloat));
  return {
      at::empty_strided(input.sizes(), code_strides(input, output_layout),
                        input.options().dtype(dtype)),
      scales,
  };
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> quantize_fp8_with_stats_meta(
    const at::Tensor& input,
    int64_t contract_dim,
    at::ScalarType dtype,
    int64_t block_outer,
    int64_t block_contract,
    bool stochastic_rounding,
    const std::string& output_layout) {
  auto [codes, scales] = quantize_fp8_meta(
      input, contract_dim, dtype, block_outer, block_contract, stochastic_rounding,
      output_layout);
  return {codes, scales, at::empty({5}, input.options().dtype(at::kFloat))};
}

}  // namespace

TORCH_LIBRARY_FRAGMENT(aot_kernel, m) {
  m.def(
      "quantize_fp8(Tensor x, int contract_dim, ScalarType dtype, int block_outer, "
      "int block_contract, bool stochastic_rounding=False, str output_layout=\"row_major\") -> (Tensor, Tensor)",
      {at::Tag::nondeterministic_seeded});
  m.def(
      "quantize_fp8_with_stats(Tensor x, int contract_dim, ScalarType dtype, int block_outer, "
      "int block_contract, bool stochastic_rounding=False, str output_layout=\"row_major\") -> (Tensor, Tensor, Tensor)",
      {at::Tag::nondeterministic_seeded});
}

TORCH_LIBRARY_IMPL(aot_kernel, Meta, m) {
  m.impl("quantize_fp8", TORCH_FN(quantize_fp8_meta));
  m.impl("quantize_fp8_with_stats", TORCH_FN(quantize_fp8_with_stats_meta));
}

TORCH_LIBRARY_IMPL(aot_kernel, CUDA, m) {
  m.impl("quantize_fp8", TORCH_FN(quantize_fp8_cuda));
  m.impl("quantize_fp8_with_stats", TORCH_FN(quantize_fp8_with_stats_cuda));
}
