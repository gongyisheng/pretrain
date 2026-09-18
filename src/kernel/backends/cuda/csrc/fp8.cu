#include "quantize.cuh"

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

dim3 make_launch_grid(int64_t blocks) {
  constexpr int64_t kMaxGridX = std::numeric_limits<int32_t>::max();
  constexpr int64_t kMaxGridY = 65535;
  const int64_t grid_x = std::min(blocks, kMaxGridX);
  const int64_t grid_y = (blocks + grid_x - 1) / grid_x;
  TORCH_CHECK(grid_y <= kMaxGridY, "quantize_fp8 tensor is too large to launch");
  return dim3(grid_x, grid_y);
}

template <bool kE5m2>
__device__ __forceinline__ void accumulate_statistics(
    QuantizationStatistics* partial, float source, uint8_t code, float scale) {
  accumulate_quantization_statistics(
      partial, source, decode_fp8<kE5m2>(code) * scale);
}

// this is for compute codes
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

__device__ __forceinline__ float reduce_strict_maximum(float maximum) {
  constexpr int kWarps = kThreads / 32;
  __shared__ float warp_maximum[kWarps];
  const int lane = threadIdx.x % 32;
  const int warp = threadIdx.x / 32;
  constexpr unsigned int kWarpMask = 0xffffffff;
  for (int offset = 16; offset > 0; offset >>= 1) {
    maximum = strict_max(maximum, __shfl_down_sync(kWarpMask, maximum, offset));
  }
  if (lane == 0) warp_maximum[warp] = maximum;
  __syncthreads();
  maximum = threadIdx.x < kWarps ? warp_maximum[threadIdx.x] : 0.0f;
  if (warp == 0) {
    for (int offset = 16; offset > 0; offset >>= 1) {
      maximum = strict_max(maximum, __shfl_down_sync(kWarpMask, maximum, offset));
    }
    if (lane == 0) warp_maximum[0] = maximum;
  }
  __syncthreads();
  return warp_maximum[0];
}

template <typename input_t, bool kE5m2, bool kStochastic>
__global__ void quantize_blockwise_1d_kernel(
    QuantizeParams params,
    at::PhiloxCudaState philox) {
  const int64_t tile = grid_index();
  const int64_t tile_count =
      params.batch * params.row * params.col_block_count;
  if (tile >= tile_count) return;

  const int64_t contract_group = tile % params.col_block_count;
  const int64_t remaining = tile / params.col_block_count;
  const int64_t row = remaining % params.row;
  const int64_t batch = remaining / params.row;
  const int64_t contract_start = contract_group * params.cols_per_block;
  const int64_t contract_end = min(contract_start + params.cols_per_block, params.col);

  float maximum = 0.0f;
  for (int64_t col = contract_start + threadIdx.x; col < contract_end;
       col += blockDim.x) {
    const float value = static_cast<float>(
        params.input_data<input_t>()[input_offset(params, batch, row, col)]);
    maximum = strict_max(maximum, fabsf(value));
  }
  __shared__ float shared_scale;
  const float maximum_value = reduce_strict_maximum(maximum);
  if (threadIdx.x == 0) {
    shared_scale = compute_scale(
        maximum_value, kE5m2 ? 57344.0f : 448.0f);
    params.scale_data<float>()[scale_offset(params, batch, row, contract_group)] =
        shared_scale;
  }
  __syncthreads();

  curandStatePhilox4_32_10_t random_state;
  if constexpr (kStochastic) {
    const auto seeds = at::cuda::philox::unpack(philox);
    curand_init(
        static_cast<unsigned long long>(std::get<0>(seeds)),
        static_cast<unsigned long long>(tile * blockDim.x + threadIdx.x),
        static_cast<unsigned long long>(std::get<1>(seeds)),
        &random_state);
  }
  QuantizationStatistics statistics;
  constexpr float qmax = kE5m2 ? 57344.0f : 448.0f;
  for (int64_t col = contract_start + threadIdx.x; col < contract_end;
       col += blockDim.x) {
    const float source = static_cast<float>(
        params.input_data<input_t>()[input_offset(params, batch, row, col)]);
    float value = source / shared_scale;
    if (!isnan(value)) value = fminf(fmaxf(value, -qmax), qmax);
    uint8_t code;
    if constexpr (kStochastic) {
      code = encode_fp8_stochastic<kE5m2>(
          value, isnan(value) ? 0u : curand(&random_state));
    } else {
      code = encode_fp8_rne<kE5m2>(value);
    }
    params.code_data<uint8_t>()[output_offset(params, batch, row, col)] = code;
    if (params.statistics != nullptr) {
      accumulate_statistics<kE5m2>(&statistics, source, code, shared_scale);
    }
  }
  if (params.statistics != nullptr) {
    quantization_statistics_epilogue<kThreads>(
        statistics, params.statistics, params.batch * params.row * params.col);
  }
}

template <typename input_t, bool kE5m2, bool kStochastic>
__global__ void quantize_blockwise_2d_kernel(
    QuantizeParams params,
    at::PhiloxCudaState philox) {
  const int64_t tile = grid_index();
  const int64_t tile_count =
      params.batch * params.row_block_count * params.col_block_count;
  if (tile >= tile_count) return;

  const int64_t col_group = tile % params.col_block_count;
  const int64_t remaining = tile / params.col_block_count;
  const int64_t row_group = remaining % params.row_block_count;
  const int64_t batch = remaining / params.row_block_count;
  const int64_t row_start = row_group * params.rows_per_block;
  const int64_t col_start = col_group * params.cols_per_block;
  const int64_t row_end = min(row_start + params.rows_per_block, params.row);
  const int64_t col_end = min(col_start + params.cols_per_block, params.col);
  const int64_t tile_cols = col_end - col_start;
  const int64_t tile_elements = (row_end - row_start) * tile_cols;

  float maximum = 0.0f;
  for (int64_t flat = threadIdx.x; flat < tile_elements; flat += blockDim.x) {
    const int64_t row = row_start + flat / tile_cols;
    const int64_t col = col_start + flat % tile_cols;
    const float value = static_cast<float>(
        params.input_data<input_t>()[input_offset(params, batch, row, col)]);
    maximum = strict_max(maximum, fabsf(value));
  }
  __shared__ float shared_scale;
  const float maximum_value = reduce_strict_maximum(maximum);
  if (threadIdx.x == 0) {
    shared_scale = compute_scale(maximum_value, kE5m2 ? 57344.0f : 448.0f);
    if (params.contract_dim == -1) {
      for (int64_t row = row_start; row < row_end; ++row) {
        params.scale_data<float>()[scale_offset(params, batch, row, col_group)] =
            shared_scale;
      }
    } else {
      for (int64_t col = col_start; col < col_end; ++col) {
        params.scale_data<float>()[scale_offset(params, batch, row_group, col)] =
            shared_scale;
      }
    }
  }
  __syncthreads();

  curandStatePhilox4_32_10_t random_state;
  if constexpr (kStochastic) {
    const auto seeds = at::cuda::philox::unpack(philox);
    curand_init(
        static_cast<unsigned long long>(std::get<0>(seeds)),
        static_cast<unsigned long long>(tile * blockDim.x + threadIdx.x),
        static_cast<unsigned long long>(std::get<1>(seeds)),
        &random_state);
  }
  QuantizationStatistics statistics;
  constexpr float qmax = kE5m2 ? 57344.0f : 448.0f;
  for (int64_t flat = threadIdx.x; flat < tile_elements; flat += blockDim.x) {
    const int64_t row = row_start + flat / tile_cols;
    const int64_t col = col_start + flat % tile_cols;
    const float source = static_cast<float>(
        params.input_data<input_t>()[input_offset(params, batch, row, col)]);
    float value = source / shared_scale;
    if (!isnan(value)) value = fminf(fmaxf(value, -qmax), qmax);
    uint8_t code;
    if constexpr (kStochastic) {
      code = encode_fp8_stochastic<kE5m2>(
          value, isnan(value) ? 0u : curand(&random_state));
    } else {
      code = encode_fp8_rne<kE5m2>(value);
    }
    params.code_data<uint8_t>()[output_offset(params, batch, row, col)] = code;
    if (params.statistics != nullptr) {
      accumulate_statistics<kE5m2>(&statistics, source, code, shared_scale);
    }
  }
  if (params.statistics != nullptr) {
    quantization_statistics_epilogue<kThreads>(
        statistics, params.statistics, params.batch * params.row * params.col);
  }
}

template <typename input_t, bool kE5m2, bool kStochastic, typename block_shape>
__global__ void quantize_blockwise_1d_packed_kernel(
    QuantizeParams params,
    at::PhiloxCudaState philox) {
  const int64_t outer_size =
      params.contract_dim == -1 ? params.row : params.col;
  const int64_t contract_size =
      params.contract_dim == -1 ? params.col : params.row;
  const int64_t contract_groups = params.contract_dim == -1
      ? params.col_block_count
      : params.row_block_count;
  constexpr int kWarps = kThreads / 32;
  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  const int64_t tile = grid_index() * kWarps + warp;
  const int64_t tile_count = params.batch * outer_size * contract_groups;
  const bool valid_tile = tile < tile_count;
  if (!valid_tile && params.statistics == nullptr) {
    return;
  }
  QuantizationStatistics statistics;
  if (valid_tile) {
    const int64_t contract_group = tile % contract_groups;
    const int64_t remaining = tile / contract_groups;
    const int64_t outer = remaining % outer_size;
    const int64_t batch = remaining / outer_size;
    constexpr int kValuesPerLane = (block_shape::dim1 + 31) / 32;
    const int64_t contract_start = contract_group * block_shape::dim1;
    float maximum = 0.0f;
    float values[kValuesPerLane];
    #pragma unroll
    for (int value_index = 0; value_index < kValuesPerLane; ++value_index) {
      const int64_t contract = contract_start + lane + value_index * 32;
      if (contract < contract_size && value_index * 32 + lane < block_shape::dim1) {
        const int64_t row = params.contract_dim == -1 ? outer : contract;
        const int64_t col = params.contract_dim == -1 ? contract : outer;
        values[value_index] = static_cast<float>(params.input_data<input_t>()[
            input_offset(params, batch, row, col)]);
        maximum = strict_max(maximum, fabsf(values[value_index]));
      }
    }
    constexpr unsigned int kWarpMask = 0xffffffff;
    for (int offset = 16; offset > 0; offset >>= 1) {
      maximum = strict_max(maximum, __shfl_down_sync(kWarpMask, maximum, offset));
    }
    maximum = __shfl_sync(kWarpMask, maximum, 0);
    const float scale = compute_scale(maximum, kE5m2 ? 57344.0f : 448.0f);
    if (lane == 0) {
      const int64_t row = params.contract_dim == -1 ? outer : contract_group;
      const int64_t col = params.contract_dim == -1 ? contract_group : outer;
      params.scale_data<float>()[scale_offset(params, batch, row, col)] = scale;
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
        if (contract < contract_size && value_index * 32 + lane < block_shape::dim1) {
          float value = values[value_index] / scale;
          if (!isnan(value)) {
            constexpr float qmax = kE5m2 ? 57344.0f : 448.0f;
            value = fminf(fmaxf(value, -qmax), qmax);
          }
          const int64_t row = params.contract_dim == -1 ? outer : contract;
          const int64_t col = params.contract_dim == -1 ? contract : outer;
          const uint8_t code = encode_fp8_stochastic<kE5m2>(
              value, isnan(value) ? 0u : curand(&random_state));
          params.code_data<uint8_t>()[output_offset(params, batch, row, col)] = code;
          if (params.statistics != nullptr) accumulate_statistics<kE5m2>(&statistics, values[value_index], code, scale);
        }
      }
    } else {
      #pragma unroll
      for (int value_index = 0; value_index < kValuesPerLane; ++value_index) {
        const int64_t contract = contract_start + lane + value_index * 32;
        if (contract < contract_size && value_index * 32 + lane < block_shape::dim1) {
          float value = values[value_index] / scale;
          if (!isnan(value)) {
            constexpr float qmax = kE5m2 ? 57344.0f : 448.0f;
            value = fminf(fmaxf(value, -qmax), qmax);
          }
          const int64_t row = params.contract_dim == -1 ? outer : contract;
          const int64_t col = params.contract_dim == -1 ? contract : outer;
          const uint8_t code = encode_fp8_rne<kE5m2>(value);
          params.code_data<uint8_t>()[output_offset(params, batch, row, col)] = code;
          if (params.statistics != nullptr) accumulate_statistics<kE5m2>(&statistics, values[value_index], code, scale);
        }
      }
    }
  }
  if (params.statistics != nullptr) {
    quantization_statistics_epilogue<kThreads>(
        statistics, params.statistics, params.batch * params.row * params.col);
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

template <typename input_t, bool kE5m2, bool kStochastic, typename block_shape>
__global__ void quantize_blockwise_1d_contiguous_kernel(
    const input_t* input,
    uint8_t* codes,
    float* scales,
    int rows,
    int cols,
    int contract_groups,
    at::PhiloxCudaState philox,
    float* statistics) {
  constexpr int kWarps = kThreads / 32;
  constexpr int kValuesPerLane = block_shape::dim1 == 256 ? 8 : 4;
  constexpr int kVectorsPerLane = kValuesPerLane / 4;
  constexpr int kLanesPerTile = block_shape::dim1 / kValuesPerLane;
  constexpr int kTilesPerWarp = 32 / kLanesPerTile;
  static_assert(kLanesPerTile > 0 && kLanesPerTile <= 32);
  static_assert(32 % kLanesPerTile == 0);

  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  const int tile_in_warp = lane / kLanesPerTile;
  const int lane_in_tile = lane % kLanesPerTile;
  const int warp_index =
      (static_cast<int>(blockIdx.y) * static_cast<int>(gridDim.x) +
       static_cast<int>(blockIdx.x)) *
          kWarps +
      warp;
  const int tile = warp_index * kTilesPerWarp + tile_in_warp;
  const int tile_count = rows * contract_groups;
  const bool valid_tile = tile < tile_count;
  const int offset = valid_tile
      ? tile * block_shape::dim1 + lane_in_tile * kValuesPerLane
      : 0;

  float values[kValuesPerLane];
  QuantizationStatistics partial;
  float maximum = 0.0f;
  if (valid_tile) {
    #pragma unroll
    for (int vector_index = 0; vector_index < kVectorsPerLane; ++vector_index) {
      load_contiguous_four(
          input, offset + vector_index * 4, values + vector_index * 4);
    }
    #pragma unroll
    for (int value_index = 0; value_index < kValuesPerLane; ++value_index) {
      maximum = strict_max(maximum, fabsf(values[value_index]));
    }
  }
  constexpr unsigned int kWarpMask = 0xffffffff;
  #pragma unroll
  for (int reduction_offset = kLanesPerTile / 2; reduction_offset > 0;
       reduction_offset >>= 1) {
    maximum = strict_max(
        maximum,
        __shfl_down_sync(kWarpMask, maximum, reduction_offset, kLanesPerTile));
  }
  const float scale = __shfl_sync(
      kWarpMask,
      compute_scale(maximum, kE5m2 ? 57344.0f : 448.0f),
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
        const uint8_t code = encode_fp8_stochastic<kE5m2>(
            value, isnan(value) ? 0u : curand(&random_state));
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
  if (statistics != nullptr) {
    quantization_statistics_epilogue<kThreads>(
        partial, statistics, static_cast<int64_t>(rows) * cols);
  }
}

template <typename input_t, bool kE5m2, typename block_shape>
__global__ void quantize_blockwise_1d_transposed_rne_kernel(
    const input_t* input,
    uint8_t* codes,
    float* scales,
    int batches,
    int rows,
    int cols,
    int contract_groups,
    float* statistics) {
  constexpr int kOuterTile = 32;
  constexpr int kThreadsPerOuter = 8;
  constexpr int kValuesPerThread = block_shape::dim1 / kThreadsPerOuter;
  static_assert(block_shape::dim1 % kThreadsPerOuter == 0);

  const int tile = static_cast<int>(blockIdx.y) * static_cast<int>(gridDim.x) +
      static_cast<int>(blockIdx.x);
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
  const int contract_start = contract_group * block_shape::dim1;
  const int valid_outers = min(kOuterTile, rows - outer_start);
  QuantizationStatistics partial;

  __shared__ float shared_values[kOuterTile * (block_shape::dim1 + 1)];
  for (int flat = threadIdx.x; flat < kOuterTile * block_shape::dim1;
       flat += blockDim.x) {
    const int contract = flat / kOuterTile;
    const int outer = flat - contract * kOuterTile;
    if (outer < valid_outers) {
      const int input_offset = batch * rows * cols +
          (contract_start + contract) * rows + outer_start + outer;
      shared_values[outer * (block_shape::dim1 + 1) + contract] =
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
          outer * (block_shape::dim1 + 1) + value_offset];
      maximum = strict_max(maximum, fabsf(value));
    }
  }
  constexpr unsigned int kWarpMask = 0xffffffff;
  #pragma unroll
  for (int reduction_offset = kThreadsPerOuter / 2; reduction_offset > 0;
       reduction_offset >>= 1) {
    maximum = strict_max(
        maximum,
        __shfl_down_sync(kWarpMask, maximum, reduction_offset, kThreadsPerOuter));
  }
  const float scale = __shfl_sync(
      kWarpMask,
      compute_scale(maximum, kE5m2 ? 57344.0f : 448.0f),
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
    const int shared_offset = outer * (block_shape::dim1 + 1) + value_offset;
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
      const int shared_offset = outer * (block_shape::dim1 + 1) + value_offset;
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
    quantization_statistics_epilogue<kThreads>(
        partial, statistics, static_cast<int64_t>(batches) * rows * cols);
  }
}

template <typename input_t, bool kE5m2, typename block_shape>
__global__ void quantize_blockwise_1d_tiled_column_rne_kernel(
    const input_t* input,
    uint8_t* codes,
    float* scales,
    int batches,
    int rows,
    int cols,
    int contract_groups,
    float* statistics) {
  constexpr int kOuterTile = 32;
  constexpr int kThreadsPerOuter = 8;
  constexpr int kValuesPerThread = block_shape::dim1 / kThreadsPerOuter;
  static_assert(block_shape::dim1 % kThreadsPerOuter == 0);

  const int tile = static_cast<int>(blockIdx.y) * static_cast<int>(gridDim.x) +
      static_cast<int>(blockIdx.x);
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
  const int contract_start = contract_group * block_shape::dim1;
  const int valid_outers = min(kOuterTile, cols - outer_start);
  QuantizationStatistics partial;

  __shared__ float shared_values[kOuterTile * (block_shape::dim1 + 1)];
  __shared__ float shared_scales[kOuterTile];
  for (int flat = threadIdx.x; flat < kOuterTile * block_shape::dim1;
       flat += blockDim.x) {
    const int outer = flat / block_shape::dim1;
    const int contract = flat - outer * block_shape::dim1;
    if (outer < valid_outers) {
      const int input_offset = batch * rows * cols +
          (outer_start + outer) * rows + contract_start + contract;
      shared_values[outer * (block_shape::dim1 + 1) + contract] =
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
      maximum = strict_max(
          maximum,
          fabsf(shared_values[outer * (block_shape::dim1 + 1) + value_offset]));
    }
  }
  constexpr unsigned int kWarpMask = 0xffffffff;
  #pragma unroll
  for (int reduction_offset = kThreadsPerOuter / 2; reduction_offset > 0;
       reduction_offset >>= 1) {
    maximum = strict_max(
        maximum,
        __shfl_down_sync(kWarpMask, maximum, reduction_offset, kThreadsPerOuter));
  }
  if (valid_outer && segment == 0) {
    const float scale = compute_scale(maximum, kE5m2 ? 57344.0f : 448.0f);
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
      const int shared_offset = outer * (block_shape::dim1 + 1) + value_offset;
      shared_values[shared_offset] = normalize_fp8_rne<kE5m2>(
          shared_values[shared_offset], scale, inverse_scale);
    }
  }
  __syncthreads();

  const int pairs = kOuterTile * block_shape::dim1 / 2;
  for (int pair = threadIdx.x; pair < pairs; pair += blockDim.x) {
    const int flat = pair * 2;
    const int contract = flat / kOuterTile;
    const int output_outer = flat - contract * kOuterTile;
    if (output_outer < valid_outers) {
      const int shared_offset = output_outer * (block_shape::dim1 + 1) + contract;
      const float first = shared_values[shared_offset];
      const int output_offset =
          (batch * rows + contract_start + contract) * cols + outer_start + output_outer;
      if (output_outer + 1 < valid_outers) {
        const float second = shared_values[shared_offset + block_shape::dim1 + 1];
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
        const uint8_t code = encode_fp8_rne<kE5m2>(first);
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
    quantization_statistics_epilogue<kThreads>(
        partial, statistics, static_cast<int64_t>(batches) * rows * cols);
  }
}

template <typename input_t, bool kE5m2, bool kStochastic>
__global__ void quantize_rows_coalesced_kernel(
    QuantizeParams params,
    at::PhiloxCudaState philox) {
  const int64_t outer_size =
      params.contract_dim == -1 ? params.row : params.col;
  const int64_t contract_size =
      params.contract_dim == -1 ? params.col : params.row;
  const int64_t contract_block_size =
      params.contract_dim == -1 ? params.cols_per_block : params.rows_per_block;
  const int64_t contract_groups = params.contract_dim == -1
      ? params.col_block_count
      : params.row_block_count;
  constexpr int kWarps = kThreads / 32;
  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  const int64_t outer_tiles = (outer_size + 31) / 32;
  const int64_t tile = grid_index() * kWarps + warp;
  const int64_t tile_count = params.batch * contract_groups * outer_tiles;
  const bool valid_tile = tile < tile_count;
  if (!valid_tile && params.statistics == nullptr) {
    return;
  }
  const int64_t outer_tile = tile % outer_tiles;
  const int64_t remaining = tile / outer_tiles;
  const int64_t contract_group = remaining % contract_groups;
  const int64_t batch = remaining / contract_groups;
  const int64_t outer = outer_tile * 32 + lane;
  const bool valid_outer = valid_tile && outer < outer_size;
  if (!valid_outer && params.statistics == nullptr) {
    return;
  }
  QuantizationStatistics statistics;
  if (valid_outer) {
    const int64_t contract_start = contract_group * contract_block_size;
    const int64_t contract_end =
        min(contract_start + contract_block_size, contract_size);
    float maximum = 0.0f;
    for (int64_t contract = contract_start; contract < contract_end; ++contract) {
      const float value = static_cast<float>(params.input_data<input_t>()[
          input_offset(params, batch, contract, outer)]);
      maximum = strict_max(maximum, fabsf(value));
    }
    const float scale = compute_scale(maximum, kE5m2 ? 57344.0f : 448.0f);
    params.scale_data<float>()[scale_offset(
        params, batch, contract_group, outer)] = scale;
    curandStatePhilox4_32_10_t random_state;
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
      const float source = static_cast<float>(params.input_data<input_t>()[
          input_offset(params, batch, contract, outer)]);
      float value = source;
      value /= scale;
      if (!isnan(value)) {
        value = fminf(fmaxf(value, -qmax), qmax);
      }
      const int64_t code_offset = output_offset(params, batch, contract, outer);
      uint8_t code;
      if constexpr (kStochastic) code = encode_fp8_stochastic<kE5m2>(
              value, isnan(value) ? 0u : curand(&random_state));
      else code = encode_fp8_rne<kE5m2>(value);
      params.code_data<uint8_t>()[code_offset] = code;
      if (params.statistics != nullptr) accumulate_statistics<kE5m2>(&statistics, source, code, scale);
    }
  }
  if (params.statistics != nullptr) {
    quantization_statistics_epilogue<kThreads>(
        statistics, params.statistics, params.batch * params.row * params.col);
  }
}

template <typename input_t>
__global__ void rowwise_partial_amax_kernel(
    QuantizeParams params,
    float* partials,
    int64_t split_count) {
  const int64_t outer_size =
      params.contract_dim == -1 ? params.row : params.col;
  const int64_t contract_size =
      params.contract_dim == -1 ? params.col : params.row;
  constexpr int kWarps = kThreads / 32;
  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  const int64_t outer_tiles = (outer_size + 31) / 32;
  const int64_t tile = grid_index() * kWarps + warp;
  const int64_t tile_count = params.batch * split_count * outer_tiles;
  if (tile >= tile_count) {
    return;
  }
  const int64_t outer_tile = tile % outer_tiles;
  const int64_t remaining = tile / outer_tiles;
  const int64_t split = remaining % split_count;
  const int64_t batch = remaining / split_count;
  const int64_t outer = outer_tile * 32 + lane;
  if (outer >= outer_size) {
    return;
  }
  const int64_t contract_start = split * kRowwiseSplit;
  const int64_t contract_end = min(contract_start + kRowwiseSplit, contract_size);
  float maximum = 0.0f;
  for (int64_t contract = contract_start; contract < contract_end; ++contract) {
    const float value = static_cast<float>(params.input_data<input_t>()[
        input_offset(params, batch, contract, outer)]);
    maximum = strict_max(maximum, fabsf(value));
  }
  partials[(batch * split_count + split) * outer_size + outer] = maximum;
}

template <bool kE5m2>
__global__ void rowwise_scale_kernel(
    QuantizeParams params,
    const float* partials,
    int64_t split_count) {
  const int64_t outer_size =
      params.contract_dim == -1 ? params.row : params.col;
  const int64_t outer = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t batch = blockIdx.y;
  if (outer >= outer_size) {
    return;
  }
  float maximum = 0.0f;
  for (int64_t split = 0; split < split_count; ++split) {
    maximum = strict_max(
        maximum, partials[(batch * split_count + split) * outer_size + outer]);
  }
  params.scale_data<float>()[scale_offset(params, batch, 0, outer)] = compute_scale(
      maximum, kE5m2 ? 57344.0f : 448.0f);
}

template <typename input_t, bool kE5m2, bool kStochastic>
__global__ void rowwise_encode_kernel(
    QuantizeParams params,
    at::PhiloxCudaState philox,
    int64_t split_count) {
  const int64_t outer_size =
      params.contract_dim == -1 ? params.row : params.col;
  const int64_t contract_size =
      params.contract_dim == -1 ? params.col : params.row;
  constexpr int kWarps = kThreads / 32;
  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  const int64_t outer_tiles = (outer_size + 31) / 32;
  const int64_t tile = grid_index() * kWarps + warp;
  const int64_t tile_count = params.batch * split_count * outer_tiles;
  const bool valid_tile = tile < tile_count;
  if (!valid_tile && params.statistics == nullptr) {
    return;
  }
  const int64_t outer_tile = tile % outer_tiles;
  const int64_t remaining = tile / outer_tiles;
  const int64_t split = remaining % split_count;
  const int64_t batch = remaining / split_count;
  const int64_t outer = outer_tile * 32 + lane;
  const bool valid_outer = valid_tile && outer < outer_size;
  if (!valid_outer && params.statistics == nullptr) {
    return;
  }
  QuantizationStatistics statistics;
  if (valid_outer) {
    const float scale = params.scale_data<float>()[scale_offset(
        params, batch, 0, outer)];
    const int64_t contract_start = split * kRowwiseSplit;
    const int64_t contract_end = min(contract_start + kRowwiseSplit, contract_size);
    curandStatePhilox4_32_10_t random_state;
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
      const float source = static_cast<float>(params.input_data<input_t>()[
          input_offset(params, batch, contract, outer)]);
      float value = source;
      value /= scale;
      if (!isnan(value)) {
        value = fminf(fmaxf(value, -qmax), qmax);
      }
      const int64_t code_offset = output_offset(params, batch, contract, outer);
      uint8_t code;
      if constexpr (kStochastic) code = encode_fp8_stochastic<kE5m2>(
              value, isnan(value) ? 0u : curand(&random_state));
      else code = encode_fp8_rne<kE5m2>(value);
      params.code_data<uint8_t>()[code_offset] = code;
      if (params.statistics != nullptr) accumulate_statistics<kE5m2>(&statistics, source, code, scale);
    }
  }
  if (params.statistics != nullptr) {
    quantization_statistics_epilogue<kThreads>(
        statistics, params.statistics, params.batch * params.row * params.col);
  }
}

template <typename input_t>
__global__ void tensorwise_partial_amax_kernel(
    QuantizeParams params, float* partials, int64_t partials_per_batch) {
  const int64_t partial = static_cast<int64_t>(blockIdx.x);
  const int64_t batch = static_cast<int64_t>(blockIdx.y);
  int64_t index =
      (partial * kThreads + threadIdx.x) * kTensorwiseValues;
  const int64_t elements = params.row * params.col;
  float maximum = 0.0f;
  int64_t row = index / params.col;
  int64_t col = index - row * params.col;
  #pragma unroll
  for (int value_index = 0; value_index < kTensorwiseValues; ++value_index) {
    if (index < elements) {
      const float value = static_cast<float>(params.input_data<input_t>()[
          input_offset(params, batch, row, col)]);
      maximum = strict_max(maximum, fabsf(value));
      ++col;
      if (col == params.col) {
        col = 0;
        ++row;
      }
      ++index;
    }
  }
  constexpr unsigned int kWarpMask = 0xffffffff;
  for (int offset = 16; offset > 0; offset >>= 1) {
    maximum = strict_max(maximum, __shfl_down_sync(kWarpMask, maximum, offset));
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
      maximum = strict_max(
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
      maximum = strict_max(maximum, fabsf(values[value_index]));
    }
  }
  constexpr unsigned int kWarpMask = 0xffffffff;
  for (int offset = 16; offset > 0; offset >>= 1) {
    maximum = strict_max(maximum, __shfl_down_sync(kWarpMask, maximum, offset));
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
      maximum = strict_max(
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
    maximum = strict_max(maximum, partials[batch * partials_per_batch + index]);
  }
  constexpr unsigned int kWarpMask = 0xffffffff;
  for (int offset = 16; offset > 0; offset >>= 1) {
    maximum = strict_max(maximum, __shfl_down_sync(kWarpMask, maximum, offset));
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
      maximum = strict_max(
          maximum, __shfl_down_sync(kWarpMask, maximum, offset));
    }
    if (lane == 0) {
      matrix_scales[batch] = compute_scale(
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
  const int64_t elements = params.row * params.col;
  int64_t index =
      (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) * kTensorwiseValues;
  int64_t row = index / params.col;
  int64_t col = index - row * params.col;
  constexpr float qmax = kE5m2 ? 57344.0f : 448.0f;
  const float scale = matrix_scales[batch];
  curandStatePhilox4_32_10_t random_state;
  QuantizationStatistics statistics;
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
      const float source = static_cast<float>(params.input_data<input_t>()[
          input_offset(params, batch, row, col)]);
      float value = source;
      value /= scale;
      if (!isnan(value)) {
        value = fminf(fmaxf(value, -qmax), qmax);
      }
      const int64_t code_offset = output_offset(params, batch, row, col);
      uint8_t code;
      if constexpr (kStochastic) code = encode_fp8_stochastic<kE5m2>(
            value, isnan(value) ? 0u : curand(&random_state));
      else code = encode_fp8_rne<kE5m2>(value);
      params.code_data<uint8_t>()[code_offset] = code;
      if (params.statistics != nullptr) accumulate_statistics<kE5m2>(&statistics, source, code, scale);
      ++col;
      if (col == params.col) {
        col = 0;
        ++row;
      }
      ++index;
    }
  }
  if (params.statistics != nullptr) {
    quantization_statistics_epilogue<kThreads>(
        statistics, params.statistics, params.batch * params.row * params.col);
  }
}

template <typename input_t, bool kE5m2, bool kStochastic>
__global__ void tensorwise_encode_dense_kernel(
    const input_t* input,
    uint8_t* codes,
    const float* matrix_scales,
    int elements,
    at::PhiloxCudaState philox,
    float* statistics) {
  const int64_t batch = blockIdx.y;
  const int index = (blockIdx.x * blockDim.x + threadIdx.x) * kTensorwiseValues;
  QuantizationStatistics partial;
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
      const uint8_t code = encode_fp8_stochastic<kE5m2>(
            value, isnan(value) ? 0u : curand(&random_state));
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
    quantization_statistics_epilogue<kThreads>(
        partial, statistics, static_cast<int64_t>(gridDim.y) * elements);
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
  const int64_t elements_per_batch = params.row * params.col;
  const int64_t partials_per_batch =
      (elements_per_batch + kThreads * kTensorwiseValues - 1) /
      (kThreads * kTensorwiseValues);
  const bool dense =
      elements_per_batch <= std::numeric_limits<int>::max() &&
      elements_per_batch % kTensorwiseValues == 0 &&
      params.input_col_stride == 1 && params.input_row_stride == params.col &&
      (params.batch == 1 || params.input_batch_stride == elements_per_batch) &&
      params.code_col_stride == 1 && params.code_row_stride == params.col &&
      (params.batch == 1 || params.code_batch_stride == elements_per_batch) &&
      reinterpret_cast<uintptr_t>(params.input) %
              (sizeof(input_t) == sizeof(float) ? alignof(float4) : alignof(uint2)) ==
          0 &&
      reinterpret_cast<uintptr_t>(params.code_data<uint8_t>()) % alignof(uint32_t) == 0;
  if (dense) {
    tensorwise_partial_amax_dense_kernel<input_t>
        <<<dim3(partials_per_batch, params.batch), kThreads, 0, stream>>>(
            params.input_data<input_t>(),
            partials.data_ptr<float>(), static_cast<int>(elements_per_batch),
            partials_per_batch);
    tensorwise_scale_kernel<kE5m2><<<params.batch, kThreads, 0, stream>>>(
        partials.data_ptr<float>(), scales.data_ptr<float>(), partials_per_batch);
    const int64_t code_blocks =
        (elements_per_batch + kThreads * kTensorwiseValues - 1) /
        (kThreads * kTensorwiseValues);
    tensorwise_encode_dense_kernel<input_t, kE5m2, kStochastic>
        <<<dim3(code_blocks, params.batch), kThreads, 0, stream>>>(
            params.input_data<input_t>(), params.code_data<uint8_t>(),
            scales.data_ptr<float>(), static_cast<int>(elements_per_batch), philox,
            params.statistics);
    return;
  }
  tensorwise_partial_amax_kernel<input_t><<<dim3(partials_per_batch, params.batch),
                                             kThreads, 0, stream>>>(
      params, partials.data_ptr<float>(), partials_per_batch);
  tensorwise_scale_kernel<kE5m2><<<params.batch, kThreads, 0, stream>>>(
      partials.data_ptr<float>(), scales.data_ptr<float>(), partials_per_batch);
  const int64_t code_blocks =
      (elements_per_batch + kThreads * kTensorwiseValues - 1) /
      (kThreads * kTensorwiseValues);
  tensorwise_encode_kernel<input_t, kE5m2, kStochastic>
      <<<dim3(code_blocks, params.batch), kThreads, 0, stream>>>(
          params, scales.data_ptr<float>(), philox);
}

template <typename input_t, bool kE5m2, bool kStochastic>
void launch_blockwise1d(
    QuantizeParams params,
    at::PhiloxCudaState philox,
    cudaStream_t stream) {
  const int64_t outer_size =
      params.contract_dim == -1 ? params.row : params.col;
  const int64_t contract_block_size =
      params.contract_dim == -1 ? params.cols_per_block : params.rows_per_block;
  const int64_t outer_groups = params.contract_dim == -1
      ? params.row_block_count
      : params.col_block_count;
  const int64_t contract_groups = params.contract_dim == -1
      ? params.col_block_count
      : params.row_block_count;
  const int64_t tiles = params.batch * outer_groups * contract_groups;
  if (params.contract_dim == -2) {
    constexpr int kWarps = kThreads / 32;
    const int64_t outer_tiles = (outer_size + 31) / 32;
    const int64_t groups = params.batch * contract_groups * outer_tiles;
    const dim3 grid = make_launch_grid((groups + kWarps - 1) / kWarps);
    quantize_rows_coalesced_kernel<input_t, kE5m2, kStochastic>
        <<<grid, kThreads, 0, stream>>>(params, philox);
    return;
  }
  constexpr int kWarps = kThreads / 32;
  if (contract_block_size == 16) {
    const dim3 grid = make_launch_grid((tiles + kWarps - 1) / kWarps);
    quantize_blockwise_1d_packed_kernel<input_t, kE5m2, kStochastic, BlockShape<1, 16>>
        <<<grid, kThreads, 0, stream>>>(params, philox);
    return;
  }
  if (contract_block_size == 32) {
    const dim3 grid = make_launch_grid((tiles + kWarps - 1) / kWarps);
    quantize_blockwise_1d_packed_kernel<input_t, kE5m2, kStochastic, BlockShape<1, 32>>
        <<<grid, kThreads, 0, stream>>>(params, philox);
    return;
  }
  if (contract_block_size == 64) {
    const dim3 grid = make_launch_grid((tiles + kWarps - 1) / kWarps);
    quantize_blockwise_1d_packed_kernel<input_t, kE5m2, kStochastic, BlockShape<1, 64>>
        <<<grid, kThreads, 0, stream>>>(params, philox);
    return;
  }
  if (contract_block_size == 128) {
    const dim3 grid = make_launch_grid((tiles + kWarps - 1) / kWarps);
    quantize_blockwise_1d_packed_kernel<input_t, kE5m2, kStochastic, BlockShape<1, 128>>
        <<<grid, kThreads, 0, stream>>>(params, philox);
    return;
  }
  if (contract_block_size == 256) {
    const dim3 grid = make_launch_grid((tiles + kWarps - 1) / kWarps);
    quantize_blockwise_1d_packed_kernel<input_t, kE5m2, kStochastic, BlockShape<1, 256>>
        <<<grid, kThreads, 0, stream>>>(params, philox);
    return;
  }
  const dim3 grid = make_launch_grid(tiles);
  quantize_blockwise_1d_kernel<input_t, kE5m2, kStochastic>
      <<<grid, kThreads, 0, stream>>>(params, philox);
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
    const dim3 grid = make_launch_grid(
        (tiles + kWarps * kTilesPerWarp - 1) / (kWarps * kTilesPerWarp));
    quantize_blockwise_1d_contiguous_kernel<input_t, kE5m2, kStochastic, BlockShape<1, 16>>
        <<<grid, kThreads, 0, stream>>>(
            input.const_data_ptr<input_t>(),
            reinterpret_cast<uint8_t*>(codes.data_ptr()),
            scales.data_ptr<float>(), rows, cols, contract_groups, philox,
            statistics);
    return;
  }
  if (block_contract == 32) {
    constexpr int kTilesPerWarp = 4;
    const dim3 grid = make_launch_grid(
        (tiles + kWarps * kTilesPerWarp - 1) / (kWarps * kTilesPerWarp));
    quantize_blockwise_1d_contiguous_kernel<input_t, kE5m2, kStochastic, BlockShape<1, 32>>
        <<<grid, kThreads, 0, stream>>>(
            input.const_data_ptr<input_t>(),
            reinterpret_cast<uint8_t*>(codes.data_ptr()),
            scales.data_ptr<float>(), rows, cols, contract_groups, philox,
            statistics);
    return;
  }
  if (block_contract == 64) {
    constexpr int kTilesPerWarp = 2;
    const dim3 grid = make_launch_grid(
        (tiles + kWarps * kTilesPerWarp - 1) / (kWarps * kTilesPerWarp));
    quantize_blockwise_1d_contiguous_kernel<input_t, kE5m2, kStochastic, BlockShape<1, 64>>
        <<<grid, kThreads, 0, stream>>>(
            input.const_data_ptr<input_t>(),
            reinterpret_cast<uint8_t*>(codes.data_ptr()),
            scales.data_ptr<float>(), rows, cols, contract_groups, philox,
            statistics);
    return;
  }
  if (block_contract == 128) {
    const dim3 grid = make_launch_grid((tiles + kWarps - 1) / kWarps);
    quantize_blockwise_1d_contiguous_kernel<input_t, kE5m2, kStochastic, BlockShape<1, 128>>
        <<<grid, kThreads, 0, stream>>>(
            input.const_data_ptr<input_t>(),
            reinterpret_cast<uint8_t*>(codes.data_ptr()),
            scales.data_ptr<float>(), rows, cols, contract_groups, philox,
            statistics);
    return;
  }
  if (block_contract == 256) {
    const dim3 grid = make_launch_grid((tiles + kWarps - 1) / kWarps);
    quantize_blockwise_1d_contiguous_kernel<input_t, kE5m2, kStochastic, BlockShape<1, 256>>
        <<<grid, kThreads, 0, stream>>>(
            input.const_data_ptr<input_t>(),
            reinterpret_cast<uint8_t*>(codes.data_ptr()),
            scales.data_ptr<float>(), rows, cols, contract_groups, philox,
            statistics);
  }
}

template <typename input_t, bool kE5m2>
void launch_blockwise1d_transposed_rne(
    const at::Tensor& input,
    const at::Tensor& codes,
    const at::Tensor& scales,
    int block_contract,
    cudaStream_t stream,
    float* statistics) {
  const int batches = input.dim() == 3 ? static_cast<int>(input.size(0)) : 1;
  const int rows = static_cast<int>(input.size(-2));
  const int cols = static_cast<int>(input.size(-1));
  const int contract_groups = cols / block_contract;
  const int outer_tiles = (rows + 31) / 32;
  const int tiles = batches * contract_groups * outer_tiles;
  const dim3 grid = make_launch_grid(tiles);
  if (block_contract == 16) {
    quantize_blockwise_1d_transposed_rne_kernel<input_t, kE5m2, BlockShape<1, 16>>
        <<<grid, kThreads, 0, stream>>>(
            input.const_data_ptr<input_t>(), reinterpret_cast<uint8_t*>(codes.data_ptr()),
            scales.data_ptr<float>(), batches, rows, cols, contract_groups,
            statistics);
  } else if (block_contract == 32) {
    quantize_blockwise_1d_transposed_rne_kernel<input_t, kE5m2, BlockShape<1, 32>>
        <<<grid, kThreads, 0, stream>>>(
            input.const_data_ptr<input_t>(), reinterpret_cast<uint8_t*>(codes.data_ptr()),
            scales.data_ptr<float>(), batches, rows, cols, contract_groups,
            statistics);
  } else if (block_contract == 64) {
    quantize_blockwise_1d_transposed_rne_kernel<input_t, kE5m2, BlockShape<1, 64>>
        <<<grid, kThreads, 0, stream>>>(
            input.const_data_ptr<input_t>(), reinterpret_cast<uint8_t*>(codes.data_ptr()),
            scales.data_ptr<float>(), batches, rows, cols, contract_groups,
            statistics);
  } else if (block_contract == 128) {
    quantize_blockwise_1d_transposed_rne_kernel<input_t, kE5m2, BlockShape<1, 128>>
        <<<grid, kThreads, 0, stream>>>(
            input.const_data_ptr<input_t>(), reinterpret_cast<uint8_t*>(codes.data_ptr()),
            scales.data_ptr<float>(), batches, rows, cols, contract_groups,
            statistics);
  } else {
    quantize_blockwise_1d_transposed_rne_kernel<input_t, kE5m2, BlockShape<1, 256>>
        <<<grid, kThreads, 0, stream>>>(
            input.const_data_ptr<input_t>(), reinterpret_cast<uint8_t*>(codes.data_ptr()),
            scales.data_ptr<float>(), batches, rows, cols, contract_groups,
            statistics);
  }
}

template <typename input_t, bool kE5m2>
void launch_blockwise1d_transposed_column_rne(
    const at::Tensor& input,
    const at::Tensor& codes,
    const at::Tensor& scales,
    int block_contract,
    cudaStream_t stream,
    float* statistics) {
  const int batches = input.dim() == 3 ? static_cast<int>(input.size(0)) : 1;
  const int rows = static_cast<int>(input.size(-2));
  const int cols = static_cast<int>(input.size(-1));
  const int contract_groups = rows / block_contract;
  const int outer_tiles = (cols + 31) / 32;
  const int tiles = batches * contract_groups * outer_tiles;
  const dim3 grid = make_launch_grid(tiles);
  if (block_contract == 16) {
    quantize_blockwise_1d_tiled_column_rne_kernel<input_t, kE5m2, BlockShape<1, 16>>
        <<<grid, kThreads, 0, stream>>>(
            input.const_data_ptr<input_t>(), reinterpret_cast<uint8_t*>(codes.data_ptr()),
            scales.data_ptr<float>(), batches, rows, cols, contract_groups,
            statistics);
  } else if (block_contract == 32) {
    quantize_blockwise_1d_tiled_column_rne_kernel<input_t, kE5m2, BlockShape<1, 32>>
        <<<grid, kThreads, 0, stream>>>(
            input.const_data_ptr<input_t>(), reinterpret_cast<uint8_t*>(codes.data_ptr()),
            scales.data_ptr<float>(), batches, rows, cols, contract_groups,
            statistics);
  } else if (block_contract == 64) {
    quantize_blockwise_1d_tiled_column_rne_kernel<input_t, kE5m2, BlockShape<1, 64>>
        <<<grid, kThreads, 0, stream>>>(
            input.const_data_ptr<input_t>(), reinterpret_cast<uint8_t*>(codes.data_ptr()),
            scales.data_ptr<float>(), batches, rows, cols, contract_groups,
            statistics);
  } else if (block_contract == 128) {
    quantize_blockwise_1d_tiled_column_rne_kernel<input_t, kE5m2, BlockShape<1, 128>>
        <<<grid, kThreads, 0, stream>>>(
            input.const_data_ptr<input_t>(), reinterpret_cast<uint8_t*>(codes.data_ptr()),
            scales.data_ptr<float>(), batches, rows, cols, contract_groups,
            statistics);
  } else {
    quantize_blockwise_1d_tiled_column_rne_kernel<input_t, kE5m2, BlockShape<1, 256>>
        <<<grid, kThreads, 0, stream>>>(
            input.const_data_ptr<input_t>(), reinterpret_cast<uint8_t*>(codes.data_ptr()),
            scales.data_ptr<float>(), batches, rows, cols, contract_groups,
            statistics);
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
    float* statistics) {
  if (e5m2) {
    launch_blockwise1d_transposed_rne<input_t, true>(
        input, codes, scales, block_contract, stream, statistics);
  } else {
    launch_blockwise1d_transposed_rne<input_t, false>(
        input, codes, scales, block_contract, stream, statistics);
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
    float* statistics) {
  if (e5m2) {
    launch_blockwise1d_transposed_column_rne<input_t, true>(
        input, codes, scales, block_contract, stream, statistics);
  } else {
    launch_blockwise1d_transposed_column_rne<input_t, false>(
        input, codes, scales, block_contract, stream, statistics);
  }
}

template <typename input_t, bool kE5m2, bool kStochastic>
void launch_rowwise(
    QuantizeParams params,
    const at::Tensor& partials,
    at::PhiloxCudaState philox,
    cudaStream_t stream) {
  const int64_t outer_size =
      params.contract_dim == -1 ? params.row : params.col;
  const int64_t contract_size =
      params.contract_dim == -1 ? params.col : params.row;
  if (params.contract_dim == -1) {
    const int64_t tiles = params.batch * outer_size;
    const dim3 grid = make_launch_grid(tiles);
    quantize_blockwise_1d_kernel<input_t, kE5m2, kStochastic>
        <<<grid, kThreads, 0, stream>>>(params, philox);
    return;
  }
  if (contract_size < 1024) {
    constexpr int kWarps = kThreads / 32;
    const int64_t outer_tiles = (outer_size + 31) / 32;
    const int64_t groups = params.batch * outer_tiles;
    const dim3 grid = make_launch_grid((groups + kWarps - 1) / kWarps);
    quantize_rows_coalesced_kernel<input_t, kE5m2, kStochastic>
        <<<grid, kThreads, 0, stream>>>(params, philox);
    return;
  }
  constexpr int kWarps = kThreads / 32;
  const int64_t split_count = (contract_size + kRowwiseSplit - 1) / kRowwiseSplit;
  const int64_t outer_tiles = (outer_size + 31) / 32;
  const int64_t groups = params.batch * split_count * outer_tiles;
  const dim3 partial_grid = make_launch_grid((groups + kWarps - 1) / kWarps);
  rowwise_partial_amax_kernel<input_t><<<partial_grid, kThreads, 0, stream>>>(
      params, partials.data_ptr<float>(), split_count);
  const int64_t scale_blocks = (outer_size + kThreads - 1) / kThreads;
  rowwise_scale_kernel<kE5m2><<<dim3(scale_blocks, params.batch), kThreads, 0, stream>>>(
      params, partials.data_ptr<float>(), split_count);
  rowwise_encode_kernel<input_t, kE5m2, kStochastic>
      <<<partial_grid, kThreads, 0, stream>>>(params, philox, split_count);
}

template <typename input_t, bool kE5m2, bool kStochastic>
void launch_blockwise2d(
    QuantizeParams params,
    at::PhiloxCudaState philox,
    cudaStream_t stream) {
  const int64_t tiles =
      params.batch * params.row_block_count * params.col_block_count;
  const dim3 grid = make_launch_grid(tiles);
  quantize_blockwise_2d_kernel<input_t, kE5m2, kStochastic>
      <<<grid, kThreads, 0, stream>>>(params, philox);
}

template <typename input_t, bool kE5m2, bool kStochastic>
void launch_quantize_impl(
    const at::Tensor& scales,
    const at::Tensor& partials,
    QuantizeParams params,
    at::PhiloxCudaState philox,
    cudaStream_t stream,
    bool tensorwise,
    bool rowwise) {
  const int64_t outer_block_size =
      params.contract_dim == -1 ? params.rows_per_block : params.cols_per_block;
  if (tensorwise) {
    launch_tensorwise<input_t, kE5m2, kStochastic>(
        scales, partials, params, philox, stream);
  } else if (rowwise) {
    launch_rowwise<input_t, kE5m2, kStochastic>(params, partials, philox, stream);
  } else if (outer_block_size == 1) {
    launch_blockwise1d<input_t, kE5m2, kStochastic>(params, philox, stream);
  } else {
    launch_blockwise2d<input_t, kE5m2, kStochastic>(params, philox, stream);
  }
}

template <typename input_t>
void launch_quantize(
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
          scales, partials, params, philox, stream, tensorwise, rowwise);
    } else {
      launch_quantize_impl<input_t, true, false>(
          scales, partials, params, philox, stream, tensorwise, rowwise);
    }
  } else if (stochastic_rounding) {
    launch_quantize_impl<input_t, false, true>(
        scales, partials, params, philox, stream, tensorwise, rowwise);
  } else {
    launch_quantize_impl<input_t, false, false>(
        scales, partials, params, philox, stream, tensorwise, rowwise);
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

  const int64_t rows = input.size(-2);
  const int64_t cols = input.size(-1);
  int64_t rows_per_block;
  int64_t cols_per_block;
  if (tensorwise) {
    rows_per_block = rows;
    cols_per_block = cols;
  } else if (contract_dim == -1) {
    rows_per_block = block_outer;
    cols_per_block = block_contract == 0 ? cols : block_contract;
  } else {
    rows_per_block = block_contract == 0 ? rows : block_contract;
    cols_per_block = block_outer;
  }
  const int64_t row_block_count =
      (rows + rows_per_block - 1) / rows_per_block;
  const int64_t col_block_count =
      (cols + cols_per_block - 1) / cols_per_block;
  QuantizeParams params;
  params.input = input.const_data_ptr();
  params.codes = codes.data_ptr();
  params.scale = scales.data_ptr();
  params.statistics =
      statistics.defined() ? statistics.data_ptr<float>() : nullptr;
  params.batch = input.dim() == 3 ? input.size(0) : 1;
  params.row = rows;
  params.col = cols;
  params.input_batch_stride = input.dim() == 3 ? input.stride(0) : 0;
  params.input_row_stride = input.stride(-2);
  params.input_col_stride = input.stride(-1);
  params.code_batch_stride = codes.dim() == 3 ? codes.stride(0) : 0;
  params.code_row_stride = codes.stride(-2);
  params.code_col_stride = codes.stride(-1);
  params.scale_batch_stride = scales.dim() == 3 ? scales.stride(0) : 0;
  params.scale_row_stride = scales.stride(-2);
  params.scale_col_stride = scales.stride(-1);
  params.rows_per_block = rows_per_block;
  params.cols_per_block = cols_per_block;
  params.row_block_count =
      row_block_count;
  params.col_block_count =
      col_block_count;
  params.contract_dim = static_cast<int>(contract_dim);
  const int64_t tensorwise_partials = tensorwise
      ? (rows * cols + kThreads * kTensorwiseValues - 1) /
          (kThreads * kTensorwiseValues)
      : 0;
  const int64_t rowwise_splits = rowwise && contract_dim == -2 && rows >= 1024
      ? (rows + kRowwiseSplit - 1) / kRowwiseSplit
      : 0;
  const at::Tensor partials = tensorwise
      ? at::empty({params.batch, tensorwise_partials}, input.options().dtype(at::kFloat))
      : rowwise_splits > 0
      ? at::empty({params.batch, rowwise_splits, cols}, input.options().dtype(at::kFloat))
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
            stream.stream(), params.statistics);
      } else if (fast_transposed_column_blockwise1d_rne) {
        launch_blockwise1d_transposed_column_rne<float>(
            input, codes, scales, static_cast<int>(block_contract), e5m2,
            stream.stream(), params.statistics);
      } else {
        launch_quantize<float>(
            scales, partials, params, e5m2, stochastic_rounding, philox,
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
            stream.stream(), params.statistics);
      } else if (fast_transposed_column_blockwise1d_rne) {
        launch_blockwise1d_transposed_column_rne<at::Half>(
            input, codes, scales, static_cast<int>(block_contract), e5m2,
            stream.stream(), params.statistics);
      } else {
        launch_quantize<at::Half>(
            scales, partials, params, e5m2, stochastic_rounding, philox,
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
            stream.stream(), params.statistics);
      } else if (fast_transposed_column_blockwise1d_rne) {
        launch_blockwise1d_transposed_column_rne<at::BFloat16>(
            input, codes, scales, static_cast<int>(block_contract), e5m2,
            stream.stream(), params.statistics);
      } else {
        launch_quantize<at::BFloat16>(
            scales, partials, params, e5m2, stochastic_rounding, philox,
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
  auto [codes, scales] = quantize_fp8_cuda_impl(
      input, contract_dim, dtype, block_outer, block_contract, stochastic_rounding,
      output_layout, statistics);
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
