#include <ATen/core/enum_tag.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <curand_kernel.h>
#include <torch/extension.h>

#include <ATen/cuda/PhiloxUtils.cuh>
#include "quantize.cuh"
#include <algorithm>
#include <cstdint>
#include <limits>
#include <mutex>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>

namespace {

__device__ __forceinline__ int64_t local_offset(
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

template <bool kInputColumnMajor>
__device__ __forceinline__ int32_t logical_row(
    int32_t position, int32_t rows, int32_t cols) {
  return kInputColumnMajor ? position % rows : position / cols;
}

template <bool kInputColumnMajor>
__device__ __forceinline__ int32_t logical_col(
    int32_t position, int32_t rows, int32_t cols) {
  return kInputColumnMajor ? position / rows : position % cols;
}

__device__ __forceinline__ uint8_t encode_e8m0(float maximum) {
  if (isnan(maximum)) return 255;
  const float exponent = ceilf(log2f(maximum / 448.0f));
  return static_cast<uint8_t>(fminf(fmaxf(exponent, -127.0f), 127.0f) + 127.0f);
}

__device__ __forceinline__ float decode_e8m0(uint8_t code) {
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

__global__ void finalize_stats_kernel(
    const float* partials,
    int64_t count,
    int64_t numel,
    float* stats
) {
  float sums[4] = {};
  for (int64_t index = threadIdx.x; index < count; index += blockDim.x) {
#pragma unroll
    for (int value = 0; value < 4; ++value) sums[value] += partials[index * 4 + value];
  }
  __shared__ float warp_sums[4][8];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
#pragma unroll
  for (int value = 0; value < 4; ++value) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      sums[value] += __shfl_down_sync(0xffffffffu, sums[value], offset);
    }
    if (lane == 0) warp_sums[value][warp] = sums[value];
  }
  __syncthreads();
  if (warp == 0) {
#pragma unroll
    for (int value = 0; value < 4; ++value) {
      float sum = lane < blockDim.x / 32 ? warp_sums[value][lane] : 0.0f;
#pragma unroll
      for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffffu, sum, offset);
      }
      if (lane == 0) stats[value == 3 ? 4 : value] = sum;
    }
    if (lane == 0) stats[3] = static_cast<float>(numel);
  }
}

template <int outer_size, int inner_size>
struct ScaleShape {
  static constexpr int outer = outer_size;
  static constexpr int inner = inner_size;
};

template <typename input_t, typename scale_shape, bool contract_contiguous,
          bool input_column_major, bool transpose_output, bool stochastic_rounding,
          bool collect_stats>
struct Tile {
  static constexpr int rows = scale_shape::outer != 1 ? scale_shape::inner
      : input_column_major
      ? (contract_contiguous ? (transpose_output ? scale_shape::inner : 512) : 32)
      : (contract_contiguous ? (transpose_output ? 32 : 2) : scale_shape::inner);
  static constexpr int cols = scale_shape::outer != 1 ? scale_shape::inner
      : input_column_major
      ? (contract_contiguous ? (transpose_output ? 32 : 2) : scale_shape::inner)
      : (contract_contiguous ? (transpose_output ? scale_shape::inner : 512) : 32);
  static constexpr bool vector = contract_contiguous && scale_shape::outer == 1;
  static constexpr bool stream = vector && !transpose_output;
  static constexpr int threads =
      vector ? rows * cols / 16
             : (scale_shape::outer == 32 && scale_shape::inner == 32 &&
                        sizeof(input_t) == 2 &&
                        !stochastic_rounding && !collect_stats
                    ? 64
                    : 256);
  static constexpr int values = rows * cols / threads;
};

template <typename input_t, typename scale_shape, int contract_dim,
          bool input_column_major,
          bool transpose_output, bool stochastic_rounding, bool collect_stats>
__global__ void quantize_mxfp8_kernel(
    QuantizeParams params,
    int64_t row_tiles,
    int64_t col_tiles,
    int64_t first_row_tile,
    int64_t first_col_tile,
    int64_t first_batch,
    bool narrow_offsets,
    at::PhiloxCudaState philox
) {
  constexpr bool contract_contiguous =
      contract_dim == (input_column_major ? -2 : -1);
  using tile = Tile<input_t, scale_shape, contract_contiguous, input_column_major,
                    transpose_output,
                    stochastic_rounding, collect_stats>;
  float* const stats_partials = params.statistics_partials;
  const int64_t row_tile = first_row_tile +
      (input_column_major ? blockIdx.x : blockIdx.y);
  const int64_t col_tile = first_col_tile +
      (input_column_major ? blockIdx.y : blockIdx.x);
  const int64_t batch = first_batch + blockIdx.z;
  const int64_t tile_index = input_column_major
      ? row_tile + row_tiles * (col_tile + col_tiles * batch)
      : col_tile + col_tiles * (row_tile + row_tiles * batch);
  int64_t row_origin, col_origin;
  if constexpr (tile::stream) {
    const int64_t contract_size = contract_dim == -1 ? params.cols : params.rows;
    const int64_t padded_contract =
        ((contract_size - 1) / scale_shape::inner + 1) *
        scale_shape::inner;
    const int64_t first = (contract_dim == -1 ? col_tile : row_tile) *
                          tile::rows * tile::cols +
                          static_cast<int64_t>(threadIdx.x) * tile::values;
    const int64_t outer_origin = first / padded_contract;
    const int64_t contract_origin = first % padded_contract;
    row_origin = contract_dim == -1 ? outer_origin : contract_origin;
    col_origin = contract_dim == -1 ? contract_origin : outer_origin;
  } else {
    row_origin = row_tile * tile::rows;
    col_origin = col_tile * tile::cols;
  }
  const int32_t row_size = static_cast<int32_t>(min(
      max(params.rows - row_origin, int64_t{0}),
      int64_t{tile::stream ? (contract_dim == -2 ? tile::values : 1) : tile::rows}));
  const int32_t col_size = static_cast<int32_t>(min(
      max(params.cols - col_origin, int64_t{0}),
      int64_t{tile::stream ? (contract_dim == -1 ? tile::values : 1) : tile::cols}));
  const bool valid_tile = row_size != 0 && col_size != 0;
  const input_t* input = nullptr;
  uint8_t* codes = nullptr;
  uint8_t* scales = nullptr;
  if (valid_tile) {
    input = params.input_data<input_t>() +
        input_offset(params, batch, row_origin, col_origin);
    codes = params.code_data<uint8_t>() +
        output_offset(params, batch, row_origin, col_origin);
    const int64_t scale_row =
        contract_dim == -1 ? row_origin : row_origin / scale_shape::inner;
    const int64_t scale_col =
        contract_dim == -1 ? col_origin / scale_shape::inner : col_origin;
    scales = params.scale_data<uint8_t>() +
        scale_offset(params, batch, scale_row, scale_col);
  }
  const int lane = threadIdx.x % 32;
  const int warp = threadIdx.x / 32;
  float values[tile::values] = {};
  float source_values[tile::values];

  if constexpr (tile::vector) {
    const int first = tile::stream ? 0 : threadIdx.x * tile::values;
    const int32_t row = logical_row<input_column_major>(first, tile::rows, tile::cols);
    const int32_t col = logical_col<input_column_major>(first, tile::rows, tile::cols);
    const input_t* source = row < row_size && col < col_size
        ? input + local_offset(row, col, params.input_row_stride,
                              params.input_col_stride, narrow_offsets)
        : nullptr;
    if (row < row_size && col < col_size &&
        (input_column_major ? row + tile::values <= row_size
                            : col + tile::values <= col_size) &&
        (input_column_major ? params.input_row_stride : params.input_col_stride) == 1 &&
        reinterpret_cast<uintptr_t>(source) % alignof(uint4) == 0) {
      if constexpr (std::is_same_v<input_t, float>) {
#pragma unroll
        for (int index = 0; index < tile::values; index += 4) {
          const float4 packed =
              *reinterpret_cast<const float4*>(source + index);
          values[index] = packed.x;
          values[index + 1] = packed.y;
          values[index + 2] = packed.z;
          values[index + 3] = packed.w;
        }
      } else {
#pragma unroll
        for (int index = 0; index < tile::values; index += 8) {
          const uint4 packed = *reinterpret_cast<const uint4*>(source + index);
          const uint32_t words[4] = {packed.x, packed.y, packed.z, packed.w};
#pragma unroll
          for (int element = 0; element < 8; ++element) {
            const uint16_t bits = static_cast<uint16_t>(words[element / 2] >>
                                                        ((element & 1) * 16));
            if constexpr (std::is_same_v<input_t, at::Half>) {
              values[index + element] = __half2float(__ushort_as_half(bits));
            } else {
              values[index + element] =
                  __uint_as_float(static_cast<uint32_t>(bits) << 16);
            }
          }
        }
      }
    } else {
#pragma unroll
      for (int index = 0; index < tile::values; ++index) {
        const int32_t value_row = input_column_major ? row + index : row;
        const int32_t value_col = input_column_major ? col : col + index;
        if (value_row < row_size && value_col < col_size) {
          values[index] = static_cast<float>(input[local_offset(
              value_row, value_col, params.input_row_stride,
              params.input_col_stride, narrow_offsets)]);
        }
      }
    }
  } else {
#pragma unroll
    for (int index = 0; index < tile::values; ++index) {
      const int position = threadIdx.x + index * tile::threads;
      const int32_t row = logical_row<input_column_major>(
          position, tile::rows, tile::cols);
      const int32_t col = logical_col<input_column_major>(
          position, tile::rows, tile::cols);
      if (row < row_size && col < col_size) {
        values[index] = static_cast<float>(input[local_offset(
            row, col, params.input_row_stride, params.input_col_stride,
            narrow_offsets)]);
      }
    }
  }

  if constexpr (collect_stats) {
#pragma unroll
    for (int index = 0; index < tile::values; ++index) {
      source_values[index] = values[index];
    }
  }

  // A thread's registers belong to one scale group (or one square tile).
  float maximum = 0.0f;
  int has_nan = 0;
#pragma unroll
  for (int index = 0; index < tile::values; ++index) {
    maximum = fmaxf(maximum, fabsf(values[index]));
    has_nan |= isnan(values[index]);
  }
  uint8_t scale;
  if constexpr (tile::vector) {
    constexpr int group_lanes = scale_shape::inner / tile::values;
    const int group_lane = lane % group_lanes;
    const unsigned mask = ((1u << group_lanes) - 1) << (lane - group_lane);
#pragma unroll
    for (int offset = group_lanes / 2; offset > 0; offset /= 2) {
      maximum =
          fmaxf(maximum, __shfl_down_sync(mask, maximum, offset, group_lanes));
    }
    has_nan = __any_sync(mask, has_nan);
    scale = group_lane == 0 ? encode_e8m0(has_nan ? nanf("") : maximum) : 0;
    scale = static_cast<uint8_t>(
        __shfl_sync(mask, static_cast<int>(scale), 0, group_lanes));
    const int first = tile::stream ? 0 : threadIdx.x * tile::values;
    const int32_t row = logical_row<input_column_major>(first, tile::rows, tile::cols);
    const int32_t col = logical_col<input_column_major>(first, tile::rows, tile::cols);
    if (group_lane == 0 && row < row_size && col < col_size) {
      const int32_t scale_row = contract_dim == -1 ? row : row / scale_shape::inner;
      const int32_t scale_col = contract_dim == -1 ? col / scale_shape::inner : col;
      scales[local_offset(scale_row, scale_col, params.scale_row_stride,
                          params.scale_col_stride, narrow_offsets)] = scale;
    }
  } else if constexpr (scale_shape::outer == 1) {
    __shared__ float partial_maximum[tile::threads / 32][32];
    __shared__ int partial_nan[tile::threads / 32][32];
    __shared__ uint8_t group_scales[32];
    partial_maximum[warp][lane] = maximum;
    partial_nan[warp][lane] = has_nan;
    __syncthreads();
    if (warp == 0) {
#pragma unroll
      for (int other = 1; other < tile::threads / 32; ++other) {
        maximum = fmaxf(maximum, partial_maximum[other][lane]);
        has_nan |= partial_nan[other][lane];
      }
      group_scales[lane] = encode_e8m0(has_nan ? nanf("") : maximum);
      const int32_t row = logical_row<input_column_major>(
          lane, tile::rows, tile::cols);
      const int32_t col = logical_col<input_column_major>(
          lane, tile::rows, tile::cols);
      if (row < row_size && col < col_size) {
        const int32_t scale_row =
            contract_dim == -1 ? row : row / scale_shape::inner;
        const int32_t scale_col =
            contract_dim == -1 ? col / scale_shape::inner : col;
        scales[local_offset(scale_row, scale_col, params.scale_row_stride,
                            params.scale_col_stride, narrow_offsets)] =
            group_scales[lane];
      }
    }
    __syncthreads();
    scale = group_scales[lane];
  } else {
    __shared__ float warp_maximum[tile::threads / 32];
    __shared__ int warp_nan[tile::threads / 32];
    __shared__ uint8_t tile_scale;
#pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
      maximum = fmaxf(maximum, __shfl_down_sync(0xffffffffu, maximum, offset));
    }
    has_nan = __any_sync(0xffffffffu, has_nan);
    if (lane == 0) {
      warp_maximum[warp] = maximum;
      warp_nan[warp] = has_nan;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
#pragma unroll
      for (int other = 1; other < tile::threads / 32; ++other) {
        maximum = fmaxf(maximum, warp_maximum[other]);
        has_nan |= warp_nan[other];
      }
      tile_scale = encode_e8m0(has_nan ? nanf("") : maximum);
    }
    __syncthreads();
    scale = tile_scale;
    if (threadIdx.x < scale_shape::outer) {
      const int32_t row = contract_dim == -1 ? threadIdx.x : 0;
      const int32_t col = contract_dim == -1 ? 0 : threadIdx.x;
      if (row < row_size && col < col_size) {
        const int32_t scale_row =
            contract_dim == -1 ? row : row / scale_shape::inner;
        const int32_t scale_col =
            contract_dim == -1 ? col / scale_shape::inner : col;
        scales[local_offset(scale_row, scale_col, params.scale_row_stride,
                            params.scale_col_stride, narrow_offsets)] = scale;
      }
    }
  }

  const float reciprocal = decode_e8m0(scale);
#pragma unroll
  for (int index = 0; index < tile::values; ++index) {
    values[index] *= reciprocal;
    if (!isnan(values[index]))
      values[index] = fminf(fmaxf(values[index], -448.0f), 448.0f);
  }
  uint8_t encoded[tile::values];
  if constexpr (stochastic_rounding) {
    const auto seeds = at::cuda::philox::unpack(philox);
    curandStatePhilox4_32_10_t state;
    curand_init(static_cast<unsigned long long>(std::get<0>(seeds)),
                static_cast<unsigned long long>(tile_index * tile::threads +
                                                threadIdx.x),
                static_cast<unsigned long long>(std::get<1>(seeds)), &state);
#pragma unroll
    for (int index = 0; index < tile::values; index += 4) {
      const uint4 random = curand4(&state);
      const uint32_t words[4] = {random.x, random.y, random.z, random.w};
#pragma unroll
      for (int element = 0; element < 4 && index + element < tile::values;
           ++element) {
        encoded[index + element] =
            encode_fp8_stochastic<false>(values[index + element], words[element]);
      }
    }
  } else if constexpr (tile::values % 2 == 0) {
#pragma unroll
    for (int index = 0; index < tile::values; index += 2) {
      const uint16_t packed = __nv_cvt_float2_to_fp8x2(
          make_float2(values[index], values[index + 1]), __NV_SATFINITE,
          __NV_E4M3);
      encoded[index] = static_cast<uint8_t>(packed);
      encoded[index + 1] = static_cast<uint8_t>(packed >> 8);
    }
  } else {
    encoded[0] = encode_fp8_rne<false>(values[0]);
  }

  if constexpr (transpose_output) {
    // Padding preserves coalesced stores without shared-memory bank conflicts.
    __shared__ uint8_t transposed[input_column_major ? tile::cols : tile::rows]
                                [(input_column_major ? tile::rows : tile::cols) + 4];
#pragma unroll
    for (int index = 0; index < tile::values; ++index) {
      const int position = tile::vector
          ? threadIdx.x * tile::values + index
          : threadIdx.x + index * tile::threads;
      const int32_t row = logical_row<input_column_major>(
          position, tile::rows, tile::cols);
      const int32_t col = logical_col<input_column_major>(
          position, tile::rows, tile::cols);
      transposed[input_column_major ? col : row]
                [input_column_major ? row : col] = encoded[index];
    }
    __syncthreads();
#pragma unroll
    for (int index = 0; index < tile::values; ++index) {
      const int position = threadIdx.x + index * tile::threads;
      const int32_t row = logical_row<!input_column_major>(
          position, tile::rows, tile::cols);
      const int32_t col = logical_col<!input_column_major>(
          position, tile::rows, tile::cols);
      if (row < row_size && col < col_size) {
        codes[local_offset(row, col, params.code_row_stride,
                           params.code_col_stride, narrow_offsets)] =
            transposed[input_column_major ? col : row]
                      [input_column_major ? row : col];
      }
    }
  } else if constexpr (tile::vector) {
    const int first = tile::stream ? 0 : threadIdx.x * tile::values;
    const int32_t row = logical_row<input_column_major>(first, tile::rows, tile::cols);
    const int32_t col = logical_col<input_column_major>(first, tile::rows, tile::cols);
    uint8_t* destination = row < row_size && col < col_size
        ? codes + local_offset(row, col, params.code_row_stride,
                              params.code_col_stride, narrow_offsets)
        : nullptr;
    if (row < row_size && col < col_size &&
        (input_column_major ? row + tile::values <= row_size
                            : col + tile::values <= col_size) &&
        reinterpret_cast<uintptr_t>(destination) % alignof(uint4) == 0) {
      uint32_t words[4] = {};
#pragma unroll
      for (int index = 0; index < 16; ++index) {
        words[index / 4] |= static_cast<uint32_t>(encoded[index])
                            << (8 * (index % 4));
      }
      *reinterpret_cast<uint4*>(destination) = {words[0], words[1], words[2],
                                                words[3]};
    } else {
#pragma unroll
      for (int index = 0; index < tile::values; ++index) {
        const int32_t value_row = input_column_major ? row + index : row;
        const int32_t value_col = input_column_major ? col : col + index;
        if (value_row < row_size && value_col < col_size) {
          codes[local_offset(value_row, value_col, params.code_row_stride,
                             params.code_col_stride, narrow_offsets)] = encoded[index];
        }
      }
    }
  } else {
#pragma unroll
    for (int index = 0; index < tile::values; ++index) {
      const int position = threadIdx.x + index * tile::threads;
      const int32_t row = logical_row<input_column_major>(
          position, tile::rows, tile::cols);
      const int32_t col = logical_col<input_column_major>(
          position, tile::rows, tile::cols);
      if (row < row_size && col < col_size) {
        codes[local_offset(row, col, params.code_row_stride,
                           params.code_col_stride, narrow_offsets)] =
            encoded[index];
      }
    }
  }

  if constexpr (collect_stats) {
    QuantizationStatistics statistics;
#pragma unroll
    for (int index = 0; index < tile::values; ++index) {
      const int position = tile::vector
          ? (tile::stream ? index : threadIdx.x * tile::values + index)
          : threadIdx.x + index * tile::threads;
      const int32_t row = logical_row<input_column_major>(
          position, tile::rows, tile::cols);
      const int32_t col = logical_col<input_column_major>(
          position, tile::rows, tile::cols);
      if (row < row_size && col < col_size) {
        const float source = source_values[index];
        const float reconstructed = decode_fp8<false>(encoded[index]) /
                                    decode_e8m0(scale);
        accumulate_quantization_statistics(
            &statistics, source, reconstructed, (encoded[index] & 0x7f) == 0);
      }
    }
    __shared__ float stats_warp_sums[4][8];
    float stat_values[4] = {
        statistics.src_sq,
        statistics.err_sq,
        statistics.under,
        statistics.nonzero,
    };
#pragma unroll
    for (int value = 0; value < 4; ++value) {
#pragma unroll
      for (int offset = 16; offset > 0; offset >>= 1) {
        stat_values[value] += __shfl_down_sync(0xffffffffu, stat_values[value], offset);
      }
      if (lane == 0) stats_warp_sums[value][warp] = stat_values[value];
    }
    __syncthreads();
    if (warp == 0) {
#pragma unroll
      for (int value = 0; value < 4; ++value) {
        float sum = lane < tile::threads / 32 ? stats_warp_sums[value][lane] : 0.0f;
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
          sum += __shfl_down_sync(0xffffffffu, sum, offset);
        }
        if (lane == 0) stats_partials[tile_index * 4 + value] = sum;
      }
    }
  }
}

template <typename input_t, typename scale_shape, int contract_dim,
          bool input_column_major,
          bool transpose_output, bool stochastic_rounding, bool collect_stats>
void launch_quantize_mxfp8(
    const at::Tensor& input,
    QuantizeParams params,
    at::PhiloxCudaState philox,
    cudaStream_t stream,
    at::Tensor* stats
) {
  constexpr bool contract_contiguous =
      contract_dim == (input_column_major ? -2 : -1);
  using tile = Tile<input_t, scale_shape, contract_contiguous, input_column_major,
                    transpose_output,
                    stochastic_rounding, collect_stats>;
  const int64_t limit = std::numeric_limits<int64_t>::max();
  int64_t row_tiles;
  int64_t col_tiles;
  if constexpr (tile::stream) {
    const int64_t contract_size = contract_dim == -1 ? params.cols : params.rows;
    const int64_t outer_size = contract_dim == -1 ? params.rows : params.cols;
    const int64_t groups = (contract_size - 1) / scale_shape::inner + 1;
    TORCH_CHECK(groups <= limit / scale_shape::inner,
                "MXFP8 padded row size exceeds int64 range");
    const int64_t padded_contract = groups * scale_shape::inner;
    TORCH_CHECK(outer_size <= limit / padded_contract,
                "MXFP8 padded input size exceeds int64 range");
    constexpr int tile_elements = tile::rows * tile::cols;
    const int64_t stream_tiles =
        (padded_contract * outer_size - 1) / tile_elements + 1;
    TORCH_CHECK(stream_tiles <= limit / tile_elements,
                "MXFP8 padded tile positions exceed int64 range");
    row_tiles = contract_dim == -1 ? 1 : stream_tiles;
    col_tiles = contract_dim == -1 ? stream_tiles : 1;
  } else {
    row_tiles = (params.rows - 1) / tile::rows + 1;
    col_tiles = (params.cols - 1) / tile::cols + 1;
  }
  TORCH_CHECK(row_tiles <= limit / col_tiles &&
                  params.batches <= limit / (row_tiles * col_tiles),
              "MXFP8 tile count exceeds int64 range");
  const int64_t count = params.batches * row_tiles * col_tiles;
  TORCH_CHECK(count <= limit / tile::threads,
              "MXFP8 logical thread count exceeds int64 range");

  const auto fits_int32 = [](int64_t rows, int64_t cols,
                             int64_t row_stride, int64_t col_stride) {
    const int64_t limit = std::numeric_limits<int32_t>::max();
    if (row_stride > limit || col_stride > limit) return false;
    const int64_t row_offset = (rows - 1) * row_stride;
    return row_offset <= limit &&
           (cols - 1) * col_stride <= limit - row_offset;
  };
  constexpr int local_rows = tile::stream
      ? (contract_dim == -2 ? tile::values : 1) : tile::rows;
  constexpr int local_cols = tile::stream
      ? (contract_dim == -1 ? tile::values : 1) : tile::cols;
  constexpr int scale_rows = contract_dim == -1
      ? local_rows : (local_rows + scale_shape::inner - 1) / scale_shape::inner;
  constexpr int scale_cols = contract_dim == -1
      ? (local_cols + scale_shape::inner - 1) / scale_shape::inner : local_cols;
  // Global bases stay 64-bit; narrow only offsets within a tile.
  const bool narrow_offsets =
      fits_int32(local_rows, local_cols,
                 params.input_row_stride, params.input_col_stride) &&
      fits_int32(local_rows, local_cols,
                 params.code_row_stride, params.code_col_stride) &&
      fits_int32(scale_rows, scale_cols,
                 params.scale_row_stride, params.scale_col_stride);
  at::Tensor partials;
  if constexpr (collect_stats) {
    partials = at::empty({count, 4}, input.options().dtype(at::kFloat));
    params.statistics_partials = partials.data_ptr<float>();
  }
  const auto* device = at::cuda::getCurrentDeviceProperties();
  for (int64_t first_batch = 0; first_batch < params.batches;) {
    const int64_t batch_chunk =
        std::min(params.batches - first_batch, int64_t{device->maxGridSize[2]});
    for (int64_t first_row = 0; first_row < row_tiles;) {
      const int64_t row_chunk = std::min(
          row_tiles - first_row, int64_t{device->maxGridSize[input_column_major ? 0 : 1]});
      for (int64_t first_col = 0; first_col < col_tiles;) {
        const int64_t col_chunk = std::min(
            col_tiles - first_col, int64_t{device->maxGridSize[input_column_major ? 1 : 0]});
        const dim3 grid(input_column_major ? row_chunk : col_chunk,
                        input_column_major ? col_chunk : row_chunk, batch_chunk);
        quantize_mxfp8_kernel<input_t, scale_shape, contract_dim, input_column_major,
                              transpose_output, stochastic_rounding, collect_stats>
            <<<grid, tile::threads, 0, stream>>>(
                params, row_tiles, col_tiles, first_row, first_col, first_batch,
                narrow_offsets, philox);
        first_col += col_chunk;
      }
      first_row += row_chunk;
    }
    first_batch += batch_chunk;
  }
  if constexpr (collect_stats) {
    finalize_stats_kernel<<<1, 256, 0, stream>>>(
        params.statistics_partials, count, input.numel(), stats->data_ptr<float>());
  }
}

// Keep runtime dispatch separate from the shared kernel body.
#define BOOL_DISPATCH(CONDITION, NAME, ...) \
  if (CONDITION) {                          \
    constexpr bool NAME = true;             \
    __VA_ARGS__;                            \
  } else {                                  \
    constexpr bool NAME = false;            \
    __VA_ARGS__;                            \
  }

template <typename input_t, typename scale_shape>
void dispatch_layout(
    const at::Tensor& input,
    QuantizeParams params,
    int64_t contract_dim,
    bool input_column_major,
    bool transpose_output,
    bool stochastic_rounding,
    at::PhiloxCudaState philox,
    cudaStream_t stream,
    at::Tensor* stats
) {
  BOOL_DISPATCH(
      contract_dim == -1, kLastDim,
      BOOL_DISPATCH(
          input_column_major, kInputColumnMajor,
          BOOL_DISPATCH(
              transpose_output, kTransposeOutput,
              BOOL_DISPATCH(
                  stochastic_rounding, kStochasticRounding,
                  BOOL_DISPATCH(
                      stats != nullptr, kCollectStats,
                      launch_quantize_mxfp8<
                          input_t,
                          scale_shape,
                          (kLastDim ? -1 : -2),
                          kInputColumnMajor,
                          kTransposeOutput,
                          kStochasticRounding,
                          kCollectStats>(
                          input, params, philox, stream,
                          stats))))))
}

template <typename input_t>
void dispatch_scale_shape(
    const at::Tensor& input,
    QuantizeParams params,
    int64_t block_size,
    int64_t block_outer,
    int64_t contract_dim,
    bool input_column_major,
    bool transpose_output,
    bool stochastic_rounding,
    at::PhiloxCudaState philox,
    cudaStream_t stream,
    at::Tensor* stats
) {
#define LAUNCH_SIZE(SIZE)                                                   \
  BOOL_DISPATCH(                                                            \
      block_outer != 1, kSquare,                                           \
      dispatch_layout<input_t, ScaleShape<(kSquare ? SIZE : 1), SIZE>>(    \
          input, params, contract_dim, input_column_major,                 \
          transpose_output, stochastic_rounding, philox, stream, stats))
  switch (block_size) {
    case 16:
      LAUNCH_SIZE(16);
      break;
    case 32:
      LAUNCH_SIZE(32);
      break;
    case 64:
      LAUNCH_SIZE(64);
      break;
    case 128:
      LAUNCH_SIZE(128);
      break;
  }
#undef LAUNCH_SIZE
}
#undef BOOL_DISPATCH

std::tuple<at::Tensor, at::Tensor> quantize_mxfp8_meta(
    const at::Tensor& input,
    int64_t contract_dim,
    int64_t block_outer,
    int64_t block_size,
    bool stochastic_rounding,
    const std::string& output_layout
) {
  auto scale_sizes = input.sym_sizes().vec();
  const int64_t contract_axis = contract_dim + input.dim();
  scale_sizes[contract_axis] =
      (scale_sizes[contract_axis] + block_size - 1) / block_size;
  const auto options = input.options().dtype(at::kFloat8_e4m3fn);
  at::Tensor codes;
  if (output_layout == "column_major") {
    const auto rows = input.sym_size(-2), cols = input.sym_size(-1);
    const std::vector<c10::SymInt> strides =
        input.dim() == 3 ? std::vector<c10::SymInt>{rows * cols, 1, rows}
                         : std::vector<c10::SymInt>{1, rows};
    codes = at::empty_strided_symint(input.sym_sizes(), strides, options);
  } else {
    codes = at::empty_symint(input.sym_sizes(), options);
  }
  return {codes, at::empty_symint(scale_sizes,
                                  input.options().dtype(at::kFloat8_e8m0fnu))};
}

std::tuple<at::Tensor, at::Tensor> quantize_mxfp8_impl(
    const at::Tensor& input,
    int64_t contract_dim,
    int64_t block_outer,
    int64_t block_size,
    bool stochastic_rounding,
    const std::string& output_layout,
    at::Tensor* stats
) {
  TORCH_CHECK(input.is_cuda(), "quantize_mxfp8 requires a CUDA tensor");
  TORCH_CHECK(input.dim() == 2 || input.dim() == 3,
              "quantize_mxfp8 requires a 2D or 3D tensor");
  TORCH_CHECK(input.scalar_type() == at::kFloat ||
                  input.scalar_type() == at::kHalf ||
                  input.scalar_type() == at::kBFloat16,
              "quantize_mxfp8 requires float32, float16, or bfloat16 input");
  TORCH_CHECK(contract_dim == -1 || contract_dim == -2,
              "quantize_mxfp8 contract_dim must be -2 or -1");
  TORCH_CHECK(block_size == 16 || block_size == 32 || block_size == 64 ||
                  block_size == 128,
              "quantize_mxfp8 block_size must be 16, 32, 64, or 128");
  TORCH_CHECK(block_outer == 1 || block_outer == block_size,
              "quantize_mxfp8 block_outer must be 1 or block_size");
  TORCH_CHECK(output_layout == "row_major" || output_layout == "column_major",
              "quantize_mxfp8 output_layout must be row_major or column_major");
  TORCH_CHECK(std::all_of(input.strides().begin(), input.strides().end(),
                          [](int64_t stride) { return stride >= 0; }),
              "quantize_mxfp8 requires nonnegative strides");
  c10::cuda::CUDAGuard device_guard(input.device());
  auto [codes, scales] =
      quantize_mxfp8_meta(input, contract_dim, block_outer, block_size,
                          stochastic_rounding, output_layout);
  if (stats != nullptr) *stats = at::zeros({5}, input.options().dtype(at::kFloat));
  if (input.numel() == 0) return {codes, scales};

  const bool input_column_major = input.stride(-2) < input.stride(-1);
  QuantizeParams params;
  params.input = input.const_data_ptr();
  params.codes = codes.data_ptr();
  params.scales = scales.data_ptr();
  params.statistics = stats != nullptr ? stats->data_ptr<float>() : nullptr;
  params.batches = input.dim() == 3 ? input.size(0) : 1;
  params.rows = input.size(-2);
  params.cols = input.size(-1);
  params.input_batch_stride = input.dim() == 3 ? input.stride(0) : 0;
  params.input_row_stride = input.stride(-2);
  params.input_col_stride = input.stride(-1);
  params.code_batch_stride = codes.dim() == 3 ? codes.stride(0) : 0;
  params.code_row_stride = codes.stride(-2);
  params.code_col_stride = codes.stride(-1);
  params.scale_batch_stride = scales.dim() == 3 ? scales.stride(0) : 0;
  params.scale_row_stride = scales.stride(-2);
  params.scale_col_stride = scales.stride(-1);
  params.rows_per_block = contract_dim == -1 ? block_outer : block_size;
  params.cols_per_block = contract_dim == -1 ? block_size : block_outer;
  params.row_block_count = (params.rows + params.rows_per_block - 1) /
      params.rows_per_block;
  params.col_block_count = (params.cols + params.cols_per_block - 1) /
      params.cols_per_block;
  params.contract_dim = static_cast<int>(contract_dim);
  const bool transpose_output =
      input_column_major != (output_layout == "column_major");
  at::PhiloxCudaState philox;
  if (stochastic_rounding) {
    auto* generator = at::get_generator_or_default<at::CUDAGeneratorImpl>(
        std::nullopt,
        at::cuda::detail::getDefaultCUDAGenerator(input.get_device()));
    std::lock_guard<std::mutex> lock(generator->mutex_);
    const int64_t draws = std::max(int64_t{16}, block_size * block_size / 256);
    philox = generator->philox_cuda_state(draws);
  }
  const auto stream = at::cuda::getCurrentCUDAStream(input.get_device());
#define LAUNCH_TYPE(TYPE)                                                      \
  dispatch_scale_shape<TYPE>(input, params, block_size,                         \
                             block_outer, contract_dim, input_column_major,     \
                             transpose_output, stochastic_rounding, philox,     \
                             stream.stream(), stats)
  switch (input.scalar_type()) {
    case at::kFloat:
      LAUNCH_TYPE(float);
      break;
    case at::kHalf:
      LAUNCH_TYPE(at::Half);
      break;
    case at::kBFloat16:
      LAUNCH_TYPE(at::BFloat16);
      break;
    default:
      TORCH_CHECK(false, "unsupported MXFP8 input dtype");
  }
#undef LAUNCH_TYPE
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {codes, scales};
}

std::tuple<at::Tensor, at::Tensor> quantize_mxfp8_cuda(
    const at::Tensor& input,
    int64_t contract_dim,
    int64_t block_outer,
    int64_t block_size,
    bool stochastic_rounding,
    const std::string& output_layout
) {
  return quantize_mxfp8_impl(input, contract_dim, block_outer, block_size,
                              stochastic_rounding, output_layout, nullptr);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> quantize_mxfp8_with_stats_cuda(
    const at::Tensor& input,
    int64_t contract_dim,
    int64_t block_outer,
    int64_t block_size,
    bool stochastic_rounding,
    const std::string& output_layout
) {
  at::Tensor stats;
  auto [codes, scales] = quantize_mxfp8_impl(
      input, contract_dim, block_outer, block_size, stochastic_rounding,
      output_layout, &stats);
  return {codes, scales, stats};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> quantize_mxfp8_with_stats_meta(
    const at::Tensor& input,
    int64_t contract_dim,
    int64_t block_outer,
    int64_t block_size,
    bool stochastic_rounding,
    const std::string& output_layout
) {
  auto [codes, scales] = quantize_mxfp8_meta(
      input, contract_dim, block_outer, block_size, stochastic_rounding,
      output_layout);
  return {codes, scales, at::empty_symint({5}, input.options().dtype(at::kFloat))};
}

}  // namespace

TORCH_LIBRARY_FRAGMENT(aot_kernel, m) {
  m.def(
      "quantize_mxfp8(Tensor x, int contract_dim, int block_outer, int "
      "block_size, "
      "bool stochastic_rounding, str output_layout=\"row_major\") -> (Tensor, "
      "Tensor)",
      {at::Tag::nondeterministic_seeded});
  m.def(
      "quantize_mxfp8_with_stats(Tensor x, int contract_dim, int block_outer, int "
      "block_size, bool stochastic_rounding, str output_layout=\"row_major\") -> "
      "(Tensor, Tensor, Tensor)",
      {at::Tag::nondeterministic_seeded});
}
TORCH_LIBRARY_IMPL(aot_kernel, Meta, m) {
  m.impl("quantize_mxfp8", TORCH_FN(quantize_mxfp8_meta));
  m.impl("quantize_mxfp8_with_stats", TORCH_FN(quantize_mxfp8_with_stats_meta));
}
TORCH_LIBRARY_IMPL(aot_kernel, CUDA, m) {
  m.impl("quantize_mxfp8", TORCH_FN(quantize_mxfp8_cuda));
  m.impl("quantize_mxfp8_with_stats", TORCH_FN(quantize_mxfp8_with_stats_cuda));
}
