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

struct UnsignedMax {
  __device__ __forceinline__ uint32_t operator()(uint32_t lhs, uint32_t rhs) const {
    return max(lhs, rhs);
  }
};

struct Mxfp8LaunchParams {
  int64_t row_tiles;
  int64_t col_tiles;
  int64_t first_row_tile;
  int64_t first_col_tile;
  int64_t first_batch;
  bool narrow_offsets;
  bool swizzled_scales;
  bool initialize_swizzled_padding;
  int64_t logical_scale_rows;
  int64_t logical_scale_cols;
};

__device__ __forceinline__ void write_mxfp8_scale(
    const QuantizeParams& params,
    const Mxfp8LaunchParams& launch,
    uint8_t* scales,
    int64_t scale_row_origin,
    int64_t scale_col_origin,
    int32_t scale_row,
    int32_t scale_col,
    uint8_t value) {
  if (!launch.swizzled_scales) {
    scales[tile_element_offset(
        scale_row, scale_col, params.scale_row_stride, params.scale_col_stride,
        launch.narrow_offsets)] = value;
    return;
  }
  const int64_t logical_row =
      params.contract_dim == -1 ? scale_row_origin + scale_row
                                : scale_col_origin + scale_col;
  const int64_t logical_col =
      params.contract_dim == -1 ? scale_col_origin + scale_col
                                : scale_row_origin + scale_row;
  swizzled_32_4_4_scale_epilogue(
      scales, value, logical_row, logical_col, launch.logical_scale_rows,
      launch.logical_scale_cols);
}

template <typename input_t, typename block_shape_t, int kContractDim,
          bool kTransposeOutput,
          bool kStochasticRounding, bool kCollectStats>
struct Mxfp8LaunchConfig {
  using input_type = input_t;
  using block_shape = block_shape_t;
  static constexpr int contract_dim = kContractDim;
  static constexpr bool transpose_output = kTransposeOutput;
  static constexpr bool stochastic_rounding = kStochasticRounding;
  static constexpr bool collect_stats = kCollectStats;
  static constexpr bool contract_contiguous =
      contract_dim == -1;
  static constexpr bool contiguous_per_thread =
      contract_contiguous && block_shape::dim0 == 1;
  static constexpr bool flattened_traversal =
      contiguous_per_thread && !transpose_output;
  static constexpr int tile_rows = block_shape::dim0 != 1 ? block_shape::dim1
      : (contract_contiguous ? (transpose_output ? 32 : 2) : block_shape::dim1);
  static constexpr int tile_cols = block_shape::dim0 != 1 ? block_shape::dim1
      : (contract_contiguous ? (transpose_output ? block_shape::dim1 : 512) : 32);
  static constexpr int threads_per_block = contiguous_per_thread
      ? tile_rows * tile_cols / 16
      : (block_shape::dim0 == 32 && block_shape::dim1 == 32 &&
                 sizeof(input_t) == 2 && !stochastic_rounding && !collect_stats
             ? 64
             : 256);
  static constexpr int elements_per_thread =
      tile_rows * tile_cols / threads_per_block;
};

template <typename launch_config>
__global__ void quantize_mxfp8_kernel(
    QuantizeParams params,
    Mxfp8LaunchParams launch,
    at::PhiloxCudaState philox
) {
  using input_t = typename launch_config::input_type;
  using block_shape = typename launch_config::block_shape;
  constexpr int contract_dim = launch_config::contract_dim;
  constexpr bool transpose_output = launch_config::transpose_output;
  constexpr bool stochastic_rounding = launch_config::stochastic_rounding;
  constexpr bool collect_stats = launch_config::collect_stats;
  constexpr bool contiguous_per_thread = launch_config::contiguous_per_thread;
  constexpr bool flattened_traversal = launch_config::flattened_traversal;
  constexpr int tile_rows = launch_config::tile_rows;
  constexpr int tile_cols = launch_config::tile_cols;
  constexpr int threads_per_block = launch_config::threads_per_block;
  constexpr int elements_per_thread = launch_config::elements_per_thread;
  const int64_t row_tile = launch.first_row_tile + blockIdx.y;
  const int64_t col_tile = launch.first_col_tile + blockIdx.x;
  const int64_t batch = launch.first_batch + blockIdx.z;
  const int64_t tile_index =
      col_tile + launch.col_tiles * (row_tile + launch.row_tiles * batch);
  // Padding writes are disjoint from logical scales, so no grid barrier is needed.
  if (launch.initialize_swizzled_padding) {
    swizzled_32_4_4_scale_epilogue(
        params.scale_data<uint8_t>(), 0, 0, 0, launch.logical_scale_rows,
        launch.logical_scale_cols, true);
  }
  int64_t row_origin, col_origin;
  if constexpr (flattened_traversal) {
    const int64_t contract_size = params.col;
    const int64_t padded_contract =
        ((contract_size - 1) / block_shape::dim1 + 1) *
        block_shape::dim1;
    const int64_t first = col_tile *
                          tile_rows * tile_cols +
                          static_cast<int64_t>(threadIdx.x) * elements_per_thread;
    row_origin = first / padded_contract;
    col_origin = first % padded_contract;
  } else {
    row_origin = row_tile * tile_rows;
    col_origin = col_tile * tile_cols;
  }
  const int32_t row_size = static_cast<int32_t>(min(
      max(params.row - row_origin, int64_t{0}),
      int64_t{flattened_traversal ? 1 : tile_rows}));
  const int32_t col_size = static_cast<int32_t>(min(
      max(params.col - col_origin, int64_t{0}),
      int64_t{flattened_traversal ? elements_per_thread : tile_cols}));
  const bool valid_tile = row_size != 0 && col_size != 0;
  const input_t* input = nullptr;
  uint8_t* codes = nullptr;
  uint8_t* scales = nullptr;
  int64_t scale_row_origin = 0;
  int64_t scale_col_origin = 0;
  if (valid_tile) {
    input = params.input_data<input_t>() +
        input_offset(params, batch, row_origin, col_origin);
    codes = params.code_data<uint8_t>() +
        output_offset(params, batch, row_origin, col_origin);
    scale_row_origin =
        contract_dim == -1 ? row_origin : row_origin / block_shape::dim1;
    scale_col_origin =
        contract_dim == -1 ? col_origin / block_shape::dim1 : col_origin;
    scales = params.scale_data<uint8_t>();
    if (!launch.swizzled_scales) {
      scales += scale_offset(
          params, batch, scale_row_origin, scale_col_origin);
    }
  }
  const int lane = threadIdx.x % 32;
  const int warp = threadIdx.x / 32;
  float values[elements_per_thread] = {};
  float source_values[elements_per_thread];

  if constexpr (contiguous_per_thread) {
    const int first = flattened_traversal ? 0 : threadIdx.x * elements_per_thread;
    const int32_t row = first / tile_cols;
    const int32_t col = first % tile_cols;
    const input_t* source = row < row_size && col < col_size
        ? input + tile_element_offset(row, col, params.input_row_stride,
                                      params.input_col_stride, launch.narrow_offsets)
        : nullptr;
    if (row < row_size && col < col_size &&
        col + elements_per_thread <= col_size &&
        params.input_col_stride == 1 &&
        reinterpret_cast<uintptr_t>(source) % alignof(uint4) == 0) {
      if constexpr (std::is_same_v<input_t, float>) {
#pragma unroll
        for (int index = 0; index < elements_per_thread; index += 4) {
          const float4 packed =
              *reinterpret_cast<const float4*>(source + index);
          values[index] = packed.x;
          values[index + 1] = packed.y;
          values[index + 2] = packed.z;
          values[index + 3] = packed.w;
        }
      } else {
#pragma unroll
        for (int index = 0; index < elements_per_thread; index += 8) {
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
      for (int index = 0; index < elements_per_thread; ++index) {
        const int32_t value_row = row;
        const int32_t value_col = col + index;
        if (value_row < row_size && value_col < col_size) {
          values[index] = static_cast<float>(input[tile_element_offset(
              value_row, value_col, params.input_row_stride,
              params.input_col_stride, launch.narrow_offsets)]);
        }
      }
    }
  } else {
#pragma unroll
    for (int index = 0; index < elements_per_thread; ++index) {
      const int position = threadIdx.x + index * threads_per_block;
      const int32_t row = position / tile_cols;
      const int32_t col = position % tile_cols;
      if (row < row_size && col < col_size) {
        values[index] = static_cast<float>(input[tile_element_offset(
            row, col, params.input_row_stride, params.input_col_stride,
            launch.narrow_offsets)]);
      }
    }
  }

  if constexpr (collect_stats) {
#pragma unroll
    for (int index = 0; index < elements_per_thread; ++index) {
      source_values[index] = values[index];
    }
  }

  // A thread's registers belong to one scale group (or one square tile).
  uint32_t maximum_bits = 0;
#pragma unroll
  for (int index = 0; index < elements_per_thread; ++index) {
    // Absolute float bits sort by magnitude, with NaNs above infinity.
    maximum_bits = max(maximum_bits, __float_as_uint(values[index]) & 0x7fffffffu);
  }
  uint8_t scale;
  if constexpr (contiguous_per_thread) {
    constexpr int group_lanes = block_shape::dim1 / elements_per_thread;
    const int group_lane = lane % group_lanes;
    const unsigned mask = ((1u << group_lanes) - 1) << (lane - group_lane);
    maximum_bits = cuda_reduce::warp_reduce<group_lanes>(
        maximum_bits, UnsignedMax{}, mask);
    scale = group_lane == 0 ? encode_fp8_e8m0(__uint_as_float(maximum_bits)) : 0;
    scale = static_cast<uint8_t>(
        __shfl_sync(mask, static_cast<int>(scale), 0, group_lanes));
    const int first = flattened_traversal ? 0 : threadIdx.x * elements_per_thread;
    const int32_t row = first / tile_cols;
    const int32_t col = first % tile_cols;
    if (group_lane == 0 && row < row_size && col < col_size) {
      const int32_t scale_row = contract_dim == -1 ? row : row / block_shape::dim1;
      const int32_t scale_col = contract_dim == -1 ? col / block_shape::dim1 : col;
      write_mxfp8_scale(
          params, launch, scales, scale_row_origin, scale_col_origin, scale_row,
          scale_col, scale);
    }
  } else if constexpr (block_shape::dim0 == 1) {
    __shared__ uint32_t partial_maximum[threads_per_block / 32][32];
    __shared__ uint8_t group_scales[32];
    partial_maximum[warp][lane] = maximum_bits;
    __syncthreads();
    if (warp == 0) {
#pragma unroll
      for (int other = 1; other < threads_per_block / 32; ++other) {
        maximum_bits = max(maximum_bits, partial_maximum[other][lane]);
      }
      group_scales[lane] = encode_fp8_e8m0(__uint_as_float(maximum_bits));
      const int32_t row = lane / tile_cols;
      const int32_t col = lane % tile_cols;
      if (row < row_size && col < col_size) {
        const int32_t scale_row =
            contract_dim == -1 ? row : row / block_shape::dim1;
        const int32_t scale_col =
            contract_dim == -1 ? col / block_shape::dim1 : col;
        write_mxfp8_scale(
            params, launch, scales, scale_row_origin, scale_col_origin, scale_row,
            scale_col, group_scales[lane]);
      }
    }
    __syncthreads();
    scale = group_scales[lane];
  } else {
    __shared__ uint32_t warp_maximum[threads_per_block / 32];
    __shared__ uint8_t tile_scale;
    maximum_bits = cuda_reduce::warp_reduce<32>(
        maximum_bits, UnsignedMax{}, 0xffffffffu);
    if (lane == 0) warp_maximum[warp] = maximum_bits;
    __syncthreads();
    if (threadIdx.x == 0) {
#pragma unroll
      for (int other = 1; other < threads_per_block / 32; ++other) {
        maximum_bits = max(maximum_bits, warp_maximum[other]);
      }
      tile_scale = encode_fp8_e8m0(__uint_as_float(maximum_bits));
    }
    __syncthreads();
    scale = tile_scale;
    if (threadIdx.x < block_shape::dim0) {
      const int32_t row = contract_dim == -1 ? threadIdx.x : 0;
      const int32_t col = contract_dim == -1 ? 0 : threadIdx.x;
      if (row < row_size && col < col_size) {
        const int32_t scale_row =
            contract_dim == -1 ? row : row / block_shape::dim1;
        const int32_t scale_col =
            contract_dim == -1 ? col / block_shape::dim1 : col;
        write_mxfp8_scale(
            params, launch, scales, scale_row_origin, scale_col_origin, scale_row,
            scale_col, scale);
      }
    }
  }

  const float reciprocal = decode_fp8_e8m0(scale);
#pragma unroll
  for (int index = 0; index < elements_per_thread; ++index) {
    values[index] *= reciprocal;
  }
  uint8_t encoded[elements_per_thread];
  if constexpr (stochastic_rounding) {
    const auto seeds = at::cuda::philox::unpack(philox);
    curandStatePhilox4_32_10_t state;
    curand_init(static_cast<unsigned long long>(std::get<0>(seeds)),
                static_cast<unsigned long long>(tile_index * threads_per_block +
                                                threadIdx.x),
                static_cast<unsigned long long>(std::get<1>(seeds)), &state);
#pragma unroll
    for (int index = 0; index < elements_per_thread; index += 4) {
      const uint4 random = curand4(&state);
      const uint32_t words[4] = {random.x, random.y, random.z, random.w};
#pragma unroll
      for (int element = 0; element < 4 && index + element < elements_per_thread;
           ++element) {
        encoded[index + element] =
            encode_fp8_stochastic<false>(values[index + element], words[element]);
      }
    }
  } else if constexpr (elements_per_thread % 2 == 0) {
#pragma unroll
    for (int index = 0; index < elements_per_thread; index += 2) {
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
    __shared__ uint8_t transposed[tile_rows][tile_cols + 4];
#pragma unroll
    for (int index = 0; index < elements_per_thread; ++index) {
      const int position = contiguous_per_thread
          ? threadIdx.x * elements_per_thread + index
          : threadIdx.x + index * threads_per_block;
      const int32_t row = position / tile_cols;
      const int32_t col = position % tile_cols;
      transposed[row][col] = encoded[index];
    }
    __syncthreads();
#pragma unroll
    for (int index = 0; index < elements_per_thread; ++index) {
      const int position = threadIdx.x + index * threads_per_block;
      const int32_t row = position % tile_rows;
      const int32_t col = position / tile_rows;
      if (row < row_size && col < col_size) {
        codes[tile_element_offset(row, col, params.code_row_stride,
                                  params.code_col_stride, launch.narrow_offsets)] =
            transposed[row][col];
      }
    }
  } else if constexpr (contiguous_per_thread) {
    const int first = flattened_traversal ? 0 : threadIdx.x * elements_per_thread;
    const int32_t row = first / tile_cols;
    const int32_t col = first % tile_cols;
    uint8_t* destination = row < row_size && col < col_size
        ? codes + tile_element_offset(row, col, params.code_row_stride,
                                      params.code_col_stride, launch.narrow_offsets)
        : nullptr;
    if (row < row_size && col < col_size &&
        col + elements_per_thread <= col_size &&
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
      for (int index = 0; index < elements_per_thread; ++index) {
        const int32_t value_row = row;
        const int32_t value_col = col + index;
        if (value_row < row_size && value_col < col_size) {
          codes[tile_element_offset(value_row, value_col, params.code_row_stride,
                                    params.code_col_stride, launch.narrow_offsets)] = encoded[index];
        }
      }
    }
  } else {
#pragma unroll
    for (int index = 0; index < elements_per_thread; ++index) {
      const int position = threadIdx.x + index * threads_per_block;
      const int32_t row = position / tile_cols;
      const int32_t col = position % tile_cols;
      if (row < row_size && col < col_size) {
        codes[tile_element_offset(row, col, params.code_row_stride,
                                  params.code_col_stride, launch.narrow_offsets)] =
            encoded[index];
      }
    }
  }

  if constexpr (collect_stats) {
    QuantizationStatistics partial;
#pragma unroll
    for (int index = 0; index < elements_per_thread; ++index) {
      const int position = contiguous_per_thread
          ? (flattened_traversal ? index
                                 : threadIdx.x * elements_per_thread + index)
          : threadIdx.x + index * threads_per_block;
      const int32_t row = position / tile_cols;
      const int32_t col = position % tile_cols;
      if (row < row_size && col < col_size) {
        const uint8_t code = encoded[index];
        accumulate_quantization_statistics(
            &partial, source_values[index],
            decode_fp8<false>(code) / decode_fp8_e8m0(scale));
      }
    }
    quantization_statistics_epilogue<threads_per_block>(
        partial, params.statistics, params.batch * params.row * params.col);
  }
}

template <typename launch_config>
void launch_quantize_mxfp8(
    QuantizeParams params,
    bool swizzled_scales,
    at::PhiloxCudaState philox,
    cudaStream_t stream
) {
  using block_shape = typename launch_config::block_shape;
  constexpr int contract_dim = launch_config::contract_dim;
  constexpr bool flattened_traversal = launch_config::flattened_traversal;
  constexpr int tile_rows = launch_config::tile_rows;
  constexpr int tile_cols = launch_config::tile_cols;
  constexpr int threads_per_block = launch_config::threads_per_block;
  constexpr int elements_per_thread = launch_config::elements_per_thread;
  const int64_t limit = std::numeric_limits<int64_t>::max();
  int64_t row_tiles;
  int64_t col_tiles;
  if constexpr (flattened_traversal) {
    const int64_t contract_size = params.col;
    const int64_t outer_size = params.row;
    const int64_t groups = (contract_size - 1) / block_shape::dim1 + 1;
    TORCH_CHECK(groups <= limit / block_shape::dim1,
                "MXFP8 padded row size exceeds int64 range");
    const int64_t padded_contract = groups * block_shape::dim1;
    TORCH_CHECK(outer_size <= limit / padded_contract,
                "MXFP8 padded input size exceeds int64 range");
    constexpr int tile_elements = tile_rows * tile_cols;
    const int64_t stream_tiles =
        (padded_contract * outer_size - 1) / tile_elements + 1;
    TORCH_CHECK(stream_tiles <= limit / tile_elements,
                "MXFP8 padded tile positions exceed int64 range");
    row_tiles = 1;
    col_tiles = stream_tiles;
  } else {
    row_tiles = (params.row - 1) / tile_rows + 1;
    col_tiles = (params.col - 1) / tile_cols + 1;
  }
  TORCH_CHECK(row_tiles <= limit / col_tiles &&
                  params.batch <= limit / (row_tiles * col_tiles),
              "MXFP8 tile count exceeds int64 range");
  const int64_t count = params.batch * row_tiles * col_tiles;
  TORCH_CHECK(count <= limit / threads_per_block,
              "MXFP8 logical thread count exceeds int64 range");

  constexpr int local_rows = flattened_traversal ? 1 : tile_rows;
  constexpr int local_cols = flattened_traversal ? elements_per_thread : tile_cols;
  constexpr int scale_rows = contract_dim == -1
      ? local_rows : (local_rows + block_shape::dim1 - 1) / block_shape::dim1;
  constexpr int scale_cols = contract_dim == -1
      ? (local_cols + block_shape::dim1 - 1) / block_shape::dim1 : local_cols;
  // Global bases stay 64-bit; narrow only offsets within a tile.
  const bool narrow_offsets =
      offsets_fit_int32(local_rows, local_cols,
                        params.input_row_stride, params.input_col_stride) &&
      offsets_fit_int32(local_rows, local_cols,
                        params.code_row_stride, params.code_col_stride) &&
      offsets_fit_int32(scale_rows, scale_cols,
                        params.scale_row_stride, params.scale_col_stride);
  const auto* device = at::cuda::getCurrentDeviceProperties();
  for (int64_t first_batch = 0; first_batch < params.batch;) {
    const int64_t batch_chunk =
        std::min(params.batch - first_batch, int64_t{device->maxGridSize[2]});
    for (int64_t first_row = 0; first_row < row_tiles;) {
      const int64_t row_chunk = std::min(
          row_tiles - first_row, int64_t{device->maxGridSize[1]});
      for (int64_t first_col = 0; first_col < col_tiles;) {
        const int64_t col_chunk = std::min(
            col_tiles - first_col, int64_t{device->maxGridSize[0]});
        const dim3 grid(col_chunk, row_chunk, batch_chunk);
        const Mxfp8LaunchParams launch{
            row_tiles,
            col_tiles,
            first_row,
            first_col,
            first_batch,
            narrow_offsets,
            swizzled_scales,
            swizzled_scales && first_batch == 0 && first_row == 0 &&
                first_col == 0,
            params.contract_dim == -1 ? params.row : params.col,
            params.contract_dim == -1 ? params.col_block_count
                                      : params.row_block_count};
        quantize_mxfp8_kernel<launch_config>
            <<<grid, threads_per_block, 0, stream>>>(params, launch, philox);
        first_col += col_chunk;
      }
      first_row += row_chunk;
    }
    first_batch += batch_chunk;
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

template <typename input_t, typename block_shape>
void dispatch_layout(
    QuantizeParams params,
    bool output_column_major,
    bool swizzled_scales,
    bool stochastic_rounding,
    at::PhiloxCudaState philox,
    cudaStream_t stream
) {
  const bool input_column_major =
      params.input_row_stride < params.input_col_stride;
  const bool transpose_output = input_column_major != output_column_major;
  // Normalize coordinates without changing physical tiles or Philox assignment.
  if (input_column_major) {
    std::swap(params.row, params.col);
    std::swap(params.input_row_stride, params.input_col_stride);
    std::swap(params.code_row_stride, params.code_col_stride);
    std::swap(params.scale_row_stride, params.scale_col_stride);
    std::swap(params.rows_per_block, params.cols_per_block);
    std::swap(params.row_block_count, params.col_block_count);
    params.contract_dim = -3 - params.contract_dim;
  }
  BOOL_DISPATCH(
      params.contract_dim == -1, kLastDim,
      BOOL_DISPATCH(
          transpose_output, kTransposeOutput,
          BOOL_DISPATCH(
              stochastic_rounding, kStochasticRounding,
              BOOL_DISPATCH(
                  params.statistics != nullptr, kCollectStats,
                  launch_quantize_mxfp8<
                      Mxfp8LaunchConfig<
                          input_t, block_shape, (kLastDim ? -1 : -2),
                          kTransposeOutput, kStochasticRounding, kCollectStats>>(
                      params, swizzled_scales, philox, stream)))))
}

template <typename input_t>
void dispatch_block_shape(
    QuantizeParams params,
    bool output_column_major,
    bool swizzled_scales,
    bool stochastic_rounding,
    at::PhiloxCudaState philox,
    cudaStream_t stream
) {
  const int64_t block_size = params.contract_dim == -1
      ? params.cols_per_block : params.rows_per_block;
  const bool square_block =
      params.rows_per_block != 1 && params.cols_per_block != 1;
#define LAUNCH_SIZE(SIZE)                                                  \
  BOOL_DISPATCH(                                                          \
      square_block, kSquare,                                              \
      dispatch_layout<input_t, BlockShape<(kSquare ? SIZE : 1), SIZE>>(      \
          params, output_column_major, swizzled_scales, stochastic_rounding, \
          philox, stream))
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
    const std::string& output_layout,
    const std::string& scale_layout
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
  const auto scale_options = input.options().dtype(at::kFloat8_e8m0fnu);
  if (scale_layout == "row_major") {
    return {codes, at::empty_symint(scale_sizes, scale_options)};
  }
  TORCH_CHECK(scale_layout == "swizzled_32_4_4",
              "quantize_mxfp8 scale_layout must be row_major or swizzled_32_4_4");
  TORCH_CHECK(input.dim() == 2 && block_size == 32 &&
                  (block_outer == 1 || block_outer == 32),
              "quantize_mxfp8 swizzled_32_4_4 scales require a 2D input, "
              "block_size=32, and block_outer=1 or 32");
  const c10::SymInt logical_rows = contract_dim == -1
      ? scale_sizes[0] : scale_sizes[1];
  const c10::SymInt logical_blocks = contract_dim == -1
      ? scale_sizes[1] : scale_sizes[0];
  const c10::SymInt padded_rows = (logical_rows + 127) / 128 * 128;
  const c10::SymInt padded_blocks = (logical_blocks + 3) / 4 * 4;
  return {codes, at::empty_symint(std::vector<c10::SymInt>{
                                      padded_rows * padded_blocks},
                                  scale_options)};
}

std::tuple<at::Tensor, at::Tensor> quantize_mxfp8_impl(
    const at::Tensor& input,
    int64_t contract_dim,
    int64_t block_outer,
    int64_t block_size,
    bool stochastic_rounding,
    const std::string& output_layout,
    const std::string& scale_layout,
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
  TORCH_CHECK(scale_layout == "row_major" || scale_layout == "swizzled_32_4_4",
              "quantize_mxfp8 scale_layout must be row_major or swizzled_32_4_4");
  const bool swizzled_scales = scale_layout == "swizzled_32_4_4";
  TORCH_CHECK(!swizzled_scales ||
                  (input.dim() == 2 && block_size == 32 &&
                   (block_outer == 1 || block_outer == 32)),
              "quantize_mxfp8 swizzled_32_4_4 scales require a 2D input, "
              "block_size=32, and block_outer=1 or 32");
  TORCH_CHECK(std::all_of(input.strides().begin(), input.strides().end(),
                          [](int64_t stride) { return stride >= 0; }),
              "quantize_mxfp8 requires nonnegative strides");
  c10::cuda::CUDAGuard device_guard(input.device());
  auto [codes, scales] =
      quantize_mxfp8_meta(input, contract_dim, block_outer, block_size,
                          stochastic_rounding, output_layout, scale_layout);
  if (stats != nullptr) *stats = at::zeros({5}, input.options().dtype(at::kFloat));
  const auto stream = at::cuda::getCurrentCUDAStream(input.get_device());
  if (input.numel() == 0) return {codes, scales};

  QuantizeParams params;
  params.input = input.const_data_ptr();
  params.codes = codes.data_ptr();
  params.statistics = stats != nullptr ? stats->data_ptr<float>() : nullptr;
  params.batch = input.dim() == 3 ? input.size(0) : 1;
  params.row = input.size(-2);
  params.col = input.size(-1);
  params.input_batch_stride = input.dim() == 3 ? input.stride(0) : 0;
  params.input_row_stride = input.stride(-2);
  params.input_col_stride = input.stride(-1);
  params.code_batch_stride = codes.dim() == 3 ? codes.stride(0) : 0;
  params.code_row_stride = codes.stride(-2);
  params.code_col_stride = codes.stride(-1);
  if (swizzled_scales) {
    const int64_t cols = (input.size(contract_dim) - 1) / block_size + 1;
    params.scale_row_stride = contract_dim == -1 ? cols : 1;
    params.scale_col_stride = contract_dim == -1 ? 1 : cols;
  } else {
    params.scale_batch_stride = scales.dim() == 3 ? scales.stride(0) : 0;
    params.scale_row_stride = scales.stride(-2);
    params.scale_col_stride = scales.stride(-1);
  }
  params.scale = scales.data_ptr();
  params.rows_per_block = contract_dim == -1 ? block_outer : block_size;
  params.cols_per_block = contract_dim == -1 ? block_size : block_outer;
  params.row_block_count = (params.row + params.rows_per_block - 1) /
      params.rows_per_block;
  params.col_block_count = (params.col + params.cols_per_block - 1) /
      params.cols_per_block;
  params.contract_dim = static_cast<int>(contract_dim);
  // Output strides are ambiguous for singleton dimensions, so preserve the
  // requested layout instead of inferring it from the output tensor.
  const bool output_column_major = output_layout == "column_major";
  at::PhiloxCudaState philox;
  if (stochastic_rounding) {
    auto* generator = at::get_generator_or_default<at::CUDAGeneratorImpl>(
        std::nullopt,
        at::cuda::detail::getDefaultCUDAGenerator(input.get_device()));
    std::lock_guard<std::mutex> lock(generator->mutex_);
    const int64_t draws = std::max(int64_t{16}, block_size * block_size / 256);
    philox = generator->philox_cuda_state(draws);
  }
#define LAUNCH_TYPE(TYPE)                                                 \
  dispatch_block_shape<TYPE>(                                             \
      params, output_column_major, swizzled_scales, stochastic_rounding,   \
      philox, stream.stream())
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
    const std::string& output_layout,
    const std::string& scale_layout
) {
  return quantize_mxfp8_impl(input, contract_dim, block_outer, block_size,
                              stochastic_rounding, output_layout, scale_layout,
                              nullptr);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> quantize_mxfp8_with_stats_cuda(
    const at::Tensor& input,
    int64_t contract_dim,
    int64_t block_outer,
    int64_t block_size,
    bool stochastic_rounding,
    const std::string& output_layout,
    const std::string& scale_layout
) {
  at::Tensor stats;
  auto [codes, scales] = quantize_mxfp8_impl(
      input, contract_dim, block_outer, block_size, stochastic_rounding,
      output_layout, scale_layout, &stats);
  return {codes, scales, stats};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> quantize_mxfp8_with_stats_meta(
    const at::Tensor& input,
    int64_t contract_dim,
    int64_t block_outer,
    int64_t block_size,
    bool stochastic_rounding,
    const std::string& output_layout,
    const std::string& scale_layout
) {
  auto [codes, scales] = quantize_mxfp8_meta(
      input, contract_dim, block_outer, block_size, stochastic_rounding,
      output_layout, scale_layout);
  return {codes, scales, at::empty_symint({5}, input.options().dtype(at::kFloat))};
}

}  // namespace

TORCH_LIBRARY_FRAGMENT(aot_kernel, m) {
  m.def(
      "quantize_mxfp8(Tensor x, int contract_dim, int block_outer, int "
      "block_size, "
      "bool stochastic_rounding, str output_layout=\"row_major\", str "
      "scale_layout=\"row_major\") -> (Tensor, Tensor)",
      {at::Tag::nondeterministic_seeded});
  m.def(
      "quantize_mxfp8_with_stats(Tensor x, int contract_dim, int block_outer, int "
      "block_size, bool stochastic_rounding, str output_layout=\"row_major\", "
      "str scale_layout=\"row_major\") -> (Tensor, Tensor, Tensor)",
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
