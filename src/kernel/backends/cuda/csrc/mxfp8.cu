#include <ATen/core/enum_tag.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <curand_kernel.h>
#include <torch/extension.h>

#include <ATen/cuda/PhiloxUtils.cuh>
#include <algorithm>
#include <cstdint>
#include <limits>
#include <mutex>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>

namespace {

struct Layout {
  int64_t batches, slow_size, fast_size;
  int64_t input_batch_stride, input_slow_stride, input_fast_stride;
  int64_t code_batch_stride, code_slow_stride, code_fast_stride;
  int64_t scale_batch_stride, scale_slow_stride, scale_fast_stride;
};

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

__device__ __forceinline__ uint8_t encode_e4m3_round_to_near(float value) {
  return __nv_cvt_float_to_fp8(value, __NV_SATFINITE, __NV_E4M3);
}

__device__ __forceinline__ uint8_t encode_e4m3_stochastic(
    float value,
    uint32_t random
) {
  if (!isnan(value)) {
    const int exponent = ((__float_as_int(value) >> 23) & 0xff) - 3;
    const float ulp = __int_as_float(max(exponent, 118) << 23);
    const float lower = floorf(value / ulp) * ulp;
    value = static_cast<float>(random) * 0x1.0p-32f < (value - lower) / ulp
                ? lower + ulp
                : lower;
  }
  return encode_e4m3_round_to_near(value);
}

__device__ __forceinline__ float decode_e4m3(uint8_t code) {
  const float magnitude = (code & 0x7f) < 8
      ? static_cast<float>(code & 7) * 0x1p-9f
      : ((code & 0x7f) == 0x7f
             ? nanf("")
             : ldexpf(1.0f + static_cast<float>(code & 7) * 0.125f,
                       static_cast<int>((code >> 3) & 0xf) - 7));
  return code & 0x80 ? -magnitude : magnitude;
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

template <typename scale_shape, bool contract_fast,
          bool transpose_output>
struct Tile {
  // Streaming groups use wide tiles; transposes need both tile axes populated.
  static constexpr int fast =
      scale_shape::outer != 1
          ? (contract_fast ? scale_shape::inner : scale_shape::outer)
          : (contract_fast ? (transpose_output ? scale_shape::inner : 512) : 32);
  static constexpr int slow =
      scale_shape::outer != 1
          ? (contract_fast ? scale_shape::outer : scale_shape::inner)
          : (contract_fast ? (transpose_output ? 32 : 2) : scale_shape::inner);
  static constexpr bool vector = contract_fast && scale_shape::outer == 1;
  static constexpr bool stream = vector && !transpose_output;
  static constexpr int threads = vector ? fast * slow / 16 : 256;
  static constexpr int values = fast * slow / threads;
};

template <typename input_t, typename scale_shape, int contract_dim,
          bool input_column_major,
          bool transpose_output, bool stochastic_rounding, bool collect_stats>
__global__ void quantize_mxfp8_kernel(
    const input_t* input,
    uint8_t* codes,
    uint8_t* scales,
    Layout layout,
    int64_t fast_tiles,
    int64_t slow_tiles,
    at::PhiloxCudaState philox,
    float* stats_partials
) {
  constexpr bool contract_fast =
      contract_dim == (input_column_major ? -2 : -1);
  using tile = Tile<scale_shape, contract_fast, transpose_output>;
  const int64_t tile_index = static_cast<int64_t>(blockIdx.x) +
                             static_cast<int64_t>(blockIdx.y) * gridDim.x;
  if (tile_index >= layout.batches * slow_tiles * fast_tiles) return;
  int64_t fast_start, slow_start, batch;
  if constexpr (tile::stream) {
    // Pack complete scale groups across rows, including narrow matrices.
    const int64_t padded_fast =
        (layout.fast_size + scale_shape::inner - 1) / scale_shape::inner *
        scale_shape::inner;
    const int64_t first = (tile_index % fast_tiles) * tile::fast * tile::slow +
                          threadIdx.x * tile::values;
    slow_start = first / padded_fast;
    fast_start = first % padded_fast;
    batch = tile_index / fast_tiles;
  } else {
    fast_start = (tile_index % fast_tiles) * tile::fast;
    const int64_t remaining = tile_index / fast_tiles;
    slow_start = (remaining % slow_tiles) * tile::slow;
    batch = remaining / slow_tiles;
  }

  const int lane = threadIdx.x % 32;
  const int warp = threadIdx.x / 32;
  float values[tile::values] = {};
  float source_values[tile::values];

  if constexpr (tile::vector) {
    const int first = tile::stream ? 0 : threadIdx.x * tile::values;
    const int64_t slow = slow_start + first / tile::fast;
    const int64_t fast = fast_start + first % tile::fast;
    const input_t* source = input + batch * layout.input_batch_stride +
                            slow * layout.input_slow_stride +
                            fast * layout.input_fast_stride;
    if (slow < layout.slow_size && fast + tile::values <= layout.fast_size &&
        layout.input_fast_stride == 1 &&
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
        if (slow < layout.slow_size && fast + index < layout.fast_size) {
          values[index] =
              static_cast<float>(source[index * layout.input_fast_stride]);
        }
      }
    }
  } else {
#pragma unroll
    for (int index = 0; index < tile::values; ++index) {
      const int position = threadIdx.x + index * tile::threads;
      const int64_t slow = slow_start + position / tile::fast;
      const int64_t fast = fast_start + position % tile::fast;
      if (slow < layout.slow_size && fast < layout.fast_size) {
        values[index] =
            static_cast<float>(input[batch * layout.input_batch_stride +
                                     slow * layout.input_slow_stride +
                                     fast * layout.input_fast_stride]);
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
    const int64_t slow = slow_start + first / tile::fast;
    const int64_t fast = fast_start + first % tile::fast;
    if (group_lane == 0 && slow < layout.slow_size && fast < layout.fast_size) {
      scales[batch * layout.scale_batch_stride +
             slow * layout.scale_slow_stride +
             (fast / scale_shape::inner) * layout.scale_fast_stride] = scale;
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
      if (fast_start + lane < layout.fast_size) {
        scales[batch * layout.scale_batch_stride +
               (slow_start / scale_shape::inner) * layout.scale_slow_stride +
               (fast_start + lane) * layout.scale_fast_stride] =
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
      const int64_t slow = slow_start + (contract_fast ? threadIdx.x : 0);
      const int64_t fast = fast_start + (contract_fast ? 0 : threadIdx.x);
      if (slow < layout.slow_size && fast < layout.fast_size) {
        scales[batch * layout.scale_batch_stride +
               (contract_fast ? slow : slow / scale_shape::inner) *
                   layout.scale_slow_stride +
               (contract_fast ? fast / scale_shape::inner : fast) *
                   layout.scale_fast_stride] = scale;
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
            encode_e4m3_stochastic(values[index + element], words[element]);
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
    encoded[0] = encode_e4m3_round_to_near(values[0]);
  }

  if constexpr (transpose_output) {
    // Four-byte padding avoids bank conflicts when reading the other tile axis.
    __shared__ uint8_t transposed[tile::slow][tile::fast + 4];
#pragma unroll
    for (int index = 0; index < tile::values; ++index) {
      const int position = tile::vector ? threadIdx.x * tile::values + index
                                        : threadIdx.x + index * tile::threads;
      transposed[position / tile::fast][position % tile::fast] = encoded[index];
    }
    __syncthreads();
#pragma unroll
    for (int index = 0; index < tile::values; ++index) {
      const int position = threadIdx.x + index * tile::threads;
      const int local_slow = position % tile::slow;
      const int local_fast = position / tile::slow;
      const int64_t slow = slow_start + local_slow;
      const int64_t fast = fast_start + local_fast;
      if (slow < layout.slow_size && fast < layout.fast_size) {
        codes[batch * layout.code_batch_stride +
              slow * layout.code_slow_stride + fast * layout.code_fast_stride] =
            transposed[local_slow][local_fast];
      }
    }
  } else if constexpr (tile::vector) {
    const int first = tile::stream ? 0 : threadIdx.x * tile::values;
    const int64_t slow = slow_start + first / tile::fast;
    const int64_t fast = fast_start + first % tile::fast;
    uint8_t* destination = codes + batch * layout.code_batch_stride +
                           slow * layout.code_slow_stride +
                           fast * layout.code_fast_stride;
    if (slow < layout.slow_size && fast + tile::values <= layout.fast_size &&
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
        if (slow < layout.slow_size && fast + index < layout.fast_size) {
          destination[index * layout.code_fast_stride] = encoded[index];
        }
      }
    }
  } else {
#pragma unroll
    for (int index = 0; index < tile::values; ++index) {
      const int position = threadIdx.x + index * tile::threads;
      const int64_t slow = slow_start + position / tile::fast;
      const int64_t fast = fast_start + position % tile::fast;
      if (slow < layout.slow_size && fast < layout.fast_size) {
        codes[batch * layout.code_batch_stride +
              slow * layout.code_slow_stride + fast * layout.code_fast_stride] =
            encoded[index];
      }
    }
  }

  if constexpr (collect_stats) {
    float src_sq = 0.0f, err_sq = 0.0f, under = 0.0f, nonzero = 0.0f;
#pragma unroll
    for (int index = 0; index < tile::values; ++index) {
      int64_t slow, fast;
      if constexpr (tile::vector) {
        const int first = tile::stream ? 0 : threadIdx.x * tile::values;
        slow = slow_start + first / tile::fast;
        fast = fast_start + first % tile::fast + index;
      } else {
        const int position = threadIdx.x + index * tile::threads;
        slow = slow_start + position / tile::fast;
        fast = fast_start + position % tile::fast;
      }
      if (slow < layout.slow_size && fast < layout.fast_size) {
        const float source = source_values[index];
        const float reconstructed = decode_e4m3(encoded[index]) /
                                    decode_e8m0(scale);
        src_sq += source * source;
        const float error = source - reconstructed;
        err_sq += error * error;
        const bool source_nonzero = source != 0.0f;
        under += source_nonzero && (encoded[index] & 0x7f) == 0;
        nonzero += source_nonzero;
      }
    }
    __shared__ float stats_warp_sums[4][8];
    float stat_values[4] = {src_sq, err_sq, under, nonzero};
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
    const at::Tensor& codes,
    const at::Tensor& scales,
    Layout layout,
    at::PhiloxCudaState philox,
    cudaStream_t stream,
    at::Tensor* stats
) {
  constexpr bool contract_fast =
      contract_dim == (input_column_major ? -2 : -1);
  using tile = Tile<scale_shape, contract_fast, transpose_output>;
  const int64_t padded_fast =
      (layout.fast_size + scale_shape::inner - 1) / scale_shape::inner *
      scale_shape::inner;
  const int64_t fast_tiles =
      tile::stream
          ? (padded_fast * layout.slow_size + tile::fast * tile::slow - 1) /
                (tile::fast * tile::slow)
          : (layout.fast_size + tile::fast - 1) / tile::fast;
  const int64_t slow_tiles =
      tile::stream ? 1 : (layout.slow_size + tile::slow - 1) / tile::slow;
  const int64_t count = layout.batches * slow_tiles * fast_tiles;
  const int64_t grid_x = std::min(
      count, static_cast<int64_t>(std::numeric_limits<int32_t>::max()));
  const int64_t grid_y = (count + grid_x - 1) / grid_x;
  TORCH_CHECK(grid_y <= 65535, "quantize_mxfp8 tensor is too large to launch");
  at::Tensor partials;
  float* stats_partials = nullptr;
  if constexpr (collect_stats) {
    partials = at::empty({count, 4}, input.options().dtype(at::kFloat));
    stats_partials = partials.data_ptr<float>();
  }
  quantize_mxfp8_kernel<input_t, scale_shape, contract_dim, input_column_major,
                        transpose_output, stochastic_rounding, collect_stats>
      <<<dim3(grid_x, grid_y), tile::threads, 0, stream>>>(
          input.const_data_ptr<input_t>(),
          reinterpret_cast<uint8_t*>(codes.data_ptr()),
          reinterpret_cast<uint8_t*>(scales.data_ptr()), layout, fast_tiles,
          slow_tiles, philox, stats_partials);
  if constexpr (collect_stats) {
    finalize_stats_kernel<<<1, 256, 0, stream>>>(
        stats_partials, count, input.numel(), stats->data_ptr<float>());
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
    const at::Tensor& codes,
    const at::Tensor& scales,
    Layout layout,
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
                          input, codes, scales, layout, philox, stream,
                          stats))))))
}

template <typename input_t>
void dispatch_scale_shape(
    const at::Tensor& input,
    const at::Tensor& codes,
    const at::Tensor& scales,
    Layout layout,
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
          input, codes, scales, layout, contract_dim, input_column_major,  \
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
  const int slow_axis = input_column_major ? -1 : -2;
  const int fast_axis = input_column_major ? -2 : -1;
  const Layout layout = {input.dim() == 3 ? input.size(0) : 1,
                         input.size(slow_axis),
                         input.size(fast_axis),
                         input.dim() == 3 ? input.stride(0) : 0,
                         input.stride(slow_axis),
                         input.stride(fast_axis),
                         codes.dim() == 3 ? codes.stride(0) : 0,
                         codes.stride(slow_axis),
                         codes.stride(fast_axis),
                         scales.dim() == 3 ? scales.stride(0) : 0,
                         scales.stride(slow_axis),
                         scales.stride(fast_axis)};
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
  dispatch_scale_shape<TYPE>(input, codes, scales, layout, block_size,         \
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
