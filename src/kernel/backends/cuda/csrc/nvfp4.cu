#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <ATen/cuda/PhiloxUtils.cuh>
#include <c10/cuda/CUDAGuard.h>
#include <curand_kernel.h>
#include <torch/library.h>

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <cstdint>
#include <mutex>
#include <tuple>
#include <vector>

namespace {

constexpr int kContractBlock = 16;
constexpr float kFp8Min = 0x1p-9f;
constexpr float kFp8Max = 448.0f;

struct MatrixLayout {
  int64_t batch;
  int64_t rows;
  int64_t cols;
  int64_t stride_batch;
  int64_t stride_row;
  int64_t stride_col;
  int64_t code_stride_batch;
  int64_t code_stride_row;
  int64_t code_stride_col;
};

template <int kQmax>
__device__ __forceinline__ float sanitize_magnitude(float value) {
  if (isnan(value) || isinf(value)) {
    return static_cast<float>(kQmax);
  }
  return fminf(fabsf(value), static_cast<float>(kQmax));
}

// Positive encodings are ordered; round the discarded mantissa bits upward.
__device__ __forceinline__ uint8_t encode_e4m3_ceil(float value) {
  if (isnan(value)) {
    return 0x7f;
  }
  value = fminf(fmaxf(value, kFp8Min), kFp8Max);
  if (value <= 0x1p-6f) {
    return static_cast<uint8_t>(ceilf(value * 512.0f));
  }
  return static_cast<uint8_t>(((__float_as_uint(value) + 0xfffff) >> 20) - 960);
}

__device__ __forceinline__ float decode_e4m3(uint8_t code) {
  if (code < 8) {
    return static_cast<float>(code) * 0x1p-9f;
  }
  if (code == 127) {
    return nanf("");
  }
  return __uint_as_float((static_cast<unsigned>(code) + 960) << 20);
}

__device__ __forceinline__ float decode_e2m1(uint8_t code) {
  constexpr float values[] = {0.0f, 0.5f, 1.0f, 1.5f,
                              2.0f, 3.0f, 4.0f, 6.0f};
  float value = values[code & 7];
  return code & 8 ? -value : value;
}

template <int kQmax>
__device__ __forceinline__ uint8_t encode_e2m1_rne(float value, float scale) {
  if (isnan(value) || isnan(scale)) {
    return 7;
  }
  // E4M3 scales times E2M1 midpoints are exact FP32 values.
  float magnitude = fabsf(value);
  if constexpr (kQmax == 4) {
    magnitude = fminf(magnitude, 4.0f * scale);
  }
  uint8_t code = static_cast<uint8_t>(magnitude > 0.25f * scale) +
      static_cast<uint8_t>(magnitude >= 0.75f * scale) +
      static_cast<uint8_t>(magnitude > 1.25f * scale) +
      static_cast<uint8_t>(magnitude >= 1.75f * scale) +
      static_cast<uint8_t>(magnitude > 2.5f * scale) +
      static_cast<uint8_t>(magnitude >= 3.5f * scale) +
      static_cast<uint8_t>(magnitude > 5.0f * scale);
  if (signbit(value)) {
    code |= 8;
  }
  return code;
}

template <int kQmax>
__device__ __forceinline__ uint8_t encode_e2m1_sr(
    float value,
    unsigned random_bits) {
  float magnitude = sanitize_magnitude<kQmax>(value);
  float step = magnitude >= 4.0f ? 2.0f : magnitude >= 2.0f ? 1.0f : 0.5f;
  float lower = floorf(magnitude / step) * step;
  float probability = (magnitude - lower) / step;
  float random = static_cast<float>(random_bits >> 8) * 0x1p-24f;
  float rounded = lower + (random < probability ? step : 0.0f);
  int code = rounded >= 6.0f ? 7 : rounded >= 4.0f ? 6 :
      rounded >= 3.0f ? 5 : rounded >= 2.0f ? 4 :
      rounded >= 1.5f ? 3 : rounded >= 1.0f ? 2 :
      rounded >= 0.5f ? 1 : 0;
  if (signbit(value) && !isnan(value)) {
    code |= 8;
  }
  return static_cast<uint8_t>(code);
}

__device__ __forceinline__ unsigned block_max(unsigned maximum) {
  __shared__ unsigned warp_maxima[8];
  int lane = threadIdx.x & 31;
  int warp = threadIdx.x >> 5;
  for (int offset = 16; offset > 0; offset >>= 1) {
    maximum = max(maximum, __shfl_down_sync(0xffffffff, maximum, offset));
  }
  if (lane == 0) {
    warp_maxima[warp] = maximum;
  }
  __syncthreads();
  maximum = lane < blockDim.x / 32 ? warp_maxima[lane] : 0;
  for (int offset = 16; offset > 0; offset >>= 1) {
    maximum = max(maximum, __shfl_down_sync(0xffffffff, maximum, offset));
  }
  return __shfl_sync(0xffffffff, maximum, 0);
}

template <typename scalar_t, int kQmax>
__global__ void global_amax_kernel(
    const scalar_t* input,
    MatrixLayout layout,
    float* partials,
    int partial_count) {
  constexpr int kChunk = 2048;
  int64_t matrix_size = layout.rows * layout.cols;
  int64_t batch = blockIdx.y;
  bool contiguous = (layout.stride_col == 1 && layout.stride_row == layout.cols) ||
      (layout.stride_row == 1 && layout.stride_col == layout.rows);
  unsigned maximum = 0;
  for (int offset = threadIdx.x; offset < kChunk; offset += blockDim.x) {
    int64_t linear = static_cast<int64_t>(blockIdx.x) * kChunk + offset;
    if (linear < matrix_size) {
      int64_t index = linear;
      if (!contiguous) {
        int64_t row = linear / layout.cols;
        index = row * layout.stride_row + (linear - row * layout.cols) * layout.stride_col;
      }
      float value = static_cast<float>(input[batch * layout.stride_batch + index]);
      maximum = max(maximum, isnan(value) ? 0x7fffffffu : __float_as_uint(fabsf(value)));
    }
  }
  maximum = block_max(maximum);
  if (threadIdx.x == 0) {
    float value = __uint_as_float(maximum);
    partials[batch * partial_count + blockIdx.x] = partial_count == 1 && !isnan(value)
        ? fmaxf(value * (1.0f / (kQmax * kFp8Max)), 1.0e-30f) : value;
  }
}

template <int kQmax>
__global__ void global_scale_kernel(const float* partials, float* global_scale, int count) {
  unsigned maximum = 0;
  for (int index = threadIdx.x; index < count; index += blockDim.x) {
    maximum = max(maximum, __float_as_uint(partials[static_cast<int64_t>(blockIdx.x) * count + index]));
  }
  maximum = block_max(maximum);
  if (threadIdx.x == 0) {
    float value = __uint_as_float(maximum);
    global_scale[blockIdx.x] = isnan(value) ? value :
        fmaxf(value * (1.0f / (kQmax * kFp8Max)), 1.0e-30f);
  }
}

template <typename scalar_t, int kQmax, int kOuterBlock, bool kContractLast,
          bool kStochastic, bool kCollectStats>
__global__ void quantize_kernel(
    const scalar_t* input,
    MatrixLayout layout,
    const float* global_scale,
    uint8_t* codes,
    uint8_t* scales,
    at::PhiloxCudaState philox_args,
    float* stats_partials) {
  constexpr int kValues = kOuterBlock == 1 ? 16 : 8;
  constexpr int kTileThreads = kOuterBlock == 1 ? 1 : 32;
  constexpr int kTilesPerCta = 128 / kTileThreads;
  int contract_groups = (kContractLast ? layout.cols : layout.rows) / 16;
  int outer_size = kContractLast ? layout.rows : layout.cols;
  int outer_groups = (outer_size + kOuterBlock - 1) / kOuterBlock;
  int tile = blockIdx.x * kTilesPerCta + threadIdx.x / kTileThreads;
  const bool tile_valid =
      tile < static_cast<int64_t>(outer_groups) * contract_groups;
  if (!tile_valid && !kCollectStats) return;
  int64_t batch = blockIdx.y;
  bool contract_contiguous = kContractLast
      ? layout.stride_col <= layout.stride_row : layout.stride_row <= layout.stride_col;
  int outer_group = contract_contiguous ? tile / contract_groups : tile % outer_groups;
  int contract_group = contract_contiguous ? tile % contract_groups : tile / outer_groups;
  int lane = threadIdx.x % kTileThreads;
  int outer = outer_group * kOuterBlock;
  int contraction = contract_group * 16;
  if constexpr (kOuterBlock == 16) {
    outer += contract_contiguous ? lane / 8 : lane % 16;
    contraction += contract_contiguous ? (lane % 8) * 2 : (lane / 16) * 8;
  }
  int64_t row = kContractLast ? outer : contraction;
  int64_t col = kContractLast ? contraction : outer;
  int64_t stride = kContractLast ? layout.stride_col : layout.stride_row;
  bool valid = tile_valid && outer < outer_size;
  float values[kValues];
  float source_values[kValues];
  unsigned maximum = 0;
  float global = global_scale == nullptr ? 1.0f : global_scale[batch];
  #pragma unroll
  for (int index = 0; index < kValues; ++index) {
    int outer_offset = kOuterBlock == 16 && contract_contiguous ? (index / 2) * 4 : 0;
    int contract_offset = kOuterBlock == 16 && contract_contiguous ? index % 2 : index;
    int64_t outer_stride = kContractLast ? layout.stride_row : layout.stride_col;
    bool value_valid = tile_valid && outer + outer_offset < outer_size;
    float value = value_valid ? static_cast<float>(input[batch * layout.stride_batch +
        row * layout.stride_row + col * layout.stride_col +
        outer_offset * outer_stride + contract_offset * stride]) : 0.0f;
    source_values[index] = value;
    if (global_scale != nullptr && value_valid) {
      value /= global;
    }
    values[index] = value;
    maximum = max(maximum, isnan(value) ? 0x7fffffffu : __float_as_uint(fabsf(value)));
  }
  if constexpr (kOuterBlock == 16) {
    for (int offset = 16; offset > 0; offset >>= 1) {
      maximum = max(maximum, __shfl_down_sync(0xffffffff, maximum, offset));
    }
    maximum = __shfl_sync(0xffffffff, maximum, 0);
  }
  uint8_t scale_code = encode_e4m3_ceil(
      __uint_as_float(maximum) * (1.0f / static_cast<float>(kQmax)));
  float scale = decode_e4m3(scale_code);
  int scale_outer = outer_group * kOuterBlock + lane;
  if (tile_valid && lane < kOuterBlock && scale_outer < outer_size) {
    int64_t scale_index = kContractLast
        ? (batch * layout.rows + scale_outer) * contract_groups + contract_group
        : (batch * contract_groups + contract_group) * layout.cols + scale_outer;
    scales[scale_index] = scale_code;
  }
  if (!valid && !kCollectStats) return;
  int64_t code_index = kContractLast
      ? batch * layout.code_stride_batch + row * layout.code_stride_row +
            (col / 2) * layout.code_stride_col
      : batch * layout.code_stride_batch + (row / 2) * layout.code_stride_row +
            col * layout.code_stride_col;
  curandStatePhilox4_32_10_t rng;
  float inverse_scale;
  if constexpr (kStochastic) {
    auto seeds = at::cuda::philox::unpack(philox_args);
    curand_init(std::get<0>(seeds), code_index, std::get<1>(seeds), &rng);
    inverse_scale = __frcp_rn(scale);
  }
  uint4 random;
  uint8_t encoded[kValues] = {};
  if (valid) {
  #pragma unroll
  for (int pair = 0; pair < kValues / 2; ++pair) {
    uint8_t low;
    uint8_t high;
    if constexpr (kStochastic) {
      if (pair % 2 == 0) {
        random = curand4(&rng);
      }
      low = encode_e2m1_sr<kQmax>(values[2 * pair] * inverse_scale, pair % 2 == 0 ? random.x : random.z);
      high = encode_e2m1_sr<kQmax>(values[2 * pair + 1] * inverse_scale, pair % 2 == 0 ? random.y : random.w);
    } else {
      low = encode_e2m1_rne<kQmax>(values[2 * pair], scale);
      high = encode_e2m1_rne<kQmax>(values[2 * pair + 1], scale);
    }
    encoded[2 * pair] = low;
    encoded[2 * pair + 1] = high;
    int outer_offset = kOuterBlock == 16 && contract_contiguous ? pair * 4 : 0;
    int contract_offset = kOuterBlock == 16 && contract_contiguous ? 0 : pair;
    if (outer + outer_offset < outer_size) {
      const int64_t code_row = kContractLast ? row + outer_offset
                                              : row + contract_offset * 2;
      const int64_t code_col = kContractLast ? col + contract_offset * 2
                                              : col + outer_offset;
      codes[batch * layout.code_stride_batch +
            (kContractLast ? code_row : code_row / 2) * layout.code_stride_row +
            (kContractLast ? code_col / 2 : code_col) * layout.code_stride_col] =
          low | (high << 4);
    }
  }
  }

  if constexpr (kCollectStats) {
    float src_sq = 0.0f, err_sq = 0.0f, under = 0.0f, nonzero = 0.0f;
#pragma unroll
    for (int index = 0; index < kValues; ++index) {
      const int outer_offset = kOuterBlock == 16 && contract_contiguous
          ? (index / 2) * 4 : 0;
      const int contract_offset = kOuterBlock == 16 && contract_contiguous
          ? index % 2 : index;
      if (outer + outer_offset < outer_size) {
        const float source = source_values[index];
        float reconstructed = decode_e2m1(encoded[index]) * scale;
        if (global_scale != nullptr) reconstructed *= global;
        src_sq += source * source;
        const float error = source - reconstructed;
        err_sq += error * error;
        const bool source_nonzero = source != 0.0f;
        under += source_nonzero && (encoded[index] & 7) == 0;
        nonzero += source_nonzero;
      }
    }
    __shared__ float stats_warp_sums[4][4];
    const int stats_lane = threadIdx.x & 31;
    const int stats_warp = threadIdx.x >> 5;
    float stat_values[4] = {src_sq, err_sq, under, nonzero};
#pragma unroll
    for (int value = 0; value < 4; ++value) {
#pragma unroll
      for (int offset = 16; offset > 0; offset >>= 1) {
        stat_values[value] += __shfl_down_sync(0xffffffffu, stat_values[value], offset);
      }
      if (stats_lane == 0) stats_warp_sums[value][stats_warp] = stat_values[value];
    }
    __syncthreads();
    if (stats_warp == 0) {
#pragma unroll
      for (int value = 0; value < 4; ++value) {
        float sum = stats_lane < 4 ? stats_warp_sums[value][stats_lane] : 0.0f;
        sum += __shfl_down_sync(0xffffffffu, sum, 16);
        sum += __shfl_down_sync(0xffffffffu, sum, 8);
        sum += __shfl_down_sync(0xffffffffu, sum, 4);
        sum += __shfl_down_sync(0xffffffffu, sum, 2);
        sum += __shfl_down_sync(0xffffffffu, sum, 1);
        if (stats_lane == 0) {
          const int64_t partial_index = static_cast<int64_t>(blockIdx.y) * gridDim.x + blockIdx.x;
          stats_partials[partial_index * 4 + value] = sum;
        }
      }
    }
  }
}

__global__ void finalize_nvfp4_stats_kernel(const float* partials, int64_t count,
                                            int64_t numel, float* stats) {
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

void check_arguments(
    const at::Tensor& input,
    int64_t contract_dim,
    at::IntArrayRef block_shape,
    double qmax,
    const std::string& output_layout) {
  TORCH_CHECK(input.is_cuda() || input.is_meta(),
              "quantize_nvfp4 expects a CUDA tensor");
  TORCH_CHECK(input.dim() == 2 || input.dim() == 3,
              "quantize_nvfp4 expects a rank-2 or rank-3 tensor");
  TORCH_CHECK(contract_dim == -1 || contract_dim == -2,
              "contract_dim must be -2 or -1");
  TORCH_CHECK(block_shape.size() == 2 &&
                  (block_shape[0] == 1 || block_shape[0] == 16) &&
                  block_shape[1] == 16,
              "block_shape must be (1, 16) or (16, 16)");
  TORCH_CHECK(input.scalar_type() == at::kFloat ||
                  input.scalar_type() == at::kHalf ||
                  input.scalar_type() == at::kBFloat16,
              "quantize_nvfp4 expects float32, float16, or bfloat16 input");
  TORCH_CHECK(input.sym_size(contract_dim) % kContractBlock == 0,
              "the contraction dimension must be a multiple of 16");
  TORCH_CHECK(input.sym_numel() > 0,
              "quantize_nvfp4 does not support empty matrices");
  TORCH_CHECK(qmax == 4.0 || qmax == 6.0,
              "quantize_nvfp4 qmax must be 4 or 6");
  TORCH_CHECK(output_layout == "row_major" || output_layout == "column_major",
              "quantize_nvfp4 output_layout must be row_major or column_major");
  for (auto stride : input.sym_strides()) {
    TORCH_CHECK(stride >= 0, "quantize_nvfp4 expects non-negative input strides");
  }
}

std::vector<c10::SymInt> output_shape(const at::Tensor& input, int64_t contract_dim) {
  auto shape = input.sym_sizes().vec();
  shape[shape.size() + contract_dim] /= 2;
  return shape;
}

std::vector<c10::SymInt> scale_shape(const at::Tensor& input, int64_t contract_dim) {
  auto shape = input.sym_sizes().vec();
  shape[shape.size() + contract_dim] /= kContractBlock;
  return shape;
}

at::Tensor allocate_codes(const at::Tensor& input, int64_t contract_dim,
                          const std::string& output_layout) {
  const auto shape = output_shape(input, contract_dim);
  const auto options = input.options().dtype(at::kByte);
  if (output_layout == "row_major") return at::empty_symint(shape, options);
  const auto rows = shape[shape.size() - 2];
  const auto cols = shape[shape.size() - 1];
  const std::vector<c10::SymInt> strides = input.dim() == 3
      ? std::vector<c10::SymInt>{rows * cols, 1, rows}
      : std::vector<c10::SymInt>{1, rows};
  return at::empty_strided_symint(shape, strides, options);
}

template <typename scalar_t, int kQmax, bool kCollectStats>
void launch_quantize_impl(
    const at::Tensor& input,
    const MatrixLayout& layout,
    const at::Tensor& codes,
    const at::Tensor& scales,
    const c10::optional<at::Tensor>& global_scale,
    int64_t contract_dim,
    int64_t outer_block,
    bool stochastic_rounding,
    at::PhiloxCudaState philox_args,
    cudaStream_t stream,
    float* stats_partials) {
  int64_t contract_groups =
      (contract_dim == -1 ? layout.cols : layout.rows) / kContractBlock;
  int64_t outer_size = contract_dim == -1 ? layout.rows : layout.cols;
  int64_t outer_groups = (outer_size + outer_block - 1) / outer_block;
  int64_t tiles = outer_groups * contract_groups;
  int tiles_per_cta = outer_block == 1 ? 128 : 4;
  dim3 grid((tiles + tiles_per_cta - 1) / tiles_per_cta, layout.batch);
  const float* global = global_scale ? global_scale->data_ptr<float>() : nullptr;
  const auto* input_ptr = input.data_ptr<scalar_t>();
  auto* code_ptr = codes.data_ptr<uint8_t>();
  auto* scale_ptr = reinterpret_cast<uint8_t*>(scales.data_ptr());
#define LAUNCH_KERNEL(OUTER, CONTRACT, STOCHASTIC)                             \
  quantize_kernel<scalar_t, kQmax, OUTER, CONTRACT, STOCHASTIC, kCollectStats><<< \
      grid, 128, 0, stream>>>(input_ptr, layout, global, code_ptr, scale_ptr, \
                               philox_args, stats_partials)
  if (contract_dim == -1) {
    if (outer_block == 1) {
      if (stochastic_rounding) {
        LAUNCH_KERNEL(1, true, true);
      } else {
        LAUNCH_KERNEL(1, true, false);
      }
    } else if (stochastic_rounding) {
      LAUNCH_KERNEL(16, true, true);
    } else {
      LAUNCH_KERNEL(16, true, false);
    }
  } else if (outer_block == 1) {
    if (stochastic_rounding) {
      LAUNCH_KERNEL(1, false, true);
    } else {
      LAUNCH_KERNEL(1, false, false);
    }
  } else if (stochastic_rounding) {
    LAUNCH_KERNEL(16, false, true);
  } else {
    LAUNCH_KERNEL(16, false, false);
  }
#undef LAUNCH_KERNEL
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename scalar_t, int kQmax>
void launch_quantize(
    const at::Tensor& input,
    const MatrixLayout& layout,
    const at::Tensor& codes,
    const at::Tensor& scales,
    const c10::optional<at::Tensor>& global_scale,
    int64_t contract_dim,
    int64_t outer_block,
    bool stochastic_rounding,
    at::PhiloxCudaState philox_args,
    cudaStream_t stream,
    float* stats_partials) {
  if (stats_partials != nullptr) {
    launch_quantize_impl<scalar_t, kQmax, true>(
        input, layout, codes, scales, global_scale, contract_dim, outer_block,
        stochastic_rounding, philox_args, stream, stats_partials);
  } else {
    launch_quantize_impl<scalar_t, kQmax, false>(
        input, layout, codes, scales, global_scale, contract_dim, outer_block,
        stochastic_rounding, philox_args, stream, nullptr);
  }
}

std::tuple<at::Tensor, at::Tensor, c10::optional<at::Tensor>> quantize_nvfp4_impl(
    const at::Tensor& input,
    int64_t contract_dim,
    at::IntArrayRef block_shape,
    bool enable_global_scale,
    bool stochastic_rounding,
    double qmax,
    const std::string& output_layout,
    at::Tensor* stats) {
  check_arguments(input, contract_dim, block_shape, qmax, output_layout);
  c10::cuda::CUDAGuard device_guard(input.device());
  MatrixLayout layout{
      input.dim() == 3 ? input.size(0) : 1, input.size(-2), input.size(-1),
      input.dim() == 3 ? input.stride(0) : 0, input.stride(-2), input.stride(-1),
      0, 0, 0};
  at::Tensor codes = allocate_codes(input, contract_dim, output_layout);
  layout.code_stride_batch = codes.dim() == 3 ? codes.stride(0) : 0;
  layout.code_stride_row = codes.stride(-2);
  layout.code_stride_col = codes.stride(-1);
  at::Tensor scales = at::empty_symint(scale_shape(input, contract_dim),
                                 input.options().dtype(at::kFloat8_e4m3fn));
  c10::optional<at::Tensor> global_scale;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(input.get_device());
  if (enable_global_scale) {
    global_scale = at::empty({layout.batch}, input.options().dtype(at::kFloat));
    int blocks = static_cast<int>((layout.rows * layout.cols + 2047) / 2048);
    auto partials = blocks == 1 ? *global_scale :
        at::empty({layout.batch, blocks}, input.options().dtype(at::kFloat));
    if (qmax == 4.0) {
      AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, input.scalar_type(),
                                      "quantize_nvfp4_global_amax", [&] {
        global_amax_kernel<scalar_t, 4><<<dim3(blocks, layout.batch), 256, 0, stream>>>(
            input.data_ptr<scalar_t>(), layout, partials.data_ptr<float>(), blocks);
      });
    } else {
      AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, input.scalar_type(),
                                      "quantize_nvfp4_global_amax", [&] {
        global_amax_kernel<scalar_t, 6><<<dim3(blocks, layout.batch), 256, 0, stream>>>(
            input.data_ptr<scalar_t>(), layout, partials.data_ptr<float>(), blocks);
      });
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    if (blocks > 1) {
      if (qmax == 4.0) {
        global_scale_kernel<4><<<layout.batch, 256, 0, stream>>>(
            partials.data_ptr<float>(), global_scale->data_ptr<float>(), blocks);
      } else {
        global_scale_kernel<6><<<layout.batch, 256, 0, stream>>>(
            partials.data_ptr<float>(), global_scale->data_ptr<float>(), blocks);
      }
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
  }

  at::PhiloxCudaState philox_args;
  if (stochastic_rounding) {
    // Each thread owns a Philox subsequence and consumes at most 16 values.
    auto generator = at::cuda::detail::getDefaultCUDAGenerator(input.get_device());
    std::lock_guard<std::mutex> lock(generator.mutex());
    philox_args = static_cast<at::CUDAGeneratorImpl*>(generator.unsafeGetGeneratorImpl())
                      ->philox_cuda_state(16);
  }
  at::Tensor partials;
  float* stats_partials = nullptr;
  if (stats != nullptr) {
    *stats = at::zeros({5}, input.options().dtype(at::kFloat));
    const int64_t contract_groups =
        (contract_dim == -1 ? layout.cols : layout.rows) / kContractBlock;
    const int64_t outer_size = contract_dim == -1 ? layout.rows : layout.cols;
    const int64_t outer_groups = (outer_size + block_shape[0] - 1) / block_shape[0];
    const int64_t tiles = outer_groups * contract_groups;
    const int64_t tiles_per_cta = block_shape[0] == 1 ? 128 : 4;
    const int64_t blocks = (tiles + tiles_per_cta - 1) / tiles_per_cta * layout.batch;
    partials = at::empty({blocks, 4}, input.options().dtype(at::kFloat));
    stats_partials = partials.data_ptr<float>();
  }
  if (qmax == 4.0) {
    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, input.scalar_type(),
                                    "quantize_nvfp4", [&] {
      launch_quantize<scalar_t, 4>(input, layout, codes, scales, global_scale,
                                   contract_dim, block_shape[0], stochastic_rounding,
                                   philox_args, stream, stats_partials);
    });
  } else {
    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, input.scalar_type(),
                                    "quantize_nvfp4", [&] {
      launch_quantize<scalar_t, 6>(input, layout, codes, scales, global_scale,
                                   contract_dim, block_shape[0], stochastic_rounding,
                                   philox_args, stream, stats_partials);
    });
  }
  if (stats != nullptr) {
    finalize_nvfp4_stats_kernel<<<1, 256, 0, stream>>>(
        stats_partials, partials.size(0), input.numel(), stats->data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return {codes, scales, global_scale};
}

std::tuple<at::Tensor, at::Tensor, c10::optional<at::Tensor>> quantize_nvfp4_cuda(
    const at::Tensor& input, int64_t contract_dim, at::IntArrayRef block_shape,
    bool enable_global_scale, bool stochastic_rounding, double qmax,
    const std::string& output_layout) {
  return quantize_nvfp4_impl(input, contract_dim, block_shape, enable_global_scale,
                              stochastic_rounding, qmax, output_layout, nullptr);
}

std::tuple<at::Tensor, at::Tensor, c10::optional<at::Tensor>, at::Tensor>
quantize_nvfp4_with_stats_cuda(
    const at::Tensor& input, int64_t contract_dim, at::IntArrayRef block_shape,
    bool enable_global_scale, bool stochastic_rounding, double qmax,
    const std::string& output_layout) {
  at::Tensor stats;
  auto [codes, scales, global_scale] = quantize_nvfp4_impl(
      input, contract_dim, block_shape, enable_global_scale, stochastic_rounding,
      qmax, output_layout, &stats);
  return {codes, scales, global_scale, stats};
}

std::tuple<at::Tensor, at::Tensor, c10::optional<at::Tensor>> quantize_nvfp4_meta(
    const at::Tensor& input,
    int64_t contract_dim,
    at::IntArrayRef block_shape,
    bool enable_global_scale,
    bool stochastic_rounding,
    double qmax,
    const std::string& output_layout) {
  check_arguments(input, contract_dim, block_shape, qmax, output_layout);
  auto codes = allocate_codes(input, contract_dim, output_layout);
  auto scales = at::empty_symint(scale_shape(input, contract_dim),
                          input.options().dtype(at::kFloat8_e4m3fn));
  c10::optional<at::Tensor> global_scale;
  if (enable_global_scale) {
    global_scale = at::empty_symint({input.dim() == 3 ? input.sym_size(0) : c10::SymInt(1)},
                             input.options().dtype(at::kFloat));
  }
  return {codes, scales, global_scale};
}

std::tuple<at::Tensor, at::Tensor, c10::optional<at::Tensor>, at::Tensor>
quantize_nvfp4_with_stats_meta(
    const at::Tensor& input, int64_t contract_dim, at::IntArrayRef block_shape,
    bool enable_global_scale, bool stochastic_rounding, double qmax,
    const std::string& output_layout) {
  auto [codes, scales, global_scale] = quantize_nvfp4_meta(
      input, contract_dim, block_shape, enable_global_scale, stochastic_rounding,
      qmax, output_layout);
  return {codes, scales, global_scale,
          at::empty_symint({5}, input.options().dtype(at::kFloat))};
}

}  // namespace

TORCH_LIBRARY_FRAGMENT(aot_kernel, m) {
  m.def(
      "quantize_nvfp4(Tensor x, int contract_dim, int[] block_shape, bool "
      "enable_global_scale=True, bool stochastic_rounding=False, float qmax=6, "
      "str output_layout=\"row_major\") -> (Tensor, "
      "Tensor, Tensor?)",
      {at::Tag::nondeterministic_seeded});
  m.def(
      "quantize_nvfp4_with_stats(Tensor x, int contract_dim, int[] block_shape, bool "
      "enable_global_scale=True, bool stochastic_rounding=False, float qmax=6, "
      "str output_layout=\"row_major\") -> (Tensor, Tensor, Tensor?, Tensor)",
      {at::Tag::nondeterministic_seeded});
}

TORCH_LIBRARY_IMPL(aot_kernel, Meta, m) {
  m.impl("quantize_nvfp4", TORCH_FN(quantize_nvfp4_meta));
  m.impl("quantize_nvfp4_with_stats", TORCH_FN(quantize_nvfp4_with_stats_meta));
}

TORCH_LIBRARY_IMPL(aot_kernel, CUDA, m) {
  m.impl("quantize_nvfp4", TORCH_FN(quantize_nvfp4_cuda));
  m.impl("quantize_nvfp4_with_stats", TORCH_FN(quantize_nvfp4_with_stats_cuda));
}
