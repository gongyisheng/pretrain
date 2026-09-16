#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <ATen/cuda/PhiloxUtils.cuh>
#include <ATen/core/enum_tag.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>

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
constexpr int kValuesPerThread = 8;
constexpr float kMinScale = 1.0e-30f;

struct Params {
  const void* input;
  int8_t* codes;
  float* scales;
  float* statistics;
  float* statistics_partials;
  int64_t batches;
  int64_t rows;
  int64_t cols;
  int64_t batch_stride;
  int64_t row_stride;
  int64_t col_stride;
  int64_t code_batch_stride;
  int64_t code_row_stride;
  int64_t code_col_stride;
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
  TORCH_CHECK(rows <= kMaxGridY, "quantize_int8 tensor is too large to launch");
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

__device__ __forceinline__ int64_t input_offset(
    const Params& params, int64_t batch, int64_t row, int64_t col) {
  return batch * params.batch_stride + row * params.row_stride +
      col * params.col_stride;
}

__device__ __forceinline__ int64_t output_offset(
    const Params& params, int64_t batch, int64_t row, int64_t col) {
  return batch * params.code_batch_stride + row * params.code_row_stride +
      col * params.code_col_stride;
}

__device__ __forceinline__ int64_t scale_offset(
    const Params& params, int64_t batch, int64_t outer, int64_t contract_group) {
  if (params.contract_is_col) {
    return (batch * params.rows + outer) * params.contract_groups + contract_group;
  }
  return (batch * params.contract_groups + contract_group) * params.cols + outer;
}

template <typename input_t>
__device__ __forceinline__ float load_value(
    const Params& params, int64_t batch, int64_t row, int64_t col) {
  return static_cast<float>(static_cast<const input_t*>(params.input)[
      input_offset(params, batch, row, col)]);
}

template <bool kStochastic>
__device__ __forceinline__ int8_t quantize_value(
    float value, float scale, float qmax, curandStatePhilox4_32_10_t* state) {
  float normalized = value / scale;
  if (isnan(normalized)) {
    return 0;
  }
  normalized = fminf(fmaxf(normalized, -qmax), qmax);
  if constexpr (kStochastic) {
    const float lower = floorf(normalized);
    const float probability = normalized - lower;
    const float random = static_cast<float>(curand(state)) * 0x1.0p-32f;
    return static_cast<int8_t>(random < probability ? lower + 1.0f : lower);
  }
  return static_cast<int8_t>(nearbyintf(normalized));
}

struct StatisticsPartial {
  float src_sq = 0.0f;
  float err_sq = 0.0f;
  float under = 0.0f;
  float nonzero = 0.0f;
};

template <bool kStatistics>
__device__ __forceinline__ void accumulate_statistics(
    StatisticsPartial* partial, float source, int8_t code, float scale) {
  if constexpr (kStatistics) {
    const bool source_nonzero = source != 0.0f;
    partial->src_sq += source * source;
    const float error = source - static_cast<float>(code) * scale;
    partial->err_sq += error * error;
    partial->under += source_nonzero && code == 0 ? 1.0f : 0.0f;
    partial->nonzero += source_nonzero ? 1.0f : 0.0f;
  }
}

template <bool kStatistics>
__device__ __forceinline__ void reduce_statistics(
    StatisticsPartial partial, float* statistics_partials, int64_t partial_index) {
  if constexpr (kStatistics) {
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
      float* destination = statistics_partials + partial_index * 4;
      destination[0] = values[0][0];
      destination[1] = values[1][0];
      destination[2] = values[2][0];
      destination[3] = values[3][0];
    }
  }
}

__global__ void finalize_statistics_kernel(
    const float* partials, float* statistics, int64_t count, int64_t numel) {
  float values[4] = {};
  for (int64_t index = threadIdx.x; index < count; index += blockDim.x) {
    const float* partial = partials + index * 4;
    values[0] += partial[0];
    values[1] += partial[1];
    values[2] += partial[2];
    values[3] += partial[3];
  }
  __shared__ float reduced[4][kThreads];
  reduced[0][threadIdx.x] = values[0];
  reduced[1][threadIdx.x] = values[1];
  reduced[2][threadIdx.x] = values[2];
  reduced[3][threadIdx.x] = values[3];
  __syncthreads();
  for (int width = kThreads / 2; width > 0; width >>= 1) {
    if (threadIdx.x < width) {
      reduced[0][threadIdx.x] += reduced[0][threadIdx.x + width];
      reduced[1][threadIdx.x] += reduced[1][threadIdx.x + width];
      reduced[2][threadIdx.x] += reduced[2][threadIdx.x + width];
      reduced[3][threadIdx.x] += reduced[3][threadIdx.x + width];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    statistics[0] = reduced[0][0];
    statistics[1] = reduced[1][0];
    statistics[2] = reduced[2][0];
    statistics[3] = static_cast<float>(numel);
    statistics[4] = reduced[3][0];
  }
}

template <typename input_t, bool kStochastic, bool kStatistics>
__global__ void quantize_tile_kernel(
    Params params,
    float qmax,
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
  const int64_t outer_size = params.contract_is_col ? params.rows : params.cols;
  const int64_t contract_size = params.contract_is_col ? params.cols : params.rows;
  const int64_t outer_start = outer_group * params.block_outer;
  const int64_t contract_start = contract_group * params.block_contract;
  const int64_t outer_end = min(outer_start + params.block_outer, outer_size);
  const int64_t contract_end =
      min(contract_start + params.block_contract, contract_size);

  float maximum = 0.0f;
  for (int64_t outer = outer_start; outer < outer_end; ++outer) {
    for (int64_t contract = contract_start + threadIdx.x; contract < contract_end;
         contract += blockDim.x) {
      const int64_t row = params.contract_is_col ? outer : contract;
      const int64_t col = params.contract_is_col ? contract : outer;
      maximum = max_with_nan(maximum, fabsf(load_value<input_t>(params, batch, row, col)));
    }
  }
  __shared__ float shared_maximum[kThreads];
  shared_maximum[threadIdx.x] = maximum;
  __syncthreads();
  for (int width = kThreads / 2; width > 0; width >>= 1) {
    if (threadIdx.x < width) {
      shared_maximum[threadIdx.x] = max_with_nan(
          shared_maximum[threadIdx.x], shared_maximum[threadIdx.x + width]);
    }
    __syncthreads();
  }
  __shared__ float shared_scale;
  if (threadIdx.x == 0) {
    shared_scale = scale_from_amax(shared_maximum[0], qmax);
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
  for (int64_t outer = outer_start; outer < outer_end; ++outer) {
    for (int64_t contract = contract_start + threadIdx.x; contract < contract_end;
         contract += blockDim.x) {
      const int64_t row = params.contract_is_col ? outer : contract;
      const int64_t col = params.contract_is_col ? contract : outer;
      const int64_t code_offset = output_offset(params, batch, row, col);
      const float value = load_value<input_t>(params, batch, row, col);
      const int8_t code = quantize_value<kStochastic>(
          value, shared_scale, qmax, &random_state);
      params.codes[code_offset] = code;
      accumulate_statistics<kStatistics>(&statistics, value, code, shared_scale);
    }
  }
  reduce_statistics<kStatistics>(statistics, params.statistics_partials,
                                 grid_index(blocks_per_row));
}

template <typename input_t, bool kStochastic, bool kStatistics>
__global__ void quantize_blockwise_1d_kernel(
    Params params,
    float qmax,
    at::PhiloxCudaState philox,
    int64_t blocks_per_row) {
  constexpr int kWarps = kThreads / 32;
  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  const int64_t tile = grid_index(blocks_per_row) * kWarps + warp;
  const int64_t tile_count = params.batches * params.outer_groups * params.contract_groups;
  const bool valid_tile = tile < tile_count;
  if (!valid_tile) {
    if constexpr (!kStatistics) return;
  }
  const int64_t contract_group = tile % params.contract_groups;
  const int64_t remaining = tile / params.contract_groups;
  const int64_t outer = remaining % params.outer_groups;
  const int64_t batch = remaining / params.outer_groups;
  const int64_t contract_size = params.contract_is_col ? params.cols : params.rows;
  const int64_t contract_start = contract_group * params.block_contract;
  const int64_t contract_end = valid_tile
      ? min(contract_start + params.block_contract, contract_size)
      : contract_start;
  float maximum = 0.0f;
  for (int64_t contract = contract_start + lane; contract < contract_end;
       contract += 32) {
    const int64_t row = params.contract_is_col ? outer : contract;
    const int64_t col = params.contract_is_col ? contract : outer;
    maximum = max_with_nan(maximum, fabsf(load_value<input_t>(params, batch, row, col)));
  }
  for (int width = 16; width > 0; width >>= 1) {
    maximum = max_with_nan(maximum, __shfl_down_sync(0xffffffff, maximum, width));
  }
  const float scale = __shfl_sync(0xffffffff, scale_from_amax(maximum, qmax), 0);
  if (valid_tile && lane == 0) {
    params.scales[scale_offset(params, batch, outer, contract_group)] = scale;
  }
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
  for (int64_t contract = contract_start + lane; contract < contract_end;
       contract += 32) {
    const int64_t row = params.contract_is_col ? outer : contract;
    const int64_t col = params.contract_is_col ? contract : outer;
    const float value = load_value<input_t>(params, batch, row, col);
    const int8_t code = quantize_value<kStochastic>(value, scale, qmax, &random_state);
    params.codes[output_offset(params, batch, row, col)] = code;
    accumulate_statistics<kStatistics>(&statistics, value, code, scale);
  }
  reduce_statistics<kStatistics>(statistics, params.statistics_partials,
                                 grid_index(blocks_per_row));
}

template <typename input_t>
__global__ void tensorwise_partial_amax_kernel(
    Params params,
    float* partials,
    int64_t partials_per_batch,
    int64_t blocks_per_row) {
  const int64_t tile = grid_index(blocks_per_row);
  if (tile >= params.batches * partials_per_batch) {
    return;
  }
  const int64_t batch = tile / partials_per_batch;
  const int64_t partial = tile % partials_per_batch;
  const int64_t elements = params.rows * params.cols;
  const int64_t start = partial * kThreads * kValuesPerThread;
  const int64_t end = min(start + kThreads * kValuesPerThread, elements);
  float maximum = 0.0f;
  for (int64_t linear = start + threadIdx.x; linear < end; linear += blockDim.x) {
    const int64_t row = linear / params.cols;
    const int64_t col = linear % params.cols;
    maximum = max_with_nan(maximum, fabsf(load_value<input_t>(params, batch, row, col)));
  }
  __shared__ float shared_maximum[kThreads];
  shared_maximum[threadIdx.x] = maximum;
  __syncthreads();
  for (int width = kThreads / 2; width > 0; width >>= 1) {
    if (threadIdx.x < width) {
      shared_maximum[threadIdx.x] = max_with_nan(
          shared_maximum[threadIdx.x], shared_maximum[threadIdx.x + width]);
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    partials[batch * partials_per_batch + partial] = shared_maximum[0];
  }
}

__global__ void tensorwise_scale_kernel(
    const float* partials, float* scales, int64_t partials_per_batch, float qmax) {
  float maximum = 0.0f;
  for (int64_t partial = threadIdx.x; partial < partials_per_batch;
       partial += blockDim.x) {
    maximum = max_with_nan(maximum, partials[blockIdx.x * partials_per_batch + partial]);
  }
  __shared__ float shared_maximum[kThreads];
  shared_maximum[threadIdx.x] = maximum;
  __syncthreads();
  for (int width = kThreads / 2; width > 0; width >>= 1) {
    if (threadIdx.x < width) {
      shared_maximum[threadIdx.x] = max_with_nan(
          shared_maximum[threadIdx.x], shared_maximum[threadIdx.x + width]);
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    scales[blockIdx.x] = scale_from_amax(shared_maximum[0], qmax);
  }
}

template <typename input_t, bool kStochastic, bool kStatistics>
__global__ void tensorwise_encode_kernel(
    Params params,
    const float* scales,
    float qmax,
    at::PhiloxCudaState philox,
    int64_t blocks_per_row,
    int64_t blocks_per_batch) {
  const int64_t tile = grid_index(blocks_per_row);
  if (tile >= params.batches * blocks_per_batch) {
    return;
  }
  const int64_t batch = tile / blocks_per_batch;
  const int64_t block = tile % blocks_per_batch;
  const int64_t elements = params.rows * params.cols;
  const int64_t start = block * kThreads * kValuesPerThread;
  const int64_t end = min(start + kThreads * kValuesPerThread, elements);
  curandStatePhilox4_32_10_t random_state;
  StatisticsPartial statistics;
  if constexpr (kStochastic) {
    const auto seeds = at::cuda::philox::unpack(philox);
    curand_init(
        static_cast<unsigned long long>(std::get<0>(seeds)),
        static_cast<unsigned long long>(tile * blockDim.x +
                                        threadIdx.x),
        static_cast<unsigned long long>(std::get<1>(seeds)),
        &random_state);
  }
  for (int64_t linear = start + threadIdx.x; linear < end; linear += blockDim.x) {
    const int64_t row = linear / params.cols;
    const int64_t col = linear % params.cols;
    const float value = load_value<input_t>(params, batch, row, col);
    const int8_t code = quantize_value<kStochastic>(
        value, scales[batch], qmax, &random_state);
    params.codes[output_offset(params, batch, row, col)] = code;
    accumulate_statistics<kStatistics>(&statistics, value, code, scales[batch]);
  }
  reduce_statistics<kStatistics>(statistics, params.statistics_partials,
                                 grid_index(blocks_per_row));
}

std::vector<int64_t> scale_shape(
    const at::Tensor& input,
    int64_t contract_dim,
    int64_t block_outer,
    int64_t block_contract) {
  std::vector<int64_t> shape(input.sizes().begin(), input.sizes().end());
  const int64_t axis = contract_dim + input.dim();
  const int64_t outer_axis = axis == input.dim() - 1 ? axis - 1 : axis + 1;
  if (block_outer == 0) {
    shape[axis] = 1;
    return shape;
  }
  if (block_contract == 0) {
    shape[axis] = 1;
  } else {
    shape[axis] = (shape[axis] + block_contract - 1) / block_contract;
  }
  if (block_outer > 1) {
    shape[outer_axis] = input.size(outer_axis);
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

std::vector<int64_t> code_strides(
    const at::Tensor& input, const std::string& output_layout) {
  TORCH_CHECK(output_layout == "row_major" || output_layout == "column_major",
              "quantize_int8 output_layout must be row_major or column_major");
  const int64_t rows = input.size(-2);
  const int64_t cols = input.size(-1);
  if (output_layout == "row_major") {
    if (input.dim() == 2) return {cols, 1};
    return {rows * cols, cols, 1};
  }
  if (input.dim() == 2) return {1, rows};
  return {rows * cols, 1, rows};
}

void check_arguments(
    const at::Tensor& input,
    int64_t contract_dim,
    int64_t block_outer,
    int64_t block_contract,
    int64_t bits) {
  TORCH_CHECK(input.dim() == 2 || input.dim() == 3,
              "quantize_int8 requires a 2D or 3D tensor");
  TORCH_CHECK(input.scalar_type() == at::kFloat || input.scalar_type() == at::kHalf ||
                  input.scalar_type() == at::kBFloat16,
              "quantize_int8 requires float32, float16, or bfloat16 input");
  TORCH_CHECK(contract_dim == -2 || contract_dim == -1,
              "quantize_int8 contract_dim must be -2 or -1");
  TORCH_CHECK(bits >= 4 && bits <= 8, "quantize_int8 bits must be from 4 to 8");
  TORCH_CHECK((block_outer == 0 && block_contract == 0) ||
                  (block_outer == 1 && block_contract == 0) ||
                  (block_contract > 0 &&
                   (block_outer == 1 || block_outer == block_contract)),
              "quantize_int8 block shape must be (0, 0), (1, 0), (1, B), or (B, B)");
  TORCH_CHECK(input.numel() > 0, "quantize_int8 requires a nonempty tensor");
  TORCH_CHECK(std::all_of(input.strides().begin(), input.strides().end(),
                          [](int64_t stride) { return stride >= 0; }),
              "quantize_int8 requires nonnegative strides");
}

template <typename input_t, bool kStochastic, bool kStatistics>
void launch_quantize(
    const at::Tensor& scales,
    const at::Tensor& partials,
    Params params,
    float qmax,
    bool tensorwise,
    at::PhiloxCudaState philox,
    int64_t statistics_partial_count,
    cudaStream_t stream) {
  if (tensorwise) {
    const int64_t elements = params.rows * params.cols;
    const int64_t partials_per_batch =
        (elements + kThreads * kValuesPerThread - 1) / (kThreads * kValuesPerThread);
    const auto partial_grid = make_launch_grid(params.batches * partials_per_batch);
    tensorwise_partial_amax_kernel<input_t><<<partial_grid.grid, kThreads, 0, stream>>>(
        params, partials.data_ptr<float>(), partials_per_batch,
        partial_grid.blocks_per_row);
    tensorwise_scale_kernel<<<params.batches, kThreads, 0, stream>>>(
        partials.data_ptr<float>(), scales.data_ptr<float>(), partials_per_batch, qmax);
    tensorwise_encode_kernel<input_t, kStochastic, kStatistics><<<
        partial_grid.grid, kThreads, 0, stream>>>(
        params, scales.data_ptr<float>(), qmax, philox, partial_grid.blocks_per_row,
        partials_per_batch);
    if constexpr (kStatistics) {
      finalize_statistics_kernel<<<1, kThreads, 0, stream>>>(
          params.statistics_partials, params.statistics, statistics_partial_count,
          params.batches * params.rows * params.cols);
    }
    return;
  }
  const int64_t tiles = params.batches * params.outer_groups * params.contract_groups;
  if (params.block_outer == 1 && params.block_contract > 0) {
    constexpr int kWarps = kThreads / 32;
    const auto grid = make_launch_grid((tiles + kWarps - 1) / kWarps);
    quantize_blockwise_1d_kernel<input_t, kStochastic, kStatistics><<<
        grid.grid, kThreads, 0, stream>>>(params, qmax, philox, grid.blocks_per_row);
    if constexpr (kStatistics) {
      finalize_statistics_kernel<<<1, kThreads, 0, stream>>>(
          params.statistics_partials, params.statistics, statistics_partial_count,
          params.batches * params.rows * params.cols);
    }
    return;
  }
  const auto grid = make_launch_grid(tiles);
  quantize_tile_kernel<input_t, kStochastic, kStatistics><<<
      grid.grid, kThreads, 0, stream>>>(params, qmax, philox, grid.blocks_per_row);
  if constexpr (kStatistics) {
    finalize_statistics_kernel<<<1, kThreads, 0, stream>>>(
        params.statistics_partials, params.statistics, statistics_partial_count,
        params.batches * params.rows * params.cols);
  }
}

std::tuple<at::Tensor, at::Tensor> quantize_int8_impl(
    const at::Tensor& input,
    int64_t contract_dim,
    int64_t block_outer,
    int64_t block_contract,
    int64_t bits,
    bool stochastic_rounding,
    const std::string& output_layout,
    float* statistics,
    const at::Tensor& statistics_partials) {
  TORCH_CHECK(input.is_cuda(), "quantize_int8 requires a CUDA tensor");
  check_arguments(input, contract_dim, block_outer, block_contract, bits);
  c10::cuda::CUDAGuard device_guard(input.device());
  const bool tensorwise = block_outer == 0;
  const bool contract_is_col = contract_dim == -1;
  const int64_t rows = input.size(-2);
  const int64_t cols = input.size(-1);
  const int64_t outer_size = contract_is_col ? rows : cols;
  const int64_t contract_size = contract_is_col ? cols : rows;
  const int64_t actual_block_outer = tensorwise ? 1 : block_outer;
  const int64_t actual_block_contract = tensorwise || block_contract == 0
      ? contract_size
      : block_contract;
  const auto output_shape = scale_shape(input, contract_dim, block_outer, block_contract);
  const at::Tensor codes = at::empty_strided(
      input.sizes(), code_strides(input, output_layout), input.options().dtype(at::kChar));
  const at::Tensor scales = tensorwise
      ? at::empty(tensorwise_storage_shape(input, contract_dim),
                  input.options().dtype(at::kFloat)).expand(output_shape)
      : at::empty(output_shape, input.options().dtype(at::kFloat));
  const int64_t batches = input.dim() == 3 ? input.size(0) : 1;
  Params params{
      input.const_data_ptr(),
      codes.data_ptr<int8_t>(),
      scales.data_ptr<float>(),
      statistics,
      statistics_partials.defined() ? statistics_partials.data_ptr<float>() : nullptr,
      batches,
      rows,
      cols,
      input.dim() == 3 ? input.stride(0) : 0,
      input.stride(-2),
      input.stride(-1),
      input.dim() == 3 ? codes.stride(0) : 0,
      codes.stride(-2),
      codes.stride(-1),
      actual_block_outer,
      actual_block_contract,
      tensorwise ? 1 : (outer_size + actual_block_outer - 1) / actual_block_outer,
      tensorwise || block_contract == 0
          ? 1
          : (contract_size + actual_block_contract - 1) / actual_block_contract,
      contract_is_col};
  const at::Tensor partials = tensorwise
      ? at::empty(
            {batches, (rows * cols + kThreads * kValuesPerThread - 1) /
                          (kThreads * kValuesPerThread)},
            input.options().dtype(at::kFloat))
      : at::Tensor();
  at::PhiloxCudaState philox;
  if (stochastic_rounding) {
    auto generator = at::cuda::detail::getDefaultCUDAGenerator(input.get_device());
    std::lock_guard<std::mutex> lock(generator.mutex());
    auto* cuda_generator = static_cast<at::CUDAGeneratorImpl*>(
        generator.unsafeGetGeneratorImpl());
    philox = cuda_generator->philox_cuda_state(
        ((static_cast<uint64_t>(input.numel()) + 3) / 4) * 4);
  }
  const float qmax = static_cast<float>((1 << (bits - 1)) - 1);
  const auto stream = at::cuda::getCurrentCUDAStream(input.get_device());
#define LAUNCH(TYPE)                                                          \
  if (statistics != nullptr) {                                                \
    if (stochastic_rounding) {                                                \
      launch_quantize<TYPE, true, true>(                                      \
          scales, partials, params, qmax, tensorwise, philox,                   \
          statistics_partials.numel() / 4, stream.stream());                    \
    } else {                                                                  \
      launch_quantize<TYPE, false, true>(                                     \
          scales, partials, params, qmax, tensorwise, philox,                   \
          statistics_partials.numel() / 4, stream.stream());                    \
    }                                                                         \
  } else if (stochastic_rounding) {                                           \
    launch_quantize<TYPE, true, false>(                                      \
        scales, partials, params, qmax, tensorwise, philox, 0, stream.stream()); \
  } else {                                                                    \
    launch_quantize<TYPE, false, false>(                                     \
        scales, partials, params, qmax, tensorwise, philox, 0, stream.stream()); \
  }
  switch (input.scalar_type()) {
    case at::kFloat:
      LAUNCH(float);
      break;
    case at::kHalf:
      LAUNCH(at::Half);
      break;
    case at::kBFloat16:
      LAUNCH(at::BFloat16);
      break;
    default:
      TORCH_CHECK(false, "quantize_int8 has an unsupported input dtype");
  }
#undef LAUNCH
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {codes, scales};
}

std::tuple<at::Tensor, at::Tensor> quantize_int8_cuda(
    const at::Tensor& input,
    int64_t contract_dim,
    int64_t block_outer,
    int64_t block_contract,
    int64_t bits,
    bool stochastic_rounding,
    const std::string& output_layout) {
  return quantize_int8_impl(
      input, contract_dim, block_outer, block_contract, bits, stochastic_rounding,
      output_layout, nullptr, at::Tensor());
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> quantize_int8_with_stats_cuda(
    const at::Tensor& input,
    int64_t contract_dim,
    int64_t block_outer,
    int64_t block_contract,
    int64_t bits,
    bool stochastic_rounding,
    const std::string& output_layout) {
  const bool tensorwise = block_outer == 0;
  const bool contract_is_col = contract_dim == -1;
  const int64_t rows = input.size(-2);
  const int64_t cols = input.size(-1);
  const int64_t batches = input.dim() == 3 ? input.size(0) : 1;
  const int64_t outer_size = contract_is_col ? rows : cols;
  const int64_t contract_size = contract_is_col ? cols : rows;
  const int64_t actual_block_outer = tensorwise ? 1 : block_outer;
  const int64_t actual_block_contract = tensorwise || block_contract == 0
      ? contract_size
      : block_contract;
  const int64_t outer_groups = tensorwise
      ? 1
      : (outer_size + actual_block_outer - 1) / actual_block_outer;
  const int64_t contract_groups = tensorwise || block_contract == 0
      ? 1
      : (contract_size + actual_block_contract - 1) / actual_block_contract;
  const int64_t tiles = batches * outer_groups * contract_groups;
  const int64_t statistics_blocks = tensorwise
      ? batches * ((rows * cols + kThreads * kValuesPerThread - 1) /
                   (kThreads * kValuesPerThread))
      : block_outer == 1 && block_contract > 0
      ? (tiles + kThreads / 32 - 1) / (kThreads / 32)
      : tiles;
  const at::Tensor statistics = at::zeros({5}, input.options().dtype(at::kFloat));
  const at::Tensor statistics_partials = at::zeros(
      {statistics_blocks, 4}, input.options().dtype(at::kFloat));
  auto [codes, scales] = quantize_int8_impl(
      input, contract_dim, block_outer, block_contract, bits, stochastic_rounding,
      output_layout, statistics.data_ptr<float>(), statistics_partials);
  switch (input.scalar_type()) {
    case at::kFloat:
    case at::kHalf:
    case at::kBFloat16:
      break;
    default:
      TORCH_CHECK(false, "quantize_int8 has an unsupported input dtype");
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {codes, scales, statistics};
}

std::tuple<at::Tensor, at::Tensor> quantize_int8_meta(
    const at::Tensor& input,
    int64_t contract_dim,
    int64_t block_outer,
    int64_t block_contract,
    int64_t bits,
    bool stochastic_rounding,
    const std::string& output_layout) {
  check_arguments(input, contract_dim, block_outer, block_contract, bits);
  const auto output_shape = scale_shape(input, contract_dim, block_outer, block_contract);
  const at::Tensor scales = block_outer == 0
      ? at::empty(tensorwise_storage_shape(input, contract_dim),
                  input.options().dtype(at::kFloat)).expand(output_shape)
      : at::empty(output_shape, input.options().dtype(at::kFloat));
  const at::Tensor codes = at::empty_strided(
      input.sizes(), code_strides(input, output_layout), input.options().dtype(at::kChar));
  return {codes, scales};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> quantize_int8_with_stats_meta(
    const at::Tensor& input,
    int64_t contract_dim,
    int64_t block_outer,
    int64_t block_contract,
    int64_t bits,
    bool stochastic_rounding,
    const std::string& output_layout) {
  auto [codes, scales] = quantize_int8_meta(
      input, contract_dim, block_outer, block_contract, bits, stochastic_rounding,
      output_layout);
  return {codes, scales, at::empty({5}, input.options().dtype(at::kFloat))};
}

}  // namespace

TORCH_LIBRARY_FRAGMENT(aot_kernel, m) {
  m.def(
      "quantize_int8(Tensor x, int contract_dim, int block_outer, int block_contract, "
      "int bits, bool stochastic_rounding=False, str output_layout=\"row_major\") -> (Tensor, Tensor)",
      {at::Tag::nondeterministic_seeded});
  m.def(
      "quantize_int8_with_stats(Tensor x, int contract_dim, int block_outer, int block_contract, "
      "int bits, bool stochastic_rounding=False, str output_layout=\"row_major\") -> (Tensor, Tensor, Tensor)",
      {at::Tag::nondeterministic_seeded});
}

TORCH_LIBRARY_IMPL(aot_kernel, Meta, m) {
  m.impl("quantize_int8", TORCH_FN(quantize_int8_meta));
  m.impl("quantize_int8_with_stats", TORCH_FN(quantize_int8_with_stats_meta));
}

TORCH_LIBRARY_IMPL(aot_kernel, CUDA, m) {
  m.impl("quantize_int8", TORCH_FN(quantize_int8_cuda));
  m.impl("quantize_int8_with_stats", TORCH_FN(quantize_int8_with_stats_cuda));
}
