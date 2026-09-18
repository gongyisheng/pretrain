#include "quantize.cuh"

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

dim3 make_launch_grid(int64_t blocks) {
  constexpr int64_t kMaxGridX = std::numeric_limits<int32_t>::max();
  constexpr int64_t kMaxGridY = 65535;
  const int64_t grid_x = std::min(blocks, kMaxGridX);
  const int64_t grid_y = (blocks + grid_x - 1) / grid_x;
  TORCH_CHECK(grid_y <= kMaxGridY, "quantize_int8 tensor is too large to launch");
  return dim3(grid_x, grid_y);
}

template <typename input_t, typename index_t>
__device__ __forceinline__ float load_value(
    const QuantizeParams& params, int64_t batch, int64_t row, int64_t col) {
  return static_cast<float>(params.input_data<input_t>()[
      input_offset<index_t>(params, batch, row, col)]);
}

template <typename index_t>
__device__ __forceinline__ int64_t tensorwise_local_offset(
    index_t start_col,
    index_t position,
    int64_t cols,
    int64_t row_stride,
    int64_t col_stride) {
  const index_t local_col = start_col + position;
  const index_t row_delta = local_col / static_cast<index_t>(cols);
  const index_t col_delta =
      local_col - row_delta * static_cast<index_t>(cols) - start_col;
  return static_cast<int64_t>(
      row_delta * static_cast<index_t>(row_stride) +
      col_delta * static_cast<index_t>(col_stride));
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

template <typename input_t, typename index_t, bool kStochastic, bool kStatistics>
__global__ void quantize_tile_kernel(
    QuantizeParams params,
    float qmax,
    at::PhiloxCudaState philox) {
  const int64_t outer_groups =
      params.contract_dim == -1 ? params.row_block_count : params.col_block_count;
  const int64_t contract_groups =
      params.contract_dim == -1 ? params.col_block_count : params.row_block_count;
  const int64_t block_outer =
      params.contract_dim == -1 ? params.rows_per_block : params.cols_per_block;
  const int64_t block_contract =
      params.contract_dim == -1 ? params.cols_per_block : params.rows_per_block;
  const int64_t tile = grid_index();
  const int64_t tile_count =
      params.batch * outer_groups * contract_groups;
  if (tile >= tile_count) {
    return;
  }
  const int64_t contract_group = tile % contract_groups;
  const int64_t remaining = tile / contract_groups;
  const int64_t outer_group = remaining % outer_groups;
  const int64_t batch = remaining / outer_groups;
  const int64_t outer_size =
      params.contract_dim == -1 ? params.row : params.col;
  const int64_t contract_size =
      params.contract_dim == -1 ? params.col : params.row;
  const int64_t outer_start = outer_group * block_outer;
  const int64_t contract_start = contract_group * block_contract;
  const int64_t outer_end = min(outer_start + block_outer, outer_size);
  const int64_t contract_end =
      min(contract_start + block_contract, contract_size);

  float maximum = 0.0f;
  for (int64_t outer = outer_start; outer < outer_end; ++outer) {
    for (int64_t contract = contract_start + threadIdx.x; contract < contract_end;
         contract += blockDim.x) {
      const int64_t row = params.contract_dim == -1 ? outer : contract;
      const int64_t col = params.contract_dim == -1 ? contract : outer;
      maximum = strict_max(
          maximum, fabsf(load_value<input_t, index_t>(params, batch, row, col)));
    }
  }
  __shared__ float shared_maximum[kThreads];
  shared_maximum[threadIdx.x] = maximum;
  __syncthreads();
  for (int width = kThreads / 2; width > 0; width >>= 1) {
    if (threadIdx.x < width) {
      shared_maximum[threadIdx.x] = strict_max(
          shared_maximum[threadIdx.x], shared_maximum[threadIdx.x + width]);
    }
    __syncthreads();
  }
  __shared__ float shared_scale;
  if (threadIdx.x == 0) {
    shared_scale = compute_scale(shared_maximum[0], qmax);
    for (int64_t outer = outer_start; outer < outer_end; ++outer) {
      const int64_t row = params.contract_dim == -1 ? outer : contract_group;
      const int64_t col = params.contract_dim == -1 ? contract_group : outer;
      params.scale_data<float>()[scale_offset<index_t>(
          params, batch, row, col)] = shared_scale;
    }
  }
  __syncthreads();

  curandStatePhilox4_32_10_t random_state;
  QuantizationStatistics statistics;
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
      const int64_t row = params.contract_dim == -1 ? outer : contract;
      const int64_t col = params.contract_dim == -1 ? contract : outer;
      const int64_t code_offset = output_offset<index_t>(params, batch, row, col);
      const float value = load_value<input_t, index_t>(params, batch, row, col);
      const int8_t code = quantize_value<kStochastic>(
          value, shared_scale, qmax, &random_state);
      params.code_data<int8_t>()[code_offset] = code;
      if constexpr (kStatistics) {
        accumulate_quantization_statistics(
            &statistics, value, static_cast<float>(code) * shared_scale);
      }
    }
  }
  if constexpr (kStatistics) {
    quantization_statistics_epilogue<kThreads>(
        statistics, params.statistics, params.batch * params.row * params.col);
  }
}

template <typename input_t, typename index_t, bool kStochastic, bool kStatistics>
__global__ void quantize_blockwise_1d_kernel(
    QuantizeParams params,
    float qmax,
    at::PhiloxCudaState philox) {
  const int64_t outer_groups =
      params.contract_dim == -1 ? params.row_block_count : params.col_block_count;
  const int64_t contract_groups =
      params.contract_dim == -1 ? params.col_block_count : params.row_block_count;
  const int64_t block_contract =
      params.contract_dim == -1 ? params.cols_per_block : params.rows_per_block;
  constexpr int kWarps = kThreads / 32;
  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  const int64_t tile = grid_index() * kWarps + warp;
  const int64_t tile_count = params.batch * outer_groups * contract_groups;
  const bool valid_tile = tile < tile_count;
  if (!valid_tile) {
    if constexpr (!kStatistics) return;
  }
  const int64_t contract_group = tile % contract_groups;
  const int64_t remaining = tile / contract_groups;
  const int64_t outer = remaining % outer_groups;
  const int64_t batch = remaining / outer_groups;
  const int64_t contract_size =
      params.contract_dim == -1 ? params.col : params.row;
  const int64_t contract_start = contract_group * block_contract;
  const int64_t contract_end = valid_tile
      ? min(contract_start + block_contract, contract_size)
      : contract_start;
  float maximum = 0.0f;
  for (int64_t contract = contract_start + lane; contract < contract_end;
       contract += 32) {
    const int64_t row = params.contract_dim == -1 ? outer : contract;
    const int64_t col = params.contract_dim == -1 ? contract : outer;
    maximum = strict_max(
        maximum, fabsf(load_value<input_t, index_t>(params, batch, row, col)));
  }
  for (int width = 16; width > 0; width >>= 1) {
    maximum = strict_max(maximum, __shfl_down_sync(0xffffffff, maximum, width));
  }
  const float scale = __shfl_sync(0xffffffff, compute_scale(maximum, qmax), 0);
  if (valid_tile && lane == 0) {
    const int64_t row = params.contract_dim == -1 ? outer : contract_group;
    const int64_t col = params.contract_dim == -1 ? contract_group : outer;
    params.scale_data<float>()[scale_offset<index_t>(
        params, batch, row, col)] = scale;
  }
  curandStatePhilox4_32_10_t random_state;
  QuantizationStatistics statistics;
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
    const int64_t row = params.contract_dim == -1 ? outer : contract;
    const int64_t col = params.contract_dim == -1 ? contract : outer;
    const float value = load_value<input_t, index_t>(params, batch, row, col);
    const int8_t code = quantize_value<kStochastic>(value, scale, qmax, &random_state);
    params.code_data<int8_t>()[output_offset<index_t>(params, batch, row, col)] = code;
    if constexpr (kStatistics) {
      accumulate_quantization_statistics(
          &statistics, value, static_cast<float>(code) * scale);
    }
  }
  if constexpr (kStatistics) {
    quantization_statistics_epilogue<kThreads>(
        statistics, params.statistics, params.batch * params.row * params.col);
  }
}

template <typename input_t, typename index_t>
__global__ void tensorwise_partial_amax_kernel(
    QuantizeParams params,
    float* partials,
    int64_t partials_per_batch) {
  const int64_t tile = grid_index();
  if (tile >= params.batch * partials_per_batch) {
    return;
  }
  const int64_t batch = tile / partials_per_batch;
  const int64_t partial = tile % partials_per_batch;
  const int64_t elements = params.row * params.col;
  const int64_t start = partial * kThreads * kValuesPerThread;
  const int64_t start_row = start / params.col;
  const index_t start_col = static_cast<index_t>(start - start_row * params.col);
  const int64_t input_base = batch * params.input_batch_stride +
      start_row * params.input_row_stride +
      static_cast<int64_t>(start_col) * params.input_col_stride;
  float maximum = 0.0f;
  for (index_t position = static_cast<index_t>(threadIdx.x);
       position < kThreads * kValuesPerThread;
       position += static_cast<index_t>(blockDim.x)) {
    if (start + static_cast<int64_t>(position) >= elements) {
      break;
    }
    const int64_t input_offset = input_base + tensorwise_local_offset(
        start_col, position, params.col, params.input_row_stride,
        params.input_col_stride);
    const float value = static_cast<float>(params.input_data<input_t>()[input_offset]);
    maximum = strict_max(maximum, fabsf(value));
  }
  __shared__ float shared_maximum[kThreads];
  shared_maximum[threadIdx.x] = maximum;
  __syncthreads();
  for (int width = kThreads / 2; width > 0; width >>= 1) {
    if (threadIdx.x < width) {
      shared_maximum[threadIdx.x] = strict_max(
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
    maximum = strict_max(maximum, partials[blockIdx.x * partials_per_batch + partial]);
  }
  __shared__ float shared_maximum[kThreads];
  shared_maximum[threadIdx.x] = maximum;
  __syncthreads();
  for (int width = kThreads / 2; width > 0; width >>= 1) {
    if (threadIdx.x < width) {
      shared_maximum[threadIdx.x] = strict_max(
          shared_maximum[threadIdx.x], shared_maximum[threadIdx.x + width]);
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    scales[blockIdx.x] = compute_scale(shared_maximum[0], qmax);
  }
}

template <typename input_t, typename index_t, bool kStochastic, bool kStatistics>
__global__ void tensorwise_encode_kernel(
    QuantizeParams params,
    const float* scales,
    float qmax,
    at::PhiloxCudaState philox,
    int64_t blocks_per_batch) {
  const int64_t tile = grid_index();
  if (tile >= params.batch * blocks_per_batch) {
    return;
  }
  const int64_t batch = tile / blocks_per_batch;
  const int64_t block = tile % blocks_per_batch;
  const int64_t elements = params.row * params.col;
  const int64_t start = block * kThreads * kValuesPerThread;
  const int64_t start_row = start / params.col;
  const index_t start_col = static_cast<index_t>(start - start_row * params.col);
  const int64_t input_base = batch * params.input_batch_stride +
      start_row * params.input_row_stride +
      static_cast<int64_t>(start_col) * params.input_col_stride;
  const int64_t output_base = batch * params.code_batch_stride +
      start_row * params.code_row_stride +
      static_cast<int64_t>(start_col) * params.code_col_stride;
  curandStatePhilox4_32_10_t random_state;
  QuantizationStatistics statistics;
  if constexpr (kStochastic) {
    const auto seeds = at::cuda::philox::unpack(philox);
    curand_init(
        static_cast<unsigned long long>(std::get<0>(seeds)),
        static_cast<unsigned long long>(tile * blockDim.x +
                                        threadIdx.x),
        static_cast<unsigned long long>(std::get<1>(seeds)),
        &random_state);
  }
  for (index_t position = static_cast<index_t>(threadIdx.x);
       position < kThreads * kValuesPerThread;
       position += static_cast<index_t>(blockDim.x)) {
    if (start + static_cast<int64_t>(position) >= elements) {
      break;
    }
    const int64_t input_offset = input_base + tensorwise_local_offset(
        start_col, position, params.col, params.input_row_stride,
        params.input_col_stride);
    const int64_t output_offset = output_base + tensorwise_local_offset(
        start_col, position, params.col, params.code_row_stride,
        params.code_col_stride);
    const float value = static_cast<float>(params.input_data<input_t>()[input_offset]);
    const int8_t code = quantize_value<kStochastic>(
        value, scales[batch], qmax, &random_state);
    params.code_data<int8_t>()[output_offset] = code;
    if constexpr (kStatistics) {
      accumulate_quantization_statistics(
          &statistics, value, static_cast<float>(code) * scales[batch]);
    }
  }
  if constexpr (kStatistics) {
    quantization_statistics_epilogue<kThreads>(
        statistics, params.statistics, params.batch * params.row * params.col);
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

template <typename input_t, typename index_t, bool kStochastic, bool kStatistics>
void launch_quantize(
    const at::Tensor& scales,
    const at::Tensor& partials,
    QuantizeParams params,
    float qmax,
    bool tensorwise,
    at::PhiloxCudaState philox,
    cudaStream_t stream) {
  const int64_t outer_groups =
      params.contract_dim == -1 ? params.row_block_count : params.col_block_count;
  const int64_t contract_groups =
      params.contract_dim == -1 ? params.col_block_count : params.row_block_count;
  const int64_t block_outer =
      params.contract_dim == -1 ? params.rows_per_block : params.cols_per_block;
  const int64_t block_contract =
      params.contract_dim == -1 ? params.cols_per_block : params.rows_per_block;
  if (tensorwise) {
    const int64_t elements = params.row * params.col;
    const int64_t partials_per_batch =
        (elements + kThreads * kValuesPerThread - 1) / (kThreads * kValuesPerThread);
    const auto partial_grid = make_launch_grid(params.batch * partials_per_batch);
    tensorwise_partial_amax_kernel<input_t, index_t><<<
        partial_grid, kThreads, 0, stream>>>(
        params, partials.data_ptr<float>(), partials_per_batch);
    tensorwise_scale_kernel<<<params.batch, kThreads, 0, stream>>>(
        partials.data_ptr<float>(), scales.data_ptr<float>(), partials_per_batch, qmax);
    tensorwise_encode_kernel<input_t, index_t, kStochastic, kStatistics><<<
        partial_grid, kThreads, 0, stream>>>(
        params, scales.data_ptr<float>(), qmax, philox, partials_per_batch);
    return;
  }
  const int64_t tiles = params.batch * outer_groups * contract_groups;
  if (block_outer == 1 && block_contract > 0) {
    constexpr int kWarps = kThreads / 32;
    const auto grid = make_launch_grid((tiles + kWarps - 1) / kWarps);
    quantize_blockwise_1d_kernel<input_t, index_t, kStochastic, kStatistics><<<
        grid, kThreads, 0, stream>>>(params, qmax, philox);
    return;
  }
  const auto grid = make_launch_grid(tiles);
  quantize_tile_kernel<input_t, index_t, kStochastic, kStatistics><<<
      grid, kThreads, 0, stream>>>(params, qmax, philox);
}

std::tuple<at::Tensor, at::Tensor> quantize_int8_impl(
    const at::Tensor& input,
    int64_t contract_dim,
    int64_t block_outer,
    int64_t block_contract,
    int64_t bits,
    bool stochastic_rounding,
    const std::string& output_layout,
    float* statistics) {
  TORCH_CHECK(input.is_cuda(), "quantize_int8 requires a CUDA tensor");
  check_arguments(input, contract_dim, block_outer, block_contract, bits);
  c10::cuda::CUDAGuard device_guard(input.device());
  const bool tensorwise = block_outer == 0;
  const bool contract_is_col = contract_dim == -1;
  const int64_t rows = input.size(-2);
  const int64_t cols = input.size(-1);
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
  QuantizeParams params;
  params.input = input.const_data_ptr();
  params.codes = codes.data_ptr();
  params.scale = scales.data_ptr();
  params.statistics = statistics;
  params.batch = batches;
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
  params.rows_per_block = tensorwise ? rows
      : contract_is_col ? actual_block_outer : actual_block_contract;
  params.cols_per_block = tensorwise ? cols
      : contract_is_col ? actual_block_contract : actual_block_outer;
  params.row_block_count =
      (rows + params.rows_per_block - 1) / params.rows_per_block;
  params.col_block_count =
      (cols + params.cols_per_block - 1) / params.cols_per_block;
  params.contract_dim = static_cast<int>(contract_dim);
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
  const bool use_int32_indices = can_use_int32_indices(params);
#define LAUNCH_INDEX(TYPE, INDEX)                                             \
  if (statistics != nullptr) {                                                \
    if (stochastic_rounding) {                                                \
      launch_quantize<TYPE, INDEX, true, true>(                               \
          scales, partials, params, qmax, tensorwise, philox, stream.stream()); \
    } else {                                                                  \
      launch_quantize<TYPE, INDEX, false, true>(                              \
          scales, partials, params, qmax, tensorwise, philox, stream.stream()); \
    }                                                                         \
  } else if (stochastic_rounding) {                                           \
    launch_quantize<TYPE, INDEX, true, false>(                               \
        scales, partials, params, qmax, tensorwise, philox, stream.stream()); \
  } else {                                                                    \
    launch_quantize<TYPE, INDEX, false, false>(                              \
        scales, partials, params, qmax, tensorwise, philox, stream.stream()); \
  }
#define LAUNCH(TYPE)                                                          \
  if (use_int32_indices) {                                                    \
    LAUNCH_INDEX(TYPE, int32_t);                                              \
  } else {                                                                    \
    LAUNCH_INDEX(TYPE, int64_t);                                              \
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
#undef LAUNCH_INDEX
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
      output_layout, nullptr);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> quantize_int8_with_stats_cuda(
    const at::Tensor& input,
    int64_t contract_dim,
    int64_t block_outer,
    int64_t block_contract,
    int64_t bits,
    bool stochastic_rounding,
    const std::string& output_layout) {
  const at::Tensor statistics = at::zeros({5}, input.options().dtype(at::kFloat));
  auto [codes, scales] = quantize_int8_impl(
      input, contract_dim, block_outer, block_contract, bits, stochastic_rounding,
      output_layout, statistics.data_ptr<float>());
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
