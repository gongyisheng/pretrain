#include "mxfp8_grouped.cuh"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace {

constexpr int64_t kScaleBlock = 32;
constexpr int64_t kScaleRows = 128;
constexpr int64_t kScaleColumns = 4;

__device__ __forceinline__ int64_t round_up(int64_t value, int64_t multiple) {
  return ((value + multiple - 1) / multiple) * multiple;
}

__device__ __forceinline__ uint8_t to_e8m0(float value) {
  return __nv_cvt_float_to_e8m0(value, __NV_SATFINITE, cudaRoundPosInf);
}

__global__ void pack_and_build_metadata(
    const int32_t* offsets,
    const float* scale_a,
    const float* scale_b,
    int64_t group_count,
    int64_t n,
    int64_t k,
    int64_t blocks,
    int64_t padded_blocks,
    int64_t aq_stride_m,
    int64_t bq_stride_group,
    int64_t bq_stride_n,
    int64_t out_stride_m,
    uint8_t* packed_a_scales,
    uint8_t* packed_b_scales,
    const uint8_t* empty_a,
    uint8_t* empty_out,
    const uint8_t* empty_b_scale,
    int64_t* m,
    int64_t* n_array,
    int64_t* k_array,
    int64_t* lda,
    int64_t* ldb,
    int64_t* ldc,
    int64_t* ldd,
    int64_t* a_pointers,
    int64_t* b_pointers,
    int64_t* c_pointers,
    int64_t* d_pointers,
    int64_t* a_scale_pointers,
    int64_t* b_scale_pointers,
    const uint8_t* aq,
    const uint8_t* bq,
    const uint8_t* out) {
  const int64_t group = blockIdx.x;
  if (group >= group_count) {
    return;
  }

  const int64_t end = static_cast<int64_t>(offsets[group]);
  const int64_t start = group == 0 ? 0 : static_cast<int64_t>(offsets[group - 1]);
  const int64_t group_m = end - start;
  const int64_t effective_m = group_m == 0 ? 1 : group_m;
  const int64_t padded_m = round_up(group_m, kScaleRows);
  const int64_t padded_n = round_up(n, kScaleRows);

  // `packed_b_scales` has variable-size regions. Its prefix can be derived
  // entirely on device from the cumulative offsets; group counts are tiny in
  // MoE and this avoids a host round-trip for offsets.
  int64_t padded_m_prefix = 0;
  for (int64_t previous = 0; previous < group; ++previous) {
    const int64_t previous_end = static_cast<int64_t>(offsets[previous]);
    const int64_t previous_start =
        previous == 0 ? 0 : static_cast<int64_t>(offsets[previous - 1]);
    padded_m_prefix += round_up(previous_end - previous_start, kScaleRows);
  }

  if (threadIdx.x == 0) {
    m[group] = effective_m;
    n_array[group] = n;
    k_array[group] = k;
    lda[group] = bq_stride_n;
    ldb[group] = aq_stride_m;
    ldc[group] = out_stride_m;
    ldd[group] = out_stride_m;
    a_pointers[group] = reinterpret_cast<int64_t>(bq + group * bq_stride_group);
    b_pointers[group] = reinterpret_cast<int64_t>(
        group_m == 0 ? empty_a : aq + start * aq_stride_m);
    c_pointers[group] = reinterpret_cast<int64_t>(
        group_m == 0 ? empty_out : out + start * out_stride_m * 2);
    d_pointers[group] = reinterpret_cast<int64_t>(
        group_m == 0 ? empty_out : out + start * out_stride_m * 2);
    a_scale_pointers[group] = reinterpret_cast<int64_t>(
        packed_a_scales + group * padded_n * padded_blocks);
    b_scale_pointers[group] = reinterpret_cast<int64_t>(group_m == 0
        ? empty_b_scale
        : packed_b_scales + padded_m_prefix * padded_blocks);
  }

  // cuBLASLt's VEC32 scale layout is a 128x4 tiled swizzle. A corresponds to
  // Bq here (B^T @ A^T), so its source scales are [N, blocks].
  for (int64_t linear = threadIdx.x;
       linear < padded_n * padded_blocks;
       linear += blockDim.x) {
    const int64_t row = linear / padded_blocks;
    const int64_t block = linear % padded_blocks;
    const int64_t tile_row = row / kScaleRows;
    const int64_t tile_block = block / kScaleColumns;
    const int64_t swizzled =
        ((((tile_row * (padded_blocks / kScaleColumns) + tile_block) * 32 +
           (row % 32)) *
              4 +
          ((row % kScaleRows) / 32)) *
             4 +
         (block % kScaleColumns));
    const float value = row < n && block < blocks
        ? scale_b[(group * blocks + block) * n + row]
        : 1.0f;
    packed_a_scales[group * padded_n * padded_blocks + swizzled] = to_e8m0(value);
  }

  // B corresponds to Aq and its source scales are [M, blocks].
  for (int64_t linear = threadIdx.x;
       linear < padded_m * padded_blocks;
       linear += blockDim.x) {
    const int64_t row = linear / padded_blocks;
    const int64_t block = linear % padded_blocks;
    const int64_t tile_row = row / kScaleRows;
    const int64_t tile_block = block / kScaleColumns;
    const int64_t swizzled =
        ((((tile_row * (padded_blocks / kScaleColumns) + tile_block) * 32 +
           (row % 32)) *
              4 +
          ((row % kScaleRows) / 32)) *
             4 +
         (block % kScaleColumns));
    const float value = row < group_m && block < blocks
        ? scale_a[(start + row) * blocks + block]
        : 1.0f;
    packed_b_scales[padded_m_prefix * padded_blocks + swizzled] = to_e8m0(value);
  }
}

__global__ void validate_offsets(
    const int32_t* offsets, int64_t group_count, int64_t rows, int32_t* valid) {
  const int64_t group = blockIdx.x * blockDim.x + threadIdx.x;
  if (group >= group_count) {
    return;
  }
  const int64_t end = static_cast<int64_t>(offsets[group]);
  const int64_t start = group == 0 ? 0 : static_cast<int64_t>(offsets[group - 1]);
  if (start < 0 || end < start || end > rows ||
      (group == group_count - 1 && end != rows)) {
    atomicExch(valid, 0);
  }
}

__global__ void fill_neutral_scale(uint8_t* scale, int64_t size) {
  for (int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
       index < size;
       index += blockDim.x * gridDim.x) {
    scale[index] = to_e8m0(1.0f);
  }
}

}  // namespace

void validate_mxfp8_grouped_offsets(const at::Tensor& offsets, int64_t rows) {
  const auto valid = at::ones({1}, offsets.options().dtype(at::kInt));
  constexpr int kThreads = 256;
  const int64_t blocks = (offsets.size(0) + kThreads - 1) / kThreads;
  const auto stream = at::cuda::getCurrentCUDAStream(offsets.device().index());
  validate_offsets<<<blocks, kThreads, 0, stream.stream()>>>(
      offsets.data_ptr<int32_t>(),
      offsets.size(0),
      rows,
      valid.data_ptr<int32_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  int32_t host_valid = 0;
  C10_CUDA_CHECK(cudaMemcpyAsync(
      &host_valid,
      valid.data_ptr<int32_t>(),
      sizeof(host_valid),
      cudaMemcpyDeviceToHost,
      stream.stream()));
  C10_CUDA_CHECK(cudaStreamSynchronize(stream.stream()));
  TORCH_CHECK(
      host_valid == 1,
      "scaled_grouped_gemm_mxfp8: offsets must be nondecreasing, within [0, M], and end at M");
}

Mxfp8GroupedMetadata build_mxfp8_grouped_metadata(
    const at::Tensor& aq,
    const at::Tensor& bq,
    const at::Tensor& scale_a,
    const at::Tensor& scale_b,
    const at::Tensor& offsets,
    const at::Tensor& out) {
  const int64_t group_count = offsets.size(0);
  const int64_t m_total = aq.size(0);
  const int64_t n = bq.size(2);
  const int64_t k = aq.size(1);
  const int64_t blocks = (k + kScaleBlock - 1) / kScaleBlock;
  const int64_t padded_blocks = ((blocks + kScaleColumns - 1) / kScaleColumns) * kScaleColumns;
  const int64_t padded_n = ((n + kScaleRows - 1) / kScaleRows) * kScaleRows;

  const auto int_options = offsets.options().dtype(at::kLong);
  const auto byte_options = scale_a.options().dtype(at::kByte);
  Mxfp8GroupedMetadata metadata{
      at::empty({group_count}, int_options),
      at::empty({group_count}, int_options),
      at::empty({group_count}, int_options),
      at::empty({group_count}, int_options),
      at::empty({group_count}, int_options),
      at::empty({group_count}, int_options),
      at::empty({group_count}, int_options),
      at::empty({group_count}, int_options),
      at::empty({group_count}, int_options),
      at::empty({group_count}, int_options),
      at::empty({group_count}, int_options),
      at::empty({group_count}, int_options),
      at::empty({group_count}, int_options),
      at::empty({group_count * padded_n * padded_blocks}, byte_options),
      at::empty({(m_total + (kScaleRows - 1) * group_count) * padded_blocks}, byte_options),
      at::zeros({k}, byte_options),
      at::zeros({n}, byte_options),
      at::empty({kScaleRows * padded_blocks}, byte_options),
  };

  constexpr int kThreads = 256;
  const auto stream = at::cuda::getCurrentCUDAStream(aq.device().index());
  fill_neutral_scale<<<1, kThreads, 0, stream.stream()>>>(
      metadata.empty_b_scale.data_ptr<uint8_t>(), metadata.empty_b_scale.numel());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  pack_and_build_metadata<<<group_count, kThreads, 0, stream.stream()>>>(
      offsets.data_ptr<int32_t>(),
      scale_a.data_ptr<float>(),
      scale_b.data_ptr<float>(),
      group_count,
      n,
      k,
      blocks,
      padded_blocks,
      aq.stride(0),
      bq.stride(0),
      bq.stride(2),
      out.stride(0),
      metadata.packed_a_scales.data_ptr<uint8_t>(),
      metadata.packed_b_scales.data_ptr<uint8_t>(),
      metadata.empty_a.data_ptr<uint8_t>(),
      metadata.empty_out.data_ptr<uint8_t>(),
      metadata.empty_b_scale.data_ptr<uint8_t>(),
      metadata.m.data_ptr<int64_t>(),
      metadata.n.data_ptr<int64_t>(),
      metadata.k.data_ptr<int64_t>(),
      metadata.lda.data_ptr<int64_t>(),
      metadata.ldb.data_ptr<int64_t>(),
      metadata.ldc.data_ptr<int64_t>(),
      metadata.ldd.data_ptr<int64_t>(),
      metadata.a_pointers.data_ptr<int64_t>(),
      metadata.b_pointers.data_ptr<int64_t>(),
      metadata.c_pointers.data_ptr<int64_t>(),
      metadata.d_pointers.data_ptr<int64_t>(),
      metadata.a_scale_pointers.data_ptr<int64_t>(),
      metadata.b_scale_pointers.data_ptr<int64_t>(),
      static_cast<const uint8_t*>(aq.data_ptr()),
      static_cast<const uint8_t*>(bq.data_ptr()),
      static_cast<uint8_t*>(out.data_ptr()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return metadata;
}
