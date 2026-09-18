#pragma once

#include <math_constants.h>
#include <cuda_runtime.h>

namespace cuda_reduce {

template <typename T>
struct Sum {
  __device__ __forceinline__ T operator()(T lhs, T rhs) const {
    return lhs + rhs;
  }

  __device__ __forceinline__ T identity() const {
    return T{0};
  }
};

// Matches fmaxf: a single NaN is ignored. Callers that propagate NaNs provide
// their own operation.
struct Fmax {
  __device__ __forceinline__ float operator()(float lhs, float rhs) const {
    return fmaxf(lhs, rhs);
  }

  __device__ __forceinline__ float identity() const {
    return -CUDART_INF_F;
  }
};

// T must support __shfl_down_sync. All lanes in each participating subgroup
// must be active; only subgroup lane 0 receives the reduced value.
template <int kWidth, typename T, typename Op>
__device__ __forceinline__ T warp_reduce(T value, Op op, unsigned mask) {
  static_assert(kWidth > 0 && kWidth <= 32 && (kWidth & (kWidth - 1)) == 0);
#pragma unroll
  for (int offset = kWidth / 2; offset > 0; offset >>= 1) {
    value = op(value, __shfl_down_sync(mask, value, offset, kWidth));
  }
  return value;
}

// Array overload: reduces each field independently across threads, in place.
// All threads in a 1D block with blockDim.x == kThreads must participate.
// Only thread 0 receives the reduced values. Fields share one block barrier.
// Call __syncthreads() before reusing this specialization's shared storage.
template <int kThreads, int kValues, typename T, typename Op>
__device__ __forceinline__ void block_reduce(T (&values)[kValues], Op op) {
  static_assert(kThreads >= 32 && kThreads <= 1024 && kThreads % 32 == 0);
  constexpr int kWarps = kThreads / 32;
  __shared__ T warp_values[kValues][kWarps];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;

#pragma unroll
  for (int field = 0; field < kValues; ++field) {
    values[field] = warp_reduce<32>(values[field], op, 0xffffffffu);
    if (lane == 0) warp_values[field][warp] = values[field];
  }
  __syncthreads();

  if (warp == 0) {
#pragma unroll
    for (int field = 0; field < kValues; ++field) {
      values[field] = lane < kWarps ? warp_values[field][lane] : op.identity();
      values[field] = warp_reduce<32>(values[field], op, 0xffffffffu);
    }
  }
}

// Scalar overload: delegates to the array overload with one field.
// All threads in a 1D block with blockDim.x == kThreads must participate.
// Only thread 0 receives the reduced value.
template <int kThreads, typename T, typename Op>
__device__ __forceinline__ T block_reduce(T value, Op op) {
  T values[1] = {value};
  block_reduce<kThreads>(values, op);
  return values[0];
}

}  // namespace cuda_reduce
