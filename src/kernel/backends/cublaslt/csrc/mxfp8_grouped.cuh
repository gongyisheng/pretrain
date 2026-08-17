#pragma once

#include <torch/extension.h>

struct Mxfp8GroupedMetadata {
  at::Tensor m;
  at::Tensor n;
  at::Tensor k;
  at::Tensor lda;
  at::Tensor ldb;
  at::Tensor ldc;
  at::Tensor ldd;
  at::Tensor a_pointers;
  at::Tensor b_pointers;
  at::Tensor c_pointers;
  at::Tensor d_pointers;
  at::Tensor a_scale_pointers;
  at::Tensor b_scale_pointers;
  at::Tensor packed_a_scales;
  at::Tensor packed_b_scales;
  at::Tensor empty_a;
  at::Tensor empty_out;
  at::Tensor empty_b_scale;
};

void validate_mxfp8_grouped_offsets(const at::Tensor& offsets, int64_t rows);

Mxfp8GroupedMetadata build_mxfp8_grouped_metadata(
    const at::Tensor& aq,
    const at::Tensor& bq,
    const at::Tensor& scale_a,
    const at::Tensor& scale_b,
    const at::Tensor& offsets,
    const at::Tensor& out);
