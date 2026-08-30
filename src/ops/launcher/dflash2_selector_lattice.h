#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

void dflash2_select_candidates_launch(const Tensor& logits, Tensor& out_ids, Tensor& out_values,
                                      cudaStream_t stream);

void dflash2_selector_lattice_launch(const Tensor& hidden_pos, const Tensor& successor,
                                     const Tensor& predecessor, const Tensor& candidates,
                                     const Tensor& unary, std::int32_t packed_width,
                                     std::int32_t block_tokens, Tensor& out, cudaStream_t stream);

} // namespace ninfer::ops::detail