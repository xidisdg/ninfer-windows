#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

// Launches the DFlash2 predecessor-id block shift. See
// ninfer/ops/dflash2_predecessor_ids.h for the full contract.
void dflash2_predecessor_ids_launch(const Tensor& candidate_ids, const Tensor& anchor_ids,
                                    std::int32_t block_tokens, Tensor& out, cudaStream_t stream);

} // namespace ninfer::ops::detail
