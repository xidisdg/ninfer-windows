#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

// Launches the DFlash2 two-tap dynamic grouped depthwise causal conv (one
// side). See ninfer/ops/dflash2_dynamic_conv.h for the full contract.
void dflash2_dynamic_conv_launch(const Tensor& x, const Tensor& dynamic, const Tensor& base, int side,
                                 std::int32_t width, Tensor& out, cudaStream_t stream);

} // namespace ninfer::ops::detail