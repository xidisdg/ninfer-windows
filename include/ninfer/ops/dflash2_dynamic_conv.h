#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops {

// Fixed DFlash2 dynamic-convolution geometry: a two-tap (kernel=2) grouped
// depthwise causal convolution, computed block-locally (the causal shift resets
// at the start of every draft block of `width` tokens).
inline constexpr std::int32_t kDFlash2ConvKernel    = 2;
inline constexpr std::int32_t kDFlash2ConvGroupSize = 16;

/**
 * Op: DFlash2 two-tap dynamic (input-dependent) grouped depthwise causal conv.
 *
 * x, dynamic, out are contiguous BF16. `width` is the draft block size (tokens
 * per sequence); T = x.ne[1] = width * batch. The conv has kernel kDFlash2ConvKernel
 * (2 taps) and group size kDFlash2ConvGroupSize (16), so hidden must be a
 * multiple of 16 and G = hidden / 16 groups.
 *
 *   x      : [hidden, T]  the (already projected) input, per-token
 *   dynamic: [2*kernel*G, T]  the per-token dynamic coefficients, the output of a
 *             [2*kernel*G, hidden] projection of x. Row r = g + G*(tap + kernel*side)
 *             holds group g's coefficient for (tap, side).
 *   base   : [2, kernel, hidden]  the static base kernel (side outermost, then tap,
 *             then channel), checkpoint layout, shared by both sides' taps.
 *   out    : [hidden, T]  result, same layout as x.
 *   side   : 0 (input side) or 1 (output side) selects which base slice and which
 *             dynamic rows (side 0 -> rows g + G*tap ; side 1 -> g + G*(tap + kernel)).
 *   width  : block size; the tap-1 term reads x[., t-1] within the same block and is
 *             zero for the first token of each block (t % width == 0).
 *
 * For channel c (group g = c / 16) and token t (in-block position j = t % width):
 *
 *   out[c,t] = (dyn[g+G*0+G*k*side, t] + base[side,0,c]) * x[c,t]
 *            + (dyn[g+G*1+G*k*side, t] + base[side,1,c]) * (j>=1 ? x[c,t-1] : 0)
 *
 * Accumulated in FP32, stored BF16. x and out must not alias (out is completely
 * overwritten); no caller workspace is used. This is the DFlash2 drafter's
 * per-layer attention/MLP dynamic convolution (see the qwen3.8-27b DFlash2
 * artifact); it runs on every draft layer's attention-in/out and mlp-in/out.
 */
void dflash2_dynamic_conv(const Tensor& x, const Tensor& dynamic, const Tensor& base, int side,
                          std::int32_t width, Tensor& out, cudaStream_t stream);

} // namespace ninfer::ops