// ninfer::ops::detail - dflash2_dynamic_conv CUDA kernel and launch.
//
// DFlash2 drafter: two-tap (kernel=2) grouped depthwise causal 1-D convolution
// whose per-group coefficients are the sum of a static base kernel and a
// dynamic low-rank projection of the input, applied block-locally (the causal
// shift resets at every draft-block boundary, block = `width` tokens).

#include "ops/launcher/dflash2_dynamic_conv.h"

#include "ninfer/ops/dflash2_dynamic_conv.h"

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops::detail {
namespace {

constexpr std::int32_t kThreads = 256;

__global__ void dflash2_dynamic_conv_kernel(const __nv_bfloat16* __restrict__ x,
                                            const __nv_bfloat16* __restrict__ dynamic,
                                            const __nv_bfloat16* __restrict__ base,
                                            __nv_bfloat16* __restrict__ out, std::int32_t hidden,
                                            std::int32_t tokens, std::int32_t width,
                                            std::int32_t side) {
    constexpr std::int32_t kernel = kDFlash2ConvKernel;        // 2
    constexpr std::int32_t group_size = kDFlash2ConvGroupSize; // 16
    const std::int32_t group_count = hidden / group_size;     // G = 320

    // Layouts are dim0-fastest throughout: x/out are [hidden, tokens] with the
    // per-token hidden vector contiguous (offset(c,t) = c + hidden*t), and
    // dynamic is [2*kernel*G, tokens] with offset(r,t) = r + rows*t.
    //
    // Dynamic row for (group g, tap, side): g + G * (tap + kernel * side).
    const std::int32_t dyn_row0 = side * (kernel * group_count); // tap 0
    const std::int32_t dyn_row1 = group_count + dyn_row0;        // tap 1
    const std::int32_t dyn_rows = kernel * 2 * group_count;
    // Base row for (tap, side): base is laid out [side, tap, hidden] in memory
    // (channel innermost), so row = (side * kernel + tap) * hidden.
    const std::int32_t base_off0 = side * (kernel * hidden);
    const std::int32_t base_off1 = hidden + base_off0;

    const std::int64_t total = static_cast<std::int64_t>(hidden) * tokens;
    for (std::int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total;
         idx += static_cast<std::int64_t>(blockDim.x) * gridDim.x) {
        const std::int32_t c = static_cast<std::int32_t>(idx % hidden);
        const std::int32_t t = static_cast<std::int32_t>(idx / hidden);
        const std::int32_t g = c / group_size;

        const float x0 = __bfloat162float(x[idx]);
        const float x_prev =
            (t % width) >= 1 ? __bfloat162float(x[idx - static_cast<std::int64_t>(hidden)]) : 0.0f;

        const std::size_t dyn0 =
            static_cast<std::size_t>(dyn_row0) + static_cast<std::size_t>(dyn_rows) * t;
        const std::size_t dyn1 =
            static_cast<std::size_t>(dyn_row1) + static_cast<std::size_t>(dyn_rows) * t;
        const float w0 = __bfloat162float(dynamic[dyn0 + g]) + __bfloat162float(base[base_off0 + c]);
        const float w1 = __bfloat162float(dynamic[dyn1 + g]) + __bfloat162float(base[base_off1 + c]);

        out[idx] = __float2bfloat16(w0 * x0 + w1 * x_prev);
    }
}

} // namespace

void dflash2_dynamic_conv_launch(const Tensor& x, const Tensor& dynamic, const Tensor& base, int side,
                                 std::int32_t width, Tensor& out, cudaStream_t stream) {
    const std::int32_t hidden = x.ne[0];
    const std::int32_t tokens = x.ne[1];
    const std::uint32_t total = static_cast<std::uint32_t>(hidden) * static_cast<std::uint32_t>(tokens);
    const std::uint32_t grid = (total + kThreads - 1) / kThreads;
    dflash2_dynamic_conv_kernel<<<grid, kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const __nv_bfloat16*>(dynamic.data),
        static_cast<const __nv_bfloat16*>(base.data), static_cast<__nv_bfloat16*>(out.data), hidden,
        tokens, width, side);
}

} // namespace ninfer::ops::detail