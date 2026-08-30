// ninfer::ops - dflash2_dynamic_conv wrapper: public api validation and launcher dispatch.
#include "ninfer/ops/dflash2_dynamic_conv.h"

#include "ops/launcher/dflash2_dynamic_conv.h" // detail::dflash2_dynamic_conv_launch

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

std::int64_t numel_checked(const Tensor& t, const char* label) {
    bool has_zero = false;
    for (int d = 0; d < 4; ++d) {
        if (t.ne[d] < 0) {
            throw std::invalid_argument(std::string("dflash2_dynamic_conv: ") + label +
                                        " dimensions must be nonnegative");
        }
        if (t.ne[d] == 0) { has_zero = true; }
    }
    if (has_zero) { return 0; }

    std::int64_t total = 1;
    for (int d = 0; d < 4; ++d) {
        if (total > std::numeric_limits<std::int64_t>::max() / t.ne[d]) {
            throw std::overflow_error("dflash2_dynamic_conv: tensor size overflows int64");
        }
        total *= t.ne[d];
    }
    return total;
}

void require_token_shape(const Tensor& t, const char* label, std::int32_t rows, std::int32_t tokens) {
    if (t.ne[0] != rows || t.ne[1] != tokens || t.ne[2] != 1 || t.ne[3] != 1) {
        throw std::invalid_argument(std::string("dflash2_dynamic_conv: ") + label +
                                    " must have shape [rows, tokens]");
    }
}

void require_accessible(const Tensor& x, const Tensor& dynamic, const Tensor& base, const Tensor& out) {
    if (!x.is_contiguous() || !dynamic.is_contiguous() || !base.is_contiguous() ||
        !out.is_contiguous()) {
        throw std::invalid_argument("dflash2_dynamic_conv: all tensors must be contiguous");
    }
    if (x.data == nullptr || dynamic.data == nullptr || base.data == nullptr || out.data == nullptr) {
        throw std::invalid_argument("dflash2_dynamic_conv: all tensor data pointers must be non-null");
    }
}

} // namespace

void dflash2_dynamic_conv(const Tensor& x, const Tensor& dynamic, const Tensor& base, int side,
                          std::int32_t width, Tensor& out, cudaStream_t stream) {
    if (x.dtype != DType::BF16 || dynamic.dtype != DType::BF16 || base.dtype != DType::BF16 ||
        out.dtype != DType::BF16) {
        throw std::invalid_argument("dflash2_dynamic_conv: all tensors must be BF16");
    }

    if (x.ne[2] != 1 || x.ne[3] != 1) {
        throw std::invalid_argument("dflash2_dynamic_conv: x must have shape [hidden, tokens]");
    }
    const std::int32_t hidden = x.ne[0];
    const std::int32_t tokens = x.ne[1];
    if (hidden <= 0 || hidden % kDFlash2ConvGroupSize != 0 || tokens <= 0) {
        throw std::invalid_argument(
            "dflash2_dynamic_conv: hidden must be positive and a multiple of the group size");
    }
    if (width < 1 || tokens % width != 0) {
        throw std::invalid_argument(
            "dflash2_dynamic_conv: tokens must be a positive multiple of the block width");
    }
    if (side < 0 || side >= kDFlash2ConvKernel) {
        throw std::invalid_argument("dflash2_dynamic_conv: side must be 0 or 1");
    }

    const std::int32_t groups = hidden / kDFlash2ConvGroupSize;
    require_token_shape(dynamic, "dynamic", kDFlash2ConvKernel * kDFlash2ConvKernel * groups, tokens);
    require_token_shape(out, "out", hidden, tokens);
    if (numel_checked(base, "base") !=
        static_cast<std::int64_t>(kDFlash2ConvKernel * kDFlash2ConvKernel) * hidden) {
        throw std::invalid_argument(
            "dflash2_dynamic_conv: base must have kernel*kernel*hidden elements "
            "[side, tap, hidden] layout");
    }
    (void)numel_checked(x, "x");
    (void)numel_checked(dynamic, "dynamic");
    (void)numel_checked(out, "out");

    require_accessible(x, dynamic, base, out);
    if (out.data == x.data) {
        throw std::invalid_argument("dflash2_dynamic_conv: out must not alias x");
    }

    detail::dflash2_dynamic_conv_launch(x, dynamic, base, side, width, out, stream);
}

} // namespace ninfer::ops