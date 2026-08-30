// ninfer::ops - dflash2_predecessor_ids wrapper: public api validation and launcher dispatch.
#include "ninfer/ops/dflash2_predecessor_ids.h"

#include "ops/launcher/dflash2_predecessor_ids.h" // detail::dflash2_predecessor_ids_launch

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
            throw std::invalid_argument(std::string("dflash2_predecessor_ids: ") + label +
                                        " dimensions must be nonnegative");
        }
        if (t.ne[d] == 0) { has_zero = true; }
    }
    if (has_zero) { return 0; }

    std::int64_t total = 1;
    for (int d = 0; d < 4; ++d) {
        if (total > std::numeric_limits<std::int64_t>::max() / t.ne[d]) {
            throw std::overflow_error("dflash2_predecessor_ids: tensor size overflows int64");
        }
        total *= t.ne[d];
    }
    return total;
}

void require_2d(const Tensor& t, const char* label, std::int32_t rows, std::int32_t tokens) {
    if (t.ne[0] != rows || t.ne[1] != tokens || t.ne[2] != 1 || t.ne[3] != 1) {
        throw std::invalid_argument(std::string("dflash2_predecessor_ids: ") + label +
                                    " must have shape [rows, tokens]");
    }
}

} // namespace

void dflash2_predecessor_ids(const Tensor& candidate_ids, const Tensor& anchor_ids,
                             std::int32_t block_tokens, Tensor& out, cudaStream_t stream) {
    if (candidate_ids.dtype != DType::I32 || anchor_ids.dtype != DType::I32 ||
        out.dtype != DType::I32) {
        throw std::invalid_argument("dflash2_predecessor_ids: all tensors must be I32");
    }

    const std::int32_t top_k = candidate_ids.ne[0];
    const std::int32_t tokens = candidate_ids.ne[1];
    if (top_k <= 0 || tokens <= 0 || block_tokens < 2 || tokens % block_tokens != 0) {
        throw std::invalid_argument(
            "dflash2_predecessor_ids: candidate_ids must be [top_k, T] with T a multiple of "
            "block_tokens >= 2");
    }
    const std::int32_t batch = tokens / block_tokens;
    if (anchor_ids.ne[0] != batch || anchor_ids.ne[1] != 1 || anchor_ids.ne[2] != 1 ||
        anchor_ids.ne[3] != 1) {
        throw std::invalid_argument("dflash2_predecessor_ids: anchor_ids must be [batch]");
    }
    require_2d(out, "out", top_k, tokens);

    if (!candidate_ids.is_contiguous() || !anchor_ids.is_contiguous() || !out.is_contiguous()) {
        throw std::invalid_argument("dflash2_predecessor_ids: tensors must be contiguous");
    }
    if (candidate_ids.data == nullptr || anchor_ids.data == nullptr || out.data == nullptr) {
        throw std::invalid_argument("dflash2_predecessor_ids: tensor data pointers must be non-null");
    }

    (void)numel_checked(candidate_ids, "candidate_ids");
    (void)numel_checked(anchor_ids, "anchor_ids");
    (void)numel_checked(out, "out");

    detail::dflash2_predecessor_ids_launch(candidate_ids, anchor_ids, block_tokens, out, stream);
}

} // namespace ninfer::ops
