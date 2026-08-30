// ninfer::ops - dflash2_selector_lattice wrapper: public api validation and launcher dispatch.
#include "ninfer/ops/dflash2_selector_lattice.h"

#include "ops/launcher/dflash2_selector_lattice.h"

#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

constexpr std::int32_t kSelTopK = kDFlash2SelectorTopK;

void require_shape_2d(const Tensor& t, const char* label, std::int32_t rows, std::int32_t tokens,
                      DType dtype, const char* dtype_name) {
    if (t.dtype != dtype) {
        throw std::invalid_argument(std::string("dflash2_selector: ") + label + " must be " +
                                    dtype_name);
    }
    if (t.ne[0] != rows || t.ne[1] != tokens || t.ne[2] != 1 || t.ne[3] != 1) {
        throw std::invalid_argument(std::string("dflash2_selector: ") + label +
                                    " must have shape [rows, tokens]");
    }
}

void require_accessible(const Tensor& a, const Tensor& b, const Tensor& c, const Tensor& d) {
    if (!a.is_contiguous() || !b.is_contiguous() || !c.is_contiguous() || !d.is_contiguous()) {
        throw std::invalid_argument("dflash2_selector: tensors must be contiguous");
    }
    if (a.data == nullptr || b.data == nullptr || c.data == nullptr || d.data == nullptr) {
        throw std::invalid_argument("dflash2_selector: tensor data pointers must be non-null");
    }
}

} // namespace

void dflash2_select_candidates(const Tensor& logits, Tensor& out_ids, Tensor& out_values,
                               cudaStream_t stream) {
    if (logits.dtype != DType::BF16) {
        throw std::invalid_argument("dflash2_select_candidates: logits must be BF16");
    }
    if (out_ids.dtype != DType::I32 || out_values.dtype != DType::FP32) {
        throw std::invalid_argument(
            "dflash2_select_candidates: out_ids must be I32 and out_values FP32");
    }
    const std::int32_t vocab = logits.ne[0];
    const std::int32_t tokens = logits.ne[1];
    if (logits.ne[2] != 1 || logits.ne[3] != 1 || vocab <= kSelTopK || tokens <= 0) {
        throw std::invalid_argument(
            "dflash2_select_candidates: logits must be [vocab, T] with vocab > 16");
    }
    if (out_ids.ne[0] != kSelTopK || out_ids.ne[1] != tokens || out_ids.ne[2] != 1 ||
        out_ids.ne[3] != 1) {
        throw std::invalid_argument("dflash2_select_candidates: out_ids must be [16, T]");
    }
    if (out_values.ne[0] != kSelTopK || out_values.ne[1] != tokens || out_values.ne[2] != 1 ||
        out_values.ne[3] != 1) {
        throw std::invalid_argument("dflash2_select_candidates: out_values must be [16, T]");
    }
    if (!logits.is_contiguous() || !out_ids.is_contiguous() || !out_values.is_contiguous()) {
        throw std::invalid_argument("dflash2_select_candidates: tensors must be contiguous");
    }
    if (logits.data == nullptr || out_ids.data == nullptr || out_values.data == nullptr) {
        throw std::invalid_argument("dflash2_select_candidates: data pointers must be non-null");
    }

    detail::dflash2_select_candidates_launch(logits, out_ids, out_values, stream);
}

void dflash2_selector_lattice(const Tensor& hidden_pos, const Tensor& successor,
                              const Tensor& predecessor, const Tensor& candidates,
                              const Tensor& unary, std::int32_t packed_width,
                              std::int32_t block_tokens, Tensor& out, cudaStream_t stream) {
    if (hidden_pos.dtype != DType::BF16 || successor.dtype != DType::BF16 ||
        predecessor.dtype != DType::BF16) {
        throw std::invalid_argument(
            "dflash2_selector_lattice: hidden_pos/successor/predecessor must be BF16");
    }
    if (candidates.dtype != DType::I32 || unary.dtype != DType::FP32 || out.dtype != DType::FP32) {
        throw std::invalid_argument(
            "dflash2_selector_lattice: candidates must be I32, unary/out FP32");
    }

    const std::int32_t tokens = hidden_pos.ne[1];
    if (hidden_pos.ne[0] != kDFlash2SelectorRank || hidden_pos.ne[2] != 1 ||
        hidden_pos.ne[3] != 1 || tokens <= 0) {
        throw std::invalid_argument(
            "dflash2_selector_lattice: hidden_pos must be [rank, T] with T > 0");
    }
    if (successor.ne[0] != kDFlash2SelectorRank || successor.ne[1] != kSelTopK ||
        successor.ne[2] != tokens || successor.ne[3] != 1) {
        throw std::invalid_argument(
            "dflash2_selector_lattice: successor must be [rank, top_k, T]");
    }
    if (predecessor.ne[0] != kDFlash2SelectorRank || predecessor.ne[1] != kSelTopK ||
        predecessor.ne[2] != tokens || predecessor.ne[3] != 1) {
        throw std::invalid_argument(
            "dflash2_selector_lattice: predecessor must be [rank, top_k, T]");
    }
    if (candidates.ne[0] != kSelTopK || candidates.ne[1] != tokens || candidates.ne[2] != 1 ||
        candidates.ne[3] != 1) {
        throw std::invalid_argument("dflash2_selector_lattice: candidates must be [top_k, T]");
    }
    if (unary.ne[0] != kSelTopK || unary.ne[1] != tokens || unary.ne[2] != 1 || unary.ne[3] != 1) {
        throw std::invalid_argument("dflash2_selector_lattice: unary must be [top_k, T]");
    }
    const std::int32_t k = kSelTopK;
    if (packed_width < k + k * k) {
        throw std::invalid_argument(
            "dflash2_selector_lattice: packed_width must be at least top_k + top_k*top_k");
    }
    if (out.ne[0] != packed_width || out.ne[1] != tokens || out.ne[2] != 1 || out.ne[3] != 1) {
        throw std::invalid_argument(
            "dflash2_selector_lattice: out must be [packed_width, T]");
    }
    if (block_tokens < 2 || tokens % block_tokens != 0) {
        throw std::invalid_argument(
            "dflash2_selector_lattice: T must be a multiple of a block_tokens >= 2");
    }

    require_accessible(hidden_pos, successor, predecessor, candidates);
    if (!unary.is_contiguous() || !out.is_contiguous() || unary.data == nullptr ||
        out.data == nullptr) {
        throw std::invalid_argument("dflash2_selector_lattice: tensors must be contiguous");
    }

    detail::dflash2_selector_lattice_launch(hidden_pos, successor, predecessor, candidates, unary,
                                            packed_width, block_tokens, out, stream);
}

} // namespace ninfer::ops