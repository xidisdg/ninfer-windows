#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops {

// Fixed DFlash2 candidate-selector geometry (the 27B checkpoint): top-16
// candidates per draft position, 256-dimensional codebooks.
inline constexpr std::int32_t kDFlash2SelectorTopK = 16;
inline constexpr std::int32_t kDFlash2SelectorRank = 256;

/**
 * Op: DFlash2 candidate selection (top-16 of the draft logits).
 *
 * For each column t of `logits` (one draft position), selects the
 * kDFlash2SelectorTopK (16) best candidates:
 *
 *   out_ids[s, t]    = token id of the s-th candidate
 *   out_values[s, t] = its logit
 *
 * Candidates are sorted by value descending; a lower token id breaks ties.
 * `logits` is contiguous BF16 [vocab, T] (the draft output-head logits as
 * emitted by ops::linear; values are read in FP32). T > 0 and vocab > 16.
 * `out_ids` is I32 [16, T], `out_values` is FP32 [16, T] (converted on read).
 * Selection compares the BF16 values in FP32; lower token ids break ties, so
 * the result is deterministic for a given input.
 */
void dflash2_select_candidates(const Tensor& logits, Tensor& out_ids, Tensor& out_values,
                               cudaStream_t stream);

/**
 * Op: DFlash2 candidate-selector lattice.
 *
 * The DFlash2 drafter exposes no raw logits to the host. For every draft
 * position it packs one `packed_width`-wide FP32 row (the "lattice"); the
 * host later traces a path through the rows, position by position, picking
 * the next predecessor from the successor scores of the current one.
 *
 *   row[c, t], c in [0, top_k)                : candidate token id (as FP32),
 *                                               same order as `candidates`
 *   row[c, t], c in [top_k, top_k + top_k^2) : score of successor s =
 *                                               (c - top_k) % top_k conditioned
 *                                               on predecessor p =
 *                                               (c - top_k) / top_k
 *   row[c, t], c >= top_k + top_k^2           : zero padding
 *
 * `packed_width` must be at least top_k + top_k*top_k (272); the 27B caller
 * uses the hidden size (5120).
 *
 * Per draft token t (block position b = t % block_tokens; rows for b == 0
 * are zero-filled — the first block slot is the committed anchor token):
 *
 *   scores[p, s] = dot_r( successor[s, r, t], bf16(predecessor[p, r, t]
 *                 * hidden_pos[r, t]) ) + unary[s, t]
 *
 * with FP32 accumulation over the rank (256) dimension. The bf16(...) factor
 * reproduces the reference graph's element-wise BF16 product before the FP32
 * codebook dot product. `hidden_pos` is the selector hidden projection of
 * the draft *input* token embeddings (rank x hidden GEMM applied by the
 * caller via ops::linear on the [hidden, T] BF16 input-embedding tensor);
 * `successor`/`predecessor` are the two selector codebook tables gathered
 * on the candidate ids (the caller gathers via ops::embedding on a
 * [top_k, T]-flattened id list, then reshapes to [rank, top_k, T]). For the
 * first scored position (b == 1) the caller repeats the anchor token id
 * top_k times in the predecessor ids, which reproduces the reference's
 * broadcast of the single anchor codebook row.
 *
 * Logical shapes (all contiguous, dim0 fastest):
 *   hidden_pos:    [rank, T]       BF16
 *   successor:     [rank, top_k, T] BF16
 *   predecessor:   [rank, top_k, T] BF16
 *   candidates:    [top_k, T]      I32
 *   unary:         [top_k, T]      FP32
 *   out:           [packed_width, T] FP32
 *   T = block_tokens * batch, block_tokens the draft block size (e.g. 8).
 */
void dflash2_selector_lattice(const Tensor& hidden_pos, const Tensor& successor,
                              const Tensor& predecessor, const Tensor& candidates,
                              const Tensor& unary, std::int32_t packed_width,
                              std::int32_t block_tokens, Tensor& out, cudaStream_t stream);

} // namespace ninfer::ops