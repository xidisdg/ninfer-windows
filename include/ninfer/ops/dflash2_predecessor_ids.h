#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops {

/**
 * Op: DFlash2 predecessor candidate ids — the per-position candidate lists
 * shifted one block column, with the anchor token broadcast on the first
 * scored positions.
 *
 * The selector lattice scores position pos's successors conditioned on the
 * *predecessor* candidates:
 *
 *   out[k, t], t = b * block_tokens + pos
 *     pos == 0 : anchor_ids[b]            (the zeroed anchor column)
 *     pos == 1 : anchor_ids[b]            (the anchor broadcast)
 *     pos >= 2 : candidate_ids[k, t - 1]  (the previous block column)
 *
 * Logical shapes (all contiguous I32, dim0 fastest):
 *   candidate_ids: [top_k, T]   T = block_tokens * batch
 *   anchor_ids   : [batch]      one committed anchor token per block
 *   out          : [top_k, T]
 *
 * The anchor column (pos == 0) is never scored (the lattice zero-fills its
 * rows); its ids exist so the caller can gather the predecessor codebook
 * over the whole [top_k, T] id list with a single ops::embedding call,
 * matching the reference graph's per-position gather (anchor at pos 1,
 * candidate_ids[pos - 1] beyond).
 */
void dflash2_predecessor_ids(const Tensor& candidate_ids, const Tensor& anchor_ids,
                             std::int32_t block_tokens, Tensor& out, cudaStream_t stream);

} // namespace ninfer::ops
