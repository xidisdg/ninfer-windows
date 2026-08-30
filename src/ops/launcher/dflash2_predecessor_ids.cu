// ninfer::ops::detail - dflash2_predecessor_ids CUDA kernel and launch.
//
// The DFlash2 selector lattice conditions each position's successor scores
// on the previous position's candidate list (the anchor token on the first
// scored position). This kernel builds the shifted id list the caller
// gathers the predecessor codebook over with a single ops::embedding call.

#include "ops/launcher/dflash2_predecessor_ids.h"

#include "ninfer/ops/dflash2_predecessor_ids.h"

#include <cstdint>

namespace ninfer::ops::detail {
namespace {

constexpr std::int32_t kThreads = 256;

__global__ void dflash2_predecessor_ids_kernel(const std::int32_t* __restrict__ candidate_ids,
                                               const std::int32_t* __restrict__ anchor_ids,
                                               std::int32_t* __restrict__ out, std::int32_t top_k,
                                               std::int32_t tokens, std::int32_t block_tokens) {
    const std::int64_t total = static_cast<std::int64_t>(top_k) * tokens;
    for (std::int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total;
         idx += static_cast<std::int64_t>(blockDim.x) * gridDim.x) {
        // Layout is dim0-fastest {top_k, tokens}: flat idx = slot + top_k * column.
        const std::int32_t slot = static_cast<std::int32_t>(idx % top_k);
        const std::int32_t t = static_cast<std::int32_t>(idx / top_k);
        const std::int32_t pos = t % block_tokens;
        if (pos == 0) {
            // Anchor column: the committed anchor token (never scored).
            out[idx] = anchor_ids[t / block_tokens];
        } else if (pos == 1) {
            // First scored position: the anchor broadcast.
            out[idx] = anchor_ids[t / block_tokens];
        } else {
            // One block column back in the same block (same predecessor slot).
            out[idx] = candidate_ids[idx - top_k];
        }
    }
}

} // namespace

void dflash2_predecessor_ids_launch(const Tensor& candidate_ids, const Tensor& anchor_ids,
                                    std::int32_t block_tokens, Tensor& out, cudaStream_t stream) {
    const std::int32_t top_k = candidate_ids.ne[0];
    const std::int32_t tokens = candidate_ids.ne[1];
    const std::uint32_t total = static_cast<std::uint32_t>(top_k) * static_cast<std::uint32_t>(tokens);
    const std::uint32_t grid = (total + kThreads - 1) / kThreads;
    dflash2_predecessor_ids_kernel<<<grid, kThreads, 0, stream>>>(
        static_cast<const std::int32_t*>(candidate_ids.data),
        static_cast<const std::int32_t*>(anchor_ids.data), static_cast<std::int32_t*>(out.data),
        top_k, tokens, block_tokens);
}

} // namespace ninfer::ops::detail
