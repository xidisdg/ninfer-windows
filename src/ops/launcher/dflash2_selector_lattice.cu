// ninfer::ops::detail - dflash2 selector kernels and launches.
//
// Candidate selection (top-16 per draft position) and the candidate-selector
// lattice packing consumed by the host path-trace in the DFlash2 decode
// round. See ninfer/ops/dflash2_selector_lattice.h for the packed-row
// contract.

#include "ops/launcher/dflash2_selector_lattice.h"

#include "ninfer/ops/dflash2_selector_lattice.h"

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops::detail {
namespace {

// ---------------------------------------------------------------------------
// dflash2_select_candidates
//
// One block per token column; every thread keeps a private sorted top-16
// (value desc, lower id first) while scanning the vocab with a grid-stride
// loop; the per-thread lists merge in shared memory and are sorted once.
// ---------------------------------------------------------------------------

constexpr std::int32_t kSelThreads = 128;

// Monotonic float -> uint32 (negative < positive, NaN top, -inf bottom); the
// key keeps this order in the high 32 bits (so the descending sort ranks
// larger logits first) and inverts the token id in the low 32 bits so lower
// ids break ties.
__device__ unsigned float_key(float value) {
    const unsigned u = __float_as_uint(value);
    return (u & 0x80000000u) ? ~u : (u ^ 0x80000000u);
}

__device__ unsigned long long sel_key(float value, std::int32_t id) {
    return (static_cast<unsigned long long>(float_key(value)) << 32) |
           static_cast<unsigned>(0x7fffffffu - (id & 0x7fffffff));
}

__device__ float sel_key_value(unsigned long long key) {
    const unsigned fk = static_cast<unsigned>(key >> 32);
    // Exact inverse of float_key: positive floats were keyed as u ^ 0x80000000u
    // (sign bit set), negatives as ~u (sign bit clear).
    const unsigned u = (fk & 0x80000000u) ? (fk ^ 0x80000000u) : ~fk;
    return __uint_as_float(u);
}

__global__ void dflash2_select_candidates_kernel(const __nv_bfloat16* __restrict__ logits,
                                                 std::int32_t vocab, std::int32_t tokens,
                                                 std::int32_t* __restrict__ out_ids,
                                                 float* __restrict__ out_values) {
    constexpr std::int32_t k = kDFlash2SelectorTopK;
    (void)tokens;

    const std::int32_t t = blockIdx.x;
    const __nv_bfloat16* col = logits + static_cast<std::int64_t>(t) * vocab;

    // Per-thread private top-k (value desc, lower id first on ties).
    float my_val[k];
    std::int32_t my_id[k];
    std::int32_t n_real = 0;
#pragma unroll
    for (std::int32_t i = 0; i < k; ++i) {
        my_val[i] = __uint_as_float(0xff800000u);  // -inf
        my_id[i] = 0;
    }
    for (std::int32_t id = threadIdx.x; id < vocab; id += kSelThreads) {
        const float v = __bfloat162float(col[id]);
        if (v <= my_val[k - 1]) { continue; }  // cannot enter the top-k
        std::int32_t slot = k - 1;
        while (slot > 0 && v > my_val[slot - 1]) {
            my_val[slot] = my_val[slot - 1];
            my_id[slot] = my_id[slot - 1];
            --slot;
        }
        // The run is sorted desc: real entries occupy the prefix [0, n_real),
        // -inf sentinels the suffix. The insertion point is never past the
        // boundary, so slot == n_real is exactly the "append past the
        // prefix" case that extends it; slot < n_real shifts within the
        // real prefix (overflowing the first sentinel off the end).
        if (slot == n_real) { ++n_real; }
        my_val[slot] = v;
        my_id[slot] = id;
    }

    // Block pool: the per-thread runs merged (keys carry the total order, so
    // a max-elimination picks the winners in order).
    __shared__ unsigned long long pool_keys[kSelThreads * k];
    __shared__ std::int32_t pool_ids[kSelThreads * k];
#pragma unroll
    for (std::int32_t i = 0; i < k; ++i) {
        const std::int32_t base = threadIdx.x * k + i;
        if (i < n_real) {
            pool_keys[base] = sel_key(my_val[i], my_id[i]);
            pool_ids[base] = my_id[i];
        } else {
            pool_keys[base] = 0;  // sentinel slot: not a live candidate
            pool_ids[base] = 0;
        }
    }
    __syncthreads();

    __shared__ unsigned long long reduce_keys[kSelThreads];
    __shared__ std::int32_t reduce_ids[kSelThreads];
    __shared__ unsigned long long winner_key;

    // 16 rounds of block-max over the live pool; the winning entry is
    // recorded and eliminated. Each thread owns pool entries
    // [threadIdx.x*k, (threadIdx.x+1)*k), so the elimination (zeroing the
    // winner's unique key) touches only its owner's region — no cross-thread
    // writes, no race.
    for (std::int32_t winner = 0; winner < k; ++winner) {
        unsigned long long best_key = 0;
        std::int32_t best_id = 0;
        bool found = false;
        const std::int32_t base = threadIdx.x * k;
#pragma unroll
        for (std::int32_t i = 0; i < k; ++i) {
            const unsigned long long key = pool_keys[base + i];
            if (key != 0 && (!found || key > best_key)) {
                best_key = key;
                best_id = pool_ids[base + i];
                found = true;
            }
        }
        reduce_keys[threadIdx.x] = found ? best_key : 0;
        reduce_ids[threadIdx.x] = best_id;
        __syncthreads();

        for (std::int32_t n = kSelThreads; n > 1; n >>= 1) {
            if (static_cast<std::int32_t>(threadIdx.x) < n / 2) {
                const unsigned long long a = reduce_keys[threadIdx.x];
                const unsigned long long b = reduce_keys[threadIdx.x + n / 2];
                if (b > a) {
                    reduce_keys[threadIdx.x] = b;
                    reduce_ids[threadIdx.x] = reduce_ids[threadIdx.x + n / 2];
                }
            }
            __syncthreads();
        }

        if (threadIdx.x == 0) {
            winner_key = reduce_keys[0];
            if (reduce_keys[0] != 0) {
                out_ids[static_cast<std::int64_t>(t) * k + winner] = reduce_ids[0];
                out_values[static_cast<std::int64_t>(t) * k + winner] =
                    sel_key_value(reduce_keys[0]);
            } else {
                out_ids[static_cast<std::int64_t>(t) * k + winner] = 0;
                out_values[static_cast<std::int64_t>(t) * k + winner] = 0.0f;
            }
        }
        __syncthreads();

        // Parallel elimination: the owner thread zeroes the winner's key.
        const std::int32_t zbase = threadIdx.x * k;
#pragma unroll
        for (std::int32_t i = 0; i < k; ++i) {
            if (pool_keys[zbase + i] == winner_key) { pool_keys[zbase + i] = 0; }
        }
        __syncthreads();
    }
}

// ---------------------------------------------------------------------------
// dflash2_selector_lattice
//
// Zero-fill kernel: anchor columns (block position 0) and the padding region
// of every column run through one grid-stride pass BEFORE the score kernel
// so no two warps race on the same cells.
//
// Score kernel: one warp per (predecessor, token); the warp reduces the
// rank dimension of successor[s, r, t] . bf16(predecessor[p, r, t] *
// hidden_pos[r, t]) across its 32 lanes (rank = 32 lanes x 8 per lane),
// adds the successor's unary logit, and writes that predecessor's 16-score
// band of the packed row. The ids band is written by predecessor warp 0.
// ---------------------------------------------------------------------------

__global__ void dflash2_lattice_zero_kernel(float* out, std::int32_t packed_width, std::int32_t tokens,
                                            std::int32_t block_tokens) {
    const std::int32_t k2 = kDFlash2SelectorTopK + kDFlash2SelectorTopK * kDFlash2SelectorTopK;
    const std::int64_t n = static_cast<std::int64_t>(tokens) * packed_width;
    for (std::int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += static_cast<std::int64_t>(blockDim.x) * gridDim.x) {
        const std::int32_t t = static_cast<std::int32_t>(i / packed_width);
        const std::int32_t c = static_cast<std::int32_t>(i % packed_width);
        if (t % block_tokens != 0 && c < k2) { continue; }  // scored by the score kernel
        out[i] = 0.0f;
    }
}

__global__ void dflash2_lattice_scores_kernel(const __nv_bfloat16* __restrict__ hidden_pos,
                                              const __nv_bfloat16* __restrict__ successor,
                                              const __nv_bfloat16* __restrict__ predecessor,
                                              const std::int32_t* __restrict__ candidates,
                                              const float* __restrict__ unary, float* __restrict__ out,
                                              std::int32_t packed_width, std::int32_t tokens,
                                              std::int32_t block_tokens) {
    constexpr std::int32_t k = kDFlash2SelectorTopK;
    constexpr std::int32_t kLane = kDFlash2SelectorRank / 32;  // 8 rank elements per lane

    const std::int32_t t = blockIdx.y;
    if (t % block_tokens == 0) { return; }  // anchor column stays zero
    const std::int32_t p = static_cast<std::int32_t>(blockIdx.x);
    const unsigned lane = static_cast<unsigned>(threadIdx.x);

    const std::int32_t pbase = (t * k + p) * kDFlash2SelectorRank;
    __nv_bfloat16 prod[kLane];
#pragma unroll
    for (std::int32_t i = 0; i < kLane; ++i) {
        const std::int32_t r = lane + i * 32;
        prod[i] = __float2bfloat16(__bfloat162float(predecessor[pbase + r]) *
                                   __bfloat162float(hidden_pos[t * kDFlash2SelectorRank + r]));
    }

    float acc[k];
#pragma unroll
    for (std::int32_t s = 0; s < k; ++s) {
        const __nv_bfloat16* srow = successor + (static_cast<std::int64_t>(t * k + s)) * kDFlash2SelectorRank;
        float sum = 0.0f;
#pragma unroll
        for (std::int32_t i = 0; i < kLane; ++i) {
            const std::int32_t r = lane + i * 32;
            sum += __bfloat162float(srow[r]) * __bfloat162float(prod[i]);
        }
#pragma unroll
        for (unsigned mask = 16u; mask > 0u; mask >>= 1) { sum += __shfl_xor_sync(0xffffffffu, sum, mask); }
        // uniform across the warp; the candidate's own unary logit
        sum += unary[t * k + s];
        acc[s] = sum;
    }

    if (lane == 0) {
        const std::int64_t row = static_cast<std::int64_t>(t) * packed_width;
        for (std::int32_t s = 0; s < k; ++s) { out[row + k + p * k + s] = acc[s]; }
        if (p == 0) {
            for (std::int32_t s = 0; s < k; ++s) {
                out[row + s] = static_cast<float>(candidates[t * k + s]);
            }
        }
    }
}

} // namespace

void dflash2_select_candidates_launch(const Tensor& logits, Tensor& out_ids, Tensor& out_values,
                                      cudaStream_t stream) {
    const std::int32_t tokens = logits.ne[1];
    const std::int32_t vocab = logits.ne[0];
    dflash2_select_candidates_kernel<<<tokens, kSelThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(logits.data), vocab, tokens,
        static_cast<std::int32_t*>(out_ids.data), static_cast<float*>(out_values.data));
}

void dflash2_selector_lattice_launch(const Tensor& hidden_pos, const Tensor& successor,
                                     const Tensor& predecessor, const Tensor& candidates,
                                     const Tensor& unary, std::int32_t packed_width,
                                     std::int32_t block_tokens, Tensor& out, cudaStream_t stream) {
    const std::int32_t tokens = hidden_pos.ne[1];
    {
        const std::int64_t cells = static_cast<std::int64_t>(tokens) * packed_width;
        const std::uint32_t grid =
            static_cast<std::uint32_t>((cells + 255) / 256);
        dflash2_lattice_zero_kernel<<<grid, 256, 0, stream>>>(static_cast<float*>(out.data),
                                                              packed_width, tokens, block_tokens);
    }
    dim3 grid(kDFlash2SelectorTopK, tokens, 1);
    dflash2_lattice_scores_kernel<<<grid, 32, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(hidden_pos.data),
        static_cast<const __nv_bfloat16*>(successor.data),
        static_cast<const __nv_bfloat16*>(predecessor.data),
        static_cast<const std::int32_t*>(candidates.data),
        static_cast<const float*>(unary.data), static_cast<float*>(out.data), packed_width, tokens,
        block_tokens);
}

} // namespace ninfer::ops::detail