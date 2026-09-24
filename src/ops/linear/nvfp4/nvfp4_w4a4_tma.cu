#include "ops/linear/nvfp4/nvfp4_w4a4_tma_launch.h"

#include "core/device.h"
#include "ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_output.cuh"
#include "ops/linear/nvfp4/nvfp4_config.h"
#include "ops/linear/nvfp4/nvfp4_w4a4_mma.cuh"
#include "ops/linear/nvfp4/nvfp4_w4a4_tma.cuh"
#include "ops/linear_add/nvfp4/nvfp4_linear_add_epilogue.cuh"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

namespace ninfer::ops::detail {
namespace {

using TmaM256N128 = Nvfp4W4a4TmaSchedule<256, 3, 1>;
// K128 consumes 64 code bytes per row. Prefetch the adjacent half-line for the next K tile.
using TmaM256N128Prefetch128B = Nvfp4W4a4TmaSchedule<256, 3, 1, CU_TENSOR_MAP_L2_PROMOTION_L2_128B>;

constexpr std::int32_t kQueryRows  = 6144;
constexpr std::int32_t kKeyRows    = 1024;
constexpr std::int32_t kGateRows   = 6144;
constexpr std::int32_t kKeyBegin   = kQueryRows;
constexpr std::int32_t kGateBegin  = kKeyBegin + kKeyRows;
constexpr std::int32_t kValueBegin = kGateBegin + kGateRows;

struct AttentionOutput {
    __nv_bfloat16* query;
    __nv_bfloat16* key;
    __nv_bfloat16* gate;
    __nv_bfloat16* value;

    __device__ __forceinline__ __nv_bfloat16* destination(std::int32_t parent_row,
                                                          std::int32_t token) const {
        if (parent_row < kKeyBegin) {
            return query + static_cast<std::int64_t>(token) * kQueryRows + parent_row;
        }
        if (parent_row < kGateBegin) {
            return key + static_cast<std::int64_t>(token) * kKeyRows + parent_row - kKeyBegin;
        }
        if (parent_row < kValueBegin) {
            return gate + static_cast<std::int64_t>(token) * kGateRows + parent_row - kGateBegin;
        }
        return value + static_cast<std::int64_t>(token) * kKeyRows + parent_row - kValueBegin;
    }

    __device__ __forceinline__ void store_vector(std::int32_t parent_row, std::int32_t token,
                                                 uint4 values) const {
        store_vec(destination(parent_row, token), values);
    }
};

static_assert((kQueryRows % TmaM256N128::kBlockN) == 0);
static_assert((kKeyRows % TmaM256N128::kBlockN) == 0);
static_assert((kGateRows % TmaM256N128::kBlockN) == 0);

#ifdef _WIN32
// MSVC cannot pass an alignas(128) struct by value as a kernel parameter (C2719), so the
// kernel takes a pointer to a device descriptor block. The H2D copy of that block is
// captured as a CUDA Graph node whose host source is re-read on every replay, so the
// source must outlive the graph: a bounded pinned-host cache keeps distinct descriptor
// bytes alive for the process (one entry per geometry/shape combination).
struct PinnedDescriptorSource {
    std::array<std::uint8_t, sizeof(Nvfp4W4a4TmaDescriptors)> bytes{};
    void* host = nullptr;
};

struct PinnedDescriptorCache {
    // Pinned buffers are allocated up front, before any stream capture can run:
    // cudaHostAlloc is not permitted while a stream is capturing, and the first descriptor
    // upload of a captured graph happens inside the capture.
    static constexpr std::size_t kPinnedPoolSize = 64;

    std::vector<void*> pinned_pool;
    std::vector<PinnedDescriptorSource> entries;

    PinnedDescriptorCache() {
        pinned_pool.reserve(kPinnedPoolSize);
        for (std::size_t i = 0; i < kPinnedPoolSize; ++i) {
            void* host = nullptr;
            CUDA_CHECK(cudaHostAlloc(&host, sizeof(Nvfp4W4a4TmaDescriptors),
                                     cudaHostAllocDefault));
            pinned_pool.push_back(host);
        }
    }

    // Entries are grow-only: a freed pinned buffer could still be the host source of a
    // captured graph's H2D copy node, so entries are never released while the process runs.
    void* source_for(const Nvfp4W4a4TmaDescriptors& descriptor) {
        const auto* bytes = reinterpret_cast<const std::uint8_t*>(&descriptor);
        for (auto& entry : entries) {
            if (std::equal(bytes, bytes + entry.bytes.size(), entry.bytes.data())) {
                return entry.host;
            }
        }
        PinnedDescriptorSource entry;
        std::copy_n(bytes, entry.bytes.size(), entry.bytes.begin());
        if (!pinned_pool.empty()) {
            entry.host = pinned_pool.back();
            pinned_pool.pop_back();
        } else {
            CUDA_CHECK(cudaHostAlloc(&entry.host, sizeof(Nvfp4W4a4TmaDescriptors),
                                     cudaHostAllocDefault));
        }
        std::memcpy(entry.host, bytes, sizeof(Nvfp4W4a4TmaDescriptors));
        entries.push_back(std::move(entry));
        return entries.back().host;
    }
};

PinnedDescriptorCache pinned_descriptor_cache;

struct Nvfp4TmaDescriptorBlock {
    Nvfp4W4a4TmaDescriptors* device = nullptr;
    cudaStream_t stream             = nullptr;

    explicit Nvfp4TmaDescriptorBlock(cudaStream_t stream) : stream(stream) {
        CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&device),
                                   sizeof(Nvfp4W4a4TmaDescriptors), stream));
    }

    Nvfp4TmaDescriptorBlock(const Nvfp4TmaDescriptorBlock&)            = delete;
    Nvfp4TmaDescriptorBlock& operator=(const Nvfp4TmaDescriptorBlock&) = delete;

    ~Nvfp4TmaDescriptorBlock() {
        if (device == nullptr) { return; }
        CUDA_CHECK(cudaFreeAsync(device, stream));
    }
};
#endif

template <class Geometry, class Schedule, class Epilogue, class Output>
void launch_tma(const std::uint8_t* activation_codes, const std::uint8_t* activation_scales,
                const std::uint8_t* weight_codes, const std::uint8_t* weight_scales,
                std::int32_t tokens, float alpha, Epilogue epilogue, Output output,
                cudaStream_t stream) {
    const Nvfp4W4a4TmaDescriptors descriptors =
        make_nvfp4_w4a4_tma_descriptors<Geometry, Schedule::kBlockM>(
            activation_codes, activation_scales, weight_codes, weight_scales, tokens,
            Schedule::kWeightCodePromotion);
    constexpr std::size_t kSharedBytes = sizeof(Nvfp4W4a4TmaSharedStorage<Schedule>);
    static const bool kConfigured      = [] {
        CUDA_CHECK(cudaFuncSetAttribute(nvfp4_w4a4_tma_kernel<Geometry, Schedule, Epilogue, Output>,
                                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                                             static_cast<int>(kSharedBytes)));
        return true;
    }();
    (void)kConfigured;

    const dim3 grid(Geometry::kOutputRows / Schedule::kBlockN, tokens / Schedule::kBlockM);
#ifdef _WIN32
    void* descriptor_host = pinned_descriptor_cache.source_for(descriptors);
    Nvfp4TmaDescriptorBlock block(stream);
    CUDA_CHECK(cudaMemcpyAsync(block.device, descriptor_host, sizeof(descriptors),
                               cudaMemcpyHostToDevice, stream));
    nvfp4_w4a4_tma_kernel<Geometry, Schedule, Epilogue, Output>
        <<<grid, Schedule::kThreads, kSharedBytes, stream>>>(block.device, alpha, epilogue, output);
#else
    nvfp4_w4a4_tma_kernel<Geometry, Schedule, Epilogue, Output>
        <<<grid, Schedule::kThreads, kSharedBytes, stream>>>(descriptors, alpha, epilogue, output);
#endif
    CUDA_CHECK(cudaGetLastError());
}

template <class Geometry, class Schedule = TmaM256N128>
void launch_linear(const std::uint8_t* activation_codes, const std::uint8_t* activation_scales,
                   const std::uint8_t* weight_codes, const std::uint8_t* weight_scales,
                   __nv_bfloat16* output, std::int32_t tokens, float alpha, cudaStream_t stream) {
    launch_tma<Geometry, Schedule>(activation_codes, activation_scales, weight_codes, weight_scales,
                                   tokens, alpha, Nvfp4IdentityEpilogue{},
                                   Nvfp4ContiguousOutput{output, Geometry::kOutputRows}, stream);
}

} // namespace

void launch_nvfp4_w4a4_tma_linear(Nvfp4GeometryId problem, const std::uint8_t* activation_codes,
                                  const std::uint8_t* activation_scales,
                                  const std::uint8_t* weight_codes, const std::uint8_t* weight_scales,
                                  __nv_bfloat16* output,
                                  std::int32_t tokens, float alpha, cudaStream_t stream) {
    switch (problem) {
    case Nvfp4GeometryId::N14336K5120:
        launch_linear<Nvfp4N14336K5120>(activation_codes, activation_scales, weight_codes,
                                        weight_scales, output, tokens, alpha, stream);
        return;
    case Nvfp4GeometryId::N16384K5120:
        launch_linear<Nvfp4N16384K5120>(activation_codes, activation_scales, weight_codes,
                                        weight_scales, output, tokens, alpha, stream);
        return;
    case Nvfp4GeometryId::N34816K5120:
        launch_linear<Nvfp4N34816K5120, TmaM256N128Prefetch128B>(
            activation_codes, activation_scales, weight_codes, weight_scales, output, tokens, alpha,
            stream);
        return;
    case Nvfp4GeometryId::N5120K6144:
        launch_linear<Nvfp4N5120K6144>(activation_codes, activation_scales, weight_codes,
                                       weight_scales, output, tokens, alpha, stream);
        return;
    case Nvfp4GeometryId::N5120K17408:
        launch_linear<Nvfp4N5120K17408>(activation_codes, activation_scales, weight_codes,
                                        weight_scales, output, tokens, alpha, stream);
        return;
    }
}

void launch_nvfp4_w4a4_tma_attention(const std::uint8_t* activation_codes,
                                     const std::uint8_t* activation_scales,
                                     const std::uint8_t* weight_codes,
                                     const std::uint8_t* weight_scales, __nv_bfloat16* query,
                                     __nv_bfloat16* gate, __nv_bfloat16* key, __nv_bfloat16* value,
                                     std::int32_t tokens, float alpha, cudaStream_t stream) {
    launch_tma<Nvfp4N14336K5120, TmaM256N128>(activation_codes, activation_scales, weight_codes,
                                              weight_scales, tokens, alpha, Nvfp4IdentityEpilogue{},
                                              AttentionOutput{query, key, gate, value}, stream);
}

void launch_nvfp4_w4a4_tma_gdn(const std::uint8_t* activation_codes,
                               const std::uint8_t* activation_scales,
                               const std::uint8_t* weight_codes, const std::uint8_t* weight_scales,
                               __nv_bfloat16* qkv, __nv_bfloat16* z, std::int32_t tokens,
                               float alpha, cudaStream_t stream) {
    launch_tma<Nvfp4N16384K5120, TmaM256N128>(activation_codes, activation_scales, weight_codes,
                                              weight_scales, tokens, alpha, Nvfp4IdentityEpilogue{},
                                              Nvfp4GdnInputOutput{qkv, z}, stream);
}

template <class Geometry>
void launch_linear_add(const std::uint8_t* activation_codes, const std::uint8_t* activation_scales,
                       const std::uint8_t* weight_codes, const std::uint8_t* weight_scales,
                       __nv_bfloat16* residual, std::int32_t tokens, float alpha,
                       cudaStream_t stream) {
    launch_tma<Geometry, TmaM256N128>(
        activation_codes, activation_scales, weight_codes, weight_scales, tokens, alpha,
        Nvfp4AddResidualEpilogue{residual, Geometry::kOutputRows},
        Nvfp4ContiguousOutput{residual, Geometry::kOutputRows}, stream);
}

void launch_nvfp4_w4a4_tma_linear_add(Nvfp4GeometryId problem, const std::uint8_t* activation_codes,
                                      const std::uint8_t* activation_scales,
                                      const std::uint8_t* weight_codes,
                                      const std::uint8_t* weight_scales, __nv_bfloat16* residual,
                                      std::int32_t tokens, float alpha, cudaStream_t stream) {
    switch (problem) {
    case Nvfp4GeometryId::N5120K6144:
        launch_linear_add<Nvfp4N5120K6144>(activation_codes, activation_scales, weight_codes,
                                           weight_scales, residual, tokens, alpha, stream);
        return;
    case Nvfp4GeometryId::N5120K17408:
        launch_linear_add<Nvfp4N5120K17408>(activation_codes, activation_scales, weight_codes,
                                            weight_scales, residual, tokens, alpha, stream);
        return;
    case Nvfp4GeometryId::N14336K5120:
    case Nvfp4GeometryId::N16384K5120:
    case Nvfp4GeometryId::N34816K5120:
        return;
    }
}

} // namespace ninfer::ops::detail