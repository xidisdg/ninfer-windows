#include "ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_w4a4_tma_launch.h"

#include "core/device.h"
#include "ops/linear/nvfp4/nvfp4_config.h"
#include "ops/linear/nvfp4/nvfp4_w4a4_tma.cuh"
#include "ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_w4a4_tma.cuh"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace ninfer::ops::detail {
namespace {

using M256N128S3 = Nvfp4W4a4TmaSchedule<256, 3, 1>;

template <class Geometry, class Schedule>
Nvfp4W4a4TmaDescriptors make_descriptors(const std::uint8_t* activation_codes,
                                         const std::uint8_t* activation_scales,
                                         const std::uint8_t* weight_codes,
                                         const std::uint8_t* weight_scales, std::int32_t tokens) {
    constexpr std::uint32_t kCodeColumns  = 64;
    constexpr std::uint32_t kScaleColumns = 16;
    constexpr std::uint32_t kPairN        = Schedule::kBlockN / 2;
    constexpr std::uint64_t kWeightScaleBytes =
        static_cast<std::uint64_t>(Geometry::kOutputRows) * Geometry::kInputRows / 16;

    Nvfp4W4a4TmaDescriptors descriptors{};
    descriptors.a_codes = nvfp4_make_tma_2d(
        const_cast<std::uint8_t*>(activation_codes), CU_TENSOR_MAP_DATA_TYPE_UINT8,
        Geometry::kCodeBytesPerRow, tokens, Geometry::kCodeBytesPerRow, kCodeColumns,
        Schedule::kBlockM, CU_TENSOR_MAP_SWIZZLE_64B, "encode LinearSwiGLU activation codes TMA");
    descriptors.b_codes = nvfp4_make_tma_2d(
        const_cast<std::uint8_t*>(weight_codes), CU_TENSOR_MAP_DATA_TYPE_UINT8,
        Geometry::kCodeBytesPerRow, Geometry::kOutputRows, Geometry::kCodeBytesPerRow, kCodeColumns,
        kPairN, CU_TENSOR_MAP_SWIZZLE_64B, "encode LinearSwiGLU weight codes TMA");
    descriptors.a_scales = nvfp4_make_tma_2d(
        const_cast<std::uint8_t*>(activation_scales), CU_TENSOR_MAP_DATA_TYPE_UINT8,
        Geometry::kGroupsPerRow, tokens, Geometry::kGroupsPerRow, kScaleColumns, Schedule::kBlockM,
        CU_TENSOR_MAP_SWIZZLE_NONE, "encode LinearSwiGLU activation scales TMA");
    descriptors.b_scales =
        nvfp4_make_tma_2d(const_cast<std::uint8_t*>(weight_scales), CU_TENSOR_MAP_DATA_TYPE_UINT8,
                          16, kWeightScaleBytes / 16, 16, 16, 64, CU_TENSOR_MAP_SWIZZLE_NONE,
                          "encode LinearSwiGLU weight scales TMA");
    return descriptors;
}

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

struct Nvfp4LinearSwiGluTmaDescriptorBlock {
    Nvfp4W4a4TmaDescriptors* device = nullptr;
    cudaStream_t stream             = nullptr;

    explicit Nvfp4LinearSwiGluTmaDescriptorBlock(cudaStream_t stream) : stream(stream) {
        CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&device),
                                   sizeof(Nvfp4W4a4TmaDescriptors), stream));
    }

    Nvfp4LinearSwiGluTmaDescriptorBlock(const Nvfp4LinearSwiGluTmaDescriptorBlock&) = delete;
    Nvfp4LinearSwiGluTmaDescriptorBlock&
    operator=(const Nvfp4LinearSwiGluTmaDescriptorBlock&) = delete;

    ~Nvfp4LinearSwiGluTmaDescriptorBlock() {
        if (device == nullptr) { return; }
        CUDA_CHECK(cudaFreeAsync(device, stream));
    }
};
#endif

} // namespace

void launch_nvfp4_linear_swiglu_w4a4_tma(const std::uint8_t* activation_codes,
                                         const std::uint8_t* activation_scales,
                                         const std::uint8_t* weight_codes,
                                         const std::uint8_t* weight_scales, __nv_bfloat16* output,
                                         std::int32_t tokens, float alpha, cudaStream_t stream) {
    if (tokens < M256N128S3::kBlockM || (tokens % M256N128S3::kBlockM) != 0) {
        throw std::invalid_argument(
            "nvfp4 LinearSwiGLU TMA requires a positive M256 full-tile token count");
    }

    using Geometry                     = Nvfp4N34816K5120;
    constexpr std::size_t kSharedBytes = sizeof(Nvfp4LinearSwiGluTmaSharedStorage<M256N128S3>);
    static const bool kConfigured      = [] {
        CUDA_CHECK(cudaFuncSetAttribute(nvfp4_linear_swiglu_w4a4_tma_kernel<Geometry, M256N128S3>,
                                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                                             static_cast<int>(kSharedBytes)));
        return true;
    }();
    (void)kConfigured;

    const Nvfp4W4a4TmaDescriptors descriptors = make_descriptors<Geometry, M256N128S3>(
        activation_codes, activation_scales, weight_codes, weight_scales, tokens);
    constexpr int kPairN = M256N128S3::kBlockN / 2;
    const dim3 grid((Geometry::kOutputRows / 2) / kPairN, tokens / M256N128S3::kBlockM);
#ifdef _WIN32
    void* descriptor_host = pinned_descriptor_cache.source_for(descriptors);
    Nvfp4LinearSwiGluTmaDescriptorBlock block(stream);
    CUDA_CHECK(cudaMemcpyAsync(block.device, descriptor_host, sizeof(descriptors),
                               cudaMemcpyHostToDevice, stream));
    nvfp4_linear_swiglu_w4a4_tma_kernel<Geometry, M256N128S3>
        <<<grid, M256N128S3::kThreads, kSharedBytes, stream>>>(block.device, alpha, output);
#else
    nvfp4_linear_swiglu_w4a4_tma_kernel<Geometry, M256N128S3>
        <<<grid, M256N128S3::kThreads, kSharedBytes, stream>>>(descriptors, alpha, output);
#endif
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail