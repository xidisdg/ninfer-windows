#pragma once

#include <ninfer/targets/qwen3_6/startup_features.h>
#include <ninfer/targets/qwen3_6/vision.h>

#include "core/tensor.h"

#include <array>
#include <cstddef>
#include <optional>

namespace ninfer {

class DeviceArena;

namespace targets::qwen3_6 {

template <class ProjectionPayload, class PostMixerPayload>
struct FullAttentionWeights {
    Tensor input_norm;
    ProjectionPayload projection;
    Tensor query_norm;
    Tensor key_norm;
    Weight output;
    Tensor post_attention_norm;
    PostMixerPayload post_mixer;
};

template <class ProjectionPayload, class PostMixerPayload>
struct GdnWeights {
    Tensor input_norm;
    ProjectionPayload projection;
    Tensor convolution;
    Tensor norm;
    Weight output;
    Tensor post_attention_norm;
    PostMixerPayload post_mixer;
};

template <class AttentionPayload, class PostMixerPayload>
struct MtpWeights {
    Weight input_projection;
    Tensor embedding_norm;
    Tensor hidden_norm;
    Tensor input_norm;
    AttentionPayload attention;
    Tensor query_norm;
    Tensor key_norm;
    Weight output;
    Tensor post_attention_norm;
    PostMixerPayload post_mixer;
    Tensor final_norm;
};

struct OptimizedProposalWeights {
    Weight head;
    Tensor token_ids;
};

struct DFlashLayerWeights {
    Tensor input_norm;
    Weight query_key_value;
    Weight context_key;
    Weight context_value;
    Tensor query_norm;
    Tensor key_norm;
    Weight attention_output;
    Tensor post_attention_norm;
    Weight gate_up;
    Weight down;
};

template <std::size_t Layers>
struct DFlashWeights {
    Weight feature_projection;
    Tensor context_norm;
    std::array<DFlashLayerWeights, Layers> layers;
    Tensor final_norm;
};

// DFlash2 (block-diffusion drafter v2) per-layer weights. Inherits the DFlash1 layer
// shape (a v2 layer IS a v1 layer: same norms/projections, so the shared runtime's
// v1-shaped dflash code compiles unchanged against the v2 payload); v2 layers
// additionally run two-tap dynamic convolutions on the attention and MLP sublayers.
// The convolution base kernels are BF16 `[side, tap, hidden]` (checkpoint layout,
// kept as-is by the converter); the kernel projections are W8 rows
// `[2 * kernel * (hidden / group), hidden]` (row = group + group_count * (tap + kernel * side)).
struct DFlash2LayerWeights : DFlashLayerWeights {
    Tensor attention_conv_base;
    Weight attention_conv_projection;
    Tensor mlp_conv_base;
    Weight mlp_conv_projection;
};

// DFlash2 drafter payload: all-sliding-window layers, no private output head (the
// target head is aliased, as in DFlash1), plus the candidate selector codebooks
// ([vocab, rank]) and hidden projection ([rank, hidden]) consumed by the on-device
// lattice build.
template <std::size_t Layers>
struct DFlash2Weights {
    Weight feature_projection;
    Tensor context_norm;
    std::array<DFlash2LayerWeights, Layers> layers;
    Tensor final_norm;
    Weight selector_predecessor_codebook;
    Weight selector_successor_codebook;
    Weight selector_hidden_projection;
};

template <class FullProjectionPayload, class GdnProjectionPayload, class MainPostMixerPayload,
          class MtpAttentionPayload, class MtpPostMixerPayload, class DFlashPayload,
          std::size_t FullAttentionLayers, std::size_t GdnLayers>
struct ModelView {
    using FullLayer = FullAttentionWeights<FullProjectionPayload, MainPostMixerPayload>;
    using GdnLayer  = GdnWeights<GdnProjectionPayload, MainPostMixerPayload>;
    using MtpLayer  = MtpWeights<MtpAttentionPayload, MtpPostMixerPayload>;
    using DFlash    = DFlashPayload;

    DeviceArena* weights_arena = nullptr;
    Weight token_embedding;
    std::array<FullLayer, FullAttentionLayers> full_layers;
    std::array<GdnLayer, GdnLayers> gdn_layers;
    Tensor final_norm;
    Weight output_head;
    StartupFeatures features;
    std::optional<OptimizedProposalWeights> optimized_proposal;
    std::optional<MtpLayer> mtp;
    std::optional<DFlashPayload> dflash;
    std::optional<VisionWeights> vision;
};

} // namespace targets::qwen3_6
} // namespace ninfer
