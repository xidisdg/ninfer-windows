"""Persistent-object contract for the Qwen3.8-27B groupwise-int + DFlash2 artifact.

The Text, optimized draft-head, MTP, and Vision objects are byte-identical in
identity and order to the registered ``qwen3.8-27b/groupwise-int`` profile.
The 66 DFlash2 tensor objects are appended after the Vision merger objects.
The DFlash2 draft component is part of the same artifact image and does not
define a second artifact or an optional profile.

DFlash2 (vs DFlash1 on the 35B target) adds, per draft layer, two-tap dynamic
convolutions on the attention and MLP sublayers (a static ``[2,2,5120]``
base kernel plus a ``[1280,5120]`` low-rank kernel projection), and replaces
per-position draft sampling with a candidate-path selector: two full-vocab
codebooks (rank 256) and a ``[256,5120]`` hidden projection consumed by the
on-device lattice build.
"""

from __future__ import annotations

from tools.convert.qwen3_6.common.inventory import tensor_spec

from . import inventory as groupwise_inventory

MODEL_ID = "qwen3.8-27b"
WEIGHTS_ID = "groupwise-int-dflash2"
TARGET_KEY = "qwen3_8_27b"

BF16 = groupwise_inventory.BF16
W8 = groupwise_inventory.W8
FORMAT_NAMES = groupwise_inventory.FORMAT_NAMES
LAYOUT_NAMES = groupwise_inventory.LAYOUT_NAMES
ResourceSpec = groupwise_inventory.ResourceSpec
StoredObjectSpec = groupwise_inventory.StoredObjectSpec
TensorSpec = groupwise_inventory.TensorSpec

FULL_ATTENTION_LAYERS = groupwise_inventory.FULL_ATTENTION_LAYERS
GDN_LAYERS = groupwise_inventory.GDN_LAYERS
RESOURCE_SPECS = groupwise_inventory.RESOURCE_SPECS

# DFlash2 draft-model geometry (z-lab/Qwen3.8-27B-DFlash2).
DFLASH2_LAYERS = tuple(range(5))
DFLASH2_HIDDEN = 5120
DFLASH2_INTERMEDIATE = 17408
DFLASH2_QUERY_HEADS = 32
DFLASH2_KV_HEADS = 8
DFLASH2_HEAD_DIM = 128
DFLASH2_QUERY_SIZE = DFLASH2_QUERY_HEADS * DFLASH2_HEAD_DIM
DFLASH2_KV_SIZE = DFLASH2_KV_HEADS * DFLASH2_HEAD_DIM
DFLASH2_INTERMEDIATE_SIZE = DFLASH2_INTERMEDIATE
DFLASH2_BLOCK_SIZE = 8
DFLASH2_SLIDING_WINDOW = 2048
DFLASH2_CONV_KERNEL_SIZE = 2
DFLASH2_CONV_GROUP_SIZE = 16
DFLASH2_CONV_PROJECTION_ROWS = (
    2 * DFLASH2_CONV_KERNEL_SIZE * (DFLASH2_HIDDEN // DFLASH2_CONV_GROUP_SIZE)
)
DFLASH2_SELECTOR_RANK = 256
DFLASH2_SELECTOR_TOP_K = 16
DFLASH2_SELECTOR_VOCAB = 248320
DFLASH2_MASK_TOKEN_ID = 248070
DFLASH2_TARGET_LAYER_IDS = (5, 19, 33, 47, 61)
DFLASH2_FEATURE_INPUT = (
    len(DFLASH2_TARGET_LAYER_IDS) * DFLASH2_HIDDEN
)


def _build_dflash2_specs() -> tuple[TensorSpec, ...]:
    specs: list[TensorSpec] = [
        tensor_spec(
            "dflash2/feature_projection",
            (DFLASH2_HIDDEN, DFLASH2_FEATURE_INPUT),
            W8,
        ),
        tensor_spec("dflash2/context_norm", (DFLASH2_HIDDEN,), BF16),
    ]
    for layer in DFLASH2_LAYERS:
        prefix = f"dflash2/layers/{layer}/"
        specs.extend(
            (
                tensor_spec(prefix + "input_norm", (DFLASH2_HIDDEN,), BF16),
                tensor_spec(
                    prefix + "attention/query_key_value",
                    (
                        DFLASH2_QUERY_SIZE + 2 * DFLASH2_KV_SIZE,
                        DFLASH2_HIDDEN,
                    ),
                    W8,
                ),
                tensor_spec(prefix + "attention/query_norm", (DFLASH2_HEAD_DIM,), BF16),
                tensor_spec(prefix + "attention/key_norm", (DFLASH2_HEAD_DIM,), BF16),
                tensor_spec(
                    prefix + "attention/output",
                    (DFLASH2_HIDDEN, DFLASH2_QUERY_SIZE),
                    W8,
                ),
                tensor_spec(
                    prefix + "attention/attention_conv_base",
                    (
                        DFLASH2_CONV_KERNEL_SIZE,
                        2,
                        DFLASH2_HIDDEN,
                    ),
                    BF16,
                ),
                tensor_spec(
                    prefix + "attention/attention_conv_projection",
                    (DFLASH2_CONV_PROJECTION_ROWS, DFLASH2_HIDDEN),
                    W8,
                ),
                tensor_spec(prefix + "post_attention_norm", (DFLASH2_HIDDEN,), BF16),
                tensor_spec(
                    prefix + "mlp/gate_up",
                    (2 * DFLASH2_INTERMEDIATE_SIZE, DFLASH2_HIDDEN),
                    W8,
                ),
                tensor_spec(
                    prefix + "mlp/down",
                    (DFLASH2_HIDDEN, DFLASH2_INTERMEDIATE_SIZE),
                    W8,
                ),
                tensor_spec(
                    prefix + "mlp/mlp_conv_base",
                    (
                        DFLASH2_CONV_KERNEL_SIZE,
                        2,
                        DFLASH2_HIDDEN,
                    ),
                    BF16,
                ),
                tensor_spec(
                    prefix + "mlp/mlp_conv_projection",
                    (DFLASH2_CONV_PROJECTION_ROWS, DFLASH2_HIDDEN),
                    W8,
                ),
            )
        )
    specs.extend(
        (
            tensor_spec("dflash2/final_norm", (DFLASH2_HIDDEN,), BF16),
            tensor_spec(
                "dflash2/selector_predecessor_codebook",
                (DFLASH2_SELECTOR_VOCAB, DFLASH2_SELECTOR_RANK),
                W8,
            ),
            tensor_spec(
                "dflash2/selector_successor_codebook",
                (DFLASH2_SELECTOR_VOCAB, DFLASH2_SELECTOR_RANK),
                W8,
            ),
            tensor_spec(
                "dflash2/selector_hidden_projection",
                (DFLASH2_SELECTOR_RANK, DFLASH2_HIDDEN),
                W8,
            ),
        )
    )
    return tuple(specs)


TEXT_CORE_TENSOR_SPECS = groupwise_inventory.TEXT_CORE_TENSOR_SPECS
DRAFT_HEAD_TENSOR_SPECS = groupwise_inventory.DRAFT_HEAD_TENSOR_SPECS
MTP_TENSOR_SPECS = groupwise_inventory.MTP_TENSOR_SPECS
VISION_TENSOR_SPECS = groupwise_inventory.VISION_TENSOR_SPECS
DFLASH2_TENSOR_SPECS = _build_dflash2_specs()

TENSOR_SPECS = (
    TEXT_CORE_TENSOR_SPECS
    + DRAFT_HEAD_TENSOR_SPECS
    + MTP_TENSOR_SPECS
    + VISION_TENSOR_SPECS
    + DFLASH2_TENSOR_SPECS
)
OBJECT_SPECS: tuple[StoredObjectSpec, ...] = RESOURCE_SPECS + TENSOR_SPECS

FORMAT_COUNTS = {
    numeric_format: sum(spec.format == numeric_format for spec in TENSOR_SPECS)
    for numeric_format in FORMAT_NAMES
}
LAYOUT_COUNTS = {
    layout: sum(spec.layout == layout for spec in TENSOR_SPECS)
    for layout in LAYOUT_NAMES
}

LOGICAL_ROW_VIEW_SPECS = groupwise_inventory.LOGICAL_ROW_VIEW_SPECS
ALIAS_SPECS = groupwise_inventory.ALIAS_SPECS