"""Persistent-object contract for the Qwen3.8-27B NVFP4 + DFlash2 artifact.

The Text, optimized draft-head, MTP, and Vision objects follow the registered
``qwen3.8-27b/nvfp4`` profile (NVFP4 MLP layers 0-55 with per-tensor divisors,
FP8 row-scaled attention/MLP layers 56-63 and FP8 row-scaled embeddings and
output head). The 66 DFlash2 tensor objects — byte-compatible with the
draft component of the ``groupwise-int-dflash2`` artifact — are appended
after the Vision merger objects. The draft component is part of the same
artifact image and does not define a second artifact or an optional profile.
"""

from __future__ import annotations

from . import inventory_dflash2 as dflash2_inventory
from . import inventory_nvfp4 as nvfp4_inventory

MODEL_ID = nvfp4_inventory.MODEL_ID
WEIGHTS_ID = "nvfp4-dflash2"
TARGET_KEY = nvfp4_inventory.TARGET_KEY

BF16 = nvfp4_inventory.BF16
W8 = nvfp4_inventory.W8
NVFP4 = nvfp4_inventory.NVFP4
FP8 = nvfp4_inventory.FP8
FP32 = nvfp4_inventory.FP32
I32 = nvfp4_inventory.I32
Q4 = nvfp4_inventory.Q4
Q5 = nvfp4_inventory.Q5
Q6 = nvfp4_inventory.Q6
FORMAT_NAMES = nvfp4_inventory.FORMAT_NAMES
LAYOUT_NAMES = nvfp4_inventory.LAYOUT_NAMES
CONTIGUOUS_LAYOUT = nvfp4_inventory.CONTIGUOUS_LAYOUT
ROW_SPLIT_LAYOUT = nvfp4_inventory.ROW_SPLIT_LAYOUT
BLOCK_SCALE_LAYOUT = nvfp4_inventory.BLOCK_SCALE_LAYOUT
ROW_SCALE_LAYOUT = nvfp4_inventory.ROW_SCALE_LAYOUT
ResourceSpec = nvfp4_inventory.ResourceSpec
StoredObjectSpec = nvfp4_inventory.StoredObjectSpec
TensorSpec = nvfp4_inventory.TensorSpec

FULL_ATTENTION_LAYERS = nvfp4_inventory.FULL_ATTENTION_LAYERS
GDN_LAYERS = nvfp4_inventory.GDN_LAYERS
NVFP4_MLP_LAYERS = nvfp4_inventory.NVFP4_MLP_LAYERS
FP8_MLP_LAYERS = nvfp4_inventory.FP8_MLP_LAYERS

RESOURCE_SPECS = nvfp4_inventory.RESOURCE_SPECS
TEXT_CORE_TENSOR_SPECS = nvfp4_inventory.TEXT_CORE_TENSOR_SPECS
DRAFT_HEAD_TENSOR_SPECS = nvfp4_inventory.DRAFT_HEAD_TENSOR_SPECS
MTP_TENSOR_SPECS = nvfp4_inventory.MTP_TENSOR_SPECS
VISION_TENSOR_SPECS = nvfp4_inventory.VISION_TENSOR_SPECS
DFLASH2_TENSOR_SPECS = dflash2_inventory.DFLASH2_TENSOR_SPECS
DFLASH2_LAYERS = dflash2_inventory.DFLASH2_LAYERS

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

LOGICAL_ROW_VIEW_SPECS = nvfp4_inventory.LOGICAL_ROW_VIEW_SPECS
ALIAS_SPECS = nvfp4_inventory.ALIAS_SPECS

NVFP4_TENSOR_SPECS = tuple(
    spec for spec in TENSOR_SPECS if spec.format == NVFP4
)
FP8_TENSOR_SPECS = tuple(spec for spec in TENSOR_SPECS if spec.format == FP8)
INPUT_SCALE_DIVISOR_SPECS = tuple(
    spec
    for spec in TENSOR_SPECS
    if spec.format == FP32 and spec.name.endswith("/input_scale_divisor")
)


def validate_inventory() -> None:
    names = tuple(spec.name for spec in OBJECT_SPECS)
    if len(names) != len(set(names)):
        raise ValueError("Qwen3.8 NVFP4 DFlash2 inventory contains duplicate names")
    if (
        len(RESOURCE_SPECS),
        len(TEXT_CORE_TENSOR_SPECS),
        len(DRAFT_HEAD_TENSOR_SPECS),
        len(MTP_TENSOR_SPECS),
        len(VISION_TENSOR_SPECS),
        len(DFLASH2_TENSOR_SPECS),
        len(TENSOR_SPECS),
        len(OBJECT_SPECS),
    ) != (6, 771, 2, 12, 333, 66, 1184, 1190):
        raise ValueError("Qwen3.8 NVFP4 DFlash2 inventory is incomplete")
    if FORMAT_COUNTS != {
        BF16: 566,
        FP32: 208,
        I32: 1,
        Q4: 55,
        Q5: 54,
        Q6: 1,
        W8: 41,
        NVFP4: 112,
        FP8: 146,
    }:
        raise ValueError(f"unexpected numeric allocation: {FORMAT_COUNTS}")
    if LAYOUT_COUNTS != {
        CONTIGUOUS_LAYOUT: 775,
        ROW_SPLIT_LAYOUT: 151,
        BLOCK_SCALE_LAYOUT: 112,
        ROW_SCALE_LAYOUT: 146,
    }:
        raise ValueError(f"unexpected layout allocation: {LAYOUT_COUNTS}")


validate_inventory()


__all__ = [
    "ALIAS_SPECS",
    "BLOCK_SCALE_LAYOUT",
    "CONTIGUOUS_LAYOUT",
    "DFLASH2_LAYERS",
    "DFLASH2_TENSOR_SPECS",
    "DRAFT_HEAD_TENSOR_SPECS",
    "FP8",
    "FP8_MLP_LAYERS",
    "FP8_TENSOR_SPECS",
    "FULL_ATTENTION_LAYERS",
    "GDN_LAYERS",
    "MODEL_ID",
    "MTP_TENSOR_SPECS",
    "NVFP4",
    "NVFP4_MLP_LAYERS",
    "NVFP4_TENSOR_SPECS",
    "OBJECT_SPECS",
    "TARGET_KEY",
    "TENSOR_SPECS",
    "TEXT_CORE_TENSOR_SPECS",
    "VISION_TENSOR_SPECS",
    "WEIGHTS_ID",
]
