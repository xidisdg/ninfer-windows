"""Hugging Face source recipe for the DFlash2 draft component of the
Qwen3.8-27B groupwise-int-dflash2 artifact.

The DFlash2 checkpoint (``z-lab/Qwen3.8-27B-DFlash2``) is a single-file
safetensors with 81 BF16 tensors: a feature-fusion projection, two norms,
five draft decoder layers (attention + dense MLP + the two-tap dynamic
convolution pairs), a final norm, and the candidate-path selector codebooks.
Every checkpoint tensor is consumed exactly once; the exact-name preflight
rejects any inventory drift on either side.
"""

from __future__ import annotations

from pathlib import Path

from tools.convert.qwen3_6.common.recipe import (
    Concat,
    ShardReader,
    SourcePreflight,
    TensorRecipe,
    preflight_source_reader,
    source,
    source_requirements as _recipe_source_requirements,
    validate_recipe_coverage as _validate_recipe_coverage,
)

from . import inventory_dflash2 as inventory


def _build_dflash2_recipes() -> tuple[TensorRecipe, ...]:
    hidden = inventory.DFLASH2_HIDDEN
    intermediate = inventory.DFLASH2_INTERMEDIATE_SIZE
    query = inventory.DFLASH2_QUERY_SIZE
    kv = inventory.DFLASH2_KV_SIZE
    head = inventory.DFLASH2_HEAD_DIM
    conv_base = (
        inventory.DFLASH2_CONV_KERNEL_SIZE,
        2,
        hidden,
    )
    conv_proj = (inventory.DFLASH2_CONV_PROJECTION_ROWS, hidden)

    recipes: list[TensorRecipe] = [
        TensorRecipe(
            "dflash2/feature_projection",
            source("fc.weight", (hidden, inventory.DFLASH2_FEATURE_INPUT)),
        ),
        TensorRecipe(
            "dflash2/context_norm",
            source("hidden_norm.weight", (hidden,)),
        ),
    ]
    for layer in inventory.DFLASH2_LAYERS:
        source_prefix = f"layers.{layer}."
        object_prefix = f"dflash2/layers/{layer}/"
        recipes.extend(
            (
                TensorRecipe(
                    object_prefix + "input_norm",
                    source(source_prefix + "input_layernorm.weight", (hidden,)),
                ),
                TensorRecipe(
                    object_prefix + "attention/query_key_value",
                    Concat(
                        (
                            source(
                                source_prefix + "self_attn.q_proj.weight",
                                (query, hidden),
                            ),
                            source(
                                source_prefix + "self_attn.k_proj.weight",
                                (kv, hidden),
                            ),
                            source(
                                source_prefix + "self_attn.v_proj.weight",
                                (kv, hidden),
                            ),
                        ),
                        0,
                    ),
                ),
                TensorRecipe(
                    object_prefix + "attention/query_norm",
                    source(source_prefix + "self_attn.q_norm.weight", (head,)),
                ),
                TensorRecipe(
                    object_prefix + "attention/key_norm",
                    source(source_prefix + "self_attn.k_norm.weight", (head,)),
                ),
                TensorRecipe(
                    object_prefix + "attention/output",
                    source(source_prefix + "self_attn.o_proj.weight", (hidden, query)),
                ),
                TensorRecipe(
                    object_prefix + "attention/attention_conv_base",
                    source(source_prefix + "attention_conv.base_kernel", conv_base),
                ),
                TensorRecipe(
                    object_prefix + "attention/attention_conv_projection",
                    source(
                        source_prefix + "attention_conv.kernel_projection.weight",
                        conv_proj,
                    ),
                ),
                TensorRecipe(
                    object_prefix + "post_attention_norm",
                    source(
                        source_prefix + "post_attention_layernorm.weight",
                        (hidden,),
                    ),
                ),
                TensorRecipe(
                    object_prefix + "mlp/gate_up",
                    Concat(
                        (
                            source(
                                source_prefix + "mlp.gate_proj.weight",
                                (intermediate, hidden),
                            ),
                            source(
                                source_prefix + "mlp.up_proj.weight",
                                (intermediate, hidden),
                            ),
                        ),
                        0,
                    ),
                ),
                TensorRecipe(
                    object_prefix + "mlp/down",
                    source(
                        source_prefix + "mlp.down_proj.weight",
                        (hidden, intermediate),
                    ),
                ),
                TensorRecipe(
                    object_prefix + "mlp/mlp_conv_base",
                    source(source_prefix + "mlp_conv.base_kernel", conv_base),
                ),
                TensorRecipe(
                    object_prefix + "mlp/mlp_conv_projection",
                    source(
                        source_prefix + "mlp_conv.kernel_projection.weight",
                        conv_proj,
                    ),
                ),
            )
        )
    recipes.extend(
        (
            TensorRecipe(
                "dflash2/final_norm",
                source("norm.weight", (hidden,)),
            ),
            TensorRecipe(
                "dflash2/selector_predecessor_codebook",
                source(
                    "candidate_selector.predecessor_codebook",
                    (inventory.DFLASH2_SELECTOR_VOCAB, inventory.DFLASH2_SELECTOR_RANK),
                ),
            ),
            TensorRecipe(
                "dflash2/selector_successor_codebook",
                source(
                    "candidate_selector.successor_codebook",
                    (inventory.DFLASH2_SELECTOR_VOCAB, inventory.DFLASH2_SELECTOR_RANK),
                ),
            ),
            TensorRecipe(
                "dflash2/selector_hidden_projection",
                source(
                    "candidate_selector.hidden_projection.weight",
                    (inventory.DFLASH2_SELECTOR_RANK, hidden),
                ),
            ),
        )
    )
    return tuple(recipes)


DFLASH2_RECIPE_SPECS = _build_dflash2_recipes()
DFLASH2_RECIPES_BY_NAME = {
    item.object_name: item for item in DFLASH2_RECIPE_SPECS
}


def validate_recipe_coverage() -> None:
    """Pair the DFlash2 recipe exactly with its inventory component."""

    _validate_recipe_coverage(DFLASH2_RECIPE_SPECS, inventory.DFLASH2_TENSOR_SPECS)
    if len(DFLASH2_RECIPE_SPECS) != len(inventory.DFLASH2_TENSOR_SPECS):
        raise ValueError("DFlash2 recipe count drifted from the inventory")
    if len(DFLASH2_RECIPES_BY_NAME) != len(DFLASH2_RECIPE_SPECS):
        raise ValueError("DFlash2 recipe object names are not unique")


def dflash2_source_requirements() -> dict[str, "object"]:
    return _recipe_source_requirements(DFLASH2_RECIPE_SPECS)


def preflight_dflash2_sources(model_dir: str | Path) -> SourcePreflight:
    """Exact-name and exact-shape check against the single-file checkpoint."""

    model = Path(model_dir)
    with ShardReader.from_file(model / "model.safetensors") as reader:
        actual_names = set(reader.names)
        requirements = dflash2_source_requirements()
        required_names = set(requirements)
        if actual_names != required_names:
            missing = sorted(required_names - actual_names)
            extra = sorted(actual_names - required_names)
            details = []
            if missing:
                details.append(f"missing={missing[:8]!r}")
            if extra:
                details.append(f"extra={extra[:8]!r}")
            raise ValueError(
                "DFlash2 checkpoint source inventory differs from its exact "
                "tensor contract"
                + (": " + ", ".join(details) if details else "")
            )
        return preflight_source_reader(reader, DFLASH2_RECIPE_SPECS)


__all__ = [
    "DFLASH2_RECIPES_BY_NAME",
    "DFLASH2_RECIPE_SPECS",
    "dflash2_source_requirements",
    "preflight_dflash2_sources",
    "validate_recipe_coverage",
]