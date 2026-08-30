"""Convert the registered Qwen3.8-27B checkpoint and the DFlash2 draft
checkpoint into one complete artifact.

Canonical invocation::

    python3 -m tools.convert.qwen3_8_27b.convert_dflash2 \
      --model /path/to/Qwen3.8-27B \
      --dflash2-model /path/to/Qwen3.8-27B-DFlash2 \
      --out out/qwen3_8_27b_dflash2.ninfer

The Text, optimized draft-head, MTP, and Vision components are produced by the
registered ``qwen3.8-27b/groupwise-int`` pipeline (shared Qwen3.6-27B source
recipe, official-resource hash checks, and the shortlist derivation). The
DFlash2 draft component is appended from the second checkpoint. The base
portion is byte-identical to ``groupwise-int``; only the DFlash2 objects are
added.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Mapping, Sequence

import torch

from tools.artifact.container import (
    ArtifactIdentity,
    ArtifactObject,
    ArtifactWriter,
)
from tools.convert.common.quantize import pick_device
from tools.convert.common.safetensors import ShardReader
from tools.convert.qwen3_6.common import conversion as family_conversion
from tools.convert.qwen3_6.common.recipe import SourcePreflight, materialize_recipe
from tools.convert.qwen3_6_27b import convert as qwen3_6_convert
from tools.convert.qwen3_6_27b import draft_head
from tools.convert.qwen3_6_27b import recipe as groupwise_recipe
from tools.convert.qwen3_8_27b import convert as groupwise_convert

# inventory_dflash2 is a strict superset of the groupwise-int inventory: it
# re-exports the base Text/draft-head/MTP/Vision specs and adds the DFlash2
# draft component. The converter addresses it as ``inventory``.
from . import inventory_dflash2 as inventory
from . import recipe_dflash2 as dflash2_recipe


RECIPE_ID = "qwen3_8_27b_groupwise-int-dflash2-v1"

DFLASH2_REPOSITORY = "z-lab/Qwen3.8-27B-DFlash2"

# Fixed source identities. The DFlash2 checkpoint is a single-file
# safetensors (no index); its geometry is validated from config.json below.
_DFLASH2_ROOT_CONFIG: dict[str, object] = {
    "architectures": ["DFlash2DraftModel"],
    "model_type": "qwen3",
    "attention_bias": False,
    "attention_dropout": 0.0,
    "is_causal": False,
    "head_dim": 128,
    "hidden_act": "silu",
    "hidden_size": 5120,
    "intermediate_size": 17408,
    "max_position_embeddings": 262144,
    "num_attention_heads": 32,
    "num_hidden_layers": 5,
    "num_key_value_heads": 8,
    "num_target_layers": 64,
    "rms_norm_eps": 1e-06,
    "tie_word_embeddings": False,
    "use_cache": True,
    "use_sliding_window": True,
    "sliding_window": 2048,
    "vocab_size": 248320,
}
_DFLASH2_DRAFT_CONFIG: dict[str, object] = {
    "block_size": 8,
    "conv_group_size": 16,
    "conv_kernel_size": 2,
    "mask_token_id": 248070,
    "selector_rank": 256,
    "selector_top_k": 16,
    "target_layer_ids": [5, 19, 33, 47, 61],
}

ResourcePayload = family_conversion.ResourcePayload
ObjectPlan = family_conversion.ObjectPlan


@dataclass(frozen=True, slots=True)
class ConversionPreflight:
    model_dir: Path
    dflash2_model_dir: Path
    base_config_summary: dict[str, object]
    dflash2_config_summary: dict[str, object]
    base_source: SourcePreflight
    dflash2_source: SourcePreflight
    resources: tuple[ResourcePayload, ...]
    draft: draft_head.DraftHeadContext
    object_plan: ObjectPlan


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def validate_dflash2_config(config: Mapping[str, object]) -> dict[str, object]:
    """Validate every DFlash2 checkpoint fact that fixes storage or execution."""

    family_conversion.check_members("dflash2 config", config, _DFLASH2_ROOT_CONFIG)
    draft = config.get("dflash_config")
    if not isinstance(draft, Mapping):
        raise ValueError("DFlash2 config.json is missing dflash_config")
    family_conversion.check_members(
        "dflash2 config.dflash_config", draft, _DFLASH2_DRAFT_CONFIG
    )
    rope = config.get("rope_parameters")
    if not isinstance(rope, Mapping):
        raise ValueError("DFlash2 config.json is missing rope_parameters")
    if rope.get("rope_type") != "default" or rope.get("rope_theta") != 10000000:
        raise ValueError("DFlash2 rope_parameters do not match the target")
    layer_types = config.get("layer_types")
    if not isinstance(layer_types, list) or tuple(layer_types) != (
        "sliding_attention",
    ) * len(inventory.DFLASH2_LAYERS):
        raise ValueError(
            "DFlash2 layer_types must be all-sliding for the registered "
            f"{len(inventory.DFLASH2_LAYERS)}-layer schedule"
        )
    return {
        name: config[name] for name in _DFLASH2_ROOT_CONFIG
    } | {
        "dflash_config": {name: draft[name] for name in _DFLASH2_DRAFT_CONFIG},
        "rope_parameters": {name: rope[name] for name in rope},
    }


def preflight_inventory() -> None:
    """Prove the target-private inventory before any payload is written."""

    counts = (
        len(inventory.RESOURCE_SPECS),
        len(inventory.TEXT_CORE_TENSOR_SPECS),
        len(inventory.DRAFT_HEAD_TENSOR_SPECS),
        len(inventory.MTP_TENSOR_SPECS),
        len(inventory.VISION_TENSOR_SPECS),
        len(inventory.DFLASH2_TENSOR_SPECS),
        len(inventory.TENSOR_SPECS),
        len(inventory.OBJECT_SPECS),
    )
    if counts != (6, 771, 2, 12, 333, 66, 1184, 1190):
        raise ValueError(f"target inventory is incomplete: {counts}")
    groupwise_recipe.validate_recipe_coverage()
    dflash2_recipe.validate_recipe_coverage()


def build_object_plan(resources: Mapping[str, bytes]) -> ObjectPlan:
    preflight_inventory()
    return family_conversion.build_object_plan(inventory.OBJECT_SPECS, resources)


def preflight_conversion(
    model_dir: str | Path,
    dflash2_model_dir: str | Path,
) -> ConversionPreflight:
    """Complete config, source, shortlist, and offset work before writing."""

    model = Path(model_dir)
    dflash2_model = Path(dflash2_model_dir)
    base_config_summary = qwen3_6_convert.validate_config(
        family_conversion.load_json(model / "config.json")
    )
    dflash2_config_summary = validate_dflash2_config(
        family_conversion.load_json(dflash2_model / "config.json")
    )
    preflight_inventory()
    base_source = groupwise_recipe.preflight_sources(model)
    dflash2_source = dflash2_recipe.preflight_dflash2_sources(dflash2_model)

    resources = groupwise_convert.load_resources(model)
    resource_map = {resource.name: resource.data for resource in resources}
    object_plan = build_object_plan(resource_map)

    ranking = _repo_root() / draft_head.DEFAULT_RANKING
    draft = draft_head.compute_shortlist(ranking, model)
    return ConversionPreflight(
        model_dir=model,
        dflash2_model_dir=dflash2_model,
        base_config_summary=base_config_summary,
        dflash2_config_summary=dflash2_config_summary,
        base_source=base_source,
        dflash2_source=dflash2_source,
        resources=resources,
        draft=draft,
        object_plan=object_plan,
    )


def materialize_tensor(
    spec: inventory.TensorSpec,
    reader: ShardReader,
    draft: draft_head.DraftHeadContext,
) -> torch.Tensor:
    if spec.name.startswith("dflash2/"):
        tensor = materialize_recipe(
            dflash2_recipe.DFLASH2_RECIPES_BY_NAME[spec.name],
            reader,
        )
    else:
        tensor = qwen3_6_convert.materialize_tensor(spec, reader, draft)
    if tuple(tensor.shape) != spec.shape:
        raise ValueError(
            f"{spec.name}: materialized shape {tuple(tensor.shape)} != {spec.shape}"
        )
    return tensor


def encode_tensor_payload(
    tensor: torch.Tensor,
    spec: inventory.TensorSpec,
    device: str | torch.device,
) -> bytes:
    return family_conversion.encode_tensor_payload(tensor, spec, device)


def build_conversion_report(
    *,
    model_dir: str | Path,
    dflash2_model_dir: str | Path,
    out_path: str | Path,
    arguments: Mapping[str, object],
    base_config_summary: Mapping[str, object],
    dflash2_config_summary: Mapping[str, object],
    base_source_preflight: SourcePreflight,
    dflash2_source_preflight: SourcePreflight,
    objects: Sequence[ArtifactObject],
    elapsed_seconds: float,
    final_bytes: int,
    device: torch.device,
    ranking_path: str | Path,
    revision: str | None = None,
    environment: Mapping[str, object] | None = None,
) -> dict[str, object]:
    combined_source = SourcePreflight(
        recipe_count=(
            base_source_preflight.recipe_count
            + dflash2_source_preflight.recipe_count
        ),
        source_tensor_count=(
            base_source_preflight.source_tensor_count
            + dflash2_source_preflight.source_tensor_count
        ),
        source_shard_count=(
            base_source_preflight.source_shard_count
            + dflash2_source_preflight.source_shard_count
        ),
        source_dtype_counts={
            "BF16": (
                base_source_preflight.source_dtype_counts.get("BF16", 0)
                + dflash2_source_preflight.source_dtype_counts.get("BF16", 0)
            )
        },
    )
    report = family_conversion.build_conversion_report(
        identity=ArtifactIdentity(inventory.MODEL_ID, inventory.WEIGHTS_ID),
        target_key=inventory.TARGET_KEY,
        recipe_id=RECIPE_ID,
        repo_root=_repo_root(),
        model_dir=model_dir,
        out_path=out_path,
        arguments=arguments,
        config_summary={
            "base": dict(base_config_summary),
            "dflash2": dict(dflash2_config_summary),
        },
        source_preflight=combined_source,
        objects=objects,
        elapsed_seconds=elapsed_seconds,
        final_bytes=final_bytes,
        device=device,
        ranking_path=ranking_path,
        revision=revision,
        environment_summary=environment,
    )
    report["source"]["base_model_path"] = report["source"].pop("model_path")
    report["source"]["dflash2_model_path"] = str(Path(dflash2_model_dir).resolve())
    report["source"]["dflash2_repository"] = DFLASH2_REPOSITORY
    report["source_preflight"] = {
        "base": {
            "recipes": base_source_preflight.recipe_count,
            "tensors": base_source_preflight.source_tensor_count,
            "shards": base_source_preflight.source_shard_count,
            "dtypes": dict(base_source_preflight.source_dtype_counts),
        },
        "dflash2": {
            "recipes": dflash2_source_preflight.recipe_count,
            "tensors": dflash2_source_preflight.source_tensor_count,
            "files": dflash2_source_preflight.source_shard_count,
            "dtypes": dict(dflash2_source_preflight.source_dtype_counts),
        },
        "combined": {
            "recipes": combined_source.recipe_count,
            "tensors": combined_source.source_tensor_count,
            "files": combined_source.source_shard_count,
            "dtypes": dict(combined_source.source_dtype_counts),
        },
    }
    report["draft_head"] = {
        "rows": draft_head.DRAFT_HEAD_N,
        "tokenizer_vocab_size": draft_head.TOKENIZER_VOCAB_SIZE,
        "shared_semantic_vocabulary": True,
    }
    report["dflash2"] = {
        "repository": DFLASH2_REPOSITORY,
        "layers": len(inventory.DFLASH2_LAYERS),
        "block_size": inventory.DFLASH2_BLOCK_SIZE,
        "sliding_window": inventory.DFLASH2_SLIDING_WINDOW,
        "conv_kernel_size": inventory.DFLASH2_CONV_KERNEL_SIZE,
        "conv_group_size": inventory.DFLASH2_CONV_GROUP_SIZE,
        "selector_rank": inventory.DFLASH2_SELECTOR_RANK,
        "selector_top_k": inventory.DFLASH2_SELECTOR_TOP_K,
        "target_layer_ids": list(inventory.DFLASH2_TARGET_LAYER_IDS),
    }
    report["quantization"] = {
        "encoder_profile": "MAXABS_F16_RECIP_RNE_V1",
        "component_tensor_bytes": {
            "dflash2": family_conversion.tensor_payload_bytes(
                inventory.DFLASH2_TENSOR_SPECS
            ),
            "total": family_conversion.tensor_payload_bytes(inventory.TENSOR_SPECS),
        },
    }
    return report


def convert(
    model_dir: str | Path,
    dflash2_model_dir: str | Path,
    out_path: str | Path,
    *,
    device: str | torch.device = "cuda",
) -> Path:
    """Run the complete target conversion and return its report path."""

    started = time.perf_counter()
    model = Path(model_dir)
    dflash2_model = Path(dflash2_model_dir)
    output = Path(out_path)
    requested_device = str(device)
    resolved_device = pick_device(device)
    preflight = preflight_conversion(model, dflash2_model)

    print(
        f"preflight complete: {len(preflight.object_plan.objects)} objects, "
        f"base={preflight.base_source.source_tensor_count} source tensors, "
        f"dflash2={preflight.dflash2_source.source_tensor_count} source tensors, "
        f"device={resolved_device}",
        flush=True,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    resources = {resource.name: resource.data for resource in preflight.resources}
    with ArtifactWriter(
        output,
        ArtifactIdentity(inventory.MODEL_ID, inventory.WEIGHTS_ID),
        preflight.object_plan.specs,
    ) as writer:
        if writer.objects != preflight.object_plan.objects:
            raise RuntimeError("writer object plan differs from completed preflight")
        index = 0

        def write_payload(spec, payload: bytes) -> None:
            nonlocal index
            writer.write(spec.name, payload)
            index += 1
            print(
                f"[{index}/{len(inventory.OBJECT_SPECS)}] {spec.name}",
                flush=True,
            )

        for spec in inventory.RESOURCE_SPECS:
            write_payload(spec, resources[spec.name])

        base_specs = inventory.TENSOR_SPECS[
            : -len(inventory.DFLASH2_TENSOR_SPECS)
        ]
        with ShardReader(model) as reader:
            for spec in base_specs:
                tensor = materialize_tensor(spec, reader, preflight.draft)
                payload = encode_tensor_payload(tensor, spec, resolved_device)
                del tensor
                write_payload(spec, payload)
                del payload

        with ShardReader.from_file(dflash2_model / "model.safetensors") as reader:
            for spec in inventory.DFLASH2_TENSOR_SPECS:
                tensor = materialize_tensor(spec, reader, preflight.draft)
                payload = encode_tensor_payload(tensor, spec, resolved_device)
                del tensor
                write_payload(spec, payload)
                del payload

    elapsed = time.perf_counter() - started
    final_bytes = output.stat().st_size
    ranking = _repo_root() / draft_head.DEFAULT_RANKING
    arguments = {
        "model": str(model_dir),
        "dflash2_model": str(dflash2_model_dir),
        "out": str(out_path),
        "device": requested_device,
    }
    report = build_conversion_report(
        model_dir=model,
        dflash2_model_dir=dflash2_model,
        out_path=output,
        arguments=arguments,
        base_config_summary=preflight.base_config_summary,
        dflash2_config_summary=preflight.dflash2_config_summary,
        base_source_preflight=preflight.base_source,
        dflash2_source_preflight=preflight.dflash2_source,
        objects=preflight.object_plan.objects,
        elapsed_seconds=elapsed,
        final_bytes=final_bytes,
        device=resolved_device,
        ranking_path=ranking,
    )
    report_path = Path(str(output) + ".conversion.json")
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(
        f"complete: {final_bytes} bytes in {elapsed:.1f}s; report={report_path}",
        flush=True,
    )
    return report_path


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--dflash2-model", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    convert(
        args.model,
        args.dflash2_model,
        args.out,
        device=args.device,
    )


if __name__ == "__main__":
    main()