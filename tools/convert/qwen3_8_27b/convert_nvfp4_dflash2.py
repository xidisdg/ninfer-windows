"""Build the registered Qwen3.8-27B NVFP4 + DFlash2 artifact.

The Text, optimized draft-head, MTP, and Vision components are produced by
the registered ``qwen3.8-27b/nvfp4`` dual-source pipeline (official BF16
checkpoint for the direct/FP8 paths, the NVFP4 checkpoint for the NVFP4 MLP
layers and their input divisors). The DFlash2 draft component is appended
from the third checkpoint (z-lab/Qwen3.8-27B-DFlash2) and is byte-compatible
with the draft component of the ``groupwise-int-dflash2`` artifact.

Canonical invocation::

    python3 -m tools.convert.qwen3_8_27b.convert_nvfp4_dflash2 \
      --model /path/to/Qwen3.8-27B \
      --quantized-model /path/to/Qwen3.8-27B-NVFP4 \
      --dflash2-model /path/to/Qwen3.8-27B-DFlash2 \
      --out out/qwen3_8_27b_nvfp4_dflash2.ninfer
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Iterable, Mapping, Sequence

import torch

from tools.artifact.container import (
    ArtifactIdentity,
    ArtifactObject,
    ArtifactWriter,
)
from tools.artifact.layouts import (
    encode_direct,
    encode_fp8_row_scaled,
    encode_nvfp4,
)
from tools.convert.common.quantize import pick_device
from tools.convert.common.safetensors import ShardReader
from tools.convert.qwen3_6.common import conversion as family_conversion
from tools.convert.qwen3_6.common import recipe as family_recipe
from tools.convert.qwen3_6_27b import convert as family_config
from tools.convert.qwen3_6_27b import draft_head

from . import convert as base_convert
from . import convert_dflash2
from . import convert_nvfp4
from . import fp8_embedding
from . import inventory_nvfp4_dflash2 as inventory
from . import recipe_dflash2 as dflash2_recipe
from . import recipe_nvfp4 as recipe


RECIPE_ID = "qwen3_8_27b_nvfp4-dflash2-v1"
OUTPUT_BASENAME = "qwen3_8_27b_nvfp4_dflash2.ninfer"
DFLASH2_REPOSITORY = "z-lab/Qwen3.8-27B-DFlash2"


@dataclass(frozen=True, slots=True)
class ConversionPreflight:
    official_dir: Path
    quantized_dir: Path
    dflash2_dir: Path
    config_summary: dict[str, object]
    dflash2_config_summary: dict[str, object]
    official_source: family_recipe.SourcePreflight
    quantized_source: family_recipe.SourcePreflight
    dflash2_source: family_recipe.SourcePreflight
    resources: tuple[family_conversion.ResourcePayload, ...]
    draft: draft_head.DraftHeadContext
    object_plan: family_conversion.ObjectPlan


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def preflight_inventory() -> None:
    inventory.validate_inventory()
    recipe.validate_recipe()
    dflash2_recipe.validate_recipe_coverage()


def build_object_plan(
    resources: Mapping[str, bytes],
) -> family_conversion.ObjectPlan:
    preflight_inventory()
    return family_conversion.build_object_plan(inventory.OBJECT_SPECS, resources)


def preflight_conversion(
    official_dir: str | Path,
    quantized_dir: str | Path,
    dflash2_model_dir: str | Path,
) -> ConversionPreflight:
    official = Path(official_dir)
    quantized = Path(quantized_dir)
    dflash2_model = Path(dflash2_model_dir)
    convert_nvfp4._validate_index(official)
    convert_nvfp4._validate_index(quantized)

    official_config = family_conversion.load_json(official / "config.json")
    if official_config.get("quantization_config") is not None:
        raise ValueError("official source must not declare quantization_config")
    official_summary = family_config.validate_config(official_config)
    quantized_summary = convert_nvfp4._validate_quantized_config(
        family_conversion.load_json(quantized / "config.json")
    )
    if official_summary != quantized_summary:
        raise ValueError("official and quantized source model configs do not match")
    dflash2_summary = convert_dflash2.validate_dflash2_config(
        family_conversion.load_json(dflash2_model / "config.json")
    )
    preflight_inventory()

    with ShardReader(official) as official_reader:
        official_source = recipe.preflight_official_sources(official_reader)
    with ShardReader(quantized) as quantized_reader:
        quantized_source = recipe.preflight_quantized_metadata(quantized_reader)
    dflash2_source = dflash2_recipe.preflight_dflash2_sources(dflash2_model)

    resources = base_convert.load_resources(official)
    resource_map = {resource.name: resource.data for resource in resources}
    object_plan = build_object_plan(resource_map)
    ranking = _repo_root() / draft_head.DEFAULT_RANKING
    draft = draft_head.compute_shortlist(ranking, official)
    return ConversionPreflight(
        official_dir=official,
        quantized_dir=quantized,
        dflash2_dir=dflash2_model,
        config_summary=official_summary,
        dflash2_config_summary=dflash2_summary,
        official_source=official_source,
        quantized_source=quantized_source,
        dflash2_source=dflash2_source,
        resources=resources,
        draft=draft,
        object_plan=object_plan,
    )


def _materialize_dflash2_tensor(
    spec: inventory.TensorSpec,
    reader: ShardReader,
    device: torch.device,
) -> bytes:
    selected = dflash2_recipe.DFLASH2_RECIPES_BY_NAME[spec.name]
    tensor = family_recipe.materialize_recipe(selected, reader)
    if tuple(tensor.shape) != spec.shape:
        raise ValueError(
            f"{spec.name}: materialized shape {tuple(tensor.shape)} != {spec.shape}"
        )
    payload = family_conversion.encode_tensor_payload(tensor, spec, device)
    del tensor
    return payload


def _build_report(
    *,
    preflight: ConversionPreflight,
    output: Path,
    arguments: Mapping[str, object],
    objects: Sequence[ArtifactObject],
    elapsed_seconds: float,
    final_bytes: int,
    device: torch.device,
) -> dict[str, object]:
    ranking = _repo_root() / draft_head.DEFAULT_RANKING
    combined_source = family_recipe.SourcePreflight(
        recipe_count=(
            preflight.official_source.recipe_count
            + preflight.quantized_source.recipe_count
            + preflight.dflash2_source.recipe_count
        ),
        source_tensor_count=(
            preflight.official_source.source_tensor_count
            + preflight.quantized_source.source_tensor_count
            + preflight.dflash2_source.source_tensor_count
        ),
        source_shard_count=(
            preflight.official_source.source_shard_count
            + preflight.quantized_source.source_shard_count
            + preflight.dflash2_source.source_shard_count
        ),
        source_dtype_counts={
            "BF16": (
                preflight.official_source.source_dtype_counts.get("BF16", 0)
                + preflight.quantized_source.source_dtype_counts.get("BF16", 0)
                + preflight.dflash2_source.source_dtype_counts.get("BF16", 0)
            ),
        },
    )
    report = family_conversion.build_conversion_report(
        identity=ArtifactIdentity(inventory.MODEL_ID, inventory.WEIGHTS_ID),
        target_key=inventory.TARGET_KEY,
        recipe_id=RECIPE_ID,
        repo_root=_repo_root(),
        model_dir=preflight.official_dir,
        out_path=output,
        arguments=arguments,
        config_summary=preflight.config_summary,
        source_preflight=combined_source,
        objects=objects,
        elapsed_seconds=elapsed_seconds,
        final_bytes=final_bytes,
        device=device,
        ranking_path=ranking,
    )
    report["source"] = {
        "official": {
            "repository": recipe.BASE_REPOSITORY,
            "revision": recipe.BASE_REVISION,
            "model_path": str(preflight.official_dir.resolve()),
        },
        "quantized": {
            "repository": recipe.QUANTIZED_REPOSITORY,
            "revision": recipe.QUANTIZED_REVISION,
            "model_path": str(preflight.quantized_dir.resolve()),
        },
        "dflash2": {
            "repository": DFLASH2_REPOSITORY,
            "model_path": str(preflight.dflash2_dir.resolve()),
        },
        "ranking_path": str(ranking.resolve()),
    }
    report["source_preflight"] = {
        "official": {
            "recipes": preflight.official_source.recipe_count,
            "tensors": preflight.official_source.source_tensor_count,
            "shards": preflight.official_source.source_shard_count,
            "dtypes": dict(preflight.official_source.source_dtype_counts),
        },
        "quantized": {
            "recipes": preflight.quantized_source.recipe_count,
            "tensors": preflight.quantized_source.source_tensor_count,
            "shards": preflight.quantized_source.source_shard_count,
            "dtypes": dict(preflight.quantized_source.source_dtype_counts),
            "source_fp8_matrices": len(recipe.FP8_SOURCES),
            "source_nvfp4_matrices": len(recipe.NVFP4_SOURCES),
        },
        "dflash2": {
            "recipes": preflight.dflash2_source.recipe_count,
            "tensors": preflight.dflash2_source.source_tensor_count,
            "files": preflight.dflash2_source.source_shard_count,
            "dtypes": dict(preflight.dflash2_source.source_dtype_counts),
        },
    }
    report["embedding_encoder"] = fp8_embedding.ENCODER_PROFILE
    report["dflash2"] = {
        "repository": DFLASH2_REPOSITORY,
        "layers": len(inventory.DFLASH2_LAYERS),
    }
    return report


def convert(
    official_dir: str | Path,
    quantized_dir: str | Path,
    dflash2_model_dir: str | Path,
    out_path: str | Path,
    *,
    device: str | torch.device = "cuda",
) -> Path:
    """Run the closed tri-source conversion and return its report path."""

    started = time.perf_counter()
    output = Path(out_path)
    if output.name != OUTPUT_BASENAME:
        raise ValueError(
            f"NVFP4 DFlash2 converter output basename must be {OUTPUT_BASENAME!r}"
        )
    requested_device = str(device)
    resolved_device = pick_device(device)
    preflight = preflight_conversion(official_dir, quantized_dir, dflash2_model_dir)

    print(
        f"preflight complete: {len(preflight.object_plan.objects)} objects, "
        f"{len(recipe.FP8_SOURCES)} FP8 and "
        f"{len(recipe.NVFP4_SOURCES)} NVFP4 source matrices, "
        f"device={resolved_device}",
        flush=True,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    resources = {resource.name: resource.data for resource in preflight.resources}
    draft_ids = draft_head.materialize_draft_head_token_ids(preflight.draft)
    derived = {draft_head.DRAFT_HEAD_TOKEN_IDS_OBJECT: draft_ids}
    with ShardReader(preflight.official_dir) as official_reader, ShardReader(
        preflight.quantized_dir
    ) as quantized_reader, ShardReader(preflight.dflash2_dir) as dflash2_reader:
        with ArtifactWriter(
            output,
            ArtifactIdentity(inventory.MODEL_ID, inventory.WEIGHTS_ID),
            preflight.object_plan.specs,
        ) as writer:
            if writer.objects != preflight.object_plan.objects:
                raise RuntimeError(
                    "writer object plan differs from completed preflight"
                )
            for index, spec in enumerate(inventory.OBJECT_SPECS, start=1):
                payload: bytes | Iterable[bytes]
                if isinstance(spec, inventory.ResourceSpec):
                    payload = resources[spec.name]
                elif spec.name.startswith("dflash2/"):
                    payload = _materialize_dflash2_tensor(
                        spec, dflash2_reader, resolved_device
                    )
                elif spec.name == "text/token_embedding":
                    payload = fp8_embedding.iter_reader_payload(
                        official_reader,
                        recipe.OFFICIAL_EMBEDDING_SOURCE.name,
                        spec.shape,
                    )
                elif spec.name in recipe.FP8_WEIGHTS_BY_NAME:
                    payload = convert_nvfp4._encode_fp8_weight(spec, quantized_reader)
                elif spec.name in recipe.NVFP4_WEIGHTS_BY_NAME:
                    payload = convert_nvfp4._encode_nvfp4_weight(spec, quantized_reader)
                elif spec.name in recipe.INPUT_DIVISORS_BY_NAME:
                    scalar = recipe.materialize_input_divisor(
                        recipe.INPUT_DIVISORS_BY_NAME[spec.name],
                        quantized_reader,
                    )
                    payload = encode_direct(scalar, inventory.FP32)
                elif spec.name in recipe.QUANTIZED_DIRECT_BY_NAME:
                    tensor = convert_nvfp4._materialize_direct(spec, quantized_reader)
                    payload = encode_direct(tensor, spec.format)
                    del tensor
                else:
                    tensor = convert_nvfp4._materialize_official(
                        spec, official_reader, derived
                    )
                    payload = family_conversion.encode_tensor_payload(
                        tensor, spec, resolved_device
                    )
                    del tensor
                writer.write(spec.name, payload)
                del payload
                print(
                    f"[{index}/{len(inventory.OBJECT_SPECS)}] {spec.name}",
                    flush=True,
                )

    elapsed = time.perf_counter() - started
    final_bytes = output.stat().st_size
    arguments = {
        "model": str(official_dir),
        "quantized_model": str(quantized_dir),
        "dflash2_model": str(dflash2_model_dir),
        "out": str(out_path),
        "device": requested_device,
    }
    report = _build_report(
        preflight=preflight,
        output=output,
        arguments=arguments,
        objects=preflight.object_plan.objects,
        elapsed_seconds=elapsed,
        final_bytes=final_bytes,
        device=resolved_device,
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
    parser.add_argument("--quantized-model", required=True, type=Path)
    parser.add_argument("--dflash2-model", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    arguments = parser.parse_args(argv)
    convert(
        arguments.model,
        arguments.quantized_model,
        arguments.dflash2_model,
        arguments.out,
        device=arguments.device,
    )


if __name__ == "__main__":
    main()
