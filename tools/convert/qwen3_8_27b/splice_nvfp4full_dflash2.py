"""Splice the DFlash2 draft component onto the nvfp4full artifact.

Produces identity ``qwen3.8-27b/nvfp4full-dflash2``:

* the full ``nvfp4full`` payload (1259 objects) is copied byte-for-byte,
* the 66 draft objects of a ``qwen3.8-27b/nvfp4-dflash2`` artifact
  (phaseonx11 image, ``dflash/*`` naming) are re-based onto 4096-aligned
  offsets and appended after the base payload. Object names are remapped
  onto the registered ``dflash2/*`` contract, and any object whose stored
  encoding differs from the registered spec (13 tensors: the two-tap conv
  kernel projections and the selector codebooks, stored as plain BF16 in the
  source image) is decoded and re-encoded with the canonical encoder before
  the splice.

The result is re-validated through the container reader (per-object byte
sizes, alignment, ranges).
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import torch

from tools.artifact.container import (
    Artifact,
    ArtifactIdentity,
    MAGIC,
    PREFIX,
    PREFIX_BYTES,
    PAYLOAD_ALIGNMENT,
    encode_directory,
    object_alignment,
    plan_objects,
)
from tools.artifact.layouts import align_up, encoded_size
from tools.convert.qwen3_6.common.conversion import encode_tensor_payload
from tools.convert.qwen3_8_27b.inventory_dflash2 import DFLASH2_TENSOR_SPECS

CHUNK = 1 << 25
NEW_WEIGHTS_ID = "nvfp4full-dflash2"


def rename_dflash(name: str) -> str:
    """Map phaseonx11-style ``dflash/*`` names onto the registered names."""
    if name == "dflash/selector/predecessor_codebook":
        return "dflash2/selector_predecessor_codebook"
    if name == "dflash/selector/successor_codebook":
        return "dflash2/selector_successor_codebook"
    if name == "dflash/selector/hidden_projection":
        return "dflash2/selector_hidden_projection"
    for conv_suffix, target_sub, base_name in (
        ("attention_conv/base_kernel", "attention/", "attention_conv_base"),
        ("attention_conv/kernel_projection", "attention/", "attention_conv_projection"),
        ("mlp_conv/base_kernel", "mlp/", "mlp_conv_base"),
        ("mlp_conv/kernel_projection", "mlp/", "mlp_conv_projection"),
    ):
        if name.endswith("/" + conv_suffix):
            layer = name.split("/")[2]
            return f"dflash2/layers/{layer}/{target_sub}{base_name}"
    if name.startswith("dflash/"):
        return "dflash2/" + name[len("dflash/"):]
    raise ValueError(f"not a dflash object: {name}")


def read_bytes(path: Path, offset: int, count: int) -> bytes:
    with open(path, "rb") as f:
        f.seek(offset)
        data = f.read(count)
    if len(data) != count:
        raise RuntimeError(f"short read at {offset} in {path}")
    return data


def write_stream(out, path: Path, offset: int, count: int, label: str) -> None:
    t0 = time.time()
    with open(path, "rb") as f:
        f.seek(offset)
        remaining = count
        while remaining > 0:
            chunk = f.read(min(CHUNK, remaining))
            if not chunk:
                raise RuntimeError(f"short read while streaming {label}")
            out.write(chunk)
            remaining -= len(chunk)
    print(f"  {label}: {count / (1 << 30):.2f} GiB in {time.time() - t0:.1f}s", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="nvfp4full artifact path")
    ap.add_argument("--src", required=True,
                    help="nvfp4-dflash2 artifact path (draft component source)")
    ap.add_argument("--out", required=True, help="output artifact path")
    args = ap.parse_args()

    base = Artifact(args.base)
    src = Artifact(args.src)
    print(f"base identity: {base.identity.model_id}/{base.identity.weights_id} "
          f"({len(base.objects)} objects, "
          f"{base.file_bytes - base.payload_offset} payload bytes)", flush=True)
    print(f"src  identity: {src.identity.model_id}/{src.identity.weights_id} "
          f"({len(src.objects)} objects, "
          f"{src.file_bytes - src.payload_offset} payload bytes)", flush=True)
    assert base.identity.weights_id == "nvfp4full", "base must be nvfp4full"
    assert src.identity.weights_id == "nvfp4-dflash2", "src must be nvfp4-dflash2"

    raw = [o for o in src.objects if o.name.startswith("dflash/")]
    assert len(raw) == len(DFLASH2_TENSOR_SPECS) == 66, f"draft object count {len(raw)}"
    raw.sort(key=lambda o: o.offset)

    specs_by_name = {s.name: s for s in DFLASH2_TENSOR_SPECS}
    entries = []  # (spec, file_object, needs_reencode)
    for o in raw:
        spec = specs_by_name[rename_dflash(o.name)]
        compatible = (
            spec.format == o.format and spec.layout == o.layout
            and tuple(spec.shape) == tuple(o.shape)
        )
        entries.append((spec, o, not compatible))
    n_reenc = sum(1 for _, _, r in entries if r)
    print(f"draft objects: {len(entries)}, compatible={len(entries) - n_reenc}, "
          f"to-reencode={n_reenc}", flush=True)

    base_payload = base.file_bytes - base.payload_offset
    region_base = align_up(base_payload, PAYLOAD_ALIGNMENT)

    # Re-plan the draft region with the canonical spec encodings.
    from tools.artifact.container import TensorSpec as ContainerTensorSpec
    planned = plan_objects(tuple(
        ContainerTensorSpec(spec.name, tuple(spec.shape), spec.format, spec.layout)
        for spec, _, _ in entries
    ))
    for (spec, o, _), p in zip(entries, planned):
        assert p.name == spec.name
        assert p.offset % object_alignment(p) == 0, f"{p.name}: unaligned"
        # region_base is 4096-aligned and every object alignment divides 4096,
        # so the re-based offsets stay aligned as well.
        assert p.bytes == encoded_size(spec.layout, spec.format, spec.shape)

    new_objects = list(base.objects) + [
        replace(p, offset=region_base + p.offset) for p in planned
    ]
    identity = ArtifactIdentity(base.identity.model_id, NEW_WEIGHTS_ID)
    directory = encode_directory(identity, new_objects)
    payload_start = align_up(PREFIX_BYTES + len(directory), PAYLOAD_ALIGNMENT)
    print(f"new directory: {len(directory)} bytes, {len(new_objects)} objects, "
          f"payload starts at {payload_start}", flush=True)

    t0 = time.time()
    out = open(args.out, "wb")
    try:
        out.write(PREFIX.pack(MAGIC, len(directory)))
        out.write(directory)
        out.write(b"\x00" * (payload_start - PREFIX_BYTES - len(directory)))
        write_stream(out, Path(args.base), base.payload_offset, base_payload,
                     "base payload")
        out.write(b"\x00" * (region_base - base_payload))
        region_pos = 0  # region-relative cursor
        for (spec, o, reenc), p in zip(entries, planned):
            if p.offset > region_pos:
                out.write(b"\x00" * (p.offset - region_pos))
                region_pos = p.offset
            if reenc:
                raw_bytes = read_bytes(Path(args.src), src.payload_offset + o.offset,
                                       o.bytes)
                tensor = (
                    torch.from_numpy(
                        np.frombuffer(raw_bytes, dtype=np.uint16).reshape(list(o.shape))
                    ).view(torch.bfloat16)
                )
                payload = encode_tensor_payload(tensor, spec, "cpu")
                assert len(payload) == p.bytes, (
                    f"{spec.name}: encoded {len(payload)} != planned {p.bytes}"
                )
                out.write(payload)
                print(f"  re-encoded {spec.name} "
                      f"({o.format}/{o.layout} -> {spec.format}/{spec.layout}, "
                      f"{o.bytes} -> {p.bytes} bytes)", flush=True)
                del tensor
            else:
                write_stream(out, Path(args.src), src.payload_offset + o.offset,
                             o.bytes, spec.name)
            region_pos = p.offset + p.bytes
    finally:
        out.close()
    print(f"wrote {args.out} in {time.time() - t0:.1f}s", flush=True)

    check = Artifact(args.out)
    assert check.identity.weights_id == NEW_WEIGHTS_ID
    assert len(check.objects) == len(new_objects)
    for spec, o, reenc in entries:
        obj = check.find(spec.name)
        a = check.payload(obj)
        if reenc:
            continue  # re-encoded payload validated by the container byte check
        b = src.payload(o)
        assert bytes(a[:65536]) == bytes(b[:65536]), f"{spec.name}: payload mismatch"
        assert len(a) == len(b)
    print("verification passed: container re-validation + payload spot-checks OK",
          flush=True)
    print(f"DONE: {len(check.objects)} objects, "
          f"{check.file_bytes / (1 << 30):.2f} GiB", flush=True)
    base.close()
    src.close()
    check.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
