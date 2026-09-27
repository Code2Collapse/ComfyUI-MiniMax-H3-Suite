"""Apply H3 Hybrid HDR LoRA to a ComfyUI MODEL patcher."""

from __future__ import annotations

import os
from typing import Any

from mmx_utils.h3_hybrid import key_in_block_range, parse_block_index


def _filter_patches_by_block(loaded: dict, block_start: int, block_end: int) -> dict:
    out = {}
    for target_key in loaded:
        blk = parse_block_index(target_key)
        if blk is None or key_in_block_range(target_key, block_start, block_end):
            out[target_key] = loaded[target_key]
    return out


def apply_hybrid_hdr_lora(
    model,
    lora_path: str,
    strength: float,
    block_start: int,
    block_end: int,
) -> tuple[Any, str]:
    """Merge Hybrid HDR LoRA patches at constant strength (no hook keyframes)."""
    try:
        import comfy.lora
        import comfy.utils
    except Exception as exc:
        raise RuntimeError("MiniMaxH3_HybridHDR needs ComfyUI (comfy.lora, comfy.utils).") from exc

    if not lora_path or not os.path.isfile(lora_path):
        raise FileNotFoundError(f"Hybrid HDR LoRA not found: {lora_path!r}")

    lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
    new_model = model.clone()
    key_map = comfy.lora.model_lora_keys_unet(new_model.model, {})
    loaded = comfy.lora.load_lora(lora, key_map, log_missing=False)
    filtered = _filter_patches_by_block(loaded, int(block_start), int(block_end))
    n = len(new_model.add_patches(filtered, float(strength)))
    if loaded and n == 0:
        # Silent no-op otherwise: the render would simply look unchanged.
        raise ValueError(
            f"None of the {len(loaded)} layers in {os.path.basename(lora_path)!r} matched this "
            f"model. Use the Ref2VA build of the SAME variant the LoRA was extracted from "
            f"(pruned with pruned, full with full), and check the block range "
            f"({block_start}-{block_end})."
        )
    report = (
        f"Hybrid HDR LoRA applied: {os.path.basename(lora_path)!r} "
        f"strength={strength} blocks={block_start}-{block_end} "
        f"patches={n}/{len(loaded)}"
    )
    return new_model, report
