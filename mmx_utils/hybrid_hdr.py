"""Apply an H3 Hybrid HDR LoRA to a ComfyUI MODEL patcher.

Two paths (A26 3.5):
- ``apply_hybrid_hdr_lora``: the legacy constant-strength MERGE (comfy.lora.load_lora + ModelPatcher.add_patches);
- ``apply_hybrid_hdr_scheduled``: the sigma-scheduled BYPASS (mmx_utils/h3_scheduled_adapter.py): one sampler run,
  strength read from the actual sigma of every model call, optional audio-row exclusion. EXPERIMENTAL until a real
  H3 sampling run validates it.

Block filtering is explicit (A26 3.6): keys inside [block_start, block_end] are "matched", keys outside are
"skipped", keys with no block index (global parameters: embeddings, refiner, heads, AdaLN tables) are kept only when
``include_global_parameters`` is on. The counts go into the report.
"""

from __future__ import annotations

import os
from typing import Any

from mmx_utils.h3_hybrid import key_in_block_range, parse_block_index


def split_by_block(loaded: dict, block_start: int, block_end: int, include_global_parameters: bool = True):
    """(kept patches, counts). Counts: matched block parameters, global parameters (kept or not), skipped."""
    out, matched, glob, skipped = {}, 0, 0, 0
    for target_key in loaded:
        blk = parse_block_index(target_key)
        if blk is None:
            glob += 1
            if include_global_parameters:
                out[target_key] = loaded[target_key]
        elif key_in_block_range(target_key, block_start, block_end):
            matched += 1
            out[target_key] = loaded[target_key]
        else:
            skipped += 1
    return out, {"matched_block_parameters": matched, "global_parameters": glob,
                 "global_included": bool(include_global_parameters), "skipped_out_of_range": skipped}


def _filter_patches_by_block(loaded: dict, block_start: int, block_end: int) -> dict:
    """Legacy behaviour (global parameters kept); kept for callers of the old helper."""
    return split_by_block(loaded, block_start, block_end, True)[0]


def _load(model, lora_path: str):
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
    return new_model, comfy.lora.load_lora(lora, key_map, log_missing=False)


def _no_match(loaded, lora_path, block_start, block_end):
    return ValueError(
        f"None of the {len(loaded)} layers in {os.path.basename(lora_path)!r} matched this "
        f"model. Use the Ref2VA build of the SAME variant the LoRA was extracted from "
        f"(pruned with pruned, full with full), and check the block range "
        f"({block_start}-{block_end}).")


def _counts_text(c: dict, applied: int, total: int) -> str:
    cov = (100.0 * applied / total) if total else 0.0
    return (f"matched block params={c['matched_block_parameters']} global={c['global_parameters']} "
            f"({'included' if c['global_included'] else 'excluded'}) skipped={c['skipped_out_of_range']} "
            f"applied={applied}/{total} ({cov:.1f}%)")


def apply_hybrid_hdr_lora(
    model,
    lora_path: str,
    strength: float,
    block_start: int,
    block_end: int,
    include_global_parameters: bool = True,
) -> tuple[Any, str]:
    """Merge Hybrid HDR LoRA patches at constant strength (no hook keyframes)."""
    new_model, loaded = _load(model, lora_path)
    filtered, counts = split_by_block(loaded, int(block_start), int(block_end), include_global_parameters)
    n = len(new_model.add_patches(filtered, float(strength)))
    if loaded and n == 0:
        # Silent no-op otherwise: the render would simply look unchanged.
        raise _no_match(loaded, lora_path, block_start, block_end)
    report = (
        f"Hybrid HDR LoRA applied: {os.path.basename(lora_path)!r} "
        f"strength={strength} blocks={block_start}-{block_end} "
        f"patches={n}/{len(loaded)}\n" + _counts_text(counts, n, len(loaded))
    )
    return new_model, report


def apply_hybrid_hdr_scheduled(
    model,
    lora_path: str,
    schedule,
    block_start: int,
    block_end: int,
    apply_to: str = "all rows",
    include_global_parameters: bool = True,
) -> tuple[Any, str]:
    """Install the LoRA once as sigma-scheduled bypass hooks (see h3_scheduled_adapter)."""
    from mmx_utils import h3_scheduled_adapter as S
    from mmx_utils.turbo_lora import _int8_fused_fc2

    new_model, loaded = _load(model, lora_path)
    filtered, counts = split_by_block(loaded, int(block_start), int(block_end), include_global_parameters)
    dm = new_model.model.diffusion_model
    mods = sorted({k[len("diffusion_model."):-len(".weight")] for k in filtered if k.startswith("diffusion_model.")
                   and k.endswith(".weight")})
    fused = [f"diffusion_model.{m}.weight" for m in _int8_fused_fc2(dm, mods)]
    res = S.install(new_model, filtered, schedule, apply_to, skip=fused)
    n = res["hooked"]
    if loaded and n == 0:
        raise _no_match(loaded, lora_path, block_start, block_end)
    sch = schedule
    shape = "constant" if sch.constant else (
        f"sigma {sch.start_sigma:g} -> {sch.end_sigma:g}, ramps in {sch.ramp_in:g} / out {sch.ramp_out:g}, "
        f"{sch.curve}")
    report = (
        f"Hybrid HDR LoRA (scheduled bypass, EXPERIMENTAL - not yet validated on a real H3 run): "
        f"{os.path.basename(lora_path)!r} max strength={sch.maximum_strength} ({shape}) "
        f"blocks={block_start}-{block_end} rows={apply_to}\n"
        + _counts_text(counts, n, len(loaded))
        + (f"\nskipped (fused int8 fc2, cannot carry a scheduled hook)={res['skipped_fused_int8']}"
           if res["skipped_fused_int8"] else "")
        + (f"\nunsupported adapter types (LoCon mid / DoRA)={res['unsupported']}" if res["unsupported"] else "")
        + ("\naudio rows: the adapter adds nothing to them; audio can still change through attention to the "
           "adapted video rows" if apply_to != "all rows" else "")
    )
    return new_model, report
