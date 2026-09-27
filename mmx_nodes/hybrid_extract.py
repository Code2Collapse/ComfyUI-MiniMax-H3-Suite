"""Offline Hybrid HDR LoRA extraction from ComfyUI (GPU box)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from comfy_api.latest import io

_PKG = Path(__file__).resolve().parents[1]
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from mmx_utils.h3_hybrid_extract import extract_hybrid_lora

try:
    import folder_paths
    import comfy.model_management as mm
except Exception:  # noqa: BLE001
    folder_paths = None  # type: ignore[assignment]
    mm = None


def _checkpoint_names() -> list[str]:
    if folder_paths is None:
        return []
    names: list[str] = []
    for folder in ("diffusion_models", "unet"):
        try:
            names.extend(folder_paths.get_filename_list(folder))
        except Exception:
            continue
    return sorted(set(names)) or ["<no checkpoints>"]


def _resolve_checkpoint(combo_name: str, path_override: str) -> str:
    override = (path_override or "").strip()
    if override:
        if not os.path.isfile(override):
            raise FileNotFoundError(f"Checkpoint not found: {override!r}")
        return override
    if folder_paths is None:
        raise RuntimeError("MiniMaxH3_HybridExtract needs ComfyUI folder_paths.")
    for folder in ("diffusion_models", "unet"):
        try:
            full = folder_paths.get_full_path(folder, combo_name)
        except Exception:
            full = None
        if full and os.path.isfile(full):
            return full
    raise FileNotFoundError(f"Checkpoint {combo_name!r} not found in diffusion_models or unet")


def _resolve_device(choice: str) -> str:
    if choice == "cpu":
        return "cpu"
    if choice == "cuda":
        return "cuda"
    if mm is not None:
        try:
            dev = mm.get_torch_device()
            return str(dev).split(":")[0] if ":" in str(dev) else str(dev)
        except Exception:
            pass
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def _loras_dir() -> Path:
    if folder_paths is None:
        raise RuntimeError("MiniMaxH3_HybridExtract needs ComfyUI folder_paths.")
    return Path(folder_paths.get_folder_paths("loras")[0])


class MiniMaxH3_HybridExtract(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        ckpt_names = _checkpoint_names()
        return io.Schema(
            node_id="MiniMaxH3_HybridExtract",
            display_name="H3 Hybrid HDR Extract",
            category="MiniMax H3/Sampling",
            description=(
                "Extract Hybrid HDR LoRA from matched Ref2VA / FL2VA / Singularity checkpoints. "
                "Streams one tensor at a time — safe on the GPU box. Saves into the loras folder."
            ),
            inputs=[
                io.Combo.Input("ref_name", options=ckpt_names, default=ckpt_names[0]),
                io.String.Input(
                    "ref_path",
                    default="",
                    tooltip="Optional full path override (use for GGUF in unet/).",
                ),
                io.Combo.Input("fl_name", options=ckpt_names, default=ckpt_names[0]),
                io.String.Input("fl_path", default=""),
                io.Combo.Input("sing_name", options=ckpt_names, default=ckpt_names[0]),
                io.String.Input("sing_path", default=""),
                io.Int.Input("rank", default=256, min=8, max=512),
                io.Boolean.Input(
                    "fl_projection",
                    default=True,
                    tooltip="ON: H = (S-R) - a(F-R). OFF: H = S-R plain delta.",
                ),
                io.Int.Input("block_start", default=25, min=0, max=49),
                io.Int.Input("block_end", default=49, min=0, max=49),
                io.Float.Input(
                    "edge_margin",
                    default=1.1,
                    min=1.0,
                    max=3.0,
                    step=0.05,
                    tooltip=(
                        "Marchenko-Pastur edge multiplier when quantisation noise floor is known "
                        "(int8/bf16/fp16). Singular values above edge are kept."
                    ),
                ),
                io.Combo.Input(
                    "device",
                    options=["auto", "cpu", "cuda"],
                    default="auto",
                ),
                io.String.Input(
                    "output_name",
                    default="h3_hybrid_hdr.safetensors",
                    tooltip="Filename written into the first loras folder.",
                ),
            ],
            outputs=[
                io.String.Output("lora_name"),
                io.String.Output("report"),
            ],
        )

    @classmethod
    def execute(
        cls,
        ref_name,
        ref_path="",
        fl_name=None,
        fl_path="",
        sing_name=None,
        sing_path="",
        rank=256,
        fl_projection=True,
        block_start=25,
        block_end=49,
        edge_margin=1.1,
        device="auto",
        output_name="h3_hybrid_hdr.safetensors",
    ) -> io.NodeOutput:
        if int(block_start) > int(block_end):
            raise ValueError("block_start must be <= block_end")

        ref = _resolve_checkpoint(ref_name, ref_path)
        fl = _resolve_checkpoint(fl_name, fl_path)
        sing = _resolve_checkpoint(sing_name, sing_path)

        out_name = (output_name or "h3_hybrid_hdr.safetensors").strip()
        if not out_name.endswith(".safetensors"):
            out_name += ".safetensors"
        out_path = _loras_dir() / out_name
        dev = _resolve_device(device)

        pbar_holder: dict = {"pbar": None}

        def progress_init(total: int):
            try:
                import comfy.utils as cu

                pbar_holder["pbar"] = cu.ProgressBar(max(1, total))
            except Exception:
                pbar_holder["pbar"] = None

        def progress_cb():
            if pbar_holder["pbar"] is not None:
                pbar_holder["pbar"].update(1)

        def interrupt_check():
            if mm is not None:
                mm.throw_exception_if_processing_interrupted()

        summary = extract_hybrid_lora(
            ref,
            fl,
            sing,
            str(out_path),
            rank=int(rank),
            no_fl_projection=not bool(fl_projection),
            block_start=int(block_start),
            block_end=int(block_end),
            edge_margin=float(edge_margin),
            device=dev,
            progress_cb=progress_cb,
            progress_init=progress_init,
            interrupt_check=interrupt_check,
        )
        return io.NodeOutput(out_name, summary["markdown"])
