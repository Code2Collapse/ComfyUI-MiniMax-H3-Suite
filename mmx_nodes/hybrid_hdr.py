"""Apply extracted Hybrid HDR LoRA to a Ref2VA MODEL at constant strength."""

from __future__ import annotations

import sys
from pathlib import Path

from comfy_api.latest import io

_PKG = Path(__file__).resolve().parents[1]
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from mmx_utils.hybrid_hdr import apply_hybrid_hdr_lora

try:
    import folder_paths
except Exception:  # noqa: BLE001
    folder_paths = None  # type: ignore[assignment]


def _lora_names() -> list[str]:
    if folder_paths is None:
        return []
    try:
        return list(folder_paths.get_filename_list("loras"))
    except Exception:
        return []


class MiniMaxH3_HybridHDR(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        names = _lora_names() or ["<no loras folder>"]
        return io.Schema(
            node_id="MiniMaxH3_HybridHDR",
            display_name="H3 Hybrid HDR LoRA",
            category="MiniMax H3/Sampling",
            description=(
                "Apply the extracted Hybrid HDR LoRA to a Ref2VA MODEL at constant strength. "
                "Removes Singularity's HDR training without the FL2VA mixture damage.\n\n"
                "For late-step HDR only, use a two-stage workflow: sample the first steps with "
                "plain Ref2VA, then continue with this patched MODEL (KSamplerAdvanced or "
                "SamplerCustomAdvanced start/end steps — not SplitSigmas on H3)."
            ),
            inputs=[
                io.Model.Input("model"),
                io.Combo.Input("lora_name", options=names, default=names[0]),
                io.Float.Input(
                    "strength",
                    default=1.0,
                    min=0.0,
                    max=2.0,
                    step=0.01,
                    tooltip="LoRA merge strength. Try 0.5 for a gentle HDR lift.",
                ),
                io.Int.Input(
                    "block_start",
                    default=25,
                    min=0,
                    max=49,
                    tooltip="First DiT block to patch (inclusive). Singularity merge used b25-49.",
                ),
                io.Int.Input(
                    "block_end",
                    default=49,
                    min=0,
                    max=49,
                    tooltip="Last DiT block to patch (inclusive).",
                ),
            ],
            outputs=[
                io.Model.Output("model"),
                io.String.Output("report"),
            ],
        )

    @classmethod
    def fingerprint_inputs(cls, model, lora_name, strength=1.0, block_start=25, block_end=49):
        return f"{lora_name}|{strength}|{block_start}|{block_end}"

    @classmethod
    def execute(cls, model, lora_name, strength=1.0, block_start=25, block_end=49) -> io.NodeOutput:
        if folder_paths is None:
            raise RuntimeError(
                "MiniMaxH3_HybridHDR needs ComfyUI folder_paths to resolve LoRA files."
            )
        if int(block_start) > int(block_end):
            raise ValueError("block_start must be <= block_end")
        path = folder_paths.get_full_path("loras", lora_name)
        patched, report = apply_hybrid_hdr_lora(
            model,
            path,
            float(strength),
            int(block_start),
            int(block_end),
        )
        return io.NodeOutput(patched, report)
