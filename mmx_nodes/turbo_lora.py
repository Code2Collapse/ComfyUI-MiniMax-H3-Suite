# Apache License 2.0 — MiniMax-H3 Turbo LoRA reference
# PORTED FROM: comfyui-minimax-h3-turbo :: __init__.py @ 4274783a23afcfdbea3b4876cb79effd6c510785

"""Apply the MiniMax-H3 Turbo LoRA and stash the step contract on the MODEL."""

from __future__ import annotations

import sys
from pathlib import Path

from comfy_api.latest import io

_PKG = Path(__file__).resolve().parents[1]
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from mmx_utils.turbo_lora import TURBO_STEP_CONTRACT, apply_turbo_lora

try:
    import folder_paths
except Exception:  # not just ImportError
    folder_paths = None  # type: ignore[assignment]


def _lora_names() -> list[str]:
    if folder_paths is None:
        return []
    try:
        return list(folder_paths.get_filename_list("loras"))
    except Exception:
        return []


class MiniMaxH3_TurboLoRA(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        names = _lora_names() or ["<no loras folder>"]
        return io.Schema(
            node_id="MiniMaxH3_TurboLoRA",
            display_name="H3 Turbo LoRA",
            category="MiniMax H3/Sampling",
            description=(
                "Apply the MiniMax-H3 Turbo LoRA. Stashes step contract "
                f"{TURBO_STEP_CONTRACT} (target/warn/refuse) on "
                "transformer_options for MiniMaxH3_LegalScheduler to enforce — "
                "wire model → LegalScheduler (optional model input) → sampler. "
                "low_vram=ON merges weights (softer on quantized bases); "
                "OFF=bypass (sharper, more VRAM)."
            ),
            inputs=[
                io.Model.Input("model"),
                io.Combo.Input("lora_name", options=names, default=names[0]),
                io.Float.Input(
                    "strength",
                    default=1.0,
                    min=-10.0,
                    max=10.0,
                    step=0.01,
                ),
                io.Boolean.Input(
                    "low_vram",
                    default=False,
                    tooltip=(
                        "OFF (default): bypass path — sharpest, more peak VRAM. "
                        "ON: merge into weights — lowest VRAM, softer on int8/fp8 bases."
                    ),
                ),
            ],
            outputs=[
                io.Model.Output("model"),
                io.String.Output("report"),
            ],
        )

    @classmethod
    def fingerprint_inputs(cls, model, lora_name, strength=1.0, low_vram=False):
        return f"{lora_name}|{strength}|{low_vram}"

    @classmethod
    def execute(cls, model, lora_name, strength=1.0, low_vram=False) -> io.NodeOutput:
        if folder_paths is None:
            raise RuntimeError(
                "MiniMaxH3_TurboLoRA needs ComfyUI folder_paths to resolve LoRA files."
            )
        path = folder_paths.get_full_path("loras", lora_name)
        patched = apply_turbo_lora(model, path, float(strength), low_vram=bool(low_vram))
        mode = "merge" if low_vram else "bypass"
        report = (
            f"Turbo LoRA applied: {lora_name!r} strength={strength} mode={mode}\n"
            f"step contract {TURBO_STEP_CONTRACT} stashed on transformer_options — "
            "connect this MODEL to LegalScheduler's optional model input."
        )
        return io.NodeOutput(patched, report)
