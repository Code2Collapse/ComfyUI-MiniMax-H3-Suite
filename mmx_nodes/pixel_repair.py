"""N5 — H3 Pixel Repair (snow · specks · colour blotches)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from comfy_api.latest import io

_PKG = Path(__file__).resolve().parents[1]
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from mmx_utils.device import run_with_cpu_fallback  # noqa: E402
from mmx_utils.pixel_repair import repair_frames  # noqa: E402

try:
    import comfy.model_management as mm
except Exception:  # not just ImportError: the comfy_kitchen skew raises AttributeError
    mm = None


class MiniMaxH3_PixelRepair(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MiniMaxH3_PixelRepair",
            display_name="H3 Pixel Repair (snow · specks · colour blotches)",
            category="MiniMax H3/Finish",
            description=(
                "Clean impulse snow and local chroma blotches after colour matching.\n\n"
                "Impulse repair uses a motion guard so real detail on moving clips is never "
                "touched. Blotch repair works in Oklab and leaves lightness alone."
            ),
            inputs=[
                io.Image.Input(
                    "images",
                    tooltip="Frames to clean — usually after Reference Colour Match.",
                ),
                io.Mask.Input(
                    "mask",
                    optional=True,
                    tooltip=(
                        "Restrict repairs to this region. Default: whole frame. "
                        "Use to spare a deliberate effect area."
                    ),
                ),
                io.Float.Input(
                    "impulse",
                    default=0.5,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip=(
                        "Salt-and-snow sensitivity. Higher = more aggressive. Turn down if "
                        "fine sparkle on the plate is being erased."
                    ),
                ),
                io.Float.Input(
                    "blotch",
                    default=0.5,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip=(
                        "Chroma blotch pull-back (red cheeks, colour specks). Higher = stronger. "
                        "Turn down if legitimate blush or makeup is being flattened."
                    ),
                ),
                io.Boolean.Input(
                    "debug",
                    default=False,
                    tooltip="Include extra per-frame counts in the report JSON.",
                ),
            ],
            outputs=[
                io.Image.Output("images"),
                io.Mask.Output(
                    "damage_mask",
                    tooltip="Union of impulse and blotch damage detected per frame.",
                ),
                io.String.Output("report"),
            ],
        )

    @classmethod
    def execute(
        cls,
        images,
        impulse=0.5,
        blotch=0.5,
        debug=False,
        mask=None,
    ) -> io.NodeOutput:
        dev = images.device
        if mm is not None:
            try:
                dev = mm.get_torch_device()
            except Exception:
                dev = images.device

        out, damage, report_dict = run_with_cpu_fallback(
            lambda d: repair_frames(
                images,
                mask,
                float(impulse),
                float(blotch),
                bool(debug),
                device=d,
            ),
            device=dev,
            label="MiniMaxH3_PixelRepair",
        )
        report_text = json.dumps(report_dict, separators=(",", ":")) + "\n\n" + report_dict["summary"]
        return io.NodeOutput(out, damage, report_text)
