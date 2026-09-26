"""N4 — H3 Reference Colour Match (identity tone vs <Picture 1>)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from comfy_api.latest import io

_PKG = Path(__file__).resolve().parents[1]
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from mmx_utils.device import run_with_cpu_fallback  # noqa: E402
from mmx_utils.reference_color_match import reference_color_match_frames  # noqa: E402

try:
    import comfy.model_management as mm
except Exception:  # not just ImportError: the comfy_kitchen skew raises AttributeError
    mm = None


class MiniMaxH3_ReferenceColorMatch(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MiniMaxH3_ReferenceColorMatch",
            display_name="H3 Reference Colour Match (identity tone vs <Picture 1>)",
            category="MiniMax H3/Finish",
            description=(
                "Hold the regenerated subject's colour to the reference still (<Picture 1>) "
                "in Oklab chroma after Plate Restore.\n\n"
                "Lightness is left alone by default so the plate's lighting wins. The report "
                "calls out how much red drift (+a in Oklab) was removed per frame."
            ),
            inputs=[
                io.Image.Input(
                    "images",
                    tooltip="Full frames after Plate Restore.",
                ),
                io.Mask.Input(
                    "edit_mask",
                    tooltip=(
                        "Full-frame edit mask — Plate Restore's restore_mask output is exactly "
                        "this. White = subject to match."
                    ),
                ),
                io.Image.Input(
                    "reference",
                    tooltip="Single reference still (<Picture 1>). Resized to frame size if needed.",
                ),
                io.Mask.Input(
                    "reference_mask",
                    optional=True,
                    tooltip="Optional mask on the reference (e.g. skin only). Unmasked = whole frame.",
                ),
                io.Float.Input(
                    "strength",
                    default=0.6,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip=(
                        "Blend toward the reference correction. Turn down if skin looks plastic "
                        "or the plate colour bleeds into the subject."
                    ),
                ),
                io.Combo.Input(
                    "mode",
                    options=["mean_std", "histogram"],
                    default="mean_std",
                    tooltip=(
                        "mean_std — match chroma mean and spread (stable on video).\n"
                        "histogram — CDF match per chroma channel (stronger, can flicker)."
                    ),
                ),
                io.Boolean.Input(
                    "match_lightness",
                    default=False,
                    tooltip=(
                        "Also match Oklab L. Leave off so the plate's lighting and shadows "
                        "on the subject are preserved."
                    ),
                ),
                io.Int.Input(
                    "temporal_smooth",
                    default=5,
                    min=1,
                    max=31,
                    tooltip=(
                        "Median window (frames) for smoothing the correction over time. "
                        "Raise on noisy segments; lower if the subject colour changes on purpose."
                    ),
                ),
            ],
            outputs=[
                io.Image.Output("images"),
                io.String.Output(
                    "report",
                    tooltip="Per-frame chroma shift removed, with Oklab +a (red) called out.",
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        images,
        edit_mask,
        reference,
        strength=0.6,
        mode="mean_std",
        match_lightness=False,
        temporal_smooth=5,
        reference_mask=None,
    ) -> io.NodeOutput:
        dev = images.device
        if mm is not None:
            try:
                dev = mm.get_torch_device()
            except Exception:
                dev = images.device

        out, report_dict = run_with_cpu_fallback(
            lambda d: reference_color_match_frames(
                images,
                edit_mask,
                reference,
                reference_mask,
                float(strength),
                str(mode),
                bool(match_lightness),
                int(temporal_smooth),
                device=d,
            ),
            device=dev,
            label="MiniMaxH3_ReferenceColorMatch",
        )
        report_text = json.dumps(report_dict, separators=(",", ":")) + "\n\n" + report_dict["summary"]
        return io.NodeOutput(out, report_text)
