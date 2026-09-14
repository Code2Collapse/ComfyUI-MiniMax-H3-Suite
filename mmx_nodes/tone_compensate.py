"""MiniMaxH3_ToneCompensate — remove the tone step at a chained-segment seam.

PORTED FROM: ComfyUI-MiniMaxH3-ToneCompensate (author rkfg, MIT License).
V3 schema, report output, input validation and the UI payload are added here.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from comfy_api.latest import io

_PKG = Path(__file__).resolve().parents[1]
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from mmx_utils.tone_compensate import (  # noqa: E402
    ToneCompensateError,
    compensate,
    compensate_report,
)


class MiniMaxH3_ToneCompensate(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MiniMaxH3_ToneCompensate",
            display_name="H3 Tone Compensate (seam brightness)",
            category="MiniMax H3/Long",
            description=(
                "Remove the brightness step where two chained H3 segments meet.\n\n"
                "H3's denoiser carries a tone bias: a generated segment tends to run "
                "darker than the keyframes it was anchored to. Chain segments into a "
                "long shot and that bias becomes a visible STEP at every seam - the "
                "classic 'my infinite video pulses' complaint.\n\n"
                "This measures the bias on the overlap, which is the stretch the new "
                "segment regenerated from the previous one, and takes it back out of "
                "the whole segment. It reports the drift it measured, so you can tell "
                "whether the seam was a tone problem at all before chasing it."
            ),
            inputs=[
                io.Image.Input(
                    "source",
                    tooltip="The PREVIOUS segment. Its last frames are the anchor the "
                            "new segment was told to continue from. You can pass the "
                            "whole segment; only the tail overlap is used."),
                io.Image.Input(
                    "target",
                    tooltip="The NEW segment, straight out of the sampler. Its first "
                            "frames are the model's regeneration of the source tail."),
                io.Combo.Input(
                    "mode", options=["frame_shift", "gain_bias", "lut"],
                    default="frame_shift",
                    tooltip="frame_shift - per-frame additive shift. The overlap is a "
                            "REGENERATION, not a pixel-wise transform of the source, "
                            "so matching the mean is honest and a per-pixel curve is "
                            "not. Start here.\n"
                            "gain_bias - one affine per channel; extrapolates cleanly "
                            "when the whole segment is uniformly lifted or crushed.\n"
                            "lut - per-pixel tone curve. Catches nonlinear drift but "
                            "OVERFITS on regenerated content, because those pixels "
                            "genuinely differ from the source."),
                io.Int.Input(
                    "overlap", default=48, min=1, max=4096,
                    tooltip="How many frames the two segments share - the KEYFRAME "
                            "count used when generating, not the segment length. "
                            "48 is 2s at 24 fps. Getting this wrong fits the "
                            "correction over frames that were never meant to match, "
                            "which is the usual reason this node 'does nothing'."),
                io.Int.Input(
                    "lut_bins", default=64, min=16, max=512, optional=True,
                    tooltip="Aggregation bins for lut mode only. More bins follow the "
                            "curve more closely and overfit sooner."),
            ],
            outputs=[
                io.Image.Output(display_name="images",
                                tooltip="The corrected target segment."),
                io.String.Output(display_name="report",
                                 tooltip="Measured drift per channel, and anything "
                                         "that had to be adjusted."),
            ],
        )

    @classmethod
    def fingerprint_inputs(cls, source, target, mode, overlap, lut_bins=64):
        return f"{id(source)}|{id(target)}|{mode}|{overlap}|{lut_bins}"

    @classmethod
    def execute(cls, source, target, mode, overlap, lut_bins=64) -> io.NodeOutput:
        try:
            out, stats = compensate(source, target, mode, overlap, lut_bins)
        except ToneCompensateError as exc:
            raise ValueError(f"H3 Tone Compensate: {exc}") from exc

        # Only NodeOutput.ui reaches the browser, so the per-frame drift is
        # serialised for the seam-drift plot.
        return io.NodeOutput(
            out,
            compensate_report(stats),
            ui={"mmx_tone_drift": [json.dumps(stats)]},
        )
